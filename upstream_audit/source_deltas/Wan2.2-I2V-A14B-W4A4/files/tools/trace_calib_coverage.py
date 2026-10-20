#!/usr/bin/env python3
"""Trace calibration coverage versus CFG inference coverage for Wan I2V.

The current act calibration path records conditional Linear inputs only. This
script compares that calibration-like path with true CFG-style cond/uncond
forwards using shared target hooks and compact streaming summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from quant.patch_scope import HookScope, find_module
from quant.stream_stats import summarize_tensor
from quant.trace_context import TraceContext, write_metadata
from quant.wan_smooth import _build_schedule, _register_linear_input_cast_hooks
from tools.trace_wan_adaln_norms import _build_pipe, _load_manifest, _prep_sample, _size_to_hw
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LOGGER = logging.getLogger("trace_calib_coverage")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Trace Wan I2V calibration coverage")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=5)
    p.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--guide_scale", type=float, default=None)
    p.add_argument("--target_regex", type=str, default="")
    p.add_argument("--blocks", type=str, default="0,10,20,30,39")
    p.add_argument("--modules", type=str, default="self_attn.q,self_attn.o,cross_attn.q,cross_attn.o,ffn.0,ffn.2")
    p.add_argument("--max_targets", type=int, default=96)
    p.add_argument("--mode", type=str, default="both", choices=["calib", "cfg", "both"])
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    p.add_argument("--list_linears", action="store_true", default=False)
    return p.parse_args()


@dataclass
class CoverageStat:
    calls: int = 0
    numel: int = 0
    first_step: Optional[int] = None
    last_step: Optional[int] = None
    absmax_max: float = 0.0
    abs_p50_sum: float = 0.0
    abs_p90_sum: float = 0.0
    abs_p99_sum: float = 0.0
    rms_sum: float = 0.0
    p99_over_p50_sum: float = 0.0
    absmax_over_p99_sum: float = 0.0

    def update(self, summary: Dict[str, Any], step_idx: Optional[int]) -> None:
        self.calls += 1
        self.numel += int(summary.get("numel", 0))
        self.absmax_max = max(self.absmax_max, float(summary.get("absmax", 0.0)))
        self.abs_p50_sum += float(summary.get("abs_p50", 0.0))
        self.abs_p90_sum += float(summary.get("abs_p90", 0.0))
        self.abs_p99_sum += float(summary.get("abs_p99", 0.0))
        self.rms_sum += float(summary.get("rms", 0.0))
        self.p99_over_p50_sum += float(summary.get("p99_over_p50", 0.0))
        self.absmax_over_p99_sum += float(summary.get("absmax_over_p99", 0.0))
        if step_idx is not None:
            step_idx = int(step_idx)
            self.first_step = step_idx if self.first_step is None else min(self.first_step, step_idx)
            self.last_step = step_idx if self.last_step is None else max(self.last_step, step_idx)

    def row(self, *, phase: str, branch: str, expert: str, module: str) -> Dict[str, Any]:
        denom = max(1, self.calls)
        return {
            "phase": phase,
            "branch": branch,
            "expert": expert,
            "module": module,
            "calls": int(self.calls),
            "numel": int(self.numel),
            "first_step": "" if self.first_step is None else int(self.first_step),
            "last_step": "" if self.last_step is None else int(self.last_step),
            "absmax_max": self.absmax_max,
            "abs_p50_mean": self.abs_p50_sum / denom,
            "abs_p90_mean": self.abs_p90_sum / denom,
            "abs_p99_mean": self.abs_p99_sum / denom,
            "rms_mean": self.rms_sum / denom,
            "p99_over_p50_mean": self.p99_over_p50_sum / denom,
            "absmax_over_p99_mean": self.absmax_over_p99_sum / denom,
        }


class CoverageCollector:
    def __init__(self, ctx: TraceContext) -> None:
        self.ctx = ctx
        self.stats: Dict[Tuple[str, str, str, str], CoverageStat] = {}
        self.raw_timesteps: Dict[str, Dict[int, float]] = {"high": {}, "low": {}}
        self.phase: str = ""

    def begin_phase(self, phase: str) -> None:
        self.phase = str(phase)

    def begin_step(self, *, step_idx: int, timestep: float, expert: str, model_name: str) -> None:
        self.ctx.begin_step(step_idx=step_idx, timestep=timestep, expert=expert, model_name=model_name)
        self.raw_timesteps[expert][int(step_idx)] = float(timestep)

    @torch.no_grad()
    def observe(self, expert: str, module_name: str, x: torch.Tensor) -> None:
        if self.ctx.expert != expert:
            return
        summary = summarize_tensor(x)
        key = (self.phase, self.ctx.branch, expert, module_name)
        self.stats.setdefault(key, CoverageStat()).update(summary, self.ctx.step_idx)

    def rows(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for (phase, branch, expert, module), stat in sorted(self.stats.items()):
            out.append(stat.row(phase=phase, branch=branch, expert=expert, module=module))
        return out


def _parse_csv_text(text: str) -> List[str]:
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def _parse_csv_ints(text: str) -> List[int]:
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


def _build_target_regex(args: argparse.Namespace) -> re.Pattern[str]:
    if args.target_regex:
        return re.compile(args.target_regex)
    blocks = _parse_csv_ints(args.blocks)
    modules = _parse_csv_text(args.modules)
    block_pat = "|".join(str(v) for v in blocks)
    module_pat = "|".join(re.escape(v) for v in modules)
    return re.compile(rf"(^|\.)blocks\.({block_pat})\.({module_pat})$")


def _matched_linears(model: nn.Module, pattern: re.Pattern[str], max_targets: int) -> List[str]:
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear) and pattern.search(name)]
    if max_targets > 0:
        names = names[: int(max_targets)]
    return names


def _register_hooks(model: nn.Module, expert: str, module_names: Sequence[str], collector: CoverageCollector) -> List[Any]:
    handles: List[Any] = []
    for name in module_names:
        module = find_module(model, name)
        if not isinstance(module, nn.Linear):
            continue

        def _make_hook(module_name: str):
            def _hook(_module: nn.Module, inp: Tuple[Any, ...]) -> None:
                if not inp or not torch.is_tensor(inp[0]):
                    return
                collector.observe(expert, module_name, inp[0])

            return _hook

        handles.append(module.register_forward_pre_hook(_make_hook(name), with_kwargs=False))
    return handles


def _run_model(model: nn.Module, latent: torch.Tensor, timestep: torch.Tensor, model_args: Dict[str, Any]) -> torch.Tensor:
    return model([latent], t=timestep, **model_args)[0]


def _write_outputs(out_dir: Path, collector: CoverageCollector, metadata: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collector.rows()
    coverage_csv = out_dir / "calib_coverage.csv"
    fieldnames = [
        "phase",
        "branch",
        "expert",
        "module",
        "calls",
        "numel",
        "first_step",
        "last_step",
        "absmax_max",
        "abs_p50_mean",
        "abs_p90_mean",
        "abs_p99_mean",
        "rms_mean",
        "p99_over_p50_mean",
        "absmax_over_p99_mean",
    ]
    with coverage_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    delta_rows = _distribution_delta_rows(rows)
    delta_csv = out_dir / "cond_uncond_distribution_delta.csv"
    delta_fields = [
        "expert",
        "module",
        "cond_calls",
        "uncond_calls",
        "call_coverage_ratio_uncond_over_cond",
        "abs_p99_uncond_over_cond",
        "rms_uncond_over_cond",
        "absmax_uncond_over_cond",
    ]
    with delta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=delta_fields)
        writer.writeheader()
        writer.writerows(delta_rows)

    metadata["raw_timesteps"] = {expert: {str(k): v for k, v in steps.items()} for expert, steps in collector.raw_timesteps.items()}
    metadata["num_coverage_rows"] = len(rows)
    metadata["num_delta_rows"] = len(delta_rows)
    metadata["outputs"] = {"coverage_csv": str(coverage_csv), "cond_uncond_delta_csv": str(delta_csv)}
    write_metadata(out_dir / "metadata.json", metadata)


def _distribution_delta_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        lookup[(row["phase"], row["branch"], row["expert"], row["module"])] = row
    out: List[Dict[str, Any]] = []
    modules = sorted({(r["expert"], r["module"]) for r in rows if r["phase"] == "cfg"})
    for expert, module in modules:
        cond = lookup.get(("cfg", "cond", expert, module))
        uncond = lookup.get(("cfg", "uncond", expert, module))
        if cond is None or uncond is None:
            continue
        out.append(
            {
                "expert": expert,
                "module": module,
                "cond_calls": int(cond["calls"]),
                "uncond_calls": int(uncond["calls"]),
                "call_coverage_ratio_uncond_over_cond": _ratio(uncond["calls"], cond["calls"]),
                "abs_p99_uncond_over_cond": _ratio(uncond["abs_p99_mean"], cond["abs_p99_mean"]),
                "rms_uncond_over_cond": _ratio(uncond["rms_mean"], cond["rms_mean"]),
                "absmax_uncond_over_cond": _ratio(uncond["absmax_max"], cond["absmax_max"]),
            }
        )
    return out


def _ratio(num: Any, den: Any) -> float:
    return float(num) / max(float(den), 1e-30)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0")
    if args.sample_steps <= 0:
        raise ValueError("--sample_steps must be > 0")

    samples = _load_manifest(Path(args.manifest), max_samples=int(args.max_samples))
    if not samples:
        raise ValueError("No valid manifest samples found.")
    pattern = _build_target_regex(args)
    pipe = _build_pipe(args)
    model_by_expert = {"high": pipe.high_noise_model, "low": pipe.low_noise_model}
    model_name_by_expert = {"high": "high_noise_model", "low": "low_noise_model"}
    target_modules = {expert: _matched_linears(model_by_expert[expert], pattern, int(args.max_targets)) for expert in ("high", "low")}
    for expert, names in target_modules.items():
        LOGGER.info("%s matched target Linear modules: %d", expert, len(names))
        for name in names:
            LOGGER.info("%s | %s", expert, name)
    if args.list_linears:
        return
    if not any(target_modules.values()):
        raise ValueError("No target Linear modules matched.")

    ctx = TraceContext()
    collector = CoverageCollector(ctx)
    cfg = WAN_CONFIGS["i2v-A14B"]
    frame_num = cfg.frame_num
    shift = cfg.sample_shift if args.shift is None else float(args.shift)
    guide_scale = cfg.sample_guide_scale if args.guide_scale is None else (float(args.guide_scale), float(args.guide_scale))
    boundary = float(pipe.boundary * pipe.num_train_timesteps)
    use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)

    cast_handles: List[Any] = []
    try:
        with HookScope() as hook_scope:
            for expert, model in model_by_expert.items():
                cast_handles.extend(_register_linear_input_cast_hooks(model))
                hook_scope.extend(_register_hooks(model, expert, target_modules[expert], collector))
            sample_iter: Iterable[Dict[str, str]] = samples
            if tqdm is not None and not args.disable_tqdm:
                sample_iter = tqdm(samples, desc="coverage samples", unit="sample", dynamic_ncols=True)
            for sample_idx, sample in enumerate(sample_iter):
                torch.manual_seed(int(args.seed) + sample_idx)
                ctx.begin_sample(sample_id=sample["sample_id"], prompt=sample["prompt"])
                item = _prep_sample(pipe, sample, frame_num=frame_num, target_hw=_size_to_hw(args.size))
                if args.mode in ("calib", "both"):
                    _run_calib_like(pipe, item, collector, ctx, model_by_expert, model_name_by_expert, boundary, shift, args, use_amp)
                if args.mode in ("cfg", "both"):
                    _run_cfg_like(pipe, item, collector, ctx, model_by_expert, model_name_by_expert, boundary, shift, guide_scale, args, use_amp)
    finally:
        for handle in cast_handles:
            handle.remove()

    _write_outputs(
        Path(args.out_dir),
        collector,
        {
            "ckpt_dir": args.ckpt_dir,
            "manifest": args.manifest,
            "max_samples": int(args.max_samples),
            "sample_steps": int(args.sample_steps),
            "size": args.size,
            "sample_solver": args.sample_solver,
            "shift": shift,
            "guide_scale": guide_scale,
            "target_regex": pattern.pattern,
            "target_modules": target_modules,
            "mode": args.mode,
            "definition": "calib phase runs conditional-only denoise updates; cfg phase runs cond+uncond forwards and CFG scheduler updates.",
        },
    )


def _run_calib_like(
    pipe: Any,
    item: Dict[str, Any],
    collector: CoverageCollector,
    ctx: TraceContext,
    model_by_expert: Dict[str, nn.Module],
    model_name_by_expert: Dict[str, str],
    boundary: float,
    shift: float,
    args: argparse.Namespace,
    use_amp: bool,
) -> None:
    collector.begin_phase("calib")
    ctx.begin_branch("cond")
    latent = item["latent"].to(device=pipe.device, dtype=pipe.param_dtype)
    y = item["y"].to(device=pipe.device, dtype=pipe.param_dtype)
    context = [t.to(device=pipe.device, dtype=pipe.param_dtype) for t in item["context"]["cond"]]
    model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}
    scheduler, timesteps = _build_schedule(args.sample_solver, int(args.sample_steps), shift, pipe.num_train_timesteps, pipe.device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype, enabled=use_amp):
        for step_idx, t_scalar in enumerate([float(t.item()) for t in timesteps]):
            expert = "high" if t_scalar >= boundary else "low"
            collector.begin_step(step_idx=step_idx, timestep=t_scalar, expert=expert, model_name=model_name_by_expert[expert])
            timestep = torch.tensor([int(t_scalar)], device=pipe.device, dtype=torch.long)
            t_for_step = torch.tensor(t_scalar, device=pipe.device, dtype=timesteps.dtype)
            noise_pred = _run_model(model_by_expert[expert], latent, timestep, model_args)
            latent = scheduler.step(noise_pred.unsqueeze(0), t_for_step, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)


def _run_cfg_like(
    pipe: Any,
    item: Dict[str, Any],
    collector: CoverageCollector,
    ctx: TraceContext,
    model_by_expert: Dict[str, nn.Module],
    model_name_by_expert: Dict[str, str],
    boundary: float,
    shift: float,
    guide_scale: Tuple[float, float],
    args: argparse.Namespace,
    use_amp: bool,
) -> None:
    collector.begin_phase("cfg")
    latent = item["latent"].to(device=pipe.device, dtype=pipe.param_dtype)
    y = item["y"].to(device=pipe.device, dtype=pipe.param_dtype)
    contexts = {
        branch: [t.to(device=pipe.device, dtype=pipe.param_dtype) for t in item["context"][branch]] for branch in ("cond", "uncond")
    }
    scheduler, timesteps = _build_schedule(args.sample_solver, int(args.sample_steps), shift, pipe.num_train_timesteps, pipe.device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype, enabled=use_amp):
        for step_idx, t_scalar in enumerate([float(t.item()) for t in timesteps]):
            expert = "high" if t_scalar >= boundary else "low"
            model = model_by_expert[expert]
            timestep = torch.tensor([int(t_scalar)], device=pipe.device, dtype=torch.long)
            preds: Dict[str, torch.Tensor] = {}
            for branch in ("cond", "uncond"):
                ctx.begin_branch(branch)
                collector.begin_step(step_idx=step_idx, timestep=t_scalar, expert=expert, model_name=model_name_by_expert[expert])
                model_args = {"context": contexts[branch], "seq_len": int(item["seq_len"]), "y": [y]}
                preds[branch] = _run_model(model, latent, timestep, model_args)
            scale = float(guide_scale[1] if expert == "high" else guide_scale[0])
            noise_pred = preds["uncond"] + scale * (preds["cond"] - preds["uncond"])
            t_for_step = torch.tensor(t_scalar, device=pipe.device, dtype=timesteps.dtype)
            latent = scheduler.step(noise_pred.unsqueeze(0), t_for_step, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)


if __name__ == "__main__":
    main()
