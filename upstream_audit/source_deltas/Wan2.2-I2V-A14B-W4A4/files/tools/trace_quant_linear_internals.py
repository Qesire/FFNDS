#!/usr/bin/env python3
"""Trace QuantLinear/SVDQuant runtime internals for Wan I2V.

Observed path for QuantLinearWithBranch:
    raw -> smooth -> activation QDQ -> main output -> low-rank branch output -> total output

Only compact summaries are written to JSONL; full activations are not retained.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from quant.mxfp4_stats import activation_mxfp4_group_stats
from quant.patch_scope import PatchScope
from quant.quant_linear import QuantLinear, QuantLinearWithBranch
from quant.stream_stats import compare_tensors, summarize_tensor
from quant.trace_context import JsonlTraceWriter, TraceContext, write_metadata
from quant.wan_smooth import _build_schedule, _register_linear_input_cast_hooks
from tools.infer_wan_i2v_svdquant import _apply_ptq_state_to_model, _load_i2v_ptq_states, _parse_keep_fp_blocks
from tools.stat_wan_i2v_activation_qdq_error import _module_meta, _select_quant_targets
from tools.stat_wan_i2v_isolated_layer_error import _build_filter_regex, _build_pipe, _expert_names, _load_manifest, _prep_sample, _size_to_hw
from tools.stat_wan_i2v_propagated_layer_error import _remove_hooks, _run_model
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LOGGER = logging.getLogger("trace_quant_linear_internals")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Trace Wan I2V QuantLinear internals")
    p.add_argument("--ckpt_dir", required=True, help="BF16 checkpoint directory used to instantiate model skeleton")
    p.add_argument("--quant_ckpt_dir", default="", help="Checkpoint directory for quant skeleton; defaults to --ckpt_dir")
    p.add_argument("--ptq_dir", required=True, help="ptq_stats.pt path/dir or legacy split PTQ directory")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=5)
    p.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--experts", type=str, default="both", choices=["high", "low", "both"])
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--target_regex", type=str, default="")
    p.add_argument("--blocks", type=str, default="", help="Optional comma-separated block ids")
    p.add_argument("--modules", type=str, default="", help="Optional module suffixes, e.g. self_attn.q,ffn.2")
    p.add_argument("--list_linears", action="store_true", default=False)
    p.add_argument("--low_keep_fp_blocks", type=str, default="")
    p.add_argument("--high_keep_fp_blocks", type=str, default="")
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)
    p.add_argument("--offload_models", action="store_true", default=True)
    p.add_argument("--no_offload_models", action="store_false", dest="offload_models")
    p.add_argument("--max_targets", type=int, default=32, help="Cap traced modules for exploratory runs; <=0 means no cap")
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    return p.parse_args()


def _inner_quant_linear(module: nn.Module) -> QuantLinear:
    if isinstance(module, QuantLinearWithBranch):
        return module.quant_linear
    if isinstance(module, QuantLinear):
        return module
    raise TypeError(f"Expected QuantLinear or QuantLinearWithBranch, got {type(module).__name__}")


def _apply_max_targets(target_modules: Dict[str, List[str]], max_targets: int) -> None:
    if max_targets <= 0:
        return
    remaining = int(max_targets)
    for expert in ("high", "low"):
        names = target_modules.get(expert, [])
        target_modules[expert] = names[:remaining]
        remaining = max(0, remaining - len(target_modules[expert]))


class QuantLinearInternalsTracer:
    def __init__(self, ctx: TraceContext, writer: JsonlTraceWriter) -> None:
        self.ctx = ctx
        self.writer = writer

    def emit(self, point: str, x: torch.Tensor, *, module_name: str, extra: Optional[Dict[str, Any]] = None) -> None:
        rec = self.ctx.base_record(point=point, module_name=module_name, block_idx=_block_idx(module_name), tensor=x, extra=extra)
        rec.update(summarize_tensor(x))
        self.writer.write(rec)

    def emit_compare(self, point: str, ref: torch.Tensor, got: torch.Tensor, *, module_name: str, extra: Optional[Dict[str, Any]] = None) -> None:
        rec = self.ctx.base_record(point=point, module_name=module_name, block_idx=_block_idx(module_name), tensor=got, extra=extra)
        rec.update(compare_tensors(ref, got, prefix="error_"))
        self.writer.write(rec)

    def patch_model(self, model: nn.Module, module_names: Sequence[str], scope: PatchScope) -> None:
        module_map = dict(model.named_modules())
        for name in module_names:
            module = module_map.get(name)
            if isinstance(module, QuantLinearWithBranch):
                scope.patch_method(module, "forward", self._make_branch_forward(name))
            elif isinstance(module, QuantLinear):
                scope.patch_method(module, "forward", self._make_quant_forward(name))

    def _quant_meta(self, qlinear: QuantLinear) -> Dict[str, Any]:
        clip_ratio, exp_offset, scale_method = qlinear.get_current_act_quant_params()
        return {
            "act_group_size": int(qlinear.act_group_size),
            "act_clip_ratio": float(clip_ratio),
            "act_exp_offset": int(exp_offset),
            "act_scale_method": str(scale_method),
            "scheme": str(qlinear.scheme),
            "smooth_enabled": bool(qlinear.smooth_enabled.item()),
            "input_channels_dim": int(qlinear.input_channels_dim.item()),
            "quantize_input": bool(qlinear.quantize_input),
            "skip_input_quant": bool(qlinear.skip_input_quant),
            "quant_method": str(qlinear.quant_method),
        }

    def _mxfp4_extra(self, qlinear: QuantLinear, x: torch.Tensor) -> Dict[str, Any]:
        if qlinear.scheme != "mxfp4":
            return {}
        clip_ratio, exp_offset, scale_method = qlinear.get_current_act_quant_params()
        return activation_mxfp4_group_stats(
            x,
            group_size=int(qlinear.act_group_size),
            clip_ratio=float(clip_ratio),
            channels_dim=int(qlinear.input_channels_dim.item()),
            exp_offset=int(exp_offset),
            scale_method=str(scale_method),
        )

    def _make_branch_forward(self, module_name: str):
        tracer = self

        def traced_forward(self: QuantLinearWithBranch, x: torch.Tensor) -> torch.Tensor:
            qlinear = self.quant_linear
            meta = tracer._quant_meta(qlinear)
            tracer.emit("ql_raw", x, module_name=module_name, extra={**meta, "has_branch": self.branch is not None})
            x_smooth = qlinear.smooth_input_only(x)
            tracer.emit("ql_smooth", x_smooth, module_name=module_name, extra=meta)
            tracer.emit_compare("ql_smooth_delta", x, x_smooth, module_name=module_name)
            x_main = qlinear._apply_input_quant_qdq(x_smooth)
            qdq_extra = {**meta, **tracer._mxfp4_extra(qlinear, x_smooth)}
            tracer.emit("ql_qdq", x_main, module_name=module_name, extra=qdq_extra)
            tracer.emit_compare("ql_qdq_error", x_smooth, x_main, module_name=module_name, extra=qdq_extra)
            out_main = qlinear.forward_preprocessed(x_main, out_dtype=x.dtype)
            tracer.emit("ql_main_out", out_main, module_name=module_name, extra=meta)
            if self.branch is not None:
                out_branch = self.branch(x_smooth)
                branch_meta = {"branch_rank": int(getattr(self.branch, "rank", -1))}
                tracer.emit("ql_branch_out", out_branch, module_name=module_name, extra=branch_meta)
                tracer.emit_compare("ql_branch_vs_main", out_main, out_branch, module_name=module_name, extra=branch_meta)
                out = out_main + out_branch
            else:
                out = out_main
            tracer.emit("ql_total_out", out, module_name=module_name, extra=meta)
            tracer.emit_compare("ql_total_vs_main", out_main, out, module_name=module_name, extra=meta)
            return out

        return traced_forward

    def _make_quant_forward(self, module_name: str):
        tracer = self

        def traced_forward(self: QuantLinear, x: torch.Tensor) -> torch.Tensor:
            meta = tracer._quant_meta(self)
            tracer.emit("ql_raw", x, module_name=module_name, extra={**meta, "has_branch": False})
            x_smooth = self.smooth_input_only(x)
            tracer.emit("ql_smooth", x_smooth, module_name=module_name, extra=meta)
            x_main = self._apply_input_quant_qdq(x_smooth)
            qdq_extra = {**meta, **tracer._mxfp4_extra(self, x_smooth)}
            tracer.emit("ql_qdq", x_main, module_name=module_name, extra=qdq_extra)
            tracer.emit_compare("ql_qdq_error", x_smooth, x_main, module_name=module_name, extra=qdq_extra)
            out = self.forward_preprocessed(x_main, out_dtype=x.dtype)
            tracer.emit("ql_total_out", out, module_name=module_name, extra=meta)
            return out

        return traced_forward


def _block_idx(module_name: str) -> Optional[int]:
    m = re.search(r"(?:^|\.)blocks\.(\d+)\.", module_name)
    return None if m is None else int(m.group(1))


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
    pattern = _build_filter_regex(args)
    quant_ckpt_dir = args.quant_ckpt_dir or args.ckpt_dir

    q_pipe = _build_pipe(quant_ckpt_dir, args)
    low_state, high_state, low_ptq_path, high_ptq_path = _load_i2v_ptq_states(args.ptq_dir)
    _apply_ptq_state_to_model(q_pipe.low_noise_model, low_state, "low_noise_model", low_ptq_path, keep_fp_blocks=_parse_keep_fp_blocks(args.low_keep_fp_blocks))
    _apply_ptq_state_to_model(q_pipe.high_noise_model, high_state, "high_noise_model", high_ptq_path, keep_fp_blocks=_parse_keep_fp_blocks(args.high_keep_fp_blocks))
    q_pipe.high_noise_model.cpu()
    q_pipe.low_noise_model.cpu()
    if q_pipe.device.type == "cuda":
        torch.cuda.empty_cache()

    q_models = {"high": q_pipe.high_noise_model, "low": q_pipe.low_noise_model}
    target_modules: Dict[str, List[str]] = {}
    for expert in _expert_names(args.experts):
        names, _modules, _meta = _select_quant_targets(q_models[expert], pattern)
        target_modules[expert] = names
        LOGGER.info("%s matched quant modules before cap: %d", expert, len(names))
        for name in names:
            LOGGER.info("%s | %s", expert, name)
    if args.list_linears:
        return
    _apply_max_targets(target_modules, int(args.max_targets))
    if not any(target_modules.values()):
        raise ValueError("No target quantized modules found.")

    out_dir = Path(args.out_dir)
    writer = JsonlTraceWriter(out_dir / "quant_linear_internals.jsonl")
    ctx = TraceContext()
    tracer = QuantLinearInternalsTracer(ctx, writer)
    cast_handles: List[Any] = []
    try:
        with PatchScope() as patch_scope:
            for expert in _expert_names(args.experts):
                cast_handles.extend(_register_linear_input_cast_hooks(q_models[expert]))
                tracer.patch_model(q_models[expert], target_modules[expert], patch_scope)

            cfg = WAN_CONFIGS["i2v-A14B"]
            frame_num = cfg.frame_num
            shift = cfg.sample_shift if args.shift is None else float(args.shift)
            boundary = float(q_pipe.boundary * q_pipe.num_train_timesteps)
            use_amp = q_pipe.device.type == "cuda" and q_pipe.param_dtype in (torch.float16, torch.bfloat16)
            model_name_by_expert = {"high": "high_noise_model", "low": "low_noise_model"}
            sample_iter: Iterable[Any] = samples
            if tqdm is not None and not args.disable_tqdm:
                sample_iter = tqdm(samples, desc="quant internals samples", unit="sample", dynamic_ncols=True)
            active_expert: Optional[str] = None
            for sample_idx, sample in enumerate(sample_iter):
                torch.manual_seed(int(args.seed) + sample_idx)
                ctx.begin_sample(sample_id=sample.sample_id, prompt=sample.prompt)
                item = _prep_sample(q_pipe, sample, frame_num=frame_num, target_hw=_size_to_hw(args.size))
                latent = item["latent"].to(device=q_pipe.device, dtype=q_pipe.param_dtype)
                y = item["y"].to(device=q_pipe.device, dtype=q_pipe.param_dtype)
                context = [t.to(device=q_pipe.device, dtype=q_pipe.param_dtype) for t in item["context"]]
                model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}
                scheduler, timesteps = _build_schedule(
                    sample_solver=args.sample_solver,
                    sampling_steps=int(args.sample_steps),
                    shift=shift,
                    num_train_timesteps=q_pipe.num_train_timesteps,
                    device=q_pipe.device,
                )
                timesteps_list = [float(t.item()) for t in timesteps]
                step_iter: Iterable[Tuple[int, float]] = list(enumerate(timesteps_list))
                if tqdm is not None and not args.disable_tqdm:
                    step_iter = tqdm(step_iter, desc=f"{sample.sample_id} denoise", unit="step", leave=False, dynamic_ncols=True)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=q_pipe.param_dtype, enabled=use_amp):
                    for step_idx, t_scalar in step_iter:
                        expert = "high" if t_scalar >= boundary else "low"
                        if expert not in _expert_names(args.experts):
                            continue
                        if args.offload_models and active_expert != expert:
                            for name, model in q_models.items():
                                model.to(q_pipe.device if name == expert else "cpu")
                            active_expert = expert
                            if q_pipe.device.type == "cuda":
                                torch.cuda.empty_cache()
                        elif not args.offload_models:
                            q_models[expert].to(q_pipe.device)

                        ctx.begin_branch("single")
                        ctx.begin_step(step_idx=step_idx, timestep=t_scalar, expert=expert, model_name=model_name_by_expert[expert])
                        timestep = torch.tensor([int(t_scalar)], device=q_pipe.device, dtype=torch.long)
                        t_for_step = torch.tensor(t_scalar, device=q_pipe.device, dtype=timesteps.dtype)
                        noise_pred = _run_model(q_models[expert], latent, timestep, model_args)
                        latent = scheduler.step(noise_pred.unsqueeze(0), t_for_step, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)
                writer.flush()
    finally:
        _remove_hooks(cast_handles)
        writer.close()

    write_metadata(
        out_dir / "metadata.json",
        {
            "ckpt_dir": args.ckpt_dir,
            "quant_ckpt_dir": quant_ckpt_dir,
            "ptq_dir": args.ptq_dir,
            "manifest": args.manifest,
            "max_samples": int(args.max_samples),
            "sample_steps": int(args.sample_steps),
            "size": args.size,
            "experts": _expert_names(args.experts),
            "target_regex": "" if pattern is None else pattern.pattern,
            "target_modules": target_modules,
            "max_targets": int(args.max_targets),
            "outputs": {"jsonl": str(out_dir / "quant_linear_internals.jsonl")},
        },
    )


if __name__ == "__main__":
    main()
