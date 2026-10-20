#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from quant.patch_scope import PatchScope
from quant.wan_smooth import _build_schedule
from tools.trace_wan_adaln_norms import (
    _build_pipe,
    _branches,
    _experts,
    _load_manifest,
    _parse_csv_ints,
    _prep_sample,
    _size_to_hw,
)
from wan.configs import WAN_CONFIGS
from wan.modules.model import WanAttentionBlock


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Round6 teacher-forced AdaLN paired tensor dataset")
    p.add_argument("--mode", required=True, choices=["cache", "trace"])

    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--variant", default="bf16", choices=["bf16", "quant_empty", "quant_policy"])
    p.add_argument("--ptq_dir", default="")
    p.add_argument("--act_policy_json", default="")
    p.add_argument("--latent_cache", default="")

    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=6)
    p.add_argument("--size", type=str, default="480*832")
    p.add_argument("--blocks", type=str, default="0")
    p.add_argument("--branches", type=str, default="cond", choices=["cond", "uncond", "both"])
    p.add_argument("--experts", type=str, default="both", choices=["high", "low", "both"])
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)

    p.add_argument("--max_rows", type=int, default=64)
    p.add_argument("--max_channels", type=int, default=128)
    p.add_argument("--row_seed", type=int, default=20260620)
    p.add_argument("--channel_seed", type=int, default=20260621)

    p.add_argument("--run_label", default="", help="Human-readable intervention label stored in metadata.")
    p.add_argument(
        "--intervention",
        default="none",
        choices=[
            "none",
            "fixed_clip_global",
            "fixed_clip_target",
            "ffn_gate_target",
            "fixed_clip_target_ffn_gate",
        ],
    )
    p.add_argument("--clip_ratio", type=float, default=1.0)
    p.add_argument("--clip_exp_offset", type=int, default=0)
    p.add_argument("--clip_scale_method", type=str, default="safe_ceil")
    p.add_argument("--target_block", type=int, default=-1)
    p.add_argument("--ffn_gate_alpha", type=float, default=1.0)
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    return p.parse_args()


def _stable_int(text: str, salt: int = 0) -> int:
    h = hashlib.sha256((str(salt) + "|" + text).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def _sample_indices(n: int, k: int, key: str, salt: int) -> torch.Tensor:
    n = int(n)
    k = min(int(k), n)
    if n <= 0 or k <= 0:
        return torch.empty((0,), dtype=torch.long)
    if k >= n:
        return torch.arange(n, dtype=torch.long)
    g = torch.Generator(device="cpu")
    g.manual_seed(_stable_int(key, salt=salt))
    return torch.randperm(n, generator=g)[:k].sort().values


def _flatten_rows_cpu(x: torch.Tensor) -> torch.Tensor:
    xd = x.detach()
    if xd.ndim == 0:
        return xd.to(torch.float32).reshape(1, 1).cpu().contiguous()
    c = int(xd.shape[-1])
    x2 = xd.to(torch.float32).reshape(-1, c)
    return torch.nan_to_num(x2.cpu().contiguous(), nan=0.0, posinf=0.0, neginf=0.0)


def _stats_cpu(x: torch.Tensor, prefix: str = "") -> Dict[str, Any]:
    x = torch.nan_to_num(x.detach().to(torch.float32).cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = x.reshape(-1)
    if flat.numel() == 0:
        return {prefix + "numel": 0}
    abs_flat = flat.abs()
    n = int(abs_flat.numel())

    def q(v: float) -> float:
        if n == 1:
            return float(abs_flat[0].item())
        return float(torch.quantile(abs_flat, torch.tensor(v, dtype=torch.float32)).item())

    abs_p50 = q(0.50)
    abs_p99 = q(0.99)
    absmax = float(abs_flat.max().item())

    out: Dict[str, Any] = {
        prefix + "numel": n,
        prefix + "shape": "x".join(map(str, x.shape)),
        prefix + "mean": float(flat.mean().item()),
        prefix + "std": float(flat.std(unbiased=False).item()) if n > 1 else 0.0,
        prefix + "rms": float(torch.sqrt(torch.mean(flat * flat)).item()),
        prefix + "absmax": absmax,
        prefix + "abs_p50": abs_p50,
        prefix + "abs_p90": q(0.90),
        prefix + "abs_p95": q(0.95),
        prefix + "abs_p99": abs_p99,
        prefix + "abs_p999": q(0.999),
        prefix + "absmax_over_p99": absmax / max(abs_p99, 1e-30),
        prefix + "p99_over_p50": abs_p99 / max(abs_p50, 1e-30),
    }

    if x.ndim >= 2:
        x2 = x.reshape(-1, int(x.shape[-1]))
        ch_absmax = x2.abs().amax(dim=0)
        ch_rms = torch.sqrt(torch.mean(x2 * x2, dim=0))
        out[prefix + "rows"] = int(x2.shape[0])
        out[prefix + "channels"] = int(x2.shape[-1])
        out[prefix + "channel_absmax_max"] = float(ch_absmax.max().item())
        out[prefix + "channel_absmax_p99"] = float(torch.quantile(ch_absmax, torch.tensor(0.99)).item())
        out[prefix + "channel_rms_max"] = float(ch_rms.max().item())
        out[prefix + "channel_rms_p99"] = float(torch.quantile(ch_rms, torch.tensor(0.99)).item())
    return out


def _slice_tensor(
    x: torch.Tensor,
    *,
    key: str,
    max_rows: int,
    max_channels: int,
    row_seed: int,
    channel_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    x2 = _flatten_rows_cpu(x)
    rows_total, channels_total = int(x2.shape[0]), int(x2.shape[1])

    row_idx = _sample_indices(rows_total, max_rows, key=key + "|rows", salt=row_seed)
    ch_idx = _sample_indices(channels_total, max_channels, key=key + "|channels", salt=channel_seed)

    sliced = x2.index_select(0, row_idx).index_select(1, ch_idx).contiguous()
    meta = {
        "rows_total": rows_total,
        "channels_total": channels_total,
        "rows_saved": int(row_idx.numel()),
        "channels_saved": int(ch_idx.numel()),
    }
    return sliced, row_idx, ch_idx, meta


def _e_slice(e_tensor: torch.Tensor, ch_idx: torch.Tensor) -> torch.Tensor:
    x2 = _flatten_rows_cpu(e_tensor)
    if x2.shape[-1] == ch_idx.numel():
        return x2
    ch = ch_idx.clamp(max=x2.shape[-1] - 1)
    return x2.index_select(1, ch).contiguous()


def _setup_quant_if_needed(pipe: Any, args: argparse.Namespace, *, shift: float) -> None:
    if args.variant == "bf16":
        return
    if not args.ptq_dir:
        raise ValueError("--ptq_dir is required for quant variants")

    from quant.act_policy import (
        build_wan_i2v_timestep_schedule,
        load_act_policy_json,
        maybe_patch_model_forward_for_act_context,
        set_quant_modules_act_policy,
    )
    from tools.infer_wan_i2v_svdquant import (
        _apply_ptq_state_to_model,
        _load_i2v_ptq_states,
    )

    low_state, high_state, low_ptq_path, high_ptq_path = _load_i2v_ptq_states(args.ptq_dir)

    _apply_ptq_state_to_model(
        pipe.low_noise_model,
        low_state,
        "low_noise_model",
        low_ptq_path,
        keep_fp_blocks=set(),
    )
    _apply_ptq_state_to_model(
        pipe.high_noise_model,
        high_state,
        "high_noise_model",
        high_ptq_path,
        keep_fp_blocks=set(),
    )

    policy = None
    if args.variant == "quant_policy":
        if not args.act_policy_json:
            raise ValueError("--act_policy_json is required for quant_policy")
        policy = load_act_policy_json(args.act_policy_json)

    set_quant_modules_act_policy(pipe.low_noise_model, "low_noise_model", policy)
    set_quant_modules_act_policy(pipe.high_noise_model, "high_noise_model", policy)

    if args.variant == "quant_policy":
        timesteps, boundary_timestep, expert_timestep_schedule = build_wan_i2v_timestep_schedule(
            num_train_timesteps=pipe.num_train_timesteps,
            boundary=pipe.boundary,
            sample_solver=args.sample_solver,
            sampling_steps=int(args.sample_steps),
            shift=float(shift),
            device=pipe.device,
        )
        maybe_patch_model_forward_for_act_context(
            pipe.low_noise_model,
            "low",
            expert_timestep_schedule=expert_timestep_schedule,
        )
        maybe_patch_model_forward_for_act_context(
            pipe.high_noise_model,
            "high",
            expert_timestep_schedule=expert_timestep_schedule,
        )


def _force_model_to_device(model: nn.Module, device: torch.device, label: str) -> None:
    model.to(device)
    bad = []
    for name, p in model.named_parameters(recurse=True):
        if p.device != device:
            bad.append(("param", name, str(p.device)))
            if len(bad) >= 8:
                break
    if len(bad) < 8:
        for name, b in model.named_buffers(recurse=True):
            if b.device != device:
                bad.append(("buffer", name, str(b.device)))
                if len(bad) >= 8:
                    break
    if bad:
        raise RuntimeError(f"{label} still has tensors off device {device}: {bad}")




def _make_fixed_clip_policy(clip_ratio: float, exp_offset: int = 0, scale_method: str = "safe_ceil") -> dict:
    return {
        "__default__": {
            "clip_ratio": float(clip_ratio),
            "exp_offset": int(exp_offset),
            "scale_method": str(scale_method),
        }
    }


def _apply_fixed_clip_intervention(
    model: nn.Module,
    *,
    clip_ratio: float,
    exp_offset: int,
    scale_method: str,
    scope: str,
    target_block: int,
) -> int:
    from quant.quant_linear import QuantLinearWithBranch

    policy = _make_fixed_clip_policy(
        clip_ratio=float(clip_ratio),
        exp_offset=int(exp_offset),
        scale_method=str(scale_method),
    )

    count = 0
    touched = 0
    prefix = f"blocks.{int(target_block)}."

    for name, module in model.named_modules():
        if not isinstance(module, QuantLinearWithBranch):
            continue

        module.set_module_name(name)
        count += 1

        if scope == "global":
            module.set_act_policy(policy)
            touched += 1
        elif scope == "target":
            if name.startswith(prefix):
                module.set_act_policy(policy)
                touched += 1
            else:
                module.set_act_policy(None)
        else:
            raise ValueError(f"unknown fixed clip intervention scope: {scope}")

    print(
        f"[round7] fixed clip intervention scope={scope} target_block={target_block} "
        f"clip={clip_ratio} exp_offset={exp_offset} scale_method={scale_method} "
        f"touched={touched}/{count}",
        flush=True,
    )
    return touched


def _apply_round7_intervention(model: nn.Module, args: argparse.Namespace, expert: str) -> None:
    if args.intervention == "none":
        return

    if args.intervention == "fixed_clip_global":
        _apply_fixed_clip_intervention(
            model,
            clip_ratio=float(args.clip_ratio),
            exp_offset=int(args.clip_exp_offset),
            scale_method=str(args.clip_scale_method),
            scope="global",
            target_block=int(args.target_block),
        )
        return

    if args.intervention in ("fixed_clip_target", "fixed_clip_target_ffn_gate"):
        if int(args.target_block) < 0:
            raise ValueError("--target_block is required for target-scoped intervention")
        _apply_fixed_clip_intervention(
            model,
            clip_ratio=float(args.clip_ratio),
            exp_offset=int(args.clip_exp_offset),
            scale_method=str(args.clip_scale_method),
            scope="target",
            target_block=int(args.target_block),
        )
        return

    if args.intervention == "ffn_gate_target":
        return

    raise ValueError(f"unknown intervention: {args.intervention}")


class DatasetWriter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self.index_rows: List[Dict[str, Any]] = []

    def add_record(self, rec: Dict[str, Any]) -> None:
        rec_id = len(self.records)
        rec["record_id"] = rec_id
        self.records.append(rec)
        self.index_rows.append({
            "record_id": rec_id,
            "variant": rec["variant"],
            "sample_id": rec["sample_id"],
            "branch": rec["branch"],
            "expert": rec["expert"],
            "block_idx": rec["block_idx"],
            "step_idx": rec["step_idx"],
            "timestep": rec["timestep"],
            "point": rec["point"],
            "rows_saved": rec["slice"].shape[0],
            "channels_saved": rec["slice"].shape[1],
            "rows_total": rec["rows_total"],
            "channels_total": rec["channels_total"],
        })

    def save(self, metadata: Dict[str, Any]) -> None:
        torch.save({"metadata": metadata, "records": self.records}, self.out_dir / "samples.pt")
        with (self.out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "record_id", "variant", "sample_id", "branch", "expert",
                "block_idx", "step_idx", "timestep", "point",
                "rows_saved", "channels_saved", "rows_total", "channels_total",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.index_rows)
        (self.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


class TeacherForcedTracer:
    def __init__(self, writer: DatasetWriter, args: argparse.Namespace) -> None:
        self.writer = writer
        self.args = args
        self.sample_id = ""
        self.branch = ""
        self.step_idx = -1
        self.timestep = float("nan")
        self.expert = ""
        self.model_name = ""

    def set_context(self, *, sample_id: str, branch: str, step_idx: int, timestep: float, expert: str, model_name: str) -> None:
        self.sample_id = str(sample_id)
        self.branch = str(branch)
        self.step_idx = int(step_idx)
        self.timestep = float(timestep)
        self.expert = str(expert)
        self.model_name = str(model_name)

    def patch_model(self, model: nn.Module, block_indices: Sequence[int], scope: PatchScope) -> None:
        for block_idx in block_indices:
            if block_idx < 0 or block_idx >= len(model.blocks):
                continue
            block = model.blocks[block_idx]
            if isinstance(block, WanAttentionBlock):
                scope.patch_method(block, "forward", self._make_block_forward(block_idx))

    def emit(self, *, point: str, block_idx: int, tensor: torch.Tensor, e_tensors: Dict[str, torch.Tensor]) -> None:
        key = "|".join([
            self.sample_id,
            self.branch,
            self.expert,
            str(block_idx),
            str(self.step_idx),
            point,
            "shape",
            "x".join(map(str, tensor.shape)),
        ])

        sliced, row_idx, ch_idx, meta = _slice_tensor(
            tensor,
            key=key,
            max_rows=int(self.args.max_rows),
            max_channels=int(self.args.max_channels),
            row_seed=int(self.args.row_seed),
            channel_seed=int(self.args.channel_seed),
        )

        e_stats = {}
        for name, val in e_tensors.items():
            e_stats.update(_stats_cpu(_e_slice(val, ch_idx), prefix=f"{name}_slice_"))

        rec: Dict[str, Any] = {
            "variant": self.args.variant,
            "run_label": self.args.run_label or self.args.variant,
            "intervention": self.args.intervention,
            "clip_ratio": float(self.args.clip_ratio),
            "target_block": int(self.args.target_block),
            "ffn_gate_alpha": float(self.args.ffn_gate_alpha),
            "teacher_forced": True,
            "sample_id": self.sample_id,
            "branch": self.branch,
            "expert": self.expert,
            "model_name": self.model_name,
            "block_idx": int(block_idx),
            "step_idx": int(self.step_idx),
            "timestep": float(self.timestep),
            "point": point,
            "slice": sliced.to(torch.float16),
            "row_idx": row_idx.to(torch.int32),
            "channel_idx": ch_idx.to(torch.int32),
            "e_stats": e_stats,
            **meta,
            **_stats_cpu(sliced, prefix="slice_"),
        }
        self.writer.add_record(rec)

    def _make_block_forward(self, block_idx: int):
        tracer = self

        def traced_forward(self: WanAttentionBlock, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
            assert e.dtype == torch.float32
            with torch.amp.autocast("cuda", dtype=torch.float32):
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

            e_shift_self = e[0].squeeze(2)
            e_scale_self = e[1].squeeze(2)
            e_gate_self = e[2].squeeze(2)
            e_shift_ffn = e[3].squeeze(2)
            e_scale_ffn = e[4].squeeze(2)
            e_gate_ffn = e[5].squeeze(2)

            e_tensors = {
                "e_shift_self": e_shift_self,
                "e_scale_self": e_scale_self,
                "e_gate_self": e_gate_self,
                "e_shift_ffn": e_shift_ffn,
                "e_scale_ffn": e_scale_ffn,
                "e_gate_ffn": e_gate_ffn,
                "one_plus_e_scale_self": 1 + e_scale_self,
                "one_plus_e_scale_ffn": 1 + e_scale_ffn,
            }

            norm1_out = self.norm1(x)
            self_in = norm1_out.float() * (1 + e_scale_self) + e_shift_self
            tracer.emit(point="self_adaln_in", block_idx=block_idx, tensor=self_in, e_tensors=e_tensors)

            y = self.self_attn(self_in, seq_lens, grid_sizes, freqs)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                self_update = y * e_gate_self
                tracer.emit(point="self_residual_update", block_idx=block_idx, tensor=self_update, e_tensors=e_tensors)
                x = x + self_update

            norm3_out = self.norm3(x)
            cross_out = self.cross_attn(norm3_out, context, context_lens)
            x = x + cross_out

            norm2_out = self.norm2(x)
            ffn_in = norm2_out.float() * (1 + e_scale_ffn) + e_shift_ffn
            tracer.emit(point="ffn_adaln_in", block_idx=block_idx, tensor=ffn_in, e_tensors=e_tensors)

            y = self.ffn(ffn_in)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                ffn_update = y * e_gate_ffn

                # Round7 optional residual/gate damper.
                # This is deliberately target-block only and conservative.
                if (
                    tracer.args.intervention in ("ffn_gate_target", "fixed_clip_target_ffn_gate")
                    and int(tracer.args.target_block) == int(block_idx)
                ):
                    ffn_update = ffn_update * float(tracer.args.ffn_gate_alpha)

                tracer.emit(point="ffn_residual_update", block_idx=block_idx, tensor=ffn_update, e_tensors=e_tensors)
                x = x + ffn_update

            tracer.emit(point="block_out", block_idx=block_idx, tensor=x, e_tensors=e_tensors)
            return x

        return traced_forward


def _common_pipe_and_sample(args: argparse.Namespace):
    samples = _load_manifest(Path(args.manifest), max_samples=int(args.max_samples))
    if not samples:
        raise ValueError("No valid manifest samples found")

    pipe = _build_pipe(args)
    cfg = WAN_CONFIGS["i2v-A14B"]
    frame_num = cfg.frame_num
    shift = cfg.sample_shift if args.shift is None else float(args.shift)

    sample = samples[0]
    torch.manual_seed(int(args.seed))
    item = _prep_sample(pipe, sample, frame_num=frame_num, target_hw=_size_to_hw(args.size))

    return pipe, sample, item, shift


def run_cache(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe, sample, item, shift = _common_pipe_and_sample(args)

    boundary = float(pipe.boundary * pipe.num_train_timesteps)
    use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)

    model_by_expert = {"high": pipe.high_noise_model, "low": pipe.low_noise_model}

    latent0 = item["latent"].to(device=pipe.device, dtype=pipe.param_dtype)
    y = item["y"].to(device=pipe.device, dtype=pipe.param_dtype)

    scheduler, timesteps = _build_schedule(
        sample_solver=args.sample_solver,
        sampling_steps=int(args.sample_steps),
        shift=shift,
        num_train_timesteps=pipe.num_train_timesteps,
        device=pipe.device,
    )
    timesteps_list = [float(t.item()) for t in timesteps]

    # Critical for teacher-forced Stage B:
    # cache all conditioning tensors so trace jobs never call _prep_sample()
    # and therefore never reopen unstable manifest image paths.
    static = {
        "seq_len": int(item["seq_len"]),
        "y": item["y"].detach().cpu(),
        "context_by_branch": {
            branch: [t.detach().cpu() for t in item["context"][branch]]
            for branch in _branches(args.branches)
        },
    }

    cache_records: List[Dict[str, Any]] = []

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype, enabled=use_amp):
        for branch in _branches(args.branches):
            latent = latent0.clone()
            context = [t.to(device=pipe.device, dtype=pipe.param_dtype) for t in item["context"][branch]]
            model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}

            for step_idx, t_scalar in enumerate(timesteps_list):
                expert = "high" if t_scalar >= boundary else "low"

                cache_records.append({
                    "sample_id": sample["sample_id"],
                    "prompt": sample["prompt"],
                    "branch": branch,
                    "step_idx": int(step_idx),
                    "timestep": float(t_scalar),
                    "expert": expert,
                    "latent_before": latent.detach().cpu(),
                })

                timestep = torch.tensor([int(t_scalar)], device=pipe.device, dtype=torch.long)
                t_for_step = torch.tensor(t_scalar, device=pipe.device, dtype=timesteps.dtype)

                noise_pred = model_by_expert[expert]([latent], t=timestep, **model_args)[0]
                latent = scheduler.step(
                    noise_pred.unsqueeze(0),
                    t_for_step,
                    latent.unsqueeze(0),
                    return_dict=False,
                )[0].squeeze(0)

    payload = {
        "metadata": {
            "version": "teacher_forced_latent_cache_v2_with_conditioning",
            "manifest": args.manifest,
            "ckpt_dir": args.ckpt_dir,
            "sample_id": sample["sample_id"],
            "sample_steps": int(args.sample_steps),
            "branches": _branches(args.branches),
            "size": args.size,
            "seed": int(args.seed),
            "sample_solver": args.sample_solver,
            "shift": float(shift),
            "contains_static_conditioning": True,
            "static_keys": ["seq_len", "y", "context_by_branch"],
        },
        "static": static,
        "records": cache_records,
    }

    torch.save(payload, out_dir / "bf16_latent_cache.pt")
    (out_dir / "metadata.json").write_text(json.dumps(payload["metadata"], indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", out_dir / "bf16_latent_cache.pt", "records", len(cache_records))
    print("STATIC seq_len", static["seq_len"])
    print("STATIC branches", list(static["context_by_branch"].keys()))


def run_trace(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.latent_cache:
        raise ValueError("--latent_cache is required for mode=trace")

    cache = torch.load(args.latent_cache, map_location="cpu")
    cache_records = cache["records"]
    static = cache.get("static", None)

    if static is None:
        raise RuntimeError(
            "latent cache does not contain static conditioning. "
            "Please regenerate Stage A cache with the patched script."
        )

    blocks = _parse_csv_ints(args.blocks)

    # Important: Stage B builds only the model pipeline.
    # It must NOT call _prep_sample(), because that reopens manifest image files.
    pipe = _build_pipe(args)
    cfg = WAN_CONFIGS["i2v-A14B"]
    shift = cfg.sample_shift if args.shift is None else float(args.shift)

    _setup_quant_if_needed(pipe, args, shift=float(shift))

    model_by_expert = {"high": pipe.high_noise_model, "low": pipe.low_noise_model}
    model_name_by_expert = {"high": "high_noise_model", "low": "low_noise_model"}

    for expert in _experts(args.experts):
        _force_model_to_device(model_by_expert[expert], pipe.device, f"{args.variant}:{expert}")

    # Round7 intervention must be applied after PTQ replacement / searched policy setup.
    # For bf16 this is a no-op; for quant variants it modifies activation policy.
    for expert in _experts(args.experts):
        _apply_round7_intervention(model_by_expert[expert], args, expert)

    if pipe.device.type == "cuda":
        torch.cuda.empty_cache()

    seq_len = int(static["seq_len"])
    y = static["y"].to(device=pipe.device, dtype=pipe.param_dtype)
    context_by_branch = {
        branch: [t.to(device=pipe.device, dtype=pipe.param_dtype) for t in tensors]
        for branch, tensors in static["context_by_branch"].items()
    }

    use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)

    writer = DatasetWriter(out_dir)
    tracer = TeacherForcedTracer(writer, args)

    metadata = {
        "version": "adaln_teacher_forced_dataset_v2_no_image_stageB",
        "variant": args.variant,
        "run_label": args.run_label or args.variant,
        "intervention": args.intervention,
        "clip_ratio": float(args.clip_ratio),
        "clip_exp_offset": int(args.clip_exp_offset),
        "clip_scale_method": str(args.clip_scale_method),
        "target_block": int(args.target_block),
        "ffn_gate_alpha": float(args.ffn_gate_alpha),
        "teacher_forced": True,
        "stageB_reads_image": False,
        "latent_cache": args.latent_cache,
        "ckpt_dir": args.ckpt_dir,
        "ptq_dir": args.ptq_dir,
        "act_policy_json": args.act_policy_json,
        "manifest": args.manifest,
        "max_samples": int(args.max_samples),
        "sample_steps": int(args.sample_steps),
        "size": args.size,
        "blocks": blocks,
        "branches": _branches(args.branches),
        "experts": _experts(args.experts),
        "sample_solver": args.sample_solver,
        "seed": int(args.seed),
        "max_rows": int(args.max_rows),
        "max_channels": int(args.max_channels),
        "row_seed": int(args.row_seed),
        "channel_seed": int(args.channel_seed),
        "points": [
            "self_adaln_in",
            "ffn_adaln_in",
            "self_residual_update",
            "ffn_residual_update",
            "block_out",
        ],
    }

    selected_experts = set(_experts(args.experts))
    selected_branches = set(_branches(args.branches))

    with PatchScope() as scope:
        for expert in selected_experts:
            tracer.patch_model(model_by_expert[expert], blocks, scope)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype, enabled=use_amp):
            for rec in cache_records:
                branch = rec["branch"]
                expert = rec["expert"]
                if branch not in selected_branches:
                    continue
                if expert not in selected_experts:
                    continue

                if branch not in context_by_branch:
                    raise KeyError(f"branch {branch!r} missing in cached context_by_branch")

                latent = rec["latent_before"].to(device=pipe.device, dtype=pipe.param_dtype)
                context = context_by_branch[branch]
                model_args = {"context": context, "seq_len": seq_len, "y": [y]}

                tracer.set_context(
                    sample_id=rec["sample_id"],
                    branch=branch,
                    step_idx=int(rec["step_idx"]),
                    timestep=float(rec["timestep"]),
                    expert=expert,
                    model_name=model_name_by_expert[expert],
                )

                model_by_expert[expert].to(pipe.device)
                timestep = torch.tensor([int(float(rec["timestep"]))], device=pipe.device, dtype=torch.long)
                _ = model_by_expert[expert]([latent], t=timestep, **model_args)[0]

    writer.save(metadata)


def main() -> None:
    args = parse_args()
    if args.mode == "cache":
        run_cache(args)
    elif args.mode == "trace":
        run_trace(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
