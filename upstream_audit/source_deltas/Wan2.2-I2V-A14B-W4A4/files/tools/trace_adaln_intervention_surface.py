#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from quant.stream_stats import compare_tensors, summarize_tensor
from quant.trace_context import JsonlTraceWriter, TraceContext, write_metadata
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
    p = argparse.ArgumentParser("Collect AdaLN-conditioned intervention surfaces")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=6)
    p.add_argument("--size", type=str, default="480*832")
    p.add_argument("--blocks", type=str, default="0,16,22,37")
    p.add_argument("--branches", type=str, default="cond", choices=["cond", "uncond", "both"])
    p.add_argument("--experts", type=str, default="both", choices=["high", "low", "both"])
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)
    p.add_argument("--clip_grid", type=str, default="1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.6")
    p.add_argument("--exp_grid", type=str, default="-2,-1,0,1,2")
    p.add_argument("--gate_grid", type=str, default="1.0,0.95,0.9,0.85,0.8,0.75,0.7")
    p.add_argument("--act_group_size", type=int, default=32)
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    return p.parse_args()


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _compact_feature(prefix: str, x: torch.Tensor) -> Dict[str, Any]:
    """Summarize AdaLN feature on CPU sampled rows.

    Never run summarize_tensor/quantile directly on CUDA here, because CUDA
    quantile/scatter can trigger device-side asserts and poison the context.
    """
    try:
        xs = _sample_lastdim_rows_cpu(x, max_rows=512)
    except NameError:
        xd = x.detach()
        xs = xd.to(torch.float32).reshape(-1, int(xd.shape[-1])).cpu().contiguous()
    s = summarize_tensor(xs, include_channel_stats=True, include_token_norms=False)
    keep = [
        "rms", "absmax", "abs_p99", "abs_p999", "absmax_over_p99",
        "p99_over_p50", "channel_absmax_max", "channel_absmax_p99",
        "channel_rms_max", "channel_rms_p99",
    ]
    return {f"{prefix}_{k}": s.get(k) for k in keep if k in s}



@torch.no_grad()


def _sample_lastdim_rows_cpu(x: torch.Tensor, max_rows: int = 512) -> torch.Tensor:
    """Uniformly sample rows over all non-channel dimensions and move to CPU.

    Keeps the full channel dimension so MXFP4 group-of-32 structure is preserved.
    """
    with torch.no_grad():
        xd = x.detach()
        if xd.ndim == 0:
            return xd.to(torch.float32).reshape(1, 1).cpu().contiguous()
        x2 = xd.to(torch.float32).reshape(-1, int(xd.shape[-1]))
        n = int(x2.shape[0])
        if n > int(max_rows):
            idx = torch.linspace(0, n - 1, steps=int(max_rows), device=x2.device).long()
            x2 = x2.index_select(0, idx)
        return x2.cpu().contiguous()


def _mxfp4_qdq_cpu_no_pack(
    x: torch.Tensor,
    group_size: int = 32,
    clip_ratio: float = 1.0,
    exp_offset: int = 0,
    fp4_max: float = 6.0,
) -> torch.Tensor:
    """CPU-only MXFP4-ish QDQ surface.

    This intentionally avoids pack/unpack and CUDA kernels. It is for diagnostic
    intervention-surface ranking, not a replacement for runtime quant kernels.
    """
    if group_size != 32:
        raise ValueError(f"expected group_size=32, got {group_size}")
    if clip_ratio <= 0:
        raise ValueError(f"clip_ratio must be > 0, got {clip_ratio}")

    x32 = torch.nan_to_num(x.detach().to(torch.float32).cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    orig_shape = tuple(x32.shape)
    if x32.ndim == 0:
        x32 = x32.reshape(1, 1)
    c = int(x32.shape[-1])
    x2 = x32.reshape(-1, c)

    pad = (group_size - (c % group_size)) % group_size
    if pad:
        x2 = torch.nn.functional.pad(x2, (0, pad))
    c_pad = int(x2.shape[-1])
    xg = x2.reshape(-1, c_pad // group_size, group_size)

    amax = xg.abs().amax(dim=-1) * float(clip_ratio)
    zero_mask = amax == 0
    safe_amax = torch.where(zero_mask, torch.ones_like(amax), amax)
    exp = torch.ceil(torch.log2(safe_amax / float(fp4_max))) + int(exp_offset)
    exp = torch.where(zero_mask, torch.zeros_like(exp), exp)
    exp = exp.clamp(-127, 127)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32), exp).clamp_min(1e-30)

    xs = (xg / scale.unsqueeze(-1)).clamp(-float(fp4_max), float(fp4_max))
    ax = xs.abs()

    # FP4 E2M1 positive values and thresholds:
    # values:     0, .5, 1, 1.5, 2, 3, 4, 6
    # thresholds: .25,.75,1.25,1.75,2.5,3.5,5
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    idx = torch.bucketize(ax.reshape(-1), thresholds, right=False).reshape_as(ax).long()
    mag = values[idx]
    sign = torch.where(xs < 0, -torch.ones_like(mag), torch.ones_like(mag))
    dq = mag * sign * scale.unsqueeze(-1)

    out = dq.reshape(-1, c_pad)[..., :c].reshape(orig_shape)
    return out.contiguous()


def _mxfp4_surface_rows(
    *,
    ctx: TraceContext,
    tensor: torch.Tensor,
    point: str,
    module_name: str,
    block_idx: int,
    clip_grid: Sequence[float],
    exp_grid: Sequence[int],
    act_group_size: int,
    feature_extra: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_stats = summarize_tensor(tensor, include_channel_stats=True, include_token_norms=True)

    for clip_ratio in clip_grid:
        for exp_offset in exp_grid:
            qdq = _qdq_mxfp4_activation_groupwise(
                tensor,
                group_size=int(act_group_size),
                clip_ratio=float(clip_ratio),
                channels_dim=-1,
                exp_offset=int(exp_offset),
                scale_method="safe_ceil",
            )
            err = compare_tensors(tensor, qdq, prefix="qdq_")
            rec = ctx.base_record(
                point=f"{point}_mxfp4_surface",
                module_name=module_name,
                block_idx=block_idx,
                tensor=tensor,
                extra={
                    "source_point": point,
                    "intervention_type": "quantizer_only",
                    "clip_ratio": float(clip_ratio),
                    "exp_offset": int(exp_offset),
                    "act_group_size": int(act_group_size),
                    **feature_extra,
                    **err,
                },
            )
            for k in [
                "rms", "absmax", "abs_p99", "abs_p999", "absmax_over_p99",
                "p99_over_p50", "channel_absmax_max", "channel_absmax_p99",
                "token_l2_p99", "token_l2_max",
            ]:
                if k in base_stats:
                    rec[f"src_{k}"] = base_stats[k]
            rows.append(rec)
    return rows


@torch.no_grad()
def _gate_rows(
    *,
    ctx: TraceContext,
    update: torch.Tensor,
    point: str,
    module_name: str,
    block_idx: int,
    gate_grid: Sequence[float],
    feature_extra: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    update_cpu = _sample_lastdim_rows_cpu(update, max_rows=512)
    base = summarize_tensor(update_cpu, include_channel_stats=True, include_token_norms=True)
    base_absmax = float(base.get("absmax", 0.0) or 0.0)
    base_p99 = float(base.get("abs_p99", 0.0) or 0.0)
    base_rms = float(base.get("rms", 0.0) or 0.0)

    for alpha in gate_grid:
        damped = update_cpu * float(alpha)
        s = summarize_tensor(damped, include_channel_stats=True, include_token_norms=True)
        rec = ctx.base_record(
            point=f"{point}_gate_damper_proxy",
            module_name=module_name,
            block_idx=block_idx,
            tensor=damped,
            extra={
                "source_point": point,
                "intervention_type": "gate_damper_proxy_cpu_sampled",
                "surface_device": "cpu",
                "surface_rows": int(update_cpu.reshape(-1, update_cpu.shape[-1]).shape[0]),
                "surface_channels": int(update_cpu.shape[-1]),
                "gate_alpha": float(alpha),
                "base_rms": base_rms,
                "base_absmax": base_absmax,
                "base_abs_p99": base_p99,
                "damped_rms": s.get("rms"),
                "damped_absmax": s.get("absmax"),
                "damped_abs_p99": s.get("abs_p99"),
                "damped_absmax_over_p99": s.get("absmax_over_p99"),
                "rms_ratio_vs_base": float(s.get("rms", 0.0) / max(base_rms, 1e-30)),
                "absmax_ratio_vs_base": float(s.get("absmax", 0.0) / max(base_absmax, 1e-30)),
                **feature_extra,
            },
        )
        rows.append(rec)
    return rows


class AdaLNInterventionSurfaceTracer:
    def __init__(
        self,
        ctx: TraceContext,
        writer: JsonlTraceWriter,
        *,
        clip_grid: Sequence[float],
        exp_grid: Sequence[int],
        gate_grid: Sequence[float],
        act_group_size: int,
    ) -> None:
        self.ctx = ctx
        self.writer = writer
        self.clip_grid = list(clip_grid)
        self.exp_grid = list(exp_grid)
        self.gate_grid = list(gate_grid)
        self.act_group_size = int(act_group_size)

    def patch_model(self, model: nn.Module, block_indices: Sequence[int], scope: PatchScope) -> None:
        for block_idx in block_indices:
            if block_idx < 0 or block_idx >= len(model.blocks):
                continue
            block = model.blocks[block_idx]
            if isinstance(block, WanAttentionBlock):
                scope.patch_method(block, "forward", self._make_block_forward(block_idx))

    def _write_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        for r in rows:
            self.writer.write(r)

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

            e_features: Dict[str, Any] = {}
            for name, ten in [
                ("e_shift_self", e_shift_self),
                ("e_scale_self", e_scale_self),
                ("e_gate_self", e_gate_self),
                ("e_shift_ffn", e_shift_ffn),
                ("e_scale_ffn", e_scale_ffn),
                ("e_gate_ffn", e_gate_ffn),
                ("one_plus_e_scale_self", 1 + e_scale_self),
                ("one_plus_e_scale_ffn", 1 + e_scale_ffn),
            ]:
                e_features.update(_compact_feature(name, ten))

            norm1_out = self.norm1(x)
            self_in = norm1_out.float() * (1 + e_scale_self) + e_shift_self
            tracer._write_many(_mxfp4_surface_rows(
                ctx=tracer.ctx,
                tensor=self_in,
                point="self_adaln_in",
                module_name=f"blocks.{block_idx}.self_adaln",
                block_idx=block_idx,
                clip_grid=tracer.clip_grid,
                exp_grid=tracer.exp_grid,
                act_group_size=tracer.act_group_size,
                feature_extra=e_features,
            ))

            y = self.self_attn(self_in, seq_lens, grid_sizes, freqs)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                self_update = y * e_gate_self
                tracer._write_many(_gate_rows(
                    ctx=tracer.ctx,
                    update=self_update,
                    point="self_residual_update",
                    module_name=f"blocks.{block_idx}.self_residual",
                    block_idx=block_idx,
                    gate_grid=tracer.gate_grid,
                    feature_extra=e_features,
                ))
                x = x + self_update

            norm3_out = self.norm3(x)
            cross_out = self.cross_attn(norm3_out, context, context_lens)
            x = x + cross_out

            norm2_out = self.norm2(x)
            ffn_in = norm2_out.float() * (1 + e_scale_ffn) + e_shift_ffn
            tracer._write_many(_mxfp4_surface_rows(
                ctx=tracer.ctx,
                tensor=ffn_in,
                point="ffn_adaln_in",
                module_name=f"blocks.{block_idx}.ffn_adaln",
                block_idx=block_idx,
                clip_grid=tracer.clip_grid,
                exp_grid=tracer.exp_grid,
                act_group_size=tracer.act_group_size,
                feature_extra=e_features,
            ))

            y = self.ffn(ffn_in)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                ffn_update = y * e_gate_ffn
                tracer._write_many(_gate_rows(
                    ctx=tracer.ctx,
                    update=ffn_update,
                    point="ffn_residual_update",
                    module_name=f"blocks.{block_idx}.ffn_residual",
                    block_idx=block_idx,
                    gate_grid=tracer.gate_grid,
                    feature_extra=e_features,
                ))
                x = x + ffn_update
            return x

        return traced_forward


def main() -> None:
    args = parse_args()
    samples = _load_manifest(Path(args.manifest), max_samples=int(args.max_samples))
    if not samples:
        raise ValueError("No valid manifest samples found")

    blocks = _parse_csv_ints(args.blocks)
    clip_grid = _parse_floats(args.clip_grid)
    exp_grid = _parse_ints(args.exp_grid)
    gate_grid = _parse_floats(args.gate_grid)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlTraceWriter(out_dir / "adaln_intervention_surface.jsonl")
    ctx = TraceContext()
    tracer = AdaLNInterventionSurfaceTracer(
        ctx,
        writer,
        clip_grid=clip_grid,
        exp_grid=exp_grid,
        gate_grid=gate_grid,
        act_group_size=int(args.act_group_size),
    )

    pipe = _build_pipe(args)

    cfg = WAN_CONFIGS["i2v-A14B"]
    frame_num = cfg.frame_num
    shift = cfg.sample_shift if args.shift is None else float(args.shift)
    boundary = float(pipe.boundary * pipe.num_train_timesteps)
    use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)

    models = {"high": pipe.high_noise_model, "low": pipe.low_noise_model}
    model_name_by_expert = {"high": "high_noise_model", "low": "low_noise_model"}

    try:
        with PatchScope() as scope:
            for expert in _experts(args.experts):
                tracer.patch_model(models[expert], blocks, scope)

            active_expert = None
            for sample_idx, sample in enumerate(samples):
                torch.manual_seed(int(args.seed) + sample_idx)
                ctx.begin_sample(sample_id=sample["sample_id"], prompt=sample["prompt"])
                item = _prep_sample(pipe, sample, frame_num=frame_num, target_hw=_size_to_hw(args.size))
                base_latent = item["latent"].to(device=pipe.device, dtype=pipe.param_dtype)
                y = item["y"].to(device=pipe.device, dtype=pipe.param_dtype)
                scheduler, timesteps = _build_schedule(
                    sample_solver=args.sample_solver,
                    sampling_steps=int(args.sample_steps),
                    shift=shift,
                    num_train_timesteps=pipe.num_train_timesteps,
                    device=pipe.device,
                )
                timesteps_list = [float(t.item()) for t in timesteps]

                for branch in _branches(args.branches):
                    ctx.begin_branch(branch)
                    latent = base_latent.clone()
                    context = [t.to(device=pipe.device, dtype=pipe.param_dtype) for t in item["context"][branch]]
                    model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}

                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype, enabled=use_amp):
                        for step_idx, t_scalar in enumerate(timesteps_list):
                            expert = "high" if t_scalar >= boundary else "low"
                            if expert not in _experts(args.experts):
                                continue
                            if active_expert != expert:
                                for name, model in models.items():
                                    model.to(pipe.device if name == expert else "cpu")
                                active_expert = expert
                                if pipe.device.type == "cuda":
                                    torch.cuda.empty_cache()

                            ctx.begin_step(
                                step_idx=step_idx,
                                timestep=t_scalar,
                                expert=expert,
                                model_name=model_name_by_expert[expert],
                            )
                            timestep = torch.tensor([int(t_scalar)], device=pipe.device, dtype=torch.long)
                            t_for_step = torch.tensor(t_scalar, device=pipe.device, dtype=timesteps.dtype)
                            noise_pred = models[expert]([latent], t=timestep, **model_args)[0]
                            latent = scheduler.step(
                                noise_pred.unsqueeze(0),
                                t_for_step,
                                latent.unsqueeze(0),
                                return_dict=False,
                            )[0].squeeze(0)
                    writer.flush()
    finally:
        writer.close()

    write_metadata(
        out_dir / "metadata.json",
        {
            "ckpt_dir": args.ckpt_dir,
            "manifest": args.manifest,
            "max_samples": int(args.max_samples),
            "sample_steps": int(args.sample_steps),
            "size": args.size,
            "blocks": blocks,
            "branches": _branches(args.branches),
            "experts": _experts(args.experts),
            "clip_grid": clip_grid,
            "exp_grid": exp_grid,
            "gate_grid": gate_grid,
            "act_group_size": int(args.act_group_size),
            "outputs": {"jsonl": str(out_dir / "adaln_intervention_surface.jsonl")},
        },
    )


if __name__ == "__main__":
    main()
