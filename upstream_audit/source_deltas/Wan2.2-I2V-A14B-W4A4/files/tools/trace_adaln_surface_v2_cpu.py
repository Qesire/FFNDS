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
from quant.trace_context import write_metadata
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

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Trace AdaLN intervention surface V2, CPU sampled")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
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

    p.add_argument("--clip_grid", type=str, default="1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.6")
    p.add_argument("--exp_grid", type=str, default="-2,-1,0,1,2")
    p.add_argument("--gate_grid", type=str, default="1.0,0.95,0.9,0.85,0.8,0.75,0.7")
    p.add_argument("--act_group_size", type=int, default=32)
    p.add_argument("--max_surface_rows", type=int, default=256)

    p.add_argument("--disable_tqdm", action="store_true", default=False)
    return p.parse_args()


def _parse_float_csv(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8")

    def write(self, rec: Dict[str, Any]) -> None:
        self.f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    def flush(self) -> None:
        self.f.flush()

    def close(self) -> None:
        self.f.close()


class RunContext:
    def __init__(self) -> None:
        self.sample_id = ""
        self.prompt = ""
        self.branch = ""
        self.step_idx = -1
        self.timestep = float("nan")
        self.expert = ""
        self.model_name = ""

    def begin_sample(self, sample_id: str, prompt: str) -> None:
        self.sample_id = sample_id
        self.prompt = prompt

    def begin_branch(self, branch: str) -> None:
        self.branch = branch

    def begin_step(self, step_idx: int, timestep: float, expert: str, model_name: str) -> None:
        self.step_idx = int(step_idx)
        self.timestep = float(timestep)
        self.expert = expert
        self.model_name = model_name

    def base(self, *, point: str, module_name: str, block_idx: int, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "branch": self.branch,
            "step_idx": self.step_idx,
            "timestep": self.timestep,
            "expert": self.expert,
            "model_name": self.model_name,
            "block_idx": int(block_idx),
            "module_name": module_name,
            "point": point,
        }
        if extra:
            rec.update(extra)
        return rec


@torch.no_grad()
def _sample_rows_cpu(x: torch.Tensor, max_rows: int = 256) -> torch.Tensor:
    """Sample over non-channel dimensions and move to CPU.

    Keeps full last/channel dimension. No CUDA quantile, no CUDA QDQ, no CUDA gather.
    """
    xd = x.detach()
    if xd.ndim == 0:
        return xd.to(torch.float32).reshape(1, 1).cpu().contiguous()

    c = int(xd.shape[-1])
    x2 = xd.to(torch.float32).reshape(-1, c)
    n = int(x2.shape[0])

    if n > int(max_rows):
        step = max(1, math.ceil(n / int(max_rows)))
        x2 = x2[::step][: int(max_rows)]

    xcpu = x2.cpu().contiguous()
    return torch.nan_to_num(xcpu, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def _stats_cpu(x: torch.Tensor, prefix: str = "") -> Dict[str, Any]:
    x = torch.nan_to_num(x.detach().to(torch.float32).cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = x.reshape(-1)
    abs_flat = flat.abs()
    n = int(abs_flat.numel())

    if n == 0:
        return {prefix + "numel": 0}

    def q(v: float) -> float:
        if n == 1:
            return float(abs_flat[0].item())
        return float(torch.quantile(abs_flat, torch.tensor(v, dtype=torch.float32)).item())

    rms = float(torch.sqrt(torch.mean(flat * flat)).item())
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item()) if n > 1 else 0.0
    absmax = float(abs_flat.max().item())
    abs_p50 = q(0.50)
    abs_p90 = q(0.90)
    abs_p95 = q(0.95)
    abs_p99 = q(0.99)
    abs_p999 = q(0.999)

    out: Dict[str, Any] = {
        prefix + "numel": n,
        prefix + "shape": "x".join(map(str, x.shape)),
        prefix + "mean": mean,
        prefix + "std": std,
        prefix + "rms": rms,
        prefix + "absmax": absmax,
        prefix + "abs_p50": abs_p50,
        prefix + "abs_p90": abs_p90,
        prefix + "abs_p95": abs_p95,
        prefix + "abs_p99": abs_p99,
        prefix + "abs_p999": abs_p999,
        prefix + "absmax_over_p99": absmax / max(abs_p99, 1e-30),
        prefix + "p99_over_p50": abs_p99 / max(abs_p50, 1e-30),
    }

    if x.ndim >= 2:
        x2 = x.reshape(-1, int(x.shape[-1]))
        ch_absmax = x2.abs().amax(dim=0)
        ch_rms = torch.sqrt(torch.mean(x2 * x2, dim=0))
        out[prefix + "channel_absmax_max"] = float(ch_absmax.max().item())
        out[prefix + "channel_absmax_p99"] = float(torch.quantile(ch_absmax, torch.tensor(0.99)).item())
        out[prefix + "channel_rms_max"] = float(ch_rms.max().item())
        out[prefix + "channel_rms_p99"] = float(torch.quantile(ch_rms, torch.tensor(0.99)).item())
        out[prefix + "channels"] = int(x2.shape[-1])
        out[prefix + "rows"] = int(x2.shape[0])

    return out


@torch.no_grad()
def _compare_cpu(a: torch.Tensor, b: torch.Tensor, prefix: str = "") -> Dict[str, Any]:
    a = torch.nan_to_num(a.detach().to(torch.float32).cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    b = torch.nan_to_num(b.detach().to(torch.float32).cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    diff = b - a

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    d_flat = diff.reshape(-1)

    a_l2 = float(torch.linalg.vector_norm(a_flat).item())
    b_l2 = float(torch.linalg.vector_norm(b_flat).item())
    d_l2 = float(torch.linalg.vector_norm(d_flat).item())
    rmse = float(torch.sqrt(torch.mean(d_flat * d_flat)).item())
    denom = max(a_l2 * b_l2, 1e-30)
    cosine = float(torch.dot(a_flat, b_flat).item() / denom)

    return {
        prefix + "l2_ref": a_l2,
        prefix + "l2_test": b_l2,
        prefix + "l2_error": d_l2,
        prefix + "relative_l2": d_l2 / max(a_l2, 1e-30),
        prefix + "rmse": rmse,
        prefix + "cosine": cosine,
    }


@torch.no_grad()
def _mxfp4_qdq_cpu(
    x: torch.Tensor,
    group_size: int = 32,
    clip_ratio: float = 1.0,
    exp_offset: int = 0,
    fp4_max: float = 6.0,
) -> torch.Tensor:
    """CPU diagnostic MXFP4-like QDQ.

    This is for ranking intervention surfaces only. It avoids CUDA kernels and
    avoids pack/unpack. It approximates E2M1 values with {0, .5, 1, 1.5, 2, 3, 4, 6}.
    """
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")

    x = torch.nan_to_num(x.detach().to(torch.float32).cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    orig_shape = tuple(x.shape)

    if x.ndim == 0:
        x = x.reshape(1, 1)

    c = int(x.shape[-1])
    x2 = x.reshape(-1, c)

    pad = (int(group_size) - (c % int(group_size))) % int(group_size)
    if pad:
        x2 = torch.nn.functional.pad(x2, (0, pad))

    c_pad = int(x2.shape[-1])
    xg = x2.reshape(-1, c_pad // int(group_size), int(group_size))

    amax = xg.abs().amax(dim=-1) * float(clip_ratio)
    zero = amax <= 0
    safe_amax = torch.where(zero, torch.ones_like(amax), amax)

    exp = torch.ceil(torch.log2(safe_amax / float(fp4_max))) + int(exp_offset)
    exp = torch.where(zero, torch.zeros_like(exp), exp)
    exp = exp.clamp(-127, 127)
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32), exp).clamp_min(1e-30)

    xs = (xg / scale.unsqueeze(-1)).clamp(-float(fp4_max), float(fp4_max))
    ax = xs.abs()

    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    idx = torch.bucketize(ax.reshape(-1), thresholds, right=False).reshape_as(ax).long()
    mag = values[idx]
    sign = torch.where(xs < 0, -torch.ones_like(mag), torch.ones_like(mag))
    dq = mag * sign * scale.unsqueeze(-1)

    out = dq.reshape(-1, c_pad)[..., :c].reshape(orig_shape)
    return out.contiguous()


def _feature_stats(prefix: str, x: torch.Tensor, max_rows: int) -> Dict[str, Any]:
    xs = _sample_rows_cpu(x, max_rows=max_rows)
    return _stats_cpu(xs, prefix=prefix + "_")


class AdaLNSurfaceTracerV2:
    def __init__(
        self,
        ctx: RunContext,
        writer: JsonlWriter,
        *,
        clip_grid: Sequence[float],
        exp_grid: Sequence[int],
        gate_grid: Sequence[float],
        act_group_size: int,
        max_surface_rows: int,
    ) -> None:
        self.ctx = ctx
        self.writer = writer
        self.clip_grid = list(clip_grid)
        self.exp_grid = list(exp_grid)
        self.gate_grid = list(gate_grid)
        self.act_group_size = int(act_group_size)
        self.max_surface_rows = int(max_surface_rows)

    def patch_model(self, model: nn.Module, block_indices: Sequence[int], scope: PatchScope) -> None:
        for block_idx in block_indices:
            if block_idx < 0 or block_idx >= len(model.blocks):
                continue
            block = model.blocks[block_idx]
            if isinstance(block, WanAttentionBlock):
                scope.patch_method(block, "forward", self._make_block_forward(block_idx))

    def _emit_quant_surface(
        self,
        *,
        point: str,
        module_name: str,
        block_idx: int,
        x: torch.Tensor,
        features: Dict[str, Any],
    ) -> None:
        x_cpu = _sample_rows_cpu(x, max_rows=self.max_surface_rows)
        src_stats = _stats_cpu(x_cpu, prefix="src_")

        for clip_ratio in self.clip_grid:
            for exp_offset in self.exp_grid:
                qdq = _mxfp4_qdq_cpu(
                    x_cpu,
                    group_size=self.act_group_size,
                    clip_ratio=float(clip_ratio),
                    exp_offset=int(exp_offset),
                )
                rec = self.ctx.base(
                    point=f"{point}_mxfp4_surface",
                    module_name=module_name,
                    block_idx=block_idx,
                    extra={
                        "source_point": point,
                        "intervention_type": "quantizer_only_cpu_sampled_v2",
                        "surface_device": "cpu",
                        "surface_rows": int(x_cpu.reshape(-1, x_cpu.shape[-1]).shape[0]),
                        "surface_channels": int(x_cpu.shape[-1]),
                        "clip_ratio": float(clip_ratio),
                        "exp_offset": int(exp_offset),
                        "act_group_size": int(self.act_group_size),
                        **features,
                        **src_stats,
                        **_compare_cpu(x_cpu, qdq, prefix="qdq_"),
                    },
                )
                self.writer.write(rec)

    def _emit_gate_surface(
        self,
        *,
        point: str,
        module_name: str,
        block_idx: int,
        update: torch.Tensor,
        features: Dict[str, Any],
    ) -> None:
        u_cpu = _sample_rows_cpu(update, max_rows=self.max_surface_rows)
        base_stats = _stats_cpu(u_cpu, prefix="base_")

        for alpha in self.gate_grid:
            damped = u_cpu * float(alpha)
            damp_stats = _stats_cpu(damped, prefix="damped_")
            rec = self.ctx.base(
                point=f"{point}_gate_damper_surface",
                module_name=module_name,
                block_idx=block_idx,
                extra={
                    "source_point": point,
                    "intervention_type": "gate_damper_cpu_sampled_v2",
                    "surface_device": "cpu",
                    "surface_rows": int(u_cpu.reshape(-1, u_cpu.shape[-1]).shape[0]),
                    "surface_channels": int(u_cpu.shape[-1]),
                    "gate_alpha": float(alpha),
                    **features,
                    **base_stats,
                    **damp_stats,
                },
            )
            self.writer.write(rec)

    def _emit_stats_only(
        self,
        *,
        point: str,
        module_name: str,
        block_idx: int,
        x: torch.Tensor,
        features: Dict[str, Any] | None = None,
    ) -> None:
        x_cpu = _sample_rows_cpu(x, max_rows=self.max_surface_rows)
        rec = self.ctx.base(
            point=f"{point}_stats",
            module_name=module_name,
            block_idx=block_idx,
            extra={
                "source_point": point,
                "intervention_type": "stats_only_cpu_sampled_v2",
                "surface_device": "cpu",
                "surface_rows": int(x_cpu.reshape(-1, x_cpu.shape[-1]).shape[0]),
                "surface_channels": int(x_cpu.shape[-1]),
                **(features or {}),
                **_stats_cpu(x_cpu, prefix="src_"),
            },
        )
        self.writer.write(rec)

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

            features: Dict[str, Any] = {}
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
                features.update(_feature_stats(name, ten, tracer.max_surface_rows))

            tracer._emit_stats_only(
                point="block_in",
                module_name=f"blocks.{block_idx}",
                block_idx=block_idx,
                x=x,
                features=features,
            )

            norm1_out = self.norm1(x)
            self_in = norm1_out.float() * (1 + e_scale_self) + e_shift_self
            tracer._emit_quant_surface(
                point="self_adaln_in",
                module_name=f"blocks.{block_idx}.self_adaln",
                block_idx=block_idx,
                x=self_in,
                features=features,
            )

            y = self.self_attn(self_in, seq_lens, grid_sizes, freqs)

            with torch.amp.autocast("cuda", dtype=torch.float32):
                self_update = y * e_gate_self
                tracer._emit_gate_surface(
                    point="self_residual_update",
                    module_name=f"blocks.{block_idx}.self_residual",
                    block_idx=block_idx,
                    update=self_update,
                    features=features,
                )
                x = x + self_update

            norm3_out = self.norm3(x)
            cross_out = self.cross_attn(norm3_out, context, context_lens)
            x = x + cross_out

            norm2_out = self.norm2(x)
            ffn_in = norm2_out.float() * (1 + e_scale_ffn) + e_shift_ffn
            tracer._emit_quant_surface(
                point="ffn_adaln_in",
                module_name=f"blocks.{block_idx}.ffn_adaln",
                block_idx=block_idx,
                x=ffn_in,
                features=features,
            )

            y = self.ffn(ffn_in)

            with torch.amp.autocast("cuda", dtype=torch.float32):
                ffn_update = y * e_gate_ffn
                tracer._emit_gate_surface(
                    point="ffn_residual_update",
                    module_name=f"blocks.{block_idx}.ffn_residual",
                    block_idx=block_idx,
                    update=ffn_update,
                    features=features,
                )
                x = x + ffn_update

            tracer._emit_stats_only(
                point="block_out",
                module_name=f"blocks.{block_idx}",
                block_idx=block_idx,
                x=x,
                features=features,
            )
            return x

        return traced_forward


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_manifest(Path(args.manifest), max_samples=int(args.max_samples))
    if not samples:
        raise ValueError("No valid manifest samples found")

    blocks = _parse_csv_ints(args.blocks)
    clip_grid = _parse_float_csv(args.clip_grid)
    exp_grid = _parse_int_csv(args.exp_grid)
    gate_grid = _parse_float_csv(args.gate_grid)

    writer = JsonlWriter(out_dir / "adaln_intervention_surface_v2.jsonl")
    ctx = RunContext()

    pipe = _build_pipe(args)

    tracer = AdaLNSurfaceTracerV2(
        ctx,
        writer,
        clip_grid=clip_grid,
        exp_grid=exp_grid,
        gate_grid=gate_grid,
        act_group_size=int(args.act_group_size),
        max_surface_rows=int(args.max_surface_rows),
    )

    cfg = WAN_CONFIGS["i2v-A14B"]
    frame_num = cfg.frame_num
    shift = cfg.sample_shift if args.shift is None else float(args.shift)
    boundary = float(pipe.boundary * pipe.num_train_timesteps)
    use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)

    model_by_expert = {"high": pipe.high_noise_model, "low": pipe.low_noise_model}
    model_name_by_expert = {"high": "high_noise_model", "low": "low_noise_model"}

    metadata = {
        "version": "v2_cpu_sampled_no_stream_stats",
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
        "max_surface_rows": int(args.max_surface_rows),
        "outputs": {"jsonl": str(out_dir / "adaln_intervention_surface_v2.jsonl")},
    }

    try:
        with PatchScope() as scope:
            for expert in _experts(args.experts):
                tracer.patch_model(model_by_expert[expert], blocks, scope)

            sample_iter: Iterable[Dict[str, str]] = samples
            if tqdm is not None and not args.disable_tqdm:
                sample_iter = tqdm(samples, desc="adaln surface v2", unit="sample", dynamic_ncols=True)

            for sample_idx, sample in enumerate(sample_iter):
                torch.manual_seed(int(args.seed) + sample_idx)
                ctx.begin_sample(sample_id=sample["sample_id"], prompt=sample["prompt"])

                item = _prep_sample(pipe, sample, frame_num=frame_num, target_hw=_size_to_hw(args.size))
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

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=pipe.param_dtype, enabled=use_amp):
                    for branch in _branches(args.branches):
                        latent = latent0.clone()
                        ctx.begin_branch(branch)

                        context = [t.to(device=pipe.device, dtype=pipe.param_dtype) for t in item["context"][branch]]
                        model_args = {"context": context, "seq_len": int(item["seq_len"]), "y": [y]}

                        for step_idx, t_scalar in enumerate(timesteps_list):
                            expert = "high" if t_scalar >= boundary else "low"
                            if expert not in _experts(args.experts):
                                continue

                            ctx.begin_step(
                                step_idx=step_idx,
                                timestep=t_scalar,
                                expert=expert,
                                model_name=model_name_by_expert[expert],
                            )

                            timestep = torch.tensor([int(t_scalar)], device=pipe.device, dtype=torch.long)
                            t_for_step = torch.tensor(t_scalar, device=pipe.device, dtype=timesteps.dtype)

                            noise_pred = model_by_expert[expert]([latent], t=timestep, **model_args)[0]
                            latent = scheduler.step(
                                noise_pred.unsqueeze(0),
                                t_for_step,
                                latent.unsqueeze(0),
                                return_dict=False,
                            )[0].squeeze(0)

                        writer.flush()
    finally:
        writer.close()

    write_metadata(out_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
