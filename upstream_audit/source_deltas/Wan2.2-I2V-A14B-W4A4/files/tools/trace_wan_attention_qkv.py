#!/usr/bin/env python3
"""Trace Wan self/cross attention QKV, RoPE, and cross-KV stability.

The script monkey-patches selected attention modules and emits only reduced
statistics. It is intended for remote exploratory runs with small sample/step
counts and narrow block selections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from quant.patch_scope import PatchScope
from quant.stream_stats import compare_tensors, norm_internal_stats, summarize_tensor
from quant.trace_context import JsonlTraceWriter, TraceContext, write_metadata
from quant.wan_smooth import _build_schedule
from tools.trace_wan_adaln_norms import _branches, _build_pipe, _experts, _load_manifest, _prep_sample, _size_to_hw
from wan.configs import WAN_CONFIGS
from wan.modules.attention import flash_attention
from wan.modules.model import WanCrossAttention, WanSelfAttention, rope_apply

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Trace Wan I2V attention QKV and cross-KV stability")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=5)
    p.add_argument("--size", type=str, default="832*480")
    p.add_argument("--blocks", type=str, default="0,10,20,30,39")
    p.add_argument("--branches", type=str, default="cond", choices=["cond", "uncond", "both"])
    p.add_argument("--experts", type=str, default="both", choices=["high", "low", "both"])
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)
    p.add_argument("--trace_self", action="store_true", default=True)
    p.add_argument("--trace_cross", action="store_true", default=True)
    p.add_argument("--kv_hash_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    p.add_argument("--list_blocks", action="store_true", default=False)
    return p.parse_args()


def _parse_csv_ints(text: str) -> List[int]:
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


def _hash_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _tensor_hash(x: torch.Tensor, dtype: torch.dtype) -> str:
    # Hashes are for stability checks only; full tensor bytes are not retained.
    y = x.detach().to(device="cpu", dtype=dtype, non_blocking=False).contiguous()
    return hashlib.sha256(y.numpy().tobytes()).hexdigest()[:24]


class KVStabilityTracker:
    """Track cross-attention K/V stability with hashes and summary deltas."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.first: Dict[Tuple[str, str, str, str, int, str], Dict[str, Any]] = {}

    def observe(self, ctx: TraceContext, *, block_idx: int, point: str, x: torch.Tensor) -> Dict[str, Any]:
        key = (ctx.sample_id, ctx.branch, ctx.expert, ctx.model_name, int(block_idx), point)
        stats = summarize_tensor(x, include_channel_stats=False, include_token_norms=False)
        digest = _tensor_hash(x, self.dtype)
        if key not in self.first:
            self.first[key] = {"hash": digest, "stats": stats, "step_idx": ctx.step_idx, "timestep": ctx.timestep}
            return {
                "kv_first_step_idx": ctx.step_idx,
                "kv_first_timestep": ctx.timestep,
                "kv_hash": digest,
                "kv_hash_equal_to_first": True,
                "kv_absmax_delta_vs_first": 0.0,
                "kv_rms_delta_vs_first": 0.0,
            }
        first = self.first[key]
        first_stats = first["stats"]
        return {
            "kv_first_step_idx": first["step_idx"],
            "kv_first_timestep": first["timestep"],
            "kv_hash": digest,
            "kv_hash_equal_to_first": digest == first["hash"],
            "kv_absmax_delta_vs_first": float(stats.get("absmax", 0.0) - first_stats.get("absmax", 0.0)),
            "kv_rms_delta_vs_first": float(stats.get("rms", 0.0) - first_stats.get("rms", 0.0)),
        }


class AttentionQKVTracer:
    def __init__(self, ctx: TraceContext, writer: JsonlTraceWriter, kv_tracker: KVStabilityTracker) -> None:
        self.ctx = ctx
        self.writer = writer
        self.kv_tracker = kv_tracker

    def emit(self, point: str, x: torch.Tensor, *, module_name: str, block_idx: int, extra: Optional[Dict[str, Any]] = None) -> None:
        rec = self.ctx.base_record(point=point, module_name=module_name, block_idx=block_idx, tensor=x, extra=extra)
        rec.update(summarize_tensor(x))
        self.writer.write(rec)

    def emit_compare(self, point: str, a: torch.Tensor, b: torch.Tensor, *, module_name: str, block_idx: int) -> None:
        rec = self.ctx.base_record(point=point, module_name=module_name, block_idx=block_idx, tensor=b)
        rec.update(compare_tensors(a, b, prefix="delta_"))
        self.writer.write(rec)

    def patch_model(self, model: nn.Module, block_indices: Sequence[int], scope: PatchScope, *, trace_self: bool, trace_cross: bool) -> None:
        for block_idx in block_indices:
            if block_idx < 0 or block_idx >= len(model.blocks):
                continue
            block = model.blocks[block_idx]
            if trace_self and isinstance(block.self_attn, WanSelfAttention):
                scope.patch_method(block.self_attn, "forward", self._make_self_forward(block_idx))
            if trace_cross and isinstance(block.cross_attn, WanCrossAttention):
                scope.patch_method(block.cross_attn, "forward", self._make_cross_forward(block_idx))

    def _make_self_forward(self, block_idx: int):
        tracer = self
        module_prefix = f"blocks.{block_idx}.self_attn"

        def traced_forward(self: WanSelfAttention, x, seq_lens, grid_sizes, freqs):
            b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
            tracer.emit("self_attn_in", x, module_name=module_prefix, block_idx=block_idx)
            q_raw = self.q(x)
            k_raw = self.k(x)
            v_raw = self.v(x)
            tracer.emit("self_q_raw", q_raw, module_name=f"{module_prefix}.q", block_idx=block_idx)
            tracer.emit("self_k_raw", k_raw, module_name=f"{module_prefix}.k", block_idx=block_idx)
            tracer.emit("self_v_raw", v_raw, module_name=f"{module_prefix}.v", block_idx=block_idx)

            q_norm = self.norm_q(q_raw)
            k_norm = self.norm_k(k_raw)
            q_extra = norm_internal_stats(q_raw, kind="rmsnorm", eps=float(getattr(self.norm_q, "eps", 1e-6)), normalized_shape=int(q_raw.shape[-1]))
            k_extra = norm_internal_stats(k_raw, kind="rmsnorm", eps=float(getattr(self.norm_k, "eps", 1e-6)), normalized_shape=int(k_raw.shape[-1]))
            tracer.emit("self_q_norm", q_norm, module_name=f"{module_prefix}.norm_q", block_idx=block_idx, extra=q_extra)
            tracer.emit("self_k_norm", k_norm, module_name=f"{module_prefix}.norm_k", block_idx=block_idx, extra=k_extra)
            tracer.emit_compare("self_q_norm_delta", q_raw, q_norm, module_name=f"{module_prefix}.norm_q", block_idx=block_idx)
            tracer.emit_compare("self_k_norm_delta", k_raw, k_norm, module_name=f"{module_prefix}.norm_k", block_idx=block_idx)

            q = q_norm.view(b, s, n, d)
            k = k_norm.view(b, s, n, d)
            v = v_raw.view(b, s, n, d)
            q_rope = rope_apply(q, grid_sizes, freqs)
            k_rope = rope_apply(k, grid_sizes, freqs)
            tracer.emit("self_q_rope", q_rope, module_name=f"{module_prefix}.rope_q", block_idx=block_idx)
            tracer.emit("self_k_rope", k_rope, module_name=f"{module_prefix}.rope_k", block_idx=block_idx)
            tracer.emit_compare("self_q_rope_delta", q, q_rope, module_name=f"{module_prefix}.rope_q", block_idx=block_idx)
            tracer.emit_compare("self_k_rope_delta", k, k_rope, module_name=f"{module_prefix}.rope_k", block_idx=block_idx)

            attn = flash_attention(q=q_rope, k=k_rope, v=v, k_lens=seq_lens, window_size=self.window_size)
            tracer.emit("self_attn_pre_o", attn, module_name=f"{module_prefix}.attn", block_idx=block_idx)
            out = self.o(attn.flatten(2))
            tracer.emit("self_o_out", out, module_name=f"{module_prefix}.o", block_idx=block_idx)
            return out

        return traced_forward

    def _make_cross_forward(self, block_idx: int):
        tracer = self
        module_prefix = f"blocks.{block_idx}.cross_attn"

        def traced_forward(self: WanCrossAttention, x, context, context_lens):
            b, n, d = x.size(0), self.num_heads, self.head_dim
            tracer.emit("cross_attn_in", x, module_name=module_prefix, block_idx=block_idx)
            q_raw = self.q(x)
            k_raw = self.k(context)
            v_raw = self.v(context)
            tracer.emit("cross_q_raw", q_raw, module_name=f"{module_prefix}.q", block_idx=block_idx)
            k_extra_stability = tracer.kv_tracker.observe(tracer.ctx, block_idx=block_idx, point="cross_k_raw", x=k_raw)
            v_extra_stability = tracer.kv_tracker.observe(tracer.ctx, block_idx=block_idx, point="cross_v_raw", x=v_raw)
            tracer.emit("cross_k_raw", k_raw, module_name=f"{module_prefix}.k", block_idx=block_idx, extra=k_extra_stability)
            tracer.emit("cross_v_raw", v_raw, module_name=f"{module_prefix}.v", block_idx=block_idx, extra=v_extra_stability)

            q_norm = self.norm_q(q_raw)
            k_norm = self.norm_k(k_raw)
            q_extra = norm_internal_stats(q_raw, kind="rmsnorm", eps=float(getattr(self.norm_q, "eps", 1e-6)), normalized_shape=int(q_raw.shape[-1]))
            k_extra = norm_internal_stats(k_raw, kind="rmsnorm", eps=float(getattr(self.norm_k, "eps", 1e-6)), normalized_shape=int(k_raw.shape[-1]))
            tracer.emit("cross_q_norm", q_norm, module_name=f"{module_prefix}.norm_q", block_idx=block_idx, extra=q_extra)
            tracer.emit("cross_k_norm", k_norm, module_name=f"{module_prefix}.norm_k", block_idx=block_idx, extra=k_extra)
            tracer.emit_compare("cross_q_norm_delta", q_raw, q_norm, module_name=f"{module_prefix}.norm_q", block_idx=block_idx)
            tracer.emit_compare("cross_k_norm_delta", k_raw, k_norm, module_name=f"{module_prefix}.norm_k", block_idx=block_idx)

            q = q_norm.view(b, -1, n, d)
            k = k_norm.view(b, -1, n, d)
            v = v_raw.view(b, -1, n, d)
            attn = flash_attention(q, k, v, k_lens=context_lens)
            tracer.emit("cross_attn_pre_o", attn, module_name=f"{module_prefix}.attn", block_idx=block_idx)
            out = self.o(attn.flatten(2))
            tracer.emit("cross_o_out", out, module_name=f"{module_prefix}.o", block_idx=block_idx)
            return out

        return traced_forward


def main() -> None:
    args = parse_args()
    block_indices = _parse_csv_ints(args.blocks)
    out_dir = Path(args.out_dir)
    samples = _load_manifest(Path(args.manifest), max_samples=int(args.max_samples))
    if not samples:
        raise ValueError("No valid manifest samples found.")

    pipe = _build_pipe(args)
    if args.list_blocks:
        print(f"num_blocks={len(pipe.high_noise_model.blocks)}")
        print(",".join(str(i) for i in range(len(pipe.high_noise_model.blocks))))
        return

    writer = JsonlTraceWriter(out_dir / "attention_qkv_stats.jsonl")
    ctx = TraceContext()
    kv_tracker = KVStabilityTracker(dtype=_hash_dtype(args.kv_hash_dtype))
    tracer = AttentionQKVTracer(ctx, writer, kv_tracker)
    metadata = {
        "ckpt_dir": args.ckpt_dir,
        "manifest": args.manifest,
        "max_samples": int(args.max_samples),
        "sample_steps": int(args.sample_steps),
        "size": args.size,
        "blocks": block_indices,
        "branches": _branches(args.branches),
        "experts": _experts(args.experts),
        "sample_solver": args.sample_solver,
        "trace_self": bool(args.trace_self),
        "trace_cross": bool(args.trace_cross),
        "kv_hash_dtype": args.kv_hash_dtype,
        "outputs": {"jsonl": str(out_dir / "attention_qkv_stats.jsonl")},
    }

    try:
        with PatchScope() as patch_scope:
            tracer.patch_model(pipe.high_noise_model, block_indices, patch_scope, trace_self=args.trace_self, trace_cross=args.trace_cross)
            tracer.patch_model(pipe.low_noise_model, block_indices, patch_scope, trace_self=args.trace_self, trace_cross=args.trace_cross)
            cfg = WAN_CONFIGS["i2v-A14B"]
            frame_num = cfg.frame_num
            shift = cfg.sample_shift if args.shift is None else float(args.shift)
            boundary = float(pipe.boundary * pipe.num_train_timesteps)
            use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)
            model_by_expert = {"high": pipe.high_noise_model, "low": pipe.low_noise_model}
            model_name_by_expert = {"high": "high_noise_model", "low": "low_noise_model"}

            sample_iter: Iterable[Dict[str, str]] = samples
            if tqdm is not None and not args.disable_tqdm:
                sample_iter = tqdm(samples, desc="attention qkv samples", unit="sample", dynamic_ncols=True)
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
