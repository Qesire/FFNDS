#!/usr/bin/env python3
"""Trace Wan block normalization, AdaLN, and norm quantization proxies.

This script prepares the hook/patch machinery needed for future remote runs.
It does not store activations: every observed tensor is immediately reduced to
JSONL records.
"""

from __future__ import annotations

import argparse
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
import torchvision.transforms.functional as TF
from PIL import Image

import wan
from quant.patch_scope import PatchScope
from quant.stream_stats import (
    affine_param_stats,
    compare_tensors,
    fake_int_symmetric_qdq,
    norm_internal_stats,
    summarize_tensor,
)
from quant.trace_context import JsonlTraceWriter, TraceContext, write_metadata
from quant.wan_smooth import _build_schedule
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.modules.model import WanAttentionBlock

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Trace Wan I2V norm/AdaLN distributions")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=5)
    p.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--blocks", type=str, default="0,10,20,30,39")
    p.add_argument("--branches", type=str, default="cond", choices=["cond", "uncond", "both"])
    p.add_argument("--experts", type=str, default="both", choices=["high", "low", "both"])
    p.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--t5_cpu", action="store_true", default=False)
    p.add_argument("--convert_model_dtype", action="store_true", default=True)
    p.add_argument("--norm_proxy_bits", type=int, default=8)
    p.add_argument("--disable_tqdm", action="store_true", default=False)
    p.add_argument("--list_blocks", action="store_true", default=False)
    return p.parse_args()


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _size_to_hw(size: str) -> Tuple[int, int]:
    w, h = [int(x) for x in size.split("*")]
    return h, w


def _branches(arg: str) -> List[str]:
    return ["cond", "uncond"] if arg == "both" else [arg]


def _experts(arg: str) -> List[str]:
    return ["high", "low"] if arg == "both" else [arg]


def _load_manifest(path: Path, max_samples: int) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_samples > 0 and len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            raw_image = rec.get("used_image") or rec.get("image") or rec.get("image_path")
            if raw_image is None and isinstance(rec.get("img_paths"), list) and rec["img_paths"]:
                raw_image = rec["img_paths"][0]
            if raw_image is None:
                continue
            image = Path(str(raw_image))
            if not image.is_absolute():
                candidates = [path.parent / image, Path(PROJECT_ROOT) / image, Path(PROJECT_ROOT) / "OpenS2V-Eval" / image]
                image = next((c for c in candidates if c.exists()), candidates[0])
            if not image.exists():
                continue
            samples.append(
                {
                    "sample_id": str(rec.get("videoid") or rec.get("id") or rec.get("name") or f"sample_{line_idx:06d}"),
                    "prompt": str(rec.get("prompt", "")),
                    "negative_prompt": str(rec.get("negative_prompt", "")),
                    "image": str(image.resolve()),
                }
            )
    return samples


def _build_mask(frame_num: int, lat_h: int, lat_w: int, device: torch.device) -> torch.Tensor:
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    return msk.transpose(1, 2)[0]


def _prep_sample(pipe: wan.WanI2V, sample: Dict[str, str], frame_num: int, target_hw: Tuple[int, int]) -> Dict[str, Any]:
    img = Image.open(sample["image"]).convert("RGB")
    cond_image = TF.to_tensor(img)
    cond_image = TF.resize(cond_image, size=list(target_hw), antialias=True)
    cond_image = TF.center_crop(cond_image, list(target_hw))
    c, h, w = int(cond_image.shape[0]), int(cond_image.shape[1]), int(cond_image.shape[2])
    cond_cfthw = torch.zeros((c, frame_num, h, w), device=pipe.device, dtype=torch.float32)
    cond_cfthw[:, 0] = cond_image.to(device=pipe.device, dtype=torch.float32) * 2.0 - 1.0
    cond_latent = pipe.vae.encode([cond_cfthw])[0]
    latent = torch.randn_like(cond_latent)
    _, f_lat, lat_h, lat_w = cond_latent.shape
    y = torch.cat([_build_mask(frame_num, lat_h, lat_w, pipe.device), cond_latent], dim=0)
    seq_len = f_lat * lat_h * lat_w // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int(math.ceil(seq_len / pipe.sp_size)) * pipe.sp_size

    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        cond_context = pipe.text_encoder([sample["prompt"]], pipe.device)
        n_prompt = sample["negative_prompt"] or pipe.sample_neg_prompt
        uncond_context = pipe.text_encoder([n_prompt], pipe.device)
        pipe.text_encoder.model.cpu()
    else:
        cond_context = pipe.text_encoder([sample["prompt"]], torch.device("cpu"))
        n_prompt = sample["negative_prompt"] or pipe.sample_neg_prompt
        uncond_context = pipe.text_encoder([n_prompt], torch.device("cpu"))
    return {
        "latent": latent.cpu(),
        "y": y.cpu(),
        "seq_len": int(seq_len),
        "context": {"cond": [t.cpu() for t in cond_context], "uncond": [t.cpu() for t in uncond_context]},
    }


def _build_pipe(args: argparse.Namespace) -> wan.WanI2V:
    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    pipe.high_noise_model.to(pipe.device).eval().requires_grad_(False)
    pipe.low_noise_model.to(pipe.device).eval().requires_grad_(False)
    return pipe


class AdalnNormTracer:
    def __init__(self, ctx: TraceContext, writer: JsonlTraceWriter, norm_proxy_bits: int = 8) -> None:
        self.ctx = ctx
        self.writer = writer
        self.norm_proxy_bits = int(norm_proxy_bits)

    def emit_tensor(self, point: str, x: torch.Tensor, *, module_name: str, block_idx: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rec = self.ctx.base_record(point=point, module_name=module_name, block_idx=block_idx, tensor=x, extra=extra)
        rec.update(summarize_tensor(x))
        self.writer.write(rec)
        return rec

    def emit_norm(self, name: str, norm: nn.Module, x: torch.Tensor, y: torch.Tensor, *, block_idx: int, kind: str) -> None:
        self.emit_tensor(f"{name}_in", x, module_name=f"blocks.{block_idx}.{name}", block_idx=block_idx)
        extra = norm_internal_stats(x, kind=kind, eps=float(getattr(norm, "eps", 1e-6)), normalized_shape=int(x.shape[-1]))
        extra.update(affine_param_stats(getattr(norm, "weight", None), getattr(norm, "bias", None)))
        self.emit_tensor(f"{name}_out", y, module_name=f"blocks.{block_idx}.{name}", block_idx=block_idx, extra=extra)
        y_qdq = fake_int_symmetric_qdq(y, bits=self.norm_proxy_bits, channels_dim=-1)
        err = compare_tensors(y, y_qdq, prefix="norm_output_qdq_")
        proxy = self.ctx.base_record(
            point=f"{name}_norm_quant_proxy",
            module_name=f"blocks.{block_idx}.{name}",
            block_idx=block_idx,
            tensor=y,
            extra={"proxy_bits": self.norm_proxy_bits, **err},
        )
        self.writer.write(proxy)

    def patch_model(self, model: nn.Module, block_indices: Sequence[int], scope: PatchScope) -> None:
        for block_idx in block_indices:
            if block_idx < 0 or block_idx >= len(model.blocks):
                continue
            block = model.blocks[block_idx]
            if not isinstance(block, WanAttentionBlock):
                continue
            scope.patch_method(block, "forward", self._make_block_forward(block_idx))

    def _make_block_forward(self, block_idx: int):
        tracer = self

        def traced_forward(self: WanAttentionBlock, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
            assert e.dtype == torch.float32
            tracer.emit_tensor("block_in", x, module_name=f"blocks.{block_idx}", block_idx=block_idx)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            assert e[0].dtype == torch.float32

            e_shift_self = e[0].squeeze(2)
            e_scale_self = e[1].squeeze(2)
            e_gate_self = e[2].squeeze(2)
            e_shift_ffn = e[3].squeeze(2)
            e_scale_ffn = e[4].squeeze(2)
            e_gate_ffn = e[5].squeeze(2)
            for point, tensor in (
                ("e_shift_self", e_shift_self),
                ("e_scale_self", e_scale_self),
                ("e_gate_self", e_gate_self),
                ("e_shift_ffn", e_shift_ffn),
                ("e_scale_ffn", e_scale_ffn),
                ("e_gate_ffn", e_gate_ffn),
                ("one_plus_e_scale_self", 1 + e_scale_self),
                ("one_plus_e_scale_ffn", 1 + e_scale_ffn),
            ):
                tracer.emit_tensor(point, tensor, module_name=f"blocks.{block_idx}.modulation", block_idx=block_idx)

            norm1_out = self.norm1(x)
            tracer.emit_norm("norm1", self.norm1, x, norm1_out, block_idx=block_idx, kind="layernorm")
            self_in = norm1_out.float() * (1 + e_scale_self) + e_shift_self
            tracer.emit_tensor("self_adaln_in", self_in, module_name=f"blocks.{block_idx}.self_adaln", block_idx=block_idx)
            y = self.self_attn(self_in, seq_lens, grid_sizes, freqs)
            tracer.emit_tensor("self_attn_out", y, module_name=f"blocks.{block_idx}.self_attn", block_idx=block_idx)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                self_update = y * e_gate_self
                tracer.emit_tensor("self_residual_update", self_update, module_name=f"blocks.{block_idx}.self_residual", block_idx=block_idx)
                x = x + self_update

            norm3_in = x
            norm3_out = self.norm3(norm3_in)
            if isinstance(self.norm3, nn.Identity):
                tracer.emit_tensor("norm3_out", norm3_out, module_name=f"blocks.{block_idx}.norm3", block_idx=block_idx)
            else:
                tracer.emit_norm("norm3", self.norm3, norm3_in, norm3_out, block_idx=block_idx, kind="layernorm")
            cross_out = self.cross_attn(norm3_out, context, context_lens)
            tracer.emit_tensor("cross_residual_update", cross_out, module_name=f"blocks.{block_idx}.cross_residual", block_idx=block_idx)
            x = x + cross_out

            norm2_out = self.norm2(x)
            tracer.emit_norm("norm2", self.norm2, x, norm2_out, block_idx=block_idx, kind="layernorm")
            ffn_in = norm2_out.float() * (1 + e_scale_ffn) + e_shift_ffn
            tracer.emit_tensor("ffn_adaln_in", ffn_in, module_name=f"blocks.{block_idx}.ffn_adaln", block_idx=block_idx)
            y = self.ffn(ffn_in)
            tracer.emit_tensor("ffn_out", y, module_name=f"blocks.{block_idx}.ffn", block_idx=block_idx)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                ffn_update = y * e_gate_ffn
                tracer.emit_tensor("ffn_residual_update", ffn_update, module_name=f"blocks.{block_idx}.ffn_residual", block_idx=block_idx)
                x = x + ffn_update
            tracer.emit_tensor("block_out", x, module_name=f"blocks.{block_idx}", block_idx=block_idx)
            return x

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

    writer = JsonlTraceWriter(out_dir / "adaln_norm_stats.jsonl")
    ctx = TraceContext()
    tracer = AdalnNormTracer(ctx, writer, norm_proxy_bits=int(args.norm_proxy_bits))
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
        "norm_proxy_bits": int(args.norm_proxy_bits),
        "outputs": {"jsonl": str(out_dir / "adaln_norm_stats.jsonl")},
    }

    try:
        with PatchScope() as patch_scope:
            tracer.patch_model(pipe.high_noise_model, block_indices, patch_scope)
            tracer.patch_model(pipe.low_noise_model, block_indices, patch_scope)
            cfg = WAN_CONFIGS["i2v-A14B"]
            frame_num = cfg.frame_num
            shift = cfg.sample_shift if args.shift is None else float(args.shift)
            boundary = float(pipe.boundary * pipe.num_train_timesteps)
            use_amp = pipe.device.type == "cuda" and pipe.param_dtype in (torch.float16, torch.bfloat16)
            model_by_expert = {"high": pipe.high_noise_model, "low": pipe.low_noise_model}
            model_name_by_expert = {"high": "high_noise_model", "low": "low_noise_model"}

            sample_iter: Iterable[Dict[str, str]] = samples
            if tqdm is not None and not args.disable_tqdm:
                sample_iter = tqdm(samples, desc="adaln/norm samples", unit="sample", dynamic_ncols=True)
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
