"""MXFP4 diagnostic summaries for activation tracing."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .mxfp4 import e8m0_decode, fp4_e2m1_encode


@torch.no_grad()
def activation_mxfp4_group_stats(
    x: torch.Tensor,
    *,
    group_size: int = 32,
    clip_ratio: float = 1.0,
    channels_dim: int = -1,
    exp_offset: int = 0,
    scale_method: str = "safe_ceil",
    fp4_max: float = 6.0,
) -> Dict[str, Any]:
    """Return compact MXFP4 group diagnostics for activation tensor `x`.

    This mirrors `quant_linear._qdq_mxfp4_activation_groupwise` enough to
    report scale/code/clipping statistics, but does not retain full tensors.
    """
    rec: Dict[str, Any] = {
        "mxfp4_group_size": int(group_size),
        "mxfp4_clip_ratio": float(clip_ratio),
        "mxfp4_exp_offset": int(exp_offset),
        "mxfp4_scale_method": str(scale_method),
    }
    if x.ndim == 0 or x.numel() == 0:
        return rec
    if int(group_size) != 32:
        rec["mxfp4_error"] = f"expected group_size=32, got {group_size}"
        return rec

    cdim = channels_dim if channels_dim >= 0 else x.ndim + channels_dim
    if cdim < 0 or cdim >= x.ndim:
        rec["mxfp4_error"] = f"invalid channels_dim={channels_dim} for shape={tuple(x.shape)}"
        return rec

    x32 = x.detach().to(torch.float32)
    if cdim != x32.ndim - 1:
        x32 = x32.movedim(cdim, -1)
    orig_c = int(x32.shape[-1])
    x2 = x32.reshape(-1, orig_c)
    pad = (int(group_size) - (orig_c % int(group_size))) % int(group_size)
    if pad > 0:
        x2 = F.pad(x2, (0, pad))
    c_pad = int(x2.shape[-1])
    num_groups = c_pad // int(group_size)
    xg = x2.view(-1, num_groups, int(group_size))

    amax = xg.abs().amax(dim=-1) * float(clip_ratio)
    zero_mask = amax == 0
    safe_amax = torch.where(zero_mask, torch.ones_like(amax), amax)
    if scale_method == "ocp_floor":
        block_scale_exp = torch.floor(torch.log2(safe_amax)) - 2.0
    elif scale_method == "safe_ceil":
        block_scale_exp = torch.ceil(torch.log2(safe_amax / float(fp4_max)))
    else:
        rec["mxfp4_error"] = f"unsupported scale_method={scale_method}"
        return rec
    block_scale_exp = block_scale_exp + int(exp_offset)
    block_scale_exp = torch.where(zero_mask, torch.zeros_like(block_scale_exp), block_scale_exp)
    scale_code = (block_scale_exp.to(torch.int32) + 127).clamp_(0, 254).to(torch.uint8)
    scale_code = torch.where(zero_mask, torch.full_like(scale_code, 127), scale_code)
    block_scale = e8m0_decode(scale_code, out_dtype=torch.float32)

    x_scaled = xg / block_scale.unsqueeze(-1)
    clipped = x_scaled.abs() > float(fp4_max)
    x_clamped = x_scaled.clamp(-float(fp4_max), float(fp4_max))
    codes = fp4_e2m1_encode(x_clamped)

    scale_hist = torch.bincount(scale_code.reshape(-1).to(torch.int64).cpu(), minlength=255)
    code_hist = torch.bincount(codes.reshape(-1).to(torch.int64).cpu(), minlength=16)
    group_amax = xg.abs().amax(dim=-1).reshape(-1)
    rec.update(
        {
            "mxfp4_num_groups": int(amax.numel()),
            "mxfp4_zero_group_ratio": float(zero_mask.to(torch.float32).mean().item()),
            "mxfp4_clipped_fraction": float(clipped.to(torch.float32).mean().item()),
            "mxfp4_group_absmax_mean": float(group_amax.mean().item()) if group_amax.numel() else 0.0,
            "mxfp4_group_absmax_max": float(group_amax.max().item()) if group_amax.numel() else 0.0,
            "mxfp4_scale_code_hist": [int(v) for v in scale_hist.tolist()],
            "mxfp4_fp4_code_hist": [int(v) for v in code_hist.tolist()],
        }
    )
    return rec
