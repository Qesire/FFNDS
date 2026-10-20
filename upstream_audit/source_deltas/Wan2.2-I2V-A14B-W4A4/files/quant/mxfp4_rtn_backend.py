import atexit
import inspect
import json
import os
from pathlib import Path

import torch

# Reuse Wan's original MXFP4 implementation instead of hand-writing codebook/scale.
from .mxfp4 import (
    quantize_linear_weight_mxfp4,
    dequantize_linear_weight_mxfp4,
)
from .quant_linear import _qdq_mxfp4_activation_groupwise

_STATS = {
    "weight_qdq_calls": 0,
    "weight_qdq_tensors": 0,
    "activation_qdq_calls": 0,
    "activation_qdq_tensors": 0,
    "block_size": 32,
    "weight_scale_method": "ocp_floor",
    "activation_scale_method": "safe_ceil",
    "format": "Wan_original_MXFP4_E2M1_E8M0_rtn_qdq",
    "compute_backend": "bf16_or_fp16_gemm_after_qdq",
    "uses_wan_mxfp4_py": True,
    "uses_wan_quant_linear_activation_qdq": True,
}

def reset_mxfp4_rtn_stats():
    for k in list(_STATS):
        if k.endswith("_calls") or k.endswith("_tensors"):
            _STATS[k] = 0

def get_mxfp4_rtn_stats():
    return dict(_STATS)

def _call_with_supported_kwargs(fn, *args, **kwargs):
    """
    Be robust to minor signature drift in local quant/mxfp4.py.
    Only pass kwargs supported by the function.
    """
    sig = inspect.signature(fn)
    keep = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            keep[k] = v
    return fn(*args, **keep)

@torch.no_grad()
def mxfp4_rtn_qdq_weight(
    w: torch.Tensor,
    dim: int = -1,
    block_size: int = 32,
    force_fp32: bool = True,
    scale_method: str | None = None,
):
    """
    Weight QDQ using Wan's native MXFP4 reference path:
      quantize_linear_weight_mxfp4 -> dequantize_linear_weight_mxfp4

    Expected weight layout is [out_features, in_features].
    MXFP4 group axis should be the last/in_features dimension, same as Linear.
    """
    if dim not in (-1, w.ndim - 1):
        raise ValueError(f"MXFP4 weight QDQ only supports dim=-1 / last dim, got dim={dim}, shape={tuple(w.shape)}")

    orig_dtype = w.dtype
    device = w.device
    work = w.detach()
    if force_fp32:
        work = work.float()

    scale_method = scale_method or os.environ.get("MXFP4_RTN_WEIGHT_SCALE_METHOD", "ocp_floor")
    _STATS["weight_scale_method"] = scale_method

    _STATS["weight_qdq_calls"] += 1
    _STATS["weight_qdq_tensors"] += int(w.numel())

    # Wan native quantize_linear_weight_mxfp4 returns:
    # {
    #   "meta": {..., "in_features": int, ...},
    #   "qweight_packed": uint8 tensor,
    #   "scales": uint8 E8M0 tensor,
    # }
    qstate = _call_with_supported_kwargs(
        quantize_linear_weight_mxfp4,
        work,
        group_size=block_size,
        scale_method=scale_method,
    )

    if not isinstance(qstate, dict):
        raise TypeError(f"quantize_linear_weight_mxfp4 should return dict, got {type(qstate)}")

    qweight_packed = qstate.get("qweight_packed", None)
    scale_e8m0 = qstate.get("scales", qstate.get("scale_e8m0", None))
    meta = qstate.get("meta", {})

    if not torch.is_tensor(qweight_packed):
        raise TypeError(f"bad qweight_packed in qstate: {type(qweight_packed)}")
    if not torch.is_tensor(scale_e8m0):
        raise TypeError(f"bad scales/scale_e8m0 in qstate: {type(scale_e8m0)}")

    in_features = int(meta.get("in_features", work.shape[-1]))

    dq = _call_with_supported_kwargs(
        dequantize_linear_weight_mxfp4,
        qweight_packed,
        scale_e8m0,
        in_features,
        group_size=block_size,
        out_dtype=torch.float32,
        device=device,
    )

    if not torch.is_tensor(dq):
        raise TypeError(f"Unexpected dequantize_linear_weight_mxfp4 return type: {type(dq)}")

    if tuple(dq.shape) != tuple(w.shape):
        raise RuntimeError(f"dequant shape mismatch: got {tuple(dq.shape)}, expected {tuple(w.shape)}")

    return dq.to(device=device, dtype=orig_dtype)

def mxfp4_rtn_qdq_activation(
    x: torch.Tensor,
    dim: int = -1,
    block_size: int = 32,
    force_fp32: bool = True,
    scale_method: str | None = None,
):
    """
    Activation QDQ using Wan's native activation QDQ helper:
      quant_linear._qdq_mxfp4_activation_groupwise
    """
    if dim not in (-1, x.ndim - 1):
        raise ValueError(f"MXFP4 activation QDQ only supports last dim, got dim={dim}, shape={tuple(x.shape)}")

    orig_dtype = x.dtype
    work = x.float() if force_fp32 else x

    scale_method = scale_method or os.environ.get("MXFP4_RTN_ACT_SCALE_METHOD", "safe_ceil")
    _STATS["activation_scale_method"] = scale_method

    _STATS["activation_qdq_calls"] += 1
    _STATS["activation_qdq_tensors"] += int(x.numel())

    y = _call_with_supported_kwargs(
        _qdq_mxfp4_activation_groupwise,
        work,
        group_size=block_size,
        scale_method=scale_method,
    )

    return y.to(dtype=orig_dtype)

def _write_runtime_report():
    p = os.environ.get("MXFP4_RTN_RUNTIME_REPORT_JSON", "")
    if not p:
        return
    try:
        path = Path(p)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(get_mxfp4_rtn_stats(), indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] failed to write MXFP4 runtime report: {e}", flush=True)

atexit.register(_write_runtime_report)
