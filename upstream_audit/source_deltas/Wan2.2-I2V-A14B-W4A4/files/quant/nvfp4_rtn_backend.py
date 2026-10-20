from __future__ import annotations

import atexit
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

FP4_E2M1_MAX = 6.0
FP8_E4M3FN_MAX = 448.0
DEFAULT_GROUP_SIZE = 16


@dataclass
class NVFP4Stats:
    weight_qdq_calls: int = 0
    weight_qdq_tensors: int = 0
    weight_qdq_groups: int = 0
    activation_qdq_calls: int = 0
    activation_qdq_tensors: int = 0
    activation_qdq_groups: int = 0
    skipped_tensors: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.update({
            "format": "NVFP4_E2M1_with_FP8_E4M3FN_local_scale",
            "weight_method": "one_shot_RTN_QDQ",
            "activation_method": "static_per_linear_global_scale_plus_dynamic_group16_QDQ",
            "group_size": DEFAULT_GROUP_SIZE,
            "compute_backend": "bf16_or_fp16_F.linear_after_QDQ",
            "packed_kernel": False,
        })
        return out


GLOBAL_NVFP4_STATS = NVFP4Stats()


def reset_nvfp4_rtn_stats() -> None:
    global GLOBAL_NVFP4_STATS
    GLOBAL_NVFP4_STATS = NVFP4Stats()


def get_nvfp4_rtn_stats() -> dict[str, Any]:
    return GLOBAL_NVFP4_STATS.to_dict()


def cast_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the HiF4_Sp NVFP4 E2M1 finite grid."""
    sign = torch.sign(x)
    a = torch.abs(x).to(torch.float32)
    out = torch.where(a <= 0.25, 0.0, 6.0)
    out = torch.where((a > 0.25) & (a < 0.75), 0.5, out)
    out = torch.where((a >= 0.75) & (a <= 1.25), 1.0, out)
    out = torch.where((a > 1.25) & (a < 1.75), 1.5, out)
    out = torch.where((a >= 1.75) & (a <= 2.5), 2.0, out)
    out = torch.where((a > 2.5) & (a < 3.5), 3.0, out)
    out = torch.where((a >= 3.5) & (a <= 5.0), 4.0, out)
    return (out * sign.to(torch.float32)).to(x.dtype)


def cast_to_fp8_e4m3fn(x: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Current torch has no torch.float8_e4m3fn")
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def global_scale_from_amax(amax: float | torch.Tensor, margin: float = 1.0) -> float:
    """G=448*6/(amax*margin), matching compressed-tensors NVFP4."""
    a = float(amax.detach().float().cpu().item()) if torch.is_tensor(amax) else float(amax)
    a *= float(margin)
    if not math.isfinite(a) or a <= 0.0:
        return 1.0
    g = FP8_E4M3FN_MAX * FP4_E2M1_MAX / a
    return float(g) if math.isfinite(g) and g > 0.0 else 1.0


def _validate_group_size(group_size: int) -> None:
    if int(group_size) != DEFAULT_GROUP_SIZE:
        raise ValueError(f"Native NVFP4 requires group_size=16, got {group_size}")


def _pad_last_dim(x: torch.Tensor, group_size: int):
    original = int(x.shape[-1])
    padded = ((original + group_size - 1) // group_size) * group_size
    return (x if padded == original else F.pad(x, (0, padded-original))), original


def _qdq_grouped(x: torch.Tensor, *, global_scale: torch.Tensor, group_size: int) -> torch.Tensor:
    _validate_group_size(group_size)
    original_shape = tuple(x.shape)
    work, original_last = _pad_last_dim(x.to(torch.float32), group_size)
    padded_last = int(work.shape[-1])
    grouped = work.reshape(*work.shape[:-1], padded_last // group_size, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    stored_local = torch.clamp(global_scale * (amax / FP4_E2M1_MAX), 0.0, FP8_E4M3FN_MAX)
    stored_local = cast_to_fp8_e4m3fn(stored_local)
    zero = amax == 0
    effective = stored_local / global_scale
    safe = torch.where(zero, torch.ones_like(effective), effective)
    scaled = torch.clamp(grouped / safe, -FP4_E2M1_MAX, FP4_E2M1_MAX)
    q = cast_to_fp4_e2m1(scaled)
    dq = torch.where(zero, torch.zeros_like(q), q * effective)
    dq = dq.reshape(*work.shape)[..., :original_last]
    return dq.reshape(original_shape)


@torch.no_grad()
def nvfp4_rtn_qdq_weight(weight: torch.Tensor, *, group_size: int = 16,
                           force_fp32: bool = True, quant_device: str | torch.device = "cuda:0",
                           chunk_rows: int = 128, return_device=None,
                           collect_error_stats: bool = False):
    if weight.ndim != 2:
        raise ValueError(f"NVFP4 weight must be 2D, got {tuple(weight.shape)}")
    _validate_group_size(group_size)
    if chunk_rows < 1: raise ValueError("chunk_rows must be positive")
    orig_device, orig_dtype = weight.device, weight.dtype
    return_device = torch.device(return_device) if return_device is not None else orig_device
    qdev = torch.device(quant_device)
    if qdev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amax = weight.detach().abs().amax()
    g_value = global_scale_from_amax(amax)
    g = torch.tensor(g_value, dtype=torch.float32, device=qdev)
    out = torch.empty(weight.shape, dtype=orig_dtype, device=return_device)
    err_sq = ref_sq = 0.0
    out_features, in_features = map(int, weight.shape)
    groups_per_row = (in_features + group_size - 1) // group_size
    for start in range(0, out_features, int(chunk_rows)):
        end = min(start + int(chunk_rows), out_features)
        chunk = weight.detach()[start:end].to(qdev)
        if force_fp32: chunk = chunk.float()
        dq = _qdq_grouped(chunk, global_scale=g, group_size=group_size)
        if collect_error_stats:
            err_sq += float(((dq.float()-chunk.float())**2).sum().item())
            ref_sq += float((chunk.float()**2).sum().item())
        out[start:end].copy_(dq.to(device=return_device, dtype=orig_dtype))
        del chunk, dq
    GLOBAL_NVFP4_STATS.weight_qdq_calls += 1
    GLOBAL_NVFP4_STATS.weight_qdq_tensors += int(weight.numel())
    GLOBAL_NVFP4_STATS.weight_qdq_groups += out_features * groups_per_row
    meta = {
        "shape":[out_features,in_features], "group_size":int(group_size),
        "groups":int(out_features*groups_per_row),
        "weight_amax":float(amax.detach().float().cpu().item()),
        "weight_global_scale":float(g_value), "input_dtype":str(orig_dtype),
        "input_device":str(orig_device), "quant_device":str(qdev),
        "return_device":str(return_device), "force_fp32":bool(force_fp32),
        "chunk_rows":int(chunk_rows),
    }
    if collect_error_stats: meta["relative_l2"] = math.sqrt(err_sq/max(ref_sq,1e-30))
    if qdev.type == "cuda": torch.cuda.empty_cache()
    return out, meta


@torch.no_grad()
def nvfp4_rtn_qdq_activation(x: torch.Tensor, input_global_scale, *, group_size: int=16,
                               force_fp32: bool=True) -> torch.Tensor:
    _validate_group_size(group_size)
    if torch.is_tensor(input_global_scale):
        if input_global_scale.numel()!=1: raise ValueError("input_global_scale must be scalar")
        g=input_global_scale.reshape(()).to(device=x.device,dtype=torch.float32)
    else:
        g=torch.tensor(float(input_global_scale),device=x.device,dtype=torch.float32)
    orig_dtype=x.dtype
    y=_qdq_grouped(x.float() if force_fp32 else x,global_scale=g,group_size=group_size)
    GLOBAL_NVFP4_STATS.activation_qdq_calls += 1
    GLOBAL_NVFP4_STATS.activation_qdq_tensors += int(x.numel())
    rows=int(x.numel()//max(int(x.shape[-1]),1))
    GLOBAL_NVFP4_STATS.activation_qdq_groups += rows*((int(x.shape[-1])+group_size-1)//group_size)
    return y.to(orig_dtype)


def nvfp4_backend_metadata():
    return {"quant_scheme":"nvfp4","element_format":"fp4_e2m1","element_max":6.0,
            "local_scale_format":"fp8_e4m3fn","local_scale_max":448.0,"group_size":16,
            "global_scale_formula":"448*6/per_tensor_amax","weight_method":"one_shot_RTN_QDQ",
            "activation_method":"calibrated_per_linear_global_scale_plus_dynamic_group16_local_scale",
            "weight_storage":"float_QDQ_weight","activation_storage":"float_runtime_QDQ",
            "compute_backend":"bf16_or_fp16_F.linear","packed_kernel":False,
            "accuracy_baseline_only":True}


def _write_runtime_report():
    p=os.environ.get("NVFP4_RTN_RUNTIME_REPORT_JSON","")
    if not p: return
    try:
        path=Path(p)
        if path.exists(): return
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(get_nvfp4_rtn_stats(),indent=2,ensure_ascii=False),encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] failed to write NVFP4 runtime report: {exc}",flush=True)

atexit.register(_write_runtime_report)
