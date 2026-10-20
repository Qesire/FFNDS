"""Streaming-friendly tensor reductions for quantization tracing.

The functions in this module return compact summaries and never retain input
activations. They are intended to be called inside forward hooks.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


_EPS = 1e-30


@torch.no_grad()
def summarize_tensor(
    x: torch.Tensor,
    *,
    channels_dim: int = -1,
    topk_channels: int = 8,
    include_channel_stats: bool = True,
    include_token_norms: bool = True,
) -> Dict[str, Any]:
    """Return scalar distribution statistics for a tensor."""
    if not torch.is_tensor(x):
        raise TypeError("summarize_tensor expects a torch.Tensor")
    rec: Dict[str, Any] = {
        "numel": int(x.numel()),
        "shape": [int(v) for v in x.shape],
        "dtype": str(x.dtype).replace("torch.", ""),
    }
    if x.numel() == 0:
        return rec

    x32 = x.detach().to(torch.float32)
    finite = torch.isfinite(x32)
    finite_count = int(finite.sum().item())
    rec["finite_ratio"] = float(finite_count / max(int(x32.numel()), 1))
    rec["nan_count"] = int(torch.isnan(x32).sum().item())
    rec["inf_count"] = int(torch.isinf(x32).sum().item())
    if finite_count == 0:
        return rec

    xf = x32[finite]
    xa = xf.abs()
    rec.update(
        {
            "mean": float(xf.mean().item()),
            "std": float(xf.std(unbiased=False).item()) if xf.numel() > 1 else 0.0,
            "rms": float(torch.sqrt(torch.mean(xf * xf).clamp_min(0.0)).item()),
            "abs_mean": float(xa.mean().item()),
            "min": float(xf.min().item()),
            "max": float(xf.max().item()),
            "absmax": float(xa.max().item()),
            "zero_ratio": float((xf == 0).to(torch.float32).mean().item()),
            "near_zero_1e_6_ratio": float((xa <= 1e-6).to(torch.float32).mean().item()),
        }
    )
    qs = _quantiles(xa, (0.50, 0.90, 0.95, 0.99, 0.999))
    rec.update(
        {
            "abs_p50": qs[0],
            "abs_p90": qs[1],
            "abs_p95": qs[2],
            "abs_p99": qs[3],
            "abs_p999": qs[4],
        }
    )
    median = max(qs[0], _EPS)
    p99 = max(qs[3], _EPS)
    rec["outlier_ratio_8x_median"] = float((xa > 8.0 * median).to(torch.float32).mean().item())
    rec["outlier_ratio_16x_median"] = float((xa > 16.0 * median).to(torch.float32).mean().item())
    rec["outlier_ratio_32x_median"] = float((xa > 32.0 * median).to(torch.float32).mean().item())
    rec["absmax_over_p99"] = float(rec["absmax"] / p99)
    rec["p99_over_p50"] = float(qs[3] / median)

    if include_channel_stats and x.ndim >= 1:
        rec.update(_channel_stats(x32, channels_dim=channels_dim, topk=topk_channels))
    if include_token_norms and x.ndim == 3:
        rec.update(_token_norm_stats(x32))
    return rec


@torch.no_grad()
def compare_tensors(a: torch.Tensor, b: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    """Return compact error metrics comparing two tensors of the same shape."""
    a32 = a.detach().to(torch.float32).reshape(-1)
    b32 = b.detach().to(torch.float32).reshape(-1)
    if a32.numel() != b32.numel():
        raise ValueError(f"Tensor size mismatch: {a32.numel()} vs {b32.numel()}")
    diff = b32 - a32
    mse = torch.mean(diff * diff).item() if diff.numel() else 0.0
    rmse = mse ** 0.5
    mae = torch.mean(diff.abs()).item() if diff.numel() else 0.0
    denom = torch.linalg.vector_norm(a32).item()
    rel_l2 = torch.linalg.vector_norm(diff).item() / max(denom, _EPS)
    cosine = _cosine(a32, b32)
    out = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "relative_l2": float(rel_l2),
        "max_abs_error": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "cosine": float(cosine),
        "cosine_error": float(1.0 - cosine),
    }
    if not prefix:
        return out
    return {f"{prefix}{k}": v for k, v in out.items()}


@torch.no_grad()
def norm_internal_stats(x: torch.Tensor, *, kind: str, eps: float, normalized_shape: Optional[int] = None) -> Dict[str, Any]:
    """Summarize internal reciprocal scale for LayerNorm or RMSNorm."""
    x32 = x.detach().to(torch.float32)
    if x32.numel() == 0:
        return {"norm_kind": kind}
    if kind == "rmsnorm":
        scale = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + float(eps))
    elif kind == "layernorm":
        mean = x32.mean(dim=-1, keepdim=True)
        var = (x32 - mean).pow(2).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(var + float(eps))
    else:
        raise ValueError(f"Unsupported norm kind: {kind}")
    rec = summarize_tensor(scale, include_channel_stats=False, include_token_norms=False)
    rec["norm_kind"] = kind
    if normalized_shape is not None:
        rec["normalized_shape"] = int(normalized_shape)
    return {f"norm_scale_{k}": v for k, v in rec.items()}


@torch.no_grad()
def affine_param_stats(weight: Optional[torch.Tensor], bias: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {}
    if weight is not None:
        w = summarize_tensor(weight, include_channel_stats=False, include_token_norms=False)
        rec.update({f"weight_{k}": v for k, v in w.items()})
    if bias is not None:
        b = summarize_tensor(bias, include_channel_stats=False, include_token_norms=False)
        rec.update({f"bias_{k}": v for k, v in b.items()})
    return rec


@torch.no_grad()
def fake_int_symmetric_qdq(x: torch.Tensor, *, bits: int = 8, channels_dim: Optional[int] = None) -> torch.Tensor:
    """Simple symmetric fake quantization for norm quantization proxies."""
    qmax = float(2 ** (int(bits) - 1) - 1)
    if qmax <= 0:
        raise ValueError(f"bits must be >= 2, got {bits}")
    x32 = x.detach().to(torch.float32)
    if channels_dim is None:
        scale = x32.abs().amax().clamp_min(_EPS) / qmax
    else:
        cdim = channels_dim if channels_dim >= 0 else x32.ndim + channels_dim
        reduce_dims = tuple(i for i in range(x32.ndim) if i != cdim)
        scale = x32.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(_EPS) / qmax
    q = torch.round(x32 / scale).clamp(-qmax - 1, qmax)
    return (q * scale).to(dtype=x.dtype)


@torch.no_grad()
def log2_abs_histogram(x: torch.Tensor, bins: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Return a compact log2(abs(x)) histogram."""
    edges = list(range(-30, 17)) if bins is None else [int(v) for v in bins]
    xa = x.detach().to(torch.float32).abs().reshape(-1)
    if xa.numel() == 0:
        return {"log2_abs_hist_edges": edges, "log2_abs_hist_counts": []}
    zero_count = int((xa == 0).sum().item())
    vals = torch.log2(xa.clamp_min(2.0 ** min(edges)))
    edge_t = torch.tensor(edges, device=vals.device, dtype=torch.float32)
    bucket = torch.bucketize(vals, edge_t)
    counts = torch.bincount(bucket.cpu(), minlength=len(edges) + 1).to(torch.int64)
    return {
        "log2_abs_hist_edges": edges,
        "log2_abs_hist_counts": [int(v) for v in counts.tolist()],
        "zero_count": zero_count,
    }


def _quantiles(x, qs, max_elems: int = 2000000):
    """Safe quantile for very large tensors.

    torch.quantile can fail on huge flattened tensors. For diagnostics we only
    need stable approximate percentiles, so uniformly subsample after flattening.
    """
    import torch
    if x is None:
        return [float("nan") for _ in qs]
    with torch.no_grad():
        xa = x.detach()
        if xa.numel() == 0:
            return [float("nan") for _ in qs]
        xa = xa.to(torch.float32).abs().reshape(-1)
        n = int(xa.numel())
        if n > int(max_elems):
            idx = torch.linspace(0, n - 1, steps=int(max_elems), device=xa.device).long()
            xa = xa.index_select(0, idx)
        q = torch.tensor(list(qs), dtype=torch.float32, device=xa.device)
        vals = torch.quantile(xa, q)
        return [float(v) for v in vals.detach().cpu().tolist()]


def _channel_stats(x32: torch.Tensor, *, channels_dim: int, topk: int) -> Dict[str, Any]:
    cdim = channels_dim if channels_dim >= 0 else x32.ndim + channels_dim
    if cdim < 0 or cdim >= x32.ndim:
        return {}
    xm = x32.movedim(cdim, -1) if cdim != x32.ndim - 1 else x32
    c = int(xm.shape[-1])
    if c <= 0:
        return {}
    x2 = xm.reshape(-1, c)
    ch_absmax = x2.abs().amax(dim=0)
    ch_rms = torch.sqrt(torch.mean(x2 * x2, dim=0).clamp_min(0.0))
    k = min(max(int(topk), 0), c)
    top = []
    if k > 0:
        vals, idx = torch.topk(ch_absmax, k=k)
        top = [{"index": int(i), "value": float(v)} for i, v in zip(idx.cpu().tolist(), vals.cpu().tolist())]
    ch_q = _quantiles(ch_absmax, (0.99,))
    rms_q = _quantiles(ch_rms, (0.99,))
    return {
        "channel_count": c,
        "channel_absmax_mean": float(ch_absmax.mean().item()),
        "channel_absmax_std": float(ch_absmax.std(unbiased=False).item()) if c > 1 else 0.0,
        "channel_absmax_p99": ch_q[0],
        "channel_absmax_max": float(ch_absmax.max().item()),
        "channel_rms_mean": float(ch_rms.mean().item()),
        "channel_rms_p99": rms_q[0],
        "channel_rms_max": float(ch_rms.max().item()),
        "top_channels_by_absmax": top,
    }


def _token_norm_stats(x32: torch.Tensor) -> Dict[str, Any]:
    norms = torch.linalg.vector_norm(x32, dim=-1).reshape(-1)
    q = _quantiles(norms, (0.50, 0.99))
    return {
        "token_l2_p50": q[0],
        "token_l2_p99": q[1],
        "token_l2_max": float(norms.max().item()) if norms.numel() else 0.0,
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item()
    if denom <= _EPS:
        return 1.0 if torch.linalg.vector_norm(a - b).item() <= _EPS else 0.0
    return float(torch.dot(a, b).item() / denom)
