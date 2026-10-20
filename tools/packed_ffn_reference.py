"""Whole-group packed HiF4 FFN arithmetic; online selector is supplied separately.

prepare() is offline packing for a fixed support. Callers must account for
weight movement/routing in their actual deployment. This is not a benchmark.
"""
import torch
import torch.nn.functional as F


def prepare(w1, w2, s1, s2, b1, b2, groups):
    ids = torch.as_tensor(groups, dtype=torch.long, device=w1.device)
    if (ids.ndim != 1 or len(ids) != 108 or len(torch.unique(ids)) != 108
            or bool((ids < 0).any()) or bool((ids >= 216).any())):
        raise ValueError("Exactly 108 distinct original groups required")
    # Keep source group order deterministic; no arbitrary sub-group permutation.
    ids = ids.sort().values
    channels = (ids[:, None] * 64 + torch.arange(64, device=w1.device)).reshape(-1)
    if w1.shape != (13824, 5120) or w2.shape != (5120, 13824):
        raise ValueError("Expected actual Wan FFN checkpoint layouts")
    return {"w1": w1.index_select(0, channels).contiguous(),
            "w2": w2.index_select(1, channels).contiguous(),
            "s1": s1, "s2": s2.index_select(0, channels).contiguous(),
            "b1": None if b1 is None else b1.index_select(0, channels).contiguous(),
            "b2": b2, "channels": channels, "group_ids": ids}


def forward(input_tensor, packed, *, gain=1.0):
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
    dtype = input_tensor.dtype
    v1 = qdq((input_tensor.float() / packed["s1"]).to(dtype), force_fp32=True)
    z = F.linear(v1, packed["w1"], packed["b1"])
    h = F.gelu(z, approximate="tanh")
    v2 = qdq((h.float() / packed["s2"]).to(dtype), force_fp32=True)
    # Match the frozen gain ordering: apply after Q2, then BF16 before GEMM.
    alpha = torch.as_tensor(gain, dtype=torch.float32, device=v2.device)
    if alpha.ndim == 1:
        if len(alpha) != len(v2):
            raise ValueError("Per-row gain must match actual rows")
        alpha = alpha[:, None]
    v2 = (v2.float() * alpha).to(dtype)
    return F.linear(v2, packed["w2"], packed["b2"])
