"""Offline factor/statistic operations from pinned ICC equations."""
import torch
from icc_ffn import Factors

def delta_stat(anchor, current, eps=1e-6):
    if anchor.shape != current.shape or anchor.ndim != 3: raise ValueError('expected matching B,N,C')
    mag = (current - anchor).abs().mean(dim=1)
    lo = mag.amin(dim=-1, keepdim=True); hi = mag.amax(dim=-1, keepdim=True)
    return ((mag - lo + eps) / (hi - lo + eps)).mean(dim=0)

class DeltaAccumulator:
    def __init__(self): self.sum = None; self.count = 0
    def add(self, anchor, current):
        v = delta_stat(anchor, current)
        self.sum = v.clone() if self.sum is None else self.sum + v
        self.count += 1
    def weights(self, max_ratio=None):
        if not self.count: raise ValueError('empty statistics')
        x = (self.sum / self.count).sqrt()
        if max_ratio is not None:
            if max_ratio <= 0: raise ValueError('max_ratio must be positive')
            x = x.clamp_min(x.max().item() / max_ratio)
        if not bool(torch.isfinite(x).all()): raise ValueError('nonfinite statistic scales')
        return x

def factorize(weight, rank, left_scale=None, right_scale=None, eps=1e-6, *, driver=None):
    if weight.ndim != 2 or not 1 <= rank <= min(weight.shape): raise ValueError('invalid matrix or rank')
    w = weight.to(torch.float32)
    if not bool(torch.isfinite(w).all()): raise ValueError('nonfinite weight')
    l = None if left_scale is None else left_scale.to(device=w.device, dtype=w.dtype) + eps
    r = None if right_scale is None else right_scale.to(device=w.device, dtype=w.dtype) + eps
    if l is not None:
        if l.shape != (w.shape[0],) or not bool(torch.all(l > 0)): raise ValueError('invalid left scales')
        w = l[:, None] * w
    if r is not None:
        if r.shape != (w.shape[1],) or not bool(torch.all(r > 0)): raise ValueError('invalid right scales')
        w = w * r[None, :]
    if driver not in (None, "gesvd", "gesvdj"):
        raise ValueError("driver must be None, gesvd or gesvdj; approximate gesvda is excluded")
    if driver is not None and w.device.type != "cuda":
        raise ValueError("explicit SVD driver requires CUDA")
    u, s, vh = torch.linalg.svd(w, full_matrices=False, driver=driver)
    a = u[:, :rank] * s[:rank]; b = vh[:rank, :]
    if l is not None: a = a / l[:, None]
    if r is not None: b = b / r[None, :]
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()): raise ValueError('nonfinite factors')
    return Factors(a.contiguous(), b.contiguous())
