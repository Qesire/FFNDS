"""Stage-0 ICC primitives. No model hooks or experiment execution.

Reference: ccccczzy/icc@dafee02efc89cdfbe81395a5a8a8fbeb17cf9be4.
Statistics accept one source per call and use FP32 before subtraction.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Factors:
    left: torch.Tensor
    right: torch.Tensor

    def apply(self, x):
        return F.linear(F.linear(x, self.right), self.left)


class FourState:
    """One request/expert/layer cache; caller supplies native anchor boundaries."""
    def __init__(self, first, second):
        self.first, self.second = first, second
        self.reset()

    def reset(self):
        self.key = self.cache = None

    def anchor(self, key, u, z, h, y):
        if not isinstance(key, tuple) or len(key) != 3 or any(v is None for v in key):
            raise ValueError('key must be (request, expert, layer)')
        if u.shape[:-1] != z.shape[:-1] or z.shape != h.shape or u.shape[:-1] != y.shape[:-1]:
            raise ValueError('incompatible anchor boundaries')
        self.key = key
        self.cache = tuple(v.detach().clone() for v in (u, z, h, y))
        return y

    def hit(self, key, u):
        if self.cache is None or key != self.key:
            raise RuntimeError('missing or mismatched anchor')
        ua, za, ha, ya = self.cache
        if u.shape != ua.shape or u.dtype != ua.dtype or u.device != ua.device:
            raise ValueError('input shape, dtype or device changed')
        z = self.first.apply(u - ua) + za
        h = F.gelu(z, approximate='tanh')
        return self.second.apply(h - ha) + ya


class SourceStatistics:
    """Accumulate independently normalized CA or CD channel magnitudes per source."""
    def __init__(self, family, eps=1e-6):
        if family not in ('CA', 'CD'):
            raise ValueError('family must be CA or CD')
        self.family, self.eps = family, eps
        self.sources = set()
        self.total = None

    def add(self, source, current, anchor=None):
        if source in self.sources:
            raise ValueError('duplicate source')
        if current.ndim != 3 or current.shape[0] != 1 or current.shape[1] == 0:
            raise ValueError('expected nonempty [1, tokens, channels]')
        x = current.float()
        if self.family == 'CD':
            if anchor is None or anchor.shape != current.shape:
                raise ValueError('CD requires matching anchor')
            x = x - anchor.float()
        elif anchor is not None:
            raise ValueError('CA does not take anchor')
        mag = x.abs().mean(dim=1).squeeze(0)
        if not bool(torch.isfinite(mag).all()):
            raise ValueError('nonfinite statistic')
        lo, hi = mag.min(), mag.max()
        normalized = (mag - lo + self.eps) / (hi - lo + self.eps)
        if self.total is not None and normalized.shape != self.total.shape:
            raise ValueError('channel shape changed')
        self.total = normalized.clone() if self.total is None else self.total + normalized
        self.sources.add(source)
        return mag, normalized

    def scales(self, max_ratio=1000):
        if not self.sources or max_ratio <= 0:
            raise ValueError('statistics empty or invalid max_ratio')
        mean = self.total / len(self.sources)
        root = mean.sqrt()
        return mean, root, root.clamp_min(root.max() / max_ratio)
