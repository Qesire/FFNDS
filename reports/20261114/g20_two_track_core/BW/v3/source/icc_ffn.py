"""FFN-only ICC equations; caller owns expert schedule and native gate."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class Factors:
    left: torch.Tensor   # output x rank
    right: torch.Tensor  # rank x input
    def to(self, *, device, dtype):
        """Explicit execution conversion; reject overflow and invalid factors."""
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('execution factors require float32 or bfloat16')
        left = self.left.to(device=device, dtype=dtype)
        right = self.right.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError('nonfinite factors after conversion')
        return Factors(left, right)
    def apply(self, x):
        return F.linear(F.linear(x, self.right), self.left)

class ICCFFN:
    """One expert/layer. Explicit anchor/hit prevents implicit schedule reuse."""
    MODES = ('author_four_state', 'ffnds_two_residual')
    def __init__(self, first, second, mode):
        if mode not in self.MODES: raise ValueError(mode)
        self.first, self.second, self.mode = first, second, mode
        self.clear()
        self.counts = {'anchor': 0, 'hit': 0, 'anchor_lowrank_projections': 0}
    def clear(self):
        self.cache = None
        self.input_shape = None
    def anchor(self, u, fc1, fc2):
        z = fc1(u); h = F.gelu(z, approximate='tanh'); y = fc2(h)
        self.input_shape = tuple(u.shape)
        if self.mode == 'author_four_state':
            self.cache = tuple(v.detach().clone() for v in (u, z, h, y))
        else:
            self.cache = ((z - self.first.apply(u)).detach(), (y - self.second.apply(h)).detach())
            self.counts['anchor_lowrank_projections'] += 2
        self.counts['anchor'] += 1
        return y
    def hit(self, u):
        if self.cache is None: raise RuntimeError('anchor required')
        if tuple(u.shape) != self.input_shape: raise ValueError('input shape changed')
        if self.mode == 'author_four_state':
            us, zs, hs, ys = self.cache
            z = self.first.apply(u - us) + zs
            h = F.gelu(z, approximate='tanh')
            y = self.second.apply(h - hs) + ys
        else:
            q1, q2 = self.cache
            z = self.first.apply(u) + q1
            h = F.gelu(z, approximate='tanh')
            y = self.second.apply(h) + q2
        self.counts['hit'] += 1
        return y
