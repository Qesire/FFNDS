"""Straight-through training; forward uses the unmodified native HiF4 quantizers."""
import math

import torch
import torch.nn.functional as F


def straight_through(value, quantizer):
    # Custom backward avoids subtract/add roundoff in the forward value.
    class NativeQuantize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x): return quantizer(x, force_fp32=True)
        @staticmethod
        def backward(ctx, grad): return grad
    return NativeQuantize.apply(value)


class FoldedAdapters(torch.nn.Module):
    def __init__(self, w0, w2, rank=32, seed=20261021):
        super().__init__(); self.register_buffer('w0', w0); self.register_buffer('w2', w2)
        generator = torch.Generator(device=w0.device).manual_seed(seed)
        for slot, weight in [('0', w0), ('2', w2)]:
            a = torch.randn((rank, weight.shape[1]), device=weight.device, dtype=torch.float32,
                generator=generator) / math.sqrt(weight.shape[1])
            b = torch.zeros((weight.shape[0], rank), device=weight.device, dtype=torch.float32)
            self.register_parameter('a' + slot, torch.nn.Parameter(a))
            self.register_parameter('b' + slot, torch.nn.Parameter(b))

    def weights(self, quantizer, *, original=False):
        if original: return self.w0, self.w2
        result = []
        for slot, prior in [('0', self.w0), ('2', self.w2)]:
            value = prior.float() + getattr(self, 'b' + slot) @ getattr(self, 'a' + slot)
            result.append(straight_through(value, quantizer).to(torch.bfloat16))
        return tuple(result)


def packed_forward(x, w0, b0, w2, b2, smooth2, alpha, activation_quantizer, *, training=False):
    hidden = F.gelu(F.linear(x, w0, b0), approximate='tanh')
    activation = (hidden.float() / smooth2).to(torch.bfloat16)
    activation = straight_through(activation, activation_quantizer) if training else activation_quantizer(activation, force_fp32=True)
    activation = (activation.float() * alpha).to(torch.bfloat16)
    return F.linear(activation, w2, b2)


def replay_main_rows(x, scope, w0, b0, w2, b2, smooth2, alpha, quantizer):
    shape, stride = tuple(scope['native_V1_shape']), tuple(scope['native_V1_stride'])
    if len(shape) != 2 or shape != (scope['full_rows'], 5120) or stride != (5120, 1):
        raise RuntimeError('LA1 requires observed native contiguous two-dimensional geometry')
    full = torch.zeros(shape, device=x.device, dtype=torch.bfloat16)
    ids = torch.tensor(scope['selected_rows'], device=x.device)
    full.index_copy_(0, ids, x)
    result = packed_forward(full, w0, b0, w2, b2, smooth2, alpha, quantizer)
    return result.index_select(0, ids)
