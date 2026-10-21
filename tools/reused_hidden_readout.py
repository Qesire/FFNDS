"""Fixed training-only normalization for an extra readout of frozen R256 features."""
import torch
import torch.nn.functional as F


def training_moments(hidden):
    if (hidden.ndim != 2 or hidden.shape[1] != 256 or len(hidden) < 2
            or not torch.isfinite(hidden).all()):
        raise ValueError('Finite training rows of the frozen256-wide hidden state required')
    x = hidden.double()
    mean = x.mean(0)
    scale = (x - mean).square().mean(0).sqrt().clamp_min(1e-4)
    return {'mean': mean.float(), 'scale': scale.float()}


def readout_features(hidden, moments, *, nonlinear):
    if (hidden.dtype != torch.bfloat16 or hidden.shape[-1] != 256
            or any(v.dtype != torch.float32 or v.shape != (256,) for v in moments.values())
            or set(moments) != {'mean', 'scale'} or not torch.isfinite(hidden).all()
            or any(not torch.isfinite(v).all() for v in moments.values())
            or torch.any(moments['scale'] < 1e-4)):
        raise ValueError('Exact BF16 hidden and finite FP32 training moments required')
    z = ((hidden.float() - moments['mean']) / moments['scale']).bfloat16()
    return F.gelu(z, approximate='tanh') if nonlinear else z
