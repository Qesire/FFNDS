"""Shared fixed-probe recovery and paired-value least squares."""
import torch


def pfrr(selected, teacher, probes):
    """SparsePR v2 A.3 with matched S/P, equal coverage and all value channels."""
    on_probe = selected.index_select(1, probes)
    mean = on_probe.mean(1, keepdim=True)
    centered = on_probe - mean
    inv_std = (centered.square().mean(1, keepdim=True) + 1e-6).rsqrt()
    xp = centered * inv_std
    residual = teacher - on_probe
    intercept = residual.mean(1, keepdim=True)
    gram = xp @ xp.transpose(1, 2)
    gram.diagonal(dim1=-2, dim2=-1).add_(1)
    coefficient = xp.transpose(1, 2) @ torch.linalg.solve(gram, residual - intercept)
    pred = selected + intercept + ((selected - mean)*inv_std) @ coefficient
    pred.index_copy_(1, probes, teacher)
    return pred


def paired_map(x, y, scale, ridge=1e-3):
    """[head, sample, channel] FP32 fit around diag(current standard deviation)."""
    xt = x.transpose(1, 2)
    g = xt @ x / x.shape[1]
    g.diagonal(dim1=-2, dim2=-1).add_(ridge)
    rhs = xt @ (y - x*scale) / x.shape[1]
    fit = torch.linalg.solve(g, rhs)
    fit.diagonal(dim1=-2, dim2=-1).add_(scale[:,0])
    return fit, g
