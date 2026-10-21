"""Fixed ridge heads on existing residual features; no deployment speed claim."""
import torch
import torch.nn.functional as F


def features(hidden, quadratic):
    x = hidden.float()
    if x.ndim < 2 or not torch.isfinite(x).all():
        raise ValueError('Finite row features required')
    return torch.cat([x, x.square()], dim=-1) if quadratic else x


@torch.no_grad()
def fit_head(hidden, targets, quadratic):
    x, y = features(hidden, quadratic).double(), targets.double()
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2 or not torch.isfinite(y).all():
        raise ValueError('Paired finite training rows required')
    mean, output_mean = x.mean(0), y.mean(0)
    scale = (x - mean).square().mean(0).sqrt().clamp_min(1e-4)
    z = (x - mean) / scale
    covariance = z.T @ z / len(z)
    covariance.diagonal().add_(1.)
    coefficients = torch.linalg.solve(covariance, z.T @ (y - output_mean) / len(z)) / scale[:, None]
    bias = output_mean - mean @ coefficients
    return {'weight': coefficients.T.float().contiguous(), 'bias': bias.float().contiguous()}


def predict_head(hidden, parameters, quadratic):
    return F.linear(features(hidden, quadratic), parameters['weight'], parameters['bias'])


def support(scores, keep):
    if scores.ndim != 2 or not 0 < keep <= scores.shape[1] or not torch.isfinite(scores).all():
        raise ValueError('Finite scores and valid group budget required')
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :keep]


def retained_fraction(profiles, selected):
    if (profiles.ndim != 2 or selected.ndim != 2 or len(profiles) != len(selected)
            or torch.any(selected < 0) or torch.any(selected >= profiles.shape[1])):
        raise ValueError('Paired valid profile/support rows required')
    return float(profiles.gather(1, selected).sum(1).mean())
