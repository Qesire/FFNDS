"""Training-only support and prior-centered recovery for a frozen UB1 study.

All fits are offline. Whole64 variance scoring adapts FLAP's WIFV principle;
the ridge consumer uses ordinary linear reconstruction around original W2.
These helpers do not execute W1, activation QDQ or a complete model.
"""
import numpy as np
import torch
import torch.nn.functional as F

from centered_signal_fit import fit_signal_moments
from reduced_rank_residual import fit_rrr, native_parameters


def _finite(*values):
    if any(not bool(torch.isfinite(v).all()) for v in values):
        raise ValueError('Nonfinite recovery input or output')


def group_weight_gram(weight, gate, charge, group_size=64):
    """gated W2 Gram per legal original group; reusable across training folds."""
    if weight.ndim != 2 or weight.shape[1] % group_size or gate.numel() != weight.shape[0]:
        raise ValueError('Group/weight/gate shape mismatch')
    _finite(weight, gate)
    d, h = weight.shape
    blocks = weight.double().reshape(d, h // group_size, group_size).permute(1, 2, 0)
    blocks = blocks * gate.double().reshape(1, 1, d)
    charge('group_weight_gram')
    return torch.bmm(blocks, blocks.transpose(1, 2))


def select_supports(activation, weight_gram, keep, charge):
    """Exact C_mean quadratic form and pooled WIFV-like whole-group score.

    The caller supplies training sources only. Pooled variance includes variation
    between source means as well as within each source; equal rows imply equal
    source weights here. This is a group/gate adaptation, not full FLAP software.
    """
    if activation.ndim != 2 or weight_gram.ndim != 3:
        raise ValueError('Support input dimension mismatch')
    groups, width, second = weight_gram.shape
    if second != width or activation.shape[1] != groups * width or not 0 < keep <= groups or len(activation) < 2:
        raise ValueError('Support budget or group shape mismatch')
    _finite(activation, weight_gram)
    a = activation.double()
    blocks = a.reshape(len(a), groups, width).permute(1, 2, 0)
    charge('group_activation_gram')
    activation_gram = torch.bmm(blocks, blocks.transpose(1, 2))
    cmean = (activation_gram * weight_gram).sum((1, 2)) / len(a)
    # Variance is over all N equally weighted training rows, not only the
    # within-source variances. No held-out row enters these moments.
    variance = a.var(0, unbiased=False).reshape(groups, width)
    flap = (variance * weight_gram.diagonal(dim1=1, dim2=2)).sum(1)
    _finite(cmean, flap)
    result = {'scores': {'cmean': cmean.tolist(), 'flap': flap.tolist()}}
    for name, score in [('cmean', cmean), ('flap', flap)]:
        # Stable sort chooses the smaller original group ID on an exact tie.
        result[name] = torch.sort(torch.argsort(score, descending=True, stable=True)[:keep]).values
    return result


def scalar_parameters(activation, weight, raw_bias, gate, dense_no_bias_fp32,
                      charge, rows_per_source=256):
    """Reuse UB1's centered nonnegative gain and folded-bias recipe."""
    if activation.ndim != 2 or weight.ndim != 2 or activation.shape[1] != weight.shape[1]:
        raise ValueError('Scalar feature geometry mismatch')
    n, d = len(activation), weight.shape[0]
    if n == 0 or n % rows_per_source or dense_no_bias_fp32.shape != (n, d) or raw_bias.shape != (d,) or gate.numel() != d:
        raise ValueError('Scalar source/target geometry mismatch')
    if weight.dtype != torch.bfloat16 or activation.dtype != torch.bfloat16 or dense_no_bias_fp32.dtype != torch.float32:
        raise ValueError('Scalar recipe requires BF16 features/weight and FP32 dense projection')
    _finite(activation, weight, raw_bias, gate, dense_no_bias_fp32)
    sum_s = torch.zeros(d, dtype=torch.float64, device=activation.device)
    sum_y = torch.zeros_like(sum_s)
    ss = sy = 0.
    for start in range(0, n, rows_per_source):
        charge('scalar_signal_FP32_linear')
        signal = F.linear(activation[start:start + rows_per_source].float(), weight.float())
        dense = dense_no_bias_fp32[start:start + rows_per_source]
        sum_s += signal.double().sum(0)
        sum_y += dense.double().sum(0)
        sg = (signal * gate.float().reshape(1, -1)).double()
        yg = (dense * gate.float().reshape(1, -1)).double()
        ss += float(sg.square().sum())
        sy += float((sg * yg).sum())
    fitted = fit_signal_moments(n, sum_s.cpu().numpy(), sum_y.cpu().numpy(), ss, sy,
                                gate.float().cpu().numpy())
    if fitted['centered_alpha'] is None:
        raise ValueError('Degenerate centered gain; no fallback')
    alpha = float(np.float32(fitted['centered_alpha']))
    correction = fitted['mean_dense'] - alpha * fitted['mean_signal']
    bias = (raw_bias.double() + torch.as_tensor(correction, device=raw_bias.device)).bfloat16()
    _finite(bias)
    return {'alpha': alpha, 'bias': bias,
            'diagnostics': {'training_rows': n, 'alpha_FP64': fitted['centered_alpha'],
                            'centered_denominator': fitted['centered_denominator'],
                            'centered_numerator': fitted['centered_numerator']}}


def ridge_consumer(features, target, prior_weight, prior_bias, charge, fraction=.01):
    """Centered dual ridge around the original selected W2 and scalar bias.

    Minimize ||Fc Delta - Ec||_F^2 + lambda ||Delta||_F^2,
    lambda=.01*tr(Fc Fc^T)/N, then fold Delta and an unpenalized intercept
    into one BF16 consumer. No inverse gate is needed. Gate affects support,
    scalar fitting, residual training and evaluation, not this ridge solve.
    """
    if features.ndim != 2 or target.ndim != 2 or len(features) != len(target) or len(features) < 2:
        raise ValueError('Consumer training geometry mismatch')
    n, k = features.shape
    d = target.shape[1]
    if prior_weight.shape != (d, k) or prior_bias.shape != (d,) or not np.isfinite(fraction) or fraction <= 0:
        raise ValueError('Consumer prior or regularization mismatch')
    if any(v.dtype != torch.bfloat16 for v in [features, target, prior_weight, prior_bias]):
        raise ValueError('Consumer deployment inputs and prior must be BF16')
    _finite(features, target, prior_weight, prior_bias)
    charge('consumer_ridge_fit')
    x = features.double()
    mean_x = x.mean(0, keepdim=True)
    xc = x - mean_x
    prior = F.linear(x, prior_weight.double(), prior_bias.double())
    error = target.double() - prior
    mean_e = error.mean(0, keepdim=True)
    ec = error - mean_e
    gram = xc @ xc.T
    penalty = max(float(fraction * gram.trace() / n), 1e-24)
    gram.diagonal().add_(penalty)
    chol = torch.linalg.cholesky(gram)
    dual = torch.cholesky_solve(ec, chol)
    delta = xc.T @ dual
    new_weight = (prior_weight.double() + delta.T).bfloat16().contiguous()
    new_bias = (prior_bias.double().reshape(1, -1) + mean_e - mean_x @ delta).reshape(-1).bfloat16().contiguous()
    _finite(new_weight, new_bias)
    return {'weight': new_weight, 'bias': new_bias,
            'diagnostics': {'ridge_fraction': fraction, 'ridge_lambda': penalty,
                            'training_rows': n, 'selected_features': k,
                            'centered_rank_upper_bound': min(n - 1, k),
                            'unidentified_feature_directions_at_least': max(0, k - n + 1),
                            'FP64_weight_delta_frobenius': float(delta.norm()),
                            'consumer_weight_BF16_bytes': new_weight.numel() * new_weight.element_size(),
                            'solver': 'FP64 dual Gram; Cholesky; no hyperparameter search'}}


def fit_residual(x, target, main, gate, charge):
    """The same whitened rank256 ridge recipe, re-fit to each arm's main."""
    if x.ndim != 2 or x.shape[1] != 5120 or target.shape != main.shape or target.shape != (len(x), 5120) or gate.numel() != 5120 or len(x) < 256:
        raise ValueError('Frozen R256 training geometry mismatch')
    if any(t.dtype != torch.bfloat16 for t in [x, target, main, gate]):
        raise ValueError('R256 native states must be BF16')
    _finite(x, target, main, gate)
    charge('RR_fit')
    label = (target.float() - main.float()) * gate.float().reshape(1, -1)
    model = fit_rrr(x, label, 1., 256, seed=20261021)
    return native_parameters(model, gate, 256)


def gated_metrics(prediction, target, gate):
    """Same BF16 gate product as the native caller; residual is outside scope."""
    if prediction.ndim != 2 or prediction.shape != target.shape or gate.numel() != target.shape[1] or any(x.dtype != torch.bfloat16 for x in [prediction, target, gate]):
        raise ValueError('Gate-only metric geometry/dtype mismatch')
    _finite(prediction, target, gate)
    p = (prediction * gate.reshape(1, -1)).double()
    y = (target * gate.reshape(1, -1)).double()
    diff = p - y
    sse, energy = float(diff.square().sum()), float(y.square().sum())
    return {'SSE': sse, 'target_energy': energy,
            'relative_SSE': sse / energy if energy else None,
            'relative_L2': (sse / energy) ** .5 if energy else None,
            'max_abs': float(diff.abs().max()), 'rows': len(target)}
