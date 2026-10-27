"""A consumer increment fitted to the actual old BF16 UB1 residual error.

Original support, gain, folded bias and R256 remain fixed. This is an offline
ridge reconstruction; its new BF16 execution must be measured separately.
"""
import math

import torch


def baseline_error(target, main, old_residual):
    if target.shape != main.shape or target.shape != old_residual.shape or any(x.dtype != torch.bfloat16 for x in (target, main, old_residual)):
        raise ValueError('Baseline requires matching BF16 branch tensors')
    combined = main + old_residual
    error = target.float() - combined.float()
    if not bool(torch.isfinite(error).all()):
        raise ValueError('Nonfinite baseline error')
    return error


def fit_delta_consumer(features, error_fp32, prior_weight, prior_bias, charge,
                       fraction=.01):
    """Fit Fc D≈Ec and fold D, its intercept into the selected original W2.

    E must be Y-float(BF16_add(original_main, original_R256)), without adding
    the old residual or original bias again. All features/parameters below are
    local training inputs, not online observers or additional deployment work.
    """
    if features.ndim != 2 or error_fp32.ndim != 2 or len(features) != len(error_fp32) or len(features) < 2:
        raise ValueError('Consumer delta training geometry mismatch')
    n, k = features.shape
    d = error_fp32.shape[1]
    if prior_weight.shape != (d, k) or prior_bias.shape != (d,):
        raise ValueError('Consumer delta prior geometry mismatch')
    if features.dtype != torch.bfloat16 or error_fp32.dtype != torch.float32 or prior_weight.dtype != torch.bfloat16 or prior_bias.dtype != torch.bfloat16:
        raise ValueError('BF16 features/prior and FP32 baseline error required')
    if not math.isfinite(fraction) or fraction <= 0 or any(not bool(torch.isfinite(v).all()) for v in (features, error_fp32, prior_weight, prior_bias)):
        raise ValueError('Invalid regularization or nonfinite consumer input')
    charge('consumer_delta_fit')
    x, e = features.double(), error_fp32.double()
    mean_x, mean_e = x.mean(0, keepdim=True), e.mean(0, keepdim=True)
    xc, ec = x - mean_x, e - mean_e
    gram = xc @ xc.T
    penalty = max(float(fraction * gram.trace() / n), 1e-24)
    gram.diagonal().add_(penalty)
    chol = torch.linalg.cholesky(gram)
    coefficients = torch.cholesky_solve(ec, chol)
    delta = xc.T @ coefficients
    increment_bias = (mean_e - mean_x @ delta).reshape(-1)
    weight = (prior_weight.double() + delta.T).bfloat16().contiguous()
    bias = (prior_bias.double() + increment_bias).bfloat16().contiguous()
    if not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(bias).all()):
        raise ValueError('Nonfinite folded BF16 consumer')
    return {'weight': weight, 'bias': bias,
            'diagnostics': {'training_rows': n, 'selected_features': k,
                            'ridge_fraction': fraction, 'ridge_lambda': penalty,
                            'centered_rank_upper_bound': min(n - 1, k),
                            'unidentified_feature_directions_at_least': max(0, k - n + 1),
                            'weight_delta_FP64_frobenius': float(delta.norm()),
                            'bias_increment_FP64_l2': float(increment_bias.norm()),
                            'original_R256_refitted': False,
                            'error_target': 'float(Y) - float(BF16_add(original_main, original_R256))',
                            'solver': 'FP64 centered dual ridge, Cholesky; fixed0.01 trace/N'}}
