"""Signed calibration statistics for frozen whole-K64 supports.

The greedy support is a specific feasible calibration baseline, not a claim
to the globally optimal cardinality-constrained signed reconstruction.
"""
import torch


def group_quadratic(delta_v2, weight2, target_y_change, current_gate, reference_y,
                    *, group_size=64, measured=lambda name, fn: fn()):
    """All group contributions on actual fixed calibration rows, fully paid.

Returns source-normalized Gram and target correlation, preserving signed
cross terms. It performs one batched group W2, one Gram and one target dot.
"""
    if delta_v2.ndim != 2 or delta_v2.shape[1] % group_size:
        raise ValueError('whole original hidden groups')
    rows, hidden = delta_v2.shape
    groups = hidden // group_size
    if weight2.shape != (target_y_change.shape[1], hidden) or target_y_change.shape[0] != rows:
        raise ValueError('matched actual postQ2 coordinates')
    gate = current_gate.float().reshape(1, -1)
    denominator = (reference_y.float() * gate).double().square().sum()
    if float(denominator) == 0:
        raise ValueError('zero calibration current-output reference')
    def project():
        a = delta_v2.float().reshape(rows, groups, group_size).permute(1, 0, 2).contiguous()
        b = weight2.float().reshape(weight2.shape[0], groups, group_size).permute(1, 2, 0).contiguous()
        return torch.bmm(a, b) * gate
    contributions = measured('calibration_group_W2_batched', project).reshape(groups, -1)
    target = (target_y_change.float() * gate).reshape(-1)
    gram = measured('calibration_signed_Gram', lambda: contributions @ contributions.T)
    correlation = measured('calibration_signed_target_dot', lambda: contributions @ target)
    # Source normalization is FP64 after the explicitly FP32 matrix products.
    return gram.double() / denominator, correlation.double() / denominator, {
        'rows': rows, 'groups': groups, 'group_size': group_size,
        'batched_W2_shape': [groups, rows, group_size, weight2.shape[0]],
        'Gram_shape': [groups, groups, rows * weight2.shape[0]],
        'current_gated_output_squared_norm': float(denominator),
        'target_gated_change_squared_norm': float(target.double().square().sum()),
        'signed_objective': 'actual_BF16_Y_change_vs_paid_FP32_group_projection',
        'Gram_and_target_product_dtype': 'float32_then_float64_source_normalization',
        'roundoff_can_change_nearly_tied_or_cancelling_marginals': True,
    }


def greedy_support(gram, correlation, count):
    """Forward greedy exact marginal reduction under this measured quadratic."""
    if gram.ndim != 2 or gram.shape != (correlation.numel(), correlation.numel()):
        raise ValueError('matched Gram and target correlation')
    if not 0 <= count <= correlation.numel() or not torch.isfinite(gram).all() or not torch.isfinite(correlation).all():
        raise ValueError('finite quadratic and feasible exact cardinality')
    if not torch.allclose(gram, gram.T, atol=1e-9, rtol=1e-6):
        raise ValueError('symmetric measured Gram')
    selected = torch.zeros_like(correlation, dtype=torch.bool)
    current_correlation = torch.zeros_like(correlation)
    order, gains = [], []
    for _ in range(count):
        gain = 2 * (correlation - current_correlation) - gram.diagonal()
        gain[selected] = -torch.inf
        index = int(torch.argsort(gain, descending=True, stable=True)[0])
        selected[index] = True
        current_correlation += gram[:, index]
        order.append(index)
        gains.append(float(gain[index]))
    return selected, order, gains
