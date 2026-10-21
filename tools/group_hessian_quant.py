"""Fixed64-channel native quantization with conditional Hessian compensation.

Known GPTQ/OBS transfer. The native HiF4 quantizer couples64 inputs, so each
entire group is quantized jointly; this is not canonical scalar-column GPTQ.
"""
import torch


def conditional_correction(error, upper_group, upper_future):
    """Optimal future-weight change after a fixed joint block quantization."""
    whitened = torch.linalg.solve_triangular(upper_group.T, error.T, upper=False).T
    return -(whitened @ upper_future)


@torch.no_grad()
def encode_groups(weight, covariance, quantizer, *, group_size=64, damping=.01):
    if weight.ndim != 2 or weight.shape[1] % group_size or covariance.shape != (weight.shape[1],) * 2:
        raise ValueError('Complete contiguous groups and matched covariance required')
    w, h = weight.double().clone(), covariance.double().clone()
    if not bool(torch.isfinite(w).all() and torch.isfinite(h).all()) or damping <= 0:
        raise ValueError('Finite parameters and positive damping required')
    dead = h.diag() == 0
    h[dead, dead] = 1.; w[:, dead] = 0.
    ridge = damping * h.diag().mean()
    h.diagonal().add_(ridge)
    lower = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(lower)
    upper = torch.linalg.cholesky(inverse, upper=True)
    q = torch.empty_like(w)
    for start in range(0, w.shape[1], group_size):
        end = start + group_size
        block = w[:, start:end].contiguous()
        encoded = quantizer(block)
        if encoded.shape != block.shape or not bool(torch.isfinite(encoded).all()):
            raise RuntimeError('Native joint-group quantizer returned invalid values')
        q[:, start:end] = encoded.double()
        if end < w.shape[1]:
            correction = conditional_correction(block - encoded.double(), upper[start:end, start:end], upper[start:end, end:])
            w[:, end:].add_(correction)
    return q, {'group_size': group_size, 'groups': weight.shape[1] // group_size,
               'damping_fraction': damping, 'actual_diagonal_damping': float(ridge),
               'dead_input_columns': int(dead.sum()), 'offline_arithmetic': 'FP64 covariance/Cholesky/compensation',
               'native_quantizer_group_outputs_preserved': True}
