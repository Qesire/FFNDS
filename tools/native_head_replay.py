"""Differentiable replay of the unchanged final Wan FFN-to-flow boundary.

Inputs are complete native operands, not sampled rows or invented trajectories.
Only the already deployed R256 tensors may receive gradients. Each BF16
multiply/add remains a separate operation, matching the native writeback/head.
"""
import math

import torch
import torch.nn.functional as F

from reduced_rank_residual import predict_y


PARAMETERS = ('first', 'bias_first', 'second', 'bias_second')


def patchify_flow(flow, grid):
    """Inverse of WanPostInfer.unpatchify for patch=(1,2,2)."""
    f, h, w = grid
    c = flow.shape[0]
    if tuple(flow.shape) != (c, f, h * 2, w * 2):
        raise ValueError('Flow/grid geometry mismatch')
    return flow.reshape(c, f, 1, h, 2, w, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(math.prod(grid), 4 * c)


def unpatchify_flow(tokens, grid):
    f, h, w = grid
    if tokens.shape[0] != math.prod(grid) or tokens.shape[1] % 4:
        raise ValueError('Complete native token grid required')
    c = tokens.shape[1] // 4
    return tokens.reshape(f, h, w, 1, 2, 2, c).permute(6, 0, 3, 1, 4, 2, 5).reshape(c, f, h * 2, w * 2)


def replay(data, parameters, *, eps=1e-6):
    """Keep native BF16 rounding boundaries; remove only addmm's out= buffer."""
    y = data['main'] + predict_y(data['u'], parameters)
    x = data['base'] + y * data['gate']
    x = F.layer_norm(x, (x.shape[-1],), eps=eps)
    x = x * (1 + data['scale'])
    x = x + data['shift']
    return torch.addmm(data['head_bias'], x, data['head_weight'])


def normalized_update(parameters, fraction):
    """One preregistered gradient step, fixed relative global L2 displacement."""
    if fraction <= 0 or any(v.grad is None or not bool(torch.isfinite(v.grad).all()) for v in parameters.values()):
        raise ValueError('Finite connected gradients and positive step required')
    norm = sum(v.detach().double().square().sum() for v in parameters.values()).sqrt()
    grad_norm = sum(v.grad.detach().double().square().sum() for v in parameters.values()).sqrt()
    if not bool(norm > 0 and grad_norm > 0):
        raise ValueError('Nonzero parameter and gradient norms required')
    multiplier = fraction * float(norm / grad_norm)
    updated = {k: v.detach() - multiplier * v.grad.detach() for k, v in parameters.items()}
    return updated, {'relative_global_L2_step': fraction, 'parameter_L2': float(norm),
                     'gradient_L2': float(grad_norm), 'gradient_multiplier': multiplier}
