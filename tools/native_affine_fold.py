"""Full-support controls for the existing R2 affine-backbone backlog.

The BF16 split is an attribution control. Only the all-HiF4 split respects
both proposed new main-linear precision boundaries. Neither is compressed.
"""
import torch
import torch.nn.functional as F

from onpolicy_residual import same_bits


def fold_parameters(w0, w2, b0, smooth2):
    """For post-Q1 x: A=.5*W2*diag(1/s2)*W0, bias=.5*W2*b0/s2."""
    left = w2.double() / smooth2.double().reshape(1, -1)
    return .5 * (left @ w0.double()), .5 * (left @ b0.double())


@torch.no_grad()
def native_operands(case, full, quantizer):
    scope = case['scope']
    if tuple(scope['native_V1_shape']) != (scope['full_rows'], 5120) or scope['native_V1_stride'] != [5120, 1]:
        raise RuntimeError('Original full-M contiguous FFN geometry required')
    x = torch.zeros((scope['full_rows'], 5120), device='cuda', dtype=torch.bfloat16)
    ids = torch.tensor(scope['selected_rows'], device='cuda')
    x.index_copy_(0, ids, case['V1'])
    z = F.linear(x, full['w0'], full['b0'])
    hidden = F.gelu(z, approximate='tanh')
    v2 = quantizer((hidden.float() / full['smooth2']).bfloat16(), force_fp32=True).bfloat16()
    teacher = F.linear(v2, full['w2'], case['b2'])
    if not same_bits(teacher.index_select(0, ids), case['TeacherY']):
        raise RuntimeError('Original full native teacher no longer equals captured TeacherY')
    # Preserve the original Q2 value, not GELU(z)/2 alone. This remainder
    # includes the original activation-quantization error and cast effects.
    phi = (v2.float() - .5 * (z.float() / full['smooth2'])).bfloat16()
    return x, ids, phi


@torch.no_grad()
def folded_output(x, ids, phi, full, case, affine_weight, affine_bias, quantizer, *, quantize_remainder):
    remainder = quantizer(phi, force_fp32=True).bfloat16() if quantize_remainder else phi
    base = F.linear(x, affine_weight, affine_bias)
    # Original b2 appears exactly once, in the nonlinear-remainder GEMM.
    output = base + F.linear(remainder, full['w2'], case['b2'])
    return output.index_select(0, ids)
