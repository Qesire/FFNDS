"""Original R2 learned affine backbone with retained native-Q2 remainders.

Mixed BF16 folded arithmetic is explicit. There is no extra remainder QDQ,
no quantization of the derived matrix, and no R256 correction branch.
"""
import torch
import torch.nn.functional as F

from onpolicy_residual import same_bits


def fit_channel_affine(scaled_z, native_v2):
    x, y = scaled_z.double(), native_v2.double()
    if x.shape != y.shape or x.ndim != 2 or not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError('Finite paired per-channel operands required')
    mean_x, mean_y = x.mean(0), y.mean(0)
    centered = x - mean_x
    variance = centered.square().sum(0)
    covariance = (centered * (y - mean_y)).sum(0)
    slope = torch.where(variance > 0, covariance / variance.clamp_min(torch.finfo(torch.float64).tiny), 0.)
    intercept = mean_y - slope * mean_x
    return slope.float().contiguous(), intercept.float().contiguous()


def fold_learned(full, b2, slope, intercept):
    left = full['w2'].double() * (slope.double() / full['smooth2'].double()).reshape(1, -1)
    matrix = left @ full['w0'].double()
    bias = b2.double() + left @ full['b0'].double() + full['w2'].double() @ intercept.double()
    return matrix.bfloat16().contiguous(), bias.bfloat16().contiguous()


@torch.no_grad()
def observe_native(case, full, quantizer):
    scope = case['scope']
    if scope['native_V1_shape'] != [scope['full_rows'], 5120] or scope['native_V1_stride'] != [5120, 1]:
        raise RuntimeError('Original full native FFN geometry required')
    ids = torch.tensor(scope['selected_rows'], device='cuda')
    x = torch.zeros((scope['full_rows'], 5120), dtype=torch.bfloat16, device='cuda')
    x.index_copy_(0, ids, case['V1'])
    z = F.linear(x, full['w0'], full['b0'])
    v2 = quantizer((F.gelu(z, approximate='tanh').float() / full['smooth2']).bfloat16(), force_fp32=True).bfloat16()
    teacher = F.linear(v2, full['w2'], case['b2']).index_select(0, ids)
    if not same_bits(teacher, case['TeacherY']):
        raise RuntimeError('Original full native Q2/TeacherY reconstruction differs')
    return z.index_select(0, ids).contiguous(), v2.index_select(0, ids).contiguous()


def remainder(z, v2, full, slope, intercept):
    return (v2.float() - z.float() / full['smooth2'] * slope - intercept).bfloat16()


@torch.no_grad()
def native_base(case, matrix, bias):
    ids = torch.tensor(case['scope']['selected_rows'], device='cuda')
    x = torch.zeros((case['scope']['full_rows'], 5120), dtype=torch.bfloat16, device='cuda')
    x.index_copy_(0, ids, case['V1'])
    return F.linear(x, matrix, bias).index_select(0, ids)


@torch.no_grad()
def native_prediction(case, full, matrix, bias, slope, intercept, groups, quantizer):
    groups = sorted(map(int, groups))
    if len(groups) != len(set(groups)) or any(g < 0 or g >= 216 for g in groups):
        raise ValueError('Unique original64-channel groups required')
    channels = torch.tensor([g * 64 + offset for g in groups for offset in range(64)], device='cuda')
    row_ids = torch.tensor(case['scope']['selected_rows'], device='cuda')
    x = torch.zeros((case['scope']['full_rows'], 5120), dtype=torch.bfloat16, device='cuda')
    x.index_copy_(0, row_ids, case['V1'])
    base = F.linear(x, matrix, bias)
    w0, b0 = full['w0'].index_select(0, channels).contiguous(), full['b0'].index_select(0, channels)
    w2 = full['w2'].index_select(1, channels).contiguous()
    smooth = full['smooth2'].index_select(0, channels)
    z = F.linear(x, w0, b0)
    v2 = quantizer((F.gelu(z, approximate='tanh').float() / smooth).bfloat16(), force_fp32=True).bfloat16()
    r = (v2.float() - z.float() / smooth * slope.index_select(0, channels) - intercept.index_select(0, channels)).bfloat16()
    # All original b2 and all-channel affine intercept already live in bias.
    output = base + F.linear(r, w2, None)
    return output.index_select(0, row_ids)


def greedy_fixed_support(gram, correlation, count, target_norm2):
    """One deterministic training-only pass with all signed cross terms."""
    g, b = gram.double(), correlation.double()
    if g.shape != (len(b), len(b)) or count < 0 or count > len(b):
        raise ValueError('Matched group Gram and a fixed count required')
    current = torch.zeros_like(b)
    selected, gains = [], []
    risk = float(target_norm2)
    for _ in range(count):
        improvement = 2 * b - 2 * current - g.diag()
        if selected: improvement[selected] = -torch.inf
        group = int(torch.argmax(improvement)); gain = float(improvement[group])
        selected.append(group); gains.append(gain); risk -= gain
        current.add_(g[:, group])
    return {'groups': sorted(selected), 'selection_order': selected,
            'incremental_training_risk_reductions': gains, 'ideal_training_risk': risk}
