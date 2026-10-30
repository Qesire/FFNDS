"""Draft reviewed-scope building blocks for actual full-M HiF4 replay.

No run entry point or execution authority. Callers must charge every supplied
measurement and every input read; these functions do not open model files.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from a5_hwg_delta_math_20261030 import proxy_weight, hwg
from a5_hwg_support_regularity_math_20261030 import require, shared_support

A5_RANKERS = ('a5_current_hwg', 'a5_delta_hwg', 'a5_delta_nohwg')
LOCAL_RANKERS = (*A5_RANKERS, 'paid_delta_diagonal')
GLOBAL_RANKERS = (*LOCAL_RANKERS, 'p64_rawh', 'calibration_static', 'random')


def native_replay(v1, w1, b1, w2, b2, smooth2, qdq, measured):
    """Original post-Q1→W1/GELU/smooth2/Q2/W2, including BF16 rounds."""
    require(v1.dtype == w1.dtype == w2.dtype == b1.dtype == b2.dtype == torch.bfloat16,
            'original BF16 operands')
    require(v1.ndim == 2 and w1.shape[1] == v1.shape[1]
            and w2.shape == (v1.shape[1], w1.shape[0])
            and b1.shape == (w1.shape[0],) and b2.shape == (w2.shape[0],)
            and smooth2.numel() == w1.shape[0], 'native operand geometry')
    z = measured('native_W1', lambda: F.linear(v1, w1, b1))
    h = measured('native_GELU', lambda: F.gelu(z, approximate='tanh'))
    q2_input = measured('native_smooth2', lambda: (h.float() / smooth2.float().reshape(1, -1)).to(h.dtype))
    v2 = measured('native_Q2', lambda: qdq(q2_input, force_fp32=True).to(h.dtype))
    y = measured('native_W2', lambda: F.linear(v2, w2, b2))
    return {'V1': v1, 'Z': z, 'H': h, 'Q2_input': q2_input, 'V2': v2, 'Y': y}


def verify_actual_samples(replayed, sampled, rows):
    checks = {}
    for key in ('V1', 'Z', 'H', 'Q2_input', 'V2', 'Y'):
        actual = replayed[key].index_select(0, rows).cpu()
        expected = sampled[key]
        require(actual.shape == expected.shape and actual.dtype == expected.dtype,
                'actual sample replay geometry: ' + key)
        checks[key] = bool(torch.equal(actual, expected))
        require(checks[key], 'actual captured native boundary mismatch: ' + key)
    return checks


def proxy_scores(v0, v1, w1, b1, smooth2, weight2, gate, measured):
    """Explicit two-BF16-state A5-inspired adapter, before proxy Q2.

The saved actual post-Q2 delta is the target, not this continuous proxy.
Full-M Dense proxy GEMMs are paid; no low-cost sign kernel is implied.
"""
    wp, gamma, beta = measured('proxy_weight_parameters', lambda: proxy_weight(w1))
    old_pre = measured('proxy_old_W1', lambda: F.linear(v0, wp, b1))
    new_pre = measured('proxy_new_W1', lambda: F.linear(v1, wp, b1))
    old_h = measured('proxy_old_GELU', lambda: F.gelu(old_pre, approximate='tanh'))
    new_h = measured('proxy_new_GELU', lambda: F.gelu(new_pre, approximate='tanh'))
    sensitivity = measured('current_gate_W2_sensitivity', lambda: hwg(weight2, gate))
    def fields():
        scale = smooth2.float().reshape(1, -1)
        delta = (new_h.float() - old_h.float()) / scale
        current = new_h.float() / scale
        return {'a5_current_hwg': current.abs() * sensitivity,
                'a5_delta_hwg': delta.abs() * sensitivity,
                'a5_delta_nohwg': delta.abs()}
    scores = measured('proxy_score_field_algebra', fields)
    return scores, {'weight': wp, 'gamma': gamma, 'beta': beta,
        'sensitivity': sensitivity, 'old_pre': old_pre, 'new_pre': new_pre}


def p64_score(v1, old_h, w1, b1, rows, measured):
    """Original raw-h RMS rule, paid original full-M permutation coordinates."""
    require(rows.shape == (64,) and rows.unique().numel() == 64, '64 distinct original rows')
    gathered = measured('p64_input_gather', lambda: v1.index_select(0, rows))
    old_sample = measured('p64_history_gather', lambda: old_h.index_select(0, rows))
    pre = measured('p64_W1', lambda: F.linear(gathered, w1, b1))
    current = measured('p64_GELU', lambda: F.gelu(pre, approximate='tanh'))
    return measured('p64_rawh_RMS', lambda: (current.float() - old_sample.float()).square().mean(0).sqrt())


def mean_m64_score(v0, v1, proxy_w, b1, smooth2, sensitivity, measured):
    """Explicit new mean-input arm; opposite within-tile changes can cancel."""
    def means(v):
        full, tail = divmod(v.shape[0], 64)
        result = v[:full * 64].reshape(full, 64, -1).float().mean(1)
        if tail:
            result = torch.cat((result, v[full * 64:].float().mean(0, keepdim=True)))
        return result.to(v.dtype)
    a = measured('mean_m64_old_input', lambda: means(v0))
    b = measured('mean_m64_new_input', lambda: means(v1))
    p0 = measured('mean_m64_old_W1', lambda: F.linear(a, proxy_w, b1))
    p1 = measured('mean_m64_new_W1', lambda: F.linear(b, proxy_w, b1))
    h0 = measured('mean_m64_old_GELU', lambda: F.gelu(p0, approximate='tanh'))
    h1 = measured('mean_m64_new_GELU', lambda: F.gelu(p1, approximate='tanh'))
    return measured('mean_score_field_and_expand', lambda:
        ((h1.float() - h0.float()).abs() / smooth2.float().reshape(1, -1) * sensitivity)
        .repeat_interleave(64, 0)[:v0.shape[0]])


def arm_specs():
    """58 total arms, 57 projected, plus one separately counted full delta."""
    result = [dict(name=r + '__global_K64', ranker=r, hidden_group=64, row_group=None,
                   shuffled=False) for r in GLOBAL_RANKERS]
    for ranker in LOCAL_RANKERS:
        for shuffled in (False, True):
            for row_group in (16, 32, 64):
                result.append(dict(name=f'{ranker}__{"shuffle" if shuffled else "raster"}_M{row_group}_K64',
                                   ranker=ranker, hidden_group=64, row_group=row_group, shuffled=shuffled))
        for hidden_group in (1, 64):
            result.append(dict(name=f'{ranker}__token_K{hidden_group}', ranker=ranker,
                               hidden_group=hidden_group, row_group=1, shuffled=False))
        for hidden_group in (16, 32):
            for row_group in (None, 64):
                result.append(dict(name=f'{ranker}__{"global" if row_group is None else "raster_M64"}_K{hidden_group}',
                                   ranker=ranker, hidden_group=hidden_group,
                                   row_group=row_group, shuffled=False))
    result.extend([
        dict(name='p64_rawh__global_K1_diagnostic', ranker='p64_rawh', hidden_group=1,
             row_group=None, shuffled=False),
        dict(name='mean_input_M64_delta_hwg__raster_M64_K64', ranker='mean_m64', hidden_group=64,
             row_group=64, shuffled=False),
        dict(name='oldY_only_zero_update', ranker='oldY', hidden_group=64, row_group=None,
             shuffled=False),
    ])
    require(len(result) == 58 and len({a['name'] for a in result}) == 58, 'complete arm identity')
    return result


def arm_mask(spec, scores, k, shuffle):
    score = scores[spec['ranker']]
    if spec['ranker'] == 'oldY':
        return torch.zeros_like(score, dtype=torch.bool), None
    return shared_support(score, k, hidden_group=spec['hidden_group'], row_group=spec['row_group'],
                          permutation=shuffle if spec['shuffled'] else None)


def pack_selected(indices, total_units):
    """Little-endian bits of compact row-group supports, without dense D2H."""
    require(indices.ndim == 2 and total_units % 8 == 0, 'whole byte mask units')
    mask = torch.zeros((indices.shape[0], total_units), dtype=torch.bool, device=indices.device)
    mask.scatter_(1, indices, True)
    shifts = torch.arange(8, dtype=torch.uint8, device=indices.device)
    return torch.bitwise_left_shift(mask.reshape(indices.shape[0], -1, 8).to(torch.uint8), shifts).sum(-1, dtype=torch.uint8)


def latent_ac_metrics(target_change, predicted_change, grid):
    """Feature-level AC contrasts, not decoded video motion or event scores."""
    require(target_change.shape == predicted_change.shape and target_change.ndim == 2,
            'matched teacher-state feature changes')
    require(target_change.shape[0] == grid[0] * grid[1] * grid[2], 'actual latent grid')
    a = target_change.float().reshape(grid[0], grid[1] * grid[2], -1)
    b = predicted_change.float().reshape_as(a)
    result = {}
    for name, dimension in [('latent_tau_AC', 0), ('within_tau_spatial_AC', 1)]:
        target = a - a.mean(dimension, keepdim=True)
        error = (b - b.mean(dimension, keepdim=True)) - target
        denominator = target.double().square().sum()
        result[name + '_target_energy'] = float(denominator)
        result[name + '_relative_L2'] = (float((error.double().square().sum() / denominator).sqrt())
                                          if float(denominator) > 0 else None)
    result['evidence'] = 'TEACHER_FFN_FEATURE_CHANGE_NOT_VIDEO_QUALITY'
    return result
