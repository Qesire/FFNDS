"""Draft E0 algebra and mask primitives; no acquisition or run authority.

Small synthetic fixtures may call these functions. Real saved tensor studies
need their own exact independent review and registered finite execution scope.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shared_support(scores, keep_channels, *, hidden_group=64, row_group=None,
                   permutation=None):
    """Sum squared scores before selection, with exact channel-index ties.

row_group=None gives a common whole-M mask; row_group=1 gives independent
token masks. A fixed row permutation changes grouping only, never coordinates.
The last partial row group is real, unpadded data during scoring.
"""
    require(scores.ndim == 2 and bool(torch.isfinite(scores).all()), 'finite MxH scores')
    m, h = scores.shape
    require(h % hidden_group == 0 and keep_channels % hidden_group == 0
            and 0 <= keep_channels <= h, 'exact equal-K hidden groups')
    size = m if row_group is None else row_group
    require(type(size) is int and size > 0, 'positive row group')
    if permutation is None:
        permutation = torch.arange(m, device=scores.device)
    require(permutation.shape == (m,) and permutation.dtype == torch.int64
            and torch.equal(permutation.sort().values, torch.arange(m, device=scores.device)),
            'complete fixed row permutation')
    grouped = scores.index_select(0, permutation).float().square().reshape(m, h // hidden_group, hidden_group).sum(-1)
    groups = math.ceil(m / size)
    if m % size:
        grouped = F.pad(grouped, (0, 0, 0, groups * size - m))
    energy = grouped.reshape(groups, size, h // hidden_group).sum(1)
    order = torch.argsort(energy, dim=1, descending=True, stable=True)
    selected = order[:, :keep_channels // hidden_group]
    group_mask = torch.zeros_like(energy, dtype=torch.bool).scatter_(1, selected, True)
    row_masks = group_mask.repeat_interleave(size, 0)[:m].repeat_interleave(hidden_group, 1)
    result = torch.empty_like(row_masks)
    result[permutation] = row_masks
    return result, selected.sort(dim=1).values


def tile_occupancy(mask, *, m_tile=16, k_tile=64):
    require(mask.ndim == 2 and mask.dtype == torch.bool, 'boolean MxH mask')
    m, h = mask.shape
    require(h % k_tile == 0 and m_tile > 0, 'whole physical hidden groups')
    padded_m = math.ceil(m / m_tile) * m_tile
    padded = F.pad(mask, (0, 0, 0, padded_m - m))
    occupied = padded.reshape(padded_m // m_tile, m_tile, h // k_tile, k_tile).any(3).any(1)
    return {
        'M_tile': m_tile, 'K_tile': k_tile, 'total_tiles': occupied.numel(),
        'nonempty_tiles': int(occupied.sum()),
        'all_zero_tiles': int((~occupied).sum()),
        'occupied_fraction': float(occupied.float().mean()),
        'real_selected_elements': int(mask.sum()),
        'padded_M': padded_m,
        'physical_tile_MAC_fraction_with_M_padding':
            float(occupied.sum()) * m_tile * k_tile / (m * h),
    }


def jaccard(left, right):
    require(left.dtype == right.dtype == torch.bool and left.shape == right.shape,
            'matched boolean supports')
    intersection = (left & right).sum(-1).double()
    union = (left | right).sum(-1).double()
    return torch.where(union > 0, intersection / union.clamp_min(1), 1.)


def support_summary(mask, *, grid=None):
    require(mask.ndim == 2 and mask.dtype == torch.bool, 'boolean MxH support')
    m, h = mask.shape
    require(h % 64 == 0, 'native K64 geometry')
    groups = mask.reshape(m, h // 64, 64).any(-1)
    starts = groups & ~F.pad(groups[:, :-1], (1, 0), value=False)
    out = {
        'selected_channel_fraction': float(mask.float().mean()),
        'per_row_nonempty_K64_mean': float(groups.sum(-1).double().mean()),
        'K64_union_all_rows': int(groups.any(0).sum()),
        'mean_contiguous_K64_runs_per_row': float(starts.sum(-1).double().mean()),
        'flattened_adjacent_support_Jaccard_including_raster_boundaries': float(jaccard(mask[:-1], mask[1:]).mean()) if m > 1 else None,
        'physical_tiles': [tile_occupancy(mask, m_tile=t) for t in (16, 32, 64)],
    }
    if grid is not None:
        require(len(grid) == 3 and math.prod(grid) == m, 'actual latent grid')
        tau = mask.reshape(grid[0], grid[1] * grid[2], h)
        out['latent_tau_same_spatial_support_Jaccard'] = float(jaccard(tau[:-1], tau[1:]).mean())
        out['latent_tau_same_spatial_switch_fraction'] = float((tau[:-1] != tau[1:]).float().mean())
        spatial = mask.reshape(*grid, h)
        for name, left, right in [
            ('horizontal', spatial[:, :, :-1], spatial[:, :, 1:]),
            ('vertical', spatial[:, :-1], spatial[:, 1:]),
            ('latent_tau', spatial[:-1], spatial[1:]),
        ]:
            pairs = left.numel() // h
            out[name + '_real_neighbor_pairs'] = pairs
            out[name + '_support_Jaccard'] = float(jaccard(left, right).mean()) if pairs else None
    return out


def signed_output_metrics(old_y, current_y, delta_v2, weight2, gate, mask,
                          *, full_delta_output=None, retained_delta_output=None):
    """Paid FP32 projection diagnostic, explicitly distinct from native action.

Actual original BF16 old/current Y are the references. Bias already occurs
once in oldY. delta_v2 is Q2(h1/s2)-Q2(h0/s2), never Q2(delta-h/s2).
The full-delta rounding difference is measured, not called a lower bound.
"""
    require(old_y.shape == current_y.shape and delta_v2.shape == mask.shape,
            'matched original operands/support')
    require(weight2.shape == (old_y.shape[1], delta_v2.shape[1])
            and gate.numel() == old_y.shape[1], 'postQ2 sensitivity geometry')
    require(mask.dtype == torch.bool, 'boolean support')
    require(all(bool(torch.isfinite(t).all()) for t in
                (old_y, current_y, delta_v2, weight2, gate)), 'finite signed operands')
    d = delta_v2.float()
    w = weight2.float()
    g = gate.float().reshape(1, -1)
    full = F.linear(d, w) if full_delta_output is None else full_delta_output
    require(full.shape == old_y.shape and full.dtype == torch.float32
            and bool(torch.isfinite(full).all()), 'finite FP32 full-delta reference')
    retained = (F.linear(d * mask, w) if retained_delta_output is None
                else retained_delta_output)
    require(retained.shape == old_y.shape and retained.dtype == torch.float32
            and bool(torch.isfinite(retained).all()), 'finite FP32 retained-delta output')
    omitted = full - retained
    actual_change = g * (current_y.float() - old_y.float())
    predicted_change = g * retained
    current = g * current_y.float()
    approximation = g * (old_y.float() + retained)
    floor = g * (current_y.float() - old_y.float() - full)
    omitted_g = g * omitted
    error = current - approximation
    norm2 = lambda x: x.double().square().sum()
    dot = lambda x, y: (x.double() * y.double()).sum()
    scalar = lambda x: float(x.item())
    def relative(value, target):
        denominator = norm2(target)
        return scalar((norm2(value) / denominator).sqrt()) if scalar(denominator) > 0 else None
    direction_denominator = (norm2(actual_change) * norm2(predicted_change)).sqrt()
    cosine = (scalar(dot(actual_change, predicted_change) / direction_denominator)
              if scalar(direction_denominator) > 0 else None)
    delta_energy = norm2(d)
    decomposition = norm2(floor) + norm2(omitted_g) + 2 * dot(floor, omitted_g)
    return {
        'evidence': 'PAID_TEACHER_STATE_FP32_W2_DIAGNOSTIC_NOT_NATIVE_SKIP',
        'gated_current_relative_L2': relative(error, current),
        'gated_change_relative_L2': relative(actual_change - predicted_change, actual_change),
        'gated_change_direction_cosine': cosine,
        'full_delta_rounding_difference_relative_to_current': relative(floor, current),
        'captured_postQ2_delta_energy_fraction': (scalar(norm2(d * mask) / delta_energy)
                                                if scalar(delta_energy) > 0 else None),
        'zero_gated_current_reference': scalar(norm2(current)) == 0,
        'zero_actual_gated_change': scalar(norm2(actual_change)) == 0,
        'zero_predicted_gated_change': scalar(norm2(predicted_change)) == 0,
        'zero_postQ2_delta_energy': scalar(delta_energy) == 0,
        'retained_omitted_gated_signed_cross': scalar(2 * dot(predicted_change, omitted_g)),
        'rounding_omission_signed_cross': scalar(2 * dot(floor, omitted_g)),
        'squared_error': scalar(norm2(error)),
        'signed_error_decomposition_residual': scalar(norm2(error) - decomposition),
    }


def four_step_cost_identity(keep_channels, hidden=13824, *, proxy_full_w1_calls=0,
                            proxy_fraction_of_full_w1=1.):
    """Equal W1/W2 main-MAC algebra, excluding separately measured overhead.

Each expert's first encounter is Dense. Only steps1/3 can use its own history.
There are four full FFNs; one full W1 equals half of one full FFN.
"""
    require(0 <= keep_channels <= hidden, 'valid K')
    core = .5 + .5 * keep_channels / hidden
    proxy = proxy_full_w1_calls * proxy_fraction_of_full_w1 / 8.
    return {'four_step_selected_main_MAC_fraction': core,
            'additional_proxy_main_MAC_fraction': proxy,
            'core_plus_proxy_MAC_fraction_excluding_other_costs': core + proxy,
            'first_encounter_Dense_sites': 80, 'history_eligible_sites': 80}
