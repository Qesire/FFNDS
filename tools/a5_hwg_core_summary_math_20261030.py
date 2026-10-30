"""Source-equal descriptive comparisons; no scientific I/O or authority."""
import math
import random
import statistics


def summarize_sources(source_values, *, draws=1024, seed=20261030):
    """Sources are independent units; nested sites are averaged first.

    Bootstrap ranges describe these exposed sources and do not validate a
    new-source population or turn calibration sources into confirmation.
    """
    if not source_values or draws <= 0:
        raise ValueError('finite nonempty source comparison')
    means = []
    for values in source_values:
        present = [float(v) for v in values if v is not None]
        if not present or not all(math.isfinite(v) for v in present):
            raise ValueError('each source needs a finite defined observation')
        means.append(statistics.fmean(present))
    rng = random.Random(seed)
    boot = sorted(statistics.fmean(rng.choices(means, k=len(means))) for _ in range(draws))
    return {'source_equal_mean': statistics.fmean(means),
            'source_mean_std': statistics.stdev(means) if len(means) > 1 else 0.,
            'source_means': means, 'independent_source_count': len(means),
            'defined_sites_per_source': [sum(v is not None for v in a) for a in source_values],
            'descriptive_source_bootstrap_95_percentile': [boot[int(.025 * (draws - 1))], boot[int(.975 * (draws - 1))]],
            'bootstrap_draws': draws, 'bootstrap_seed': seed,
            'new_source_or_video_quality_confirmation': False}


def metric_vector(arm):
    support, output, ac, post = [arm[k] for k in ['support', 'signed_output', 'feature_AC', 'sample_postwrite']]
    result = {k: output[k] for k in ['gated_current_relative_L2', 'gated_change_relative_L2',
        'gated_change_direction_cosine', 'captured_postQ2_delta_energy_fraction',
        'full_delta_rounding_difference_relative_to_current',
        'retained_omitted_gated_signed_cross', 'rounding_omission_signed_cross',
        'squared_error', 'signed_error_decomposition_residual']}
    result.update({k: support[k] for k in ['selected_channel_fraction', 'K64_union_all_rows',
        'mean_contiguous_K64_runs_per_row', 'latent_tau_same_spatial_switch_fraction',
        'horizontal_support_Jaccard', 'vertical_support_Jaccard', 'latent_tau_support_Jaccard']})
    for tile in support['physical_tiles']:
        result[f'M{tile["M_tile"]}_K64_occupied_fraction'] = tile['occupied_fraction']
        result[f'M{tile["M_tile"]}_K64_padded_physical_MAC_fraction'] = tile['physical_tile_MAC_fraction_with_M_padding']
    result.update({k: ac[k] for k in ['latent_tau_AC_relative_L2', 'within_tau_spatial_AC_relative_L2']})
    result.update({k: post[k] for k in ['gated_writeback_relative_L2', 'post_residual_relative_L2']})
    return result
