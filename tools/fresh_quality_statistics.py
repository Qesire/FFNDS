"""Source-cluster statistics for frozen official outputs; no evaluator/model.

Official scoring stays unchanged. MUSIQ detail units are converted only in
the downstream paired analysis. Inference repeats/frames are never samples.
"""
import math
import random
import statistics

from fresh_quality_contract import CORE


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite observed score')
    return number


def normalized_video_score(dimension, detail):
    if dimension not in CORE:
        raise ValueError('Not one of the frozen nine dimensions')
    value = finite(detail['video_results'])
    return value / 100.0 if dimension == 'imaging_quality' else value


def verify_official_aggregate(dimension, result, frame_counts):
    """Reconstruct the original single-process aggregate from saved details.

    frame_counts maps each actually decoded video path to its frame count.
    The subject scorer omits weights in its JSON, so decoded counts matter.
    """
    details = result[-1]
    if not details or len({r['video_path'] for r in details}) != len(details):
        raise ValueError('Missing/duplicate underlying video scores')
    values = [normalized_video_score(dimension, r) for r in details]
    weights = [1] * len(details)
    if dimension in ('subject_consistency', 'background_consistency'):
        weights = [frame_counts[r['video_path']] - 1 for r in details]
        if any(n < 1 for n in weights):
            raise ValueError('A consistency score requires at least two actual frames')
    if dimension == 'background_consistency':
        for row, value, weight in zip(details, values, weights):
            if (row['cnt_per_video'] != weight or
                    not math.isclose(finite(row['video_sim']), value * weight,
                                     rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError('Background official frame-pair accounting differs')
    reconstructed = math.fsum(v * w for v, w in zip(values, weights)) / sum(weights)
    aggregate = finite(result[0])
    if not math.isclose(aggregate, reconstructed, rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError('Raw official aggregate disagrees with source/units/weights')
    return reconstructed


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def paired_sources(reference, candidate, *, draws=10000, seed=20261023):
    """Paired percentile bootstrap; one common source draw for both arms.

    Relative change is ratio of equal-source means, never mean of ratios.
    A zero denominator is undefined, including in bootstrap resamples.
    These are marginal descriptive intervals, not multiple-testing proof.
    """
    if not reference or set(reference) != set(candidate) or draws < 2:
        raise ValueError('Exactly aligned nonempty underlying sources required')
    keys = sorted(reference)
    base = [finite(reference[k]) for k in keys]
    new = [finite(candidate[k]) for k in keys]
    n = len(keys)
    base_mean, new_mean = statistics.fmean(base), statistics.fmean(new)
    rng = random.Random(seed)
    delta_draws, relative_draws = [], []
    undefined_relative = 0
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        b = math.fsum(base[i] for i in indices) / n
        c = math.fsum(new[i] for i in indices) / n
        delta_draws.append(c - b)
        if b == 0:
            undefined_relative += 1
        else:
            relative_draws.append(c / b - 1)
    interval = lambda v: [quantile(v, .025), quantile(v, .975)]
    relative_ci = None if undefined_relative else interval(relative_draws)
    return {'source_indices': keys, 'independent_source_count': n,
            'reference_equal_source_mean': base_mean, 'candidate_equal_source_mean': new_mean,
            'positive_reference_mean_for_relative_quality_interpretation': base_mean > 0,
            'absolute_mean_difference': new_mean - base_mean,
            'relative_change_of_means': None if base_mean == 0 else new_mean / base_mean - 1,
            'paired_source_differences': {str(k): c - b for k, b, c in zip(keys, base, new)},
            'bootstrap': {'draws': draws, 'seed': seed, 'unit': 'underlying_conditioning_image',
                          'absolute_difference_percentile95': interval(delta_draws),
                          'relative_change_percentile95': relative_ci,
                          'undefined_zero_denominator_resamples': undefined_relative,
                          'conditional_on_nonzero_relative_interval': False,
                          'multiple_comparisons_adjusted': False},
            'automatic_quality_acceptance': False}
