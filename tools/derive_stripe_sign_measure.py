"""A finite cubature-law deduction, not a change to registered native signs.

Enumerates tiny synthetic sign sets in stdlib only. It demonstrates both a
quadratic variance improvement and its failure for a discontinuous response.
"""
import argparse
import itertools
import json
import math
from pathlib import Path


def derive():
    # Four centered, rank-three token rows; empirical covariance is identity.
    x = ((1., 1., 1.), (1., -1., -1.), (-1., 1., -1.), (-1., -1., 1.))
    signs = list(itertools.product((-1., 1.), repeat=4))
    contrast = lambda s: [math.fsum(s[i] * x[i][j] for i in range(4)) / 2 for j in range(3)]
    original = [contrast(s) for s in signs]
    retained = [contrast(s) for s in signs if len(set(s)) > 1]
    factor = math.sqrt(7 / 8)
    conditioned = [[factor * value for value in c] for c in retained]
    covariance = lambda cs: [[math.fsum(c[i] * c[j] for c in cs) / len(cs) for j in range(3)] for i in range(3)]
    target = [[float(i == j) for j in range(3)] for i in range(3)]
    old_cov, new_cov = covariance(original), covariance(conditioned)
    error = max(abs(c[i][j] - target[i][j]) for c in (old_cov, new_cov) for i in range(3) for j in range(3))
    quadratic = lambda c: c[0] ** 2 + 2 * c[1] ** 2 - .5 * c[2] ** 2 + c[0] * c[1]
    moments = lambda values: (math.fsum(values) / len(values), math.fsum(v * v for v in values) / len(values))
    mean, second = moments([quadratic(c) for c in original])
    new_mean, new_second = moments([quadratic(c) for c in conditioned])
    old_variance, new_variance = second - mean * mean, new_second - new_mean * new_mean
    predicted_drop = second / 8
    # Even radial threshold: the antithetic mean is this same discontinuous f(c).
    # It stands for a counterexample, not the actual HiF4 group quantizer.
    discontinuous = lambda c: float(math.fsum(value * value for value in c) > 2.8)
    true_mean = math.fsum(discontinuous(row) for row in x) / 4
    mse = lambda cs: math.fsum((discontinuous(c) - true_mean) ** 2 for c in cs) / len(cs)
    original_mse, conditioned_mse = mse(original), mse(conditioned)
    assert error < 1e-14 and abs(mean - new_mean) < 1e-14
    assert abs(old_variance - new_variance - predicted_drop) < 1e-13
    assert conditioned_mse > original_mse
    return {'id': 'R14_SIGN_MEASURE_QUADRATIC_VARIANCE_AND_DISCONTINUITY',
            'status': 'EXACT_FINITE_MATH_AND_SYNTHETIC_CPU_ONLY_UNREGISTERED',
            'hypothesis': 'Independent4signs spend1/8 probability on all-identical signs, giving zero contrast for centered rows. Excluding those2signs and scaling the retained contrast bysqrt(7/8) keeps covariance but changes the cubature law.',
            'theorem': 'For any centered four-row inputs, E_old(ccT)=empirical_covariance=E_new(ccT). For any homogeneous quadraticq, E_newq=E_oldq and Var_oldq-Var_newq=E_old(q^2)/8. The two excluded events haveq=0; conditioning and quadratic scaling prove the identities.',
            'original_signs': len(original), 'conditioned_signs': len(conditioned),
            'synthetic_covariance_max_error': error,
            'quadratic': {'original_mean': mean, 'conditioned_mean': new_mean,
                          'original_variance': old_variance, 'conditioned_variance': new_variance,
                          'predicted_variance_drop': predicted_drop},
            'discontinuity_counterexample': {'function': 'indicator(norm2>2.8)',
                'true_four_row_mean': true_mean, 'original_two_point_MSE': original_mse,
                'conditioned_two_point_MSE': conditioned_mse,
                'interpretation': 'Quadratic variance reduction is not a universal improvement for discontinuous responses. This is not native HiF4 data or a Wan effect.'},
            'registered_scope_unchanged': True, 'new_model_generations': 0, 'new_native_FFN_calls': 0,
            'native_registration': None,
            'next_question': 'If registered original signs show operator-limited value, separate curvature/quantizer bias from sampler variance. This deduction does not permit changing signs or automatically rerunning the fixed native instance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the first deduction receipt')
    args.output.write_text(json.dumps(derive(), indent=2) + '\n')


if __name__ == '__main__':
    main()
