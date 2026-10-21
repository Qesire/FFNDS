"""CPU-only R4 arithmetic and counterexamples; never loads Wan or initializes CUDA."""
import argparse
import itertools
import json
from decimal import Decimal, localcontext
from pathlib import Path


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def plus(a, b):
    return [x + y for x, y in zip(a, b)]


def advantage(e, d):
    return dot(e, e) - dot(plus(e, d), plus(e, d))


def build_report(root):
    reports = root / 'reports' / 'worker_20261020'
    certificate = json.loads((reports / 'BO1_independent_imaging_certificate.json').read_text(encoding='utf-8'))
    cohort = next(x for x in certificate['cohorts'] if x['cohort'] == 'disjoint20')
    with localcontext() as ctx:
        ctx.prec = 50
        dense = Decimal(cohort['D_Q_mean_decimal'])
        best = Decimal(cohort['max_possible_imaging_mean_decimal'])
        required = Decimal(cohort['required_mean_decimal'])
        drop = 1 - best / dense
        gap = required - best
        assert best < required and abs(drop - Decimal(cohort['minimum_possible_relative_drop_decimal'])) < Decimal('1e-45')
        bound = {'best_drop_percent': str(100 * drop), 'score_gap_to_acceptance': str(gap), 'gap_in_dense_percentage_points': str(100 * gap / dense)}
    op = json.loads((reports / 'OP3_summary.json').read_text(encoding='utf-8'))['CV']
    nf = json.loads((reports / 'NF2_summary.json').read_text(encoding='utf-8'))['CV']
    assert len(op) == len(nf) == 8
    assert all(x['candidate_over_frozen'] > 1 and not x['gate_pass'] for x in op)
    assert all(x['nonlinear_over_linear'] > 1 and not x['gate_pass'] for x in nf)
    linear_gains = [100 * (1 - x['linear_native_SSE'] / x['frozen_UB1_native_SSE']) for x in nf]

    # Phi spans the first coordinate; current correction is not optimized.
    e, d = [1., 1.], [-1., 0.]
    frozen_gain = advantage(e, d)
    q = lambda v: [0., v[1]]
    refit_gain = advantage(q(e), q(d))
    assert frozen_gain == 1 and refit_gain == 0
    assert advantage(q(e), [0., -.5]) == .75

    # Training-only OLS adjustment: Phi_train=(1,2), d_train=(2,4).
    coefficient = dot([1., 2.], [2., 4.]) / dot([1., 2.], [1., 2.])
    validation_delta = 7. - 3. * coefficient
    assert validation_delta == 1 and advantage([2.], [validation_delta]) == -5

    # Both individual swaps lose, but their joint effect gains.
    e, d1, d2 = [1., 0.], [-.75, 1.], [-.75, -1.]
    individual = [advantage(e, d1), advantage(e, d2)]
    joint = advantage(e, plus(d1, d2))
    assert individual == [-.0625, -.0625] and joint == .75

    # Enumerate an illustrative fixed-real contribution neighborhood.
    e = [1., -.5]
    contributions = [[1., 0.], [0., 1.], [-.4, .3], [.2, -.7]]
    linear = [-2 * dot(e, c) for c in contributions]
    keep, omit = [0, 1], [2, 3]
    upper = max([0.] + [sum(sorted((linear[j] for j in omit), reverse=True)[:k]) + sum(sorted((-linear[j] for j in keep), reverse=True)[:k]) for k in [1, 2]])
    gains = []
    for k in range(3):
        for removed in itertools.combinations(keep, k):
            for added in itertools.combinations(omit, k):
                s = [int(j in added) - int(j in removed) for j in range(4)]
                delta = [sum(s[j] * contributions[j][i] for j in range(4)) for i in range(2)]
                actual = advantage(e, delta)
                relaxed = dot(linear, s)
                assert actual <= relaxed + 1e-12 and relaxed <= upper + 1e-12
                gains.append(actual)

    # E[e|h]=0, but action-value is exactly predictable: d=h*u.
    states = [{'h': h, 'u': u, 'e': u, 'd': h*u, 'A': -2*h-1} for h in [-1, 1] for u in [-1, 1]]
    for h in [-1, 1]:
        rows = [x for x in states if x['h'] == h]
        assert sum(x['e'] for x in rows) == 0
        for row in rows:
            assert row['A'] == advantage([row['e']], [row['d']])
    oracle = sum(max(x['A'], 0) for x in states) / 4
    static = max(0, sum(x['A'] for x in states) / 4)
    assert oracle - static == .5

    # Real-arithmetic folding of a linear head on standardized h.
    h, b, c, a, mu, sigma, t = 3., 2., .5, -.4, 1., 2., .2
    separate = b*h + c + a*(h-mu)/sigma + t
    folded = (b+a/sigma)*h + c+t-a*mu/sigma
    assert abs(separate-folded) < 1e-12
    result = {
        'status': 'PASS_CPU_ARITHMETIC_AND_SYNTHETIC_CHECKS_ONLY',
        'evidence_base_commit': '9227de1f3e391341ff000c15ad805d9ca2db3084',
        'scope': 'Recomputed saved historical scalars plus synthetic real-arithmetic examples; no new model, native BF16, GPU, quality or timing results.',
        'BO1_saved_certificate_arithmetic': bound,
        'OP3_saved_summary': {'entries': len(op), 'all_fail': True, 'candidate_over_frozen_range': [min(x['candidate_over_frozen'] for x in op), max(x['candidate_over_frozen'] for x in op)]},
        'NF2_saved_summary': {'entries': len(nf), 'all_nonlinear_lose_to_linear': True, 'linear_improvement_percent_range': [min(linear_gains), max(linear_gains)]},
        'compensation_redundancy_example': {'frozen_gain': frozen_gain, 'matched_refit_gain': refit_gain, 'orthogonal_change_gain': .75},
        'train_only_refit_validation_example': {'validation_delta': validation_delta, 'validation_gain': -5},
        'joint_swap_example': {'individual_gains': individual, 'joint_gain': joint},
        'linear_relaxation_example': {'maximum_enumerated_gain': max(gains), 'upper_bound': upper, 'candidate_count': len(gains)},
        'output_vs_action_prediction_example': {'states': states, 'conditional_mean_error': 0, 'dynamic_value_over_best_static': oracle-static},
        'linear_head_real_fold_abs_difference': abs(separate-folded)
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON output; without it, print to stdout only.')
    args = parser.parse_args()
    result = build_report(Path(__file__).resolve().parents[1])
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
        print(result['status'])
    else:
        print(text, end='')
