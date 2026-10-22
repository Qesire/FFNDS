"""Derive finite local sharing limits from already saved R13 scalars.

The exact common-output bound concerns these sampled pairs only. The affine
stripe figures concern an ideal real-arithmetic surrogate and a reference
sampling design, not an executed/native stripe or a video-quality bound.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_output_summary(rows):
    baseline = math.fsum(r['hidden_pruning_pair_SSE'] for r in rows)
    lower = math.fsum(r['row_sharing_pair_SSE'] / 2 for r in rows)
    oracle = math.fsum(min(r['hidden_pruning_pair_SSE'], r['row_sharing_pair_SSE'] / 2) for r in rows)
    return {'pairs': len(rows), 'UB1_pair_SSE': baseline,
            'arbitrary_common_output_lower_SSE': lower,
            'lower_bound_over_UB1': lower / baseline,
            'pairs_with_lower_bound_below_UB1': sum(r['row_sharing_pair_SSE'] / 2 < r['hidden_pruning_pair_SSE'] for r in rows),
            'ideal_pairwise_choice_upper_gain_over_UB1': 1 - oracle / baseline}


def affine_summary(scopes):
    baseline = math.fsum(r['reference_UB1_SSE'] for r in scopes)
    anchor = math.fsum(r['reference_random_anchor_FP64_SSE'] for r in scopes)
    # E_anchor(sum_i ||R_i-R_anchor||^2) = 2 sum_i ||R_i-mean(R)||^2.
    lower = anchor / 2
    leave_one_out = []
    for row in scopes:
        old = baseline - row['reference_UB1_SSE']
        leave_one_out.append((lower - row['reference_random_anchor_FP64_SSE'] / 2) / old)
    return {'sources': len(scopes), 'observed_undirected_stripe_pairs': sum(r['stripe_pairs'] for r in scopes),
            'reference_UB1_BF16_gate_SSE': baseline,
            'ideal_common_remainder_reference_SSE': lower,
            'ideal_common_remainder_over_UB1': lower / baseline,
            'leave_one_source_out_ratio_range': [min(leave_one_out), max(leave_one_out)],
            'scope': 'Continuous affine-remainder surrogate on the saved trajectories, with reference pair-inclusion weighting; not a bound for rounded native reconstruction, and no complete four-row stripe was observed.'}


def build(axis, affine):
    return {
        'id': 'R13S2_FINITE_SPATIAL_SHARING_LIMITS',
        'evidence_level': 'EXACT_SAVED_PAIR_BOUND_AND_IDEAL_AFFINE_STRIPE_REFERENCE_ONLY',
        'common_output_pair_identity': 'For original gated targets A,B and any common output C, ||A-C||^2+||B-C||^2=0.5||A-B||^2+2||C-(A+B)/2||^2. The continuous minimizer need not be deployable or BF16-representable.',
        'common_output': [{'step': step, **common_output_summary([r for r in axis['rows'] if step == 'all' or r['step'] == step])} for step in (1, 2, 'all')],
        'affine_remainder_reference': [
            {'step': step, 'cohort': cohort, **affine_summary([r for r in affine['scopes'] if r['step'] == step and
                (cohort == 'all_exposed24' or r['R9_fit_source'] == (cohort == 'original_R9_fit_odd'))])}
            for step in range(4) for cohort in ('all_exposed24', 'original_R9_fit_odd', 'nonfit_even_development')],
        'rounding_clarification': 'The earlier off-diagonal BF16 stripe diagnostic corresponds to bypassing anchor reconstruction and writing the already paid exact native anchor output. It omits any self-reconstruction error from L_a+BF16(Y_a-L_a). It is not evidence of exact anchor reconstruction. The present mean-remainder identity uses FP64 differences only.',
        'interpretation': 'The simple common-output family has very little available local pair value here. In contrast, the affine-plus-ideal-mean-remainder reference has room at step0/2, while the one-anchor estimate mixes spatial contrast loss with anchor sampling error. This motivates examining a different mean-response operator, without making its missing mean observable or affordable by assumption.',
        'new_model_generations': 0, 'fitted_parameters': 0,
        'model_execution_eligibility_change': False,
        'limits': 'These are fixed exposed sources and sampled coordinates, not strongest fixed spatial policies, arbitrary merge/unmerge maps, downstream flow, free video, or speed. The 0.236% finite pair bound is unrelated to and does not bound the historical R12 output-splice number.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the first sharing-limit analysis')
    paths = {'axis_result': ROOT / 'research/20261022/R13_saved_compression_axis_value.json',
             'affine_result': ROOT / 'research/20261022/R13_anchored_affine_remainder.json'}
    result = build(*(json.loads(p.read_text()) for p in paths.values()))
    result['source_sha256'] = digest(__file__)
    result['input_bindings'] = {k: {'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for k, p in paths.items()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['common_output'][-1]))


if __name__ == '__main__':
    main()
