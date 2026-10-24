"""Saved-receipt comparisons and constructive CPU checks; no model dependencies."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
OUT = 'research/20261024/potential_directions_CPU.json'
INPUTS = [
    'research/20261023/C1N0_dense_value_robustness.json',
    'research/20261023/PRD_successor_reanalysis.json',
    'research/20261024/C1_boundary_target_audit.json',
    'research/20261023/C2N0_execution_summary.json',
    'research/20261023/C2N1_execution_summary.json',
    'tools/check_c1_dense_value_robustness.py',
    'tools/c1_native_probe.py',
    'tools/ub1_row_dispatch.py',
    'research/OBJECTIVE_CONTRACT.json',
    'docs/68_R12_D0等预算原生动作检验.md',
]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (float, int)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10)
    return type(a) is type(b) and a == b


def bf16(value):
    """Round a finite synthetic scalar through FP32 to BF16, ties to even."""
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def toy_q(values):
    """Deliberately width-dependent toy QDQ, NOT HiF4."""
    scale = max(max(abs(v) for v in values) / 7, 1e-10)
    return [bf16(round(v / scale) * scale) for v in values]


def dot(a, b):
    return math.fsum(x*y for x, y in zip(a, b))


def projection_cost(b, q):
    return b + q*(1-b)


def staged_fixture(promoted=(1, 4)):
    """GELU FFN: base hidden prefix reused before either W2 branch.

    Both paths use explicit scalar arithmetic, synthetic BF16 casts and toy QDQ.
    This checks dependency/order/accounting, not native GEMM bit equivalence.
    """
    rng = random.Random(20261024)
    m, d, h = 6, 8, 16
    base = list(range(4)) + list(range(8, 12))  # two nonadjacent whole groups
    complement = [j for j in range(h) if j not in base]
    x = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(m)]
    w1 = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(h)]
    w2 = [[rng.uniform(-1, 1) for _ in range(h)] for _ in range(d)]
    b1 = [rng.uniform(-.2, .2) for _ in range(h)]
    dense_bias = [rng.uniform(-.1, .1) for _ in range(d)]
    folded_bias = [rng.uniform(-.2, .2) for _ in range(d)]
    correction = [[rng.uniform(-.3, .3) for _ in range(d)] for _ in range(m)]
    alpha = 1.375
    promoted = set(promoted)
    if not promoted <= set(range(m)):
        raise ValueError('Invalid promoted rows')

    def hidden(row, ids):
        out = []
        for j in ids:
            u = bf16(dot(row, w1[j]) + b1[j])
            out.append(bf16(.5*u*(1 + math.erf(u/math.sqrt(2)))))
        return out

    def finish(v, ids, row, dense):
        v = toy_q(v)
        if not dense:
            v = [bf16(alpha*z) for z in v]
        bias = dense_bias if dense else folded_bias
        y = [bf16(dot(v, [ww[j] for j in ids]) + bias[k]) for k, ww in enumerate(w2)]
        if not dense:
            y = [bf16(z+c) for z, c in zip(y, correction[row])]
        # Gate and identity path are present in both comparator and staged path.
        return [bf16(a + .75*b) for a, b in zip(x[row], y)]

    max_error = 0.
    computed_hidden = 0
    w2_mac = 0
    for i, row in enumerate(x):
        dense = i in promoted
        ids = list(range(h)) if dense else base
        reference = finish(hidden(row, ids), ids, i, dense)
        partial = hidden(row, base)
        computed_hidden += len(base)
        if dense:
            remaining = hidden(row, complement)
            computed_hidden += len(complement)
            joined = dict(zip(base, partial)) | dict(zip(complement, remaining))
            # Assemble all hidden coordinates before the original-width Q2.
            partial = [joined[j] for j in range(h)]
        staged = finish(partial, ids, i, dense)
        max_error = max(max_error, max(abs(a-b) for a, b in zip(reference, staged)))
        w2_mac += len(ids)*d
    mac = computed_hidden*d + w2_mac
    return {'toy_rows': m, 'toy_D': d, 'toy_H': h, 'promoted_rows': len(promoted),
            'base_hidden_computations': m*len(base),
            'complement_hidden_computations': len(promoted)*len(complement),
            'projection_MAC': mac, 'dense_projection_MAC': 2*m*d*h,
            'projection_fraction': mac/(2*m*d*h),
            'expected_projection_fraction': projection_cost(.5, len(promoted)/m),
            'max_output_error_in_toy': max_error,
            'evidence': 'SYNTHETIC_BF16_TOY_QDQ_NOT_NATIVE_HIF4'}


def log_target_counterexample():
    # A has greater arithmetic expected loss, but a smaller geometric mean.
    a = [.1, 10.]
    b = [2., 2.]
    arithmetic = [math.fsum(v)/len(v) for v in (a, b)]
    geometric = [math.exp(math.fsum(math.log(x) for x in v)/len(v)) for v in (a, b)]
    return {'A_outcomes_equal_probability': a, 'B_outcomes_equal_probability': b,
            'arithmetic_means_A_B': arithmetic, 'geometric_means_A_B': geometric,
            'log_target_selects': 'B', 'expected_loss_target_selects': 'A',
            'expected_capture_regret': arithmetic[0]-arithmetic[1]}


def signed_advantage_fixture():
    e, delta = [1., 0.], [3., 0.]
    direct = dot(e, e)-dot([a-b for a, b in zip(e, delta)], [a-b for a, b in zip(e, delta)])
    # Two actions can cancel after a shared linear downstream map.
    ea, da, db = [0.], [1.], [-1.]
    single_a = dot(ea, ea)-dot(da, da)
    single_b = dot(ea, ea)-dot(db, db)
    both = dot(ea, ea)-dot([da[0]+db[0]], [da[0]+db[0]])
    return {'baseline_error': e, 'action_response': delta,
            'action_energy': dot(delta, delta), 'signed_advantage': direct,
            'identity_value': 2*dot(e, delta)-dot(delta, delta),
            'interacting_actions_sum_individual_advantages': single_a+single_b,
            'interacting_actions_joint_advantage': both}


def folded_head_fixture():
    rng = random.Random(47)
    h = [rng.uniform(-1, 1) for _ in range(256)]
    features = h + [v*v for v in h] + [dot(h, h)]
    weights = [rng.uniform(-1, 1) for _ in features]
    means = [rng.uniform(-1, 1) for _ in features]
    scales = [rng.uniform(.2, 2.) for _ in features]
    intercept = .17
    original = intercept + math.fsum(w*(v-m)/s for w, v, m, s in zip(weights, features, means, scales))
    coef = [w/s for w, s in zip(weights, scales)]
    bias = intercept - dot(coef, means)
    folded = bias + dot(coef[:256], h) + math.fsum((coef[256+j]+coef[-1])*v*v for j, v in enumerate(h))
    return {'absolute_error': abs(original-folded), 'h_dimensions': 256,
            'h_only_polynomial_coefficients': 512,
            'interpretation': 'Exact real-arithmetic folding; V1 statistics, structural intercepts and optional pre-features remain separately charged.'}


def no_completion_fixture():
    """Known contribution vectors prove existence, not pre-W1 predictability."""
    contexts = [[10., -10., 1., 1.], [1., 1., 10., -10.]]
    supports = list(itertools.combinations(range(4), 2))
    def loss(v, support):
        return (math.fsum(v) - math.fsum(v[j] for j in support))**2
    dynamic = [min(loss(v, s) for s in supports) for v in contexts]
    fixed = min(math.fsum(loss(v, s) for v in contexts)/2 for s in supports)
    magnitude = [loss(v, sorted(range(4), key=lambda j: -abs(v[j]))[:2]) for v in contexts]
    # Every single exchange loses, while a simultaneous two-group exchange wins.
    trap = [10., -10., 1., 0.]
    base = {0, 1}
    neighbors = [(base-{i}) | {j} for i in base for j in (2, 3)]
    return {'evidence': 'SYNTHETIC_EXISTENCE_AND_NEIGHBORHOOD_COUNTEREXAMPLE',
            'two_context_group_contributions': contexts, 'retained_fraction': .5,
            'dense_completion_rows': 0, 'additive_correction_branches': 0,
            'dynamic_oracle_errors': dynamic, 'best_fixed_mean_error': fixed,
            'independent_magnitude_errors': magnitude,
            'single_exchange_trap_contributions': trap,
            'base_error': loss(trap, base),
            'all_single_exchange_errors': [loss(trap, s) for s in neighbors],
            'joint_exchange_error': loss(trap, {2, 3}),
            'online_information': 'Unknown; actual omitted group vectors are offline labels only.',
            'scope': 'Real-arithmetic additive group model; native QDQ/gates and complete residual propagation require their own construction.'}


def build():
    rich = read(INPUTS[0])['results']
    old = read(INPUTS[1])['saved_native_reanalysis']
    c2 = old['C2_pooled']['scaled_capture']
    r = rich['rich_current']['fraction_of_all_sampled_measured_nextnorm_loss']
    rp = rich['rich_current_plus_pre']['fraction_of_all_sampled_measured_nextnorm_loss']
    per_source = {}
    for source, c in old['C2_grouped']['source'].items():
        value = rich['rich_current']['per_source_captured_loss'][source]/c['sampled_loss']
        per_source[source] = {'rich_current_capture': value, 'C2_capture': c['scaled_capture'],
                              'gain_pp': 100*(value-c['scaled_capture'])}
    rstd = [c['gain_pp'] for c in per_source.values()]
    boundary = read(INPUTS[2])['pooled']
    return {'id': 'POTENTIAL_DIRECTIONS_20261024',
            'evidence_base_commit': '0033725882c4db2e9fbd6f9ee0c85e20950fbe62',
            'input_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS},
            'saved_receipt_comparison': {
                'sources': 6, 'scope': 'same 72 sites, top84 of 336 sampled rows; exposed-source held-out folds',
                'rich_current_capture': r, 'rich_plus_pre_capture': rp, 'C2_capture': c2,
                'sample_label_oracle_capture': old['C2_pooled']['oracle_capture'],
                'rich_current_minus_C2_pp': 100*(r-c2), 'rich_plus_pre_minus_C2_pp': 100*(rp-c2),
                'extra_pre_over_rich_current_pp': 100*(rp-r),
                'equal_source_macro_rich_minus_C2_pp': math.fsum(rstd)/len(rstd),
                'positive_source_count': sum(v>0 for v in rstd), 'per_source': per_source},
            'latest_saved_full_row_boundary_audit': {
                'scope': '72 sites, top7917 of 31668 actual saved scalar row losses; a different selection population from the readout comparison',
                'mean_overlap': boundary['mean_top_quarter_overlap_fraction'],
                'mean_spearman': boundary['mean_post_nextnorm_Spearman']},
            'closed_attempt_completed_videos': {name: read(f'research/20261023/{name}_execution_summary.json')['completed_videos'] for name in ('C2N0', 'C2N1')},
            'log_target_counterexample': log_target_counterexample(),
            'signed_finite_value_counterexample': signed_advantage_fixture(),
            'no_completion_counterexample': no_completion_fixture(),
            'folded_h256_head': folded_head_fixture(),
            'shared_prefix_fixtures': [staged_fixture(()), staged_fixture(), staged_fixture(range(6))],
            'analytic_cost_at_b_half_q_quarter': {
                'original_or_shared_prefix_main_projection_fraction': projection_cost(.5, .25),
                'finish_UB1_then_recompute_promoted_base_W2_fraction': .5+.25*(1-.5/2),
                'R256_all_rows_fraction': 256/13824,
                'R256_stopped_rows_only_fraction': .75*256/13824,
                'main_MAC_per_site_by_step': [projection_cost(b, .25) for b in (.25, .5, .5, .75)],
                'scope': 'Projection MACs only; selector, QDQ, GELU, packing and launches remain measured costs.'},
            'new_model_calls': 0, 'new_C1_feature_refits': 0,
            'evidence': 'SAVED_AGGREGATE_REANALYSIS_AND_SYNTHETIC_CONSTRUCTION',
            'new_native_quality_or_speed_result': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = build()
    if args.check:
        if not equivalent(result, read(OUT)):
            raise SystemExit('Saved potential-direction result differs')
        print('Potential-direction saved receipts, target counterexamples and staged computation: PASS')
    else:
        (ROOT/OUT).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(OUT)
