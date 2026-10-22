"""Account for D0's native budget debit using completed, matched R7 traces.

This reads compact evidence only. The historical restorations have more main
MAC than D0; they supply a diagnostic credit/debit decomposition, not another
matched control or a deployable policy. No new model operation is performed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D0 = 'research/20261022/R12_D0_native_results.json'
R7 = 'research/20261021/R7D1_results.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(d0, r7):
    if (d0['completed_generations'] != 24 or not d0['numeric_and_cost_controls_pass']
            or not d0['repeated_N0_bitwise_equal']):
        raise ValueError('Requires all completed D0 cases and controls')
    current = {(r['index'], r['arm']): r for r in d0['rows']}
    old = {r['index']: r for r in r7['rows']}
    rows = []
    for index in (0, 5, 10, 15):
        incumbent = current[index, 'UB1']
        previous = old[index]
        if incumbent['flow_SSE'] != previous['flow_SSE']['none']:
            raise ValueError('Cross-round frozen UB1 loss differs')
        for arm, old_arm, restored_block in [('NA', 'A', 30), ('NB', 'B', 31)]:
            action = current[index, arm]
            past = previous['scopes'][old_arm]
            # Both rounds share the exact pre-W1 block30 state. A restoration
            # also gives the same block31 state, before its debit is applied.
            keys = ['norm_input_sha256', 'b0_sha256', 'b2_sha256', 'gate_sha256']
            if any(action['scopes'][0][k] != past[0][k] for k in keys):
                raise ValueError('Cross-round incoming state/bias/gate changed')
            if arm == 'NA':
                keys += ['returned_Y_sha256']
                if any(action['scopes'][0][k] != past[0][k] for k in keys):
                    raise ValueError('Original block30 full restoration differs')
                if action['scopes'][1]['norm_input_sha256'] != past[1]['norm_input_sha256']:
                    raise ValueError('Block31 debit is not evaluated on the identical incoming state')
            restored = action['scopes'][restored_block - 30]
            if (not restored['full216_original_teacher_bitwise_equal']
                    or restored['teacher_shard_sha256'] != past[restored_block - 30]['teacher_shard_sha256']):
                raise ValueError('Original full-width teacher operator changed')
            base, historical, final = incumbent['flow_SSE'], previous['flow_SSE'][old_arm], action['flow_SSE']
            credit, debit = base - historical, final - historical
            if not math.isclose(base - final, credit - debit, rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError('Credit minus debit identity failed')
            rows.append({'index': index, 'action': arm,
                'historical_restoration_credit': credit,
                'budget_and_calibration_bundle_debit': debit,
                'net_equal_budget_gain_over_UB1': base - final,
                'debit_exceeds_credit': debit > credit,
                'block30_incoming_state_bias_gate_identical': True,
                'block31_input_identical_to_historical_restoration':
                    action['scopes'][1]['norm_input_sha256'] == past[1]['norm_input_sha256'],
                'restored_layer_uses_original_teacher': True})
    pooled = {}
    for arm in ('NA', 'NB'):
        selected = [r for r in rows if r['action'] == arm]
        pooled[arm] = {key: math.fsum(r[key] for r in selected) for key in (
            'historical_restoration_credit', 'budget_and_calibration_bundle_debit', 'net_equal_budget_gain_over_UB1')}
    bundle = {str(i): current[i, 'N0']['flow_SSE'] - current[i, 'UB1']['flow_SSE'] for i in old}
    return {'id': 'R12_D0_CROSS_ROUND_BUDGET_DEBIT',
        'status': 'COMPLETED_COMPACT_EVIDENCE_ACCOUNTING_NO_MODEL_RUN',
        'evidence_level': 'FIXED_INPUT_NATIVE_FLOW_DIAGNOSTIC_POSTHOC',
        'source_sha256': sha(Path(__file__)), 'new_model_generations': 0,
        'identity': 'L_UB1 - L_D0 = (L_UB1 - L_R7_restore) - (L_D0 - L_R7_restore)',
        'historical_restore_pair_groups': 378, 'D0_pair_groups': 324,
        'historical_restore_replacement_only_main_MAC_fraction': 17334 / 34560,
        'historical_R7_measured_main_MAC_fraction_including_offline_teachers': .5 + 2 / 160,
        'historical_cost_scope': '378groups is the effective A/B replacement operator. The actual R7 diagnostic first ran the UB1 student plus both full offline teachers; its main work was51.25% dense, before auxiliary work. Neither figure is a matched50% deployment.',
        'D0_global_main_MAC_fraction': .5,
        'rows': rows, 'pooled': pooled,
        'N0_calibration_bundle_removal_penalty': bundle,
        'pooled_N0_calibration_bundle_removal_penalty': math.fsum(bundle.values()),
        'causal_scope': {
            'NA': 'Identical full block30 output and block31 input; debit jointly removes54 block31 groups and its fitted bias/gain/R256 bundle.',
            'NB': 'Identical block30 input; debit jointly shrinks block30 to108 raw groups and removes its fitted bundle. Original full block31 consumes the genuinely changed state.',
            'N0': 'Both pair positions retain162 groups while replacing the fitted bias/gain/R256 bundle by raw biases/alpha1/zero correction.'},
        'limitations': 'Posthoc decomposition on four exposed fixed-step sources. Debit is not width alone, R7 is not equal-budget, and no new policy, free-running quality or speed is established.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {'D0': ROOT / D0, 'R7D1': ROOT / R7}
    result = analyze(*(json.loads(paths[key].read_text()) for key in ('D0', 'R7D1')))
    result['evidence_sha256'] = {key: sha(path) for key, path in paths.items()}
    if args.output.exists():
        raise RuntimeError('Preserve the previous accounting result')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['pooled'], indent=2))


if __name__ == '__main__':
    main()
