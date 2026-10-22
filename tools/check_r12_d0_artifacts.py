"""Recompute the completed D0 decision and audit its public evidence limits.

Stdlib-only validation of saved records, not a native model re-execution.
"""
import hashlib
import json
import math
from pathlib import Path

from r12d0_contract import ACTIONS, INDICES, METHOD, PROTOCOL, decide, groups, residual_count

ROOT = Path(__file__).resolve().parents[1]
BASE = 'research/20261022/'


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def validate(result, current, triage, historical):
    errors = []

    def need(value, message):
        if not value:
            errors.append(message)

    need(result['completed_generations'] == 24, 'D0 requires exactly24 completed generations')
    need(result['numeric_and_cost_controls_pass'] is True and result['repeated_N0_bitwise_equal'] is True,
         'Native numeric/cost/repeat controls must pass')
    need(result['evidence_level'] == 'FIXED_INPUT_NATIVE_FLOW_ONLY', 'Fixed-step evidence relabeled')
    need(result['free_student_quality_or_FFN_speed_acceptance'] is False
         and result['deployment_policy_fitted'] is False and result['new_training_updates'] == 0,
         'D0 cannot establish a trained router, free quality or speed')
    matrix = {int(i): v for i, v in result['action_matrix'].items()}
    ub1 = {int(i): v for i, v in result['frozen_UB1_losses'].items()}
    repeats = {int(i): v for i, v in result['N0_repeat_losses'].items()}
    expected = decide(matrix, ub1, repeats)
    need(result['decision'] == expected, 'Frozen pooled gates or oracle do not reproduce')
    rows = {(r['index'], r['arm']): r for r in result['rows']}
    arms = ('DQ', 'UB1', 'N0', 'N0_repeat', 'NA', 'NB')
    need(len(rows) == len(result['rows']) == 24 and set(rows) == {(i, a) for i in INDICES for a in arms},
         'Missing, duplicate or unregistered case')
    if errors:
        return errors
    previous = {r['index']: r for r in historical['rows']}
    full = offline = 0
    for (index, arm), row in rows.items():
        execution = row['execution']
        need(row['counter_delta'] == {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600,
                                     'weight_transfers': 0, 'cache_misses': 0}, 'Wrong model linear/QDQ coverage')
        need(execution['main_MAC'] == (717256182988800 if arm == 'DQ' else 358628091494400)
             and execution['dense_main_MAC'] == 717256182988800
             and execution['actual_main_linear_calls'] == 320
             and execution['actual_global_main_MAC_fraction'] == (1. if arm == 'DQ' else .5)
             and execution['actual_residual_positions'] == residual_count(arm)
             and execution['pending_residual_predictions'] == 0
             and execution['complete_FFN_timing_or_quality_acceptance'] is False,
             'Wrong actual main/residual execution budget')
        need(row['diagnostic_video_is_DQ_by_construction'] is True, 'Diagnostic video relabeled free student')
        loss = row['flow_SSE']
        need(math.isclose(ub1[index] - loss, row['signed_alignment_gain'] - row['direction_norm2_penalty'],
                          rel_tol=1e-9, abs_tol=1e-8), 'Signed downstream gain identity failed')
        if arm in ACTIONS:
            need(loss == matrix[index][arm], 'Action matrix and actual case disagree')
        elif arm == 'N0_repeat':
            need(loss == repeats[index] == matrix[index]['N0'], 'Repeated N0 loss differs')
            need(row['scopes'] == rows[index, 'N0']['scopes'] and row['steps'] == rows[index, 'N0']['steps'],
                 'Repeated N0 native bytes differ')
        elif arm == 'DQ':
            need(loss == 0., 'Original dense flow does not reproduce')
        else:
            need(loss == ub1[index] == previous[index]['flow_SSE']['none'], 'Original UB1 loss changed')
        need([r['step'] for r in row['steps']] == [0, 1, 2, 3], 'Wrong native four-step coverage')
        for step, value in enumerate(row['steps']):
            dq = rows[index, 'DQ']['steps'][step]
            control = rows[index, 'DQ' if arm == 'DQ' else 'UB1']['steps'][step]
            need(value['input_sha256'] == dq['input_sha256'], 'Original complete-step input changed')
            if step != 3 or arm in ('DQ', 'UB1'):
                need(all(value[k] == control[k] for k in ('flow_sha256', 'post_sha256')),
                     'Frozen native comparator flow/post differs')
        need(len(row['scopes']) == 2, 'Incomplete observed native pair')
        for block, scope in zip((30, 31), row['scopes']):
            need(scope['block'] == block and scope['groups'] == groups(arm, 3, block)
                 and scope['full_rows'] == 31668 and scope['native_gate_residual_bitwise_equal'] is True
                 and scope['native_RNG_unchanged'] is True, 'Pair/gate/RNG coverage differs')
            calls = scope['offline_teacher_counter_delta']
            expected_calls = 2 if scope['groups'] == 216 else 0
            need(calls['linear_forward_calls'] == calls['activation_qdq_calls'] == expected_calls,
                 'Offline original teacher work is uncharged')
            offline += calls['linear_forward_calls']
            if scope['groups'] == 216:
                full += 1
                need(scope['full216_original_teacher_bitwise_equal'] is True
                     and scope['original_Q1_Q2_bitwise_equal'] is True, 'Full216 identity failed')
        if arm in ACTIONS:
            need(all(row['scopes'][0][k] == rows[index, 'N0']['scopes'][0][k]
                     for k in ('norm_input_sha256', 'Q1_sha256', 'b0_sha256', 'b2_sha256', 'gate_sha256')),
                 'Pre-action block30 input differs')
    need(full == result['full216_original_teacher_identities'] == 16
         and offline == result['offline_teacher_linears'] == result['offline_teacher_QDQ_calls'] == 32,
         'Original full216 identity/extra work totals differ')
    status = ('COMPLETED_FIXED_R12_D0_VALUE_GATE_PASSED_NO_FOLLOWUP_AUTHORIZATION' if expected['diagnostic_gate_pass']
              else 'CLOSED_FIXED_R12_D0_RAW_VALUE_GATE_REJECTED')
    method = next(m for m in triage['methods'] if m['id'] == METHOD)
    need(current['eligible_model_method_ids'] == [] and method['model_execution_eligibility'] != 'ELIGIBLE_REGISTERED_SCREEN',
         'Completed D0 cannot auto-authorize another model job')
    need(current['closed_method_statuses'][METHOD] == method['status'] == status,
         'Current decision/triage result disposition differs')
    record = next(r for r in current['completed_diagnostics'] if r['method_id'] == METHOD)
    need(record['result'] == BASE + 'R12_D0_native_results.json'
         and record['result_sha256'] == sha(record['result'])
         and record['status'] == status and record['protocol_sha256'] == result['protocol_sha256'],
         'Completed evidence binding differs')
    return errors


def main():
    result = read(BASE + 'R12_D0_native_results.json')
    errors = validate(result, read('research/CURRENT_DECISION.json'), read('research/20261021/method_triage.json'),
                      read('research/20261021/R7D1_results.json'))
    if result['protocol_sha256'] != hashlib.sha256(PROTOCOL.read_bytes()).hexdigest():
        errors.append('Executed D0 protocol SHA changed')
    # The active driver checked every executed source before each native case.
    # Preserve those identities against the pre-execution archive, rather than
    # requiring shared utilities at future HEADs to stay frozen forever. Public
    # CI has neither the native installation nor the original raw payloads.
    archive = read(BASE + 'R12_D0_before_execution.json')['registration']
    if (result['source_sha256'] != archive['source_sha256']
            or result['protocol_sha256'] != archive['protocol_sha256']):
        errors.append('Executed source/protocol identities differ from the pre-execution archive')
    current = read('research/CURRENT_DECISION.json')
    review = current['latest_research_review']
    for key in ('budget_debit', 'resources'):
        if sha(review[key]) != review[key + '_sha256']:
            errors.append('Completed supporting evidence SHA changed: ' + key)
    debit = read(review['budget_debit'])
    if (debit['evidence_sha256']['D0'] != sha(BASE + 'R12_D0_native_results.json')
            or debit['evidence_sha256']['R7D1'] != sha('research/20261021/R7D1_results.json')
            or debit['source_sha256'] != sha('tools/analyze_r12_budget_debit.py')):
        errors.append('Cross-round accounting source/input SHA changed')
    resource = read(review['resources'])
    if (resource['new_model_generations'] != 24 or resource['GPU_count_max'] != 2
            or resource['selected_GPU_indices'] != [0, 1]
            or resource['combined_new_HF_bytes'] != resource['worker_experiment_HF_bytes'] + resource['coordinator_HF_receipt_mirror_bytes']
            or resource['worker_experiment_HF_bytes'] != sum(r['bytes'] for r in resource['worker_HF_files'])
            or resource['combined_new_HF_bytes'] > 1 << 29
            or resource['worker_min_recorded_free_bytes'] < 80 << 30
            or resource['mirrored_cases'] != 24 or resource['mirrored_receipt_and_flow_files'] != 48):
        errors.append('Completed generation/GPU/HF/reserve resource ledger differs')
    information = read(BASE + 'R12_information_budget_discriminators.json')
    if information['source_sha256'] != sha('tools/analyze_r12_information_budget.py'):
        errors.append('Executed information/budget discriminator source SHA changed')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Completed24-case D0 native records, pooled gates, costs and closed eligibility: CONSISTENT')
    print('Saved-evidence audit only; free student quality, deployable routing and FFN speed remain unverified.')


if __name__ == '__main__':
    main()
