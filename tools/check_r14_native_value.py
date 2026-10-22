"""Reproduce the completed R14N0 local-value ledger using public scalars only."""
import hashlib
import json
import math
from pathlib import Path

from analyze_stripe_native_values import analyze
from r14_stripe_contract import CALLS, CAPS, METHOD, ORDER
from derive_stripe_sign_measure import derive
from analyze_stripe_common_reference import analyze_reference

ROOT = Path(__file__).resolve().parents[1]
BASE = 'research/20261022/'
CLOSED = 'CLOSED_FIXED_NATIVE_STRIPE_RESPONSE_AND_NORM_VALUE_REJECTED'


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(result, capture, budget, resources):
    errors = []

    def need(condition, message):
        if not condition:
            errors.append(message)

    need(result['status'] == 'COMPLETE_FIXED_NATIVE_STRIPE_VALUE'
         and result['evidence_level'] == 'ACTUAL_NATIVE_COMPLETE_STRIPE_VALUES', 'Native local evidence scope changed')
    need(result['method_id'] == capture['id'] == budget['method_id'] == METHOD, 'Native method identity changed')
    need(result['native_video_or_speed_acceptance'] is False
         and result['new_candidate_rollouts'] == result['new_fitted_parameters'] == 0,
         'Local writeback cannot claim free-video/model-speed acceptance')
    if len(result['cases']) != 4 or [case['index'] for case in result['cases']] != list(ORDER):
        errors.append('Exactly four original source action records required')
        return errors
    recomputed = analyze(result['cases'])
    need(recomputed == result['analysis'], 'Native action/fixed-mask analysis does not reproduce')
    need(capture['status'] == 'PASS_FOUR_ORIGINAL_UB1_OBSERVATIONS'
         and [case['index'] for case in capture['completed']] == list(ORDER), 'Original capture coverage changed')
    need(budget['used'] == budget['caps'] == CAPS and budget == result['budget'], 'Fixed native resource budget changed')
    debits = budget['debits']
    need(len(debits) == 36 and len({(d['index'], d['label']) for d in debits}) == 36,
         'Missing, repeated or expanded native call')
    for index in ORDER:
        calls = [d for d in debits if d['index'] == index]
        need({d['label'] for d in calls} == {'capture', *CALLS}, 'Native call family changed')
        for call in calls:
            need(call['status'] == 'COMPLETED', 'Failed or merely queued call relabelled complete')
            if call['label'] != 'capture':
                need(call['actual_linear_calls'] == len(call['actual_linear_shapes']) == CALLS[call['label']],
                     'Actual linear shapes/counts omitted')
    for key in CAPS:
        need(sum(d['charge'].get(key, 0) for d in debits) == CAPS[key], 'Pre-debit total differs from actual fixed scope')
    for case in capture['completed']:
        scope = case['scope']
        need(case['original_trajectory_bitwise'] and case['original_video_bitwise']
             and [row['step'] for row in case['trajectory']] == list(range(4)), 'Original latent/video identity missing')
        need(scope['actual_native_writeback_bitwise_control'] and all(scope['old_operand_controls'].values())
             and scope['Q1_teacher_equals_actual'] and scope['actual_native_BaseY_plus_old_residual_equals_StudentY'],
             'Original target/bias/gate/residual control failed')
        need(scope['extra_full_teacher_GEMMs'] == 2, 'Original teacher work omitted from capture')
        need(scope['grid'] == [21, 29, 52] and scope['rows'] == 256 and len(scope['selected_rows']) == 256,
             'Real complete-stripe geometry changed')
        need(case['counter_delta']['linear_forward_calls'] == case['counter_delta']['activation_qdq_calls'] == 1600
             and case['counter_delta']['cache_misses'] == 0, 'Original native model work omitted')
    need(len(result['actual_execution']) == 4 and [r['index'] for r in result['actual_execution']] == list(ORDER),
         'Final physical action coverage changed')
    for row, analysis in zip(result['actual_execution'], recomputed['per_source']):
        need(len(row['Q2_controls']) == 6 and all(c['original_Q2_bit_mismatches'] == 0
             and c['shape'] == [11875, 13824] for c in row['Q2_controls']), 'Actual native Q2 control failed')
        need(len(row['auxiliary_controls']) == 6 and all(all(c[k] for k in
             ('point_pack_exact', 'postQ2_pool_exact', 'epilogue_control_pass')) for c in row['auxiliary_controls']),
             'Native auxiliary arithmetic changed')
        need(row['norm_max_relative_error'] <= 5e-6 and row['epilogue'] == 'BF16_FUSED_NATIVE_WRITEBACK',
             'Native signal/epilogue boundary changed')
        for arm in ('fixed', 'dynamic', 'oracle'):
            need(all(row['actual_composition_controls'][arm].values()), 'Output splice relabelled executed action')
            need(math.isclose(row['actual_native_SSE'][arm], analysis[arm + '_SSE'], rel_tol=1e-12, abs_tol=1e-8)
                 and math.fsum(row['primary_group_losses'][arm]) == row['actual_native_SSE'][arm],
                 'Actual action loss differs from independently measured local values')
        need(all(row['repeat_bitwise_control'].values())
             and row['repeat_dynamic_SSE'] == row['actual_native_SSE']['dynamic'], 'Native repeat differs')
    eta = math.fsum(abs(r['repeat_dynamic_SSE'] - r['actual_native_SSE']['dynamic'])
                    for r in result['actual_execution'] if r['index'] in (0, 10))
    total = recomputed['evaluation_totals']
    support = total['dynamic_SSE'] < min(total['fixed_SSE'], total['UB1_SSE']) - 3 * eta
    need(eta == result['eta_absolute_evaluation_SSE'] and result['finite_local_dynamic_support'] == support,
         'Predeclared finite value decision changed')
    need(resources['actual_model_generations'] == 4 and resources['actual_standalone_FFN_forwards'] == 32
         and resources['extra_linear_calls_including_capture_teacher'] == 104
         and resources['standalone_actual_linear_calls'] == 96
         and resources['max_GPUs'] == 1 and resources['new_candidate_rollouts'] == resources['new_fitted_parameters'] == 0,
         'Completed model/FFN/linear resource accounting changed')
    need(resources['native_model_speed_or_video_quality_acceptance'] is False,
         'Local resource record cannot claim video/model speed')
    need(result['max_allocated_bytes'] == resources['standalone_max_allocated_bytes'] <= 32 << 30
         and resources['incremental_HF_output_bytes'] <= 1 << 29
         and resources['free_node_cache_bytes_after'] >= 80 << 30, 'VRAM/HF/node reserve cap exceeded')
    return errors


def main():
    result, capture, budget, resources = [read(BASE + name) for name in (
        'R14N0_native_results.json', 'R14N0_capture_receipt.json', 'R14N0_budget_debit.json', 'R14N0_completed_resources.json')]
    errors = validate(result, capture, budget, resources)
    protocol = read(BASE + 'R14N0_registered_protocol.json')
    archive = read(BASE + 'R14N0_before_execution.json')['registration']
    if (result['protocol_sha256'] != sha(ROOT / (BASE + 'R14N0_registered_protocol.json'))
            or result['protocol_sha256'] != archive['protocol_sha256']
            or result['source_sha256'] != protocol['source_sha256']
            or result['source_sha256'] != archive['source_sha256']):
        errors.append('Executed source/protocol/pre-execution bindings changed')
    for field in ('authorization', 'CPU_preflight', 'draft', 'backend_evidence', 'asset_readiness'):
        binding = protocol[field]
        if sha(ROOT / binding['path']) != binding['sha256']:
            errors.append('Frozen native prerequisite changed: ' + field)
    if sha(ROOT / (BASE + 'R14N0_training_fixed_mask.json')) != result['training_only_fixed_mask_sha256']:
        errors.append('Training-only fixed mask seal changed')
    current, triage = read('research/CURRENT_DECISION.json'), read('research/20261021/method_triage.json')
    method = next(m for m in triage['methods'] if m['id'] == METHOD)
    if (METHOD in current['eligible_model_method_ids'] or method['model_execution_eligibility'] == 'ELIGIBLE_REGISTERED_SCREEN'
            or method['status'] != current['closed_method_statuses'][METHOD]
            or method['status'] != CLOSED):
        errors.append('Completed fixed native diagnostic cannot auto-authorize another model job')
    cpu = read(BASE + 'R14N0_saved_failure_analysis.json')
    cpu_protocol = read(BASE + 'R14N0_saved_failure_analysis_protocol.json')
    if (cpu['source_sha256'] != sha(ROOT / 'tools/analyze_complete_stripe_failure.py')
            or cpu['protocol_sha256'] != sha(ROOT / (BASE + 'R14N0_saved_failure_analysis_protocol.json'))
            or cpu['native_result_sha256'] != sha(ROOT / (BASE + 'R14N0_native_results.json'))
            or cpu['source_sha256'] != cpu_protocol['source_sha256']):
        errors.append('Saved CPU analysis source/evidence changed')
    if (cpu['CUDA_initialized'] is not False or cpu['new_model_generations'] != 0 or cpu['new_native_FFN_calls'] != 0
            or cpu['new_original_W1_W2_calls'] != 0 or cpu['new_quantizer_calls'] != 0
            or cpu['CPU_existing_affine_projection_calls'] != 8):
        errors.append('Saved CPU reference relabelled native model work')
    if len(cpu['scopes']) != 4 or [s['index'] for s in cpu['scopes']] != list(ORDER):
        errors.append('Saved CPU source coverage changed')
    else:
        for summary in cpu['summary']:
            indices = {'training': (1, 11), 'old_nonfit_development': (0, 10), 'all_exposed': ORDER}[summary['cohort']]
            scopes = [s for s in cpu['scopes'] if s['index'] in indices]
            baseline = math.fsum(s['actual_native_UB1_SSE'] for s in scopes)
            bf16 = math.fsum(s['CPU_BF16_three_direction_teacher_mean_post_reference_SSE'] for s in scopes)
            smooth = math.fsum(s['FP64_smooth_additive_common_post_reference_SSE'] for s in scopes)
            if (baseline != summary['UB1_SSE'] or bf16 != summary['CPU_BF16_teacher_mean_post_reference_SSE']
                    or smooth != summary['FP64_smooth_common_post_reference_SSE']
                    or bf16 / baseline != summary['CPU_BF16_teacher_mean_reference_over_UB1']
                    or smooth / baseline != summary['FP64_smooth_common_post_reference_over_UB1']):
                errors.append('Saved CPU mean-reference aggregation does not reproduce')
    if derive() != read(BASE + 'R14_sign_measure_deduction.json'):
        errors.append('Quadratic variance deduction/counterexample does not reproduce')
    common = read(BASE + 'R14N0_common_mask_reference.json')
    if (analyze_reference(result) != {k: v for k, v in common.items() if k not in ('source_sha256', 'native_result_sha256')}
            or common['source_sha256'] != sha(ROOT / 'tools/analyze_stripe_common_reference.py')
            or common['native_result_sha256'] != sha(ROOT / (BASE + 'R14N0_native_results.json'))):
        errors.append('Unexecuted common-mask reference or scope changed')
    mechanism_path = ROOT / (BASE + 'R14N0_mechanism_record.json')
    mechanism = read(BASE + 'R14N0_mechanism_record.json')
    for item in mechanism['evidence'] + [mechanism['proof_source']]:
        if sha(ROOT / item['path']) != item['sha256']:
            errors.append('Native mechanism evidence/source binding changed: ' + item['path'])
    if current['latest_research_review']['id'] == METHOD:
        if (current['latest_research_review'] != triage['latest_research_review']
                or current['latest_research_review']['mechanism_record_sha256'] != sha(mechanism_path)):
            errors.append('Completed native finding pointers disagree')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('R14N0:4original captures,32actual FFN/104extra linears,24Q2 controls, sealed fixed mask and finite rejection CONSISTENT.')
    print('Saved CPU references and finite cubature proof reproduce; no new model eligibility, free-video quality or native-model speed claim.')


if __name__ == '__main__':
    main()
