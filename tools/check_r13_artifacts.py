"""Verify public R13 CPU/saved-data evidence without private tensor assets."""
import json
import math
from pathlib import Path
import statistics

from analyze_ffn_compression_axes import summarize
from analyze_spatial_sharing_limits import build, digest
from audit_anchored_affine_remainder import aggregate
from probe_pooled_nonlinear_moments import gaussian_gelu_mean, projection_cost
from prepare_complete_stripe_operand_draft import select_stripes

ROOT = Path(__file__).resolve().parents[1]
BASE = 'research/20261022/'


def read(name):
    return json.loads((ROOT / (BASE + name)).read_text())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def bound(path, expected):
    need(digest(ROOT / path) == expected, 'Executed source/evidence changed: ' + path)


def check():
    spatial, axis, affine, limits, moments = [read(name) for name in (
        'R13_spatial_residual_coherence.json', 'R13_saved_compression_axis_value.json',
        'R13_anchored_affine_remainder.json', 'R13_spatial_sharing_limits.json', 'R13_pooled_nonlinear_moments.json')]
    for result in (spatial, affine):
        expected_scopes = {(s, i) for s in range(4) for i in range(24)}
        need({(r['step'], r['index']) for r in result['scopes']} == expected_scopes and len(result['scopes']) == 96,
             'Saved source/position coverage changed')
        for path, sha in result['source_sha256'].items():
            if path.startswith('.runtime/'):
                # Clean exports omit upstream runtime; the old source audit
                # keeps its original binding. Do not require private assets.
                continue
            bound(path, sha)
        for item in result['original_collection_bindings']:
            bound(item['record'], item['sha256'])
        need(result['new_model_generations'] == 0 and result['CUDA_initialized'] is False,
             'Saved CPU evidence relabelled as model execution')
    bound(BASE + 'R13_spatial_residual_audit_protocol.json', spatial['protocol_sha256'])
    bound(BASE + 'R13_anchored_affine_audit_protocol.json', affine['protocol_sha256'])
    need(sum(s['complete_observed_horizontal4_stripes'] for s in spatial['scopes']) == 0,
         'Missing complete stripes relabelled observed')
    need(spatial['payloads_read'] == 96 and spatial['payload_bytes_checked'] == 1008643584,
         'Original payload extent changed')
    bound('tools/analyze_ffn_compression_axes.py', axis['source_sha256'])
    bound(BASE + 'R13_spatial_residual_coherence.json', axis['spatial_audit_sha256'])
    need(summarize(axis['rows']) == axis['pooled'], 'Saved equal-budget axis summary does not reproduce')
    occupied = set()
    for r in axis['rows']:
        for token in (r['left_native_row'], r['right_native_row']):
            key = r['index'], r['step'], token
            need(key not in occupied, 'Matched pair vertices overlap')
            occupied.add(key)
        need(r['step'] in (1, 2) and r['sharing_advantage'] == r['hidden_pruning_pair_SSE'] - r['row_sharing_pair_SSE'],
             'Pair action/budget arithmetic changed')
    for step, item in axis['per_step'].items():
        need(item == summarize([r for r in axis['rows'] if r['step'] == int(step)]),
             'Step axis aggregation changed')
    bound('research/20261021/R9A1_native_affine_results.json', affine['original_R9_result_sha256'])
    r9 = json.loads((ROOT / 'research/20261021/R9A1_native_affine_results.json').read_text())
    for original, reused in zip(r9['affine_matrices'], affine['original_affine_bindings']):
        need(reused['sha256'] == original['sha256'] and reused['tensor_sha256'] == original['tensor_sha256']
             and reused['original_training_indices'] == original['training_indices'], 'R9 matrix or fit exposure changed')
    need(len(affine['original_affine_bindings']) == 2 and affine['CPU_existing_affine_linear_calls'] == 96
         and affine['new_training_updates'] == 0, 'Frozen affine audit relabelled a new fit')
    for item in affine['summary']:
        cohort = item['cohort']
        scopes = [r for r in affine['scopes'] if r['step'] == item['step'] and
                  (cohort == 'all_exposed24' or r['R9_fit_source'] == (cohort == 'original_R9_fit_odd'))]
        need({k: v for k, v in item.items() if k not in ('step', 'cohort')} == aggregate(scopes),
             'Affine cohort aggregation does not reproduce')
    bound('tools/analyze_spatial_sharing_limits.py', limits['source_sha256'])
    for item in limits['input_bindings'].values():
        bound(item['path'], item['sha256'])
    need({k: v for k, v in limits.items() if k not in ('source_sha256', 'input_bindings')} == build(axis, affine),
         'Finite pair/ideal stripe identities do not reproduce')
    bound('tools/probe_pooled_nonlinear_moments.py', moments['source_sha256'])
    bound(BASE + 'R13_pooled_moment_probe_protocol.json', moments['protocol_sha256'])
    for r in moments['gaussian_exact_GELU_checks']:
        need(math.isclose(r['closed_form'], gaussian_gelu_mean(r['mean'], r['variance']), abs_tol=1e-14)
             and r['absolute_error'] < 1e-7, 'Synthetic Gaussian identity changed')
    for r in moments['joint_cubature_cases']:
        need(math.isclose(r['single_sketch_expected_SSE'], r['two_point_ensemble_bias_SSE'] + r['single_sketch_variance_SSE'], abs_tol=1e-12),
             'Ensemble bias confused with single-sketch error')
        need(r['sign_covariance_max_error'] < 1e-10 and r['quadratic_moment_max_error'] < 1e-10,
             'Covariance/quadratic identity failed')
    need(moments['full_projection_MAC_fraction'] == projection_cost() and projection_cost()['total'] <= .5,
         'Extra W1 point or odd stripe cost omitted')
    need(moments['new_model_generations'] == moments['models_loaded'] == moments['fitted_parameters'] == 0
         and moments['CUDA_initialized'] is False and moments['selection_or_native_success'] is False,
         'Toy evidence relabelled native or conditional success')
    mechanism = read('R13_mechanism_record.json')
    for item in mechanism['evidence']:
        bound(item['path'], item['sha256'])
    draft_binding = mechanism['next_operand_coordinate_draft']
    bound(draft_binding['path'], draft_binding['sha256'])
    need(mechanism['native_registration'] is None and mechanism['model_execution_eligibility_change'] is False,
         'Mechanism record cannot register a model route')
    layout = read('R13_complete_stripe_operand_DRAFT.json')
    bound('tools/prepare_complete_stripe_operand_draft.py', layout['source_sha256'])
    need(layout['new_model_generations'] == layout['new_operand_payloads'] == 0
         and layout['native_registration'] is None and layout['execution_authority'] is None,
         'Coordinate draft relabelled collected or authorized')
    for item in layout['cases']:
        groups, rows = select_stripes(item['index'])
        need(groups == item['planned_stripe_ids'] and rows == item['planned_native_row_ids']
             and item['observed_new_operands'] == 0, 'Planned native coordinates do not reproduce')
    cost = read('R13_synthetic_spatial_response_cost.json')
    bound('tools/probe_spatial_response_cost.py', cost['source_sha256'])
    bound(BASE+'R13_synthetic_cost_protocol.json', cost['protocol_sha256'])
    need(cost['models_loaded'] == cost['new_model_generations'] == cost['fitted_parameters'] == 0
         and cost['quality_or_native_model_speed_acceptance'] is False, 'Synthetic cost relabelled model success')
    need(len(cost['QDQ_controls']) == 8 and all(r['bit_mismatches']==0 for r in cost['QDQ_controls']), 'Synthetic original QDQ controls changed')
    for arm, values in cost['samples_ms'].items():
        need(len(values)==7 and statistics.median(values)==cost['median_ms'][arm], 'Whole-chain cost samples do not reproduce')
        need(cost['speedup_over_same_run_dense'][arm]==cost['median_ms']['dense']/cost['median_ms'][arm], 'Speed denominator changed')
    need(cost['pooled_dynamic_speed_gate']==(cost['speedup_over_same_run_dense']['pooled_dynamic']>=1.5), 'Direct synthetic speed gate relabelled')
    contrast = read('R13_three_direction_contrast.json')
    bound('tools/audit_three_direction_contrast.py', contrast['source_sha256'])
    bound(BASE+'R13_linear_contrast_protocol.json', contrast['protocol_sha256'])
    bound(BASE+'R13_anchored_affine_remainder.json', contrast['prior_affine_result_sha256'])
    need(len(contrast['rows'])==24 and {r['index'] for r in contrast['rows']}==set(range(24)), 'Linear rewrite coverage changed')
    discrepancy = math.fsum(r['gated_rewrite_discrepancy_SSE'] for r in contrast['rows'])
    baseline = math.fsum(r['original_UB1_gated_SSE'] for r in contrast['rows'])
    need(discrepancy==contrast['gated_rewrite_discrepancy_SSE'] and discrepancy/baseline==contrast['rewrite_discrepancy_over_UB1'], 'Linear rounding error denominator changed')
    need(contrast['fixed_CPU_precision_screen_pass']==(discrepancy/baseline<=.01)
         and contrast['projection_MAC_fraction']==projection_cost()['total']-.25*5120/(2*13824), 'Three-direction precision/budget gate changed')
    need(all(r['native_spatial_groups_observed']==0 for r in contrast['rows']) and contrast['CUDA_initialized'] is False
         and contrast['native_identity_or_speed_pass'] is False, 'Arbitrary arithmetic fixtures relabelled native spatial groups')
    current = json.loads((ROOT / 'research/CURRENT_DECISION.json').read_text())
    triage = json.loads((ROOT / 'research/20261021/method_triage.json').read_text())
    need(current['latest_static_review'] == triage['latest_static_review'], 'Live CPU review pointers disagree')
    if current['latest_static_review']['id'] == mechanism['id']:
        bound(current['latest_static_review']['result'], current['latest_static_review']['result_sha256'])
    print('R13: saved pairs, finite identities, affine rewrites and fixed synthetic A100 cost CONSISTENT.')
    print('No Wan pooled forward, conditional policy, free-video quality, original model FFN speed or eligibility certified.')


if __name__ == '__main__':
    check()
