"""Portable consistency of completed C1 evidence; no model dependencies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'research/20261023'
    read = lambda name: json.loads((base / name).read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    protocol = read('C1N0_registered_protocol.json')
    run, budget = read('C1N0_native_results.json'), read('C1N0_budget.json')
    summary, resources = read('C1N0_execution_summary.json'), read('C1N0_completed_resources.json')
    analysis = read('C1N0_analysis.json')
    assert run['status'] == 'COMPLETE_NATIVE_C1_FIXED_SCOPE'
    assert summary['status'] == 'COMPLETED_FIXED_NATIVE_C1_DIAGNOSTIC'
    assert run['budget'] == budget and budget['used'] == budget['caps'] == protocol['caps']
    assert summary['actual_budget'] == resources['actual_budget'] == protocol['caps']
    for kind, count in protocol['caps'].items():
        actual = [r for r in budget['debits'] if r['kind'] == kind]
        assert len(actual) == count and all(r['status'] == 'COMPLETED' for r in actual)
    assert len({r['name'] for r in budget['debits']}) == len(budget['debits'])
    assert len(run['requests']) == 8 and len(run['finite_scopes']) == 72
    assert [r['source_index'] for r in run['requests']] == [-1, -1] + protocol['sources']
    assert all(r['original_four_latent_identity_required_and_passed'] for r in run['requests'][:2])
    reference = next(r for r in read('R16V0_native_results.json')['requests'] if r['arm'] == 'UB1')
    expected = [{'step': r['step'], 'sha256': r['sha256']} for r in reference['trajectory']]
    for r in run['requests']:
        assert r['main_native_counter_delta'] == {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600,
                                                 'cache_misses': 0, 'weight_transfers': 0}
        assert r['free_UB1_baseline_no_resets'] and r['candidate_free_video'] is False
    for r in run['requests'][:2]:
        assert r['trajectory'] == expected and r['initial_noise_sha256'] == reference['initial_noise_sha256']
    keys = {(r['source'], r['step'], r['block']) for r in run['finite_scopes']}
    assert keys == {(s, t, b) for s in protocol['sources'] for t in protocol['steps'] for b in protocol['blocks']}
    for r, numeric in zip(run['finite_scopes'], analysis['native_whole_site_rows']):
        assert (r['source'], r['step'], r['block']) == (numeric['source'], numeric['step'], numeric['block'])
        assert r['actual_UB1_post_bitwise_identity'] and r['same_prefix_full_input_clones']
        assert r['rows'] == 31668 and r['sampled_feature_rows'] == 336
        assert r['whole_site_nextnorm_SSE']['dense_DQ'] == r['whole_site_post_SSE']['dense_DQ'] == 0
        assert [r['whole_site_nextnorm_SSE'][a] for a in analysis['arms']] == numeric['losses']
    for metric in ('whole_site_nextnorm_SSE', 'whole_site_post_SSE'):
        assert all(r[metric][a] >= 0 for r in run['finite_scopes'] for a in protocol['arms'])
    totals = analysis['results']
    oracle = sum(min(r['losses']) for r in analysis['native_whole_site_rows'])
    assert abs(oracle - totals['oracle_whole_site']['total_nextnorm_SSE']) < 1e-6
    for r in totals.values():
        assert sum(r['whole_site_action_counts'].values()) == 72
        assert abs(sum(r['per_source_SSE'].values()) - r['total_nextnorm_SSE']) < 1e-6
        assert abs(r['total_nextnorm_SSE'] - oracle - r['total_regret_to_same_action_oracle']) < 1e-6
    actions = read('C1N0_action_parameters.json')
    assert len(actions['actions']) == 12 and actions['new_fit_or_factorization'] is False
    for r in actions['actions']:
        assert len(r['calibration_payloads']) == 24
        assert set(r['actions']) == {'swap_A', 'swap_B'}
        assert all(a['R256_unchanged'] for a in r['actions'].values())
    assert resources['actual_child_exit_code'] == 0 and not resources['GPU_isolation_violations']
    assert resources['logical_output_bytes_including_public_intermediates'] <= protocol['new_output_bytes_max']
    assert resources['peak_allocated_bytes'] <= protocol['peak_allocated_bytes_max']
    for row in resources['path_transform_receipts']:
        assert sha(ROOT / row['public_path']) == row['public_sha256'] and row['numeric_values_unchanged']
    assert run['protocol_sha256'] == summary['original_protocol_sha256'] == sha(base / 'C1N0_registered_protocol.json')
    assert analysis['official_video_quality_or_Pareto_or_native_policy_speed_result'] is False
    assert not analysis['automatic_followup']
    media = read('C1N0_baseline_media.json')
    assert len(media['records']) == 8
    assert all(r['decoded_video_stream']['nb_read_frames'] == 81 for r in media['records'])
    subjects = [r for r in media['records'] if r['source_index'] in (0, 2)]
    assert len(subjects) == 2 and all(r['original_R17_UB1_four_latent_and_initial_noise_identity'] for r in subjects)
    dense = read('C1N0_dense_value_CPU.json')
    robust = read('C1N0_dense_value_robustness.json')
    assert dense['actual_saved_feature_rows'] == 72 * 336
    assert robust['saved_rows'] == 72 * 336 and robust['current_feature_dimensions'] == 523
    assert robust['added_feature_dimensions'] == 8
    for doc in (dense, robust):
        assert doc['new_model_calls'] == 0 and doc['neural_training_updates'] == 0
        assert doc['video_quality_or_actual_policy_speed_result'] is False
        for value in doc['results'].values():
            assert abs(sum(value['per_source_captured_loss'].values()) - value['captured_measured_nextnorm_loss']) < 1e-6
            assert 0 <= value['fraction_of_all_sampled_measured_nextnorm_loss'] <= 1
        for key in ('source_sha256', 'plan_sha256'):
            assert len(doc[key]) == 64
    oracle_capture = dense['results']['same336row_top84_oracle']['fraction_of_all_sampled_measured_nextnorm_loss']
    assert all(v['fraction_of_all_sampled_measured_nextnorm_loss'] <= oracle_capture for v in robust['results'].values())
    receipt = read('C1N0_dense_value_robustness.json.redaction.json')
    assert receipt['numeric_values_exact'] and receipt['public_sha256'] == sha(base / 'C1N0_dense_value_robustness.json')
    findings = read('C1N0_findings.json')
    for path, expected in findings['input_evidence_sha256'].items():
        assert sha(ROOT / path) == expected
    assert findings['primary_compensated_action_test']['oracle_whole_site_gain_over_UB1'] == 0
    assert findings['dense_exception_information']['sources_positive_vs_rich_current'] == protocol['sources']
    print('PASS C1:8 baseline videos/72 paired native scopes, exact finite budget, held-out scalar consistency; no quality/speed promotion')


if __name__ == '__main__':
    main()
