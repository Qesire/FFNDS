"""Portable completed/partial C2 numerical checks; no model dependencies."""
from pathlib import Path
from c2_contract import ARMS, CAPS, ROOT, read, sha


def main(*, prefix='C2N0', registration_key='registered_c2_smoke',
         complete_status='COMPLETE_C2_NATIVE_VIDEOS_PENDING_FULL_REVIEW'):
    base = ROOT / 'research/20261023'
    run, budget, resources = [read(base / f'{prefix}_{n}.json') for n in ('native_results', 'budget', 'completed_resources')]
    registration = read(ROOT / 'research/CURRENT_DECISION.json')[registration_key]
    p = read(ROOT / registration['protocol'])
    assert run['budget'] == budget and resources['actual_budget'] == budget['used']
    assert run['protocol_sha256'] == sha(ROOT / registration['protocol'])
    assert all(0 <= budget['used'][k] <= CAPS[k] for k in CAPS)
    for k, used in budget['used'].items(): assert len([d for d in budget['debits'] if d['kind'] == k]) == used
    assert len({d['name'] for d in budget['debits']}) == len(budget['debits'])
    assert [r['name'] for r in run['requests']] == ARMS[:len(run['requests'])]
    if run['status'] == complete_status:
        assert resources['actual_native_child_exit_code'] == 0 and budget['used'] == CAPS and len(run['requests']) == 6
        assert resources['GPU_isolation_violations'] == [] and run['resources']['sampled_isolation_pass'] is True
        if prefix == 'C2N1':
            lease = run['early_CUDA_reservation']
            assert lease['before_CPU_asset_loading'] and lease['same_model_process']
            assert lease['NVML_UUID_matches_locked_GPU'] and lease['own_process_visible_to_NVML']
        assert all(d['status'] == 'COMPLETED' for d in budget['debits'])
        control = run['requests'][0]
        expected = next(r for r in read(base / 'R16V0_native_results.json')['requests'] if r['arm'] == 'UB1')
        assert control['initial_noise_sha256'] == expected['initial_noise_sha256']
        assert control['trajectory'] == [{'step': s['step'], 'sha256': s['sha256']} for s in expected['trajectory']]
        assert len({r['initial_noise_sha256'] for r in run['requests'][1:]}) == 1
        assert len({r['image_sha256'] for r in run['requests'][1:]}) == 1
        assert len({r['prompt_sha256'] for r in run['requests'][1:]}) == 1
        candidate = run['requests'][-1]
        assert candidate['physical_row_dispatch_sites'] == 160
        assert {(r['step'], r['block']) for r in candidate['norm_statistic_events']} == {(s, b) for s in range(4) for b in range(40)}
        assert all(r['scaled_score_calls'] == 1 and r['CPU_tensor_transfers'] == 0 for r in candidate['norm_statistic_events'])
        assert all(r['score_statistics_dtype'] == 'torch.float32' for r in candidate['norm_statistic_events'])
    for r in run['requests']:
        assert r['free_student_no_resets'] and r['counter_delta'] == {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600,
                                                                   'cache_misses': 0, 'weight_transfers': 0}
    assert resources['logical_output_bytes_at_export'] <= p['new_output_bytes_max']
    for row in resources['path_transform_receipts']:
        assert sha(ROOT / row['public_path']) == row['public_sha256'] and row['numeric_values_unchanged']
    media_path = base / (prefix + '_video_review.json')
    if media_path.exists():
        media = read(media_path)
        assert media['frames_per_video'] == 81 and media['videos_decoded'] == 6 and media['total_decoded_frames'] == 486
        assert media['source_cluster_count'] == 1 and media['source_index'] == 6
        assert media['automatic_visual_PASS'] is False and media['official_quality_or_repeated_system_speed_result'] is False
        assert len(media['per_frame']) == 81
        channels = 832 * 464 * 3
        for arm, metric in media['RGB_fidelity_to_dense'].items():
            total = sum(r['observations'][arm]['RGB_SSE'] for r in media['per_frame'])
            assert abs(metric['pooled_RGB_MSE_0_255'] - total / (81 * channels)) < 1e-10
    print('PASS C2 actual native/partial budget and media evidence; no automatic quality/speed promotion')


if __name__ == '__main__': main()
