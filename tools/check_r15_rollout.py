"""Verify archived R15V0 pairing, actual path, debit and review evidence offline."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/20261023'
METHOD = 'R15V0_MEAN_CONTRAST_PAIRED_ROLLOUT'
CAPS = {'full_generations': 2, 'offline_weight_factorizations': 2, 'training_steps': 0}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(run, budget, decode, resources):
    errors = []

    def need(value, message):
        if not value:
            errors.append(message)

    need(run['id'] == budget['method_id'] == resources['method_id'] == METHOD,
         'Paired method identity differs')
    need(run['status'] == 'COMPLETE_PAIRED_FULL_VIDEO_AWAITING_DECODE_AND_REVIEW',
         'Incomplete generation cannot become full-video evidence')
    need(run['intermediate_latent_reset_to_DQ'] is False,
         'Dense resets cannot count as free candidate rollout')
    need(run['video_quality_or_Pareto_acceptance'] is False
         and resources['population_quality_or_native_speed_acceptance'] is False,
         'One-source integration smoke cannot claim population quality or native speed')
    need(budget == run['budget'] and budget['used'] == budget['caps'] == CAPS,
         'Fixed generation/factorization debit changed')
    expected_charges = [('factors', {'offline_weight_factorizations': 2}),
                        ('dense_DQ', {'full_generations': 1}),
                        ('mean_contrast_candidate', {'full_generations': 1})]
    need([(d['name'], d['charge']) for d in budget['debits']] == expected_charges
         and all(d['status'] == 'COMPLETED' for d in budget['debits']),
         'Failed, missing or expanded native work cannot disappear')
    requests = run['requests']
    if len(requests) != 2:
        return errors + ['Exactly two actual full-video arms required']
    dense, candidate = requests
    need([r['arm'] for r in requests] == ['dense_DQ', 'mean_contrast_candidate'],
         'Wrong video arms')
    need(run['matched_initial_noise'] is True
         and dense['initial_noise_sha256'] == candidate['initial_noise_sha256'],
         'Paired initial noise differs')
    for request in requests:
        c = request['counter_delta']
        need([r['step'] for r in request['trajectory']] == list(range(4))
             and all(r['shape'] == [16, 21, 58, 104] for r in request['trajectory'])
             and request['free_student_states_no_reset'] is True,
             'Actual complete four-step native state coverage missing')
        need(c == {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600,
                   'weight_transfers': 0, 'cache_misses': 0}, 'Native model coverage/fallback differs')
        need(request['performance_benchmark'] is False,
             'Cold/warm one-order wall times are not a native speed benchmark')
        need(0 < request['peak_allocated_bytes'] <= 79 << 30, 'Native allocation cap exceeded')
    events = candidate['candidate_events']
    need(not dense['candidate_events'] and len(events) == 8
         and {(e['step'], e['module']) for e in events}
         == {(s, f'blocks.20.ffn.{slot}') for s in range(4) for slot in (0, 2)},
         'Candidate must actually execute all eight intended boundaries')
    for event in events:
        first = event['module'].endswith('.0')
        shapes = ([[23751, 5120, 5120], [7917, 5120, 13824],
                   [23751, 5120, 256], [23751, 256, 13824]] if first
                  else [[7917, 13824, 5120]])
        need(event['block'] == 20 and event['grid'] == [21, 29, 52]
             and event['original_QDQ_rows'] == 31668 and event['physical_linear_MKH'] == shapes,
             'Actual original-row QDQ or physical candidate GEMMs changed')
    need(decode['decoded_every_frame'] is True and decode['paired_frames'] == 81
         and decode['size'] == [832, 464] and decode['fps'] == 16
         and decode['automatic_visual_PASS'] is False,
         'Every-frame decode is required and does not automatically grant visual PASS')
    need(resources['actual_full_generations'] == 2
         and resources['actual_offline_weight_factorizations'] == 2
         and resources['actual_candidate_FFN_positions'] == 4
         and resources['actual_candidate_linear_calls'] == 20
         and resources['incremental_HF_output_bytes'] <= 1 << 28,
         'Actual bounded integration resources differ')
    return errors


def main():
    run, budget, decode, resources = [read(BASE / name) for name in (
        'R15V0_native_rollout_results.json', 'R15V0_budget_debit.json',
        'R15V0_video_inspection.json', 'R15V0_completed_resources.json')]
    errors = validate(run, budget, decode, resources)
    protocol_path = BASE / 'R15V0_registered_protocol.json'
    protocol = read(protocol_path)
    if run['protocol_sha256'] != sha(protocol_path):
        errors.append('Executed paired protocol changed')
    for name in ('authorization', 'approved_draft'):
        binding = protocol[name]
        if sha(ROOT / binding['path']) != binding['sha256']:
            errors.append('Frozen paired prerequisite changed: ' + name)
    for name in ('R15V0_native_rollout_results.json', 'R15V0_budget_debit.json', 'R15V0_video_inspection.json'):
        receipt = read(BASE / (name + '.redaction.json'))
        if sha(BASE / name) != receipt['public_sha256'] or receipt['numeric_values_exact'] is not True:
            errors.append('Public evidence/redaction identity differs: ' + name)
    raw_run = read(BASE / 'R15V0_native_rollout_results.json.redaction.json')['original_sha256']
    if decode['raw_run_sha256'] != raw_run:
        errors.append('Decode belongs to a different generation record')
    review = read(BASE / 'R15V0_full_duration_review.json')
    if (review['reviewed_frame_ids'] != list(range(81))
            or review['automatically_granted_by_decode'] is not False
            or review['human_real_time_playback_claim'] is not False):
        errors.append('Full decoded-frame review scope changed')
    registry = read(ROOT / 'research/candidate_rollout/registry/R15V0_mean_contrast.json')
    if (review['human_full_video_review_completed'] is False
            and registry['status'] in {'ROLLOUT_PASSED_SMOKE', 'READY_FOR_PARETO_BENCHMARK'}):
        errors.append('Agent frame review cannot replace required human full-video smoke review')
    fidelity_path = BASE / 'R15V0_decoded_pair_difference.json'
    fidelity = read(fidelity_path)
    rows = fidelity['per_frame']
    channels = 832 * 464 * 3
    total_sse = sum(r['RGB_SSE'] for r in rows)
    pooled_mse = total_sse / (81 * channels)
    if (len(rows) != 81 or [r['frame'] for r in rows] != list(range(81))
            or any(r['RGB_channel_count'] != channels for r in rows)
            or total_sse != fidelity['pooled_RGB_SSE']
            or not math.isclose(pooled_mse, fidelity['pooled_RGB_MSE_0_255'], rel_tol=1e-12)
            or not math.isclose(10 * math.log10(255 ** 2 / pooled_mse),
                                fidelity['pooled_MSE_PSNR_dB'], rel_tol=1e-12)
            or fidelity['input_video_sha256'] != [r['video_sha256'] for r in run['requests']]
            or fidelity['raw_run_sha256'] != raw_run
            or fidelity['source_sha256'] != sha(ROOT / 'tools/analyze_paired_rollout_difference.py')
            or fidelity['new_model_generations'] != 0
            or fidelity['official_quality_or_native_speed_claim'] is not False):
        errors.append('Actual decoded RGB fidelity provenance, units or aggregation differs')
    for row in rows:
        mse = row['RGB_SSE'] / channels
        expected = None if mse == 0 else 10 * math.log10(255 ** 2 / mse)
        if row['PSNR_dB'] != expected or row['exact_match'] != (mse == 0):
            errors.append('Per-frame PSNR or exact-match semantics differs')
    finite_psnr = [r['PSNR_dB'] for r in rows if not r['exact_match']]
    if not math.isclose(math.fsum(finite_psnr) / len(finite_psnr),
                        fidelity['mean_finite_per_frame_PSNR_dB'], rel_tol=1e-12):
        errors.append('Mean per-frame PSNR cannot be replaced by pooled-MSE PSNR')
    receipt = read(BASE / 'R15V0_decoded_pair_difference.json.redaction.json')
    if sha(fidelity_path) != receipt['public_sha256']:
        errors.append('Decoded fidelity public receipt changed')
    current = read(ROOT / 'research/CURRENT_DECISION.json')
    if METHOD in current['eligible_model_method_ids']:
        errors.append('Completed paired budget cannot authorize another model job')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('R15V0: two actual paired videos, four native sites,81 decoded frames and spent budget CONSISTENT.')
    print('Four-site integration/agent frame review; human video smoke verdict pending; no population quality or native speed certification.')


if __name__ == '__main__':
    main()
