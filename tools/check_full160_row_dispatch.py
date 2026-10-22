"""Offline actual-budget, native-path and full-video evidence checks."""
import json
import math
from pathlib import Path

from row_dispatch_contract import ARMS, CAPS, METHOD, ROOT, read, sha

BASE = ROOT / 'research/20261023'


def validate(run, budget, review, resources):
    errors = []

    def need(value, message):
        if not value:
            errors.append(message)

    need(run['id'] == budget['method_id'] == resources['id'] == METHOD, 'Wrong actual method identity')
    need(run['status'] == 'COMPLETE_FULL160_GENERATION_AND_NATIVE_MICROBENCHMARK_AWAITING_VIDEO_REVIEW', 'Incomplete model execution')
    need(run['budget'] == budget and budget['caps'] == budget['used'] == CAPS, 'Fixed model debit changed')
    need(len(budget['debits']) == 169 and len({d['name'] for d in budget['debits']}) == 169
         and all(d['status'] == 'COMPLETED' for d in budget['debits']), 'Missing, failed or duplicate model work')
    need(run['intermediate_latent_reset_to_DQ'] is False and run['official_quality_or_Pareto_acceptance'] is False,
         'Free trajectory or one-source claim boundary changed')
    requests = run['requests']
    if len(requests) != 5:
        return errors + ['Require five actual full videos']
    need([r['arm'] for r in requests] == ARMS and len({r['initial_noise_sha256'] for r in requests}) == 1,
         'Matched arms/initial noise differ')
    need(requests[0]['original_dense_four_latents_bitwise_equal'] is True, 'Original dense trajectory not preserved')
    for request in requests:
        need([r['step'] for r in request['trajectory']] == list(range(4))
             and all(r['shape'] == [16, 21, 58, 104] for r in request['trajectory'])
             and request['free_student_states_no_reset'] is True, 'Not a complete original free student trajectory')
        need(request['counter_delta_excluding_debited_benchmarks'] == {
            'linear_forward_calls': 1600, 'activation_qdq_calls': 1600, 'cache_misses': 0, 'weight_transfers': 0},
             'Original logical runtime/fallback coverage differs')
        need(request['whole_generation_performance_benchmark'] is False, 'One-order generation observation cannot become benchmark')
        need(0 < request['peak_allocated_bytes'] <= 79 << 30, 'Peak allocation outside scope')
        events = request['dispatch_events']
        need((not events) if request['arm'] == 'dense_DQ' else
             len(events) == 160 and {(e['step'], e['block']) for e in events} == {(s, b) for s in range(4) for b in range(40)},
             'Actual full160 dispatch coverage missing')
        for event in events:
            dense, sparse, width = event['dense_rows'], event['sparse_rows'], event['sparse_width']
            need(event['arm'] == request['arm'] and not event['benchmark']
                 and dense + sparse == 31668 and width == [3456, 6912, 6912, 10368][event['step']]
                 and dense == (0 if request['arm'] == 'UB1' else 7917)
                 and event['W1_physical_MKH'] == [[sparse, 5120, width], [dense, 5120, 13824]]
                 and event['R256_physical_MKH'] == [[31668, 5120, 256], [31668, 256, 5120]],
                 'Observed compact GEMM operands/budget differ')
        core = [r for r in request['FFN_timing']['rows'] if r['kind'] == 'FFN_norm_through_gate_residual']
        need(len(core) == 160 and {(r['step'], r['block']) for r in core} == {(s, b) for s in range(4) for b in range(40)},
             'Original full160 FFN timing regions missing')
    benches = run['native_microbenchmarks']
    need(len(benches) == 4 and [b['step'] for b in benches] == list(range(4)), 'Missing four native benchmark positions')
    for bench in benches:
        need(bench['block'] == 20 and bench['actual_full_native_rows'] == 31668
             and bench['UB1_reference_bit_mismatches'] == 0 and len(bench['timing_orders']) == 5,
             'Original native bit control or timing scope differs')
        for arm in ARMS:
            row = bench['arms'][arm]
            need(len(row['cuda_ms']) == len(row['host_ms']) == 5
                 and all(math.isfinite(t) and t > 0 for t in row['cuda_ms'] + row['host_ms'])
                 and row['median_cuda_ms'] == sorted(row['cuda_ms'])[2]
                 and row['median_host_ms'] == sorted(row['host_ms'])[2], 'Invalid actual native timing aggregation')
    need(review['decoded_every_frame'] is True and review['frames_per_video'] == 81 and review['size'] == [832, 464]
         and review['fps'] == 16 and review['automatic_visual_PASS'] is False, 'Complete media decode is not visual PASS')
    need(review['input_video_sha256'] == [r['video_sha256'] for r in requests], 'Decode belongs to different video bytes')
    need(len(review['per_frame']) == 81 and [r['frame'] for r in review['per_frame']] == list(range(81)), 'Missing every-frame decoded observations')
    for arm in ARMS:
        sse = sum(row['observations'][arm]['RGB_SSE'] for row in review['per_frame'])
        count = 81 * 832 * 464 * 3
        mse = sse / count
        agg = review['RGB_fidelity_to_dense'][arm]
        need(math.isclose(agg['pooled_RGB_MSE_0_255'], mse, rel_tol=1e-12, abs_tol=1e-12), 'Decoded RGB unit/aggregation changed')
        need(agg['pooled_MSE_PSNR_dB'] == (None if not sse else 10 * math.log10(255 ** 2 / mse)), 'RGB PSNR semantics changed')
    need(resources['full_generations'] == 5 and resources['standalone_FFN_forwards'] == 164
         and resources['HF_output_bytes'] <= 1 << 29 and resources['max_GPUs'] == 1
         and resources['human_full_video_review_completed'] is True
         and resources['population_quality_or_Pareto_acceptance'] is False, 'Actual bounded resources/claim changed')
    return errors


def main():
    run, budget, review, resources = [read(BASE / name) for name in (
        'R16V0_native_results.json', 'R16V0_budget_debit.json', 'R16V0_video_review.json', 'R16V0_completed_resources.json')]
    errors = validate(run, budget, review, resources)
    protocol_path = BASE / 'R16V0_registered_protocol.json'
    protocol = read(protocol_path)
    if run['protocol_sha256'] != sha(protocol_path):
        errors.append('Executed immutable protocol changed')
    for key in ('authorization', 'approved_draft'):
        b = protocol[key]
        if sha(ROOT / b['path']) != b['sha256']:
            errors.append('Executed authorization/draft changed')
    for name in ('R16V0_native_results.json', 'R16V0_budget_debit.json', 'R16V0_video_review.json'):
        receipt = read(BASE / (name + '.redaction.json'))
        if sha(BASE / name) != receipt['public_sha256'] or not receipt['numeric_values_exact']:
            errors.append('Public original scalar receipt identity differs')
    receipt = read(BASE / 'R16V0_native_results.json.redaction.json')
    if review['raw_run_sha256'] != receipt['original_sha256']:
        errors.append('CPU review belongs to a different raw model receipt')
    human_path = BASE / 'R16V0_human_review.json'
    human = read(human_path)
    full = read(BASE / 'R16V0_full_duration_review.json')
    if (human['human_full_video_review_completed'] is not True or
            human['verdict'] != 'NO_CATASTROPHIC_COLLAPSE' or
            human['user_reply'] != '已完整观看，没有明显崩坏' or
            human['candidate_video_sha256'] != run['requests'][-1]['video_sha256'] or
            human['baseline_video_sha256'] != run['requests'][0]['video_sha256'] or
            full['human_review_sha256'] != sha(human_path) or
            full['automatically_granted_by_decode'] is not False or
            full['reviewed_frame_ids'] != list(range(81))):
        errors.append('Separate human full-playback verdict/media provenance changed')
    if METHOD in read(ROOT / 'research/CURRENT_DECISION.json')['eligible_model_method_ids']:
        errors.append('Spent full160 fixed budget cannot authorize further execution')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('R16V0:5actual full160videos,164nativeFFNs,all81frames decoded,original controls and spent budget CONSISTENT.')
    print('Human noncollapse smoke confirmed; one old source, no population official quality or repeated system benchmark certification.')


if __name__ == '__main__':
    main()
