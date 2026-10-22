"""Recalculate actual full160 results from immutable scalar/media receipts."""
import json
import math
from pathlib import Path

from row_dispatch_contract import ARMS, ROOT, atomic, read, sha


def analyze(run, review):
    benches = run['native_microbenchmarks']
    per_position = []
    total_sse = {arm: math.fsum(b['arms'][arm]['native_post_SSE_to_dense'] for b in benches) for arm in ARMS}
    best_fixed = min(('fixed_stride', 'fixed_hash'), key=lambda a: total_sse[a])
    for b in benches:
        fixed_sse = min(b['arms'][a]['native_post_SSE_to_dense'] for a in ('fixed_stride', 'fixed_hash'))
        dynamic_sse = b['arms']['dynamic_residual']['native_post_SSE_to_dense']
        per_position.append({'step': b['step'], 'block': b['block'],
                             'dynamic_local_gain_over_position_best_fixed': 1 - dynamic_sse / fixed_sse,
                             'same_input_native_SSE': {a: b['arms'][a]['native_post_SSE_to_dense'] for a in ARMS},
                             'median_native_core_ms': {a: b['arms'][a]['median_cuda_ms'] for a in ARMS}})
    median_sums = {a: math.fsum(b['arms'][a]['median_cuda_ms'] for b in benches) for a in ARMS}
    observations = {}
    for request in run['requests']:
        arm = request['arm']
        core = [r for r in request['FFN_timing']['rows'] if r['kind'] == 'FFN_norm_through_gate_residual']
        observations[arm] = {'observed_full160_native_core_cuda_ms': math.fsum(r['cuda_ms'] for r in core),
                             'observed_full160_conservative_charged_cuda_ms': request['FFN_timing']['cuda_ms'],
                             'whole_generation_seconds_including_observation': request['elapsed_seconds_with_observation'],
                             'whole_generation_performance_benchmark': False,
                             'actual_dispatch_sites': len(request['dispatch_events']),
                             'distinct_dense_row_set_SHA': len({e['dense_row_ids_sha256'] for e in request['dispatch_events']}),
                             'peak_allocated_bytes': request['peak_allocated_bytes']}
    fidelity = review['RGB_fidelity_to_dense']
    best_pixel_fixed = min(('fixed_stride', 'fixed_hash'), key=lambda a: fidelity[a]['pooled_RGB_MSE_0_255'])
    pixel_ratio = fidelity['dynamic_residual']['pooled_RGB_MSE_0_255'] / fidelity[best_pixel_fixed]['pooled_RGB_MSE_0_255']
    return {'per_position': per_position, 'local_source_cluster_count': 1,
            'pooled_four_native_position_SSE': total_sse,
            'best_single_fixed_of_two_native_controls': best_fixed,
            'dynamic_native_local_gain_over_best_single_fixed': 1 - total_sse['dynamic_residual'] / total_sse[best_fixed],
            'four_native_position_sum_of_median_ms': median_sums,
            'four_position_native_core_speed_ratio_dense_over_dynamic': median_sums['dense_DQ'] / median_sums['dynamic_residual'],
            'native_timing_scope': 'Four actual dense-state FFN inputs at block20; five counterbalanced repetitions each. This ratio does not measure all160sites or full-model system acceleration.',
            'full_generation_observations': observations,
            'RGB_fidelity_to_dense': fidelity,
            'dynamic_pooled_RGB_MSE_over_best_pixel_fixed': pixel_ratio,
            'best_pixel_fixed_of_two': best_pixel_fixed,
            'mean_frame_PSNR_and_pooled_MSE_rankings_differ': (
                fidelity['dynamic_residual']['mean_frame_PSNR_dB'] > fidelity['fixed_stride']['mean_frame_PSNR_dB'] and pixel_ratio > 1),
            'scientific_readout': 'The pre-W1 residual-magnitude policy reduces same-input native writeback SSE at all four measured positions versus both fixed row controls, but its complete free video has worse pooled decoded RGB fidelity than the better fixed pattern. This is a real local-to-trajectory ranking mismatch, not evidence that RGB deviation equals perceptual quality loss. Human noncollapse review and independent official motion/quality evaluation remain distinct.',
            'fresh_independent_quality_sources': 0, 'official_quality_models_executed': 0,
            'new_model_generations': 0, 'population_quality_or_Pareto_acceptance': False}


def main():
    base = ROOT / 'research/20261023'
    run_path, review_path = base / 'R16V0_native_results.json', base / 'R16V0_video_review.json'
    result = {'id': 'R16V0_ACTUAL_NATIVE_AND_VIDEO_READOUT', 'status': 'COMPLETED_CPU_RECALCULATION',
              'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (run_path, review_path)},
              'source_sha256': sha(__file__), **analyze(read(run_path), read(review_path))}
    target = base / 'R16V0_analysis.json'
    if target.exists():
        if read(target) != result:
            raise RuntimeError('Preserve a prior analysis; do not silently change measured aggregation')
    else:
        atomic(target, result)
    print(json.dumps({k: result[k] for k in ('dynamic_native_local_gain_over_best_single_fixed',
                     'four_position_native_core_speed_ratio_dense_over_dynamic',
                     'dynamic_pooled_RGB_MSE_over_best_pixel_fixed', 'mean_frame_PSNR_and_pooled_MSE_rankings_differ')}))


if __name__ == '__main__':
    main()
