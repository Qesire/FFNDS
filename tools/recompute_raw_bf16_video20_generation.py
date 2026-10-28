"""Recompute source-equal complete-video CORE/cost evidence from real receipts.

Stdlib only; no new metric, tensor, model, fit or GPU operation. Official quality
is a separate scope. Failed native parents remain failed in the provenance.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_video20_quality as evidence


def stats(values):
    return {'mean': statistics.fmean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values), 'source_count': len(values)}


def recompute(protocol, hf):
    plan = evidence.base.read(protocol)
    evidence.validate(plan, True)
    evidence.verify_generation(plan, hf)
    phases, _, _, parents = evidence.generation_metadata(plan, hf)
    sources = []
    for case in evidence.base.read_bound(ROOT, plan['source_cohort'])['sources']:
        row = {'source': case['native_source_ordinal'], 'source_index': case['index'],
               'image_sha256': case['image']['sha256'], 'prompt_sha256': case['prompt_sha256'],
               'stratum': case['stratum'], 'conditioning_identity': case['conditioning_identity'],
               'seed': 42, 'arms': {}, 'candidate_to_Dense': {}}
        for mode in evidence.MODES:
            phase = phases[row['source'], mode]
            core = evidence.base.read_bound(hf, phase['core_metrics'])
            report = core['metrics']
            value = {'generation': phase['generation_receipt'], 'core': phase['core_metrics'],
                     'native_evidence': phase['native_operand_transfer_evidence'],
                     'video_sha256': phase['video']['sha256'],
                     'generation_seconds': phase['generation_wall_seconds_including_encoders_decode_observers'],
                     'FFN_boundary_CUDA_seconds': phase['FFN_boundary_CUDA_seconds'],
                     'four_infer_wall_seconds': sum(s['wall_seconds'] for s in phase['stages'] if s['stage'] == 'infer'),
                     'peak_allocated_bytes': phase['peak_cuda_allocated_bytes'],
                     'peak_reserved_bytes': phase['peak_cuda_reserved_bytes'],
                     'factor_GPU_preload_seconds': phase['current_factor_family_gpu_preload_seconds'],
                     'factor_GPU_bytes': phase['current_factor_family_GPU_tensor_bytes'],
                     'requested_prefetch_H2D_bytes': phase['requested_prefetch_H2D_bytes'],
                     'skipped_original_W2_H2D_bytes': phase['skipped_original_W2_H2D_bytes'],
                     'original_GPU_buffers_retained': phase['original_GPU_buffers_retained'],
                     'order_within_triad': phase['order_within_triad'],
                     'Farneback_all_pair_mean_flow_downsampled_px': report['temporal_motion']['downsampled_Farneback_mean_flow_px']['mean'],
                     'paired_RGB_RMSE': report.get('paired_spatial', {}).get('RGB_RMSE_0to1', {}).get('mean'),
                     'mean_per_pair_Farneback_magnitude_ratio': report.get('paired_motion', {}).get('candidate_to_reference_flow_magnitude_ratio', {}).get('mean')}
            row['arms'][mode] = value
        dense = row['arms']['dense']
        for mode in evidence.MODES[1:]:
            value = row['arms'][mode]
            row['candidate_to_Dense'][mode] = {
                'generation_time_ratio': value['generation_seconds']/dense['generation_seconds'],
                'FFN_time_ratio': value['FFN_boundary_CUDA_seconds']/dense['FFN_boundary_CUDA_seconds'],
                'CORE_mean_flow_ratio': value['Farneback_all_pair_mean_flow_downsampled_px']/dense['Farneback_all_pair_mean_flow_downsampled_px']
                    if dense['Farneback_all_pair_mean_flow_downsampled_px'] > 0 else None,
                'allocated_memory_difference_bytes': value['peak_allocated_bytes']-dense['peak_allocated_bytes']}
        sources.append(row)
    fields = ('generation_seconds', 'FFN_boundary_CUDA_seconds', 'four_infer_wall_seconds',
              'peak_allocated_bytes', 'peak_reserved_bytes', 'factor_GPU_preload_seconds',
              'Farneback_all_pair_mean_flow_downsampled_px')
    summaries = {mode: {key: stats([r['arms'][mode][key] for r in sources]) for key in fields}
                 for mode in evidence.MODES}
    comparisons = {}
    for mode in evidence.MODES[1:]:
        comparison = {}
        for name, field in (('observed_generation_speed_ratio', 'generation_seconds'),
                            ('observed_FFN_boundary_speed_ratio', 'FFN_boundary_CUDA_seconds'),
                            ('observed_four_infer_speed_ratio', 'four_infer_wall_seconds')):
            comparison[name] = summaries['dense'][field]['mean']/summaries[mode][field]['mean']
        comparison['CORE_flow_ratio_of_source_equal_means'] = summaries[mode][fields[-1]]['mean']/summaries['dense'][fields[-1]]['mean']
        ratios = [{'source': r['source'], 'ratio': r['candidate_to_Dense'][mode]['CORE_mean_flow_ratio']} for r in sources]
        defined = [r for r in ratios if r['ratio'] is not None]
        comparison['CORE_mean_of_defined_source_flow_ratios'] = statistics.fmean(r['ratio'] for r in defined)
        comparison['all_source_flow_ratios_sorted'] = sorted(defined, key=lambda r: (r['ratio'], r['source']))
        comparison['source_flow_ratio_undefined'] = [r['source'] for r in ratios if r['ratio'] is None]
        comparison['paired_RGB_RMSE_source_equal'] = stats([r['arms'][mode]['paired_RGB_RMSE'] for r in sources])
        comparisons[mode] = comparison
    counts = {}
    for parent in parents:
        for key, n in parent['operation_ledger']['observed'].items():
            counts[key] = counts.get(key, 0)+n
    assert len(sources) == 20 and len({r['image_sha256'] for r in sources}) == 20
    assert counts['video_generations'] == 60 and counts['native_full_model_flows'] == 240
    assert counts['native_FFN_calls'] == 9600 and counts['BF16_factor_GEMM'] == 12800
    assert all(p['original_GPU_buffers_retained'] for r in sources for p in r['arms'].values())
    return {'status': 'COMPLETE_TWENTY_IMAGE_SIXTY_FULL_DEPTH_VIDEOS_CORE_AND_OBSERVED_COST',
            'numeric_path': 'BF16_ORIGINAL', 'quality_protocol': evidence.base.binding(protocol, ROOT),
            'native_parents': plan['parents'], 'prefix_CPU_repair': plan['prefix_repair'],
            'source_cohort': plan['source_cohort'], 'source_count': 20, 'genuine_videos': 60,
            'native_FFN_calls': 9600, 'full_model_flows': 240,
            'same_source_noise_byte_equal_triads': 20,
            'all_native_operation_counts_including_failed_scopes': counts,
            'CORE_evaluation_calls': {'native_scope_actual': counts['core_CPU_reports'], 'separate_CPU_repair': 4,
                                      'valid_final_reports': 60, 'computed_but_unsaved_in_failed_parents': 2},
            'source_equal_arm_statistics': summaries, 'comparisons': comparisons, 'per_source': sources,
            'strata': {s: sum(r['stratum'] == s for r in sources) for s in ('subject', 'background', 'camera')},
            'source_history': 'HISTORICALLY_EXPOSED_DEVELOPMENT_NOT_BLIND; all20 disjoint6fit images',
            'timing_identity': 'SINGLE_OBSERVED_ORDER_WITH_OBSERVERS; four_native_attempt_setups; no_warmups_or_repeat_SYS',
            'motion_identity': 'CORE_DOWNSAMPLED_FARNEBACK_DIAGNOSTIC_NOT_RAFT_OR_SEMANTIC_ACTION',
            'official_quality': 'SEPARATE_REGISTERED_SCOPE; consult_ACTIVE_and_actual_RUN',
            'human_fullplay_review': 'PENDING', 'fair_warmed_system_speed': 'NOT_MEASURED',
            'quality_or_speed_acceptance': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = recompute(args.protocol.resolve(), args.hf_root.resolve())
    if args.out.exists():
        raise ValueError('Fresh recomputation output required')
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'comparisons': result['comparisons']}))


if __name__ == '__main__':
    main()
