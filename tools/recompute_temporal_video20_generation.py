"""Reconstruct actual new40/reusedDense20 CORE and observed cost evidence."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_temporal_video20_quality as quality
from recompute_raw_bf16_video20_generation import stats


def recompute(protocol, hf):
    plan = quality.base.read(protocol); quality.validate(plan, True); quality.verify_generation(plan, hf)
    old, _ = quality.dense_quality(plan, hf)
    phases, _, _, _ = quality.q.generation_metadata(old, hf)
    phases = {k: v for k, v in phases.items() if k[1] == 'dense'}
    receipts, counts = [], {}
    for parent in plan['parents']:
        report = quality.base.read_bound(hf, parent['RUN'])
        receipt = quality.base.read_bound(hf, parent['exit'])
        receipts.append({**parent, 'actual_exit': receipt['child_returncode'], 'actual_outer_seconds': receipt['outer_seconds']})
        for phase in report['phases']: phases[phase['native_source_ordinal'], phase['mode']] = phase
        for key, n in report['operation_ledger']['observed'].items(): counts[key] = counts.get(key, 0)+n
    sources = []
    for case in quality.base.read_bound(ROOT, plan['source_cohort'])['sources']:
        source = case['native_source_ordinal']
        row = {'source': source, 'source_id': case['source_id'], 'source_index': case['index'],
               'image_sha256': case['image']['sha256'], 'prompt_sha256': case['prompt_sha256'],
               'stratum': case['stratum'], 'conditioning_identity': case['conditioning_identity'], 'seed': 42, 'arms': {}}
        for mode in quality.MODES:
            phase = phases[source, mode]; core = quality.base.read_bound(hf, phase['core_metrics'])['metrics']
            row['arms'][mode] = {
                'generation': phase['generation_receipt'], 'core': phase['core_metrics'],
                'native_evidence': phase['native_operand_transfer_evidence'], 'video_sha256': phase['video']['sha256'],
                'generation_origin': 'REUSED_ORIGINAL_DENSE' if mode == 'dense' else 'NEW_ACTUAL_CANDIDATE',
                'generation_seconds': phase['generation_wall_seconds_including_encoders_decode_observers'],
                'FFN_boundary_CUDA_seconds': phase['FFN_boundary_CUDA_seconds'],
                'four_infer_wall_seconds': sum(s['wall_seconds'] for s in phase['stages'] if s['stage'] == 'infer'),
                'peak_allocated_bytes': phase['peak_cuda_allocated_bytes'], 'peak_reserved_bytes': phase['peak_cuda_reserved_bytes'],
                'factor_GPU_preload_seconds': phase['current_factor_family_gpu_preload_seconds'],
                'factor_GPU_bytes': phase['current_factor_family_GPU_tensor_bytes'],
                'requested_prefetch_H2D_bytes': phase['requested_prefetch_H2D_bytes'],
                'skipped_original_W2_H2D_bytes': phase['skipped_original_W2_H2D_bytes'],
                'original_GPU_buffers_retained': phase['original_GPU_buffers_retained'],
                'order_within_pair': phase.get('order_within_pair'),
                'Farneback_all_pair_mean_flow_downsampled_px': core['temporal_motion']['downsampled_Farneback_mean_flow_px']['mean'],
                'paired_RGB_RMSE': core.get('paired_spatial', {}).get('RGB_RMSE_0to1', {}).get('mean')}
        sources.append(row)
    fields = ('generation_seconds', 'FFN_boundary_CUDA_seconds', 'four_infer_wall_seconds', 'peak_allocated_bytes',
              'peak_reserved_bytes', 'factor_GPU_preload_seconds', 'Farneback_all_pair_mean_flow_downsampled_px')
    summaries = {m: {field: stats([s['arms'][m][field] for s in sources]) for field in fields} for m in quality.MODES}
    comparisons = {}
    for mode in quality.MODES[1:]:
        comparisons[mode] = {
            'observed_generation_speed_ratio': summaries['dense']['generation_seconds']['mean']/summaries[mode]['generation_seconds']['mean'],
            'observed_FFN_boundary_speed_ratio': summaries['dense']['FFN_boundary_CUDA_seconds']['mean']/summaries[mode]['FFN_boundary_CUDA_seconds']['mean'],
            'CORE_flow_ratio_of_source_equal_means': summaries[mode][fields[-1]]['mean']/summaries['dense'][fields[-1]]['mean'],
            'paired_RGB_RMSE': stats([s['arms'][mode]['paired_RGB_RMSE'] for s in sources]),
            'all_source_CORE_flow_ratios': [{'source': s['source'], 'ratio':
                s['arms'][mode][fields[-1]]/s['arms']['dense'][fields[-1]] if s['arms']['dense'][fields[-1]] else None} for s in sources]}
    assert len(sources) == 20 and len(phases) == 60
    assert counts['video_generations'] == 40 and counts['native_full_model_flows'] == 160
    assert counts['native_FFN_calls'] == 6400 and counts['BF16_factor_GEMM'] == 12800
    assert counts['matched_reference_noise_byte_checks'] == 80 and counts['reused_Dense_registration_writes'] == 20
    return {'status': 'COMPLETE_TWENTY_SOURCES_FORTY_NEW_MATCHED_REPRESENTATION_VIDEOS',
        'quality_protocol': quality.base.binding(protocol, ROOT), 'new_native_parents': receipts,
        'reused_Dense_quality_and_native_lineage': plan['reused_Dense_quality'], 'source_cohort': plan['source_cohort'],
        'image_clusters': 20, 'new_candidate_videos': 40, 'reused_original_Dense_videos': 20,
        'new_native_FFN_calls': 6400, 'new_full_model_flows': 160, 'new_native_operation_counts': counts,
        'source_equal_arm_statistics': summaries, 'comparisons': comparisons, 'per_source': sources,
        'source_history': 'EXPOSED_DEVELOPMENT_ALL20_IMAGE_DISJOINT_FROM_NEW10_FIT_NOT_BLIND',
        'timing_identity': 'SINGLE_ORDER_DIFFERENT_NATIVE_ATTEMPT_DENSE_REUSE_INCLUDES_OBSERVERS_NOT_FAIR_WARMED_SYS',
        'motion_identity': 'ALL81_FRAME_DOWNSAMPLED_FARNEBACK_CORE_ONLY_OFFICIAL_RAFT_SEPARATE',
        'quality_or_speed_accepted': False, 'human_fullplay_review': 'PENDING', 'recompute_new_tensor_model_GPU_calls': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface')); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); result = recompute(args.protocol.resolve(), args.hf_root.resolve())
    with args.out.open('x') as stream: json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'comparisons': result['comparisons']}))


if __name__ == '__main__': main()
