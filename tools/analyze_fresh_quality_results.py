"""Reconstruct fresh-source official scores and repeated native system timing."""
import math
import os
from pathlib import Path
import re
import statistics

from fresh_quality_contract import ARMS, CAPS, CORE, METHOD, ROOT, atomic, generation_jobs, read, sha
from fresh_quality_statistics import normalized_video_score, paired_sources, quantile, verify_official_aggregate


def latency_summary(values):
    if len(values) != 5 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise RuntimeError('Exactly five positive actual timed repetitions required')
    return {'all_five_values': values, 'median': statistics.median(values),
            'p95_interpolated': quantile(values, .95), 'minimum': min(values),
            'maximum': max(values), 'mean': statistics.fmean(values),
            'median_absolute_deviation': statistics.median(abs(v - statistics.median(values)) for v in values)}


def write_once(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Preserve a prior analysis, not a silently changed aggregation')
    else:
        atomic(path, value)


def analyze(hf):
    root = hf / 'derived/ffnds/R17E0_fresh_quality'
    gen_path, decode_path, metric_path = [root / x for x in ('GENERATIONS.json', 'VIDEO_DECODE.json', 'metrics/RUN.json')]
    gen, decode, metrics, budget = [read(x) for x in (gen_path, decode_path, metric_path, root / 'BUDGET.json')]
    if (gen['status'] != 'COMPLETE_145_GENERATIONS_AWAITING_FRESH_OFFICIAL_METRICS' or
            metrics['status'] != 'COMPLETE_FIXED23_SOURCE_NINE_DIMENSION_FIVE_ARM_EVALUATION' or
            decode['status'] != 'COMPLETE_ALL115_FRESH_VIDEOS_ALL81_FRAMES_DECODED_NO_VISUAL_VERDICT' or
            budget['used'] != CAPS or len(budget['debits']) != 190 or
            any(r['status'] != 'COMPLETED' for r in budget['debits']) or
            decode['generations_sha256'] != sha(gen_path) or metrics['generation_receipt_sha256'] != sha(gen_path)):
        raise RuntimeError('Incomplete original finite generation/decode/evaluation chain')
    if [r['job'] for r in gen['completed']] != generation_jobs():
        raise RuntimeError('Actual executed source/action/order list differs')
    expected_sites = {(s, b) for s in range(4) for b in range(40)}
    requests, compact = {}, []
    for item in gen['completed']:
        path = hf / item['request_HF_relative']
        if sha(path) != item['request_sha256']:
            raise RuntimeError('Original actual generation receipt changed')
        record = read(path)
        job = record['job']
        if (record['actual_native_counter_delta'] != {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600,
                'cache_misses': 0, 'weight_transfers': 0} or not record['free_student_states_no_reset'] or
                [r['step'] for r in record['trajectory']] != list(range(4))):
            raise RuntimeError('Original native free-trajectory/control coverage differs')
        events = record['dispatch_events']
        if job['arm'] != 'dense_DQ' and (len(events) != 160 or
                {(e['step'], e['block']) for e in events} != expected_sites):
            raise RuntimeError('Candidate did not execute all160 actual FFNs')
        timing = record['native_FFN_timing']
        for kind in ('shared_modulation_preparation', 'FFN_norm_through_gate_residual'):
            rows = [r for r in timing['rows'] if r['kind'] == kind]
            if len(rows) != 160 or {(r['step'], r['block']) for r in rows} != expected_sites:
                raise RuntimeError('Actual full160 event timing coverage differs')
        if (timing['instrumentation_per_block_device_synchronization'] or
                timing['instrumentation_CPU_tensor_reads_during_request'] or
                timing['operator_internal_synchronization_excluded_from_cost']):
            raise RuntimeError('Native performance observation/cost boundary changed')
        requests[job['name']] = record
        compact.append({k: record[k] for k in ('job', 'grid', 'initial_noise_sha256', 'trajectory',
            'source_image_sha256', 'prompt_sha256', 'video_HF_relative', 'video_sha256',
            'generation_wall_seconds', 'whole_generation_performance_benchmark', 'peak_allocated_bytes', 'routing_trace')} |
            {'original_request_HF_relative': item['request_HF_relative'], 'original_request_sha256': item['request_sha256'],
             'actual_FFN_dispatch_sites': len(events), 'actual_native_counter_delta': record['actual_native_counter_delta'],
             'native_core_cuda_ms': timing['native_core_cuda_ms'], 'shared_modulation_cuda_ms': timing['shared_modulation_cuda_ms']})
    timing = {}
    for arm in ARMS:
        rows = [requests[f'timing{i}_{arm}'] for i in range(5)]
        if any(not r['whole_generation_performance_benchmark'] for r in rows):
            raise RuntimeError('Warm/quality observations cannot be timing repetitions')
        timing[arm] = {'full_generation_wall_seconds': latency_summary([r['generation_wall_seconds'] for r in rows]),
                      'full160_FFN_core_cuda_ms': latency_summary([r['native_FFN_timing']['native_core_cuda_ms'] for r in rows]),
                      'full160_FFN_plus_all_shared_modulation_cuda_ms': latency_summary([
                          r['native_FFN_timing']['native_core_cuda_ms'] + r['native_FFN_timing']['shared_modulation_cuda_ms'] for r in rows])}
    speed = {}
    for arm in ARMS[1:]:
        speed[arm] = {scope: timing['dense_DQ'][scope]['median'] / timing[arm][scope]['median'] for scope in timing[arm]}
        speed[arm]['paired_Latin_cycle_wall_speed_ratios'] = [
            requests[f'timing{i}_dense_DQ']['generation_wall_seconds'] / requests[f'timing{i}_{arm}']['generation_wall_seconds'] for i in range(5)]
    decoded = {(s['source'], arm): video for s in decode['sources'] for arm, video in s['videos'].items()}
    sources = read(ROOT / 'research/20261023/R17_fresh23_input_audit.json')['inputs']
    scores, original_official_records = {}, []
    for item in metrics['completed']:
        path = hf / item['result_HF_relative']
        if sha(path) != item['sha256']:
            raise RuntimeError('Original official evaluator receipt changed')
        record = read(path)
        dimension, arm = record['dimension'], record['arm']
        expected = {s['index'] for s in sources if dimension in s['core_dimensions']}
        details = record['official_return'][-1]
        values, counts = {}, {}
        for detail in details:
            match = re.match(r'^source(\d+)(?: |\.mp4)', Path(detail['video_path']).name)
            if match is None:
                raise RuntimeError('Unbound official video/source identity')
            source = int(match[1])
            if source in values:
                raise RuntimeError('Copied filenames cannot be additional sources')
            values[source] = normalized_video_score(dimension, detail)
            counts[detail['video_path']] = decoded[source, arm]['decoded_frames']
        if set(values) != expected or set(record['source_indices']) != expected:
            raise RuntimeError('Official dimension source population differs')
        verify_official_aggregate(dimension, record['official_return'], counts)
        if (dimension, arm) in scores:
            raise RuntimeError('Duplicate original evaluator call')
        scores[dimension, arm] = values
        original_official_records.append(record)
    if set(scores) != {(d, a) for d in CORE for a in ARMS}:
        raise RuntimeError('Require every original official nine-dimensional/five-arm call')
    quality = {}
    for dimension in CORE:
        means = {arm: statistics.fmean(scores[dimension, arm].values()) for arm in ARMS}
        quality[dimension] = {'source_indices': sorted(scores[dimension, 'dense_DQ']),
            'equal_source_means_in_official_aggregate_units': means,
            'per_source_scores_in_official_aggregate_units': {arm: scores[dimension, arm] for arm in ARMS},
            'dynamic_paired_to_each_control': {arm: paired_sources(scores[dimension, arm], scores[dimension, 'dynamic_residual']) for arm in ARMS[:-1]},
            'higher_raw_score_is_not_by_itself_a_joint_motion_quality_verdict': True}
    report = {'id': METHOD, 'status': 'COMPLETE_FIXED_SCOPE_ACTUAL_QUALITY_AND_REPEATED_NATIVE_TIMING',
        'analysis_source_sha256': sha(__file__), 'inputs_sha256': {'GENERATIONS': sha(gen_path),
            'VIDEO_DECODE': sha(decode_path), 'OFFICIAL_METRICS': sha(metric_path), 'BUDGET': sha(root / 'BUDGET.json')},
        'actual_budget': budget['used'], 'underlying_fresh_sources': 23,
        'timing_scope': 'Five counterbalanced repetitions on one old source after five actual warmups. Technical variation only; not23independent speed samples. Native observer/latent clones and operator internal synchronization charged.',
        'timing': timing, 'speed_ratios_dense_over_arm': speed, 'official_nine_quality': quality,
        'algorithm_changed_or_refit': False, 'strong_fixed_controls': ['fixed_stride', 'fixed_hash'],
        'official_full_dataset_benchmark': False, 'independent_source_statistical_unit': 'conditioning_image',
        'missing_comparators': 'Closest optimized/public prior methods under exactly matched model/quantization/scheduler/hardware remain unmeasured.',
        'automatic_quality_or_Pareto_or_paper_acceptance': False}
    return report, compact, original_official_records


def main():
    hf = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    root = hf / 'derived/ffnds/R17E0_fresh_quality'
    report, compact, official = analyze(hf)
    write_once(root / 'ANALYSIS.json', report)
    write_once(root / 'GENERATION_RECEIPTS_COMPACT.json', {'id': METHOD, 'records': compact,
        'original_per_boundary_event_and_dispatch_records_retained_in_HF': True})
    write_once(root / 'OFFICIAL_RECORDS_COMPLETE.json', {'id': METHOD, 'records': official})
    print(report['status'], flush=True)
    for dimension, row in report['official_nine_quality'].items():
        print(dimension, row['equal_source_means_in_official_aggregate_units'], flush=True)
    print('Actual median speed ratios:', report['speed_ratios_dense_over_arm'], flush=True)


if __name__ == '__main__':
    main()
