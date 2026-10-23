"""CPU readout of the completed five-warmup/25-timing prefix; no model run."""
import json
import os
from pathlib import Path

from analyze_fresh_quality_results import latency_summary, write_once
from fresh_quality_contract import ARMS, METHOD, ROOT, generation_jobs, read, sha


def analyze(hf):
    root = hf / 'derived/ffnds/R17E0_fresh_quality'
    gen = read(root / 'GENERATIONS.json')
    prefix = gen['completed'][:30]
    if len(prefix) != 30 or [r['job'] for r in prefix] != generation_jobs()[:30]:
        raise RuntimeError('Wait for the complete original fivewarm/25balancedtiming prefix')
    original = {r['arm']: r for r in read(ROOT / 'research/20261023/R16V0_native_results.json')['requests']}
    requests = {}
    expected = {(s, b) for s in range(4) for b in range(40)}
    receipts = []
    for item in prefix:
        path = hf / item['request_HF_relative']
        if sha(path) != item['request_sha256']:
            raise RuntimeError('Original actual warm/timing receipt changed')
        record = read(path)
        job, arm = record['job'], record['job']['arm']
        if (record['initial_noise_sha256'] != original[arm]['initial_noise_sha256'] or
                [r['sha256'] for r in record['trajectory']] != [r['sha256'] for r in original[arm]['trajectory']] or
                record['actual_native_counter_delta'] != {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600,
                    'cache_misses': 0, 'weight_transfers': 0} or
                record['whole_generation_performance_benchmark'] != (job['phase'] == 'timing')):
            raise RuntimeError('Warm/repeated native trajectory, coverage or performance scope differs')
        timing = record['native_FFN_timing']
        for kind in ('shared_modulation_preparation', 'FFN_norm_through_gate_residual'):
            regions = [r for r in timing['rows'] if r['kind'] == kind]
            if len(regions) != 160 or {(r['step'], r['block']) for r in regions} != expected:
                raise RuntimeError('Missing complete native event boundaries')
        if (timing['instrumentation_per_block_device_synchronization'] or
                timing['instrumentation_CPU_tensor_reads_during_request'] or
                timing['operator_internal_synchronization_excluded_from_cost']):
            raise RuntimeError('Native timing boundary differs')
        requests[job['name']] = record
        receipts.append({'job': job, 'original_HF_relative': item['request_HF_relative'],
                         'original_sha256': item['request_sha256'],
                         'generation_wall_seconds': record['generation_wall_seconds'],
                         'core_cuda_ms': timing['native_core_cuda_ms'],
                         'shared_modulation_cuda_ms': timing['shared_modulation_cuda_ms'],
                         'peak_allocated_bytes': record['peak_allocated_bytes']})
    results = {}
    for arm in ARMS:
        rows = [requests[f'timing{i}_{arm}'] for i in range(5)]
        results[arm] = {'complete_generation_wall_seconds': latency_summary([r['generation_wall_seconds'] for r in rows]),
                       'full160_FFN_core_cuda_ms': latency_summary([r['native_FFN_timing']['native_core_cuda_ms'] for r in rows]),
                       'full160_FFN_including_all_shared_modulation_cuda_ms': latency_summary([
                           r['native_FFN_timing']['native_core_cuda_ms'] + r['native_FFN_timing']['shared_modulation_cuda_ms'] for r in rows])}
    ratios = {arm: {scope: results['dense_DQ'][scope]['median'] / results[arm][scope]['median']
                   for scope in results[arm]} for arm in ARMS[1:]}
    return {'id': METHOD, 'status': 'ACTUAL_FIXED_SOURCE_COUNTERBALANCED_TIMING_PREFIX_COMPLETE',
        'source_sha256': sha(__file__), 'original_protocol_sha256': sha(ROOT / 'research/20261023/R17E0_registered_protocol.json'),
        'actual_complete_warmups': 5, 'actual_timing_generations': 25,
        'technical_repeats_per_arm': 5, 'independent_performance_sources': 1,
        'source_identity': 'old_calibration24_source0_seed42', 'GPU': 'NVIDIA A10080GBPCIe',
        'precision': 'Original HiF4 activation/weightQDQ with effective BF16 native linear arithmetic; not a packedINT4GEMM throughput claim.',
        'boundary': 'Complete pipe.generate through encoded video save and final request synchronization; frozen operator/selector/QDQ/gather/scatter, latentGPUclones/events charged. Budget bookkeeping, static mask initialization, afterrequest CPUhashes/routebitpack excluded. Model/checkpoint/runtime resident and actual warmups precede all25timings.',
        'per_arm': results, 'speed_ratios_dense_over_arm': ratios,
        'original_30_actual_request_receipts': receipts,
        'new_model_forwards_from_this_analysis': 0,
        'fresh_quality_result': None, 'paper_or_Pareto_acceptance': False,
        'complete_batch_resource_audit_pending': True,
        'interpretation': 'This is the completed true timing prefix, not the final145generation/45metric outcome. Per-request sampled-isolation checks passed in the producer; its complete-session hardware/resource receipt is produced at generator exit. Repetitions measure technical variation on one old source, not independent population speed.'}


def main():
    hf = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    result = analyze(hf)
    target = ROOT / 'research/20261023/R17E0_system_timing_prefix.json'
    write_once(target, result)
    print(result['status'])
    for arm, summary in result['per_arm'].items():
        print(arm, 'median generation seconds', summary['complete_generation_wall_seconds']['median'],
              'median native FFN ms', summary['full160_FFN_core_cuda_ms']['median'])
    print(json.dumps(result['speed_ratios_dense_over_arm'], indent=2))


if __name__ == '__main__':
    main()
