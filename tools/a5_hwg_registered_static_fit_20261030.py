"""Finite exact-reviewed signed static calibration; not a video candidate.

Requires completed, independently verified matched acquisitions. This entry
point never registers itself, selects hardware, retries, or reads dev sources.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_20261030 as guard
from a5_hwg_evidence_io_20261030 import bound_json, bound_tensors, CallLedger

SELF = 'tools/a5_hwg_registered_static_fit_20261030.py'
CALIBRATION_SOURCES = [898, 1040]
CALIBRATION_KIND = 'CAL2_898_1040_SIGNED_FORWARD_GREEDY_K64_V2'
RES = {
    'GPU': 1, 'CPU_threads': 4, 'session_max_GPUs': 4,
    'GPU_reserved_max_bytes': 32 << 30, 'host_available_min_bytes': 128 << 30,
    'peak_RSS_max_bytes': 96 << 30, 'model_forecast_bytes': 64 << 30,
    'host_max_models': 2, 'new_HF_bytes': 2 << 30,
    'free_disk_min_bytes': 6 << 30, 'maximum_single_file_bytes': 512 << 20,
    'asset_validation_read_max_bytes': 192 << 30,
    'wall_seconds': 7200, 'outer_seconds': 7320, 'automatic_retry': False,
}
CALL_LIMITS = {
    'effective_W2_transfer': 80,
    'calibration_group_W2_batched': 160,
    'calibration_signed_Gram': 160,
    'calibration_signed_target_dot': 160,
    'signed_forward_greedy': 80,
    'calibration_selected_direct_W2': 160,
    'calibration_operands_H2D': 160,
}
guard.RES = RES


def authorize_common(protocol, protocol_path, job_path, resources, limits, kind, *, gpu_cards=1):
    need = guard.need
    need(protocol['resources'] == resources and protocol['call_limits'] == limits
         and protocol['kind'] == kind, 'finite scientific resources/calls/kind differ')
    active = guard.read(ROOT / 'research/ACTIVE_RESEARCH.json')
    scopes = active['experiment_authority']['current_user_authorized_scopes']
    matches = [s for s in scopes if s.get('id') == protocol['id']]
    need(len(matches) == 1, 'exact ACTIVE scope missing')
    scope = matches[0]
    need(scope['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
         and scope['protocol_sha256'] == guard.sha(protocol_path)
         and scope['job_spec_sha256'] == guard.sha(job_path), 'ACTIVE identity differs')
    need(scope['session_max_GPUs'] == 4 and scope['max_gpu_cards'] == gpu_cards,
         'applicable four-GPU session cap')
    need(sum(s.get('assigned_GPU_slots', 0) for s in scopes
             if s.get('execution_status') == 'RUNNING_FIXED_SCOPE') <= 4,
         'registered concurrent GPU cap exceeded')
    review = guard.read(guard.check_binding(ROOT, scope['review']))
    need(review['state'] == 'REVIEWER_RECHECK_APPROVED'
         and review['recheck_disposition'] == 'APPROVE'
         and review['author_agent_id'] != review['reviewer_agent_id']
         and review['scope_id'] == protocol['id']
         and review['approval_scope_fingerprint'] == guard.canonical(protocol)
         and review['job_spec_sha256'] == guard.sha(job_path), 'independent exact recheck missing')
    for row in protocol['repo_sources']:
        guard.check_binding(ROOT, row)
        need(review['critical_source_hashes'].get(row['path']) == row['sha256'],
             'review critical source differs')
    for row in review['review_evidence']:
        guard.check_binding(ROOT, row)
    need(scope['reviewed_git_commit'] == subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(), 'frozen HEAD differs')
    job = guard.read(job_path)
    need(job['id'] == protocol['id'] and job['protocol_sha256'] == guard.sha(protocol_path)
         and Path(job['protocol_path']).resolve() == protocol_path.resolve()
         and Path(job['cwd']).resolve() == ROOT, 'exact job differs')
    from worker_identity import require_worker
    require_worker()


def authorize(protocol, protocol_path, job_path):
    authorize_common(protocol, protocol_path, job_path, RES, CALL_LIMITS,
                     CALIBRATION_KIND)
    guard.need(protocol['calibration_source_ordinals'] == CALIBRATION_SOURCES
               and protocol['keep_groups'] == 80 and protocol['hidden_group'] == 64,
               'exact static scientific scope differs')


def validate_parent(source, hf, budget, expected_cache):
    need = guard.need
    producer_protocol = guard.read(guard.check_binding(ROOT, source['producer_protocol']))
    run = bound_json(hf, source['producer_RUN'], budget)
    receipt = bound_json(hf, source['producer_EXIT'], budget)
    verified = bound_json(hf, source['verifier_RUN'], budget)
    verified_exit = bound_json(hf, source['verifier_EXIT'], budget)
    for record, ref, expected_id in [(receipt, source['producer_RUN'], source['producer_id']),
                                    (verified_exit, source['verifier_RUN'], source['producer_id'] + '_VERIFY')]:
        need(record['id'] == expected_id and record['process_exited'] is True
             and record['child_returncode'] == 0 and record['timeout'] is False
             and record['automatic_retry'] is False and record['run_sha256'] == ref['sha256'],
             'parent actual EXIT0 identity')
    need(producer_protocol['id'] == source['producer_id']
         and producer_protocol['case'] == source['case']
         and producer_protocol['cache'] == expected_cache
         and run['protocol_sha256'] == receipt['protocol_sha256']
         == verified['protocol_sha256'] == verified_exit['protocol_sha256']
         == source['producer_protocol']['sha256']
         and receipt['spec_sha256'] == source['producer_job_spec_sha256']
         and verified_exit['spec_sha256'] == source['verifier_job_spec_sha256'],
         'parent protocol/cache/job binding differs')
    need(run['status'] == 'COMPLETE_MATCHED_FULL160_CAPTURE_EXIT_PENDING'
         and run['ten_states_and_video_neutral'] is True
         and verified['status'] == 'PASS_INDEPENDENT_FILE_AND_RAW_SHA_FULL160_EXIT_PENDING'
         and verified['matched_fullM_sample_sites'] == 160
         and verified['producer_RUN'] == source['producer_RUN']
         and verified['producer_exit'] == source['producer_EXIT'], 'parent independent closure differs')
    need(run['source'] == source['case'] and run['source']['seed'] == 42
         and len(source['metadata']) == 160, 'parent source/seed/full160 differs')
    validate_metadata_manifest(source, run, hf, budget)


def validate_metadata_manifest(source, run, hf, budget):
    """Bind the exact refs verified under the original producer RUN chain."""
    need = guard.need
    need(len(run['phases']) == 2 and [v['arm'] for v in run['phases']] == ['plain', 'observed'],
         'producer original phase identity')
    manifest = bound_json(hf, run['phases'][1]['capture_manifest'], budget)
    need(manifest['status'] == 'PASS_CAPTURE_STRUCTURE_ONLY'
         and manifest['full_V1_files'] == manifest['completed_scopes'] == 160
         and len(manifest['scopes']) == 160, 'original full160 manifest')
    original = {}
    for packet in manifest['scopes']:
        step, block = packet['step'], packet['block']
        need(type(step) is int and 0 <= step < 4 and type(block) is int and 0 <= block < 40,
             'original manifest site bounds')
        key = f'step{step}_block{block:02d}'
        need(key not in original, 'duplicate original manifest site')
        original[key] = packet['metadata']
    need(source['metadata'] == original, 'metadata refs differ from independently verified producer manifest')


@guard.execution_bootstrap('run')
def run(args):
    need = guard.need
    started = guard.CURRENT_BOOTSTRAP.started
    budget = guard.CURRENT_BOOTSTRAP.budget
    p = guard.read(args.protocol)
    authorize(p, args.protocol, args.job_spec)
    private = guard.private_assets(p, args.private_assets)
    need(sorted(os.sched_getaffinity(0)) == p['cpu_affinity'], 'registered CPU affinity')
    need(len(p['sources']) == 2 and [s['case']['native_source_ordinal'] for s in p['sources']] == CALIBRATION_SOURCES,
         'only two frozen calibration sources')
    need(len({s['case']['image']['sha256'] for s in p['sources']}) == 2,
         'two distinct original image clusters')
    hf = Path(private['hf_root']).resolve()
    out = hf / p['output_HF_relative']
    need(not out.exists(), 'single attempt output already exists')
    out.mkdir()
    report = {'id': p['id'], 'protocol_sha256': guard.sha(args.protocol), 'status': 'RUNNING',
              'numeric_family': 'HIF4_CAPTURED_POSTQ2_AND_EFFECTIVE_W2',
              'kind': p['kind'], 'completed_sites': 0, 'automatic_retry': False,
              'calibration_source_ordinals': CALIBRATION_SOURCES, 'dev_tensor_files_read': 0,
              'new_video_or_native_FFN_forward_calls': 0,
              'is_global_optimal_support': False, 'is_deployable_method': False}
    monitor = None
    def check():
        need(time.monotonic() - started <= RES['wall_seconds'], 'static fit wall budget')
        need(guard.available_memory() >= RES['host_available_min_bytes'], 'static fit host reserve')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= RES['peak_RSS_max_bytes'], 'static fit RSS')
        need(shutil.disk_usage(out).free >= 4 << 30, 'static fit disk reserve')
        if monitor: monitor.check()
        if monitor and 'torch' in sys.modules:
            need(sys.modules['torch'].cuda.max_memory_reserved() <= RES['GPU_reserved_max_bytes'], 'static fit GPU reserve')
    writer = guard.Writer(out, hf, check)
    budget.check = check
    persist = lambda: writer.json(out / 'RUN.json', report)
    persist()
    try:
        with guard.HostReservation(), guard.bounded_files():
            need(shutil.disk_usage(out).free >= RES['free_disk_min_bytes'], 'initial static fit disk budget')
            for name, expected in p['runtime_versions'].items():
                need(importlib.metadata.version(name) == expected, 'runtime package differs: ' + name)
            import torch
            torch.set_num_threads(4)
            torch.set_num_interop_threads(4)
            torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            from a5_hwg_static_signed_math_20261030 import group_quadratic, greedy_support
            from startup_gpu_reservation import reserve_before_assets
            reservation, monitor, report['CUDA_reservation'] = reserve_before_assets(torch)
            for source in p['sources']: validate_parent(source, hf, budget, p['cache'])
            cache = Path(private['cache_path'])
            need(cache.stat().st_size == p['cache']['bytes'] and guard.sha(cache) == p['cache']['sha256'], 'actual cache changed')
            writer.note('CACHE_MMAP_LOAD_BEGIN_INTERNAL_IO_UNKNOWN', declared_file_bytes=p['cache']['bytes'])
            cache_values = torch.load(cache, map_location='cpu', weights_only=False, mmap=True)
            writer.note('CACHE_MMAP_LOAD_RETURN_INTERNAL_IO_UNKNOWN', status='COMPLETE')
            report['cache_loader'] = {'calls': 1, 'mmap': True, 'internal_file_IO_bytes': 'UNKNOWN_SEPARATE_FROM_EXPLICIT_SHA_READS'}
            ledger = CallLedger(CALL_LIMITS, writer.note, check, torch)
            result_masks, orders, traces, sites = [], [], [], []
            seen = [{'paths': set(), 'inodes': set()} for _ in p['sources']]
            w2_raw_hash_bytes = 0
            for expert_index, expert in enumerate(['high_noise_model', 'low_noise_model']):
                steps = [0, 1] if expert_index == 0 else [2, 3]
                for block in range(40):
                    check()
                    ledger.context = {'expert': expert, 'block': block}
                    weight = cache_values[expert][f'blocks.{block}.ffn.2']['weight_qdq']
                    need(weight.dtype == torch.bfloat16 and list(weight.shape) == [5120, 13824], 'effective W2 geometry')
                    weight_hash = writer.raw_sha(weight.contiguous())
                    w2_raw_hash_bytes += weight.numel() * weight.element_size()
                    w2 = ledger('effective_W2_transfer', lambda: weight.cuda(), shape=[5120, 13824], dtype='bfloat16')
                    source_grams, source_correlations, source_metrics, direct_operands = [], [], [], []
                    for source_index, source in enumerate(p['sources']):
                        packets = []
                        for step in steps:
                            meta_ref = source['metadata'][f'step{step}_block{block:02d}']
                            meta = bound_json(hf, meta_ref, budget)
                            packet = {'step': step, 'block': block, 'metadata': meta_ref}
                            guard.validate_packet(meta, packet, {'case': source['case'], 'cache': p['cache']},
                                                  hf, hf / source['results_HF_relative'], seen[source_index])
                            need(meta['effective_W2']['runtime_operand_sha256'] == weight_hash, 'actual captured/cache W2 mismatch')
                            packets.append(bound_tensors(hf, meta['payload'], budget, meta['tensor_schema']))
                        old, current = packets
                        ledger.context = {'source_ordinal': source['case']['native_source_ordinal'],
                                          'expert': expert, 'block': block,
                                          'rows': 1344, 'grouped_W2_shape': [216, 1344, 64, 5120],
                                          'Gram_shape': [216, 216, 1344 * 5120]}
                        delta_cpu = current['V2'].float() - old['V2'].float()
                        target_cpu = current['Y'].float() - old['Y'].float()
                        direct_operands.append((delta_cpu, target_cpu, current['gate_broadcast']))
                        delta, target, gate, reference = ledger('calibration_operands_H2D', lambda:
                            (delta_cpu.cuda(), target_cpu.cuda(), current['gate_broadcast'].cuda(), current['Y'].cuda()),
                            rows=1344, hidden=13824, output_width=5120)
                        gram, corr, info = group_quadratic(delta, w2, target, gate, reference, measured=ledger)
                        source_grams.append(gram.cpu())
                        source_correlations.append(corr.cpu())
                        source_metrics.append(info)
                        del packets, old, current, delta, target, gram, corr, gate, reference
                    gram = (source_grams[0] + source_grams[1]) / 2
                    corr = (source_correlations[0] + source_correlations[1]) / 2
                    ledger.context = {'expert': expert, 'block': block, 'independent_calibration_sources': 2}
                    support, order, gains = ledger('signed_forward_greedy', lambda: greedy_support(gram, corr, 80))
                    for si, (delta_cpu, target_cpu, gate_cpu) in enumerate(direct_operands):
                        def direct_check():
                            import torch.nn.functional as F
                            delta, target, gate = delta_cpu.cuda(), target_cpu.cuda(), gate_cpu.float().cuda().reshape(1, -1)
                            selected = support.repeat_interleave(64).cuda()
                            projected = F.linear(delta * selected, w2.float())
                            direct = float(((projected - target) * gate).double().square().sum()) / source_metrics[si]['current_gated_output_squared_norm']
                            sg, sc = source_grams[si], source_correlations[si]
                            quadratic = (source_metrics[si]['target_gated_change_squared_norm'] / source_metrics[si]['current_gated_output_squared_norm']
                                         - 2 * float(sc[support].sum()) + float(sg[support][:, support].sum()))
                            return {'direct_selected_FP32_W2_normalized_SSE': direct,
                                    'rounded_Gram_quadratic_normalized_SSE': quadratic,
                                    'signed_Gram_direct_closure_residual': quadratic - direct,
                                    'exact_objective': 'MEASURED_ROUNDED_FP32_GRAM_NOT_REAL_ARITHMETIC_OPTIMUM'}
                        source_metrics[si].update(ledger('calibration_selected_direct_W2', direct_check,
                            source_ordinal=p['sources'][si]['case']['native_source_ordinal'], shape=[1344, 13824, 5120]))
                    artifact = writer.tensor(out / f'{expert}_block{block:02d}.safetensors', {
                        'source_Gram': torch.stack(source_grams),
                        'source_correlation': torch.stack(source_correlations),
                        'source_equal_Gram': gram, 'source_equal_correlation': corr,
                        'selected_K64_groups': support,
                        'greedy_order': torch.tensor(order, dtype=torch.int32),
                        'marginal_gains': torch.tensor(gains, dtype=torch.float64),
                    })
                    sites.append({'expert': expert, 'block': block, 'artifact': artifact, 'source_metrics': source_metrics})
                    result_masks.append(support)
                    orders.append(torch.tensor(order, dtype=torch.int32))
                    traces.append(torch.tensor(gains, dtype=torch.float64))
                    report['completed_sites'] += 1
                    persist()
                    del gram, corr, w2, source_grams, source_correlations, direct_operands
            need(ledger.counts == CALL_LIMITS and all(len(v['paths']) == 480 for v in seen), 'exact all80 calibration coverage/calls')
            support_tensors = {
                'selected_groups': torch.stack(result_masks).reshape(2, 40, 216),
                'greedy_order': torch.stack(orders).reshape(2, 40, 80),
                'marginal_gains': torch.stack(traces).reshape(2, 40, 80),
            }
            report['static_support'] = writer.tensor(out / 'STATIC_K64_SUPPORT.safetensors', support_tensors)
            report['static_support_schema'] = {name: {'shape': list(tensor.shape),
                'dtype': str(tensor.dtype).removeprefix('torch.'),
                'raw_tensor_sha256': writer.raw_sha(tensor)} for name, tensor in support_tensors.items()}
            report['site_manifest'] = sites
            report['call_counts'] = ledger.counts
            report['call_records'] = ledger.records
            report['W2_explicit_raw_SHA_memory_bytes'] = w2_raw_hash_bytes
            report['status'] = 'COMPLETE_CAL2_SIGNED_GREEDY_FULL80_EXIT_PENDING'
    except BaseException as error:
        report.update(status='FAILED_STATIC_CALIBRATION_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        guard.CONTROL_READ = None
        signal.alarm(0)
        writer.check = lambda: None  # Bounded postmortem metadata, no new numerical work.
        writer.finish_pending()
        if monitor:
            monitor.close()
            report['assigned_GPU_monitor'] = monitor.report()
            if report['assigned_GPU_monitor']['violations']:
                report['status'] = 'FAILED_ASSIGNED_GPU_MONITOR_PRESERVED_NO_RETRY'
        report.update(asset_and_payload_explicit_read_ledger=budget.summary(),
                      input_explicit_memory_raw_SHA_bytes=getattr(budget, 'explicit_memory_raw_SHA_bytes', 0),
                      wall_seconds=time.monotonic() - started,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      output_accounting=writer.summary())
        if 'torch' in sys.modules:
            report['peak_CUDA_reserved_bytes'] = sys.modules['torch'].cuda.max_memory_reserved()
        persist()
        writer.journal.close()
        if monitor: monitor.check()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--job-spec', type=Path, required=True)
    ap.add_argument('--private-assets', type=Path, required=True)
    args = ap.parse_args()
    with guard.bounded_files(): run(args)


if __name__ == '__main__': main()
