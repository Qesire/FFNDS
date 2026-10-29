"""Reuse the completed environment and allow measured cold-cache SHA throughput.

The failed 300-second proof stays immutable. This separately registered CPU
scope reads the four masters afresh, copies no files and reports actual time.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import importlib.util
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('_s2_causal_v5_mirror', ROOT/'tools/raw_bf16_worker_mirror_proof.py')
mirror = importlib.util.module_from_spec(spec); spec.loader.exec_module(mirror)
native = mirror.native
ID = 'E0_S2_CAUSAL_ADDITIONAL_IDLE_A100_ENV_MASTER_CPU_V5_20261029'
OUTPUT = 'ffnds-derived/e0_s2_causal_additional_idle_a100_env_master_cpu_v5_20261029/attempt1/worker/results'
PRIOR = 'config/e0_s2_causal_idle_a100_master_cpu_20261029_v4.json'
TEMPLATE = 'config/e0_s2_new12_dense_baseline_part0_20261029_v2.json'
TOTAL = mirror.TOTAL
RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'hash_threads': 2, 'wall_seconds': 1800,
             'outer_seconds': 2100, 'new_HF_bytes': 8 << 20, 'new_local_bytes': 0,
             'automatic_retry': False}
LIMITS = {'full_payload_bytes': TOTAL, 'models': 4, 'native_headers': 2,
          'package_files_verified': 2302, 'package_bytes_verified': 42053803,
          'metadata_bytes_verified': 21454628, 'copied_bytes': 0,
          'model_download_bytes': 0, 'GPU_calls': 0, 'model_calls': 0}
SOURCES = ('tools/e0_s2_causal_worker_proof_v5.py', 'tests/test_e0_s2_causal_worker_proof_v5.py',
           'tools/raw_bf16_worker_mirror_proof.py', 'tests/test_raw_bf16_worker_mirror_proof.py',
           'tools/e0_original_bf16_videos.py', 'tools/worker_identity.py', PRIOR, TEMPLATE)
mirror.ID, mirror.RESOURCES, mirror.LIMITS = ID, RESOURCES, LIMITS


def draft():
    plan = mirror.draft(); old = native.read(ROOT/PRIOR)
    hf = Path('/home/xiang/.cache/huggingface'); out = hf/old['output_HF_relative']
    plan.update(output_HF_relative=OUTPUT, resources=RESOURCES, limits=LIMITS,
        completed_environment={'protocol': native.binding(ROOT/PRIOR, ROOT),
            'RUN': native.binding(out/'RUN.json', hf), 'environment': native.binding(out/'ENVIRONMENT.json', hf),
            'exit': native.binding(out.parent/'EXIT_RECEIPT.json', hf)},
        local_package=old['local_package'], local_root=old['local_root'],
        model_view_root=native.read(ROOT/TEMPLATE)['model_view_root'],
        model_view_metadata=native.read(ROOT/TEMPLATE)['model_view_metadata'],
        repo_sources=[native.binding(ROOT/p, ROOT) for p in sorted(set(SOURCES) | {b['path'] for b in old['repo_sources']})],
        execution_identity='New full SHA on actual worker with measured cold-cache budget; reuse only completed environment, never partial hashes')
    return plan


def validate(plan):
    expected = draft()
    expected['status'] = plan.get('status')
    expected['private_worker_config_sha256'] = plan.get('private_worker_config_sha256')
    native.need(plan == expected and plan['status'] in ('DRAFT_NOT_REGISTERED', 'FROZEN_REGISTERED'), 'Exact V5 scope differs')
    value = plan['private_worker_config_sha256']
    native.need(value is None or isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef'), 'Malformed worker binding')
    if plan['status'] == 'FROZEN_REGISTERED': native.need(value is not None, 'Actual worker not bound')


def reuse_environment(plan, hf):
    p = plan['completed_environment']; prior = native.read(native.bound(ROOT, p['protocol']))
    run = native.read(native.bound(hf, p['RUN'])); env = native.read(native.bound(hf, p['environment']))
    receipt = native.read(native.bound(hf, p['exit']))
    native.need(prior['private_worker_config_sha256'] == plan['private_worker_config_sha256']
        and prior['models'] == plan['models'] and receipt['id'] == prior['id']
        and receipt['protocol_sha256'] == p['protocol']['sha256'] and receipt['run_sha256'] == p['RUN']['sha256']
        and receipt['process_exited'] is True and type(receipt['child_returncode']) is int
        and receipt['child_returncode'] == 1 and receipt['timeout'] is False
        and run['status'] == 'FAILED_MIRROR_PROOF_PRESERVED_NO_RETRY'
        and run['error'] == 'Fresh full SHA incomplete or timed out'
        and 300 <= run['wall_seconds'] < 310
        and run['actual_full_payload_read_bytes'] == 24175968256
        and env['status'] == 'PASS_BYTE_IDENTICAL_LOCAL_PACKAGE_AND_HF_ALIAS'
        and env['copied_files'] == 2302 and env['copied_bytes'] == 42053803
        and env['GPU_calls'] == env['model_calls'] == env['model_download_bytes'] == 0,
        'Actual V4 completed environment and timed-out SHA lineage differ')
    native.model_metadata(plan)
    info = plan['local_package']; inventory = native.read(native.bound(hf, info['inventory']))
    rows = inventory['files']; root = Path(plan['local_root'])/'transformers'
    native.need(len(rows) == LIMITS['package_files_verified']
        and sum(r['bytes'] for r in rows) == LIMITS['package_bytes_verified']
        and native.canonical(rows) == info['inventory_sha256'], 'Frozen package inventory differs')
    for row in rows: native.bound(root, row)
    return {'prior': p, 'metadata_files_verified': 7, 'package_files_verified': len(rows),
            'metadata_bytes_verified': LIMITS['metadata_bytes_verified'],
            'package_bytes_verified': LIMITS['package_bytes_verified'], 'copied_bytes': 0}


def run_proof(plan, out, digest, started=None):
    start = time.monotonic() if started is None else started
    report = {'id': ID, 'protocol_sha256': digest, 'status': 'STARTED_FRESH_SHA_NO_RETRY',
        'parent_protocol_sha256': mirror.PARENT_SHA, 'models_sha256': plan['models_sha256'],
        'private_worker_config_sha256': plan['private_worker_config_sha256'],
        'model_payloads': [], 'predebited_full_payload_bytes': 0, 'actual_full_payload_read_bytes': 0,
        'native_headers': {}, 'all_CUDA_devices_hidden': True, 'NumPy_or_Torch_imported': False,
        'GPU_calls': 0, 'model_calls': 0, 'automatic_retry': False, 'copied_bytes': 0,
        'wall_budget_seconds': RESOURCES['wall_seconds']}
    def persist(): native.write(out/'RUN.json', report)
    persist(); stop = threading.Event(); timed_out = False
    def timeout(*_):
        nonlocal timed_out
        timed_out = True; stop.set()
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1, int(RESOURCES['wall_seconds'] - (time.monotonic()-start))))
    try:
        pending = iter(plan['models'])
        with ThreadPoolExecutor(max_workers=RESOURCES['hash_threads']) as pool:
            futures = {}
            def submit(item):
                report['predebited_full_payload_bytes'] += item['bytes']
                report['model_payloads'].append({'path': item['path'], 'kind': item['kind'],
                    'expected_sha256': item['sha256'], 'predebited_bytes': item['bytes'],
                    'actual_read_bytes': 0, 'status': 'PREDEBITED_STARTED'})
                persist(); futures[pool.submit(native.full_hash, item, stop)] = item
            for _ in range(RESOURCES['hash_threads']): submit(next(pending))
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    item = futures.pop(future); result = future.result(); result['kind'] = item['kind']
                    index = next(i for i, x in enumerate(report['model_payloads']) if x['path'] == item['path'])
                    report['model_payloads'][index] = result
                    report['actual_full_payload_read_bytes'] = sum(x['actual_read_bytes'] for x in report['model_payloads'])
                    persist()
                    if result['status'] != 'PASS_FRESH_FULL_SHA': stop.set()
                    if not stop.is_set():
                        item = next(pending, None)
                        if item is not None: submit(item)
        native.need(not timed_out and not stop.is_set() and len(report['model_payloads']) == 4
            and report['predebited_full_payload_bytes'] == report['actual_full_payload_read_bytes'] == TOTAL,
            'Fresh full SHA incomplete or timed out')
        for item in plan['models']:
            if item['kind'] in ('high', 'low'):
                row = next(x for x in report['model_payloads'] if x['path'] == item['path'])
                native.need(native.stat_record(item['path']) == row['stat'], 'Master changed before header')
                h = native.raw_header_check(native.header(item['path']))
                native.need(h['header_sha256'] == item['header_sha256'] and native.stat_record(item['path']) == row['stat'], 'Raw header or stat changed')
                report['native_headers'][item['kind']] = {k: h[k] for k in ('header_sha256', 'header_bytes', 'tensor_count')}
        native.need(not timed_out and time.monotonic()-start <= RESOURCES['wall_seconds']
            and not {'torch', 'numpy'} & set(sys.modules), 'Registered V5 wall or CPU-only contract exceeded')
        report['status'] = 'PASS_FRESH_WORKER_RAW_SHA_STAT_HEADER_NO_MODEL_NO_NUMPY_NO_TORCH'
    except BaseException as exc:
        stop.set(); report.update(status='FAILED_MIRROR_PROOF_PRESERVED_NO_RETRY', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        report['wall_seconds'] = time.monotonic()-start; persist()
    return report


def verify_report(plan, protocol, run):
    expected = {'id': ID, 'protocol_sha256': protocol['sha256'],
        'status': 'PASS_FRESH_WORKER_RAW_SHA_STAT_HEADER_NO_MODEL_NO_NUMPY_NO_TORCH',
        'parent_protocol_sha256': mirror.PARENT_SHA, 'models_sha256': native.canonical(plan['models']),
        'private_worker_config_sha256': plan['execution_private_worker_config_sha256'],
        'predebited_full_payload_bytes': TOTAL, 'actual_full_payload_read_bytes': TOTAL,
        'all_CUDA_devices_hidden': True, 'NumPy_or_Torch_imported': False,
        'model_calls': 0, 'GPU_calls': 0, 'copied_bytes': 0, 'automatic_retry': False,
        'wall_budget_seconds': RESOURCES['wall_seconds']}
    native.need(all(run.get(k) == v for k, v in expected.items())
        and 0 <= run.get('wall_seconds', -1) <= RESOURCES['wall_seconds'], 'Actual closed V5 proof identity or budget differs')
    rows = run['model_payloads']
    native.need(len(rows) == 4 and {r['path'] for r in rows} == {r['path'] for r in plan['models']}, 'Four raw masters required')
    for item in plan['models']:
        row = next(r for r in rows if r['path'] == item['path'])
        native.need(row.get('kind') == item['kind'] and row.get('sha256') == item['sha256']
            and row.get('status') == 'PASS_FRESH_FULL_SHA'
            and row.get('predebited_bytes') == row.get('actual_read_bytes') == item['bytes']
            and native.stat_record(item['path']) == row.get('stat'), 'Master changed after V5 full SHA')
        if item['kind'] in ('high', 'low'):
            native.need(run['native_headers'][item['kind']]['header_sha256'] == item['header_sha256'], 'Raw header differs')


def require_closed_proof(plan, hf):
    from e0_original_bf16_full_depth_collect import exited
    p = plan['execution_raw_master_CPU_proof']; protocol = native.read(native.bound(ROOT, p['protocol']))
    validate(protocol)
    native.need(protocol['status'] == 'FROZEN_REGISTERED' and protocol['models'] == plan['models']
        and protocol['private_worker_config_sha256'] == plan['execution_private_worker_config_sha256'], 'V5 protocol model/worker differs')
    run = native.read(native.bound(hf, p['RUN']))
    exited(native.read(native.bound(hf, p['launch_exit'])), p['RUN'], p['protocol']['sha256'], ID)
    mirror.require_worker()
    native.need(native.sha(mirror.private_config_path()) == plan['execution_private_worker_config_sha256'], 'Different execution worker')
    verify_report(plan, p['protocol'], run)
    return p


def run(args):
    started = time.monotonic(); plan = native.read(args.protocol); validate(plan); digest = native.sha(args.protocol)
    scope = mirror.authorize(plan, digest)
    native.need(plan['status'] == 'FROZEN_REGISTERED' and scope['protocol'] == str(args.protocol.resolve().relative_to(ROOT))
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'numpy', 'torch'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 4, 'Exact hidden-CUDA four-CPU V5 scope required')
    mirror.require_worker()
    native.need(native.sha(mirror.private_config_path()) == plan['private_worker_config_sha256'], 'Actual worker differs')
    for row in plan['repo_sources']: native.bound(ROOT, row)
    out = (args.hf_root/OUTPUT).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh V5 output required'); out.mkdir(parents=True)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Environment verification budget120s'))); signal.alarm(120)
    try: environment = reuse_environment(plan, args.hf_root)
    finally: signal.alarm(0)
    native.write(out/'ENVIRONMENT_REUSE.json', environment)
    report = run_proof(plan, out, digest, started)
    report['environment_reuse'] = native.binding(out/'ENVIRONMENT_REUSE.json', args.hf_root)
    native.write(out/'RUN.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('draft', 'run'))
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface')); args = p.parse_args()
    if args.action == 'draft':
        native.need(not args.protocol.exists(), 'Fresh V5 draft required'); native.write(args.protocol, draft())
    else: run(args)


if __name__ == '__main__': main()
