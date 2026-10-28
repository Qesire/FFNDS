"""Fresh original-BF16 master identity on a registered execution worker.

Draft and inspect are stdlib only and never read model payloads. Registered run
reads the four existing masters exactly once for SHA, plus two small headers.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import e0_original_bf16_videos as native
from worker_identity import private_config_path, require_worker

ROOT = Path(__file__).resolve().parents[1]
ID = 'E0_ORIGINAL_BF16_NEW_WORKER_MIRROR_CPU_20261029'
PARENT = 'config/e0_original_bf16_videos_20261028_v1.json'
PARENT_SHA = '2f5352e91a877e85c0c0e88d8c216f74f5124693161ad412b7226aff73c69122'
TOTAL = 69025362058
RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'hash_threads': 2, 'wall_seconds': 300,
             'outer_seconds': 360, 'automatic_retry': False}
LIMITS = {'full_payload_bytes': TOTAL, 'models': 4, 'native_headers': 2}
SOURCES = ('tools/raw_bf16_worker_mirror_proof.py', 'tests/test_raw_bf16_worker_mirror_proof.py',
           'tools/e0_original_bf16_videos.py', 'tools/worker_identity.py')


def parent():
    p = ROOT/PARENT
    native.need(native.sha(p) == PARENT_SHA, 'Original parent protocol SHA mismatch')
    obj = native.read(p)
    native.need(len(obj['models']) == 4 and {x['kind'] for x in obj['models']} == {'high','low','T5','VAE'}
                and sum(x['bytes'] for x in obj['models']) == TOTAL, 'Original master closure mismatch')
    return obj


def draft():
    old = parent()
    return {'id': ID, 'status': 'DRAFT_NOT_REGISTERED', 'parent_protocol': {'path': PARENT, 'sha256': PARENT_SHA},
            'models': old['models'], 'models_sha256': native.canonical(old['models']),
            'private_worker_config_sha256': None, 'resources': RESOURCES, 'limits': LIMITS,
            'repo_sources': [native.binding(ROOT/p, ROOT) for p in SOURCES],
            'output_HF_relative': 'ffnds-derived/e0_original_bf16_new_worker_mirror_cpu_20261029/attempt1/worker/results',
            'execution_identity': 'fresh SHA and stable stat on actual execution worker; no model, GPU, training, or retry'}


def validate(plan):
    expected = draft()
    expected['private_worker_config_sha256'] = plan.get('private_worker_config_sha256')
    expected['status'] = plan.get('status')
    native.need(plan == expected and plan['status'] in ('DRAFT_NOT_REGISTERED','FROZEN_REGISTERED'), 'Fixed proof protocol differs')
    digest = plan['private_worker_config_sha256']
    native.need(digest is None or isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest), 'Worker SHA format')
    if plan['status'] == 'FROZEN_REGISTERED': native.need(digest is not None, 'Worker identity not bound')


def authorize(plan, digest):
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    a = active['experiment_authority']; standing = active['user_continuation_authorization_20261027']
    matches = [s for s in a['current_user_authorized_scopes'] if s.get('id') == ID]
    native.need(active.get('status') == 'E0_OPEN_DISCOVERY_USER_RESUMED_REGISTERED_SCOPES_ONLY'
      and a.get('all_cpu_gpu_model_experiments') == 'USER_RESUMED_EXACT_REGISTERED_SCOPES_ONLY'
      and standing.get('status') == 'ACTIVE_CURRENT_USER_DIRECTED_SESSION'
      and standing.get('new_scope_registration_allowed') is True and standing.get('automatic_retry') is False
      and len(matches) == 1, 'Standing session authorization missing')
    scope = matches[0]
    native.need(scope.get('protocol_sha256') == digest and scope.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
      and scope.get('resources') == RESOURCES and scope.get('limits') == LIMITS, 'Exact mirror proof registration required')
    return scope


def run_proof(plan, out, digest):
    start = time.monotonic()
    report = {'id': ID, 'protocol_sha256': digest, 'status': 'STARTED_FRESH_SHA_NO_RETRY',
      'parent_protocol_sha256': PARENT_SHA, 'models_sha256': plan['models_sha256'],
      'private_worker_config_sha256': plan['private_worker_config_sha256'],
      'model_payloads': [], 'predebited_full_payload_bytes': 0, 'actual_full_payload_read_bytes': 0,
      'native_headers': {}, 'all_CUDA_devices_hidden': True, 'NumPy_or_Torch_imported': False,
      'GPU_calls': 0, 'model_calls': 0, 'automatic_retry': False}
    def persist(): native.write(out/'RUN.json', report)
    persist(); stop = threading.Event(); timed_out = False
    def timeout(*_):
        nonlocal timed_out
        timed_out = True; stop.set()
    signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCES['wall_seconds'])
    try:
        pending = iter(plan['models'])
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            def submit(item):
                report['predebited_full_payload_bytes'] += item['bytes']
                report['model_payloads'].append({'path': item['path'], 'kind': item['kind'], 'expected_sha256': item['sha256'],
                    'predebited_bytes': item['bytes'], 'actual_read_bytes': 0, 'status': 'PREDEBITED_STARTED'})
                persist()
                futures[pool.submit(native.full_hash,item,stop)] = item
            for _ in range(2): submit(next(pending))
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    item = futures.pop(future)
                    result = future.result(); result['kind'] = item['kind']
                    index = next(i for i,x in enumerate(report['model_payloads']) if x['path'] == item['path'])
                    report['model_payloads'][index] = result
                    report['actual_full_payload_read_bytes'] = sum(x['actual_read_bytes'] for x in report['model_payloads'])
                    persist()
                    if result['status'] != 'PASS_FRESH_FULL_SHA': stop.set()
                    if not stop.is_set():
                        item = next(pending,None)
                        if item is not None: submit(item)
        native.need(not timed_out and not stop.is_set() and len(report['model_payloads']) == 4
          and report['predebited_full_payload_bytes'] == report['actual_full_payload_read_bytes'] == TOTAL, 'Fresh full SHA incomplete or timed out')
        for item in plan['models']:
            if item['kind'] in ('high','low'):
                row = next(x for x in report['model_payloads'] if x['path'] == item['path'])
                native.need(native.stat_record(item['path']) == row['stat'], 'Master changed before header')
                h = native.raw_header_check(native.header(item['path']))
                native.need(h['header_sha256'] == item['header_sha256'] and native.stat_record(item['path']) == row['stat'], 'Raw header or stat changed')
                report['native_headers'][item['kind']] = {k:h[k] for k in ('header_sha256','header_bytes','tensor_count')}
                persist()
        native.need(not timed_out and time.monotonic()-start <= 300 and not {'torch','numpy'} & set(sys.modules), 'Wall or CPU-only contract exceeded')
        report['status'] = 'PASS_FRESH_WORKER_RAW_SHA_STAT_HEADER_NO_MODEL_NO_NUMPY_NO_TORCH'
    except BaseException as exc:
        stop.set(); report.update(status='FAILED_MIRROR_PROOF_PRESERVED_NO_RETRY', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-start; persist()
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=('draft','inspect','run')); ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--hf-root',type=Path,required=True); args=ap.parse_args()
    if args.command == 'draft':
        native.need(not args.protocol.exists(), 'Fresh draft required'); native.write(args.protocol,draft()); return
    plan=native.read(args.protocol); validate(plan)
    if args.command == 'inspect': print(json.dumps({'id':ID,'status':plan['status'],'models_sha256':plan['models_sha256'],'limits':LIMITS})); return
    native.need(plan['status'] == 'FROZEN_REGISTERED', 'Frozen protocol required')
    digest=native.sha(args.protocol); scope=authorize(plan,digest)
    native.need(scope.get('protocol') == str(args.protocol.resolve().relative_to(ROOT)), 'Registered protocol path differs')
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch','numpy'} & set(sys.modules) and len(os.sched_getaffinity(0)) <= 4, 'CPU affinity/import/device contract differs')
    require_worker(); native.need(native.sha(private_config_path()) == plan['private_worker_config_sha256'], 'Execution worker configuration SHA mismatch')
    out=(args.hf_root.resolve()/plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined proof output required')
    out.mkdir(parents=True); print(json.dumps({'status':run_proof(plan,out,digest)['status']}))


if __name__ == '__main__': main()
