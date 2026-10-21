"""OP2A: complete post-write payload audit and compact dataset evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from ar2_checkpoint import atomic_json, canonical_json, digest
from native_payload_integrity import verify_payload
from run_ar2_restart import ROOT
from ub1_native_data import load_ub1_collections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('Complete native payload audit is CPU only')
    runs = ROOT / 'runs/worker_restart_20261020'
    paths = [runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in [0, 1]]
    records, bindings, identity = load_ub1_collections(paths, ROOT)
    entries, sources = [], []
    for shard, path in enumerate(paths):
        report = json.loads(path.read_text())
        accepted = {entry['index']: entry for entry in report['completed']}
        for index in sorted(accepted):
            entry = accepted[index]
            case = records[index]
            parent = Path(entry['input_report']).parent
            for phase in ['plain', 'observed']:
                counter = case['native_counter_deltas'][phase]
                if (counter['linear_forward_calls'] != 1600 or counter['activation_qdq_calls'] != 1600
                        or counter['cache_misses']):
                    raise RuntimeError('Native input does not bind all original linears and QDQs')
            if any(scope['extra_full_teacher_GEMMs'] != 2 for scope in case['scopes']):
                raise RuntimeError('Every recorded teacher position must execute two full native linears')
            for scope in case['scopes']:
                if scope['rows'] != 256:
                    raise RuntimeError('Recorded row count differs from native payload contract')
                entries.append((index, scope, parent))
            sources.append({'index': index, 'source_group': case['input']['source_group'],
                'image_sha256': case['input']['input_provenance']['image_sha256'], 'seed': case['input']['seed'],
                'input_report_sha256': entry['input_report_sha256'], 'scope_count': len(case['scopes']),
                'payload_bytes': sum(scope['bytes'] for scope in case['scopes']),
                'video_sha256': case['videos']['plain']['sha256'], 'plain_observed_video_bitwise': True,
                'trajectory': case['trajectories']['plain'], 'plain_observed_latents_bitwise': True})
    if len(entries) != 3840 or len({str(Path(scope['path']).resolve()) for _, scope, _ in entries}) != 3840:
        raise RuntimeError('Complete distinct 24 by160 payload coverage required')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / 'RUN.json'
    if path.exists():
        raise RuntimeError('Full payload audit reports are immutable')
    inventory = [{'index': index, 'step': scope['step'], 'block': scope['block'],
                  'sha256': scope['sha256'], 'bytes': scope['bytes']} for index, scope, _ in entries]
    inventory.sort(key=lambda row: (row['index'], row['step'], row['block']))
    meta = {'status': 'RUNNING', 'protocol_sha256': digest(ROOT / 'config/native_dataset_audit_op2a.json'),
        'collection_bindings': bindings, 'original_UB1_policy_sha256': identity['original_UB1_policy_sha256'],
        'producer_sources': identity['sources'], 'source_count': 24, 'expected_payloads': 3840,
        'verified_payloads': 0, 'verified_bytes': 0, 'sources': sorted(sources, key=lambda row: row['index']),
        'inventory_canonical_sha256': hashlib.sha256(canonical_json(inventory).encode()).hexdigest(),
        'audit_sources': {str(ROOT / 'tools' / name): digest(ROOT / 'tools' / name) for name in
            ['audit_ub1_native_dataset.py', 'native_payload_integrity.py', 'ub1_native_data.py']},
        'quality_or_speed_acceptance': False}
    atomic_json(path, meta)
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(verify_payload, scope, parent) for _, scope, parent in entries]
            try:
                for future in as_completed(futures):
                    meta['verified_bytes'] += future.result()
                    meta['verified_payloads'] += 1
                    if meta['verified_payloads'] % 192 == 0:
                        atomic_json(path, meta)
                        print(f'OP2A verified {meta["verified_payloads"]}/3840 payloads', flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if (meta['verified_payloads'] != 3840
                or meta['verified_bytes'] != sum(row['bytes'] for row in inventory)
                or torch.cuda.is_initialized()):
            raise RuntimeError('Full native dataset audit did not cover the registered bytes')
        atomic_json(args.out / 'INVENTORY.json', inventory)
        meta.update(status='PASS_OP2A_ALL24_FULL3840_NATIVE_PAYLOAD_INTEGRITY', CUDA_initialized=False,
            elapsed_seconds=time.perf_counter() - started, actual_native_positions=3840,
            original_native_video_pairs=24, offline_extra_teacher_GEMMs=7680,
            scope='payload integrity and complete native producer/no-op evidence, not new candidate performance')
    except BaseException as error:
        meta.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(path, meta)


if __name__ == '__main__':
    main()
