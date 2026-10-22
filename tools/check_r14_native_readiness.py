"""CPU-only asset readiness for the unregistered complete-stripe draft.

Reuses the prior full mirror SHA receipt, checks current model/cache metadata,
and rehashes four original OP2 operands, inputs, videos and the frozen affine.
This deliberately does not claim a fresh checksum of all 181 GB of assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from worker_identity import require_worker

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/20261022'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_worker()
    if args.output.exists() or 'torch' in sys.modules:
        raise RuntimeError('New output and CPU-only process required')
    began = time.monotonic()
    draft_path = BASE / 'R14_native_value_DRAFT.json'
    draft = read(draft_path)
    mirror_path = BASE / 'R12_D0_worker_asset_mirror.json'
    mirror = read(mirror_path)
    if (mirror['verified_files'] != 93 or len(mirror['records']) != 93
            or mirror['status'] != 'PASS_EXACT_MODEL_AND_DQ_CACHE_MIRROR_93_SHA256'):
        raise RuntimeError('Original full-checksum mirror receipt incomplete')
    local_path = ROOT / '.runtime/node_local_model_view.json'
    cache_path = ROOT / '.runtime/hif4_rebuilt_cache.json'
    local, cache = read(local_path), read(cache_path)
    if (sha(local_path) != mirror['model_view_manifest_sha256']
            or sha(cache_path) != mirror['DQ_cache_manifest_sha256']
            or len(local['files']) != 12 or len(cache['shards']) != 80):
        raise RuntimeError('Original asset manifests changed')
    for row in local['files']:
        path = Path(row['view_path'])
        if (path.stat().st_size != row['bytes'] or not path.samefile(row['blob'])
                or not row['copy_sha256_verified']):
            raise RuntimeError('Model metadata or HF link identity changed')
    if Path(cache['path']).stat().st_size != cache['bytes']:
        raise RuntimeError('Combined cache size changed')
    mirrored = {row['sha256']: row['bytes'] for row in mirror['records']}
    for row in cache['shards']:
        if Path(row['path']).stat().st_size != mirrored[row['sha256']]:
            raise RuntimeError('Cache shard existence or size changed')
    r9 = read(ROOT / 'research/20261021/R9A1_native_affine_results.json')
    affine = next(a for a in r9['affine_matrices'] if 2 in a['shared_steps'])
    affine_path = Path(cache['path']).parents[1] / affine['HF_derived_relative']
    affine_sha = sha(affine_path)
    if affine_sha != draft['parameters']['affine_file_sha256']:
        raise RuntimeError('Original frozen affine changed')
    sources = []
    for index in (0, 1, 10, 11):
        shard = ROOT / f'reports/worker_20261020/OP2_UB1_calibration24_shard{index % 2}_RUN.json'
        entry = next(e for e in read(shard)['completed'] if e['index'] == index)
        if sha(entry['input_report']) != entry['input_report_sha256']:
            raise RuntimeError('Original source receipt changed')
        case = read(entry['input_report'])
        scope = next(s for s in case['scopes'] if (s['step'], s['block']) == (2, 20))
        input_sha = sha(case['input']['image_path'])
        if (sha(scope['path']) != scope['sha256']
                or input_sha != case['input']['input_provenance']['image_sha256']
                or sha(case['videos']['observed']['path']) != case['videos']['observed']['sha256']
                or case['trajectories']['plain'] != case['trajectories']['observed']):
            raise RuntimeError('Original operand/input/video/no-op identity changed')
        sources.append({'index': index, 'original_receipt_sha256': entry['input_report_sha256'],
                        'original_scope_sha256': scope['sha256'], 'image_sha256': input_sha,
                        'original_video_sha256': case['videos']['observed']['sha256'],
                        'seed': case['input']['seed'], 'original_trajectory': case['trajectories']['observed']})
    free = shutil.disk_usage(Path(cache['path']).parent).free
    if free < draft['resources']['node_free_reserve_bytes_min'] + draft['resources']['incremental_output_bytes_max']:
        raise RuntimeError('Cannot preserve node disk reserve')
    result = {'id': 'R14_NATIVE_DRAFT_CPU_READINESS',
              'status': 'PASS_CPU_EXISTENCE_METADATA_AND_SELECTED_SHA_NO_MODEL',
              'source_sha256': sha(__file__), 'draft_sha256': sha(draft_path),
              'prior93_SHA_receipt_sha256': sha(mirror_path), 'prior93_SHA_receipt_files': 93,
              'current93_SHA_reverification': False, 'current_model_metadata_files': 12,
              'current_combined_cache_size_bytes': cache['bytes'], 'current_cache_shards_metadata': 80,
              'frozen_affine_current_sha256': affine_sha, 'selected_original_sources': sources,
              'free_bytes_after': free, 'torch_imported': 'torch' in sys.modules,
              'new_model_generations': 0, 'elapsed_seconds': time.monotonic() - began,
              'limits': 'CPU metadata and selected old-data SHA checks only. No native registration, model load, inference or fresh full93-asset checksum.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
