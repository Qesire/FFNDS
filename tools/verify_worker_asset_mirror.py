"""Verify all original model/cache bytes after a user-directed worker mirror."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-transfers', action='store_true')
    args = parser.parse_args()
    from worker_identity import require_worker, matches_worker
    require_worker()
    if args.output.exists():
        raise RuntimeError('Preserve completed verification')
    transfer_results = [ROOT / f'runs/r12d0_native_value_attempt1/STAGE_{family}_RESULT.json'
                        for family in ('weights', 'assets')]
    if args.wait_for_transfers:
        print('Waiting for both fixed-asset transfers; no model allocated', flush=True)
        while not all(p.is_file() for p in transfer_results):
            time.sleep(10)
    for family in ('weights', 'assets'):
        p = ROOT / f'runs/r12d0_native_value_attempt1/STAGE_{family}_RESULT.json'
        if json.loads(p.read_text())['exit_code'] != 0:
            raise RuntimeError('Both exact-asset transfers must complete successfully first')
    started = time.monotonic()
    model_path = ROOT / '.runtime/node_local_model_view.json'
    cache_path = ROOT / '.runtime/hif4_rebuilt_cache.json'
    model, cache = [json.loads(p.read_text()) for p in (model_path, cache_path)]
    if not matches_worker(model['host']):
        raise RuntimeError('Unrecognized original manifest provenance')
    work = []
    for row in model['files']:
        if (not Path(row['blob']).samefile(row['view_path'])
                or not Path(row['blob']).samefile(row['path'])
                or Path(row['blob']).stat().st_size != row['bytes']):
            raise RuntimeError('Original model hardlink layout/size not preserved')
        work.append((row['blob'], row['sha256'], f'model:{row["repo"]}@{row["revision"]}/{row["filename"]}'))
    work.append((cache['path'], cache['sha256'], 'original_DQ_combined_cache'))
    work.extend((s['path'], s['sha256'], f'original_DQ_shard:{s["expert"]}/block{s["block"]}') for s in cache['shards'])
    if len(work) != 93:
        raise RuntimeError('Expected12 model blobs, one combined cache and80 shards')

    def verify(item):
        path, expected, locator = item
        actual = sha(path)
        if actual != expected:
            raise RuntimeError('Migrated asset checksum mismatch: ' + locator)
        return {'locator': locator, 'sha256': actual, 'bytes': Path(path).stat().st_size}

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(verify, work))
    free = shutil.disk_usage(cache['path']).free
    if free < (80 << 30) + (1 << 29):
        raise RuntimeError('Mirror must preserve80GiB plus the diagnostic output allowance')
    result = {'status': 'PASS_EXACT_MODEL_AND_DQ_CACHE_MIRROR_93_SHA256',
        'verification_source_sha256': sha(__file__), 'model_view_manifest_sha256': sha(model_path),
        'DQ_cache_manifest_sha256': sha(cache_path), 'verified_files': len(rows),
        'verified_payload_bytes': sum(r['bytes'] for r in rows), 'records': rows,
        'free_bytes_after': free, 'torch_imported': False, 'new_model_generations': 0,
        'elapsed_seconds': time.monotonic() - started}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
