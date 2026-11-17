"""Release unselected predictions after the exact Stage 3 decomposition completes."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def bound(spec):
    path = Path(spec['path'])
    if not path.is_absolute() or path.resolve() != path or digest(path) != spec['sha256']:
        raise ValueError(f'changed or noncanonical cached evidence: {path}')
    return read(path)


def write(path, value):
    with path.open('x', encoding='utf-8') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())


def producer(spec):
    stage = Path(spec['stage'])
    if not stage.is_absolute() or stage.resolve() != stage:
        raise ValueError('noncanonical producer stage')
    scope, start, end = (bound({'path': str(stage / name), 'sha256': spec[key]}) for name, key in (
        ('scope.json', 'scope_sha256'), ('receipt/RUN.json', 'run_sha256'), ('receipt/EXIT.json', 'exit_sha256')))
    if (start['scope_sha256'] != spec['scope_sha256'] or end['exit_code'] != 0
            or str(start['slurm_job_id']) != str(spec['job_id']) or str(end['slurm_job_id']) != str(spec['job_id'])):
        raise ValueError('producer evidence lacks successful exact scope')
    return stage / 'output', scope


def build(spec):
    grid, grid_scope = producer(spec['grid'])
    out, decomposition_scope = producer(spec['decomposition'])
    if grid_scope['operation'] != 'icc_grid' or decomposition_scope['operation'] != 'icc_decomposition':
        raise ValueError('wrong grid or decomposition scope')
    manifest, selection, scores = (bound({'path': str(grid / f'{name}.json'), 'sha256': spec[f'{name}_sha256']})
                                   for name in ('cache_manifest', 'selection', 'scores'))
    decomp = bound({'path': str(out / 'decomposition.json'), 'sha256': spec['decomposition_sha256']})
    if (scores['cell_count'] != 792 or len(scores['rows']) != 792
            or any(item['scope_sha256'] != spec['grid']['scope_sha256'] for item in (manifest, selection, scores))
            or decomp['status'] != 'DECOMPOSITION_COMPLETE' or decomp['factor_actions'] != 132
            or decomp['selection_sha256'] != spec['selection_sha256'] or len(decomp['rows']) != 66
            or decomposition_scope['selection_sha256'] != spec['selection_sha256']
            or decomposition_scope['cache_manifest_sha256'] != spec['cache_manifest_sha256']):
        raise ValueError('complete selected explanation and exact grid evidence required')
    families = {'SVD', 'CA', 'CD'}
    selected = selection['family']
    if selected not in families:
        raise ValueError('unknown selection')
    canonical_sites = {(source, expert, layer) for source in grid_scope['sources']
                       for expert in ('high', 'low') for layer in (4, 19, 34)}
    replay = {(row['source'], row['expert'], row['layer']): row for row in decomp['rows']}
    if len(canonical_sites) != 66 or set(replay) != canonical_sites:
        raise ValueError('decomposition does not cover all native pairs')
    native = {f'native__{s}__{e}__{layer:02d}.pt' for s, e, layer in canonical_sites}
    index = {row['file']: row for row in manifest['files']}
    keep, remove, keys = set(native), [], set()
    for row in scores['rows']:
        if row['rank'] != 1408 or row['mode'] != '4S':
            continue
        s, e, layer, family = (row[key] for key in ('source', 'expert', 'layer', 'family'))
        if (s, e, layer) not in canonical_sites or family not in families or (s, e, layer, family) in keys:
            raise ValueError('duplicate or unexpected primary prediction identity')
        keys.add((s, e, layer, family))
        name = f'pred__{s}__{e}__{layer:02d}__{family}.pt'
        prediction = row['prediction_cache']
        if prediction != index[name]:
            raise ValueError('prediction receipt and cache differ')
        if family == selected:
            keep.add(name)
            item = replay[s, e, layer]
            if item['family'] != family or item['prediction_cache_sha256'] != prediction['sha256']:
                raise ValueError('selected explanation did not use the exact prediction bytes')
        else:
            remove.append(prediction)
    if (len(index) != len(manifest['files']) or len(keep) != 132 or len(remove) != 132
            or set(index) != keep | {item['file'] for item in remove}):
        raise ValueError('prune set must contain exactly 132 unselected predictions')
    return {'selected_family': selected, 'cache_path': str(grid / 'cache'),
            'retained': [index[name] for name in sorted(keep)],
            'released': sorted(remove, key=lambda item: item['file']),
            'released_bytes': sum(item['bytes'] for item in remove)}


def validate_file(cache, row, *, allow_missing, check_digest):
    path = cache / row['file']
    if path.parent != cache or path.resolve() != path:
        raise ValueError(f'noncanonical cache file: {path}')
    if allow_missing and not path.exists():
        return
    if not path.is_file() or path.stat().st_size != row['bytes'] or check_digest and digest(path) != row['sha256']:
        raise ValueError(f'changed bound cache file: {path}')


def prune(spec, plan_path, receipt_path):
    plan = build(spec)
    cache = Path(plan['cache_path'])
    if cache.resolve() != cache:
        raise ValueError('noncanonical cache directory')
    resumed = plan_path.exists()
    if resumed and read(plan_path) != plan:
        raise ValueError('resumed prune plan differs')
    for item in plan['retained']:
        validate_file(cache, item, allow_missing=False, check_digest=False)
    for item in plan['released']:
        validate_file(cache, item, allow_missing=resumed, check_digest=True)
    if not resumed:
        write(plan_path, plan)
    receipt = {'status': 'PRUNED_132_UNSELECTED_PREDICTIONS', 'plan_sha256': digest(plan_path),
               'retained_files': 132, 'released_files': 132, 'released_bytes': plan['released_bytes']}
    if receipt_path.exists() and read(receipt_path) != receipt:
        raise ValueError('previous prune receipt differs')
    for item in plan['released']:
        (cache / item['file']).unlink(missing_ok=True)
    if not receipt_path.exists():
        write(receipt_path, receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    parser.add_argument('spec_sha256')
    parser.add_argument('plan', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    prune(bound({'path': str(args.spec.resolve()), 'sha256': args.spec_sha256}), args.plan, args.receipt)
