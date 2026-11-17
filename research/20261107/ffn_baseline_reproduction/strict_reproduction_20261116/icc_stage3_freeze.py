"""Assemble the selected ICC full-depth bank from successful exact factor jobs."""
import argparse
import hashlib
import json
from pathlib import Path

SIX = {(expert, layer) for expert in ('high', 'low') for layer in (4, 19, 34)}
FULL = {(expert, layer) for expert in ('high', 'low') for layer in range(40)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def passed(spec):
    stage = Path(spec['stage']).resolve()
    run = read(stage / 'receipt/RUN.json')
    exit_record = read(stage / 'receipt/EXIT.json')
    if (sha(stage / 'scope.json') != spec['scope_sha256'] or run['scope_sha256'] != spec['scope_sha256']
            or str(run['slurm_job_id']) != str(spec['job_id']) or str(exit_record['slurm_job_id']) != str(spec['job_id'])
            or exit_record['exit_code'] != 0):
        raise ValueError('missing exact completed Stage3 producer')
    proof = {'stage': str(stage), 'job_id': str(spec['job_id']), 'scope_sha256': spec['scope_sha256'],
             'run_sha256': sha(stage / 'receipt/RUN.json'), 'exit_sha256': sha(stage / 'receipt/EXIT.json')}
    return stage, proof


def assemble(spec):
    grid, grid_proof = passed(spec['grid'])
    select_path = grid / 'output/selection.json'
    score_path = grid / 'output/scores.json'
    selection, scores = read(select_path), read(score_path)
    if (sha(select_path) != spec['selection_sha256'] or sha(score_path) != spec['scores_sha256']
            or selection['scope_sha256'] != grid_proof['scope_sha256']
            or scores['scope_sha256'] != grid_proof['scope_sha256'] or scores['cell_count'] != 792
            or len(scores['rows']) != 792 or selection['family'] not in ('SVD', 'CA', 'CD')):
        raise ValueError('missing frozen complete 792-cell selection')
    family = selection['family']
    output_index = {}
    manifests = []
    for producer in spec['factor_producers']:
        stage, proof = passed(producer)
        path = stage / 'output/FACTOR_MANIFEST.json'
        if sha(path) != producer['factor_manifest_sha256']:
            raise ValueError('factor manifest SHA mismatch')
        manifest = read(path)
        rows = manifest['factor_pairs']
        if manifest['matrix_decompositions'] != 2 * len(rows):
            raise ValueError('factor production accounting differs')
        manifests.append({**proof, 'factor_manifest_sha256': sha(path)})
        for row in rows:
            identity = row['identity']
            if identity['family'] != family:
                continue
            site = identity['expert'], identity['layer']
            if site in output_index or site not in FULL:
                raise ValueError('duplicate or unexpected chosen factor')
            if (identity['rank'] != 1408 or identity['checkpoint_sha256'] != spec['checkpoint_sha256'][site[0]]
                    or identity['execution_factor_dtype'] != 'bfloat16' or identity['svd_driver'] != 'gesvd'
                    or identity['factor_split'] != 'U*S, Vh' or identity['prefix_rank512_is_same_decomposition'] is not True):
                raise ValueError('factor does not match selected numeric contract')
            expected_cal = None if family == 'SVD' else spec['clean_cal11_manifest_sha256']
            if identity['calibration_manifest_sha256'] != expected_cal:
                raise ValueError('selected calibrated factor data identity differs')
            tensor_path = Path(row['path']).resolve()
            if not tensor_path.is_relative_to(stage / 'output') or tensor_path.stat().st_size != row['bytes']:
                raise ValueError('factor file missing/outside producer or changed size')
            output_index[site] = row
    if set(output_index) != FULL:
        raise ValueError('selected factor bank is incomplete')
    for row in scores['rows']:
        if row['family'] == family and row['factor_sha256'] != output_index[row['expert'], row['layer']]['sha256']:
            raise ValueError('chosen factor at grid site was replaced after selection')
    return {'status': 'ALL_80_FROZEN', 'family': family, 'rank': 1408,
            'factor_pairs': [output_index[site] for site in sorted(FULL)], 'factor_producers': manifests,
            'grid_producer': grid_proof, 'selection_sha256': spec['selection_sha256'],
            'scores_sha256': spec['scores_sha256'], 'uses_exact_grid_factor_hashes': True,
            'runtime_factor_content_recheck_required': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('input_sha256')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if sha(args.input) != args.input_sha256:
        raise ValueError('selected-factor freeze input changed')
    result = assemble(read(args.input))
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')

