"""Validate all clean calibration shards, then reduce the four ICC scale banks."""
import argparse
import os
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone

from icc_native_job import read_json, sha, write_json
from icc_calibration_receipts import validate
from icc_calibration_driver import validate_and_plan
from icc_statistics_finalize import finalize


def run(scope_path, expected_sha):
    stage = scope_path.parent
    if scope_path.name != 'scope.json' or sha(scope_path) != expected_sha:
        raise RuntimeError('changed aggregate scope')
    scope = read_json(scope_path)
    active = read_json(stage / 'ACTIVE_RESEARCH.json')[scope['scope_id'].lower()]
    review_path = stage / 'REVIEW.json'
    review = read_json(review_path)
    if (not active['runnable'] or active['scope_sha256'] != expected_sha
            or active['review_sha256'] != sha(review_path) or review['scope_sha256'] != expected_sha
            or review['recheck_disposition'] != 'APPROVE'):
        raise RuntimeError('missing exact REVIEW/ACTIVE authorization')
    if stage != Path(scope['stage']) or not Path(sys.executable).samefile(scope['python_executable']):
        raise RuntimeError('aggregate path or interpreter identity changed')
    if sha(sys.executable) != scope['python_executable_sha256'] or socket.gethostname() != scope['host']:
        raise RuntimeError('aggregate interpreter content or CPU host changed')
    for name, expected in scope['bundle_sha256'].items():
        target = stage / name
        if target.resolve().parent != stage or sha(target) != expected:
            raise RuntimeError(f'changed aggregate source/input {name}')
    if not all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
        raise RuntimeError('one CPU thread is required')
    out = stage / 'output'
    out.mkdir()
    write_json(out / 'RUN.json', {'scope_sha256': expected_sha, 'host': socket.gethostname(),
                                  'start_utc': datetime.now(timezone.utc).isoformat(), 'wan_nfe': 0, 'videos': 0})
    rc = 1
    try:
        receipt = compute(scope, stage, out)
        write_json(out / 'SUMMARY.json', receipt)
        rc = 0
    finally:
        write_json(out / 'EXIT.json', {'exit_code': rc, 'end_utc': datetime.now(timezone.utc).isoformat()})


def compute(scope, stage, out):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.__version__ != scope['torch_version'] or torch.get_num_threads() != 1:
        raise RuntimeError('aggregation numeric runtime changed')
    manifest = stage / 'ICC_CLEAN_CAL11_BOUND.json'
    source_spec = read_json(manifest)
    jobs, budget = validate_and_plan(source_spec)
    sources = [r['source_cluster_id'] for r in source_spec['records']]
    expected_stages = {(r['stage'], r['scope_sha256'], r['job_id']) for r in scope['shards']}
    if len(expected_stages) != 8 or len(scope['shards']) != 8:
        raise ValueError('exactly eight independently frozen source shards required')
    validated, transaction_index, observed = [], {}, set()
    number_main, number_diag = 0, 0
    for shard in scope['shards']:
        result = validate(shard['stage'], shard['scope_sha256'], shard['job_id'])
        if observed.intersection(result['sources']) or result['manifest_sha256'] != sha(manifest):
            raise ValueError('duplicate source or wrong calibration input')
        observed.update(result['sources'])
        number_main += result['main_nfe']
        number_diag += result['diagnostic_nfe']
        transaction_index.update(result['transactions'])
        validation_path = out / f'VERIFIED_{shard["job_id"]}.json'
        write_json(validation_path, result)
        validated.append({'job_id': shard['job_id'], 'stage': shard['stage'], 'sha256': sha(validation_path)})
    if (observed != set(sources) or set(transaction_index) != {t.identifier for t in jobs}
            or (number_main, number_diag) != (66, 27) or budget['main_and_diagnostic_nfe_max'] != 93):
        raise ValueError('incomplete registered clean calibration cohort')
    scales, diagnostics = [], []
    for family in ('CA', 'CD'):
        for expert in ('high', 'low'):
            main = {source: transaction_index[f'{source}|{family}|{expert}:main'] for source in sources}
            source_dirs = {source: row['directory'] for source, row in main.items()}
            statistics = {(source, item['layer']): item['pt_sha256']
                          for source, row in main.items() for item in row['statistics']}
            dest = out / f'{family}_{expert}'
            scale_bank = finalize(source_dirs, expert=expert, family=family, expected_sources=sources, output=dest,
                                  oracle_receipt=stage / 'ORACLE.json', collector_file=stage / 'icc_four_boundary_collector.py',
                                  expected_statistics=statistics)
            for item in scale_bank['outputs']:
                scales.append({'family': family, 'expert': expert, 'layer': item['layer'],
                               'calibration_manifest_sha256': sha(manifest),
                               'file': {'path': str(dest / item['file']), 'sha256': item['sha256']}})
    for txn in jobs:
        if txn.branch == 'main':
            continue
        value = transaction_index[txn.identifier]
        if txn.may_reuse_main:
            diagnostics.append({'source': txn.source, 'family': txn.family, 'expert': txn.expert,
                                'branch': txn.branch, 'timesteps': list(txn.indices), 'evidence': value,
                                'run_nfe': 0})
            continue
        path = Path(value['directory']) / 'PAIRED_DIAGNOSTIC.json'
        if sha(path) != value['diagnostic_sha256']:
            raise ValueError('changed validated paired diagnostic')
        diagnostic = read_json(path)
        diagnostics.append({'source': txn.source, 'family': txn.family, 'expert': txn.expert,
                            'branch': txn.branch, 'timesteps': list(txn.indices),
                            'run_nfe': len(txn.indices), 'flow_rms': diagnostic['flow_rms'],
                            'per_layer_statistics_differences': diagnostic['per_layer_statistics_differences'],
                            'main_clean_and_initial': {'clean': diagnostic['main']['clean'], **diagnostic['main']['initial']},
                            'alternate_clean_and_initial': {'clean': diagnostic['alternate']['clean'], **diagnostic['alternate']['initial']},
                            'evidence': {'path': str(path), 'sha256': value['diagnostic_sha256']}})
    write_json(out / 'SCALES.json', {'calibration_manifest_sha256': sha(manifest), 'sources': sources, 'scales': scales})
    write_json(out / 'CAL3_PAIRED.json', {'sources': budget['cal3_source_clusters'], 'pairs': diagnostics,
                                         'registered_main_time_support': [{'source': t.source, 'family': t.family,
                                                                            'expert': t.expert, 'indices': list(t.indices)}
                                                                           for t in jobs if t.branch == 'main']})
    return {'status': 'PASS', 'source_count': len(sources), 'main_nfe': number_main, 'diagnostic_nfe': number_diag,
            'source_validation': validated, 'scale_records': len(scales), 'scales_sha256': sha(out / 'SCALES.json'),
            'diagnostics_sha256': sha(out / 'CAL3_PAIRED.json')}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('scope', type=Path)
    cli.add_argument('scope_sha256')
    args = cli.parse_args()
    run(args.scope.resolve(), args.scope_sha256)

