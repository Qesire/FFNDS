"""Verify every committed native calibration transaction against exact scope and trace."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from icc_calibration_driver import validate_and_plan


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(test, description):
    if not test:
        raise ValueError(description)


def check_transaction(txn, info, stage, trace, schedule):
    called = [trace[txn.identifier, step] for step in txn.indices]
    source_dir = txn.source.replace(':', '_')
    directory = stage / 'output' / 'cal' / source_dir / f'{txn.family}_{txn.expert}_{txn.branch}'
    require(info['status'] == 'COMMITTED' and Path(info['directory']) == directory
            and read(directory / 'TRANSACTION.json') == info, f'commit differs {txn.identifier}')
    require(info['transaction'] == txn.identifier and info['forward_flows'] == [row['flow'] for row in called],
            f'flow trace differs {txn.identifier}')
    require(info['initial']['steps'] == [schedule[step] for step in txn.indices]
            and info['initial']['expert'] == txn.expert and info['initial']['noised'] == called[0]['model_input'],
            f'initial identity differs {txn.identifier}')
    for idx, (row, step) in enumerate(zip(called, txn.indices, strict=True)):
        phase = 'CA' if txn.family == 'CA' else ('anchor' if idx == 0 else 'hit')
        require(row['step'] == step and row['phase'] == phase and row['expert'] == txn.expert
                and {k: row[k] for k in schedule[step]} == schedule[step], f'forward identity differs {txn.identifier}')
    second = called[1]['model_input'] if txn.family == 'CD' else None
    require(info['cd_second_input'] == second, f'CD follow-up differs {txn.identifier}')
    complete_path = directory / 'COMPLETE.json'
    complete = read(complete_path)
    fixed = {'source': txn.source, 'family': txn.family, 'expert': txn.expert, 'layers': 40, 'tokens': 31668,
             'identity': {'transaction': txn.identifier, **info['initial']}}
    require(all(complete[key] == value for key, value in fixed.items()), f'complete differs {txn.identifier}')
    files = []
    for layer in range(40):
        meta_path = directory / f'stats_{layer:02d}.json'
        values_path = directory / f'stats_{layer:02d}.pt'
        meta = read(meta_path)
        fields = {key: value for key, value in fixed.items() if key != 'layers'}
        values_sha = sha(values_path)
        require(all(meta[key] == value for key, value in fields.items())
                and meta['layer'] == layer and meta['stats_sha256'] == values_sha,
                f'statistics differ {txn.identifier} layer {layer}')
        files.append({'layer': layer, 'pt_sha256': values_sha, 'metadata_sha256': sha(meta_path)})
    require(not list(directory.glob('anchor_*.pt')), f'purged CD scratch remains {txn.identifier}')
    return {'directory': str(directory), 'complete_sha256': sha(complete_path),
            'transaction_sha256': sha(directory / 'TRANSACTION.json'), 'statistics': files}


def check_pair(txn, main, alternate):
    old, new = main['initial'], alternate['initial']
    changed = {key for key in old if old[key] != new[key]}
    if txn.branch == 'text_flip':
        require(changed == {'context', 'empty_context'} and main['clean'] == alternate['clean'],
                f'text comparison failed to change only context {txn.identifier}')
    else:
        require(changed == {'noised'} and main['clean'] != alternate['clean'],
                f'posterior comparison failed to change only target {txn.identifier}')


def validate(stage, scope_sha256, job_id):
    stage = Path(stage)
    scope_path, exit_path = stage / 'scope.json', stage / 'receipt' / 'EXIT.json'
    run_path = stage / 'receipt' / 'RUN.json'
    run = read(run_path)
    require(sha(scope_path) == scope_sha256 and run['scope_sha256'] == scope_sha256
            and str(run['allocation']['job_id']) == str(job_id),
            'execution scope hash mismatch')
    scope, exit_info = read(scope_path), read(exit_path)
    require(exit_info['exit_code'] == 0 and str(exit_info['slurm_job_id']) == str(job_id), 'shard EXIT failed')
    manifest_path = stage / 'ICC_CLEAN_CAL11_BOUND.json'
    require(sha(manifest_path) == scope['bundle_sha256'][manifest_path.name], 'changed CAL11 manifest')
    manifest = read(manifest_path)
    plan, _ = validate_and_plan(manifest)
    planned = [txn for txn in plan if txn.source in scope['sources']]
    require(len(set(scope['sources'])) == len(scope['sources'])
            and set(scope['sources']) == {txn.source for txn in planned}, 'shard source set mismatch')
    summary_path, nfe_path = (stage / 'output' / name for name in ('SUMMARY.json', 'NFE.jsonl'))
    summary = read(summary_path)
    expected_calls = [(txn.identifier, step) for txn in planned if not txn.may_reuse_main for step in txn.indices]
    events = [json.loads(line) for line in nfe_path.read_text(encoding='utf-8').splitlines()]
    require(len(events) == 2 * len(expected_calls) and len(expected_calls) == scope['max_nfe'],
            'NFE journal count differs')
    traces = {}
    for i, (identifier, step) in enumerate(expected_calls):
        attempt, complete = events[2 * i:2 * i + 2]
        require(attempt == {'event': 'NFE_ATTEMPT', 'transaction': identifier, 'step': step, 'total': i+1}
                and (complete['event'], complete['transaction'], complete['step']) == ('NFE_COMPLETED', identifier, step),
                f'NFE journal differs at {i+1}')
        traces[identifier, step] = complete['trace']
    main_nfe = sum(len(txn.indices) for txn in planned if txn.branch == 'main')
    require((summary['attempted_nfe'], summary['main_nfe'], summary['diagnostic_nfe'])
            == (len(expected_calls), main_nfe, len(expected_calls)-main_nfe), 'summary NFE count differs')
    transactions = summary['transactions']
    require(set(transactions) == {txn.identifier for txn in planned}, 'summary coverage differs')
    main, evidence = {}, {}
    for txn in planned:
        info = transactions[txn.identifier]
        if txn.may_reuse_main:
            require(info == {'status': 'REUSED_IDENTICAL_INPUT', 'main': txn.main_id}
                    and txn.main_id in main, f'invalid guarded main reuse {txn.identifier}')
            evidence[txn.identifier] = info
            continue
        if txn.branch == 'main':
            main[txn.main_id] = info
            evidence[txn.identifier] = check_transaction(txn, info, stage, traces, scope['schedule'])
            continue
        require(info['transaction'] == txn.identifier and info['main'] == main[txn.main_id],
                f'diagnostic pairing differs {txn.identifier}')
        receipt = check_transaction(txn, info['alternate'], stage, traces, scope['schedule'])
        check_pair(txn, info['main'], info['alternate'])
        diagnostic = Path(receipt['directory']) / 'PAIRED_DIAGNOSTIC.json'
        for value in info['flow_rms']:
            require(math.isfinite(value) and value >= 0, f'invalid paired flow RMS {txn.identifier}')
        for row in info['per_layer_statistics_differences']:
            require(set(row) == {'layer', 'u', 'z', 'h', 'y'}, f'invalid diagnostic boundaries {txn.identifier}')
            for boundary in ('u', 'z', 'h', 'y'):
                require(set(row[boundary]) == {'normalized_rms', 'magnitude_rms'}
                        and all(math.isfinite(x) and x >= 0 for x in row[boundary].values()),
                        f'invalid diagnostic statistic RMS {txn.identifier}')
        require(read(diagnostic) == info and len(info['flow_rms']) == len(txn.indices)
                and [row['layer'] for row in info['per_layer_statistics_differences']] == list(range(40)),
                f'diagnostic evidence differs {txn.identifier}')
        evidence[txn.identifier] = {**receipt, 'diagnostic_sha256': sha(diagnostic)}
    return {'status': 'PASS', 'stage': str(stage), 'scope_sha256': scope_sha256, 'job_id': str(job_id),
            'run_sha256': sha(run_path), 'exit_sha256': sha(exit_path), 'summary_sha256': sha(summary_path),
            'nfe_sha256': sha(nfe_path), 'main_nfe': main_nfe, 'diagnostic_nfe': len(expected_calls)-main_nfe,
            'manifest_sha256': sha(manifest_path), 'sources': scope['sources'], 'transactions': evidence}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', type=Path)
    parser.add_argument('scope_sha256')
    parser.add_argument('job_id')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = validate(args.stage, args.scope_sha256, args.job_id)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')

