"""Validate the interrupted CAL prefix and bind its single CD replacement."""
import hashlib
import json
from pathlib import Path

from icc_calibration_driver import validate_and_plan
from icc_calibration_receipts import check_transaction, read, require, sha


def audit_interrupted(spec, manifest):
    stage = Path(spec['stage'])
    require(stage.is_absolute(), 'interrupted stage must be absolute')
    evidence = {}
    for name, expected in spec['file_sha256'].items():
        path = (stage / name).resolve()
        require(path.is_relative_to(stage), f'interrupted evidence outside stage: {name}')
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, f'changed interrupted evidence: {name}')
        evidence[name] = data
    scope = json.loads(evidence['scope.json'])
    run = json.loads(evidence['receipt/RUN.json'])
    job_id = spec['job_id']
    require(run['scope_sha256'] == spec['file_sha256']['scope.json']
            and str(run['allocation']['job_id']) == job_id
            and job_id == str(json.loads(evidence['receipt/EXIT.json'])['slurm_job_id'])
            and f'JOB {job_id} '.encode() in evidence[f'slurm-{job_id}.err']
            and b'DUE to SIGNAL Terminated' in evidence[f'slurm-{job_id}.err']
            and not (stage / 'output/SUMMARY.json').exists(), 'original shard is not the bound interrupted run')
    manifest_path = stage / 'ICC_CLEAN_CAL11_BOUND.json'
    require(sha(manifest_path) == scope['bundle_sha256'][manifest_path.name] and read(manifest_path) == manifest,
            'different original CAL11 cohort')
    jobs, _ = validate_and_plan(manifest)
    planned = [txn for txn in jobs if txn.source in scope['sources']]
    require(len(planned) == 8 and all(txn.branch == 'main' for txn in planned), 'unexpected interrupted shard cohort')
    last = planned[-1]
    require(last.identifier == spec['transaction'] and last.family == 'CD' and last.indices == (2, 3),
            'replacement must be the final interrupted low-noise CD pair')
    calls = [(txn.identifier, step) for txn in planned for step in txn.indices]
    require(len(calls) == scope['max_nfe'] == 12, 'unexpected original call count')
    events = [json.loads(line) for line in evidence['output/NFE.jsonl'].splitlines()]
    require(len(events) == 2 * len(calls) - 1, 'interrupted pair journal differs')
    traces = {}
    for index, (identifier, step) in enumerate(calls):
        attempt = {'event': 'NFE_ATTEMPT', 'transaction': identifier, 'step': step, 'total': index + 1}
        require(events[2 * index] == attempt, 'original attempt differs')
        if index < len(calls) - 1:
            complete = events[2 * index + 1]
            require((complete['event'], complete['transaction'], complete['step']) == ('NFE_COMPLETED', identifier, step),
                    'original completion differs')
            traces[identifier, step] = complete['trace']
    transactions = {}
    for txn in planned[:-1]:
        path = stage / 'output/cal' / txn.source.replace(':', '_') / f'{txn.family}_{txn.expert}_main' / 'TRANSACTION.json'
        transactions[txn.identifier] = check_transaction(txn, read(path), stage, traces, scope['schedule'])
    source_dir = stage / 'output/cal' / last.source.replace(':', '_')
    require(not (source_dir / 'CD_low_main').exists(), 'interrupted pair already committed')
    pending = source_dir / 'CD_low_main_pending'
    encoding_path = source_dir / 'ENCODING.json'
    metadata_path = pending / 'stats_00.json'
    require(all(str(path.relative_to(stage)) in spec['file_sha256'] for path in
                (stage / 'scope.json', stage / 'receipt/RUN.json', stage / 'receipt/EXIT.json',
                 stage / f'slurm-{job_id}.err', stage / 'output/NFE.jsonl', encoding_path, metadata_path)),
            'missing exact interrupted evidence hashes')
    meta = json.loads(evidence[str(metadata_path.relative_to(stage))])
    initial = dict(meta['identity'])
    anchor = traces[last.identifier, 2]
    require((meta['source'], meta['family'], meta['expert'], meta['layer'], meta['tokens'])
            == (last.source, 'CD', 'low', 0, 31668)
            and initial.pop('transaction') == last.identifier
            and initial['noised'] == anchor['model_input']
            and initial['steps'] == [scope['schedule'][2], scope['schedule'][3]],
            'original prepared pair identity differs')
    return {'job_id': job_id, 'sources': scope['sources'], 'transaction': last.identifier, 'transactions': transactions,
            'encoding': json.loads(evidence[str(encoding_path.relative_to(stage))]), 'initial': initial, 'anchor': anchor,
            'attempts': len(calls), 'completions': len(traces), 'committed_nfe': len(calls) - 2}


class MatchingRetry:
    """Leave the native implementation unchanged and check the reproducible inputs."""
    def __init__(self, native, previous):
        self.native = native
        self.previous = previous

    def __getattr__(self, name):
        return getattr(self.native, name)

    def encode_source(self, record):
        result = self.native.encode_source(record)
        require(result['identity'] == self.previous['encoding'], 'retry encoding differs from original')
        return result

    def prepare_transaction(self, encoded, record, txn):
        require(txn.identifier == self.previous['transaction'], 'unexpected recovery transaction')
        prepared = self.native.prepare_transaction(encoded, record, txn)
        require(prepared['identity'] == self.previous['initial'], 'retry input, noise or condition differs from original')
        return prepared

    def forward(self, prepared, txn, step, phase):
        response, trace = self.native.forward(prepared, txn, step, phase)
        if phase == 'anchor':
            require(trace == self.previous['anchor'], 'retry anchor flow or input differs from original')
        return response, trace

