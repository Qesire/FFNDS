"""Admit only the completed replacement of the interrupted CD pair."""
import hashlib
import json
from pathlib import Path

from icc_calibration_driver import validate_and_plan
from icc_calibration_receipts import check_transaction, require
from icc_calibration_recovery import audit_interrupted


def validate_recovery(spec, manifest):
    previous = audit_interrupted(spec['original'], manifest)
    prefix_bytes = (json.dumps(previous['transactions'], indent=2, allow_nan=False) + '\n').encode()
    require(hashlib.sha256(prefix_bytes).hexdigest() == spec['committed_transactions_sha256'],
            'previously validated committed prefix changed')
    retry = spec['replacement']
    stage = Path(retry['stage'])
    scope_raw = (stage / 'scope.json').read_bytes()
    scope = json.loads(scope_raw)
    run = json.loads((stage / 'receipt/RUN.json').read_bytes())
    exit_record = json.loads((stage / 'receipt/EXIT.json').read_bytes())
    expected = retry['scope_sha256']
    job_id = str(retry['job_id'])
    require(hashlib.sha256(scope_raw).hexdigest() == expected
            and run['scope_sha256'] == expected and str(run['allocation']['job_id']) == job_id
            and exit_record['exit_code'] == 0 and str(exit_record['slurm_job_id']) == job_id
            and scope['cd_retry'] == spec['original'] and scope['max_nfe'] == 2,
            'replacement scope or execution receipt differs')
    txn_id = previous['transaction']
    txn = next(item for item in validate_and_plan(manifest)[0] if item.identifier == txn_id)
    manifest_raw = (stage / 'ICC_CLEAN_CAL11_BOUND.json').read_bytes()
    require(scope['sources'] == [txn.source]
            and hashlib.sha256(manifest_raw).hexdigest() == scope['bundle_sha256']['ICC_CLEAN_CAL11_BOUND.json']
            and json.loads(manifest_raw) == manifest, 'replacement source or cohort differs')
    journal = (stage / 'output/NFE.jsonl').read_bytes()
    rows = [json.loads(line) for line in journal.splitlines()]
    require(len(rows) == 4, 'replacement journal must contain two complete forwards')
    traces = {}
    for index, step in enumerate(txn.indices):
        attempt, completion = rows[index * 2:index * 2 + 2]
        require(attempt == {'event': 'NFE_ATTEMPT', 'transaction': txn_id, 'step': step, 'total': index + 1}
                and (completion['event'], completion['transaction'], completion['step']) == ('NFE_COMPLETED', txn_id, step),
                'replacement attempt or completion differs')
        traces[txn_id, step] = completion['trace']
    summary_raw = (stage / 'output/SUMMARY.json').read_bytes()
    summary = json.loads(summary_raw)
    require(summary['status'] == 'EXECUTED_ADAPTER_RESULTS_REQUIRE_VALIDATION'
            and (summary['attempted_nfe'], summary['main_nfe'], summary['diagnostic_nfe']) == (2, 2, 0)
            and set(summary['transactions']) == {txn_id}, 'missing complete atomic CD replacement')
    info = summary['transactions'][txn_id]
    encoding_path = stage / 'output/cal' / txn.source.replace(':', '_') / 'ENCODING.json'
    require(json.loads(encoding_path.read_bytes()) == previous['encoding']
            and info['initial'] == previous['initial'] and traces[txn_id, 2] == previous['anchor'],
            'persisted replacement identities or the measured step-2 anchor differ')
    result = check_transaction(txn, info, stage, traces, scope['schedule'])
    require(txn_id not in previous['transactions'], 'duplicate replacement in interrupted prefix')
    return {'status': 'PASS', 'original_job_id': previous['job_id'], 'replacement_job_id': job_id,
            'replacement_scope_sha256': expected, 'sources': previous['sources'],
            'transactions': {**previous['transactions'], txn_id: result},
            'adopted_nfe': previous['committed_nfe'] + 2,
            'attempted_nfe': previous['attempts'] + 2, 'completed_nfe': previous['completions'] + 2,
            'summary_sha256': hashlib.sha256(summary_raw).hexdigest(),
            'journal_sha256': hashlib.sha256(journal).hexdigest()}

