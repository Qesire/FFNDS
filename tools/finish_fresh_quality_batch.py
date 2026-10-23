"""Finish the already-started finite batch without another chat round trip.

One generator is already running. Wait for its exact receipt, decode on CPU,
freeze the authorized evaluator phase, run it once, and reconstruct results.
No training, model retry, replacement input or extra original forward.
"""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from fresh_quality_contract import ARMS, CAPS, CORE, METHOD, ROOT, STATE, atomic, read, sha, validate_registration
from worker_identity import ssh_arguments


def remote(command, log):
    with log.open('x') as stream:
        result = subprocess.run([*ssh_arguments(), shlex.join(command)], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Finite phase failed; no automatic retry: ' + log.name)


def current_registration():
    decision = read(ROOT / 'research/CURRENT_DECISION.json')
    registration = decision['registered_fresh_quality']
    errors = validate_registration(registration, ROOT)
    if (errors or not registration['active'] or registration['method_id'] != METHOD or
            decision['eligible_model_method_ids'] != [METHOD] or decision['execution_state'] != STATE):
        raise RuntimeError('Original active finite authority/source binding changed')
    return decision, registration


def close_eligibility(success, folder, hf):
    decision, registration = current_registration()
    triage = read(ROOT / 'research/20261021/method_triage.json')
    budget_path = hf / 'derived/ffnds/R17E0_fresh_quality/BUDGET.json'
    used = read(budget_path)['used'] if budget_path.exists() else {k: 0 for k in CAPS}
    base = ROOT / 'research/20261023'
    status = 'COMPLETED_FIXED_SCOPE_AWAITING_SCIENTIFIC_REVIEW' if success else 'STOPPED_FIXED_SCOPE_FAILURE_NO_AUTOMATIC_RETRY'
    evidence = {'id': METHOD, 'status': status, 'original_protocol_sha256': registration['protocol_sha256'],
                'actual_debits': used, 'new_scope_or_algorithm_authorized': False,
                'automatic_quality_or_Pareto_acceptance': False,
                'HF_relative_root': 'derived/ffnds/R17E0_fresh_quality'}
    analysis = hf / 'derived/ffnds/R17E0_fresh_quality/ANALYSIS.json'
    if success:
        evidence['analysis_HF_relative'] = str(analysis.relative_to(hf))
        evidence['analysis_sha256'] = sha(analysis)
    evidence_path = base / 'R17E0_execution_summary.json'
    if evidence_path.exists():
        raise RuntimeError('Preserve the first finite scope summary')
    atomic(evidence_path, evidence)
    registration.update(active=False, budget_spent=success, phase_completed_or_stopped=registration['phase'],
                        automatic_followup=False, result=str(evidence_path.relative_to(ROOT)))
    decision['execution_state'] = registration['preserved_model_closure']
    decision['eligible_model_method_ids'] = []
    decision['latest_fresh_validation'] = evidence
    decision['next_preparation']['status'] = status
    decision['current_handoff'] = 'research/20261023/HANDOFF_R17E0.md'
    triage['policy']['execution_state'] = decision['execution_state']
    triage['policy']['execution_note'] = 'R17E0 finite authorized validation completed or stopped; no automatic retry/followup. Original R16/R15/R14/D0/R4/R11 closures preserved. Read exact execution summary.'
    row = next(r for r in triage['methods'] if r['id'] == METHOD)
    row.update(status=status, model_execution_eligibility='NOT_ELIGIBLE_FIXED_SCOPE_FINISHED',
               result=str(evidence_path.relative_to(ROOT)))
    triage['latest_fresh_validation'] = evidence
    triage['next_preparation']['status'] = status
    atomic(ROOT / 'research/CURRENT_DECISION.json', decision)
    atomic(ROOT / 'research/20261021/method_triage.json', triage)
    handoff = base / 'HANDOFF_R17E0.md'
    handoff.write_text('# R17E0 finite fresh-source quality and native timing\n\n'
        + status + '. Actual debit counts: `' + json.dumps(used, sort_keys=True) + '`.\n\n'
        'Read [execution summary](R17E0_execution_summary.json), [immutable protocol](R17E0_registered_protocol.json), '
        '[direct authority](R17E0_authorization.json) and [original R16 smoke/evidence](HANDOFF_R16V0.md). '
        'Existing raw records and media remain in the named HF root with hashes; unreached phases have no results. '
        'No scope extension, retry, fitting or paper/Pareto acceptance follows. '
        'Agent review, public numeric export, interpretation and publication remain next.\n')
    readme = ROOT / 'README.md'
    text = readme.read_text()
    notice = '## R17E0固定验证批次已结束\n\n状态：`' + status + '`。实际扣账：`' + json.dumps(used, sort_keys=True) + ' `。'
    notice += '见[执行实录](research/20261023/R17E0_execution_summary.json)与[当前交接](research/20261023/HANDOFF_R17E0.md)。'
    notice += '原始结果留在HF缓存，尚待科学审阅和公开数值整理；没有自动宣称质量或Pareto通过。\n\n'
    readme.write_text(notice + text)
    agents = ROOT / 'AGENTS.md'
    text = agents.read_text()
    marker = '# Latest actual full160 dynamic execution — R16V0 completed (2026-10-23 CST)'
    if text.startswith('# Live scoped validation — R17E0 registered and running') and marker in text:
        agents.write_text('# R17E0 fixed validation scope finished (2026-10-23 CST)\n\n'
            + status + '. Read [execution summary](research/20261023/R17E0_execution_summary.json) '
            'and [current handoff](research/20261023/HANDOFF_R17E0.md). '
            'Model eligibility is empty; no retry or expanded model scope. '
            'Actual quality/timing interpretation and publication require reading the saved evidence.\n\n'
            + marker + text.split(marker, 1)[1])
    subprocess.run([os.environ.get('FFNDS_LOCAL_PYTHON', 'python3'), str(ROOT / 'tools/check_research_routes.py')], check=True)


def main():
    folder = ROOT / 'runs/r17_execution'
    lock = (folder / 'COORDINATOR.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    hf = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    root = hf / 'derived/ffnds/R17E0_fresh_quality'
    current_registration()
    print('Watching the single already-started145generation process; no duplicate generator', flush=True)
    seen = -1
    result = {'status': 'RUNNING_EXISTING_FINITE_BATCH', 'id': METHOD}
    try:
        while not (folder / 'GENERATOR_EXIT.json').exists():
            path = root / 'GENERATIONS.json'
            if path.exists():
                record = read(path)
                completed = len(record['completed'])
                if completed != seen:
                    print('Original generations completed:', completed, '/145', flush=True)
                    seen = completed
            time.sleep(10)
        if read(folder / 'GENERATOR_EXIT.json')['exit_code']:
            raise RuntimeError('Original generator failed; no retry')
        generations = read(root / 'GENERATIONS.json')
        budget = read(root / 'BUDGET.json')
        if (generations['status'] != 'COMPLETE_145_GENERATIONS_AWAITING_FRESH_OFFICIAL_METRICS' or
                len(generations['completed']) != 145 or budget['used']['full_generations'] != 145 or
                budget['used']['metric_evaluator_calls'] != 0):
            raise RuntimeError('Exact145generation phase incomplete')
        print('All145generations complete; decoding actual115freshvideos onCPU', flush=True)
        remote(['bash', str(ROOT / 'scripts/worker_cpu_exec.sh'), str(ROOT / '.venv/bin/python'),
                str(ROOT / 'tools/review_fresh_quality_videos.py')], folder / 'VIDEO_DECODE.log')
        decision, registration = current_registration()
        leaf = {'parent_method_id': METHOD, 'parent_protocol_sha256': registration['protocol_sha256'],
                'evaluator_call_count': 45, 'dimensions': CORE, 'arms': ARMS,
                'asset_manifest_sha256': sha(ROOT / '.runtime/R17_metric_assets.json'),
                'source_supplement_sha256': sha(ROOT / 'research/20261023/R17_metric_source_supplement.json'),
                'generation_receipt_sha256': sha(root / 'GENERATIONS.json'),
                'full_decode_receipt_sha256': sha(root / 'VIDEO_DECODE.json'),
                'phase_coordinator_sha256': sha(__file__),
                'CPU_decode_source_sha256': sha(ROOT / 'tools/review_fresh_quality_videos.py'),
                'analysis_source_sha256': sha(ROOT / 'tools/analyze_fresh_quality_results.py'),
                'status': 'IMMUTABLE_AUTHORIZED_OFFICIAL_METRIC_PHASE_BEFORE_EVALUATOR_MODEL_ENTRY'}
        target = ROOT / 'research/20261023/R17E0_metric_phase_protocol.json'
        if target.exists():
            raise RuntimeError('Preserve existing evaluator phase registration; no retry')
        atomic(target, leaf)
        registration['phase'] = 'METRICS'
        registration['metric_phase'] = {'protocol': str(target.relative_to(ROOT)), 'protocol_sha256': sha(target)}
        atomic(ROOT / 'research/CURRENT_DECISION.json', decision)
        subprocess.run(['python3', str(ROOT / 'tools/check_research_routes.py'), '--method-id', METHOD], check=True)
        print('Immutable evaluator phase bound; running exactly45officialcalls on one idleGPU', flush=True)
        # One selection before model entry; the locked wrapper repeats the
        # occupancy check. A race/occupied card causes a finite stop, not retry.
        selected = None
        for index in range(4):
            query = subprocess.run([*ssh_arguments(), shlex.join(['nvidia-smi', '-i', str(index),
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'])], capture_output=True, text=True)
            if query.returncode == 0 and not query.stdout.strip():
                selected = index
                break
        if selected is None:
            raise RuntimeError('No idle evaluator GPU available at the phase boundary')
        remote(['bash', str(ROOT / 'scripts/worker_idle_exec.sh'), str(selected), 'bash',
                str(ROOT / 'scripts/worker_eval_env.sh'), str(ROOT / '.runtime/envs/vbench_official/bin/python'),
                str(ROOT / 'tools/run_fresh_official_metrics.py')], folder / 'OFFICIAL_METRICS.log')
        print('All45officialcalls complete; reconstructing source-paired scores and proper timing', flush=True)
        remote(['bash', str(ROOT / 'scripts/worker_cpu_exec.sh'), str(ROOT / '.venv/bin/python'),
                str(ROOT / 'tools/analyze_fresh_quality_results.py')], folder / 'ANALYSIS.log')
        close_eligibility(True, folder, hf)
        result['status'] = 'COMPLETE_FIXED145_GENERATIONS_45_METRICS_AND_CPU_ANALYSIS'
    except BaseException as error:
        result.update(status='STOPPED_FIXED_SCOPE_NO_AUTOMATIC_RETRY', error_type=type(error).__name__)
        try:
            close_eligibility(False, folder, hf)
        except BaseException as close_error:
            result['close_error_type'] = type(close_error).__name__
        raise
    finally:
        atomic(folder / 'PIPELINE_EXIT.json', result)
        print(result['status'], flush=True)
        lock.close()


if __name__ == '__main__':
    main()
