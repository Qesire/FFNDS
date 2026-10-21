"""Run the preregistered TF2 paired diagnostic, with resumable per-source cases."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from ar2_checkpoint import atomic_json, digest
from run_on_worker import ROOT


def main():
    root = ROOT / 'runs/worker_restart_20261020'
    out = root / 'tf2_flow'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'CONTROLLER.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = ROOT / 'config/teacher_forced_flow_tf2.json'
    policy = root / 'ub1_step_budget_fit/UB1_policy.json'
    if not policy.exists():
        raise RuntimeError('Locate the frozen UB1 policy before running the diagnostic')
    meta = {'status': 'RUNNING', 'protocol_sha256': digest(protocol),
            'controller_sha256': digest(Path(__file__)), 'completed_arms': [],
            'quality_or_speed_acceptance': False}
    try:
        for arm in ['DQ', 'UB1']:
            meta['stage'] = arm
            atomic_json(out / 'RUN.json', meta)
            command = [sys.executable, str(ROOT / 'tools/run_on_worker.py'), '--gpu', '1', '--',
                       'python', 'tools/run_supplement_teacher_flow.py', '--reference-suite',
                       str(root / f'disjoint20_{arm}/RUN.json'), '--out', str(out / arm),
                       '--payload-root', f'/var/tmp/ffnds-xiang/huggingface/ffnds-derived/tf2_source_disjoint_flow_20261021/{arm}']
            if arm == 'UB1':
                command += ['--policy', str(policy), '--teacher-report', str(out / 'DQ/RUN.json')]
            atomic_json(out / f'{arm}_command.json', {'command': command})
            for attempt in range(1, 9):
                logpath = out / f'{arm}_controller_attempt{attempt}_{time.time_ns()}.log'
                with logpath.open('w') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 3:
                    time.sleep(15)
                    continue
                if result.returncode:
                    raise RuntimeError(f'{arm} diagnostic failed; preserve log {logpath}')
                break
            else:
                raise RuntimeError('GPU1 remained occupied; diagnostic was not modified')
            report = out / arm / 'RUN.json'
            value = json.loads(report.read_text())
            expected = ('PASS_TF2_FOUR_SOURCE_DENSE_CONTROLS' if arm == 'DQ'
                        else 'PASS_TF2_FOUR_SOURCE_CANDIDATE_SAME_DENSE_STATES')
            if value['status'] != expected:
                raise RuntimeError('Paired diagnostic did not finish all fixed sources')
            meta['completed_arms'].append({'arm': arm, 'path': str(report), 'sha256': digest(report)})
        rows = []
        candidate = json.loads((out / 'UB1/RUN.json').read_text())
        cases = []
        for record in candidate['completed']:
            if digest(record['path']) != record['sha256']:
                raise RuntimeError('Completed diagnostic case changed')
            cases.append(json.loads(Path(record['path']).read_text()))
        for step in range(4):
            row = {'step': step, 'source_count': 4}
            for key in ['flow_error', 'one_step_post_error']:
                stats = [case['steps'][step][key] for case in cases]
                sse = sum(v['squared_error'] for v in stats)
                norm = sum(v['reference_squared_norm'] for v in stats)
                row[key] = {'squared_error': sse, 'reference_squared_norm': norm,
                            'pooled_relative_RMSE': (sse / norm) ** .5}
            rows.append(row)
        meta.update(status='COMPLETED_TF2_SAME_DENSE_STATE_DIAGNOSTIC', stage='COMPLETE', per_step=rows,
                    scope='whole-model paired flow and one-step errors on identical D-Q states; no additive terminal-error decomposition or quality/speed acceptance')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(out / 'RUN.json', meta)


if __name__ == '__main__':
    main()
