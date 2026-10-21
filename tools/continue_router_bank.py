"""Queue one fixed RB1 proxy experiment after both supplemental arms finish."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from ar2_checkpoint import atomic_json, digest
from run_on_worker import ROOT


def main():
    r = ROOT / 'runs/worker_restart_20261020'
    out = r / 'router_bank_rb1_controller'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = {'status': 'RUNNING', 'stage': 'WAIT_FOR_FROZEN_DISJOINT20_SUPPLEMENT',
            'source_sha256': digest(Path(__file__)), 'quality_or_speed_acceptance': False}
    atomic_json(out / 'RUN.json', meta)
    try:
        prerequisite = r / 'disjoint20_pipeline/RUN.json'
        while True:
            value = json.loads(prerequisite.read_text()) if prerequisite.exists() else {}
            if value.get('status') == 'COMPLETED_FROZEN_SOURCE_DISJOINT20_SUPPLEMENT':
                break
            if value.get('status', '').startswith(('FAILED', 'NOT_RUN')):
                raise RuntimeError('Finish existing supplemental evaluation before using another GPU')
            time.sleep(10)
        meta['stage'] = 'RUN_FROZEN_FOUR_FOLD_C_SCORE_PROXY_ONLY'
        atomic_json(out / 'RUN.json', meta)
        command = [sys.executable, str(ROOT / 'tools/run_on_worker.py'), '--gpu', '0', '--',
            'python', 'tools/run_router_bank_feasibility.py', '--collections',
            str(r / 'onpolicy_calibration24/shard0/RUN.json'), str(r / 'onpolicy_calibration24/shard1/RUN.json'),
            '--student-policy', str(r / 'calibration24_residual_fit/C24_RR_policy.json'),
            '--derived-report', str(r / 'dense_v2_native_replay/RUN.json'),
            '--out', '/var/tmp/ffnds-xiang/huggingface/ffnds-derived/router_bank_rb1_20261021',
            '--report', str(r / 'router_bank_rb1/RUN.json')]
        atomic_json(out / 'command.json', {'command': command})
        for attempt in range(1, 9):
            with (out / f'attempt{attempt}.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 3:
                time.sleep(15)
                continue
            if result.returncode:
                raise RuntimeError('RB1 failed; preserve source and logs before any retry')
            break
        else:
            raise RuntimeError('GPU0 remained occupied; no RB1 coefficients changed')
        result = json.loads((r / 'router_bank_rb1/RUN.json').read_text())
        if result['status'] not in ['COMPLETED_RB1_PROXY_GATE_PASSED', 'COMPLETED_RB1_PROXY_GATE_REJECTED']:
            raise RuntimeError('RB1 did not complete its registered four-fold proxy experiment')
        meta.update(status='COMPLETED_RB1_PROXY_EXPERIMENT', stage='COMPLETE', result=result)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(out / 'RUN.json', meta)


if __name__ == '__main__':
    main()
