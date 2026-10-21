"""Run fixed OP2R replay controls after the complete OP2D state description."""
import json
from pathlib import Path
import subprocess
import sys
import time

from ar2_checkpoint import atomic_json, digest
from run_on_worker import ROOT


def main():
    runs = ROOT / 'runs/worker_restart_20261020'
    out = runs / 'op2r_replay_controller'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze_path = ROOT / 'reports/worker_20261020/OP2R_before_native_replay_source_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    meta = {'status': 'WAITING_COMPLETE_OP2D_STATE_DESCRIPTION', 'source_sha256': digest(Path(__file__)),
        'source_freeze_sha256': digest(freeze_path), 'quality_or_speed_acceptance': False}
    atomic_json(out / 'RUN.json', meta)
    try:
        while True:
            parent = runs / 'op2d_distribution_controller/RUN.json'
            if parent.exists():
                state = json.loads(parent.read_text())['status']
                if state.startswith('FAILED'):
                    raise RuntimeError('Native state description failed; replay controls are not run')
                if state == 'COMPLETED_OP2D_CPU_DESCRIPTION_ONLY':
                    break
            time.sleep(30)
        for name, sha in freeze['sources'].items():
            if digest(ROOT / name) != sha:
                raise RuntimeError('Registered replay source changed')
        if digest(ROOT / 'config/ub1_native_replay_op2r.json') != freeze['protocol_sha256']:
            raise RuntimeError('Registered native replay scope changed')
        meta['status'] = 'RUNNING_OP2R_NATIVE_REPLAY_ONLY'
        atomic_json(out / 'RUN.json', meta)
        command = [sys.executable, str(ROOT / 'tools/run_on_worker.py'), '--gpu', '0', '--',
            'python', 'tools/verify_ub1_native_replay.py', '--out', str(runs / 'op2r_native_replay')]
        for attempt in range(120):
            with (out / 'remote.log').open('ab') as handle:
                result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                break
            if result.returncode != 3:
                raise RuntimeError('Native replay failed; preserve the report and log')
            time.sleep(10)
        else:
            raise RuntimeError('Worker GPU0 did not become available')
        path = runs / 'op2r_native_replay/RUN.json'
        deadline = time.monotonic() + 90
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(2)
        value = json.loads(path.read_text())
        if value['status'] != 'PASS_OP2R_ALL_FOUR_STEPS_BOTH_NATIVE_STATE_CONTROLS':
            raise RuntimeError('Native replay scope did not pass completely')
        meta.update(status='PASS_OP2R_NATIVE_REPLAY_CONTROLS_ONLY', evidence_sha256=digest(path))
    except BaseException as error:
        meta.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(out / 'RUN.json', meta)


if __name__ == '__main__':
    main()
