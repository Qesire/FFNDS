"""Run the registered CPU state description only after all OP2 sources complete."""
import json
from pathlib import Path
import shlex
import subprocess
import time

from ar2_checkpoint import atomic_json, digest
from run_on_worker import ROOT, SSH


def main():
    runs = ROOT / 'runs/worker_restart_20261020'
    out = runs / 'op2d_distribution_controller'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze_path = ROOT / 'reports/worker_20261020/OP2D_before_native_distribution_source_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    meta = {'status': 'WAITING_COMPLETE_OP2_NATIVE_COLLECTION', 'source_sha256': digest(Path(__file__)),
            'source_freeze_sha256': digest(freeze_path), 'quality_or_speed_acceptance': False}
    atomic_json(out / 'RUN.json', meta)
    try:
        while True:
            path = runs / 'op2_native_controller/RUN.json'
            if path.exists():
                report = json.loads(path.read_text())
                if report['status'].startswith('FAILED'):
                    raise RuntimeError('OP2 native collection failed; CPU state description is not run')
                if report['status'] == 'COMPLETED_OP2_UB1_NATIVE_CALIBRATION24_ONLY':
                    break
            time.sleep(30)
        for name, sha in freeze['sources'].items():
            if digest(ROOT / name) != sha:
                raise RuntimeError('Registered native distribution producer changed')
        if digest(ROOT / 'config/onpolicy_ub1_distribution_op2d.json') != freeze['protocol_sha256']:
            raise RuntimeError('Registered native distribution protocol changed')
        command = ['bash', str(ROOT / 'scripts/worker_cpu_exec.sh'), 'python', 'tools/compare_ub1_native_states.py',
                   '--out', str(runs / 'op2d_native_distribution')]
        meta['status'] = 'RUNNING_CPU_NATIVE_DISTRIBUTION_DESCRIPTION'
        atomic_json(out / 'RUN.json', meta)
        with (out / 'remote.log').open('wb') as handle:
            subprocess.run(SSH + [shlex.join(command)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        path = runs / 'op2d_native_distribution/RUN.json'
        deadline = time.monotonic() + 90
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(2)
        report = json.loads(path.read_text())
        if report['status'] != 'COMPLETED_OP2D_NATIVE_STATE_DISTRIBUTION_DESCRIPTION':
            raise RuntimeError('Native distribution description did not finish')
        meta.update(status='COMPLETED_OP2D_CPU_DESCRIPTION_ONLY', evidence_sha256=digest(path))
    except BaseException as error:
        meta.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(out / 'RUN.json', meta)


if __name__ == '__main__':
    main()
