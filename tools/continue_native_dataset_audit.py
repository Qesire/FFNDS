"""Wait for all OP2 inputs, then audit every immutable payload on worker CPU."""
import json
from pathlib import Path
import shlex
import subprocess
import time

from ar2_checkpoint import atomic_json, digest
from run_on_worker import ROOT, SSH, wait_for_source_visibility


def main():
    runs = ROOT / 'runs/worker_restart_20261020'
    out = runs / 'op2a_audit_controller'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze_path = ROOT / 'reports/worker_20261020/OP2A_before_complete_payload_audit_source_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    meta = {'status': 'WAITING_COMPLETE_OP2_NATIVE_COLLECTION', 'source_sha256': digest(Path(__file__)),
            'source_freeze_sha256': digest(freeze_path), 'quality_or_speed_acceptance': False}
    atomic_json(out / 'RUN.json', meta)
    try:
        while True:
            parent = runs / 'op2_native_controller/RUN.json'
            if parent.exists():
                state = json.loads(parent.read_text())['status']
                if state.startswith('FAILED'):
                    raise RuntimeError('Native collection failed; full dataset audit is not run')
                if state == 'COMPLETED_OP2_UB1_NATIVE_CALIBRATION24_ONLY':
                    break
            time.sleep(30)
        for name, sha in freeze['sources'].items():
            if digest(ROOT / name) != sha:
                raise RuntimeError('Frozen full dataset audit source changed')
        if digest(ROOT / 'config/native_dataset_audit_op2a.json') != freeze['protocol_sha256']:
            raise RuntimeError('Frozen native dataset audit contract changed')
        wait_for_source_visibility()
        command = ['bash', str(ROOT / 'scripts/worker_cpu_exec.sh'), 'python',
                   'tools/audit_ub1_native_dataset.py', '--out', str(runs / 'op2a_dataset_integrity')]
        meta['status'] = 'RUNNING_COMPLETE_CPU_PAYLOAD_AUDIT'
        atomic_json(out / 'RUN.json', meta)
        with (out / 'remote.log').open('wb') as handle:
            subprocess.run(SSH + [shlex.join(command)], stdout=handle, stderr=subprocess.STDOUT, check=True)
        path = runs / 'op2a_dataset_integrity/RUN.json'
        deadline = time.monotonic() + 90
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(2)
        report = json.loads(path.read_text())
        if report['status'] != 'PASS_OP2A_ALL24_FULL3840_NATIVE_PAYLOAD_INTEGRITY':
            raise RuntimeError('Full native dataset payload audit did not pass')
        meta.update(status='PASS_OP2A_COMPLETE_NATIVE_DATASET_AUDIT', evidence_sha256=digest(path))
    except BaseException as error:
        meta.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(out / 'RUN.json', meta)


if __name__ == '__main__':
    main()
