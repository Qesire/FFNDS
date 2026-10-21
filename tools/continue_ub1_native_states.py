"""Wait for strict OP2 native controls, then collect the two UB1 calibration shards."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from ar2_checkpoint import atomic_json, canonical_json, digest
from run_on_worker import ROOT

RUNS = ROOT / 'runs/worker_restart_20261020'


def validate_controls():
    freeze_path = ROOT / 'reports/worker_20261020/OP2_before_native_control_source_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    if digest(ROOT / 'config/onpolicy_ub1_op2.json') != freeze['protocol_sha256']:
        raise RuntimeError('Registered native collection protocol changed')
    for name, sha in freeze['sources'].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError('Control/collection producer changed after freeze')
    bindings = []
    identities = []
    for arm in ['DQ', 'UB1']:
        path = RUNS / f'op2_native_controls/{arm}/RUN.json'
        if not path.exists():
            return None
        report = json.loads(path.read_text())
        if report['status'].startswith('FAILED'):
            raise RuntimeError('Native control rejected; preserve its original report')
        if report['status'] != 'PASS_OP2_BOTH_FIXED_NATIVE_CONTROLS':
            return None
        identity = report['identity']
        import hashlib
        if (hashlib.sha256(canonical_json(identity).encode()).hexdigest() != report['fingerprint']
                or identity['protocol_sha256'] != freeze['protocol_sha256']
                or identity['arm'] != arm or identity['indices'] != [0, 10]
                or {c['index'] for c in report['completed']} != {0, 10}):
            raise RuntimeError('Native control arm, protocol, fingerprint or source scope differs')
        for name, sha in identity['sources'].items():
            if digest(name) != sha:
                raise RuntimeError('Native control numerical producer changed')
        for entry in report['completed']:
            if digest(entry['input_report']) != entry['input_report_sha256']:
                raise RuntimeError('Native control input changed')
            case = json.loads(Path(entry['input_report']).read_text())
            if (case['status'] != 'PASS_OP2_NATIVE_INPUT_AND_NOOP' or not case['original_reference_bitwise']
                    or case['trajectories']['plain'] != case['trajectories']['observed']
                    or len(case['scopes']) != 160 or case['offline_extra_teacher_GEMMs'] != 320):
                raise RuntimeError('Incomplete native operator/no-op evidence')
            binding = entry['original_reference_binding']
            if digest(binding['case_report']) != binding['case_report_sha256']:
                raise RuntimeError('Original native reference changed')
            for video in case['videos'].values():
                if digest(video['path']) != video['sha256']:
                    raise RuntimeError('Native control video changed')
        identities.append(identity)
        bindings.append({'arm': arm, 'path': str(path), 'sha256': digest(path)})
    for key in ['assets', 'inputs_sha256', 'recovery_sha256', 'original_UB1_policy_sha256', 'UB1_execution_alias_sha256', 'sources']:
        if identities[0][key] != identities[1][key]:
            raise RuntimeError('The two native control arms do not share the same identity')
    return bindings


def collect(shard):
    log = ROOT / f'runs/bootstrap_20261020/op2_calibration_shard{shard}.log'
    command = [sys.executable, str(ROOT / 'tools/run_on_worker.py'), '--gpu', str(shard), '--',
               'python', 'tools/collect_ub1_native_states.py', '--stage', 'calibration', '--arm', 'UB1',
               '--shard', str(shard), '--out', str(RUNS / f'op2_ub1_calibration24/shard{shard}')]
    for attempt in range(120):
        with log.open('ab') as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return {'shard': shard, 'log': str(log), 'attempts': attempt + 1}
        if result.returncode != 3:
            raise RuntimeError(f'UB1 native shard{shard} failed; inspect its preserved log')
        time.sleep(10)
    raise RuntimeError('Configured worker GPU did not become available')


def main():
    out = RUNS / 'op2_native_controller'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = {'status': 'WAITING_NATIVE_CONTROLS', 'source_sha256': digest(Path(__file__)),
            'quality_or_speed_acceptance': False}
    atomic_json(out / 'RUN.json', meta)
    try:
        while True:
            controls = validate_controls()
            if controls:
                break
            time.sleep(20)
        meta.update(status='COLLECTING_UB1_CALIBRATION24', control_bindings=controls)
        atomic_json(out / 'RUN.json', meta)
        with ThreadPoolExecutor(max_workers=2) as pool:
            meta['shards'] = list(pool.map(collect, [0, 1]))
        meta['status'] = 'COMPLETED_OP2_UB1_NATIVE_CALIBRATION24_ONLY'
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(out / 'RUN.json', meta)


if __name__ == '__main__':
    main()
