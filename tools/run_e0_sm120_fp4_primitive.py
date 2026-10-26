"""Two bounded official NVFP4 primitive checks; never a Wan speed/quality test.

No Torch, model or video imports. The orchestrator owns no CUDA context; NVML
accepts only its one live C++ child. Wrong-architecture exit0 is rejected.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

from worker_identity import private_config_path, require_worker
from startup_gpu_reservation import gpu_uuid

ROOT = Path(__file__).resolve().parents[1]
ID = 'E0_SM120_NVFP4_PRIMITIVE_20261026'
CASES = [{'m': 128, 'n': 128, 'k': 128, 'alpha': 1., 'beta': 0., 'iterations': 0},
         {'m': 128, 'n': 256, 'k': 512, 'alpha': .5, 'beta': .5, 'iterations': 3}]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_pass(returncode, output):
    # The upstream example deliberately returns0 on unsupported hardware.
    return returncode == 0 and 'Disposition: Passed' in output and 'Disposition: Failed' not in output


def foreign_pids(rows, allowed):
    return sorted({int(row[0].strip()) for row in rows if row} - set(allowed))


def query(index, argument):
    p = subprocess.run(['nvidia-smi', '-i', index, argument, '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True, timeout=10)
    return list(csv.reader(io.StringIO(p.stdout)))


def preflight(path):
    plan = json.loads(path.read_text())
    if (plan['id'] != ID or plan['cases'] != CASES or plan['max_GPU_cards'] != 1
            or plan['max_child_processes'] != 2 or plan['timeout_seconds_per_child'] != 120
            or plan['automatic_retry'] is not False or plan['max_HF_output_bytes'] != 1048576):
        raise ValueError('Exact two-case primitive scope required')
    for name, digest in plan['bound_sources'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Bound primitive source changed')
    build_path = Path(plan['source_build_RUN'])
    if sha(build_path) != plan['source_build_RUN_sha256']:
        raise ValueError('Original compiler receipt changed')
    build = json.loads(build_path.read_text())
    if (build['status'] != 'PASS_COMPILER_ONLY_NATIVE_NOT_RUN'
            or build['id'] != plan['source_build_id']
            or build['protocol_sha256'] != plan['source_build_protocol_sha256']
            or build['source_revision'] != plan['source_revision']
            or build['binary_relative'] != plan['binary_relative']):
        raise ValueError('Successful compiler-only evidence required')
    build_protocol = ROOT / plan['source_build_protocol']
    if sha(build_protocol) != plan['source_build_protocol_sha256']:
        raise ValueError('Compiler protocol binding changed')
    source_plan = json.loads(build_protocol.read_text())
    if (source_plan['source_revision'] != plan['source_revision']
            or source_plan['target'] != plan['target']
            or source_plan['canonical_output'] != str(build_path.parent)
            or source_plan['configure_arguments'] != plan['configure_arguments']):
        raise ValueError('Compiler source/target/configuration mismatch')
    binary = build_path.parent / build['binary_relative']
    if sha(binary) != plan['binary_sha256'] or plan['binary_sha256'] != build['binary_sha256']:
        raise ValueError('Native executable changed')
    return plan, binary


def execute(path, plan, binary):
    active = json.loads((ROOT / 'research/ACTIVE_RESEARCH.json').read_text())
    allowed = [s for s in active['experiment_authority']['current_user_authorized_scopes']
               if s['id'] == ID and s['protocol_sha256'] == sha(path)
               and s['execution_status'] == 'AUTHORIZED_NOT_STARTED']
    if len(allowed) != 1 or sha(private_config_path()) != plan['private_worker_config_sha256']:
        raise RuntimeError('Exact primitive scope and private execution binding required')
    require_worker()
    index = os.environ.get('FFNDS_PHYSICAL_GPU', '')
    if index not in ['0', '1', '2', '3'] or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid(index):
        raise ValueError('One UUID-selected locked idle GPU required')
    hardware = query(index, '--query-gpu=name,compute_cap,memory.total,driver_version')
    if len(hardware) != 1 or 'Blackwell' not in hardware[0][0] or hardware[0][1].strip() != '12.0':
        raise ValueError('SM120 Blackwell primitive scope required')
    out = Path(plan['canonical_output']); out.mkdir(parents=True, exist_ok=False)
    run = {'id': ID, 'status': 'STARTED', 'protocol_sha256': sha(path),
           'binary_sha256': sha(binary), 'hardware': hardware, 'cases': [],
           'GPU_cards': 1, 'model_calls': 0, 'Wan_FFN_calls': 0, 'new_videos': 0,
           'automatic_retry': False, 'quality_accepted': False, 'observed_resources': [],
           'scope': 'SYNTHETIC_ALREADY_PACKED_E2M1_WITH_SCALES; NO_BF16_QUANTIZATION_OR_WAN_SPEED_CLAIM'}
    save = lambda: (out / 'RUN.json').write_text(json.dumps(run, indent=2) + '\n')
    save(); started = time.monotonic(); child = None; entry = None; beginning = None
    try:
        for number, case in enumerate(CASES):
            if foreign_pids(query(index, '--query-compute-apps=pid'), set()):
                raise RuntimeError('Assigned card is occupied before native child')
            command = [str(binary)] + [f'--{k}={v}' for k, v in case.items()]
            child = None
            entry = {'case': case, 'status': 'LAUNCH_ATTEMPTED',
                     'launch_attempted': True, 'child_started': False, 'child_PID': None,
                     'command': command, 'GEMMs_reserved': 1 + case['iterations'],
                     'GEMMs_completed': 'UNKNOWN_UNTIL_NATIVE_PASS'}
            run['cases'].append(entry); save()
            logfile = out / f'case{number}.log'; beginning = time.monotonic()
            with logfile.open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                entry.update(status='CHILD_STARTED', child_started=True, child_PID=child.pid)
                save()
                while True:
                    pids = query(index, '--query-compute-apps=pid,used_memory')
                    memory = query(index, '--query-gpu=memory.used')
                    foreign = foreign_pids(pids, {child.pid})
                    run['observed_resources'].append({'elapsed_seconds': time.monotonic()-started,
                        'case_index': number, 'own_child_visible': any(int(r[0].strip()) == child.pid for r in pids if r),
                        'foreign_count': len(foreign), 'physical_used_MiB': int(memory[0][0].strip())})
                    if foreign:
                        raise RuntimeError('Foreign compute appeared on assigned card')
                    if time.monotonic()-beginning > 120:
                        raise TimeoutError('Fixed primitive child time budget')
                    if child.poll() is not None:
                        break
                    time.sleep(.2)
            text = logfile.read_text(); entry.update(returncode=child.returncode,
                wall_seconds=time.monotonic()-beginning, log_sha256=sha(logfile),
                native_disposition_pass=native_pass(child.returncode, text))
            if not entry['native_disposition_pass']:
                raise RuntimeError('Native correctness disposition absent or failed')
            entry.update(status='PASS_NATIVE_PACKED_PRIMITIVE', GEMMs_completed=1 + case['iterations'])
            child = None; save()
        run['status'] = 'PASS_PACKED_PRIMITIVE_NOT_WAN_OR_QUANTIZATION_VALIDATION'
    except BaseException as error:
        run.update(status='FAILED_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        if entry is not None and entry['status'] != 'PASS_NATIVE_PACKED_PRIMITIVE':
            entry.update(status='FAILED_CHILD_TERMINAL' if entry['child_started'] else 'FAILED_BEFORE_CHILD_START',
                         returncode=None if child is None else child.returncode,
                         wall_seconds=None if beginning is None else time.monotonic()-beginning,
                         error_type=run.get('error_type'), error=run.get('error'))
        run['wall_seconds'] = time.monotonic()-started
        run['sampled_peak_physical_used_MiB'] = max((s['physical_used_MiB'] for s in run['observed_resources']), default=None)
        run['sampling_scope'] = 'NVML_CLI_POLLING_APPROX_0.2S_PLUS_QUERY_LATENCY; NOT_EXACT_ALLOCATOR_PEAK'
        run['actual_logs'] = [{'path': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size}
                              for p in sorted(out.glob('case*.log'))]
        save()
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > plan['max_HF_output_bytes']:
            run.update(status='FAILED_PRESERVED_NO_RETRY', error_type='OutputBudgetExceeded',
                       error='Primitive output cap exceeded')
            save()
            raise RuntimeError('Primitive output cap exceeded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--execute-native', action='store_true')
    args = parser.parse_args(); plan, binary = preflight(args.protocol)
    print(json.dumps({'status': 'BINARY_PROVENANCE_PREFLIGHT_ONLY', 'GPU_calls': 0}), flush=True)
    if args.execute_native:
        execute(args.protocol, plan, binary)


if __name__ == '__main__':
    main()
