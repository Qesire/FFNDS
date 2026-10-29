"""Prepare exact private jobs; root owns all ACTIVE registration and closure.

Materialization reads only protocol/source metadata and target device identities.
It never reads scientific tensors, initializes CUDA, or changes ACTIVE. A launch
requires a separately reviewed exact scope already registered by the root.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / '.runtime/chipmunk_history_encoding_20261030'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def binding(path, root=ROOT):
    path = Path(path).resolve()
    need(path.is_relative_to(root.resolve()), 'binding outside expected root')
    return {'path': str(path.relative_to(root.resolve())),
            'bytes': path.stat().st_size, 'sha256': digest(path)}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def materialize(spec):
    target = json.loads(Path(spec).read_text())
    profile = Path(target['worker_profile']).resolve()
    hf = Path(target['hf_root']).resolve()
    worker = json.loads(profile.read_text())
    need(socket.gethostname().split('.')[0] ==
         worker['execution_hostname'].split('.')[0], 'materialize on target worker')
    need(hf.is_dir() and len(target['parts']) == 4, 'four parts and existing HF root')
    raw = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        text=True, timeout=15)
    devices = {int(row[0]): row[1] for line in raw.splitlines()
               if (row := [s.strip() for s in line.split(',')])}
    sources = [binding(ROOT / path) for path in target['repo_source_paths']]
    need(len({item['path'] for item in sources}) == len(sources), 'duplicate source')
    sys.path.insert(0, str(ROOT/'tools'))
    from chipmunk_history_encoding_contract import validate_candidate
    outputs, gpu_indices = [], []
    for part, item in enumerate(target['parts']):
        folder = PRIVATE / f'part{part}'
        protocol_path = ROOT / f'config/e0_chipmunk_history_encoding_20261030_part{part}_pending_review_v1.json'
        need(not protocol_path.exists() and not folder.exists(), 'no overwrite or retry')
        draft = json.loads((ROOT / item['draft']).read_text())
        need(draft['id'] == f'E0_CHIPMUNK_HISTORY_ENCODING_20261030_PART{part}'
             and not draft['assets_finalized'], 'exact unexecuted draft')
        gpu = item['physical_gpu']
        need(type(gpu) is int and gpu in devices and gpu not in gpu_indices,
             'distinct actual GPU per case')
        affinity = item['cpu_affinity']
        need(len(affinity) == 2 and len(set(affinity)) == 2 and
             set(affinity) <= os.sched_getaffinity(0), 'two available CPU cores')
        gpu_indices.append(gpu)
        draft.update(status='REVIEW_REQUESTED_NOT_AUTHORIZED', assets_finalized=True,
                     repo_sources=sources, physical_gpu=gpu,
                     worker_config_sha256=digest(profile),
                     gpu_job_private_path=str((folder/'GPU_JOB_V1.json').relative_to(ROOT)))
        verification = f'ffnds-derived/e0_chipmunk_history_encoding_20261030/part{part}/attempt1/verification/worker/results'
        draft['verification_output_HF_relative'] = verification
        draft['read_budgets']['source_hash_bytes'] = sum(s['bytes'] for s in sources)
        validate_candidate(draft)
        protocol_digest = hashlib.sha256((json.dumps(draft, indent=2,
                                                    allow_nan=False)+'\n').encode()).hexdigest()
        jobs = {}
        for phase in ('CPU', 'GPU', 'VERIFY'):
            job_path = folder / f'{phase}_JOB_V1.json'
            scope_id = draft['id'] + {'CPU':'_CPU_PREFLIGHT', 'GPU':'', 'VERIFY':'_VERIFY'}[phase]
            owner = 'e0_chipmunk_history_encoding_verify.py' if phase == 'VERIFY' else 'e0_chipmunk_history_encoding_probe.py'
            driver = [str(ROOT/'tools'/owner), '--protocol', str(protocol_path),
                      '--hf-root', str(hf), '--job-spec', str(job_path)]
            if phase == 'CPU':
                driver.append('--preflight')
            argv = ['env', 'FFNDS_WORKER_CONFIG='+str(profile)]
            if phase == 'GPU':
                argv += ['bash', str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),
                         str(gpu), devices[gpu]]
            else:
                argv += ['CUDA_VISIBLE_DEVICES=']
            phase_affinity = affinity if phase != 'VERIFY' else affinity[:1]
            argv += ['taskset', '-c', ','.join(map(str, phase_affinity)),
                     str(ROOT/'.venv/bin/python'), '-I', '-B'] + driver
            result = hf / (draft['cpu_output_HF_relative'] if phase == 'CPU' else
                           verification if phase == 'VERIFY' else draft['output_HF_relative'])
            resources = draft[{'CPU':'cpu_preflight_resources', 'GPU':'resources',
                               'VERIFY':'verification_resources'}[phase]]
            job = {'id': scope_id, 'protocol_path':str(protocol_path),
                   'protocol_sha256':protocol_digest, 'cwd':str(ROOT), 'argv':argv,
                   'driver_argv':driver, 'cpu_affinity':phase_affinity,
                   'output_dir':str(result.parent), 'run_json':str(result/'RUN.json'),
                   'hf_root':str(hf), 'external_timeout_seconds':resources['outer_seconds']}
            if phase == 'GPU':
                job.update(gpu_launcher='scripts/worker_teal_assigned_uuid_exec.sh',
                           physical_gpu=gpu, gpu_uuid=devices[gpu],
                           checkpoint=str(Path(item['checkpoint']).resolve()))
            need(result.is_relative_to(hf) and not result.parent.exists(), 'fresh HF output')
            jobs[phase] = (job_path, job)
        write_new(protocol_path, draft)
        need(digest(protocol_path) == protocol_digest, 'protocol serialization identity')
        refs = {}
        for phase, (path, job) in jobs.items():
            write_new(path, job)
            refs[phase] = binding(path)
        outputs.append({'part':part, 'protocol':binding(protocol_path), 'jobs':refs})
    receipt = {'status':'MATERIALIZED_NOT_REGISTERED_NOT_EXECUTED', 'parts':outputs,
               'worker_profile_sha256':digest(profile), 'root_owns_ACTIVE':True}
    write_new(PRIVATE/'MATERIALIZATION_V1.json', receipt)
    return {'status':receipt['status'], 'parts':len(outputs)}


def start(kind, part):
    need(part in range(4), 'one explicit part')
    sys.path.insert(0, str(ROOT/'tools'))
    import durable_worker_job
    phase = {'runCPU':'CPU', 'runGPU':'GPU', 'runVERIFY':'VERIFY'}[kind]
    path = PRIVATE/f'part{part}'/f'{phase}_JOB_V1.json'
    job = durable_worker_job.validate(path, digest(path))
    profile_token = next(s for s in job['argv'] if s.startswith('FFNDS_WORKER_CONFIG='))
    os.environ['FFNDS_WORKER_CONFIG'] = profile_token.split('=',1)[1]
    from worker_identity import require_worker
    require_worker()
    # In particular, do not count a detached START as experiment completion.
    Path(job['output_dir']).parent.mkdir(parents=True, exist_ok=True)
    return {'status':'ACTUAL_START_ACCEPTED_NOT_COMPLETION',
            **durable_worker_job.start(path, digest(path))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('materialize','runCPU','runGPU','runVERIFY'))
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--part', type=int)
    args = parser.parse_args()
    need(args.spec is not None if args.action == 'materialize' else args.part is not None,
         'spec or explicit part required')
    print(json.dumps(materialize(args.spec) if args.action == 'materialize'
                     else start(args.action, args.part)))
