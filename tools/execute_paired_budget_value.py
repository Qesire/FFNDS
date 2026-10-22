"""Execute only the frozen24-case D0 attempt as worker GPUs become idle.

Uses the existing exclusive worker wrappers. Never evicts foreign processes,
changes environments, expands the experiment, or publishes automatically.
"""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from r12d0_contract import ROOT, PROTOCOL, METHOD, jobs, read, sha, prerequisites
from worker_identity import ssh_arguments

VIEW = '/var/tmp/ffnds-xiang/huggingface/ffnds-derived/python_source_views/transformers-5.6.2-5c310812199dbcdf/VIEW.json'


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def idle_gpus():
    probe = '''import subprocess,json
rows=[]
for gpu in (0,1):
    p=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if not p and free*2**20 >= 72*2**30: rows.append(gpu)
print(json.dumps(rows))
'''
    value = subprocess.check_output(ssh_arguments() + [shlex.join(['python3', '-c', probe])], text=True, timeout=45)
    return json.loads(value)


def stop_own_attempt(out):
    # Interrupt only this exact experiment's Python drivers, after verifying
    # owner, executable script argument and the frozen attempt argument.
    probe = '''import os,pathlib,signal,json
out=pathlib.Path(''' + repr(str(out)) + ''');stopped=[]
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        args=p.joinpath('cmdline').read_bytes().decode().split('\\0')
        if 'tools/run_paired_budget_value.py' not in args or '--out' not in args:continue
        if pathlib.Path(args[args.index('--out')+1]).resolve()!=out:continue
        os.kill(int(p.name),signal.SIGINT);stopped.append(int(p.name))
    except (OSError,ValueError,UnicodeError):continue
print(json.dumps({'interrupted_own_D0_driver_count':len(stopped)}))
'''
    remote = 'cd ' + shlex.quote(str(ROOT)) + ' && ' + shlex.join(['python3', '-c', probe])
    return subprocess.check_output(ssh_arguments() + [remote], text=True, timeout=45).strip()


def main():
    protocol = read(PROTOCOL)
    out = ROOT / protocol['execution_plan']['execution_root']
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'EXECUTOR.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'EXECUTOR.json').exists() or (out / 'STOP.json').exists():
        raise RuntimeError('Preserve the previous executor/attempt; no automatic restart')
    subprocess.run([sys.executable, 'tools/check_research_routes.py', '--method-id', METHOD], cwd=ROOT, check=True)
    phases = [[('controls', 'DQ', 0), ('controls', 'UB1', 0)],
              [('baselines', 'UB1', 0), ('baselines', 'UB1', 1)],
              [('interventions', 'UB1', 0), ('interventions', 'UB1', 1)]]
    status = {'status': 'WAITING_FOR_IDLE_WORKER_GPUS', 'protocol_sha256': sha(PROTOCOL),
              'completed_jobs': [], 'running_jobs': {}, 'model_jobs_launched': 0,
              'phase': 0, 'publication_performed': False}
    active = {}
    write(out / 'EXECUTOR.json', status)
    try:
        for phase_number, pending in enumerate(phases):
            pending = list(pending)
            status['phase'] = phase_number
            while pending or active:
                if sha(PROTOCOL) != status['protocol_sha256']:
                    raise RuntimeError('Registered protocol changed while queued/running')
                if (out / 'STOP.json').exists():
                    raise RuntimeError('Stop requested')
                for gpu, (process, name, logfile) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    logfile.close()
                    del active[gpu]
                    del status['running_jobs'][str(gpu)]
                    if code:
                        receipt = out / name / 'SOURCE_VIEW.json'
                        if code == 3 and not receipt.exists() and not (out / name / 'RUN.json').exists():
                            # Lock/foreign-process race was rejected before a
                            # native command or source-view child ever started.
                            pending.insert(0, next(j for j in phases[phase_number] if jobs(*j)[0] == name))
                            status['model_jobs_launched'] -= 1
                            continue
                        raise RuntimeError(f'{name} failed; see preserved job log/receipts (exit{code})')
                    job = read(out / name / 'RUN.json')
                    if job['status'] != 'PASS_ALL_REGISTERED_RUNS':
                        raise RuntimeError('Job exited without complete numeric controls')
                    status['completed_jobs'].append(name)
                    print(f'D0 completed {name}: {len(job["completed"])} native cases', flush=True)
                if pending:
                    available = [g for g in idle_gpus() if g not in active]
                    for gpu in available:
                        if not pending:
                            break
                        phase, baseline, shard = pending.pop(0)
                        name, _ = jobs(phase, baseline, shard)
                        folder = out / name
                        folder.mkdir(exist_ok=True)
                        command = [sys.executable, 'tools/run_on_worker.py', '--gpu', str(gpu), '--',
                            '.venv/bin/python', 'tools/with_transformers_source_view.py', '--manifest', VIEW,
                            '--receipt', str(folder / 'SOURCE_VIEW.json'), '--', 'python',
                            'tools/run_paired_budget_value.py', '--phase', phase, '--baseline', baseline,
                            '--shard', str(shard), '--out', str(out)]
                        logfile = (folder / f'launch{len(list(folder.glob("launch*.log"))) + 1}.log').open('w')
                        process = subprocess.Popen(command, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT)
                        active[gpu] = (process, name, logfile)
                        status['running_jobs'][str(gpu)] = name
                        status['model_jobs_launched'] += 1
                        print(f'D0 launching {name} on idle GPU{gpu}', flush=True)
                status['status'] = 'RUNNING_REGISTERED_NATIVE_JOBS' if active else 'WAITING_FOR_IDLE_WORKER_GPUS'
                status['updated_unix_time'] = time.time()
                write(out / 'EXECUTOR.json', status)
                if pending or active:
                    time.sleep(30)
            if phase_number < 2:
                identity = read(out / 'control_DQ/RUN.json')['scientific_identity']
                prerequisites(out, 'baselines' if phase_number == 0 else 'interventions', identity)
        command = ['bash', str(ROOT / 'scripts/worker_cpu_exec.sh'), '.venv/bin/python',
            'tools/with_transformers_source_view.py', '--manifest', VIEW,
            '--receipt', str(out / 'ANALYSIS_SOURCE_VIEW.json'), '--', 'python',
            'tools/analyze_paired_budget_value.py', '--out', str(out),
            '--output', 'research/20261022/R12_D0_native_results.json']
        status['status'] = 'ANALYZING_ALL24_NATIVE_CASES'
        write(out / 'EXECUTOR.json', status)
        with (out / 'ANALYSIS.log').open('w') as log:
            subprocess.run(ssh_arguments() + [shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
        status['status'] = 'COMPLETE_24_NATIVE_CASES_AND_FROZEN_ANALYSIS'
    except BaseException as error:
        write(out / 'STOP.json', {'reason': str(error), 'time': time.time()})
        status.update(status='STOPPED_PRESERVED', error=str(error))
        if active:
            status['interruption'] = stop_own_attempt(out)
        raise
    finally:
        write(out / 'EXECUTOR.json', status)


if __name__ == '__main__':
    main()
