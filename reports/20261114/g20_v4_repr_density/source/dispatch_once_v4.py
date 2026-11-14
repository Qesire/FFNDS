"""Submit one reviewed finite G20 V4 representation/density stage once; no retries or scope changes."""
import argparse, datetime, getpass, hashlib, json, re, subprocess, fcntl
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument('--stage', required=True)
ap.add_argument('--scope-sha', required=True)
a = ap.parse_args()
p = Path(a.stage).resolve()
dispatch_lock = (p.parent / '.g20_v4_dispatch.lock').open('a')
fcntl.flock(dispatch_lock, fcntl.LOCK_EX)
if sha(p / 'scope.json') != a.scope_sha:
    raise RuntimeError('scope hash')
s = json.loads((p / 'scope.json').read_text())
r = json.loads((p / 'INDEPENDENT_REVIEW.json').read_text())
act = json.loads((p / 'ACTIVE_RESEARCH.json').read_text())
if s['phase'] != 'G20_V4_REPR_DENSITY' or s['new_videos'] != 0 \
        or s['max_new_D4_full_executions'] != 0 \
        or s['reference_generation'] != 0 or s['reference_benchmarks'] != 0:
    raise RuntimeError('phase/reference constraint')
if r.get('scope_sha256') != a.scope_sha or r.get('disposition') != 'APPROVE' \
        or r.get('recheck_disposition') != 'APPROVE' or not r.get('reviewer_agent_id') \
        or r['reviewer_agent_id'] in s['authors'] \
        or r.get('root_self_audit_sha256') != sha(p / 'ROOT_SELF_AUDIT.txt'):
    raise RuntimeError('review')
if act['scope_sha256'] != a.scope_sha or act['review_sha256'] != sha(p / 'INDEPENDENT_REVIEW.json') \
        or act['state'] != 'AUTHORIZED_FOR_ONE_JOB' or act['session_max_gpus'] != 2:
    raise RuntimeError('ACTIVE')
for n, d in {**s['code_sha256'], **s['frozen_inputs_sha256']}.items():
    f = (p / n).resolve()
    if not f.is_relative_to(p) or sha(f) != d:
        raise RuntimeError('input ' + n)
if any((p / n).exists() for n in ('SUBMITTED.json', 'DISPATCH_INTENT.json', 'RUN.json', 'output')):
    raise RuntimeError('already attempted')
queue = subprocess.run(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%b|%D'],
                       check=True, capture_output=True, text=True).stdout
lines = queue.strip().splitlines()
if len(lines) >= 2:
    raise RuntimeError('session two-job cap')
reserved = 0
for line in lines:
    _, _, gres, nodes = line.split('|')
    match = re.fullmatch(r'(?:gres/)?gpu(?::[A-Za-z0-9_-]+)?:(\d+)', gres.strip())
    if not match:
        raise RuntimeError('unknown existing GPU allocation')
    reserved += int(match.group(1)) * int(nodes)
if reserved + 1 > 2:
    raise RuntimeError('session two-GPU cap')
cmd = ['sbatch', '--parsable', '--partition=P107-A100', '--account=competition',
       '--qos=qos_p107-a100', '--no-requeue', '--gres=gpu:A100:1',
       '--nodelist=' + ','.join(s['node']), '--job-name=' + s['scope_id'].lower(),
       '--chdir=' + str(p), '--output=' + str(p / 'slurm-%j.out'),
       '--error=' + str(p / 'slurm-%j.err'), str(p / 'job.sh'), str(p), a.scope_sha]
record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope_sha256': a.scope_sha, 'queue': queue, 'command': cmd,
          'dispatcher_sha256': sha(Path(__file__))}
with (p / 'DISPATCH_INTENT.json').open('x') as f:
    json.dump(record, f, indent=2)
proc = subprocess.run(cmd, capture_output=True, text=True)
record.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
job = proc.stdout.strip().split(';')[0]
if proc.returncode == 0 and re.fullmatch('[1-9][0-9]*', job):
    record['job_id'] = job
with (p / 'SUBMITTED.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
if 'job_id' not in record:
    raise SystemExit(2)