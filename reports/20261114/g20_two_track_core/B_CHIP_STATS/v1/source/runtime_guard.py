"""Compute-node-only execution and exact reviewed-scope verification.

This module uses only stdlib. Call it before importing torch or reading weights.
There is deliberately no real-weight local mode or bypass environment variable.
"""
from __future__ import annotations
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import shutil
import subprocess
import time
from datetime import datetime, timezone


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _command(args):
    return subprocess.run(args, check=True, text=True, capture_output=True,
                          timeout=15).stdout.strip()


def _short(host):
    return host.lower().split('.')[0]


def _memory_mib(value):
    match = re.fullmatch(r'([0-9.]+)([KMGT])', value)
    if not match:
        raise RuntimeError('Unrecognized Slurm memory quantity')
    return float(match[1])*{'K':1/1024,'M':1,'G':1024,'T':1024**2}[match[2]]


def process_peak_rss_bytes():
    if os.name != 'posix':
        return None  # Windows synthetic tests have no resource module.
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def require_compute_allocation():
    """Verify live scheduler state, allocated host and user; reject login hosts.

    Environment membership alone is insufficient: scontrol independently supplies
    the live allocation. This does not defend against a compromised scheduler.
    """
    hostname = socket.gethostname()
    if re.search(r'(^|[-.])(login|tradmin|head|gateway)', hostname, re.I):
        raise RuntimeError('LOGIN_NODE_FORBIDDEN: ' + hostname)
    job = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch(r'[1-9][0-9]*', job):
        raise RuntimeError('A real SLURM_JOB_ID is required before weight access')
    record = _command(['scontrol', 'show', 'job', '-o', job])
    values = dict(re.findall(r'(\w+)=([^\s]+)', record))
    if values.get('JobId') != job or values.get('JobState') != 'RUNNING':
        raise RuntimeError('Scheduler does not report this job RUNNING')
    owner = re.fullmatch(r'([^()]+)\(([0-9]+)\)', values.get('UserId', ''))
    if not owner or owner[1] != getpass.getuser():
        raise RuntimeError('Scheduler job owner does not match process user')
    if hasattr(os, 'getuid') and int(owner[2]) != os.getuid():
        raise RuntimeError('Scheduler uid does not match process uid')
    nodelist = values.get('NodeList', '')
    if not nodelist or nodelist in ('(null)', 'None'):
        raise RuntimeError('Scheduler allocation has no nodes')
    hosts = _command(['scontrol', 'show', 'hostnames', nodelist]).splitlines()
    if _short(hostname) not in {_short(h) for h in hosts}:
        raise RuntimeError('Current host is outside the live Slurm allocation')
    env_nodes = os.environ.get('SLURM_JOB_NODELIST', '')
    if env_nodes:
        env_hosts = _command(['scontrol', 'show', 'hostnames', env_nodes]).splitlines()
        if {_short(h) for h in env_hosts} != {_short(h) for h in hosts}:
            raise RuntimeError('SLURM_JOB_NODELIST differs from scheduler state')
    # This runner is a one-GPU scope. Accept common Slurm TRES spellings only.
    tres = values.get('AllocTRES', '')
    generic = re.search(r'(?:^|,)gres/gpu=([0-9]+)(?:,|$)', tres)
    if not generic or int(generic[1]) != 1:
        raise RuntimeError('This scope requires exactly one allocated GPU')
    if values.get('NumNodes') != '1' or not 0 < int(values.get('NumCPUs', '0')) <= 4:
        raise RuntimeError('Allocation exceeds one node / four CPUs')
    limit = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', values.get('TimeLimit', ''))
    if not limit:
        raise RuntimeError('Unrecognized or unbounded Slurm time limit')
    days, hours, minutes, seconds = (int(x or 0) for x in limit.groups())
    if not 0 < days*86400 + hours*3600 + minutes*60 + seconds <= 10800:
        raise RuntimeError('Slurm wall time exceeds 3 hours')
    mem = re.search(r'(?:^|,)mem=([0-9.]+)([KMGT])(?:,|$)', tres)
    if re.search(r'(?:^|,)mem=', tres) and not mem:
        raise RuntimeError('Present AllocTRES memory field is unparseable; fallback forbidden')
    if mem:
        memory_mib = _memory_mib(mem[1]+mem[2])
        memory_evidence = 'AllocTRES'
    else:
        # This cluster omits memory from AllocTRES. Bind BOTH scheduler request
        # fields; this is a request, not evidence of a hard cgroup limit.
        requested = re.search(r'(?:^|,)mem=([0-9.]+[KMGT])(?:,|$)', values.get('ReqTRES',''))
        if not requested:
            raise RuntimeError('Scheduler memory request is unverifiable')
        memory_mib = _memory_mib(requested[1])
        if memory_mib != _memory_mib(values.get('MinMemoryNode','')):
            raise RuntimeError('Scheduler memory request fields disagree')
        memory_evidence = 'ReqTRES_AND_MinMemoryNode; request_only_not_hard_limit'
    if not 0 < memory_mib <= 192*1024:
        raise RuntimeError('Slurm memory request/allocation exceeds 192 GiB')
    return {'hostname': hostname, 'job_id': job, 'user': owner[1],
            'allocated_hosts': hosts, 'alloc_tres': tres,
            'memory_mib': memory_mib, 'memory_evidence': memory_evidence,
            'scheduler_job_record': record, 'checked_utc': _utc()}


def _relative_file(root, name):
    root = Path(root).resolve()
    p = (root / name).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise RuntimeError('Scope file escapes root or is absent: ' + str(name))
    return p


def verify_frozen_scope(root, scope_path, expected_scope_sha):
    root=Path(root).resolve(); scope_path=Path(scope_path).resolve()
    if not scope_path.is_relative_to(root) or sha256_file(scope_path)!=expected_scope_sha or not re.fullmatch(r'[0-9a-f]{64}',expected_scope_sha): raise RuntimeError('scope identity')
    scope=json.loads(scope_path.read_text(encoding='utf8'))
    if scope.get('status')!='REVIEW_CANDIDATE' or scope.get('phase')!='FIXED_FOLD_METRICS' or scope.get('group') not in ('A','B'): raise RuntimeError('scope identity')
    review=json.loads(_relative_file(root,'INDEPENDENT_REVIEW.json').read_text(encoding='utf8'))
    if (review.get('disposition')!='APPROVE' or review.get('recheck_disposition')!='APPROVE'
        or review.get('scope_sha256')!=expected_scope_sha or not review.get('reviewer_agent_id')
        or review['reviewer_agent_id'] in scope['authors']
        or review.get('root_self_audit_sha256')!=sha256_file(root/'ROOT_SELF_AUDIT.txt')): raise RuntimeError('review identity')
    required={'runner_panel_metrics.py','preflight_panel.py','check_candidates_preflight.py','summarize_quality.py','analysis_motion.py','reduce_quality_flows.py','metric_contract.py','metric_loaders.py','runtime_guard.py','wrapper_panel.py','job_panel_metrics.sh','dispatch_once.py'}
    if set(scope['code_sha256'])!=required: raise RuntimeError('code coverage')
    needed={'panel_manifest.json','metric_closure.json','folding_references.json','old_metric_references.json','candidate_producer_proofs.json','baseline_binding.json'}
    if not needed.issubset(scope['frozen_inputs_sha256']): raise RuntimeError('unbound metric inputs')
    for name,digest in {**scope['code_sha256'],**scope['frozen_inputs_sha256']}.items():
        if sha256_file(_relative_file(root,name))!=digest: raise RuntimeError('file hash mismatch: '+name)
    active=json.loads(_relative_file(root,scope['active_registry_file']).read_text(encoding='utf8'))
    entry=active.get('scopes',{}).get(scope['scope_id'],{})
    if entry.get('review_sha256')!=sha256_file(_relative_file(root,'INDEPENDENT_REVIEW.json')): raise RuntimeError('review SHA mismatch')
    if active.get('state')!='SCOPE_FROZEN_IN_ACTIVE' or active.get('session_max_gpus')!=2 or active.get('max_concurrent_jobs')!=2 or entry.get('scope_sha256')!=expected_scope_sha or entry.get('state')!='AUTHORIZED_FOR_ONE_JOB' or entry.get('max_jobs')!=1: raise RuntimeError('active scope mismatch')
    b=scope['budget']
    if (b['gpus_per_job'],b['max_jobs'],b['cpus'],b['host_memory_gib'],b['gpu_memory_gib'],b['wall_seconds'])!=(1,1,4,192,60,7200) or not 0<b['soft_seconds']<b['wall_seconds']: raise RuntimeError('budget mismatch')
    groups={'A':{'raft','guitar','eagle','canoe_static_00'},'B':{'bicycle','tennis','spoon'}}
    ids=groups[scope['group']]
    if scope.get('arms')!=['offline_fold'] or scope.get('seeds')!=[43,44] or any(scope.get(k)!=0 for k in ('reference_generation','reference_benchmarks','reference_metric_evaluations')): raise RuntimeError('candidate/reference scope')
    if set(scope['source_ids'])!=ids or len(scope['source_ids'])!=len(ids) or scope['source_count']!=len(ids) or scope['video_count']!=2*len(ids): raise RuntimeError('source group scope')
    if not Path(scope['output_dir']).is_absolute(): raise RuntimeError('output identity')
    scope['_verified_scope_sha256']=expected_scope_sha
    return scope

def _utc():
    return datetime.now(timezone.utc).isoformat()


class RunReceipt:
    """Exclusive directory, START/EXIT, exception/signal receipt, soft deadline.

    A Slurm SIGKILL or machine failure can prevent EXIT. Such a run remains
    incomplete; a missing EXIT must never be interpreted as success.
    """
    def __init__(self, output_dir, scope, allocation):
        self.path = Path(output_dir)
        self.scope = scope
        self.allocation = allocation
        self.started = None
        self.previous = {}
        self.result = {}

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        (self.path / 'START.json').write_text(json.dumps({
            'status': 'STARTED', 'utc': _utc(), 'scope': self.scope,
            'allocation': self.allocation}, ensure_ascii=False, indent=2), encoding='utf-8')
        for name in ('SIGTERM', 'SIGUSR1', 'SIGINT'):
            sig = getattr(signal, name, None)
            if sig is not None:
                self.previous[sig] = signal.signal(sig, self._signal)
        return self

    def _signal(self, sig, frame):
        raise RuntimeError('INTERRUPTED_SIGNAL_' + str(sig))

    def check_deadline(self):
        peak = process_peak_rss_bytes()
        if peak is not None and peak > self.scope['budget'].get('host_memory_gib',192)*1024**3:
            raise MemoryError('HOST_MEMORY_BUDGET_EXCEEDED')
        if time.monotonic() - self.started >= self.scope['budget']['soft_seconds']:
            raise TimeoutError('PARTIAL_SOFT_DEADLINE')

    def __exit__(self, kind, exc, tb):
        for sig, old in self.previous.items():
            signal.signal(sig, old)
        peak = process_peak_rss_bytes()
        terminal_memory_error = None
        if exc is None and peak is not None and peak > self.scope['budget'].get('host_memory_gib',192)*1024**3:
            terminal_memory_error = MemoryError('HOST_MEMORY_BUDGET_EXCEEDED_AT_EXIT')
            exc = terminal_memory_error
        record = {'status': 'SUCCESS' if exc is None else 'FAILED_OR_PARTIAL',
                  'utc': _utc(), 'elapsed_seconds': time.monotonic() - self.started,
                  'process_peak_rss_bytes': peak,
                  'exception': None if exc is None else repr(exc), 'result': self.result}
        with (self.path / 'EXIT.json').open('x', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False)
        if terminal_memory_error is not None:
            raise terminal_memory_error
        return False
