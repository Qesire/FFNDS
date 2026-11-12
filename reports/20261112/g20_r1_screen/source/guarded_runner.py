"""One exact reviewed G20 R1 same-input screening job. Never submits or expands.

Verification mirrors the reviewed ICC/G19 guarded stages: live Slurm allocation,
independent review + recheck, ACTIVE authorization, frozen code/input hashes,
single A100 80GB, budget ceilings. Computation runs in two passes (80-site cheap
landscape, then six-site expensive fits) plus a pre-registered conditional
supplementary site. No new generation, no model forward beyond native FFN replay
on archived paired rows, no video, no benchmark.
"""
import argparse
import datetime
import json
import re
import shutil
import signal
import sys
import traceback
from pathlib import Path

from runtime_guard import require_compute_allocation, sha256_file as sha, process_peak_rss_bytes


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--scope-sha', required=True)
    a = ap.parse_args()
    stage = Path(a.stage).resolve()
    if stage != Path(__file__).resolve().parent or sha(stage / 'scope.json') != a.scope_sha:
        raise RuntimeError('stage/scope identity')
    s = json.loads((stage / 'scope.json').read_text())
    r = json.loads((stage / 'INDEPENDENT_REVIEW.json').read_text())
    active = json.loads((stage / 'ACTIVE_RESEARCH.json').read_text())
    if s['phase'] != 'G20_R1_SAME_INPUT_SCREEN' or s['status'] != 'REVIEW_CANDIDATE' \
            or s['remote_stage'] != str(stage) or s['new_videos'] != 0 \
            or s['reference_generation'] != 0 or s['reference_benchmarks'] != 0:
        raise RuntimeError('phase/reference constraint')
    if r.get('scope_sha256') != a.scope_sha or r.get('disposition') != 'APPROVE' \
            or r.get('recheck_disposition') != 'APPROVE' or not r.get('reviewer_agent_id') \
            or r['reviewer_agent_id'] in s['authors'] \
            or r.get('root_self_audit_sha256') != sha(stage / 'ROOT_SELF_AUDIT.txt'):
        raise RuntimeError('independent review')
    if active.get('scope_sha256') != a.scope_sha or active.get('review_sha256') != sha(stage / 'INDEPENDENT_REVIEW.json') \
            or active.get('state') != 'AUTHORIZED_FOR_ONE_JOB' or active.get('session_max_gpus') != 2:
        raise RuntimeError('ACTIVE')
    for n, d in {**s['code_sha256'], **s['frozen_inputs_sha256']}.items():
        p = (stage / n).resolve()
        if not p.is_relative_to(stage) or sha(p) != d:
            raise RuntimeError('bundle hash ' + n)
    for p, d in s['external_source_sha256'].items():
        if sha(p) != d:
            raise RuntimeError('source hash ' + p)
    allocation = require_compute_allocation()
    budget = s['budget']
    if allocation['hostname'].split('.')[0] != s['node'] \
            or allocation['memory_mib'] > budget['host_memory_gib'] * 1024 \
            or Path(sys.executable).resolve() != Path(s['python_executable']).resolve():
        raise RuntimeError('allocation/runtime mismatch')
    record = dict(re.findall(r'(\w+)=([^\s]+)', allocation['scheduler_job_record']))
    t = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', record['TimeLimit'])
    dd, hh, mm, ss = (int(x or 0) for x in t.groups())
    if dd * 86400 + hh * 3600 + mm * 60 + ss > budget['wall_seconds']:
        raise RuntimeError('time limit')
    if shutil.disk_usage(stage).free < budget['min_free_disk_gib'] * 1024 ** 3:
        raise RuntimeError('free disk')
    receipt = {'scope_sha256': a.scope_sha, 'allocation': allocation,
               'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(stage / 'RUN.json', receipt)
    code = 1

    def terminate(signum, frame):
        raise TimeoutError('termination signal ' + str(signum))

    signal.signal(signal.SIGTERM, terminate)
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 \
                or torch.cuda.get_device_name() != 'NVIDIA A100-SXM4-80GB':
            raise RuntimeError('requires one A100 80GB')
        if torch.__version__ != s['torch_version']:
            raise RuntimeError('torch version mismatch')
        torch.cuda.set_per_process_memory_fraction(
            budget['gpu_memory_gib'] * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory)
        from runner_screen import run
        with torch.inference_mode():
            receipt['result'] = run(stage, s, a.scope_sha)
        receipt['peak_rss_bytes'] = process_peak_rss_bytes()
        receipt['device'] = torch.cuda.get_device_name()
        receipt['total_gpu_bytes'] = torch.cuda.get_device_properties(0).total_memory
        if receipt['peak_rss_bytes'] is None or receipt['peak_rss_bytes'] > budget['host_memory_gib'] * 1024 ** 3:
            raise RuntimeError('host memory budget exceeded')
        code = 0
    except BaseException:
        receipt['traceback'] = traceback.format_exc()
        print(receipt['traceback'], flush=True)
        raise
    finally:
        receipt['exit_code'] = code
        receipt['end_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(stage / 'EXIT.json', receipt)


if __name__ == '__main__':
    main()