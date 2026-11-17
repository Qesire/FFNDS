"""Guarded A100 fixture for the actual Chip indexed-op stream boundary."""
import argparse
import hashlib
import json
import re
import resource
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def job(stage, expected):
    if sha(stage / 'scope.json') != expected:
        raise ValueError('changed stream fixture scope')
    scope = read(stage / 'scope.json')
    review_path = stage / 'REVIEW.json'
    review = read(review_path)
    active = read(stage / 'ACTIVE_RESEARCH.json')[scope['scope_id'].lower()]
    if (scope['remote_stage'] != str(stage) or scope['operation'] != 'chip_stream_bridge_fixture'
            or review['scope_sha256'] != expected or review['reviewer_agent_id'] == review['author_agent_id']
            or review['recheck_disposition'] != 'APPROVE' or not active['runnable']
            or active['scope_sha256'] != expected or active['review_sha256'] != sha(review_path)):
        raise ValueError('exact independent approval and ACTIVE fixture required')
    names = {'chip_stream_bridge_job.py', 'chip_stream_bridge_job.sh', 'chip_stream_bridge_fixture.py',
             'chip_provider.py', 'chip_wan_voxel_tail_v1.py', 'runtime_guard.py'}
    if set(scope['bundle_sha256']) != names:
        raise ValueError('stream fixture code closure incomplete')
    for name, digest in scope['bundle_sha256'].items():
        if sha(stage / name) != digest:
            raise ValueError('changed stream fixture code: ' + name)
    if not Path(sys.executable).samefile(scope['python_executable']) or sha(sys.executable) != scope['python_executable_sha256']:
        raise ValueError('stream fixture interpreter differs')
    from runtime_guard import require_compute_allocation

    allocation = require_compute_allocation()
    values = dict(re.findall(r'(\w+)=([^\s]+)', allocation['scheduler_job_record']))
    limits = scope['resources']
    wall = re.fullmatch(r'(\d+):(\d+):(\d+)', values.get('TimeLimit', ''))
    if (wall is None or not 0 < sum(int(x)*k for x, k in zip(wall.groups(), (3600, 60, 1))) <= limits['wall_seconds']
            or values.get('Account') != limits['account'] or values.get('QOS') != limits['qos']
            or values.get('Partition') != limits['partition'] or allocation['memory_mib'] > limits['host_memory_gib']*1024):
        raise ValueError('unexpected Slurm stream fixture allocation')
    output = stage / 'output'
    if scope['output'] != str(output) or output.exists() or shutil.disk_usage(stage).free < limits['min_free_disk_gib'] * 1024**3:
        raise ValueError('stream fixture output or disk scope differs')

    import torch
    import triton

    if (torch.__version__ != scope['torch_version'] or triton.__version__ != scope['triton_version']
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA A100-SXM4-80GB'):
        raise ValueError('unexpected actual runtime or GPU for stream fixture')
    device = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, limits['gpu_memory_gib']*1024**3/device.total_memory))
    started = time.monotonic()

    def check_resources():
        if time.monotonic() - started > limits['soft_seconds']:
            raise TimeoutError('bounded selector fixture timed out')
        if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits['host_memory_gib'] * 1024**3
                or torch.cuda.max_memory_reserved() > limits['gpu_memory_gib'] * 1024**3):
            raise MemoryError('bounded selector fixture memory exceeded')

    from chip_stream_bridge_fixture import run

    with torch.inference_mode():
        result = run(scope, authorize=lambda _: None, check_resources=check_resources)
    check_resources()
    print(json.dumps({'result': result, 'summary_sha256': sha(output/'SUMMARY.json'), 'allocation': allocation},
                     allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', type=Path)
    parser.add_argument('scope_sha256')
    args = parser.parse_args()
    job(args.stage.resolve(), args.scope_sha256)
