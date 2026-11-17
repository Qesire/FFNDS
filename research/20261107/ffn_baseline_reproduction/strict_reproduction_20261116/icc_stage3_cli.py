"""Reviewed Stage-3 P512 grid entry point, to be launched inside compute allocation.

Usage: python icc_stage3_cli.py /absolute/path/to/scope.json EXPECTED_SCOPE_SHA256
This CLI does not submit jobs. A real independent REVIEW and ACTIVE are required.
"""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import sys
import time


def digest(p):
    import hashlib
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main(scope_path, expected_sha):
    scope_path = Path(scope_path).resolve()
    if digest(scope_path) != expected_sha: raise RuntimeError('scope hash mismatch')
    scope = json.loads(scope_path.read_text(encoding='utf8'))
    review_path,active_path = Path(scope['review_path']),Path(scope['active_path'])
    review = json.loads(review_path.read_text(encoding='utf8'))
    active = json.loads(active_path.read_text(encoding='utf8'))[scope['active_key']]
    if (review['scope_sha256'] != expected_sha or review['recheck_disposition'] != 'APPROVE'
        or active['scope_sha256'] != expected_sha or active['review_sha256'] != digest(review_path)
        or not active['runnable']):
        raise RuntimeError('independent review / ACTIVE missing or changed')
    guard = Path(scope['runtime_guard_path']).resolve()
    if digest(guard) != scope['runtime_guard_sha256']: raise RuntimeError('runtime guard identity mismatch')
    sys.path.insert(0,str(guard.parent))
    from runtime_guard import require_compute_allocation
    allocation = require_compute_allocation()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('missing actual Slurm allocation')
    resources = scope['resources']
    job = dict(re.findall(r'(\w+)=([^\s]+)', allocation['scheduler_job_record']))
    wall = re.fullmatch(r'(\d+):(\d+):(\d+)', job['TimeLimit'])
    if wall is None or sum(int(x)*w for x,w in zip(wall.groups(),(3600,60,1))) > resources['wall_seconds']:
        raise RuntimeError('allocation wall limit exceeds scope')
    if allocation['memory_mib'] > resources['host_memory_gib']*1024:
        raise RuntimeError('allocation memory exceeds scope')
    output = Path(scope['output']).resolve()
    output_key = {'icc_decomposition':'decomposition_output','fixed_comparators':'comparator_output','icc_numeric':'numeric_output'}.get(scope['operation'])
    if output_key and Path(scope[output_key]).resolve() != output:
        raise RuntimeError('operation output path differs from reviewed output')
    out_parent = output.parent
    if shutil.disk_usage(out_parent).free < resources['min_free_disk_gib']*1024**3:
        raise RuntimeError('insufficient free output space')
    if not Path(sys.executable).samefile(scope['python_executable']) or digest(sys.executable) != scope['python_executable_sha256']:
        raise RuntimeError('Python identity mismatch')
    own_dir = Path(__file__).resolve().parent
    required_local = {'icc_stage3_cli.py','icc_stage3_assets.py','icc_local_replay.py','icc_stage3_grid.py'}
    per_operation = {'icc_grid': set(), 'icc_factor_build': {'icc_stage3_factor_build.py'},
                     'icc_author_wan_shape': {'icc_author_wan_shape_candidate.py'},
                     'icc_decomposition': {'icc_selected_decompose.py'},
                     'fixed_comparators': {'stage3_fixed_comparators.py'}, 'icc_numeric': {'icc_numeric_diagnostic.py'},
                     'q1_fit80': {'q1_cal11_reproduction.py','q1_cal11_fit80.py'},
                     'q1_v4_reconstruct': {'q1_cal11_reproduction.py','q1_v4_reconstruct_six.py'},
                     'q1_v4_remaining': {'q1_cal11_reproduction.py','q1_v4_reconstruct_six.py',
                                         'q1_v4_fixed_iterate_candidate.py','q1_v4_fit_remaining_candidate.py'}}
    operation = scope['operation']
    if operation not in per_operation: raise ValueError('unknown operation')
    missing = {str(own_dir/name) for name in required_local | per_operation[operation]} - set(scope['implementation_sha256'])
    if missing: raise RuntimeError('local implementation identity missing: '+str(sorted(missing)))
    for path, expected in scope['implementation_sha256'].items():
        if digest(path) != expected: raise RuntimeError('code identity mismatch')
    source_dir = Path(scope['bws_source_dir']).resolve()
    ridge = source_dir/'ridge_fit_core.py'
    if str(ridge) not in scope['implementation_sha256']:
        raise RuntimeError('native_ffn source is not bound')
    sys.path.insert(0,str(source_dir))
    import ridge_fit_core
    if Path(ridge_fit_core.__file__).resolve() != ridge:
        raise RuntimeError('native_ffn imported from unexpected source')
    import torch
    if torch.__version__ != scope['torch_version'] or torch.cuda.device_count()!=1 or torch.cuda.get_device_name() != 'NVIDIA A100-SXM4-80GB':
        raise RuntimeError('wrong torch or allocated GPU')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.cuda.set_per_process_memory_fraction(min(1.,resources['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory))
    started = time.monotonic()
    def check_resources():
        import resource
        if time.monotonic()-started > resources['soft_seconds']: raise TimeoutError('time budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > resources['host_memory_gib']*1024**3:
            raise MemoryError('host memory budget')
        if torch.cuda.max_memory_reserved()>resources['gpu_memory_gib']*1024**3: raise MemoryError('GPU memory budget')
        if output.exists() and sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > resources['output_bytes']:
            raise RuntimeError('output byte budget')
    operation = scope['operation']
    with torch.inference_mode():
        if operation == 'icc_grid':
            from icc_stage3_grid import execute
            result = execute(scope,scope_sha256=expected_sha,native_ffn=ridge_fit_core.native_ffn,device='cuda',
                             authorize=lambda _:None,check_resources=check_resources)
        elif operation == 'icc_factor_build':
            if str(source_dir/'icc_decompose.py') not in scope['implementation_sha256'] or str(source_dir/'icc_ffn.py') not in scope['implementation_sha256']:
                raise RuntimeError('factor producer files not bound')
            from icc_decompose import factorize
            from icc_stage3_factor_build import execute
            result = execute(scope,factorize=factorize,device='cuda',authorize=lambda _:None,check_resources=check_resources)
        elif operation == 'icc_author_wan_shape':
            author_path = str(Path(scope['author_source']['path']).resolve())
            if scope['implementation_sha256'].get(author_path) != scope['author_source']['sha256']:
                raise RuntimeError('pinned author source not bound by implementation scope')
            from icc_author_wan_shape_candidate import run
            result = run(scope,device='cuda',authorize=lambda _:None,check_resources=check_resources)
        elif operation == 'icc_decomposition':
            from icc_selected_decompose import run
            result = run(scope,scope['grid_dir'],device='cuda',authorize=lambda _:None,check_resources=check_resources)
        elif operation == 'icc_numeric':
            from icc_numeric_diagnostic import run
            result = run(scope,scope['grid_dir'],device='cuda',authorize=lambda _:None,check_resources=check_resources)
        elif operation == 'fixed_comparators':
            for name in ('b_common_replay.py','q1_fit_core.py','icc_ffn.py','icc_decompose.py',
                         'chipmunk_fullm_proxy_math.py','chipmunk_wan_packed_pregate.py'):
                if str(source_dir/name) not in scope['implementation_sha256']:
                    raise RuntimeError('comparator dependency not bound: '+name)
            import b_common_replay
            from stage3_fixed_comparators import run
            result = run(scope,common=b_common_replay,native_ffn=ridge_fit_core.native_ffn,device='cuda',
                         authorize=lambda _:None,check_resources=check_resources)
        elif operation == 'q1_fit80':
            for name in ('b_common_replay.py','q1_fit_core.py','icc_ffn.py','icc_decompose.py',
                         'chipmunk_fullm_proxy_math.py','chipmunk_wan_packed_pregate.py'):
                if str(source_dir/name) not in scope['implementation_sha256']:
                    raise RuntimeError('Q1 producer dependency not bound: '+name)
            import b_common_replay
            from q1_cal11_fit80 import run
            result = run(scope,common_replay=b_common_replay,device='cuda',authorize=lambda _:None,
                         check_resources=check_resources,deadline=started+resources['soft_seconds'])
        elif operation in ('q1_v4_reconstruct', 'q1_v4_remaining'):
            dependencies = ('b_common_replay.py','q1_fit_core.py','icc_ffn.py','icc_decompose.py',
                            'chipmunk_fullm_proxy_math.py','chipmunk_wan_packed_pregate.py')
            for name in dependencies:
                if str(source_dir/name) not in scope['implementation_sha256']:
                    raise RuntimeError('Q1 V4 dependency not bound: '+name)
            import b_common_replay
            if operation == 'q1_v4_reconstruct':
                from q1_fit_core import Problem
                from q1_v4_reconstruct_six import run
                result = run(scope,common_replay=b_common_replay,Problem=Problem,device='cuda',
                             authorize=lambda _:None,check_resources=check_resources)
            else:
                from q1_v4_fit_remaining_candidate import run
                result = run(scope,common_replay=b_common_replay,device='cuda',authorize=lambda _:None,
                             check_resources=check_resources,deadline=started+resources['soft_seconds'])
        else:
            raise ValueError('unsupported reviewed operation')
    check_resources()
    import resource
    telemetry = {'operation_wall_seconds': time.monotonic() - started,
                 'process_lifetime_peak_host_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                 'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved()}
    print(json.dumps({'result':result,'allocation':allocation,'resources':telemetry},allow_nan=False),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('scope'); p.add_argument('scope_sha256'); a=p.parse_args()
    main(a.scope,a.scope_sha256)







