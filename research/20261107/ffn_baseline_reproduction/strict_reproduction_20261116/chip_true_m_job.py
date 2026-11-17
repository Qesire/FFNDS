"""Independently guarded true-M local replay; never submits a job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

BASE = Path('/public/home/scc/pb23061276/projects/wan-quant/work/ffnds_strict_reproduction_20261116_336b7e1')
GPU = 'NVIDIA A100-SXM4-80GB'


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def bound(spec):
    path = Path(spec['path']).resolve()
    if not path.is_file() or digest(path) != spec['sha256']: raise RuntimeError(f'asset identity mismatch: {path}')
    return path


def prior(run_spec, exit_spec, expected_job):
    run, end = read(bound(run_spec)), read(bound(exit_spec))
    if (run['slurm_job_id'] != expected_job or end['slurm_job_id'] != expected_job or end['exit_code'] != 0
            or run.get('wan_nfe') != 0 or run.get('videos') != 0):
        raise RuntimeError('prerequisite execution failed or job identity changed')
    return run


def main(scope_path, expected_sha):
    start = time.monotonic()
    stage = Path.cwd().resolve()
    scope_path = Path(scope_path).resolve()
    if not stage.is_relative_to(BASE) or scope_path != stage/'scope.json' or digest(scope_path) != expected_sha:
        raise RuntimeError('invalid stage or changed scope')
    scope = read(scope_path)
    if scope['operation'] != 'chip_true_m' or scope['selector_seed'] != 42:
        raise RuntimeError('wrong operation or frozen selector seed')
    if (scope['max_predictions'], scope['max_arithmetic_pairs'], scope['num_videos'], scope['num_model_nfe']) != (24, 4, 0, 0):
        raise RuntimeError('unregistered work')
    review_path = stage/'REVIEW.json'
    review = read(review_path)
    active = read(stage/'ACTIVE_RESEARCH.json')[scope['scope_id'].lower()]
    if (not active['runnable'] or active['scope_sha256'] != expected_sha or active['review_sha256'] != digest(review_path)
            or review['scope_sha256'] != expected_sha or review['recheck_disposition'] != 'APPROVE'):
        raise RuntimeError('independent approval and ACTIVE required')
    if Path(active['scope_path']).resolve() != stage:
        raise RuntimeError('ACTIVE refers to different stage')
    if not Path(sys.executable).samefile(scope['python_executable']) or digest(sys.executable) != scope['python_executable_sha256']:
        raise RuntimeError('interpreter mismatch')
    pinned_gp64 = {'gp64_proxy_source':'f46ba6feee6905b3e875a3d4e3118ae97259a978ce6ba55413ed6d2a0a5544be',
                   'gp64_action_source':'9a3e789e15564a0b2979418f9b69e26737abc057a0afcb8132f934dadde2160b'}
    for field, identity in pinned_gp64.items():
        if scope[field]['sha256'] != identity or digest(scope[field]['path']) != identity:
            raise RuntimeError('GP64 differs from pinned reference: '+field)
    names = {'chip_true_m_job.py','chip_true_m_job.sh','chip_true_m_replay.py','chip_provider.py','chip_wan_voxel_tail_v1.py',
             'chip_tail52_valid_candidate.py','icc_stage3_assets.py','icc_local_replay.py'}
    required = {str((stage/name).resolve()) for name in names}
    if not required <= set(scope['implementation_sha256']): raise RuntimeError('missing executable source identity')
    for path, identity in scope['implementation_sha256'].items():
        if digest(path) != identity: raise RuntimeError('changed implementation')
    prereq = scope['prerequisites']
    module_run = prior(prereq['module_run'], prereq['module_exit'], '168787')
    module = read(bound(prereq['module_result']))
    selector_run = prior(prereq['selector_run'], prereq['selector_exit'], '168713')
    selector = read(bound(prereq['selector_result']))
    if (module['status'] != 'PASS' or not module['new_request_seeded'] or module['source_sha256'] != scope['provider']['source_sha256']
        or module['witnesses']['mm1_actual'] != module['witnesses']['mm1_expected']
        or module['witnesses']['mm2_actual'] != module['witnesses']['mm2_expected']
        or module['witnesses']['mm1_actual'] == module['witnesses']['mm1_round_first']
        or module['witnesses']['mm2_actual'] == module['witnesses']['mm2_add_first']
        or module_run['scope'] != 'FFNDS_STRICT_CHIP_MODULE_BOUNDED_V1'):
        raise RuntimeError('bounded author module evidence mismatch')
    if (selector_run['scope'] != 'FFNDS_STRICT_CHIP_SAFE_SELECTOR_A100_V1'
        or selector['variant'] != 'CHIPMUNK_AUTHOR_MLP_SAFE_SELECTOR_V1'
        or selector['loaded_extension_sha256'] != scope['provider']['library_sha256']
        or selector['adversarial_wan_width'] != {'count':13824,'unique':13824,'unwritten_slots':0,'copy_exact':True}):
        raise RuntimeError('safe selector evidence mismatch')
    grid_run = read(bound(prereq['grid_run'])); grid_end = read(bound(prereq['grid_exit']))
    grid_scope = bound(scope['canonical_grid_scope'])
    if (grid_run['scope_sha256'] != digest(grid_scope) or grid_end['exit_code'] != 0
            or grid_run['slurm_job_id'] != grid_end['slurm_job_id'] or grid_run['wan_nfe'] != 0 or grid_run['videos'] != 0):
        raise RuntimeError('canonical grid did not complete')
    cache = read(bound(scope['canonical_cache_manifest']))
    scores = read(bound(prereq['grid_scores']))
    if (cache['scope_sha256'] != digest(grid_scope) or scores['scope_sha256'] != digest(grid_scope)
            or scores['cell_count'] != 792 or len(scores['rows']) != 792
            or cache['status'] != 'PRESERVED_PENDING_SELECTED_FAMILY_ERROR_DECOMPOSITION'):
        raise RuntimeError('incomplete canonical grid')
    resources = scope['resources']
    if (not 0 < resources['gpu_memory_gib'] <= 78 or not 0 < resources['host_memory_gib'] <= 160 or not 0 < resources['wall_seconds'] <= 10800
            or not 0 < resources['output_bytes'] <= 2*1024**3 or resources['min_free_disk_gib'] <= 0 or not 0 < resources['soft_seconds'] < resources['wall_seconds']):
        raise RuntimeError('resource ceiling exceeds fixed scope')
    output = Path(scope['output']).resolve()
    if output != stage/'output' or output.exists(): raise RuntimeError('output must be new stage/output')
    guard = bound(scope['runtime_guard'])
    sys.path.insert(0, str(guard.parent))
    from runtime_guard import require_compute_allocation
    allocation = require_compute_allocation()
    job = dict(re.findall(r'(\w+)=([^\s]+)', allocation['scheduler_job_record']))
    wall = re.fullmatch(r'(\d+):(\d+):(\d+)', job.get('TimeLimit',''))
    if (not os.environ.get('SLURM_JOB_ID') or wall is None or job.get('Partition') != 'P107-A100' or job.get('QOS') != 'qos_p107-a100' or job.get('Account') != 'competition'
            or sum(int(x)*w for x,w in zip(wall.groups(),(3600,60,1))) > resources['wall_seconds']
            or allocation['memory_mib'] > resources['host_memory_gib']*1024):
        raise RuntimeError('unapproved allocation')
    if shutil.disk_usage(stage).free < resources['min_free_disk_gib']*1024**3: raise RuntimeError('insufficient disk')
    def terminate(signum, frame): raise TimeoutError(f'termination signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    import torch
    import triton
    if (torch.__version__ != scope['torch_version'] or triton.__version__ != scope['triton_version']
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != GPU):
        raise RuntimeError('unexpected torch, Triton or GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.cuda.set_per_process_memory_fraction(min(1., resources['gpu_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory))
    def check_resources():
        import resource
        if time.monotonic()-start > resources['soft_seconds']: raise TimeoutError('soft time budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > resources['host_memory_gib']*1024**3: raise MemoryError('host memory limit')
        if torch.cuda.max_memory_reserved() > resources['gpu_memory_gib']*1024**3: raise MemoryError('GPU memory limit')
        if shutil.disk_usage(stage).free < resources['min_free_disk_gib']*1024**3: raise RuntimeError('disk floor')
        if output.exists() and sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > resources['output_bytes']:
            raise RuntimeError('output bytes exceeded')
    from chip_true_m_replay import execute
    with torch.inference_mode():
        execute(scope, output, authorize=lambda candidate: None if candidate is scope else (_ for _ in ()).throw(RuntimeError('scope mismatch')),
                check_resources=check_resources)
    check_resources()
    with (stage/'receipt'/'RESULT.json').open('x', encoding='utf-8') as f:
        json.dump({'status':'COMPLETE', 'scope_sha256':expected_sha, 'summary_sha256':digest(output/'SUMMARY.json'),
                   'torch_seed':42, 'curand_seed_rule':'per_row_thread_linear_id + signed_int32(first_two_row_bf16)' , 'allocation':allocation,
                   'gpu_peak_reserved':torch.cuda.max_memory_reserved()}, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('scope'); parser.add_argument('scope_sha256')
    args = parser.parse_args(); main(args.scope, args.scope_sha256)
