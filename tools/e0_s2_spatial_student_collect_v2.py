"""Same spatial/student scope on a newly byte-verified, idle A100 executor.

V1 stayed in its idle queue and was cancelled before any model/GPU work.
All model arithmetic, observation geometry, budgets and byte controls are the
unchanged executed-source implementation; only executor proof and run IDs move.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('_ffnds_spatial_capture_portable_v2', ROOT/'tools/e0_s2_spatial_student_collect.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
import portable_raw_bf16_worker_proof as proof

# The V1 module imports its video dependency by name. Give this executor an
# independent dependency tree too, so its proof identity cannot affect V1.
video_spec = importlib.util.spec_from_file_location('_ffnds_spatial_capture_v2_video_dependency', ROOT/'tools/e0_raw_bf16_temporal_video20.py')
private_video = importlib.util.module_from_spec(video_spec); video_spec.loader.exec_module(private_video)
core.video, core.vm = private_video, private_video.vm
core.native, core.io = private_video.native, private_video.io
core.legacy = private_video.io.legacy

MASTER = 'config/e0_raw_bf16_additional_a100_master_cpu_20261029_v1.json'
core.BASE_ID = 'E0_S2_SPATIAL_TEACHER_STUDENT_FULL160_CAPTURE_PORTABLE_20261029_V2'
core.REQUIRED |= {'tools/e0_s2_spatial_student_collect_v2.py', 'tests/test_e0_s2_spatial_student_collect_v2.py',
                  'tools/portable_raw_bf16_worker_proof.py', 'tests/test_portable_raw_bf16_worker_proof.py', MASTER}


def output_relative(part):
    core.scope_id(part)
    return f'ffnds-derived/e0_s2_spatial_student_full160_portable_v2_20261029/part{part}/attempt1/worker/results'


core.output_relative = output_relative
# Replace only the independently loaded proof namespace used by the existing
# strict gate. Its identity/count/stat/header checks remain unchanged.
core.io.mirror = SimpleNamespace(ID=proof.ID, TOTAL=proof.TOTAL, validate=proof.validate)


def draft(hf, part):
    plan = core.draft(hf, part)
    master = core.native.read(ROOT/MASTER); proof.validate(master)
    folder = hf/master['output_HF_relative']
    master_proof = {'protocol': core.native.binding(ROOT/MASTER, ROOT),
                    'RUN': core.native.binding(folder/'RUN.json', hf),
                    'launch_exit': core.native.binding(folder.parent/'EXIT_RECEIPT.json', hf)}
    run = core.native.read(core.native.bound(hf, master_proof['RUN']))
    core.vm.exited(core.native.read(core.native.bound(hf, master_proof['launch_exit'])), master_proof['RUN'],
                   master_proof['protocol']['sha256'], proof.ID)
    core.native.need(run['status'] == 'PASS_FRESH_WORKER_RAW_SHA_STAT_HEADER_NO_MODEL_NO_NUMPY_NO_TORCH'
        and run['actual_full_payload_read_bytes'] == proof.TOTAL and plan['models'] == master['models'],
        'Fresh additional worker original-master proof required')
    old_path = ROOT/f'config/e0_s2_spatial_student_collect_part{part}_20261029_v1.json'
    old = core.native.read(old_path); old_out = hf/old['output_HF_relative']; worker = old_out.parent
    cancelled = core.native.read(worker/'ROOT_QUEUE_CANCELLATION.json'); exited = core.native.read(worker/'EXIT_RECEIPT.json')
    core.native.need(cancelled['id'] == old['id'] == exited['id'] and cancelled['model_process_started'] is False
        and exited['process_exited'] is True and type(exited['child_returncode']) is int
        and exited['child_returncode'] == -15 and exited['run_sha256'] is None and not old_out.exists(),
        'Original V1 must remain cancelled before model start')
    plan.update(execution_private_worker_config_sha256=master['private_worker_config_sha256'],
                execution_raw_master_CPU_proof=master_proof,
                migration={'cancelled_V1_protocol': core.native.binding(old_path, ROOT),
                           'queue_cancellation': core.native.binding(worker/'ROOT_QUEUE_CANCELLATION.json', hf),
                           'actual_queue_exit': core.native.binding(worker/'EXIT_RECEIPT.json', hf),
                           'old_model_calls': 0, 'old_GPU_calls': 0, 'numeric_algorithm_changed': False})
    if master['private_worker_config_sha256'] not in plan['private_worker_config_sha256']:
        plan['private_worker_config_sha256'].append(master['private_worker_config_sha256'])
    plan['repo_sources'].append(core.native.binding(old_path, ROOT))
    core.validate(plan)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('draft', 'preflight', 'run'))
    p.add_argument('--part', type=int, choices=(0, 1)); p.add_argument('--protocol', type=Path)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    p.add_argument('--local-root', type=Path); p.add_argument('--out', type=Path); args = p.parse_args()
    if args.action == 'draft':
        core.native.need(args.out is not None and not args.out.exists(), 'Fresh V2 draft required')
        core.native.write(args.out, draft(args.hf_root, args.part))
    elif args.action == 'preflight': core.preflight(args)
    else: core.run(args)


if __name__ == '__main__': main()
