"""Versioned CPU launch repair; native four-corner arithmetic remains frozen.

V1 CPU preflight exited1 before writing RUN because its nested output directory
did not exist. No GPU/native calls occurred. Create that directory explicitly
and bind original failure receipts in a fresh registered identity.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('_ffnds_s2_spatial_prefix_launch_v2', ROOT/'tools/e0_s2_spatial_prefix_replay.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
native = core.native
core.BASE = 'E0_S2_SPATIAL52_SAME_H_LOCAL_PREFIX_REPLAY_LAUNCH_V2_20261029'
core.REQUIRED |= {'tools/e0_s2_spatial_prefix_replay_v2.py', 'tests/test_e0_s2_spatial_prefix_replay_v2.py'}


def output(part):
    core.identity(part)
    return f'ffnds-derived/e0_s2_spatial52_same_h_local_prefix_replay_v2_20261029/part{part}/attempt1/worker/results'


core.output = output


def prepare_parent(args):
    native.need(args.out is not None and args.out.resolve().is_relative_to(args.hf_root.resolve())
        and not args.out.exists(), 'Fresh confined CPU result required')
    args.out.parent.mkdir(parents=True, exist_ok=True)


def verify_repair(plan, hf):
    proof = plan['CPU_launch_repair']
    old = native.read(native.bound(ROOT, proof['prior_protocol']))
    receipt = native.read(native.bound(hf, proof['actual_failed_exit']))
    error = native.bound(hf, proof['actual_stderr']).read_text()
    native.need(receipt['id'] == old['id'] and receipt['protocol_sha256'] == proof['prior_protocol']['sha256']
        and receipt['process_exited'] is True and receipt['child_returncode'] == 1 and receipt['run_sha256'] is None
        and 'FileNotFoundError' in error and '/CPU/RUN.json.tmp' in error
        and proof['prior_model_calls'] == proof['prior_GPU_calls'] == 0 and proof['native_algorithm_changed'] is False,
        'Original missing-parent CPU failure audit changed')


def draft(hf, part):
    plan = core.draft(hf, part)
    old_path = ROOT/f'config/e0_s2_spatial_prefix_replay_part{part}_20261029_v1.json'
    old = native.read(old_path)
    worker = hf/f'ffnds-derived/e0_s2_spatial_prefix_replay_cpu_20261029/part{part}/attempt1/worker'
    receipt = native.read(worker/'EXIT_RECEIPT.json')
    native.need(receipt['id'] == old['id'] and receipt['protocol_sha256'] == native.sha(old_path)
        and receipt['process_exited'] is True and receipt['child_returncode'] == 1
        and receipt['run_sha256'] is None and not (hf/old['output_HF_relative']).exists(), 'V1 no-native CPU failure identity differs')
    error = (worker/'child.stderr.log').read_text()
    native.need('FileNotFoundError' in error and '/CPU/RUN.json.tmp' in error, 'Exact missing-parent launch failure required')
    plan['CPU_launch_repair'] = {'prior_protocol': native.binding(old_path, ROOT),
        'actual_failed_exit': native.binding(worker/'EXIT_RECEIPT.json', hf),
        'actual_stderr': native.binding(worker/'child.stderr.log', hf),
        'prior_model_calls': 0, 'prior_GPU_calls': 0, 'native_algorithm_changed': False,
        'repair': 'CREATE_CONFINED_FRESH_CPU_OUTPUT_PARENT_BEFORE_UNCHANGED_PREFLIGHT'}
    plan['repo_sources'].append(native.binding(old_path, ROOT))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('draft', 'preflight', 'run'))
    p.add_argument('--part', type=int, choices=(0, 1)); p.add_argument('--protocol', type=Path)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface')); p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.action == 'draft':
        native.need(not args.out.exists(), 'Fresh V2 draft only'); native.write(args.out, draft(args.hf_root, args.part))
    elif args.action == 'preflight':
        verify_repair(native.read(args.protocol), args.hf_root); prepare_parent(args); core.preflight(args)
    else:
        verify_repair(native.read(args.protocol), args.hf_root); core.run(args)


if __name__ == '__main__': main()
