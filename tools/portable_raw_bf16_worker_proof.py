"""New bounded full-SHA proof on an additional asset-ready A100 worker."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('_ffnds_portable_master_sha_20261029', ROOT/'tools/raw_bf16_worker_mirror_proof.py')
frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
native = frozen.native
ID = 'E0_ORIGINAL_BF16_ADDITIONAL_A100_MASTER_CPU_20261029'
OUTPUT = 'ffnds-derived/e0_raw_bf16_additional_A100_master_cpu_20261029/attempt1/worker/results'
TOTAL, RESOURCES, LIMITS = frozen.TOTAL, frozen.RESOURCES, frozen.LIMITS
frozen.ID = ID  # Private namespace; original executed proof is unchanged.


def draft():
    plan = frozen.draft()
    plan['output_HF_relative'] = OUTPUT
    plan['repo_sources'] += [native.binding(ROOT/p, ROOT) for p in
        ('tools/portable_raw_bf16_worker_proof.py', 'tests/test_portable_raw_bf16_worker_proof.py')]
    return plan


def validate(plan):
    expected = draft()
    expected['private_worker_config_sha256'] = plan.get('private_worker_config_sha256')
    expected['status'] = plan.get('status')
    native.need(plan == expected and plan['id'] == ID and plan['status'] in ('DRAFT_NOT_REGISTERED', 'FROZEN_REGISTERED'),
                'Exact portable raw-master proof differs')
    value = plan['private_worker_config_sha256']
    native.need(value is None or isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef'), 'Private profile SHA malformed')
    if plan['status'] == 'FROZEN_REGISTERED': native.need(value is not None, 'Execution profile not bound')


def run(args):
    plan = native.read(args.protocol); validate(plan); sha = native.sha(args.protocol)
    frozen.authorize(plan, sha)
    native.need(plan['status'] == 'FROZEN_REGISTERED' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and not {'torch', 'numpy'} & set(sys.modules) and len(os.sched_getaffinity(0)) <= 4, 'Fresh hidden-CUDA four-CPU proof required')
    frozen.require_worker()
    native.need(native.sha(frozen.private_config_path()) == plan['private_worker_config_sha256'], 'Different private execution profile')
    for row in plan['repo_sources']: native.bound(ROOT, row)
    out = (args.hf_root/OUTPUT).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined full-SHA proof only')
    out.mkdir(parents=True)
    return frozen.run_proof(plan, out, sha)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('draft', 'run'))
    p.add_argument('--protocol', type=Path, required=True); p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    args = p.parse_args()
    if args.action == 'draft':
        native.need(not args.protocol.exists(), 'Fresh protocol required'); native.write(args.protocol, draft())
    else: run(args)


if __name__ == '__main__': main()
