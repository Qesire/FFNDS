"""Versioned launch fix for a superseded legacy GPU-authority field.

V1 CPU guards exited before RUN/assets/models. V2 preserves that failure and
binds the actual dynamic policy plus this exact two-slot execution allocation.
All native generation, source selection and numerical paths remain unchanged.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
spec = importlib.util.spec_from_file_location('_ffnds_new12_launch_v2', ROOT/'tools/e0_s2_new12_dense_baseline.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
native = core.native
EXTRA = ('tools/e0_s2_new12_dense_baseline_v2.py', 'tests/test_e0_s2_new12_dense_baseline_v2.py')


def scope_id(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two parts')
    return f'E0_S2_NEW12_DENSE_BASELINE_20261029_PART{part}_V2'


def output(part):
    scope_id(part)
    return f'ffnds-derived/e0_s2_new12_dense_baseline_v2_20261029/part{part}/attempt1/worker/results'


core.scope_id, core.output_relative = scope_id, output


def resource_authority(active):
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,
                'Continuation inactive')
    current = active['current_session_resource_constraint_20261028']
    native.need(current['mode'] == 'DYNAMIC_MAX_ACTUALLY_CALLABLE'
        and current['max_simultaneously_assigned_idle_GPUs'] is None,
        'Current user dynamic resource policy differs')
    # This pair elects to use the two genuinely asset-ready slots. This is an
    # exact finite scope bound, not a new fixed global resource restriction.
    assigned = sum(s.get('assigned_GPU_slots', 0) for s in
        active['experiment_authority']['current_user_authorized_scopes'])
    native.need(0 <= assigned <= 2, 'This frozen execution pair has at most two assigned slots')


def verify_failed_v1(plan, hf):
    proof = plan['CPU_authority_launch_repair']
    prior = native.read(native.bound(ROOT, proof['protocol']))
    receipt = native.read(native.bound(hf, proof['exit']))
    error = native.bound(hf, proof['stderr']).read_text()
    expected = f'E0_S2_NEW12_DENSE_BASELINE_20261029_PART{plan["part"]}_V1'
    native.need(prior['id'] == receipt['id'] == expected
        and proof['protocol']['sha256'] == receipt['protocol_sha256']
        and receipt['process_exited'] is True and receipt['child_returncode'] == 1
        and receipt['run_sha256'] is None and receipt['timeout'] is False
        and receipt['automatic_retry'] is False
        and 'ValueError: Required current two-GPU authority differs' in error
        and proof['actual_GPU_calls'] == proof['actual_model_calls'] == 0
        and proof['native_algorithm_changed'] is False
        and not (hf/prior['output_HF_relative']).exists(), 'Original zero-model authority failure differs')


def validate(plan, frozen=False):
    core.validate(plan, frozen)
    native.need(set(EXTRA) <= {b['path'] for b in plan['repo_sources']}
        and plan['CPU_authority_launch_repair']['native_algorithm_changed'] is False,
        'V2 launch repair source closure differs')


def authorize(plan, digest):
    validate(plan, True); active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    resource_authority(active)
    rows = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    native.need(len(rows) == 1 and rows[0].get('protocol_sha256') == digest
        and rows[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and rows[0].get('limits') == plan['limits'] and rows[0].get('resources') == core.RESOURCES
        and rows[0].get('run_ids') == plan['run_ids'] and rows[0].get('max_gpu_cards') == 1,
        'Exact V2 acquisition not registered')
    return rows[0]


core.authorize = authorize


def draft(hf, part):
    plan = core.draft(hf, part)
    prior = ROOT/f'config/e0_s2_new12_dense_baseline_part{part}_20261029_v1.json'
    worker = hf/f'ffnds-derived/e0_s2_new12_dense_baseline_cpu_20261029/part{part}/attempt1/worker'
    plan['CPU_authority_launch_repair'] = {'protocol': native.binding(prior, ROOT),
        'exit': native.binding(worker/'EXIT_RECEIPT.json', hf),
        'stderr': native.binding(worker/'child.stderr.log', hf),
        'actual_model_calls': 0, 'actual_GPU_calls': 0, 'native_algorithm_changed': False,
        'reason': 'OLD_SUPERSEDED_GLOBAL_GPU_FIELD_MISREAD_AS_THIS_NEW_FINITE_SCOPE_ALLOCATION'}
    paths = {b['path'] for b in plan['repo_sources']} | set(EXTRA) | {str(prior.relative_to(ROOT))}
    plan['repo_sources'] = [native.binding(ROOT/p, ROOT) for p in sorted(paths)]
    validate(plan); verify_failed_v1(plan, hf)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('draft', 'inspect', 'preflight', 'run'))
    p.add_argument('--part', type=int, choices=(0, 1)); p.add_argument('--protocol', type=Path)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    p.add_argument('--local-root', type=Path); p.add_argument('--out', type=Path); args = p.parse_args()
    if args.action == 'draft':
        native.need(args.out is not None and not args.out.exists(), 'Fresh draft only')
        args.out.parent.mkdir(parents=True, exist_ok=True); native.write(args.out, draft(args.hf_root, args.part))
    else:
        plan = native.read(args.protocol); validate(plan, True); verify_failed_v1(plan, args.hf_root)
        if args.action == 'inspect': print('V2_STATIC_ONLY_NO_MODEL')
        elif args.action == 'preflight': core.preflight(args)
        else: core.run(args)


if __name__ == '__main__': main()
