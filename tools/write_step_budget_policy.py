"""Freeze UB1 or its108-group identity-control policy."""
import argparse
import json
from pathlib import Path

from ar2_checkpoint import atomic_json, digest
from evaluate_residual_rr1 import open_verified
from run_ar2_restart import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent-policy', type=Path, required=True)
    ap.add_argument('--fits', type=Path, nargs=2)
    ap.add_argument('--identity-control', action='store_true')
    ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    parent = json.loads(args.parent_policy.read_text())
    if parent['experiment'] != 'RR1_CALIBRATION24_FIXED_RANK256_LAMBDA1_20261021':
        raise RuntimeError('Only the frozen C24 parent is supported')
    protocol = ROOT / 'config/step_budget_ub1.json'; changed = {}
    if not args.identity_control:
        if args.fits is None: ap.error('--fits required for actual UB1')
        for path in args.fits:
            store, report, identity = open_verified(path, 'COMPLETED_UB1_STEP40_FIT'); step = report['step']
            if (step not in [0, 3] or str(step) in changed or len(report['positions']) != 40
                    or identity['parent_policy_sha256'] != digest(args.parent_policy)
                    or identity['protocol_sha256'] != digest(protocol)
                    or report['control']['status'] != 'PASS_OLD_C24_ALPHA_BIAS_RR_PARAMETER_BITS'):
                raise RuntimeError('Both complete fixed changed-step fits required')
            changed[str(step)] = {'path': str(path.resolve()), 'sha256': digest(path), 'fingerprint': store.fingerprint}
        if set(changed) != {'0', '3'}: raise RuntimeError('Both changed steps required')
    policy = {'kind': 'STEP_BUDGET_PACKED_REFERENCE', 'experiment': 'UB1_IDENTITY_CONTROL' if args.identity_control else 'UB1_25_50_50_75_C24_FIXED_RECIPE',
        'identity_control': args.identity_control, 'groups_by_step': [108] * 4 if args.identity_control else [54, 108, 108, 162],
        'parent_policy': {'path': str(args.parent_policy.resolve()), 'sha256': digest(args.parent_policy)},
        'protocol': {'path': str(protocol), 'sha256': digest(protocol)}, 'changed_fits': changed,
        'cache_manifest': parent['cache_manifest'], 'auxiliary_compute': parent['auxiliary_compute'],
        'global_main_MAC_density': .5, 'quality_or_speed_acceptance': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text()) != policy: raise RuntimeError('Refusing policy overwrite')
    atomic_json(args.output, policy); print(json.dumps({'path': str(args.output.resolve()), 'sha256': digest(args.output)}))


if __name__ == '__main__': main()
