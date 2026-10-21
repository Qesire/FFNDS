"""Bind UB2's one changed-step fit to active and original-UB1 identity policies."""
import argparse
import json
from pathlib import Path

from ar2_checkpoint import atomic_json,digest
from evaluate_residual_rr1 import open_verified
from run_ar2_restart import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--identity-control',action='store_true')
    args = ap.parse_args()
    fit,report,identity = open_verified(args.fit,'COMPLETED_UB2_STEP1_QUARTER40_FIT')
    r = ROOT/'runs/worker_restart_20261020'
    parent = r/'ub1_step_budget_fit/UB1_policy.json'
    p = json.loads(parent.read_text())
    protocol = ROOT/'config/terminal_dense_ub2.json'
    if (len(report['positions']) != 40 or identity['frozen_UB1_policy_sha256'] != digest(parent)
            or identity['parent_policy_sha256'] != p['parent_policy']['sha256']
            or identity['protocol_sha256'] != digest(protocol)):
        raise RuntimeError('Exact frozen UB2 fit, C24 recipe and UB1 parent required')
    viewpath = ROOT/'.runtime/node_local_model_view.json'
    view = json.loads(viewpath.read_text())
    low = next(v for v in view['files'] if '/low_noise_model/' in v['view_path'] and v['filename'].endswith('.safetensors'))
    sources = [ROOT/'tools'/name for name in ['terminal_dense_runtime.py','step_budget_runtime.py',
        'trained_packed_runtime.py','reduced_rank_residual.py','resident_ffn_runtime.py']]
    sources += [ROOT/'.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py',
                ROOT/'.runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py']
    value = {'kind':'TERMINAL_DENSE_STEP_REFERENCE','experiment':'UB2_TERMINAL_DENSE_FIXED_HALF_MAIN_BUDGET_20261021',
        'identity_control':args.identity_control,'groups_by_step':[54,54,108,216],'residual_rank':256,'terminal_step':3,
        'parent_policy':{'path':str(parent),'sha256':digest(parent)},
        'step1_fit':{'path':str(args.fit.resolve()),'sha256':digest(args.fit),'fingerprint':fit.fingerprint},
        'protocol':{'path':str(protocol),'sha256':digest(protocol)},'cache_manifest':p['cache_manifest'],
        'low_raw_checkpoint':{'path':low['view_path'],'sha256':low['sha256'],'bytes':low['bytes'],
                              'node_view_manifest_sha256':digest(viewpath)},
        'runtime_sources':{str(s):digest(s) for s in sources},'nominal_global_main_MAC_density':.5,
        'expected_residual_positions':160 if args.identity_control else 120,
        'nominal_auxiliary_MAC_density':(256 if args.identity_control else 192)/13824,
        'quality_or_speed_acceptance':False}
    if args.output.exists() and json.loads(args.output.read_text()) != value:
        raise RuntimeError('Policy is immutable; preserve the old file and select a new output')
    atomic_json(args.output,value)
    print(digest(args.output))


if __name__ == '__main__':
    main()
