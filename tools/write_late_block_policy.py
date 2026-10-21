"""Bind the completed LR2 fit to immutable active and identity runtime policies."""
import argparse
import json
from pathlib import Path

from ar2_checkpoint import atomic_json, digest
from evaluate_residual_rr1 import open_verified
from run_ar2_restart import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--identity-control',action='store_true')
    args = ap.parse_args()
    fit,report,identity = open_verified(args.fit,'COMPLETED_LR2_TEN_LAST_STEP_RESIDUAL_FIT')
    r = ROOT/'runs/worker_restart_20261020'
    parent = r/'ub1_step_budget_fit/UB1_policy.json'
    cv = r/'late_block_rank_lr2/RUN.json'
    protocol = ROOT/'config/late_block_rank_lr2.json'
    if (identity['UB1_policy_sha256'] != digest(parent) or identity['CV_report_sha256'] != digest(cv)
            or identity['protocol_sha256'] != digest(protocol) or len(report['positions']) != 10):
        raise RuntimeError('Exact registered LR2 fit/parent/CV required')
    sources = [ROOT/'tools'/name for name in ['late_block_rank_runtime.py','step_budget_runtime.py',
        'trained_packed_runtime.py','reduced_rank_residual.py','resident_ffn_runtime.py']]
    value = {'kind':'LATE_BLOCK_RANK_RESIDUAL_REFERENCE','experiment':'LR2_LAST_TEN_BLOCKS_FINAL_STEP_CAPACITY_20261021',
        'identity_control':args.identity_control,'main_groups_by_step':[54,108,108,162],
        'changed_positions':[[3,b] for b in range(30,40)],'default_rank':256,'changed_rank':512,
        'parent_policy':{'path':str(parent),'sha256':digest(parent)},
        'fit_report':{'path':str(args.fit.resolve()),'sha256':digest(args.fit),'fingerprint':fit.fingerprint},
        'CV_report':{'path':str(cv),'sha256':digest(cv)},'protocol':{'path':str(protocol),'sha256':digest(protocol)},
        'cache_manifest':json.loads(parent.read_text())['cache_manifest'],
        'runtime_sources':{str(p):digest(p) for p in sources},
        'global_main_MAC_density':.5,'auxiliary_MAC_density':(256 if args.identity_control else 272)/13824,
        'quality_or_speed_acceptance':False}
    if args.output.exists() and json.loads(args.output.read_text()) != value:
        raise RuntimeError('Frozen LR2 policy changed; select a fresh output')
    atomic_json(args.output,value)
    print(digest(args.output))


if __name__ == '__main__':
    main()
