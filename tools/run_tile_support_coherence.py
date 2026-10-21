"""TC1 CPU-only survey of support transfer between native64-row tiles."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from ar2_checkpoint import atomic_json,digest
from evaluate_residual_rr1 import open_verified
from onpolicy_residual import load_collections
from router_bank_feasibility import normalized_profiles
from run_ar2_restart import ROOT
from tile_support_coherence import transfer_statistics


def bootstrap_interval(values,draws):
    x=np.asarray(values,dtype=np.float64)
    return np.quantile(x[draws].mean(1),[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise RuntimeError('TC1 is CPU only; explicitly hide CUDA')
    r=ROOT/'runs/worker_restart_20261020'
    preflight=ROOT/'reports/worker_20261020/TC1_preflight_79_tests.json'
    tests=json.loads(preflight.read_text())
    if tests['status']!='PASS_COMPLETE_CPU_UNIT_TEST_PREFLIGHT' or tests['tests_run']!=79:
        raise RuntimeError('Exact pair-enumeration/unit preflight must pass first')
    records,bindings,capture=load_collections([r/f'onpolicy_calibration24/shard{s}/RUN.json' for s in [0,1]],ROOT)
    profilepath=r/'router_bank_rb1/RUN.json'
    store,_,identity=open_verified(profilepath,'COMPLETED_RB1_PROXY_GATE_REJECTED')
    if identity['collection_bindings']!=bindings or identity['student_policy_sha256']!=capture['student_policy_sha256']:
        raise RuntimeError('Frozen teacher profiles and native coordinate source differ')
    for path,sha in identity['sources'].items():
        if digest(path)!=sha:raise RuntimeError('Frozen profile producer changed')
    hp1path=r/'hidden_profile_hp1/RUN.json';hp1=json.loads(hp1path.read_text())
    protocol=ROOT/'config/tile_support_tc1.json'
    sources=[Path(__file__),ROOT/'tools/tile_support_coherence.py',ROOT/'tools/onpolicy_residual.py',ROOT/'tools/router_bank_feasibility.py']
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=args.out/'RUN.json'
    if path.exists():raise RuntimeError('TC1 descriptions are immutable; preserve prior output')
    meta={'status':'RUNNING','identity':{'protocol_sha256':digest(protocol),'profile_report_sha256':digest(profilepath),
        'profile_fingerprint':store.fingerprint,'HP1_report_sha256':digest(hp1path),'collections':bindings,
        'sources':{str(p):digest(p) for p in sources},'preflight_sha256':digest(preflight)},
        'folds':[],'quality_or_speed_acceptance':False,'scope':'teacher-oracle pair C transfer only, not a learned selector or dynamic FFN kernel'}
    atomic_json(path,meta);started=time.perf_counter()
    try:
        for step,keep in [(2,108),(3,162)]:
            part=store.load('C_profiles',f'step{step}_block20');profiles=normalized_profiles(part['arrays']['scores']).reshape(24,256,216)
            scopes={i:next(s for s in records[i]['scopes'] if s['step']==step and s['block']==20) for i in range(24)}
            if [v['observed_payload_sha256'] for v in part['metadata']['bindings']]!=[scopes[i]['sha256'] for i in range(24)]:
                raise RuntimeError('Profile rows no longer match original native coordinates')
            for fold in [0,1]:
                train=[i for i in range(24) if i%2!=fold];val=[i for i in range(24) if i%2==fold]
                mean=profiles[train].reshape(-1,216).mean(0);static=np.argsort(-mean,kind='stable')[:keep]
                cases=[]
                for index in val:
                    scope=scopes[index];grid=scope['grid'];total=scope['full_rows']
                    if len(grid)!=3 or math.prod(grid)!=total:raise RuntimeError('Original3D grid coordinates required')
                    item=transfer_statistics(profiles[index],np.array(scope['selected_rows']),total,64,int(grid[1]*grid[2]),static,keep)
                    item.update(index=index,source_group=records[index]['input']['source_group'],
                        native_scope_sha256=scope['sha256'],sampling_seed=scope['sampling_seed'])
                    cases.append(item)
                reference=next(v for v in hp1['CV'] if v['step']==step and v['fold']==fold)
                static_mean=float(np.mean([v['row_static_C_mean'] for v in cases]));oracle_mean=float(np.mean([v['row_oracle_C_mean'] for v in cases]))
                if not math.isclose(static_mean,reference['static_retained_C_fraction'],abs_tol=1e-12,rel_tol=0) or not math.isclose(oracle_mean,reference['unrestricted_oracle_retained_C_fraction'],abs_tol=1e-12,rel_tol=0):
                    raise RuntimeError('Old HP1 static/oracle row reference differs')
                within=[v['subsets']['within_tile']['HT_gain_mean'] for v in cases]
                far=[v['subsets']['other_tile_same_frame']['HT_gain_mean'] for v in cases]
                if any(v is None for v in within+far):raise RuntimeError('Population within/same-frame-other-tile pair subsets must exist')
                difference=np.asarray(within)-np.asarray(far)
                rng=np.random.default_rng(20261021+100*step+fold);draws=rng.integers(0,12,size=(10000,12))
                observed=sum(v['subsets']['within_tile']['sample_ordered_pairs'] for v in cases)
                row={'step':step,'block':20,'fold':fold,'train_inputs':train,'validation_inputs':val,'tile':64,'keep_groups':keep,
                    'matched_static_C_fraction':static_mean,'per_row_oracle_C_fraction':oracle_mean,
                    'equal_source_within_tile_HT_gain':float(np.mean(within)),
                    'equal_source_same_frame_other_tile_HT_gain':float(np.mean(far)),
                    'equal_source_within_minus_sameframe_other_tile_HT_gain':float(difference.mean()),
                    'within_source_bootstrap95':bootstrap_interval(within,draws),
                    'within_minus_far_source_bootstrap95':bootstrap_interval(difference,draws),
                    'sample_within_ordered_pairs':observed,'source_count':12,
                    'screen_pass':float(np.mean(within))>=.02 and float(difference.mean())>=.01 and observed>=500,
                    'cases':cases}
                meta['folds'].append(row);atomic_json(path,meta)
                print('TC1 '+json.dumps({k:v for k,v in row.items() if k!='cases'}),flush=True)
        passed=len(meta['folds'])==4 and all(v['screen_pass'] for v in meta['folds'])
        if torch.cuda.is_initialized():raise RuntimeError('CPU coherence description initialized CUDA')
        meta.update(status='COMPLETED_TC1_COHERENCE_SCREEN_PASSED' if passed else 'COMPLETED_TC1_COHERENCE_SCREEN_REJECTED',
            coherence_screen_pass=passed,elapsed_seconds=time.perf_counter()-started,CUDA_initialized=False,
            interpretation='finite-population pair support transfer in known C24 states; not a learned predictor, full-tile oracle measurement, native output-risk or video result')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:atomic_json(path,meta)


if __name__=='__main__':main()
