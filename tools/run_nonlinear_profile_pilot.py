"""NP1: nonlinear versus equally expanded linear features for C prediction only."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from ar2_checkpoint import CheckpointStore,atomic_json,digest
from evaluate_residual_rr1 import open_verified
from hidden_profile_head import fit_head,predict_head,retained_fraction,support
from onpolicy_residual import load_collections,position_cases
from router_bank_feasibility import normalized_profiles
from run_ar2_restart import ROOT,tensor_sha
from run_hidden_profile_pilot import hidden_full,read_parameters
from run_late_rank_pilot import checked_store


@torch.no_grad()
def both_features(case,residual,first,smooth,quantizer):
    linear,ids=hidden_full(case,residual)
    z,zids=hidden_full(case,first)
    if not torch.equal(ids,zids):raise RuntimeError('Native feature row coordinates differ')
    nonlinear=quantizer((F.gelu(z,approximate='tanh').float()/smooth).to(z.dtype),force_fp32=True)
    return torch.cat([linear,z],dim=-1),torch.cat([linear,nonlinear],dim=-1),ids


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();r=ROOT/'runs/worker_restart_20261020'
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    if not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived')) or shutil.disk_usage('/var/tmp').free<82*1024**3:
        raise RuntimeError('Node HF cache and80GiB reserve plus2GiB allowance required')
    nfpath=r/'nonlinear_residual_nf1/RUN.json';nf,nfreport,nfid=open_verified(nfpath,'COMPLETED_NF1_NATIVE_GATE_REJECTED')
    if len(nfreport['CV'])!=4:raise RuntimeError('Finish all4 frozen NF1 output-readout folds')
    for path,sha in nfid['sources'].items():
        if digest(path)!=sha:raise RuntimeError('Frozen NF1 numerical source changed')
    records,bindings,capture=load_collections([r/f'onpolicy_calibration24/shard{s}/RUN.json' for s in [0,1]],ROOT)
    parentpath=r/'calibration24_residual_fit/C24_RR_policy.json';parent=json.loads(parentpath.read_text())
    if capture['student_policy_sha256']!=digest(parentpath) or nfid['collection_bindings']!=bindings:
        raise RuntimeError('Frozen observation rows changed')
    old=checked_store(parent['residual_fit'],'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    base=checked_store(parent['fit_reports']['low_noise_model'],'COMPLETED_CALIBRATION_PARAMETER_FIT')
    ub1path=r/'ub1_step_budget_fit/UB1_policy.json';ub1=json.loads(ub1path.read_text())
    if nfid['UB1_policy_sha256']!=digest(ub1path):raise RuntimeError('Frozen UB1 feature reference changed')
    changed=checked_store(ub1['changed_fits']['3'],'COMPLETED_UB1_STEP40_FIT')
    cb=ub1['cache_manifest']
    if digest(cb['path'])!=cb['sha256']:raise RuntimeError('Effective cache identity changed')
    cache=json.loads(Path(cb['path']).read_text())
    prior,_,priorid=open_verified(r/'router_bank_rb1/RUN.json','COMPLETED_RB1_PROXY_GATE_REJECTED')
    if priorid['collection_bindings']!=bindings:raise RuntimeError('Teacher profile collection changed')
    hp1path=r/'hidden_profile_hp1/RUN.json';hp1=json.loads(hp1path.read_text())
    if hp1['status']!='COMPLETED_HP1_PROXY_GATE_REJECTED':raise RuntimeError('Frozen HP1 reference required')
    hpsha={(v['step'],v['index']):v['hidden_sha256'] for v in hp1['controls']}
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    sources=[Path(__file__),ROOT/'tools/hidden_profile_head.py',ROOT/'tools/run_hidden_profile_pilot.py',
        ROOT/'tools/onpolicy_residual.py',ROOT/'tools/router_bank_feasibility.py',ROOT/'tools/run_late_rank_pilot.py',
        ROOT/'tools/run_nonlinear_residual_pilot.py',ROOT/'.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py']
    protocol=ROOT/'config/nonlinear_profile_np1.json'
    args.out.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    store=CheckpointStore(args.out,{'protocol_sha256':digest(protocol),'protocol':json.loads(protocol.read_text()),
        'NF1_report_sha256':digest(nfpath),'NF1_fingerprint':nf.fingerprint,'HP1_report_sha256':digest(hp1path),
        'collection_bindings':bindings,'RB1_profile_fingerprint':prior.fingerprint,'UB1_policy_sha256':digest(ub1path),
        'sources':{str(p):digest(p) for p in sources},'torch':torch.__version__,'TF32':False})
    meta={'status':'RUNNING','checkpoint_root':str(args.out),'fingerprint':store.fingerprint,'controls':[],'CV':[],
        'quality_or_speed_acceptance':False,'scope':'per-row C profile proxy; no native FFN risk or dynamic kernel/video acceptance',
        'source_exposure':'only new heads source-split, frozen main and residual already saw all24'}
    atomic_json(args.report,meta);started=time.perf_counter()
    try:
        shard=next(v for v in cache['shards'] if v['expert']=='low_noise_model' and v['block']==20)
        if digest(shard['path'])!=shard['sha256']:raise RuntimeError('Original effective shard changed')
        payload=torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)['low_noise_model']
        view=json.loads((ROOT/'.runtime/node_local_model_view.json').read_text())
        raw=next(v for v in view['files'] if '/low_noise_model/' in v['view_path'] and v['filename'].endswith('.safetensors'))
        with safe_open(raw['view_path'],framework='pt',device='cpu') as handle:b0=handle.get_tensor('blocks.20.ffn.0.bias').cuda().bfloat16()
        for step,keep in [(2,108),(3,162)]:
            info=base.load('positions',f'step{step}_block20')['metadata'];cases=position_cases(records,step,20,info)
            residual=read_parameters(old if step==2 else changed,step,changed=step==3)
            part=prior.load('C_profiles',f'step{step}_block20');profilemeta=part['metadata']
            if ([v['observed_payload_sha256'] for v in profilemeta['bindings']]!=[c['scope']['sha256'] for c in cases]
                    or profilemeta['weight_sha256']!=tensor_sha(payload['blocks.20.ffn.2']['weight_qdq'])
                    or profilemeta['gate_sha256']!=tensor_sha(cases[0]['gate'])):
                raise RuntimeError('Exact normalized teacher C data required')
            profiles=torch.from_numpy(normalized_profiles(part['arrays']['scores'])).cuda()
            for case in cases:
                if tensor_sha(b0)!=case['scope']['original_b0_sha256']:raise RuntimeError('Original raw W0 bias changed')
                full,ids=hidden_full(case,residual)
                sample=full.reshape(-1,256).index_select(0,ids)
                if tensor_sha(sample)!=hpsha[step,case['index']]:raise RuntimeError('Original residual feature changed')
                meta['controls'].append({'step':step,'index':case['index'],'HP1_hidden_bits_equal':True})
            for fold in [0,1]:
                key=f'step{step}_fold{fold}';previous=store.load('heads',key)
                if previous is None:
                    train=[c for c in cases if c['index']%2!=fold];val=[c for c in cases if c['index']%2==fold]
                    if len(train)!=12 or len(val)!=12 or sum(c['index']<12 for c in train)!=6:
                        raise RuntimeError('Fixed12/12 balanced source folds required')
                    parentpart=nf.load('heads',key);nfmeta=parentpart['metadata']
                    if nfmeta['train_inputs']!=[c['index'] for c in train] or nfmeta['validation_inputs']!=[c['index'] for c in val]:
                        raise RuntimeError('Frozen NF1 feature selection folds differ')
                    groups=torch.from_numpy(parentpart['arrays']['groups']).cuda().long()
                    if len(groups)!=4 or len(groups.unique())!=4 or groups.cpu().tolist()!=nfmeta['omitted_groups']:
                        raise RuntimeError('Exactly the same4 original omitted groups required')
                    channels=(groups[:,None]*64+torch.arange(64,device='cuda')).reshape(-1)
                    first={'first':payload['blocks.20.ffn.0']['weight_qdq'].cuda().index_select(0,channels).contiguous(),
                        'bias_first':b0.index_select(0,channels).contiguous()}
                    smooth=payload['blocks.20.ffn.2']['smooth_scale'].cuda().index_select(0,channels).contiguous()
                    if (tensor_sha(first['first'])!=nfmeta['feature_weight_sha256'] or tensor_sha(first['bias_first'])!=nfmeta['feature_raw_bias_sha256']
                            or tensor_sha(smooth)!=nfmeta['smooth2_sha256']):raise RuntimeError('Actual nonlinear feature basis differs from NF1')
                    def profile_ids(items):
                        return torch.tensor(np.concatenate([np.arange(c['index']*256,(c['index']+1)*256) for c in items]),device='cuda')
                    train_profiles,val_profiles=profiles[profile_ids(train)],profiles[profile_ids(val)]
                    static=retained_fraction(val_profiles,support(train_profiles.mean(0,keepdim=True),keep).expand(len(val_profiles),-1))
                    oracle=retained_fraction(val_profiles,support(val_profiles,keep))
                    hp=next(v for v in hp1['CV'] if v['step']==step and v['fold']==fold)
                    if not math.isclose(static,hp['static_retained_C_fraction'],abs_tol=1e-12,rel_tol=0) or not math.isclose(oracle,hp['unrestricted_oracle_retained_C_fraction'],abs_tol=1e-12,rel_tol=0):
                        raise RuntimeError('Original HP1 static/oracle proxy reference changed')
                    xs=[[],[]]
                    for case in train:
                        linear,nonlinear,ids=both_features(case,residual,first,smooth,quantizer)
                        for j,value in enumerate([linear,nonlinear]):xs[j].append(value.reshape(-1,512).index_select(0,ids))
                    heads=[fit_head(torch.cat(x),train_profiles,False) for x in xs]
                    retained=[]
                    for j,head in enumerate(heads):
                        selections=[]
                        for case in val:
                            linear,nonlinear,ids=both_features(case,residual,first,smooth,quantizer)
                            scores=predict_head([linear,nonlinear][j],head,False).reshape(-1,216).index_select(0,ids)
                            selections.append(support(scores,keep))
                        retained.append(retained_fraction(val_profiles,torch.cat(selections)))
                    gain=oracle-static;capture_fraction=(retained[1]-static)/gain if gain>0 else 0.
                    extra=retained[1]-retained[0]
                    row={'step':step,'block':20,'fold':fold,'train_inputs':[c['index'] for c in train],
                        'validation_inputs':[c['index'] for c in val],'groups':groups.cpu().tolist(),'keep_groups':keep,
                        'static_retained_C_fraction':static,'unrestricted_oracle_retained_C_fraction':oracle,
                        'matched_linear512_retained_C_fraction':retained[0],'nonlinear512_retained_C_fraction':retained[1],
                        'HP1_linear256_context_retained_C_fraction':hp['linear_native_retained_C_fraction'],
                        'oracle_minus_static':gain,'nonlinear_capture_fraction':capture_fraction,'nonlinear_minus_matched_linear':extra,
                        'gate_pass':gain>=.02 and capture_fraction>=.5 and extra>=.002,
                        'actual_original_fullM_feature_and_FP32_head_inference':True,'feature_basis_matches_NF1':True,
                        'extra_BF16_input_projection_GEMMs':1,'FP32_head_shape':[216,512]}
                    arrays={'groups':groups.cpu().numpy(),**{label+'_'+k:v.cpu().numpy() for label,head in zip(['linear','nonlinear'],heads) for k,v in head.items()}}
                    store.save('heads',key,arrays,row);previous=store.load('heads',key)
                meta['CV'].append(previous['metadata']);atomic_json(args.report,meta)
                print('NP1 '+json.dumps(previous['metadata']),flush=True)
        passed=len(meta['CV'])==4 and all(v['gate_pass'] for v in meta['CV'])
        meta.update(status='COMPLETED_NP1_PROXY_GATE_PASSED' if passed else 'COMPLETED_NP1_PROXY_GATE_REJECTED',
            proxy_gate_pass=passed,elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(args.report,meta);atomic_json(args.out/'run_meta.json',meta)


if __name__=='__main__':main()
