"""NF1: native residual correction from omitted nonlinear features, four fixed folds."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from anchored_reconstruction import bf16_bits
from ar2_checkpoint import CheckpointStore,atomic_json,digest
from evaluate_residual_rr1 import open_verified
from hidden_profile_head import fit_head
from onpolicy_residual import load_collections,native_sse,replay_rows,same_bits
from router_bank_feasibility import normalized_profiles
from run_ar2_restart import ROOT,tensor_sha
from run_folded_adapter_cv import prepare
from run_hidden_profile_pilot import hidden_full,read_parameters
from run_late_rank_pilot import checked_store,ub1_position


def native_head(features,target):
    fitted=fit_head(features,target,False)
    result={k:v.to(torch.bfloat16).contiguous() for k,v in fitted.items()}
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise RuntimeError('Nonfinite native NF1 readout parameters')
    return result


@torch.no_grad()
def nonlinear_full(case,parameters,smooth,quantizer):
    z,ids=hidden_full(case,parameters)
    h=F.gelu(z,approximate='tanh')
    h=quantizer((h.float()/smooth).to(h.dtype),force_fp32=True)
    return h,ids


@torch.no_grad()
def risk(cases,feature_parameters,smooth,quantizer,head,nonlinear):
    total=0.
    for case in cases:
        if nonlinear:
            full,ids=nonlinear_full(case,feature_parameters,smooth,quantizer)
        else:
            full,ids=hidden_full(case,feature_parameters)
        correction=F.linear(full,head['weight'],head['bias']).reshape(-1,5120).index_select(0,ids)
        total+=native_sse(case['UB1StudentY']+correction,case['TeacherY'],case['gate'])
    return total


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();r=ROOT/'runs/worker_restart_20261020'
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    terminal=r/'diverse_confirmation20_controller/RUN.json'
    state=json.loads(terminal.read_text())
    if not (state['status'].startswith('COMPLETED_DIVERSE_CONFIRMATION20_') or state['status']=='NOT_RUN_OBSERVED_UB2_QUALITY_REJECTED'):
        raise RuntimeError('Current UB2 quality/engineering/confirmation pipeline must be terminal before NF1')
    if not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived')) or shutil.disk_usage('/var/tmp').free<82*1024**3:
        raise RuntimeError('Node HF parameters and80GiB reserve plus2GiB allowance required')
    records,bindings,capture=load_collections([r/f'onpolicy_calibration24/shard{s}/RUN.json' for s in [0,1]],ROOT)
    parentpath=r/'calibration24_residual_fit/C24_RR_policy.json';parent=json.loads(parentpath.read_text())
    if digest(parentpath)!=capture['student_policy_sha256']:raise RuntimeError('Original observation student changed')
    old=checked_store(parent['residual_fit'],'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    ub1path=r/'ub1_step_budget_fit/UB1_policy.json';ub1=json.loads(ub1path.read_text())
    if ub1['parent_policy']['sha256']!=digest(parentpath):raise RuntimeError('Frozen UB1 parent changed')
    changed=checked_store(ub1['changed_fits']['3'],'COMPLETED_UB1_STEP40_FIT')
    base=checked_store(parent['fit_reports']['low_noise_model'],'COMPLETED_CALIBRATION_PARAMETER_FIT')
    cb=ub1['cache_manifest']
    if digest(cb['path'])!=cb['sha256']:raise RuntimeError('Effective cache manifest changed')
    cache=json.loads(Path(cb['path']).read_text())
    prior,_,prior_id=open_verified(r/'router_bank_rb1/RUN.json','COMPLETED_RB1_PROXY_GATE_REJECTED')
    for path,sha in prior_id['sources'].items():
        if digest(path)!=sha:raise RuntimeError('Frozen teacher-profile numerical source changed')
    if prior_id['collection_bindings']!=bindings or prior_id['student_policy_sha256']!=digest(parentpath):
        raise RuntimeError('Frozen teacher C profile source changed')
    hp1path=r/'hidden_profile_hp1/RUN.json';hp1=json.loads(hp1path.read_text())
    if hp1['status']!='COMPLETED_HP1_PROXY_GATE_REJECTED':raise RuntimeError('Preserved HP1 controls required')
    hp1_hidden={(v['step'],v['index']):v['hidden_sha256'] for v in hp1['controls']}
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    sources=[Path(__file__),ROOT/'tools/hidden_profile_head.py',ROOT/'tools/run_hidden_profile_pilot.py',
        ROOT/'tools/run_late_rank_pilot.py',ROOT/'tools/run_folded_adapter_cv.py',ROOT/'tools/folded_adapter.py',
        ROOT/'tools/onpolicy_residual.py',ROOT/'tools/router_bank_feasibility.py',
        ROOT/'.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py']
    args.out.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=ROOT/'config/nonlinear_residual_nf1.json'
    store=CheckpointStore(args.out,{'protocol_sha256':digest(protocol),'protocol':json.loads(protocol.read_text()),
        'collection_bindings':bindings,'UB1_policy_sha256':digest(ub1path),'RB1_profile_fingerprint':prior.fingerprint,
        'HP1_report_sha256':digest(hp1path),'UB2_pipeline_terminal_sha256':digest(terminal),
        'sources':{str(p):digest(p) for p in sources},'torch':torch.__version__,'TF32':False})
    meta={'status':'RUNNING','checkpoint_root':str(args.out),'fingerprint':store.fingerprint,'controls':[],'CV':[],
        'quality_or_speed_acceptance':False,'scope':'native local fixed-UB1 residual correction on OP1 C24 states, two source folds only; not new policy own-trajectory validation'}
    atomic_json(args.report,meta);started=time.perf_counter()
    try:
        for step in [2,3]:
            if step==2:
                data=prepare(records,cache,parent,old,2,quantizer)
                cases=data['cases'];control=data['control']
                groups=base.load('positions','step2_block20')['arrays']['C_groups']
                for case in cases:case['UB1StudentY']=case['StudentY']
            else:
                cases,control=ub1_position(records,cache,parent,old,changed,20,quantizer)
                groups=changed.load('positions','step3_block20')['arrays']['groups']
                for case in cases:case['UB1StudentY']=case['UB1_main']+case['UB1_frozen_R']
            residual=read_parameters(old if step==2 else changed,step,changed=step==3)
            profiles_part=prior.load('C_profiles',f'step{step}_block20')
            profile_meta=profiles_part['metadata']
            if ([v['observed_payload_sha256'] for v in profile_meta['bindings']]!=[c['scope']['sha256'] for c in cases]
                    or profile_meta['gate_sha256']!=tensor_sha(cases[0]['gate'])):
                raise RuntimeError('Exact teacher profile rows/gate required')
            profiles=torch.from_numpy(normalized_profiles(profiles_part['arrays']['scores'])).cuda()
            for case in cases:
                full,ids=hidden_full(case,residual)
                projected=F.linear(full,residual['second'],residual['bias_second']).reshape(-1,5120).index_select(0,ids)
                if not same_bits(projected,replay_rows(case['V1'],case['scope'],residual)):
                    raise RuntimeError('Original UB1 split residual native bits changed')
                case['linear_hidden']=full.reshape(-1,256).index_select(0,ids).contiguous()
                if tensor_sha(case['linear_hidden'])!=hp1_hidden[step,case['index']]:
                    raise RuntimeError('Linear control feature differs from frozen HP1')
            meta['controls'].append({'step':step,'original_controls':control,'HP1_all24_hidden_bits_equal':True})
            atomic_json(args.report,meta)
            shard=next(v for v in cache['shards'] if v['expert']=='low_noise_model' and v['block']==20)
            if digest(shard['path'])!=shard['sha256']:raise RuntimeError('Original effective shard changed')
            payload=torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)['low_noise_model']
            if tensor_sha(payload['blocks.20.ffn.2']['weight_qdq'])!=profile_meta['weight_sha256']:
                raise RuntimeError('Teacher-profile effective weight differs')
            view=json.loads((ROOT/'.runtime/node_local_model_view.json').read_text())
            raw=next(v for v in view['files'] if '/low_noise_model/' in v['view_path'] and v['filename'].endswith('.safetensors'))
            with safe_open(raw['view_path'],framework='pt',device='cpu') as handle:b0=handle.get_tensor('blocks.20.ffn.0.bias').cuda().bfloat16()
            if any(tensor_sha(b0)!=c['scope']['original_b0_sha256'] for c in cases):raise RuntimeError('Original native W0 bias changed')
            available=torch.tensor(sorted(set(range(216))-set(map(int,groups))),device='cuda')
            for fold in [0,1]:
                key=f'step{step}_fold{fold}';part=store.load('heads',key)
                if part is None:
                    train=[c for c in cases if c['index']%2!=fold];val=[c for c in cases if c['index']%2==fold]
                    if len(train)!=12 or len(val)!=12 or sum(c['index']<12 for c in train)!=6:
                        raise RuntimeError('Fixed balanced12/12 source folds required')
                    train_rows=torch.tensor(np.concatenate([np.arange(c['index']*256,(c['index']+1)*256) for c in train]),device='cuda')
                    means=profiles[train_rows].mean(0)
                    selected=available[torch.argsort(means[available],descending=True,stable=True)[:4]].sort().values
                    channels=(selected[:,None]*64+torch.arange(64,device='cuda')).reshape(-1)
                    first={'first':payload['blocks.20.ffn.0']['weight_qdq'].cuda().index_select(0,channels).contiguous(),
                        'bias_first':b0.index_select(0,channels).contiguous()}
                    smooth=payload['blocks.20.ffn.2']['smooth_scale'].cuda().index_select(0,channels).contiguous()
                    samples=[]
                    for case in train:
                        full,ids=nonlinear_full(case,first,smooth,quantizer)
                        samples.append(full.reshape(-1,256).index_select(0,ids).contiguous())
                    target=torch.cat([c['TeacherY'].float()-c['UB1StudentY'].float() for c in train])
                    linear=native_head(torch.cat([c['linear_hidden'] for c in train]),target)
                    nonlinear=native_head(torch.cat(samples),target)
                    original=sum(native_sse(c['UB1StudentY'],c['TeacherY'],c['gate']) for c in val)
                    linear_sse=risk(val,residual,None,quantizer,linear,False)
                    nonlinear_sse=risk(val,first,smooth,quantizer,nonlinear,True)
                    row={'step':step,'block':20,'fold':fold,'train_inputs':[c['index'] for c in train],
                        'validation_inputs':[c['index'] for c in val],'omitted_groups':selected.cpu().tolist(),
                        'frozen_UB1_native_SSE':original,'linear_readout_native_SSE':linear_sse,'nonlinear_readout_native_SSE':nonlinear_sse,
                        'nonlinear_over_frozen':nonlinear_sse/original,'nonlinear_over_linear':nonlinear_sse/linear_sse,
                        'gate_pass':nonlinear_sse<=.9*original and nonlinear_sse<=.95*linear_sse,
                        'actual_original_fullM_BF16_inference':True,'feature_weight_sha256':tensor_sha(first['first']),
                        'feature_raw_bias_sha256':tensor_sha(first['bias_first']),'smooth2_sha256':tensor_sha(smooth),
                        'feature_width':256,'extra_linear_control_GEMMs':1,'extra_nonlinear_GEMMs':2,
                        'extra_nonlinear_GELU_and_activation_QDQ':True}
                    arrays={'groups':selected.cpu().numpy(),**{label+'_'+k:bf16_bits(v) for label,head in [('linear',linear),('nonlinear',nonlinear)] for k,v in head.items()}}
                    store.save('heads',key,arrays,row);part=store.load('heads',key)
                meta['CV'].append(part['metadata']);atomic_json(args.report,meta)
                print('NF1 '+json.dumps(part['metadata']),flush=True)
            del cases,profiles,payload
        passed=len(meta['CV'])==4 and all(v['gate_pass'] for v in meta['CV'])
        meta.update(status='COMPLETED_NF1_NATIVE_GATE_PASSED' if passed else 'COMPLETED_NF1_NATIVE_GATE_REJECTED',
            native_gate_pass=passed,elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(args.report,meta);atomic_json(args.out/'run_meta.json',meta)


if __name__=='__main__':main()
