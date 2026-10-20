"""Run the fixed JR1 complementary reconstruction control on calibration-only folds."""
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

from anchored_reconstruction import bf16_bits,fit_reconstruction,from_bf16_bits
from ar2_checkpoint import CheckpointStore,atomic_json,digest
from evaluate_residual_rr1 import open_verified
from fit_cstatic_centered import load_training_reports
from joint_reconstruction import joint_update,output,project_rows
from reduced_rank_residual import predict_y
from run_ar2_restart import ROOT,tensor_sha
from run_residual_rr1 import decode,load_position


def position(records,cache,base,step):
    data=load_position(records,cache,base,step)
    expert='high_noise_model' if step==0 else 'low_noise_model'
    part=base.load('positions',f'step{step}_block20');arrays,info=part['arrays'],part['metadata']
    ids=torch.tensor((arrays['C_groups'][:,None]*64+np.arange(64)).reshape(-1),device='cuda')
    shard=next(s for s in cache['shards'] if s['expert']==expert and s['block']==20)
    # load_position already verified this exact shard SHA.
    payload=torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)
    prior=payload[expert]['blocks.20.ffn.2']['weight_qdq'].cuda().index_select(1,ids).contiguous()
    bias=torch.from_numpy(arrays['C_centered_gain_bias_b2']).to(device='cuda',dtype=torch.bfloat16)
    alpha=info['policies']['C']['modes']['centered_gain_bias']['alpha'];features=[]
    for index in range(12):
        scope=next(s for s in records[index]['scopes'] if s['step']==step and s['block']==20)
        with safe_open(scope['path'],framework='pt',device='cpu') as handle:
            a=handle.get_tensor('V2').cuda().index_select(1,ids)
        features.append((a.float()*alpha).to(torch.bfloat16))
    data.update(features=torch.cat(features),weight=prior,bias=bias,alpha=alpha,groups=arrays['C_groups'],base_info=info)
    if not torch.equal(project_rows(data['features'],prior,bias),data['base']):
        raise RuntimeError('Gain-preserving native main baseline did not reproduce')
    return data


def encode(model):
    return {'weight':bf16_bits(model['weight']),'bias':bf16_bits(model['bias']),
        **{'rr_'+k:bf16_bits(v) for k,v in model['residual'].items()}}


def decode_model(part):
    a=part['arrays']
    return {'weight':from_bf16_bits(a['weight'],'cuda'),'bias':from_bf16_bits(a['bias'],'cuda'),
        'residual':{k[3:]:from_bf16_bits(v,'cuda') for k,v in a.items() if k.startswith('rr_')}}


def sse(value,target,gate):
    return float(((value*gate).double()-target).square().sum())


def fit(data,mask,initial,quantizer,basis_rank):
    return joint_update(data['features'][mask],data['x'][mask],data['dense'][mask],data['base'][mask],
        data['gate'],data['weight'],data['bias'],initial,quantizer,rank=256,basis_rank=basis_rank)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--base-fits',type=Path,nargs=2,required=True)
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--yield-to-primary',type=Path)
    args=ap.parse_args()
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_weight_qdq as quantizer
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    if not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived')):
        raise RuntimeError('JR1 pilot must use the node HF derived cache')
    if shutil.disk_usage('/var/tmp').free<83*1024**3:raise RuntimeError('Keep80GiB reserve plus3GiB pilot allowance')
    args.out.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    records,bindings,capture=load_training_reports(args.collections);bases={};base_bindings=[]
    for path in args.base_fits:
        store,report,identity=open_verified(path,'COMPLETED_CALIBRATION_PARAMETER_FIT')
        if identity['training_collections']!=bindings:raise RuntimeError('Original12 base data identity differs')
        bases[report['expert']]=store;base_bindings.append({'path':str(path.resolve()),'sha256':digest(path),'fingerprint':store.fingerprint})
    parent,parent_report,parent_id=open_verified(args.parent,'COMPLETED_RR1_CV_AND_SELECTED_FINAL_FIT')
    if (parent_id['training_collections']!=bindings or parent_report['selected']['rank']!=256
            or parent_report['selected']['lambda']!=1 or parent_id['base_fits']!=base_bindings):
        raise RuntimeError('Frozen RR1 parent selection/data/base differ')
    cache_path=ROOT/'.runtime/hif4_rebuilt_cache.json';cache=json.loads(cache_path.read_text())
    if capture['assets']['cache_manifest_sha256']!=digest(cache_path):raise RuntimeError('Quantized cache changed')
    protocol_path=ROOT/'config/joint_reconstruction_jr1.json';protocol=json.loads(protocol_path.read_text())
    source_paths=[Path(__file__),ROOT/'tools/joint_reconstruction.py',ROOT/'tools/anchored_reconstruction.py',
        ROOT/'tools/reduced_rank_residual.py',ROOT/'tools/run_residual_rr1.py',
        ROOT/'.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py']
    store=CheckpointStore(args.out,{'protocol':protocol,'protocol_sha256':digest(protocol_path),
        'training_collections':bindings,'base_fits':base_bindings,'parent_report_sha256':digest(args.parent),
        'parent_fingerprint':parent.fingerprint,'cache_manifest_sha256':digest(cache_path),
        'torch':torch.__version__,'TF32':False,'sources':{str(p):digest(p) for p in source_paths}})
    meta={'status':'RUNNING','fingerprint':store.fingerprint,'checkpoint_root':str(args.out.resolve()),
        'CV':[],'quality_or_speed_acceptance':False}
    atomic_json(args.report,meta);started=time.perf_counter();positions={}
    def resource_yield():
        if args.yield_to_primary and args.yield_to_primary.exists():
            primary=json.loads(args.yield_to_primary.read_text())
            if primary['status'] in ['COMPLETED_CALIBRATION24_FROZEN_FORMAL_COMPARISON','FAILED_RESUMABLE']:
                meta.update(status='YIELDED_RESUMABLE_PRIMARY_VALIDATION',
                    elapsed_seconds=time.perf_counter()-started,resource_window_report_sha256=digest(args.yield_to_primary))
                return True
        return False
    try:
        for step in [0,2]:
            expert='high_noise_model' if step==0 else 'low_noise_model'
            data=position(records,cache,bases[expert],step);positions[step]=data
            for fold in [0,1]:
                if resource_yield():return
                name=f'step{step}_fold{fold}';previous=store.load('CV_models',name)
                if previous is None:
                    validation=torch.tensor([i%2==fold for i in range(12)],device='cuda').repeat_interleave(512);train=~validation
                    old=parent.load('CV_models',f'step{step}_fold{fold}_lambda1p0')
                    if old is None:raise RuntimeError('Frozen RR1 fold model missing')
                    full=decode(old['arrays'])
                    if {k:tensor_sha(v) for k,v in full.items()}!=old['metadata']['parameter_sha256']:
                        raise RuntimeError('Frozen RR1 native fold encoding differs')
                    initial=decode(old['arrays'],256)
                    target=(data['dense'][validation]*data['gate']).double()
                    baseline=sse(data['base'][validation]+predict_y(data['x'][validation],initial),target,data['gate'])
                    expected=next(r['native_SSE'] for r in parent_report['CV'] if r['step']==step and r['fold']==fold and r['rank']==256 and r['lambda']==1)
                    if not np.isclose(baseline,expected,rtol=1e-7,atol=1e-5):raise RuntimeError('Frozen RR1 CV native SSE did not reproduce')
                    # Matched WR-only control preserves the same activation gain.
                    wc,bc=fit_reconstruction(data['features'][train],data['dense'][train],data['base'][train],
                        data['gate'],data['weight'],data['bias'],1.)
                    wqc=quantizer(wc,force_fp32=True).to(torch.bfloat16).contiguous()
                    wr=sse(project_rows(data['features'][validation],wqc,bc),target,data['gate'])
                    wr_ideal=sse(project_rows(data['features'][validation],wc,bc),target,data['gate'])
                    model=fit(data,train,initial,quantizer,512)
                    native=output(data['features'][validation],data['x'][validation],model)
                    loss=sse(native,target,data['gate'])
                    control=output(data['features'][validation],data['x'][validation],model,fp32=True)*data['gate'].float()
                    fp32=float((control.double()-target).square().sum())
                    ideal={**model,'weight':model['pre_quant_weight']}
                    ideal_loss=sse(output(data['features'][validation],data['x'][validation],ideal),target,data['gate'])
                    row={'step':step,'fold':fold,'RR1_native_SSE':baseline,'matched_WR_native_SSE':wr,
                        'matched_WR_preQ_SSE':wr_ideal,'JR1_native_SSE':loss,'JR1_FP32_SSE':fp32,'JR1_preQ_SSE':ideal_loss,
                        'JR1_over_RR1':loss/baseline,'JR1_over_WR':loss/wr,
                        'gate_pass':loss<=.8*baseline and loss<=.9*wr,'RR1_parent_reproduced':True}
                    store.save('CV_models',name,encode(model),row);previous=store.load('CV_models',name)
                meta['CV'].append(previous['metadata']);atomic_json(args.report,meta)
                print('JR1 CV '+json.dumps(previous['metadata']),flush=True)
        meta['CV_gate_pass']=len(meta['CV'])==4 and all(r['gate_pass'] for r in meta['CV'])
        if not meta['CV_gate_pass']:
            meta['status']='COMPLETED_JR1_CV_GATE_REJECTED'
        else:
            for step,data in positions.items():
                if resource_yield():return
                name=f'step{step}_block20'
                if store.load('final_models',name) is None:
                    old=parent.load('final_models',name)
                    if old is None:raise RuntimeError('RR1 full12 pilot model missing')
                    initial=decode(old['arrays'])
                    if {k:tensor_sha(v) for k,v in initial.items()}!=old['metadata']['parameter_sha256']:
                        raise RuntimeError('RR1 full12 parameter encoding differs')
                    train=torch.ones(len(data['x']),dtype=torch.bool,device='cuda')
                    model=fit(data,train,initial,quantizer,256)
                    store.save('final_models',name,encode(model),{'step':step,'block':20,'alpha':data['alpha'],
                        'gate_sha256':data['base_info']['gate_sha256'],'original_W2_sha256':data['base_info']['weight_sha256'],
                        'weight_sha256':tensor_sha(model['weight']),'bias_sha256':tensor_sha(model['bias']),
                        'residual_parameter_sha256':{k:tensor_sha(v) for k,v in model['residual'].items()}})
            meta['status']='COMPLETED_JR1_CV_AND_FIXED_PILOT_FIT'
        meta['elapsed_seconds']=time.perf_counter()-started
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(args.report,meta);atomic_json(args.out/'run_meta.json',meta)


if __name__=='__main__':main()
