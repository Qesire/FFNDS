"""Calibrate WR1 with native weight re-quantization, selecting only on calibration."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from anchored_reconstruction import fit_reconstruction,bf16_bits,from_bf16_bits
from ar2_checkpoint import CheckpointStore,atomic_json,digest
from evaluate_residual_rr1 import open_verified
from fit_cstatic_centered import load_training_reports
from run_ar2_restart import ROOT,tensor_sha


def load_position(records,cache,model,step,block=20):
    expert='high_noise_model' if step<2 else 'low_noise_model'
    shard=next(s for s in cache['shards'] if s['expert']==expert and s['block']==block)
    if digest(shard['path'])!=shard['sha256']:raise RuntimeError('Effective W2 shard changed')
    payload=torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)
    w=payload[expert][f'blocks.{block}.ffn.2']['weight_qdq'].cuda()
    part=model.load('positions',f'step{step}_block{block:02d}')
    if part is None:raise RuntimeError('Missing frozen base')
    arrays,info=part['arrays'],part['metadata']
    if tensor_sha(w)!=info['weight_sha256']:raise RuntimeError('Frozen base W2 differs')
    channels=torch.tensor((arrays['C_groups'][:,None]*64+np.arange(64)).reshape(-1),device='cuda')
    packed=w[:,channels].contiguous()
    parameters=info['policies']['C']['modes']['centered_gain_bias']
    bias=torch.from_numpy(arrays['C_centered_gain_bias_b2']).cuda().to(torch.bfloat16)
    if tensor_sha(bias)!=parameters['b2_sha256']:raise RuntimeError('Frozen base bias differs')
    data={k:[] for k in ['x','dense','base']}
    for index in range(12):
        scope=next(s for s in records[index]['scopes'] if s['step']==step and s['block']==block)
        if scope['rows']!=512 or scope['expert']!=expert or digest(scope['path'])!=scope['sha256']:
            raise RuntimeError('Training operands differ')
        with safe_open(scope['path'],framework='pt',device='cpu') as handle:
            a,y,g,b=[handle.get_tensor(k).cuda() for k in ['V2','Y','gate','b2']]
        gate=g.reshape(1,5120)
        if tensor_sha(gate)!=info['gate_sha256'] or tensor_sha(b)!=info['original_b2_sha256']:
            raise RuntimeError('Training gate or original b2 changed')
        x=a[:,channels].contiguous()
        base=F.linear((x.float()*parameters['alpha']).to(torch.bfloat16),packed,bias)
        for k,v in [('x',x),('dense',y),('base',base)]:data[k].append(v)
    data={k:torch.cat(v) for k,v in data.items()}
    data.update(gate=gate,prior_weight=packed.float()*parameters['alpha'],prior_bias=bias,
        groups=arrays['C_groups'],base_info=info)
    return data


def native_fit(data,train,regularization,quantizer):
    w,b=fit_reconstruction(data['x'][train],data['dense'][train],data['base'][train],data['gate'],
        data['prior_weight'],data['prior_bias'],regularization)
    quantized=quantizer(w,force_fp32=True).to(torch.bfloat16).contiguous()
    if quantized.shape!=w.shape or not bool(torch.isfinite(quantized).all()):
        raise RuntimeError('Native HiF4 reconstructed weight invalid')
    return w,quantized,b


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--base-fits',type=Path,nargs=2,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_weight_qdq as quantizer
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    records,bindings,capture=load_training_reports(args.collections)
    bases,base_bindings={},[]
    for path in args.base_fits:
        model,report,identity=open_verified(path,'COMPLETED_CALIBRATION_PARAMETER_FIT')
        if identity['training_collections']!=bindings:raise RuntimeError('Base training collection differs')
        bases[report['expert']]=model;base_bindings.append({'path':str(path.resolve()),'sha256':digest(path),'fingerprint':model.fingerprint})
    cache_path=ROOT/'.runtime/hif4_rebuilt_cache.json';cache=json.loads(cache_path.read_text())
    if capture['assets']['cache_manifest_sha256']!=digest(cache_path):raise RuntimeError('Training cache differs')
    protocol_path=ROOT/'config/weight_reconstruction_wr1.json';protocol=json.loads(protocol_path.read_text())
    store=CheckpointStore(args.out,{'protocol':protocol,'protocol_sha256':digest(protocol_path),
        'training_collections':bindings,'base_fits':base_bindings,'cache_manifest_sha256':digest(cache_path),
        'torch':torch.__version__,'TF32':False,'sources':{str(p):digest(p) for p in [Path(__file__),
            ROOT/'tools/anchored_reconstruction.py',ROOT/'.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py']}})
    meta={'status':'RUNNING','fingerprint':store.fingerprint,'checkpoint_root':str(args.out.resolve()),
        'CV':[],'selected':None,'quality_or_speed_acceptance':False}
    started=time.perf_counter();atomic_json(args.report,meta)
    try:
        positions={}
        for step in [0,2]:
            expert='high_noise_model' if step==0 else 'low_noise_model'
            data=load_position(records,cache,bases[expert],step);positions[step]=data
            for fold in [0,1]:
                validation=torch.tensor([i%2==fold for i in range(12)],device='cuda').repeat_interleave(512)
                target=(data['dense'][validation]*data['gate']).double()
                base_sse=float(((data['base'][validation]*data['gate']).double()-target).square().sum())
                for regularization in protocol['lambdas']:
                    name=f'step{step}_fold{fold}_lambda{str(regularization).replace(".","p")}'
                    previous=store.load('CV',name)
                    if previous is None:
                        w,qw,b=native_fit(data,~validation,regularization,quantizer)
                        native=F.linear(data['x'][validation],qw,b)*data['gate']
                        ideal=F.linear(data['x'][validation],w,b)*data['gate']
                        sse=float((native.double()-target).square().sum())
                        ideal_sse=float((ideal.double()-target).square().sum())
                        row={'step':step,'fold':fold,'lambda':regularization,'native_SSE':sse,'base_native_SSE':base_sse,
                            'ratio':sse/base_sse,'pre_requantization_BF16_ratio':ideal_sse/base_sse,
                            'native_weight_QDQ_relative_L2':float((qw.double()-w.double()).norm()/w.double().norm())}
                        store.save('CV',name,{'weight':bf16_bits(qw),'bias':bf16_bits(b)},row)
                        previous=store.load('CV',name)
                    row=previous['metadata'];meta['CV'].append(row);atomic_json(args.report,meta)
                    print(f'CV step{step} fold{fold} lambda{regularization}: quantized/base={row["ratio"]:.6g}, preQ={row["pre_requantization_BF16_ratio"]:.6g}',flush=True)
        candidates=[]
        for regularization in protocol['lambdas']:
            rows=[r for r in meta['CV'] if r['lambda']==regularization]
            if len(rows)==4 and all(r['ratio']<=.5 for r in rows):
                candidates.append((sum(r['ratio'] for r in rows)/4,regularization))
        if not candidates:
            meta['status']='COMPLETED_WR1_NO_CV_CANDIDATE'
        else:
            _,regularization=min(candidates);meta['selected']={'lambda':regularization,'selection':'native requantized calibration-only CV'}
            for step,data in positions.items():
                name=f'step{step}_block20'
                if store.load('final_models',name) is None:
                    train=torch.ones(len(data['x']),dtype=torch.bool,device='cuda')
                    w,qw,b=native_fit(data,train,regularization,quantizer)
                    store.save('final_models',name,{'weight':bf16_bits(qw),'bias':bf16_bits(b),'groups':data['groups']},
                        {'step':step,'block':20,'lambda':regularization,'weight_sha256':tensor_sha(qw),'bias_sha256':tensor_sha(b),
                            'base_weight_sha256':data['base_info']['weight_sha256'],'gate_sha256':data['base_info']['gate_sha256']})
            meta['status']='COMPLETED_WR1_CV_AND_SELECTED_FINAL_FIT'
        meta['elapsed_seconds']=time.perf_counter()-started
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(args.report,meta);atomic_json(args.out/'run_meta.json',meta)
    print(json.dumps({k:v for k,v in meta.items() if k!='CV'},indent=2))


if __name__=='__main__':main()
