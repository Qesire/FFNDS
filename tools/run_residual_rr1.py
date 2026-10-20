"""Select low-rank residual correction using calibration-only source folds."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import CheckpointStore,atomic_json,digest
from fit_cstatic_centered import load_training_reports
from reduced_rank_residual import fit_rrr,native_parameters,predict_y
from run_ar2_restart import ROOT,tensor_sha


def encode(parameters):
    return {k:v.detach().cpu().float().numpy() for k,v in parameters.items()}


def decode(arrays,rank=None):
    result={k:torch.from_numpy(v).cuda().to(torch.bfloat16) for k,v in arrays.items()}
    if rank is not None:
        result['first']=result['first'][:rank]
        result['bias_first']=result['bias_first'][:rank]
        result['second']=result['second'][:,:rank]
    return result


def load_position(records,cache,model,step):
    expert='high_noise_model' if step==0 else 'low_noise_model'
    shard=next(s for s in cache['shards'] if s['expert']==expert and s['block']==20)
    if digest(shard['path'])!=shard['sha256']:
        raise RuntimeError('Effective W2 shard changed')
    payload=torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)
    w=payload[expert]['blocks.20.ffn.2']['weight_qdq'].cuda()
    part=model.load('positions',f'step{step}_block20')
    arrays,info=part['arrays'],part['metadata']
    if tensor_sha(w)!=info['weight_sha256']:
        raise RuntimeError('Base policy W2 differs')
    groups=arrays['C_groups']
    channels=torch.tensor((groups[:,None]*64+np.arange(64)).reshape(-1),device='cuda')
    packed=w.index_select(1,channels).contiguous()
    params=info['policies']['C']['modes']['centered_gain_bias']
    b2=torch.from_numpy(arrays['C_centered_gain_bias_b2']).cuda().to(torch.bfloat16)
    if tensor_sha(b2)!=params['b2_sha256']:
        raise RuntimeError('Native base bias encoding changed')
    data={k:[] for k in ['x','base','dense']}
    common_gate=None
    for index in range(12):
        scope=next(s for s in records[index]['scopes'] if s['step']==step and s['block']==20)
        if digest(scope['path'])!=scope['sha256']:
            raise RuntimeError('Calibration raw tensor changed')
        with safe_open(scope['path'],framework='pt',device='cpu') as handle:
            x,a,y,gate=[handle.get_tensor(k).cuda() for k in ['V1','V2','Y','gate']]
        gate=gate.reshape(1,5120)
        if tensor_sha(gate)!=info['gate_sha256'] or (common_gate is not None and tensor_sha(gate)!=tensor_sha(common_gate)):
            raise RuntimeError('Residual gate differs from fixed base policy')
        base=F.linear((a.index_select(1,channels).float()*params['alpha']).to(torch.bfloat16),packed,b2)
        data['x'].append(x);data['base'].append(base);data['dense'].append(y)
        common_gate=gate
    data={k:torch.cat(v) for k,v in data.items()}
    data['gate']=common_gate
    data['label']=(data['dense'].float()-data['base'].float())*common_gate.float()
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--base-fits',type=Path,nargs=2,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol_path=ROOT/'config/residual_rr1.json';protocol=json.loads(protocol_path.read_text())
    records,bindings,capture_identity=load_training_reports(args.collections)
    cache_path=ROOT/'.runtime/hif4_rebuilt_cache.json';cache=json.loads(cache_path.read_text())
    if capture_identity['assets']['cache_manifest_sha256']!=digest(cache_path):
        raise RuntimeError('Residual calibration cache differs')
    base_models,base_bindings={},[]
    for path in args.base_fits:
        report=json.loads(path.read_text());root=Path(report['checkpoint_root'])
        identity=json.loads((root/'identity.json').read_text())
        model=CheckpointStore(root,identity['identity'])
        if report['status']!='COMPLETED_CALIBRATION_PARAMETER_FIT' or model.fingerprint!=report['fingerprint']:
            raise RuntimeError('Fixed base fit incomplete')
        base_models[report['expert']]=model;base_bindings.append({'path':str(path.resolve()),'sha256':digest(path),'fingerprint':model.fingerprint})
    store=CheckpointStore(args.out,{'protocol':protocol,'protocol_sha256':digest(protocol_path),'training_collections':bindings,
        'base_fits':base_bindings,'cache_manifest_sha256':digest(cache_path),'torch':torch.__version__,
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/reduced_rank_residual.py',
            ROOT/'tools/fit_cstatic_centered.py',ROOT/'tools/ar2_checkpoint.py']}})
    meta={'status':'RUNNING','fingerprint':store.fingerprint,'checkpoint_root':str(args.out.resolve()),'CV':[],
          'quality_or_speed_acceptance':False}
    started=time.perf_counter();atomic_json(args.report,meta)
    try:
        positions={}
        for step in [0,2]:
            expert='high_noise_model' if step==0 else 'low_noise_model'
            data=load_position(records,cache,base_models[expert],step);positions[step]=data
            for fold in [0,1]:
                validation=torch.tensor([i%2==fold for i in range(12)],device='cuda').repeat_interleave(512)
                train=~validation
                target=(data['dense'][validation]*data['gate']).double()
                base=(data['base'][validation]*data['gate']).double()
                base_sse=float((base-target).square().sum())
                if base_sse<=0:
                    raise RuntimeError('Degenerate residual CV baseline')
                for regularization in protocol['lambdas']:
                    name=f'step{step}_fold{fold}_lambda{str(regularization).replace(".","p")}'
                    part=store.load('CV_models',name)
                    if part is None:
                        fit=fit_rrr(data['x'][train],data['label'][train],regularization,max(protocol['ranks']))
                        parameters=native_parameters(fit,data['gate'],max(protocol['ranks']))
                        store.save('CV_models',name,encode(parameters),{'step':step,'fold':fold,'lambda':regularization,
                            'parameter_sha256':{k:tensor_sha(v) for k,v in parameters.items()}})
                        part=store.load('CV_models',name)
                    for rank in protocol['ranks']:
                        parameters=decode(part['arrays'],rank)
                        predicted=predict_y(data['x'][validation],parameters)
                        candidate=((data['base'][validation]+predicted)*data['gate']).double()
                        sse=float((candidate-target).square().sum())
                        meta['CV'].append({'step':step,'fold':fold,'rank':rank,'lambda':regularization,'native_SSE':sse,
                                           'base_native_SSE':base_sse,'ratio':sse/base_sse})
                    atomic_json(args.report,meta)
                    print(f'CV step{step} fold{fold} lambda{regularization}: rank512 ratio={meta["CV"][-1]["ratio"]:.6g}',flush=True)
        finalize(args,protocol,positions,store,meta)
        meta['elapsed_seconds']=time.perf_counter()-started
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(args.report,meta);atomic_json(args.out/'run_meta.json',meta)
    print(json.dumps(meta,indent=2))


def finalize(args,protocol,positions,store,meta):
    candidates=[]
    for rank in protocol['ranks']:
        for regularization in protocol['lambdas']:
            rows=[r for r in meta['CV'] if r['rank']==rank and r['lambda']==regularization]
            if len(rows)==4 and all(r['ratio']<=.75 for r in rows):
                candidates.append((rank,sum(r['ratio'] for r in rows)/4,regularization))
    if not candidates:
        meta.update(status='COMPLETED_RR1_NO_CV_CANDIDATE',selected=None);return
    rank,_,regularization=min(candidates)
    meta['selected']={'rank':rank,'lambda':regularization,'selection':'calibration CV only'}
    for step,data in positions.items():
        name=f'step{step}_block20'
        if store.load('final_models',name) is None:
            fit=fit_rrr(data['x'],data['label'],regularization,rank)
            parameters=native_parameters(fit,data['gate'],rank)
            store.save('final_models',name,encode(parameters),{'step':step,'rank':rank,'lambda':regularization,
                'parameter_sha256':{k:tensor_sha(v) for k,v in parameters.items()}})
    meta['status']='COMPLETED_RR1_CV_AND_SELECTED_FINAL_FIT'


if __name__=='__main__':
    main()
