"""Checkpoint native CPU output-moment fits without changing the frozen RR1 base."""
import argparse
import fcntl
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import CheckpointStore,atomic_json,digest
from fit_cstatic_centered import load_training_reports
from output_variance import affine_reference,fit_output_moments
from reduced_rank_residual import predict_y
from run_ar2_restart import ROOT,tensor_sha


def verified(path,status):
    report=json.loads(Path(path).read_text());root=Path(report['checkpoint_root'])
    identity=json.loads((root/'identity.json').read_text())['identity'];store=CheckpointStore(root,identity)
    if report['status']!=status or store.fingerprint!=report['fingerprint']:
        raise RuntimeError('Incomplete frozen base parameter fit')
    return store,report,identity


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--base-policy',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--stage',choices=['pilot','full'],default='pilot')
    ap.add_argument('--gate-report',type=Path)
    ap.add_argument('--max-seconds',type=float,default=1200)
    args=ap.parse_args()
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    args.out.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    records,bindings,capture=load_training_reports(args.collections)
    policy=json.loads(args.base_policy.read_text())
    if (policy['kind']!='CALIBRATION_STATIC_PACKED_REFERENCE' or policy['support']!='C'
            or policy['mode']!='centered_gain_bias' or 'residual_fit' not in policy):
        raise RuntimeError('VM1 requires the unchanged C-centered RR1 base')
    bases={};base_bindings=[]
    for expert in ['high_noise_model','low_noise_model']:
        binding=policy['fit_reports'][expert]
        if digest(binding['path'])!=binding['sha256']:raise RuntimeError('Frozen C-base report changed')
        model,report,identity=verified(binding['path'],'COMPLETED_CALIBRATION_PARAMETER_FIT')
        if report['expert']!=expert or identity['training_collections']!=bindings:
            raise RuntimeError('VM1 base calibration identity differs')
        bases[expert]=model;base_bindings.append(binding)
    binding=policy['residual_fit']
    if digest(binding['path'])!=binding['sha256']:raise RuntimeError('Frozen residual report changed')
    residual,residual_report,residual_id=verified(binding['path'],'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    if residual_id['training_collections']!=bindings or residual_report['selected']['rank']!=256 or residual_report['selected']['lambda']!=1:
        raise RuntimeError('VM1 must retain the original12 selected rank/lambda')
    cache_path=ROOT/'.runtime/hif4_rebuilt_cache.json';cache=json.loads(cache_path.read_text())
    if capture['assets']['cache_manifest_sha256']!=digest(cache_path) or policy['cache_manifest']['sha256']!=digest(cache_path):
        raise RuntimeError('Frozen base/capture quantized cache differs')
    protocol_path=ROOT/'config/output_variance_vm1.json'
    store=CheckpointStore(args.out,{'protocol_sha256':digest(protocol_path),'base_policy_sha256':digest(args.base_policy),
        'training_collections':bindings,'cache_manifest_sha256':digest(cache_path),'base_fits':base_bindings,
        'residual_fingerprint':residual.fingerprint,'torch':torch.__version__,'host':platform.node(),'threads':8,
        'cpu_capability':torch.backends.cpu.get_cpu_capability(),
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/output_variance.py',ROOT/'tools/reduced_rank_residual.py']}})
    if args.stage=='full':
        if args.gate_report is None:ap.error('Full fitting requires the fixed pilot local gate report')
        gate=json.loads(args.gate_report.read_text())
        if gate['status']!='COMPLETED_VM1_PILOT_RISK_DIAGNOSTIC' or not gate['route_pass'] or gate['fit_fingerprint']!=store.fingerprint:
            raise RuntimeError('VM1 pilot local risk gate must pass first')
    meta={'status':'RUNNING','stage':args.stage,'fingerprint':store.fingerprint,'checkpoint_root':str(args.out.resolve()),
        'positions':[],'quality_or_speed_acceptance':False}
    atomic_json(args.report,meta);started=time.perf_counter()
    requested=[(0,20),(2,20)] if args.stage=='pilot' else [(s,b) for s in range(4) for b in range(40)]
    weight_cache={}
    try:
        for step,block in requested:
            name=f'step{step}_block{block:02d}'
            previous=store.load('positions',name)
            if previous is None:
                expert='high_noise_model' if step<2 else 'low_noise_model'
                part=bases[expert].load('positions',name);correction=residual.load('positions',name)
                if part is None or correction is None:raise RuntimeError('Frozen RR1 position missing')
                arrays,info=part['arrays'],part['metadata']
                key=(expert,block)
                if key not in weight_cache:
                    shard=next(r for r in cache['shards'] if r['expert']==expert and r['block']==block)
                    if digest(shard['path'])!=shard['sha256']:raise RuntimeError('Effective W2 shard changed')
                    payload=torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)
                    w=payload[expert][f'blocks.{block}.ffn.2']['weight_qdq']
                    weight_cache={key:w}
                w=weight_cache[key]
                if tensor_sha(w)!=info['weight_sha256']:raise RuntimeError('Frozen base W2 differs')
                ids=torch.tensor((arrays['C_groups'][:,None]*64+np.arange(64)).reshape(-1))
                packed=w[:,ids].contiguous()
                base=info['policies']['C']['modes']['centered_gain_bias']
                bias=torch.from_numpy(arrays['C_centered_gain_bias_b2']).to(torch.bfloat16)
                parameters={k:torch.from_numpy(v).to(torch.bfloat16) for k,v in correction['arrays'].items()}
                if {k:tensor_sha(v) for k,v in parameters.items()}!=correction['metadata']['parameter_sha256']:
                    raise RuntimeError('Residual parameter encoding changed')
                dense_outputs=[];candidate_outputs=[]
                for index in range(12):
                    scope=next(r for r in records[index]['scopes'] if r['step']==step and r['block']==block)
                    if digest(scope['path'])!=scope['sha256']:raise RuntimeError('Original calibration operands changed')
                    with safe_open(scope['path'],framework='pt',device='cpu') as handle:
                        x,a,y,g,b=[handle.get_tensor(k) for k in ['V1','V2','Y','gate','b2']]
                    if tensor_sha(g)!=info['gate_sha256'] or tensor_sha(b)!=info['original_b2_sha256']:
                        raise RuntimeError('Original gate or bias changed')
                    base_y=F.linear((a[:,ids].float()*base['alpha']).to(torch.bfloat16),packed,bias)
                    candidate_outputs.append(base_y+predict_y(x,parameters));dense_outputs.append(y)
                dense,candidate=torch.cat(dense_outputs),torch.cat(candidate_outputs)
                original_bias=b.reshape(5120)
                moments=fit_output_moments(dense,candidate,g,original_bias)
                corrected=affine_reference(candidate,moments['beta'],moments['shift'],original_bias)
                target=(dense*g).double();before=float(((candidate*g).double()-target).square().sum())
                after=float(((corrected*g).double()-target).square().sum())
                centered=corrected.double()-original_bias.double();centered-=centered.mean(0)
                corrected_energy=float((centered*g.double()).square().sum()/len(corrected))
                values={'beta':np.asarray([moments['beta']],dtype=np.float32),'shift':moments['shift'].numpy(),
                    'original_b2':original_bias.float().numpy()}
                metadata={k:v for k,v in moments.items() if k!='shift'}
                metadata.update({'step':step,'block':block,'expert':expert,'original_b2_sha256':tensor_sha(original_bias),'gate_sha256':info['gate_sha256'],
                    'native_training_SSE_before':before,'native_training_SSE_after':after,'native_training_risk_ratio':after/before,
                    'corrected_centered_energy_ratio':corrected_energy/moments['dense_centered_energy'],
                    'shift_FP32_sha256':tensor_sha(moments['shift']),'training_rows':len(dense)})
                store.save('positions',name,values,metadata);previous=store.load('positions',name)
                if block==20 and step in [0,2] and store.load('probe_data',name) is None:
                    store.save('probe_data',name,{'candidate':candidate[:128].float().numpy()},
                        {'precision':'exact finite BF16 values encoded in FP32','source':'CPU native RR1 reconstruction on calibration input0 sampled rows, not formal data'})
                print(f'FITTED VM1 {name}: beta={moments["beta"]:.7g}, SSE ratio={after/before:.6g}',flush=True)
                del dense,candidate,corrected,dense_outputs,candidate_outputs
            meta['positions'].append(previous['metadata']);atomic_json(args.report,meta)
            if time.perf_counter()-started>args.max_seconds and len(meta['positions'])<len(requested):
                meta.update(status='YIELDED_RESUMABLE_CPU_RESOURCE_WINDOW',elapsed_seconds=time.perf_counter()-started)
                break
        else:
            meta.update(status='COMPLETED_VM1_PILOT_MOMENT_FIT' if args.stage=='pilot' else 'COMPLETED_VM1_FULL160_MOMENT_FIT',
                elapsed_seconds=time.perf_counter()-started)
        if torch.cuda.is_initialized():raise RuntimeError('CPU coefficient fitting initialized CUDA')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(args.report,meta);atomic_json(args.out/'run_meta.json',meta)
    print(json.dumps({k:v for k,v in meta.items() if k!='positions'},indent=2))


if __name__=='__main__':main()
