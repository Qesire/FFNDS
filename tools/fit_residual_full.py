"""Fit all160 RR1 positions with the already frozen calibration-only rank/lambda."""
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

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from evaluate_residual_rr1 import open_verified
from fit_cstatic_centered import load_training_reports
from reduced_rank_residual import fit_rrr, native_parameters, predict_y
from run_ar2_restart import ROOT, tensor_sha
from run_residual_rr1 import encode


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--base-fits',type=Path,nargs=2,required=True)
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--heldout',type=Path)
    ap.add_argument('--calibration24-fixed',action='store_true',help='New registered24x256 training data; keep original selected rank/lambda')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    selection,selection_report,selection_id=open_verified(args.selection,'COMPLETED_RR1_CV_AND_SELECTED_FINAL_FIT')
    protocol_path=ROOT/('config/residual_cal24_fixed.json' if args.calibration24_fixed else 'config/residual_rr1.json')
    protocol=json.loads(protocol_path.read_text())
    if selection_id['protocol_sha256']!=digest(ROOT/'config/residual_rr1.json'):
        raise RuntimeError('RR1 protocol changed')
    if args.calibration24_fixed:
        args.input_count,args.rows=24,256
        if (digest(args.selection)!=protocol['selection_report_sha256'] or (protocol['rank'],protocol['lambda'])!=(256,1.)
                or selection_report['selected']['rank']!=protocol['rank']
                or selection_report['selected']['lambda']!=protocol['lambda'] or shutil.disk_usage(args.out).free<82*2**30):
            raise RuntimeError('Frozen hyperparameters changed or node disk reserve insufficient')
    else:
        args.input_count,args.rows=12,512
        if args.heldout is None:ap.error('Original12 route requires its independent heldout report')
        heldout=json.loads(args.heldout.read_text())
        heldout_id=json.loads((args.heldout.parent/'identity.json').read_text())
        if (heldout['status']!='COMPLETED_RR1_INDEPENDENT_HELDOUT_DIAGNOSTIC' or not heldout['route_pass']
                or heldout_id['identity']['correction_fingerprint']!=selection.fingerprint
                or heldout_id['identity']['correction_report_sha256']!=digest(args.selection)
                or digest(args.heldout.parent/'scope_summary.csv')!=heldout['final_csv_sha256']):
            raise RuntimeError('Frozen independent source gate must pass before all-position fit')
    records,bindings,capture_identity=load_training_reports(args.collections,expected_inputs=args.input_count,expected_rows=args.rows)
    if not args.calibration24_fixed and bindings!=selection_id['training_collections']:
        raise RuntimeError('Calibration identity differs from rank selection')
    if args.calibration24_fixed and capture_identity['protocol_sha256']!=digest(ROOT/'config/calibration24_ffn_operands.json'):
        raise RuntimeError('New calibration capture protocol differs')
    base_models,base_bindings={},[]
    for path in args.base_fits:
        model,report,identity=open_verified(path,'COMPLETED_CALIBRATION_PARAMETER_FIT')
        base_models[report['expert']]=model
        base_bindings.append({'path':str(path.resolve()),'sha256':digest(path),'fingerprint':model.fingerprint})
    if not args.calibration24_fixed and base_bindings!=selection_id['base_fits']:
        raise RuntimeError('Frozen base policy changed')
    if args.calibration24_fixed:
        for path in args.base_fits:
            root=Path(json.loads(path.read_text())['checkpoint_root'])
            identity=json.loads((root/'identity.json').read_text())['identity']
            if identity['training_collections']!=bindings or identity['protocol_sha256']!=digest(ROOT/'config/cstatic_centered_cal24_fit.json'):
                raise RuntimeError('Base was not trained on the same frozen24-source data')
    cache_path=ROOT/'.runtime/hif4_rebuilt_cache.json';cache=json.loads(cache_path.read_text())
    if capture_identity['assets']['cache_manifest_sha256']!=digest(cache_path):
        raise RuntimeError('Cache differs from calibration data')
    rank,regularization=[selection_report['selected'][k] for k in ['rank','lambda']]
    frozen_selection=dict(selection_report['selected'])
    if args.calibration24_fixed:
        frozen_selection['selection']='fixed from original12 RR1 CV; no cal24 or test hyperparameter selection'
    store=CheckpointStore(args.out,{'protocol_sha256':digest(protocol_path),'selection_fingerprint':selection.fingerprint,
        'selection_report_sha256':digest(args.selection),'heldout_gate_sha256':digest(args.heldout) if args.heldout else None,
        'selected':frozen_selection,'training_collections':bindings,'base_fits':base_bindings,
        'training_geometry':{'inputs':args.input_count,'rows_per_input':args.rows},
        'cache_manifest_sha256':digest(cache_path),'torch':torch.__version__,'TF32':False,
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/reduced_rank_residual.py',
            ROOT/'tools/run_residual_rr1.py',ROOT/'tools/evaluate_residual_rr1.py']}})
    meta={'status':'RUNNING','fingerprint':store.fingerprint,'checkpoint_root':str(args.out.resolve()),
        'selected':frozen_selection,'positions':[],'quality_or_speed_acceptance':False}
    started=time.perf_counter();atomic_json(args.report,meta)
    try:
        for expert,steps in [('high_noise_model',[0,1]),('low_noise_model',[2,3])]:
            for block in range(40):
                shard=next(s for s in cache['shards'] if s['expert']==expert and s['block']==block)
                if digest(shard['path'])!=shard['sha256']:
                    raise RuntimeError('Effective weight shard changed')
                payload=torch.load(shard['path'],map_location='cpu',weights_only=True,mmap=True)
                w=payload[expert][f'blocks.{block}.ffn.2']['weight_qdq'].cuda()
                for step in steps:
                    name=f'step{step}_block{block:02d}'
                    previous=store.load('positions',name)
                    if previous is not None:
                        meta['positions'].append(previous['metadata']);continue
                    part=base_models[expert].load('positions',name)
                    if part is None:
                        raise RuntimeError('Missing base policy position')
                    arrays,info=part['arrays'],part['metadata']
                    if tensor_sha(w)!=info['weight_sha256']:
                        raise RuntimeError('Base W2 differs from captured cache')
                    channels=torch.tensor((arrays['C_groups'][:,None]*64+np.arange(64)).reshape(-1),device='cuda')
                    packed=w.index_select(1,channels).contiguous()
                    base_parameters=info['policies']['C']['modes']['centered_gain_bias']
                    b2=torch.from_numpy(arrays['C_centered_gain_bias_b2']).cuda().to(torch.bfloat16)
                    if tensor_sha(b2)!=base_parameters['b2_sha256']:
                        raise RuntimeError('Base folded bias changed')
                    xs,ys,bases=[],[],[]
                    for index in range(args.input_count):
                        scope=next(s for s in records[index]['scopes'] if s['step']==step and s['block']==block)
                        if scope['expert']!=expert or scope['rows']!=args.rows or digest(scope['path'])!=scope['sha256']:
                            raise RuntimeError('Training operands changed')
                        with safe_open(scope['path'],framework='pt',device='cpu') as handle:
                            x,a,y,gate=[handle.get_tensor(k).cuda() for k in ['V1','V2','Y','gate']]
                        gate=gate.reshape(1,5120)
                        if tensor_sha(gate)!=info['gate_sha256']:
                            raise RuntimeError('Gate differs from fixed base')
                        base=F.linear((a.index_select(1,channels).float()*base_parameters['alpha']).to(torch.bfloat16),packed,b2)
                        xs.append(x);ys.append(y);bases.append(base)
                    x,y,base=[torch.cat(v) for v in [xs,ys,bases]]
                    label=(y.float()-base.float())*gate.float()
                    # Reuse precisely the selected all12 model at the two pilot
                    # positions. All other positions use the same fixed algorithm.
                    pilot=selection.load('final_models',name) if not args.calibration24_fixed and block==20 and step in [0,2] else None
                    if pilot is None:
                        fit=fit_rrr(x,label,regularization,rank)
                        parameters=native_parameters(fit,gate,rank)
                    else:
                        parameters={k:torch.from_numpy(v).cuda().to(torch.bfloat16) for k,v in pilot['arrays'].items()}
                    target=(y*gate).double()
                    base_sse=float(((base*gate).double()-target).square().sum())
                    candidate=(base+predict_y(x,parameters))*gate
                    sse=float((candidate.double()-target).square().sum())
                    position={'expert':expert,'step':step,'block':block,'rank':rank,'lambda':regularization,
                        'gate_sha256':info['gate_sha256'],'weight_sha256':info['weight_sha256'],
                        'parameter_sha256':{k:tensor_sha(v) for k,v in parameters.items()},
                        'base_native_training_SSE':base_sse,'native_training_SSE':sse,
                        'native_training_ratio':sse/base_sse,'reused_frozen_pilot':pilot is not None}
                    store.save('positions',name,encode(parameters),position)
                    meta['positions'].append(position);atomic_json(args.report,meta)
                    print(f'FITTED {expert} {name}: corrected/base training={sse/base_sse:.6g}',flush=True)
                    del xs,ys,bases,x,y,base,label,parameters,candidate,target
                del payload,w
        if len(meta['positions'])!=160:
            raise RuntimeError('Full160 positions required')
        meta.update(status='COMPLETED_RR1_FULL160_CALIBRATION_FIT',elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(args.report,meta);atomic_json(args.out/'run_meta.json',meta)
    print(json.dumps({k:v for k,v in meta.items() if k!='positions'},indent=2))


if __name__=='__main__':
    main()
