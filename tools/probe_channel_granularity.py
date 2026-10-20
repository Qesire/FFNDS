"""Compare calibration-frozen individual channels with whole-group C, post-Q2 only."""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import CheckpointStore,atomic_json,digest
from ar2_core import top_groups
from centered_signal_fit import fit_signal_moments
from fit_cstatic_centered import load_training_reports
from run_ar2_restart import ROOT,key_name,load_scopes,tensor_sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs=2,required=True)
    ap.add_argument('--matrices',type=Path,nargs=2,required=True)
    ap.add_argument('--base-fits',type=Path,nargs=2,required=True)
    ap.add_argument('--base-heldout',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    args.out.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    records,bindings,capture_id=load_training_reports(args.collections)
    bases,base_bindings={},[]
    for path in args.base_fits:
        report=json.loads(path.read_text());root=Path(report['checkpoint_root'])
        identity=json.loads((root/'identity.json').read_text());model=CheckpointStore(root,identity['identity'])
        if report['status']!='COMPLETED_CALIBRATION_PARAMETER_FIT' or model.fingerprint!=report['fingerprint'] or identity['identity']['training_collections']!=bindings:
            raise RuntimeError('Base does not match complete calibration12 fit')
        bases[report['expert']]=model;base_bindings.append({'path':str(path),'sha256':digest(path),'fingerprint':model.fingerprint})
    parent=json.loads(args.base_heldout.read_text())
    if not parent['fixed_candidate_pass']:
        raise RuntimeError('Expected completed centered-base held-out control')
    scopes,inputs=load_scopes(args.matrices)
    protocol=ROOT/'config/channel_granularity_cg1.json'
    store=CheckpointStore(args.out,{'protocol_sha256':digest(protocol),'training':bindings,'base_fits':base_bindings,
        'heldout_inputs':inputs,'base_heldout_sha256':digest(args.base_heldout),'torch':torch.__version__,
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/centered_signal_fit.py',ROOT/'tools/ar2_core.py']}})
    meta={'status':'RUNNING','fingerprint':store.fingerprint,'checkpoint_root':str(args.out.resolve()),
          'post_Q2_diagnostic_only':True,'quality_or_speed_acceptance':False}
    atomic_json(args.report,meta);started=time.perf_counter()
    try:
        for step,expert in [(0,'high_noise_model'),(2,'low_noise_model')]:
            name=f'step{step}_block20'
            if store.load('parameters',name) is not None:continue
            reference=next(data for key,data in scopes.items() if key.step==step)
            w=reference['projection'].weight_t;gate=reference['gate'];original_bias=reference['projection'].bias
            base=bases[expert].load('positions',name)
            if (tensor_sha(w)!=base['metadata']['weight_sha256'] or tensor_sha(gate)!=base['metadata']['gate_sha256']
                    or tensor_sha(original_bias)!=base['metadata']['original_b2_sha256']):
                raise RuntimeError('Base effective operands differ')
            aa=[]
            for index in range(12):
                scope=next(s for s in records[index]['scopes'] if s['step']==step and s['block']==20)
                if digest(scope['path'])!=scope['sha256']:
                    raise RuntimeError('Calibration raw tensors changed')
                with safe_open(scope['path'],framework='pt',device='cpu') as handle:
                    a,g,b=[handle.get_tensor(k) for k in ['V2','gate','b2']]
                if tensor_sha(g)!=base['metadata']['gate_sha256'] or tensor_sha(b)!=base['metadata']['original_b2_sha256']:
                    raise RuntimeError('Calibration gate or bias changed')
                aa.append(a)
            a=torch.cat(aa)
            e=a.double().square().mean(0)*((w.double()*gate.reshape(-1,1).double()).square().sum(0))
            ids=np.sort(top_groups(e.numpy(),6912))
            channels=torch.from_numpy(ids);wf=w.float()
            signal=F.linear(a[:,channels].float(),wf[:,channels].contiguous())
            dense=F.linear(a.float(),wf)
            sg=(signal*gate.float()).double();yg=(dense*gate.float()).double()
            fit=fit_signal_moments(len(a),signal.double().sum(0).numpy(),dense.double().sum(0).numpy(),
                float(sg.square().sum()),float((sg*yg).sum()),gate.float().numpy())
            if fit['centered_alpha'] is None:
                raise RuntimeError('Degenerate calibration channel policy')
            alpha=float(np.float32(fit['centered_alpha']))
            b2=(original_bias.double()+torch.from_numpy(fit['mean_dense']-alpha*fit['mean_signal'])).to(torch.bfloat16)
            group_ids=base['arrays']['C_groups']
            group_channels=(group_ids[:,None]*64+np.arange(64)).reshape(-1)
            info={'step':step,'expert':expert,'block':20,'alpha':alpha,'b2_sha256':tensor_sha(b2),
                'E_retained_individual':float(e[channels].sum()/e.sum()),
                'E_retained_whole_C_groups':float(e[group_channels].sum()/e.sum()),
                'weight_sha256':tensor_sha(w),'gate_sha256':tensor_sha(gate),'original_b2_sha256':tensor_sha(original_bias)}
            store.save('parameters',name,{'channels':ids,'b2':b2.float().numpy()},info)
            print(f'FITTED {name}: individual E retention={info["E_retained_individual"]:.6g}, wholeC={info["E_retained_whole_C_groups"]:.6g}',flush=True)
        decisions=[]
        for key,data in scopes.items():
            scope=key_name(key);part=store.load('parameters',f'step{key.step}_block20')
            values=store.load('risks',scope)
            if values is None:
                ids=torch.from_numpy(part['arrays']['channels']);bias=torch.from_numpy(part['arrays']['b2']).to(torch.bfloat16)
                w=data['projection'].weight_t[:,ids].contiguous()
                effective=data['a'][:,ids].float()*part['metadata']['alpha']
                native=F.linear(effective.to(torch.bfloat16),w,bias)*data['gate']
                fp32=F.linear(effective,w.float(),bias.float())*data['gate'].float()
                target=data['target'].double()
                arrays={'native_SSE':(native.double()-target).square().sum(1).numpy(),
                    'FP32_SSE':(fp32.double()-target).square().sum(1).numpy(),'weights':data['weights']}
                store.save('risks',scope,arrays,{'post_Q2_diagnostic_only':True});values=store.load('risks',scope)
            parent_row=next(r for r in parent['scopes'] if r['scope']==scope)
            # This ratio cancels the exact AR2 dense normalization and uniform
            # design weight. Reconstruct its denominator from the frozen base.
            base_part=bases[key.expert].load('positions',f'step{key.step}_block20')
            bp=base_part['metadata']['policies']['C']['modes']['centered_gain_bias'];ba=base_part['arrays']
            ids=torch.tensor((ba['C_groups'][:,None]*64+np.arange(64)).reshape(-1))
            base_y=F.linear((data['a'][:,ids].float()*bp['alpha']).to(torch.bfloat16),
                data['projection'].weight_t[:,ids].contiguous(),torch.from_numpy(ba['C_centered_gain_bias_b2']).to(torch.bfloat16))*data['gate']
            base_sse=float(data['weights']@((base_y.double()-data['target'].double()).square().sum(1).numpy()))
            arrays=values['arrays'];sse=float(arrays['weights']@arrays['native_SSE']);fp32=float(arrays['weights']@arrays['FP32_SSE'])
            ratio=sse/base_sse
            decisions.append({'scope':scope,**asdict(key),'ratio':ratio,'native_risk':ratio*parent_row['centered_risk'],
                'FP32_risk':fp32/base_sse*parent_row['centered_risk'],'base_native_risk':parent_row['centered_risk'],'pass':ratio<=.75})
            print(f'EVALUATED {scope}: channel/wholeC={ratio:.6g}',flush=True)
        gates={}
        for task in [0,6]:
            passed=[r for r in decisions if r['task']==task and r['pass']]
            gates[str(task)]={'passed_scopes':len(passed),'gate_pass':len(passed)>=3 and {r['seed'] for r in passed}=={20261001,20261002} and {r['step'] for r in passed}=={0,2}}
        if torch.cuda.is_initialized():raise RuntimeError('CPU diagnostic initialized CUDA')
        meta.update(status='COMPLETED_CG1_POSTQ2_DIAGNOSTIC',scopes=decisions,source_gates=gates,
            route_pass=all(r['gate_pass'] for r in gates.values()),elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:
        atomic_json(args.report,meta);atomic_json(args.out/'run_meta.json',meta)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    main()
