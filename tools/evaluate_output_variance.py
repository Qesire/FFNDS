"""Check the registered VM1 local-risk cost against the unchanged RR1 parent."""
import argparse
import csv
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
from fit_output_variance import verified
from output_variance import affine_reference
from reduced_rank_residual import predict_y
from run_ar2_restart import ROOT,key_name,load_scopes,tensor_sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrices',type=Path,nargs=2,required=True)
    ap.add_argument('--fit-report',type=Path,required=True)
    ap.add_argument('--base-policy',type=Path,required=True)
    ap.add_argument('--rr1-parent',type=Path,required=True)
    ap.add_argument('--ar2-parent',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    fit,fit_report,fit_id=verified(args.fit_report,'COMPLETED_VM1_PILOT_MOMENT_FIT')
    protocol=ROOT/'config/output_variance_vm1.json'
    if fit_id['base_policy_sha256']!=digest(args.base_policy) or fit_id['protocol_sha256']!=digest(protocol):
        raise RuntimeError('VM1 frozen base or protocol changed')
    policy=json.loads(args.base_policy.read_text());bases={}
    for expert,binding in policy['fit_reports'].items():
        if digest(binding['path'])!=binding['sha256']:raise RuntimeError('Frozen base report changed')
        bases[expert],_,_=verified(binding['path'],'COMPLETED_CALIBRATION_PARAMETER_FIT')
    residual_binding=policy['residual_fit']
    if digest(residual_binding['path'])!=residual_binding['sha256']:raise RuntimeError('Residual report changed')
    residual,_,_=verified(residual_binding['path'],'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    rr_meta=json.loads((args.rr1_parent/'run_meta.json').read_text())
    ar_meta=json.loads((args.ar2_parent/'run_meta.json').read_text())
    if (rr_meta['status']!='COMPLETED_RR1_INDEPENDENT_HELDOUT_DIAGNOSTIC'
            or digest(args.rr1_parent/'scope_summary.csv')!=rr_meta['final_csv_sha256']
            or ar_meta['status']!='COMPLETED_LOCAL_DIAGNOSTIC'
            or digest(args.ar2_parent/'scope_policy_summary.csv')!=ar_meta['final_csv_sha256']):
        raise RuntimeError('Frozen risk/normalization controls changed')
    rr_rows=list(csv.DictReader((args.rr1_parent/'scope_summary.csv').open()))
    ar_rows=list(csv.DictReader((args.ar2_parent/'scope_policy_summary.csv').open()))
    scopes,inputs=load_scopes(args.matrices)
    store=CheckpointStore(args.out,{'protocol_sha256':digest(protocol),'fit_fingerprint':fit.fingerprint,
        'base_policy_sha256':digest(args.base_policy),'rr_parent':rr_meta['fingerprint'],'ar_parent':ar_meta['fingerprint'],
        'inputs':inputs,'torch':torch.__version__,'threads':8,
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/output_variance.py',ROOT/'tools/reduced_rank_residual.py']}})
    meta={'status':'RUNNING','fingerprint':store.fingerprint,'fit_fingerprint':fit.fingerprint,
        'research_sources_reused':True,'test_targets_used_for_fit':False,'quality_or_speed_acceptance':False}
    atomic_json(args.out/'RUN.json',meta);started=time.perf_counter();summary=[]
    try:
        for key,data in scopes.items():
            name=f'step{key.step}_block{key.block:02d}';scope=key_name(key)
            vm=fit.load('positions',name);base=bases[key.expert].load('positions',name);rr=residual.load('positions',name)
            if any(v is None for v in [vm,base,rr]):raise RuntimeError('Pilot parameter missing')
            info=base['metadata'];arrays=base['arrays']
            if tensor_sha(data['projection'].weight_t)!=info['weight_sha256'] or tensor_sha(data['gate'])!=info['gate_sha256']:
                raise RuntimeError('Frozen native operands differ')
            original_bias=torch.from_numpy(vm['arrays']['original_b2']).to(torch.bfloat16)
            shift=torch.from_numpy(vm['arrays']['shift']);beta=float(vm['arrays']['beta'][0])
            if tensor_sha(original_bias)!=vm['metadata']['original_b2_sha256'] or tensor_sha(original_bias)!=tensor_sha(data['projection'].bias) or tensor_sha(shift)!=vm['metadata']['shift_FP32_sha256']:
                raise RuntimeError('Affine coefficient or original bias encoding changed')
            risk=store.load('risks',scope)
            if risk is None:
                ids=torch.tensor((arrays['C_groups'][:,None]*64+np.arange(64)).reshape(-1))
                b2=torch.from_numpy(arrays['C_centered_gain_bias_b2']).to(torch.bfloat16)
                parameters={k:torch.from_numpy(v).to(torch.bfloat16) for k,v in rr['arrays'].items()}
                if {k:tensor_sha(v) for k,v in parameters.items()}!=rr['metadata']['parameter_sha256']:
                    raise RuntimeError('Native RR1 encoding changed')
                with safe_open(data['scope']['tensor_file'],framework='pt',device='cpu') as handle:x=handle.get_tensor('V1')
                w=data['projection'].weight_t[:,ids].contiguous()
                effective=data['a'][:,ids].float()*info['policies']['C']['modes']['centered_gain_bias']['alpha']
                s=F.linear(effective.to(torch.bfloat16),w,b2)+predict_y(x,parameters)
                native=affine_reference(s,beta,shift,original_bias)*data['gate']
                before=s*data['gate']
                control_s=F.linear(effective,w.float(),b2.float())+predict_y(x,parameters,fp32=True)
                control=affine_reference(control_s,beta,shift,original_bias,fp32=True)*data['gate'].float()
                target=data['target'].double()
                values={f'{n}_SSE':(v.double()-target).square().sum(1).numpy() for n,v in
                    [('native',native),('RR1_native',before),('FP32',control)]}
                values['weights']=data['weights']
                if any(not np.isfinite(v).all() for v in values.values()):raise RuntimeError('Nonfinite VM1 risk')
                store.save('risks',scope,values,{'beta':beta});risk=store.load('risks',scope)
            values=risk['arrays'];den=float(next(r for r in ar_rows if r['scope']==scope)['denominator_T'])
            risks={k:float(values['weights']@v/den) for k,v in values.items() if k.endswith('_SSE')}
            parent=float(next(r for r in rr_rows if r['scope']==scope)['native_SSE'])
            if not np.isclose(risks['RR1_native_SSE'],parent,rtol=1e-7,atol=1e-10):
                raise RuntimeError('Native RR1 parent risk did not reproduce')
            ratio=risks['native_SSE']/parent
            summary.append({'scope':scope,**asdict(key),**risks,'beta':beta,'ratio_to_RR1':ratio,'pass':ratio<=1.10})
            print(f'EVALUATED VM1 {scope}: beta={beta:.7g}, risk/RR1={ratio:.6g}',flush=True)
        gates={}
        for task in [0,6]:
            passed=[r for r in summary if r['task']==task and r['pass']]
            gates[str(task)]={'passed_scopes':len(passed),'gate_pass':len(passed)>=3 and {r['seed'] for r in passed}=={20261001,20261002} and {r['step'] for r in passed}=={0,2}}
        if torch.cuda.is_initialized():raise RuntimeError('CPU VM1 evaluator initialized CUDA')
        meta.update(status='COMPLETED_VM1_PILOT_RISK_DIAGNOSTIC',source_gates=gates,route_pass=all(r['gate_pass'] for r in gates.values()),
            scopes=summary,elapsed_seconds=time.perf_counter()-started)
        with (args.out/'scope_summary.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        meta['scope_csv_sha256']=digest(args.out/'scope_summary.csv')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:atomic_json(args.out/'RUN.json',meta)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
