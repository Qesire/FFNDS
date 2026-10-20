"""Evaluate only a passing fixed JR1 pilot on the two reused research sources."""
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

from anchored_reconstruction import from_bf16_bits
from ar2_checkpoint import CheckpointStore,atomic_json,digest
from evaluate_residual_rr1 import open_verified
from reduced_rank_residual import predict_y
from run_ar2_restart import ROOT,key_name,load_scopes,tensor_sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrices',type=Path,nargs=2,required=True)
    ap.add_argument('--base-fits',type=Path,nargs=2,required=True)
    ap.add_argument('--pilot',type=Path,required=True)
    ap.add_argument('--RR1-fit',dest='rr1_fit',type=Path,required=True)
    ap.add_argument('--rr1-parent',type=Path,required=True)
    ap.add_argument('--ar2-parent',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    pilot,pilot_report,pilot_id=open_verified(args.pilot,'COMPLETED_JR1_CV_AND_FIXED_PILOT_FIT')
    if not pilot_report['CV_gate_pass']:raise RuntimeError('JR1 CV must pass before any held-out evaluation')
    bases={};base_bindings=[]
    for path in args.base_fits:
        model,report,identity=open_verified(path,'COMPLETED_CALIBRATION_PARAMETER_FIT')
        bases[report['expert']]=model;base_bindings.append({'path':str(path.resolve()),'sha256':digest(path),'fingerprint':model.fingerprint})
    residual,rr_report,rr_id=open_verified(args.rr1_fit,'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    if pilot_id['base_fits']!=base_bindings or rr_id['base_fits']!=base_bindings:
        raise RuntimeError('JR1 and RR1 use different frozen C-centered bases')
    parent_meta=json.loads((args.rr1_parent/'run_meta.json').read_text());ar_meta=json.loads((args.ar2_parent/'run_meta.json').read_text())
    if (parent_meta['status']!='COMPLETED_RR1_INDEPENDENT_HELDOUT_DIAGNOSTIC'
            or digest(args.rr1_parent/'scope_summary.csv')!=parent_meta['final_csv_sha256']
            or ar_meta['status']!='COMPLETED_LOCAL_DIAGNOSTIC'
            or digest(args.ar2_parent/'scope_policy_summary.csv')!=ar_meta['final_csv_sha256']):
        raise RuntimeError('Frozen parent native risks/normalization changed')
    parents=list(csv.DictReader((args.rr1_parent/'scope_summary.csv').open()))
    denominators=list(csv.DictReader((args.ar2_parent/'scope_policy_summary.csv').open()))
    scopes,inputs=load_scopes(args.matrices)
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'JOB.lock').open('w');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    store=CheckpointStore(args.out,{'pilot_report_sha256':digest(args.pilot),'pilot_fingerprint':pilot.fingerprint,
        'base_fits':base_bindings,'RR1_fingerprint':residual.fingerprint,'inputs':inputs,'torch':torch.__version__,'threads':8,
        'RR1_parent_fingerprint':parent_meta['fingerprint'],'ar2_parent_fingerprint':ar_meta['fingerprint'],
        'sources':{str(p):digest(p) for p in [Path(__file__),ROOT/'tools/reduced_rank_residual.py',ROOT/'tools/anchored_reconstruction.py']}})
    meta={'status':'RUNNING','fingerprint':store.fingerprint,'pilot_fingerprint':pilot.fingerprint,
        'research_sources_reused':True,'test_used_for_fit_or_selection':False,'quality_or_speed_acceptance':False}
    atomic_json(args.out/'RUN.json',meta);started=time.perf_counter();summary=[]
    try:
        for key,data in scopes.items():
            scope=key_name(key);name=f'step{key.step}_block{key.block:02d}'
            base=bases[key.expert].load('positions',name);jr=pilot.load('final_models',name);rr=residual.load('positions',name)
            if any(v is None for v in [base,jr,rr]):raise RuntimeError('Required fixed pilot parameters missing')
            a,m=base['arrays'],base['metadata']
            if (tensor_sha(data['projection'].weight_t)!=m['weight_sha256'] or tensor_sha(data['gate'])!=m['gate_sha256']
                    or jr['metadata']['original_W2_sha256']!=m['weight_sha256'] or jr['metadata']['gate_sha256']!=m['gate_sha256']):
                raise RuntimeError('Quantized W2 or gate identity differs')
            part=store.load('risks',scope)
            if part is None:
                ids=torch.tensor((a['C_groups'][:,None]*64+np.arange(64)).reshape(-1))
                alpha=m['policies']['C']['modes']['centered_gain_bias']['alpha']
                if jr['metadata']['alpha']!=alpha:raise RuntimeError('Online gain/cast differs from fixed RR1')
                features=(data['a'][:,ids].float()*alpha).to(torch.bfloat16)
                with safe_open(data['scope']['tensor_file'],framework='pt',device='cpu') as handle:x=handle.get_tensor('V1')
                old={k:torch.from_numpy(v).to(torch.bfloat16) for k,v in rr['arrays'].items()}
                if {k:tensor_sha(v) for k,v in old.items()}!=rr['metadata']['parameter_sha256']:
                    raise RuntimeError('Frozen RR1 parameter encoding differs')
                base_y=F.linear(features,data['projection'].weight_t[:,ids].contiguous(),
                    torch.from_numpy(a['C_centered_gain_bias_b2']).to(torch.bfloat16))
                before=(base_y+predict_y(x,old))*data['gate']
                q=from_bf16_bits(jr['arrays']['weight']);b=from_bf16_bits(jr['arrays']['bias'])
                new={k[3:]:from_bf16_bits(v) for k,v in jr['arrays'].items() if k.startswith('rr_')}
                if (tensor_sha(q)!=jr['metadata']['weight_sha256'] or tensor_sha(b)!=jr['metadata']['bias_sha256']
                        or {k:tensor_sha(v) for k,v in new.items()}!=jr['metadata']['residual_parameter_sha256']):
                    raise RuntimeError('JR1 native parameter encoding differs')
                native=(F.linear(features,q,b)+predict_y(x,new))*data['gate']
                control=(F.linear(features.float(),q.float(),b.float())+predict_y(x,new,fp32=True))*data['gate'].float()
                target=data['target'].double()
                values={n+'_SSE':(v.double()-target).square().sum(1).numpy() for n,v in
                    [('native',native),('RR1_native',before),('FP32',control)]};values['weights']=data['weights']
                if any(not np.isfinite(v).all() for v in values.values()):raise RuntimeError('Nonfinite JR1 native risk')
                store.save('risks',scope,values,{'weight_sha256':tensor_sha(q)});part=store.load('risks',scope)
            values=part['arrays'];den=float(next(r for r in denominators if r['scope']==scope)['denominator_T'])
            risks={k:float(values['weights']@v/den) for k,v in values.items() if k.endswith('_SSE')}
            parent=float(next(r for r in parents if r['scope']==scope)['native_SSE'])
            if not np.isclose(risks['RR1_native_SSE'],parent,rtol=1e-7,atol=1e-10):
                raise RuntimeError('Native RR1 parent risk did not reproduce')
            ratio=risks['native_SSE']/parent
            summary.append({'scope':scope,**asdict(key),**risks,'JR1_over_RR1':ratio,'pass':ratio<=.9})
            print(f'EVALUATED JR1 {scope}: JR1/RR1={ratio:.7g}',flush=True)
        gates={}
        for task in [0,6]:
            passed=[r for r in summary if r['task']==task and r['pass']]
            gates[str(task)]={'passed_scopes':len(passed),'gate_pass':len(passed)>=3 and
                {r['seed'] for r in passed}=={20261001,20261002} and {r['step'] for r in passed}=={0,2}}
        if torch.cuda.is_initialized():raise RuntimeError('CPU JR1 diagnostic initialized CUDA')
        csv_path=args.out/'scope_summary.csv'
        with csv_path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        meta.update(status='COMPLETED_JR1_REUSED_RESEARCH_DIAGNOSTIC',source_gates=gates,
            route_pass=all(v['gate_pass'] for v in gates.values()),scope_summary_sha256=digest(csv_path),
            elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error));raise
    finally:atomic_json(args.out/'RUN.json',meta)


if __name__=='__main__':main()
