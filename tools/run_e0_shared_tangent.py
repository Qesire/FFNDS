"""Saved-data falsification of shared-slope FFN update folding; no model action."""
import argparse
import json
from pathlib import Path
import shutil

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256,write_json
from e0_intrinsic_math import energy
from e0_gelu_tangent import derivative,folding_mac_ratio

ROOT=Path(__file__).resolve().parents[1]


def execute(a):
    import torch
    from startup_gpu_reservation import reserve_before_assets
    from worker_identity import require_worker
    require_worker();p=json.loads(a.protocol.read_text())
    if not a.execute_gpu or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('Registered A100 required')
    for f,h in p['bound_sources'].items():
        if sha256(ROOT/f)!=h:raise ValueError('Frozen source changed')
    a.out.mkdir(parents=True,exist_ok=False);snap=a.out/'EXECUTED_SOURCES';snap.mkdir()
    for f in p['bound_sources']:
        dest=snap/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest)
    shutil.copyfile(a.protocol,snap/a.protocol.name)
    r={'status':'STARTED_NOT_ACCEPTED','protocol_sha256':sha256(a.protocol),'cases':[],
       'diagnostic_projections':{'W1':0,'W2':0},'new_generations':0,'native_QDQ_calls':0,
       'new_weight_compositions':0,'training_updates':0,'quality_or_speed_acceptance':False,'automatic_retry':False}
    write_json(a.out/'RUN.json',r);monitor=None
    try:
        reservation,monitor,receipt=reserve_before_assets(torch);r['CUDA_reservation']=receipt
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        dev=lambda x:torch.as_tensor(x).to('cuda',dtype=torch.float64)
        def proj(x,w,name):
            r['diagnostic_projections'][name]+=1
            if r['diagnostic_projections'][name]>p['projection_limits'][name]:raise RuntimeError('Budget exceeded')
            return x@w.T
        accepted=json.loads((ROOT/'research/20261025/E0_SHARED_TILE_PRE_W1.json').read_text())
        cache=json.loads((ROOT/'.runtime/hif4_rebuilt_cache.json').read_text())
        def load(index,step,block):
            path=a.raw_root/f'native_source{index:02d}_attempt1/RUN.json'
            binding=next(v for v in accepted['raw_bindings'] if v.get('source_index')==index)
            if sha256(path)!=binding['sha256']:raise ValueError('Native run changed')
            run=json.loads(path.read_text());s=next(v for v in run['scopes'] if (v['step'],v['block'])==(step,block))
            path=path.parent/'observed'/s['path']
            if sha256(path)!=s['sha256']:raise ValueError('Operand changed')
            values=load_arrays(path);ids=[s['row_plan']['rows'].index(row) for row in sum(s['row_plan']['contiguous_tiles'],[])]
            return {k:dev(v[ids] if k in ['q1_post','z','gelu','q2_post'] else v) for k,v in values.items() if k in ['q1_post','z','gelu','q2_post','gate']}
        with torch.inference_mode():
            for block in p['blocks']:
                for expert,step in [('high_noise_model',1),('low_noise_model',3)]:
                    shard=next(v for v in cache['shards'] if v['expert']==expert and v['block']==block)
                    if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('Weight changed')
                    rec=torch.load(shard['path'],map_location='cpu',weights_only=False)[expert]
                    w1=rec[f'blocks.{block}.ffn.0']['weight_qdq'].to('cuda',dtype=torch.bfloat16).double()
                    w2=rec[f'blocks.{block}.ffn.2']['weight_qdq'].to('cuda',dtype=torch.bfloat16).double()
                    smooth=dev(rec[f'blocks.{block}.ffn.2']['smooth_scale']).reshape(1,-1)
                    for index in p['source_indices']:
                        cur,old=load(index,step,block),load(index,step-1,block)
                        da=proj(cur['q1_post']-old['q1_post'],w1,'W1')
                        # Operationally valid old-only slope; current a/h is
                        # used solely by explicitly posthoc analytic controls.
                        previous_derivative=derivative(old['z'])
                        native_da=cur['z']-old['z'];dh=cur['gelu']-old['gelu']
                        numerator=(native_da*dh).sum(0,keepdim=True)
                        denominator=native_da.square().sum(0,keepdim=True)
                        oracle=numerator/denominator.clamp_min(1e-30)
                        models={
                            'old_per_token_derivative':previous_derivative,
                            'old_mean_derivative_shared384':previous_derivative.mean(0,keepdim=True),
                            'old_mean_derivative_shared128':previous_derivative.reshape(3,128,-1).mean(1).repeat_interleave(128,0),
                            'posthoc_shared384_least_squares':oracle,
                        }
                        dq=cur['q2_post']-old['q2_post'];gate=cur['gate'];target=proj(dq,w2,'W2')*gate
                        item={'source_index':index,'step':step,'block':block,'results':{},
                              'linear_preactivation_reconstruction_SSE':energy(da-native_da)/energy(native_da)}
                        for name,slope in models.items():
                            approximation=proj(da*slope/smooth,w2,'W2')*gate
                            error=energy(approximation-target)/energy(target)
                            item['results'][name]={'relative_innovation_SSE':error}
                        # Isolate the best same-coordinate derivative assumption
                        # from the original Q2 representation.
                        target_h=proj(dh/smooth,w2,'W2')*gate
                        native_tangent=proj(native_da*previous_derivative/smooth,w2,'W2')*gate
                        item['continuous_GELU_old_per_token_tangent_SSE']=energy(native_tangent-target_h)/energy(target_h)
                        item['Q2_rounding_residual_over_innovation']=energy(target-target_h)/energy(target)
                        r['cases'].append(item);monitor.check()
                        print(f'tangent source{index:02d} step{step} block{block:02d}',flush=True)
                    write_json(a.out/'RUN.json',r);del w1,w2,rec
        if r['diagnostic_projections']!=p['projection_limits']:raise RuntimeError('Incomplete diagnostics')
        r['hypothetical_composition_MAC_ratio']={str(g):folding_mac_ratio(g) for g in [1,3,9,10,248]}
        r['status']='PASS_SHARED_TANGENT_DISCRIMINATION'
    except BaseException as e:
        r.update(status='FAILED_PRESERVED',error_type=type(e).__name__,error=str(e));raise
    finally:
        if monitor is not None:monitor.close();r['assigned_GPU_resources']=monitor.report()
        write_json(a.out/'RUN.json',r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','raw-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute-gpu',action='store_true');execute(p.parse_args())
