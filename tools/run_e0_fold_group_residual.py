"""Cross-source fixed folded update plus original64-group nonlinear correction.

Saved-operand mechanism discrimination only; no matrix is composed or deployed.
"""
import argparse
import json
from pathlib import Path
import shutil

from analyze_ffnds_observations import load_arrays
from analyze_e0_temporal_calibration import fit_affine
from e0_intrinsic_math import energy,group_values,top_support
from ffnds_observation_common import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]


def execute(a):
    import torch
    from startup_gpu_reservation import reserve_before_assets
    from worker_identity import require_worker
    require_worker();p=json.loads(a.protocol.read_text())
    if not a.execute_gpu or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('Registered A100 required')
    for name,h in p['bound_sources'].items():
        if sha256(ROOT/name)!=h:raise ValueError('Frozen source changed')
    a.out.mkdir(parents=True,exist_ok=False);snap=a.out/'EXECUTED_SOURCES';snap.mkdir()
    for name in p['bound_sources']:
        target=snap/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(a.protocol,snap/a.protocol.name)
    r={'status':'STARTED_NOT_ACCEPTED','protocol_sha256':sha256(a.protocol),'cases':[],
       'diagnostic_projections':{'W1':0,'full_W2':0,'batched_group_W2':0},
       'new_generations':0,'native_QDQ_calls':0,'weight_compositions':0,'training_updates':0,
       'quality_or_speed_acceptance':False,'automatic_retry':False};monitor=None
    write_json(a.out/'RUN.json',r)
    try:
        reservation,monitor,receipt=reserve_before_assets(torch);r['CUDA_reservation']=receipt
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        dev=lambda x:torch.as_tensor(x).to('cuda',dtype=torch.float64)
        def charge(name):
            r['diagnostic_projections'][name]+=1
            if r['diagnostic_projections'][name]>p['projection_limits'][name]:raise RuntimeError('Budget exceeded')
        def values(x,w,g):charge('batched_group_W2');return group_values(x,w,g)
        accepted=json.loads((ROOT/'research/20261025/E0_ALL_BLOCK_TANGENT_OBSERVATIONS.json').read_text())
        cache=json.loads((ROOT/'.runtime/hif4_rebuilt_cache.json').read_text())
        def load(index,step,block):
            path=a.raw_root/f'native_source{index:02d}_attempt1/RUN.json'
            if sha256(path)!=next(v for v in accepted['raw_bindings'] if v.get('source_index')==index)['sha256']:
                raise ValueError('Native accepted run changed')
            d=json.loads(path.read_text());s=next(v for v in d['scopes'] if (v['step'],v['block'])==(step,block))
            path=path.parent/'observed'/s['path']
            if sha256(path)!=s['sha256']:raise ValueError('Operand changed')
            arrays=load_arrays(path);rows=s['row_plan']['contiguous_tiles'][0]+s['row_plan']['uniform_rows']
            ids=[s['row_plan']['rows'].index(row) for row in rows]
            return {k:dev(v[ids] if k in ['q1_post','q2_post','ffn_output'] else v)
                    for k,v in arrays.items() if k in ['q1_post','q2_post','ffn_output','gate','full_M_old_derivative_mean']}
        with torch.inference_mode():
            for block in p['blocks']:
                for expert,step in [('high_noise_model',1),('low_noise_model',3)]:
                    shard=next(s for s in cache['shards'] if s['expert']==expert and s['block']==block)
                    if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('Weight changed')
                    rec=torch.load(shard['path'],map_location='cpu',weights_only=False)[expert]
                    w1=rec[f'blocks.{block}.ffn.0']['weight_qdq'].to('cuda',dtype=torch.bfloat16).double()
                    w2=rec[f'blocks.{block}.ffn.2']['weight_qdq'].to('cuda',dtype=torch.bfloat16).double()
                    smooth=dev(rec[f'blocks.{block}.ffn.2']['smooth_scale']).reshape(1,-1)
                    data={i:(load(i,step,block),load(i,step-1,block)) for i in p['source_indices']}
                    cases={}
                    for index in p['source_indices']:
                        other=12 if index==0 else 0
                        c,old=data[index];train_old=data[other][1]
                        charge('W1');da=(c['q1_post']-old['q1_post'])@w1.T
                        dq=c['q2_post']-old['q2_post'];gate=c['gate']
                        original_values=values(dq,w2,gate)
                        # Calibration labels and kappa are from this source only;
                        # its score will be used exclusively by the OTHER source.
                        own_residual=values(dq-da*old['full_M_old_derivative_mean']/smooth,w2,gate)
                        cal_score=own_residual[128:].square().sum(dim=(0,2));del own_residual
                        fixed_coeff=da*train_old['full_M_old_derivative_mean']/smooth
                        cross_residual=values(dq-fixed_coeff,w2,gate)
                        charge('full_W2');linear=fixed_coeff@w2.T*gate
                        native_delta=(c['ffn_output']-old['ffn_output'])*gate
                        target=c['ffn_output']*gate
                        if energy(cross_residual.sum(1)+linear-original_values.sum(1))>1e-20*energy(original_values.sum(1)):
                            raise RuntimeError('Folded base plus signed group residual identity failed')
                        cases[index]={'cal_score':cal_score,'raw_score':original_values[128:].square().sum(dim=(0,2)),
                            'original_values':original_values,'residual_values':cross_residual,'linear':linear,
                            'native_delta':native_delta,'native_current':target,'gate':gate,
                            'prior_y':old['ffn_output'],'current_y':c['ffn_output'],
                            'continuous_native_delta_SSE':energy(original_values.sum(1)-native_delta)/energy(native_delta)}
                    for index,c in cases.items():
                        other=12 if index==0 else 0;train=cases[other]
                        row={'source_index':index,'calibration_source':other,'kappa_calibration_source':other,
                             'step':step,'block':block,'results':{},
                             'continuous_native_delta_SSE':c['continuous_native_delta_SSE']}
                        def describe(prediction):
                            error=prediction-c['native_delta']
                            return {name:{'relative_current_output_SSE':energy(error[sl])/energy(c['native_current'][sl]),
                                          'relative_innovation_SSE':energy(error[sl])/energy(c['native_delta'][sl])}
                                    for name,sl in [('contiguous128',slice(0,128)),('uniform32',slice(128,160))]}
                        row['results']['fixed_linear_only']=describe(c['linear'])
                        for count in p['group_counts']:
                            for choice in ['source_fixed','posthoc']:
                                score=(train['cal_score'] if choice=='source_fixed' else
                                       c['residual_values'][128:].square().sum(dim=(0,2)))
                                mask=top_support(score,count)
                                prediction=c['linear']+c['residual_values'][:,mask,:].sum(1)
                                row['results'][f'linear_plus_{count}_{choice}']=describe(prediction)
                                raw_score=train['raw_score'] if choice=='source_fixed' else c['raw_score']
                                raw_mask=top_support(raw_score,count)
                                row['results'][f'group_only_{count}_{choice}']=describe(c['original_values'][:,raw_mask,:].sum(1))
                        # Cheap other-source affine output control, fit on only
                        # that source's original random32 native rows.
                        tc={'previous':train['prior_y'][128:].cpu().numpy(),
                            'current':train['current_y'][128:].cpu().numpy(),'gate':train['gate'].cpu().numpy()}
                        alpha,beta=fit_affine([tc])
                        prediction=(alpha*c['prior_y']+dev(beta)-c['prior_y'])*c['gate']
                        row['results']['other_source_affine']=describe(prediction)
                        r['cases'].append(row);monitor.check()
                        print(f'fixed folded/group residual source{index:02d} step{step} block{block:02d}',flush=True)
                    del data,cases,w1,w2,rec
                    write_json(a.out/'RUN.json',r)
        if r['diagnostic_projections']!=p['projection_limits']:raise RuntimeError('Incomplete diagnostics')
        r['status']='PASS_CROSS_SOURCE_FOLD_GROUP_RESIDUAL'
    except BaseException as e:
        r.update(status='FAILED_PRESERVED',error_type=type(e).__name__,error=str(e));raise
    finally:
        if monitor is not None:monitor.close();r['assigned_GPU_resources']=monitor.report()
        write_json(a.out/'RUN.json',r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','raw-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute-gpu',action='store_true');execute(p.parse_args())
