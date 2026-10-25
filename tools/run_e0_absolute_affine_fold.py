"""Source-fixed absolute FFN affine folding with original64-group corrections.

No temporal cache, composed matrix, candidate model or video is executed here.
"""
import argparse
import json
from pathlib import Path
import shutil

from analyze_ffnds_observations import load_arrays
from e0_affine_fold_math import fit_coefficients
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
       'new_generations':0,'native_QDQ_calls':0,'composed_matrices':0,'training_updates':0,
       'quality_or_speed_acceptance':False,'automatic_retry':False};monitor=None
    write_json(a.out/'RUN.json',r)
    try:
        reservation,monitor,receipt=reserve_before_assets(torch);r['CUDA_reservation']=receipt
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        dev=lambda x:torch.as_tensor(x).to('cuda',dtype=torch.float64)
        def charge(name):
            r['diagnostic_projections'][name]+=1
            if r['diagnostic_projections'][name]>p['projection_limits'][name]:raise RuntimeError('Budget exceeded')
        def groups(q,w,g):charge('batched_group_W2');return group_values(q,w,g)
        accepted=json.loads((ROOT/'research/20261025/E0_ALL_BLOCK_TANGENT_OBSERVATIONS.json').read_text())
        cache=json.loads((ROOT/'.runtime/hif4_rebuilt_cache.json').read_text())
        def load(index,step,block):
            path=a.raw_root/f'native_source{index:02d}_attempt1/RUN.json'
            if sha256(path)!=next(v for v in accepted['raw_bindings'] if v.get('source_index')==index)['sha256']:
                raise ValueError('Native run changed')
            run=json.loads(path.read_text());s=next(v for v in run['scopes'] if (v['step'],v['block'])==(step,block))
            path=path.parent/'observed'/s['path']
            if sha256(path)!=s['sha256']:raise ValueError('Native operand changed')
            values=load_arrays(path);rows=s['row_plan']['contiguous_tiles'][0]+s['row_plan']['uniform_rows']
            ids=[s['row_plan']['rows'].index(row) for row in rows]
            return {k:dev(v[ids] if k in ['q1_post','q2_post','gelu','ffn_output'] else v)
                    for k,v in values.items() if k in ['q1_post','q2_post','gelu','ffn_output','gate','b1','b2']}
        with torch.inference_mode():
            for block in p['blocks']:
                for expert,steps in [('high_noise_model',[0,1]),('low_noise_model',[2,3])]:
                    shard=next(s for s in cache['shards'] if s['expert']==expert and s['block']==block)
                    if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('Weight changed')
                    rec=torch.load(shard['path'],map_location='cpu',weights_only=False)[expert]
                    w1=rec[f'blocks.{block}.ffn.0']['weight_qdq'].to('cuda',dtype=torch.bfloat16).double()
                    w2=rec[f'blocks.{block}.ffn.2']['weight_qdq'].to('cuda',dtype=torch.bfloat16).double()
                    smooth=dev(rec[f'blocks.{block}.ffn.2']['smooth_scale']).reshape(1,-1)
                    for step in steps:
                        cases={}
                        for index in p['source_indices']:
                            c=load(index,step,block);charge('W1')
                            preact=c['q1_post']@w1.T+c['b1'].reshape(1,-1)
                            q=c['q2_post'];gate=c['gate'];models={};cal_scores={}
                            for name,labels in [('Q2_OLS',q),('GELU_OLS',c['gelu']/smooth)]:
                                slope,offset=fit_coefficients(preact[128:],labels[128:])
                                # Both calibrators preserve the training-source
                                # mean actual Q2, differing only in their slope.
                                offset=q[128:].mean(0,keepdim=True)-slope*preact[128:].mean(0,keepdim=True)
                                models[name]=(slope,offset)
                                gv=groups(q-preact*slope-offset,w2,gate)
                                cal_scores[name]=gv[128:].square().sum(dim=(0,2));del gv
                            raw=groups(q,w2,gate)
                            cases[index]={'preact':preact,'q':q,'gate':gate,'b2':c['b2'].reshape(1,-1),
                                'native_y':c['ffn_output']*gate,'models':models,'cal_scores':cal_scores,
                                'raw_values':raw,'raw_score':raw[128:].square().sum(dim=(0,2)),
                                'mean_y':c['ffn_output'][128:].mean(0,keepdim=True)}
                        for index,c in cases.items():
                            other=12 if index==0 else 0;train=cases[other]
                            row={'source_index':index,'calibration_source':other,'step':step,'block':block,'results':{}}
                            def describe(prediction):
                                error=prediction-c['native_y']
                                return {key:{'relative_current_output_SSE':energy(error[sl])/energy(c['native_y'][sl])}
                                        for key,sl in [('contiguous128',slice(0,128)),('uniform32',slice(128,160))]}
                            decoded=c['raw_values'].sum(1)+c['b2']*c['gate']
                            row['continuous_vs_native_output_SSE']=energy(decoded-c['native_y'])/energy(c['native_y'])
                            row['results']['other_source_constant_mean']=describe(train['mean_y']*c['gate'])
                            for name,(slope,offset) in train['models'].items():
                                coefficients=c['preact']*slope+offset
                                charge('full_W2');base=(coefficients@w2.T+c['b2'])*c['gate']
                                residual=groups(c['q']-coefficients,w2,c['gate'])
                                if energy(base+residual.sum(1)-decoded)>1e-20*energy(decoded):
                                    raise RuntimeError('Absolute affine plus signed group residual identity failed')
                                row['results'][name+'_linear_only']=describe(base)
                                for count in p['correction_group_counts']:
                                    for choice in ['source_fixed','posthoc']:
                                        score=(train['cal_scores'][name] if choice=='source_fixed' else
                                               residual[128:].square().sum(dim=(0,2)))
                                        mask=top_support(score,count)
                                        row['results'][f'{name}_plus{count}_{choice}']=describe(base+residual[:,mask,:].sum(1))
                                del residual
                            for count in p['matched_group_only_counts']:
                                for choice in ['source_fixed','posthoc']:
                                    score=train['raw_score'] if choice=='source_fixed' else c['raw_score']
                                    mask=top_support(score,count)
                                    row['results'][f'group_only{count}_{choice}']=describe(c['raw_values'][:,mask,:].sum(1)+c['b2']*c['gate'])
                            r['cases'].append(row);monitor.check()
                            print(f'absolute affine/group source{index:02d} step{step} block{block:02d}',flush=True)
                        del cases
                        write_json(a.out/'RUN.json',r)
                    del w1,w2,rec
        if r['diagnostic_projections']!=p['projection_limits']:raise RuntimeError('Incomplete diagnostics')
        r['status']='PASS_ABSOLUTE_AFFINE_GROUP_FOLD'
    except BaseException as e:
        r.update(status='FAILED_PRESERVED',error_type=type(e).__name__,error=str(e));raise
    finally:
        if monitor is not None:monitor.close();r['assigned_GPU_resources']=monitor.report()
        write_json(a.out/'RUN.json',r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','raw-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute-gpu',action='store_true');execute(p.parse_args())
