"""Frozen adjacent-token/pre-W1 saved-operand GPU discrimination.

All W1/W2 operations are charged diagnostics; nothing is returned to a model.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256, write_json
from e0_intrinsic_math import energy, top_support
from e0_shared_tile_math import block_means, selector_scores, shared_mask

ROOT=Path(__file__).resolve().parents[1]


def execute(args):
    import torch
    from startup_gpu_reservation import reserve_before_assets
    from worker_identity import require_worker
    require_worker()
    p=json.loads(args.protocol.read_text())
    if not args.execute_gpu or 'A100' not in torch.cuda.get_device_name(0):
        raise ValueError('Explicit registered A100 scope required')
    for name,expected in p['bound_sources'].items():
        if sha256(ROOT/name)!=expected:
            raise ValueError('Frozen source changed: '+name)
    args.out.mkdir(parents=True,exist_ok=False)
    snap=args.out/'EXECUTED_SOURCES';snap.mkdir()
    for name in p['bound_sources']:
        target=snap/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(args.protocol,snap/args.protocol.name)
    r={'status':'STARTED_NOT_ACCEPTED','protocol_sha256':sha256(args.protocol),
       'new_generations':0,'native_FFN_calls':0,'training_updates':0,
       'quality_or_speed_acceptance':False,'automatic_retry':False,
       'diagnostic_projections':{'W1':0,'W2':0},'cases':[],'native_bindings':[],'weights':[]}
    write_json(args.out/'RUN.json',r)
    monitor=None;started=time.perf_counter()
    try:
        reservation,monitor,receipt=reserve_before_assets(torch)
        r['CUDA_reservation']=receipt
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        def device(x):return torch.as_tensor(x).to(device='cuda',dtype=torch.float64)
        def project(x,w,which):
            r['diagnostic_projections'][which]+=1
            return x@w.T
        cache=json.loads((ROOT/p['cache_manifest']).read_text())
        accepted=json.loads((ROOT/p['accepted_observations']).read_text())
        originals={}
        for s in accepted['sources']:
            path=args.old_root/f"source{s['index']:02d}_attempt1/RUN.json"
            if sha256(path)!=s['run_sha256']:
                raise ValueError('Original run changed')
            originals[s['index']]=(path,json.loads(path.read_text()))
        runs={}
        for i in p['source_indices']:
            path=args.native_root/f'native_source{i:02d}_attempt1/RUN.json'
            d=json.loads(path.read_text())
            if d['status']!='PASS_NATIVE_CONTIGUOUS_OPERANDS_NOOP' or d['protocol_sha256']!=p['native_protocol_sha256']:
                raise ValueError('Accepted contiguous native run required')
            if not all(s['historical_uniform_all_five_fields_exact'] for s in d['scopes']):
                raise ValueError('Original five-field identity failed')
            runs[i]=(path,d)
            r['native_bindings'].append({'source_index':i,'run_sha256':sha256(path),
                'video_sha256':d['video_sha256'],'trajectory':d['trajectory'],
                'full_generations':d['generations_started'],'native_counts':d['native_counts']})
        def saved(index,step,block,original=False):
            path,run=(originals if original else runs)[index]
            scope=next(s for s in run['scopes'] if (s['step'],s['block'])==(step,block))
            payload=path.parent/'observed'/scope['path']
            if sha256(payload)!=scope['sha256']:
                raise ValueError('Saved operands changed')
            a=load_arrays(payload)
            rows=scope['row_plan']['uniform_rows'] if original else sum(scope['row_plan']['contiguous_tiles'],[])
            ids=[scope['row_plan']['rows'].index(x) for x in rows]
            return {k:device(v[ids] if k in ['q1_post','z','gelu','q2_post','ffn_output'] else v)
                    for k,v in a.items() if k in ['q1_post','z','gelu','q2_post','ffn_output','gate','b1','b2']},scope
        with torch.inference_mode():
            for block in p['blocks']:
                for expert,step in [('high_noise_model',1),('low_noise_model',3)]:
                    monitor.check()
                    shard=next(s for s in cache['shards'] if s['expert']==expert and s['block']==block)
                    if sha256(Path(shard['path']))!=shard['sha256']:
                        raise ValueError('Full block cache changed')
                    records=torch.load(shard['path'],map_location='cpu',weights_only=False)[expert]
                    w1=records[f'blocks.{block}.ffn.0']['weight_qdq'].to('cuda',dtype=torch.bfloat16).double()
                    w2=records[f'blocks.{block}.ffn.2']['weight_qdq'].to('cuda',dtype=torch.bfloat16).double()
                    s2=device(records[f'blocks.{block}.ffn.2']['smooth_scale']).reshape(1,-1)
                    r['weights'].append({'block':block,'expert':expert,'shard_sha256':shard['sha256']})
                    for index in p['source_indices']:
                        current,cs=saved(index,step,block);previous,ps=saved(index,step-1,block)
                        if cs['row_plan']!=ps['row_plan'] or not torch.equal(current['b1'],previous['b1']):
                            raise ValueError('Same expert rows/bias mismatch')
                        gate=current['gate'];norm=(w2*gate.T).square().sum(0)
                        q=current['q2_post'];delta=q-previous['q2_post']
                        target=project(delta,w2,'W2')*gate
                        current_output=(project(q,w2,'W2')+current['b2'].reshape(1,-1))*gate
                        current_energy=energy(current_output)
                        source_scores=[]
                        for other in originals:
                            if other==index:continue
                            c,_=saved(other,step,block,True);old,_=saved(other,step-1,block,True)
                            # Other-source current gate, not the held-source label.
                            other_norm=(w2*c['gate'].T).square().sum(0)
                            score=(c['q2_post']-old['q2_post']).square().sum(0)*other_norm
                            source_scores.append(score/score.sum())
                        fixed=torch.stack(source_scores).mean(0)
                        # Exact coefficient identity separating a linear part, the
                        # saved nonlinear remainder, and Smooth/Q2 rounding.
                        linear=(current['z']-previous['z'])/(2*s2)
                        nonlinear=(current['gelu']-previous['gelu'])/s2-linear
                        rounding=delta-linear-nonlinear
                        components=[project(v,w2,'W2')*gate for v in [linear,nonlinear,rounding]]
                        if energy(sum(components)-target)>1e-20*energy(target):
                            raise RuntimeError('Signed decomposition identity failed')
                        common={'source_index':index,'step':step,'block':block,'expert':expert,
                                'current_payload_sha256':cs['sha256'],'previous_payload_sha256':ps['sha256'],
                                'relative_current_reconstruction_SSE':energy(current_output-current['ffn_output']*gate)/energy(current['ffn_output']*gate),
                                'innovation_over_current_output_energy':energy(target)/current_energy,
                                'linear_nonlinear_rounding_energy_over_innovation':[energy(c)/energy(target) for c in components],
                                'linear_nonlinear_cosine':float((components[0]*components[1]).sum()/(components[0].norm()*components[1].norm()))}
                        for layout in ['contiguous','shuffled_control']:
                            order=torch.arange(len(delta),device='cuda')
                            if layout!='contiguous':
                                gen=torch.Generator().manual_seed(p['random_seed'])
                                order=torch.randperm(len(delta),generator=gen).to('cuda')
                            d=delta[order];x=current['q1_post'][order];oldx=previous['q1_post'][order]
                            ac=project(block_means(x,16),w1,'W1')+current['b1'].reshape(1,-1)
                            ap=project(block_means(oldx,16),w1,'W1')+previous['b1'].reshape(1,-1)
                            for tile in (p['row_tiles'] if layout=='contiguous' else [128]):
                                mu=block_means(target[order],tile)
                                mean_energy=energy(mu)*tile
                                item={**common,'layout':layout,'row_tile':tile,
                                      'tile_count':len(d)//tile,
                                      'innovation_token_mean_energy_fraction':mean_energy/energy(target),
                                      'results':{}}
                                scores=selector_scores(d,ac,ap,s2,norm,tile)
                                scores['source_fixed_LOSO']=fixed.expand(len(d)//tile,-1)
                                scores['pointwise_posthoc_diagonal']=d.square()*norm
                                for fraction in p['retained_fractions']:
                                    count=round(d.shape[1]*fraction);out={}
                                    for name,score in scores.items():
                                        mask=(top_support(score,count) if name.startswith('pointwise_')
                                              else shared_mask(score,tile,count))
                                        if not bool((mask.sum(-1)==count).all()):
                                            raise RuntimeError('Unequal channel budget')
                                        error=project(d*(~mask),w2,'W2')*gate
                                        error_mu=block_means(error,tile)
                                        mean_error=energy(error_mu)*tile
                                        total_error=energy(error)
                                        out[name]={'relative_innovation_SSE':total_error/energy(target),
                                                   'relative_current_output_SSE':total_error/current_energy,
                                                   'error_mean_part':mean_error/energy(target),
                                                   'error_centered_part':(total_error-mean_error)/energy(target)}
                                    item['results'][str(fraction)]=out
                                r['cases'].append(item)
                        print(f'shared support source{index:02d} step{step} block{block:02d} complete',flush=True)
                    del w1,w2,records
                    write_json(args.out/'RUN.json',r)
        monitor.check();r.update(status='PASS_SHARED_TILE_SAVED_DISCRIMINATION',elapsed_seconds=time.perf_counter()-started)
    except BaseException as e:
        r.update(status='FAILED_PRESERVED',error_type=type(e).__name__,error=str(e));raise
    finally:
        if monitor is not None:
            monitor.close();r['assigned_GPU_resources']=monitor.report()
        write_json(args.out/'RUN.json',r)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','native-root','old-root','out']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--execute-gpu',action='store_true');execute(ap.parse_args())
