"""Original-quantizer group closure of promising shared-column descriptions.

No QDQ/model calls: only saved operands and charged FP64 diagnostic projections.
"""
import argparse
import json
from pathlib import Path
import shutil

from analyze_ffnds_observations import load_arrays
from ffnds_observation_common import sha256,write_json
from e0_intrinsic_math import energy,top_support
from e0_shared_tile_math import block_means,selector_scores,shared_mask

ROOT=Path(__file__).resolve().parents[1]


def closure_fraction(mask, width):
    """Fraction of original groups touched, equals expanded column fraction."""
    return mask.reshape(*mask.shape[:-1],-1,width).any(-1).double().mean(-1)


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
        target=snap/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,target)
    shutil.copyfile(a.protocol,snap/a.protocol.name)
    r={'status':'STARTED_NOT_ACCEPTED','protocol_sha256':sha256(a.protocol),'cases':[],
       'diagnostic_projections':{'W1':0,'W2':0},'new_generations':0,'native_QDQ_calls':0,
       'quality_or_speed_acceptance':False,'automatic_retry':False};monitor=None
    try:
        reservation,monitor,receipt=reserve_before_assets(torch);r['CUDA_reservation']=receipt
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        dev=lambda x:torch.as_tensor(x).to('cuda',dtype=torch.float64)
        def project(x,w,name):
            r['diagnostic_projections'][name]+=1
            if r['diagnostic_projections'][name]>p['projection_limits'][name]:raise RuntimeError('Budget exceeded')
            return x@w.T
        cache=json.loads((ROOT/'.runtime/hif4_rebuilt_cache.json').read_text())
        accepted=json.loads((ROOT/'research/20261025/E0_SHARED_TILE_PRE_W1.json').read_text())
        def load(index,step,block):
            path=a.raw_root/f'native_source{index:02d}_attempt1/RUN.json'
            binding=next(v for v in accepted['raw_bindings'] if v.get('source_index')==index)
            if sha256(path)!=binding['sha256']:raise ValueError('Native accepted run changed')
            d=json.loads(path.read_text());s=next(v for v in d['scopes'] if (v['step'],v['block'])==(step,block))
            path=path.parent/'observed'/s['path']
            if sha256(path)!=s['sha256']:raise ValueError('Operand changed')
            values=load_arrays(path);ids=[s['row_plan']['rows'].index(row) for row in sum(s['row_plan']['contiguous_tiles'],[])]
            return {k:dev(v[ids] if k in ['q1_post','q2_post'] else v) for k,v in values.items() if k in ['q1_post','q2_post','gate','b1']}
        with torch.inference_mode():
            for block in p['blocks']:
                for expert,step in [('high_noise_model',1),('low_noise_model',3)]:
                    shard=next(v for v in cache['shards'] if v['expert']==expert and v['block']==block)
                    if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('Cache shard changed')
                    rec=torch.load(shard['path'],map_location='cpu',weights_only=False)[expert]
                    w1=rec[f'blocks.{block}.ffn.0']['weight_qdq'].to('cuda',dtype=torch.bfloat16).double()
                    w2=rec[f'blocks.{block}.ffn.2']['weight_qdq'].to('cuda',dtype=torch.bfloat16).double()
                    smooth=dev(rec[f'blocks.{block}.ffn.2']['smooth_scale']).reshape(1,-1)
                    for index in p['source_indices']:
                        c,old=load(index,step,block),load(index,step-1,block)
                        delta=c['q2_post']-old['q2_post'];gate=c['gate'];norm=(w2*gate.T).square().sum(0)
                        ac=project(block_means(c['q1_post'],16),w1,'W1')+c['b1'].reshape(1,-1)
                        ap=project(block_means(old['q1_post'],16),w1,'W1')+old['b1'].reshape(1,-1)
                        scores=selector_scores(delta,ac,ap,smooth,norm,128)
                        target=project(delta,w2,'W2')*gate
                        item={'source_index':index,'step':step,'block':block,'results':{}}
                        for fraction in p['retained_fractions']:
                            count=round(13824*fraction);out={}
                            for name in p['selectors']+['group64_closed_shared_diagonal']:
                                if name.startswith('group64_'):
                                    group_score=scores['posthoc_shared_diagonal'].reshape(3,216,64).sum(-1)
                                    mask=top_support(group_score,count//64).repeat_interleave(64,-1)
                                else:mask=top_support(scores[name],count)
                                error=project(delta*(~mask.repeat_interleave(128,0)),w2,'W2')*gate
                                mse=energy(error)/energy(target)
                                if not name.startswith('group64_'):
                                    reference=next(v for v in accepted['cases'] if (v['source_index'],v['step'],v['block'],v['layout'],v['row_tile'])==(index,step,block,'contiguous',128))['results'][str(fraction)][name]['relative_innovation_SSE']
                                    if abs(mse-reference)>1e-10*max(1.,reference):raise ValueError('Prior frozen selection reconstruction failed')
                                out[name]={'relative_innovation_SSE':mse,'kept_columns':count,
                                           'group_closure':{str(g):[float(v) for v in closure_fraction(mask,g)] for g in [4,8,64]}}
                            item['results'][str(fraction)]=out
                        r['cases'].append(item);monitor.check()
                        print(f'closure source{index:02d} step{step} block{block:02d}',flush=True)
                    write_json(a.out/'RUN.json',r)
                    del w1,w2,rec
        if r['diagnostic_projections']!=p['projection_limits']:raise RuntimeError('Incomplete diagnostics')
        r['status']='PASS_ORIGINAL_GROUP_CLOSURE'
    except BaseException as e:
        r.update(status='FAILED_PRESERVED',error_type=type(e).__name__,error=str(e));raise
    finally:
        if monitor is not None:monitor.close();r['assigned_GPU_resources']=monitor.report()
        write_json(a.out/'RUN.json',r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','raw-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute-gpu',action='store_true');execute(p.parse_args())
