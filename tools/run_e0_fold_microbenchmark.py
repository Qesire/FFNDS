"""Isolated native after-Q1 FFN versus folded-update arithmetic on repeated rows.

This is a shape microbenchmark, never an end-to-end generation speed test.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

from analyze_ffnds_observations import load_arrays
from e0_gelu_tangent import derivative
from ffnds_observation_common import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]


def execute(a):
    import torch
    import torch.nn.functional as F
    from startup_gpu_reservation import reserve_before_assets
    from worker_identity import require_worker
    require_worker();p=json.loads(a.protocol.read_text())
    if not a.execute_gpu or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('Frozen A100 required')
    for f,h in p['bound_sources'].items():
        if sha256(ROOT/f)!=h:raise ValueError('Frozen source changed')
    a.out.mkdir(parents=True,exist_ok=False);snap=a.out/'EXECUTED_SOURCES';snap.mkdir()
    for f in p['bound_sources']:
        dest=snap/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest)
    shutil.copyfile(a.protocol,snap/a.protocol.name)
    r={'status':'STARTED_NOT_ACCEPTED','protocol_sha256':sha256(a.protocol),'cases':[],
       'counters':{'standalone_native_after_Q1_FFN':0,'native_Q2_calls':0,
                   'FP32_matrix_compositions':0,'BF16_matrix_compositions':0,
                   'folded_update_applications':0,'full_M_derivative_reductions':0},
       'full_generations':0,'persisted_weights':0,'official_quality_calls':0,
       'quality_or_speed_acceptance':False,'automatic_retry':False}
    write_json(a.out/'RUN.json',r);monitor=None
    try:
        reservation,monitor,receipt=reserve_before_assets(torch);r['CUDA_reservation']=receipt
        for sub in ['HiF4_Sp','LightX2V','Wan2.2-I2V-A14B-W4A4']:sys.path.insert(0,str(ROOT/'.runtime/code'/sub))
        for sub in ['HiFloat4','HiFloat4/hif4_gpu']:sys.path.insert(0,str(ROOT/'.runtime/code/HiF4_Sp'/sub))
        from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        r['environment']={'torch':torch.__version__,'CUDA':torch.version.cuda,'device':torch.cuda.get_device_name(0),
                          'allow_tf32':False,'BF16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        cache=json.loads((ROOT/'.runtime/hif4_rebuilt_cache.json').read_text())
        accepted=json.loads((ROOT/'research/20261025/E0_ALL_BLOCK_TANGENT_OBSERVATIONS.json').read_text())
        def count(name):
            r['counters'][name]+=1
            if r['counters'][name]>p['call_limits'][name]:raise RuntimeError('Benchmark scope exceeded')
        def timing(call):
            for _ in range(p['warmup_calls']):call()
            durations=[]
            for _ in range(p['timed_calls']):
                torch.cuda.synchronize();start=time.perf_counter();value=call();torch.cuda.synchronize()
                durations.append(1000*(time.perf_counter()-start));del value
            return durations
        def load(index,step,block):
            path=a.raw_root/f'native_source{index:02d}_attempt1/RUN.json'
            if sha256(path)!=next(v for v in accepted['raw_bindings'] if v.get('source_index')==index)['sha256']:
                raise ValueError('Native input identity changed')
            d=json.loads(path.read_text());s=next(v for v in d['scopes'] if (v['step'],v['block'])==(step,block))
            path=path.parent/'observed'/s['path']
            if sha256(path)!=s['sha256']:raise ValueError('Input tensor changed')
            values=load_arrays(path);ids=[s['row_plan']['rows'].index(row) for row in s['row_plan']['contiguous_tiles'][0]]
            return {k:torch.as_tensor(v[ids] if k in ['q1_post','z','ffn_output'] else v).to('cuda')
                    for k,v in values.items() if k in ['q1_post','z','ffn_output','b1','b2','gate','full_M_old_derivative_mean']}
        def repeat(x):return x.repeat(math.ceil(31668/len(x)),1)[:31668].contiguous()
        def energy(x):return float(x.double().square().sum())
        with torch.inference_mode():
            for block in p['blocks']:
                for expert,step in [('high_noise_model',1),('low_noise_model',3)]:
                    shard=next(s for s in cache['shards'] if s['expert']==expert and s['block']==block)
                    if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('Weight shard changed')
                    rec=torch.load(shard['path'],map_location='cpu',weights_only=False)[expert]
                    w1=rec[f'blocks.{block}.ffn.0']['weight_qdq'].to('cuda',dtype=torch.bfloat16)
                    w2=rec[f'blocks.{block}.ffn.2']['weight_qdq'].to('cuda',dtype=torch.bfloat16)
                    smooth=rec[f'blocks.{block}.ffn.2']['smooth_scale'].to('cuda',dtype=torch.float32).reshape(1,-1)
                    for index in p['source_indices']:
                        c,old=load(index,step,block),load(index,step-1,block)
                        x=c['q1_post'].bfloat16();oldx=old['q1_post'].bfloat16();oldy=old['ffn_output'].bfloat16()
                        b1,b2=c['b1'].bfloat16().reshape(-1),c['b2'].bfloat16().reshape(-1)
                        gate=c['gate'].bfloat16();slope=old['full_M_old_derivative_mean'].float()
                        def native(value):
                            count('standalone_native_after_Q1_FFN')
                            hidden=F.gelu(F.linear(value,w1,b1),approximate='tanh')
                            u=(hidden.float()/smooth).bfloat16()
                            count('native_Q2_calls');q=native_hif4_activation_qdq(u,force_fp32=True).bfloat16()
                            return F.linear(q,w2,b2)*gate
                        label=c['ffn_output'].bfloat16()*gate
                        quality=native(x)
                        item={'source_index':index,'step':step,'block':block,
                              'native_128_vs_saved_full_M_relative_SSE':energy(quality-label)/energy(label),
                              'methods':{}}
                        xf,oxf,oyf=repeat(x),repeat(oldx),repeat(oldy)
                        zold=repeat(old['z'].bfloat16())
                        def mean_derivative():
                            count('full_M_derivative_reductions')
                            total=torch.zeros(13824,device='cuda',dtype=torch.float64)
                            for begin in range(0,31668,512):
                                total.add_(derivative(zold[begin:begin+512].float()).sum(0,dtype=torch.float64))
                            return total/31668
                        item['full_M_native_after_Q1_FFN_ms']=timing(lambda:native(xf))
                        item['full_M_old_derivative_ms']=timing(mean_derivative)
                        for mode in ['FP32','BF16']:
                            def compose():
                                count(mode+'_matrix_compositions')
                                scaled=w2.float()*(slope/smooth)
                                return (scaled@w1.float()).bfloat16() if mode=='FP32' else scaled.bfloat16()@w1
                            matrix=compose()
                            def folded(value,previous,cache_y):
                                count('folded_update_applications')
                                return (cache_y+F.linear(value-previous,matrix))*gate
                            y=folded(x,oldx,oldy)
                            item['methods'][mode]={'sample128_native_output_relative_SSE':energy(y-label)/energy(label),
                                'matrix_build_ms':timing(compose),
                                'full_M_application_ms':timing(lambda:folded(xf,oxf,oyf))}
                            del matrix,y
                        r['cases'].append(item);monitor.check();write_json(a.out/'RUN.json',r)
                        print(f'fold benchmark source{index:02d} step{step} block{block:02d}',flush=True)
                        del xf,oxf,oyf,zold
                    del w1,w2,rec
        if r['counters']!=p['call_limits']:raise RuntimeError('Incomplete isolated benchmark')
        r['status']='PASS_ISOLATED_NATIVE_FOLD_MICROBENCHMARK'
    except BaseException as e:
        r.update(status='FAILED_PRESERVED',error_type=type(e).__name__,error=str(e));raise
    finally:
        if monitor is not None:monitor.close();r['assigned_GPU_resources']=monitor.report()
        write_json(a.out/'RUN.json',r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['protocol','raw-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute-gpu',action='store_true');execute(p.parse_args())
