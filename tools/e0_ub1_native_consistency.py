"""UB1 saved-row consistency; strict baseline and exact registered scope."""
import argparse
import hashlib
import importlib
import os
import shutil
import signal
import sys
import time
import json
from pathlib import Path

ID='E0_UB1_NATIVE_CONSISTENCY_20261027'
ROOT=Path(__file__).resolve().parents[1]
EXPECTED={(source,step,block) for source in (0,12) for step in range(4) for block in (0,39)}
# Per case: dense original-M W1/W2 and Q2, plus full-M and M160 each
# (narrow W1, selected/narrow W2, two R256 linears, narrow Q2).
LIMITS={'BF16_GEMM':192,'native_Q2':48,'full_model_calls':0,'fits':0,'video_generations':0}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''): h.update(chunk)
    return h.hexdigest()


def authorize(plan,digest,active):
    if plan.get('id')!=ID or plan.get('limits')!=LIMITS: raise ValueError('Scope/budget mismatch')
    if {(x['source'],x['step'],x['block']) for x in plan['cases']}!=EXPECTED or len(plan['cases'])!=16: raise ValueError('Case coverage mismatch')
    scopes=active['experiment_authority']['current_user_authorized_scopes']
    if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in scopes): raise ValueError('Protocol SHA not authorized')


def validate_rows(rows,total=31668,count=160):
    if len(rows)!=count or len(set(rows))!=count or any(type(v) is not int or v<0 or v>=total for v in rows): raise ValueError('Invalid row coordinates')


def verify_bindings(items,root):
    root=Path(root).resolve()
    for x in items:
        path=(root/x['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path)!=x['sha256']: raise ValueError('Asset path/SHA mismatch')


def count_call(counts,key,limits=LIMITS):
    if counts.get(key,0)>=limits[key]: raise RuntimeError('Budget exceeded before call')
    counts[key]=counts.get(key,0)+1


def bit_equal(a,b):
    import torch
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.view(torch.int16),b.view(torch.int16))


def compare(a,b):
    import torch
    if a.shape!=b.shape or a.dtype!=torch.bfloat16 or b.dtype!=torch.bfloat16: raise ValueError('Comparison requires matching BF16 shape and dtype')
    x=a.double();y=b.double();d=x-y;den=y.norm().item()
    return {'bitwise_equal':bit_equal(a,b),'mismatched_elements':int((a.view(torch.int16)!=b.view(torch.int16)).sum().item()),'relative_l2':d.norm().item()/den if den else None,'max_abs':d.abs().max().item()}


def padded(sample,rows,total):
    import torch
    validate_rows(rows,total=total,count=sample.shape[0]);ids=torch.tensor(rows,device=sample.device)
    out=torch.zeros((total,sample.shape[1]),device=sample.device,dtype=sample.dtype);out.index_copy_(0,ids,sample)
    return out,ids


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n');os.replace(temp,path)


def tensor_sha(value):
    import torch
    return hashlib.sha256(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def validate_scope(case,run):
    if run.get('source_index')!=case['source'] or run.get('seed')!=case['seed']: raise ValueError('Run source/seed mismatch')
    matches=[s for s in run['scopes'] if (s['step'],s['block'])==(case['step'],case['block'])]
    if len(matches)!=1: raise ValueError('Run scope mismatch')
    scope=matches[0]
    if scope['expert']!=case['expert'] or scope['sha256']!=case['panel_sha256'] or scope['path']!=Path(case['panel']).name or scope['row_plan']['rows']!=case['rows']: raise ValueError('Run panel/rows mismatch')
    return scope['row_plan']


def load_coeff(main_path,main_meta,rr_path,rr_meta,step,block,device,original_w2_sha,original_b2_sha):
    import numpy as np
    import torch
    def part(path,meta):
        metadata=json.loads(meta.read_text())
        if metadata.get('status')!='COMPLETE' or metadata.get('payload_sha256')!=sha(path): raise ValueError('Coefficient payload identity mismatch')
        with np.load(path,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        return arrays,metadata['data']
    label=f'step{step}_block{block:02d}'
    if main_path.name!=label+'.npz' or main_meta.name!=label+'.json' or rr_path.name!=label+'.npz' or rr_meta.name!=label+'.json': raise ValueError('Coefficient position path differs')
    arrays,info=part(main_path,main_meta)
    if step in (0,3) and (info.get('step'),info.get('block'))!=(step,block): raise ValueError('Main coefficient position differs')
    if info.get('original_W2_sha256',info.get('weight_sha256'))!=original_w2_sha or info.get('original_b2_sha256')!=original_b2_sha: raise ValueError('Original W2/b2 identity differs')
    if step in (0,3):
        groups=arrays['groups'];b2=arrays['b2'];alpha=info['alpha'];rr={k[3:]:v for k,v in arrays.items() if k.startswith('rr_')}; expected_b2=info['b2_sha256'];expected_rr=info['RR_parameter_sha256']
    else:
        groups=arrays['C_groups'];b2=arrays['C_centered_gain_bias_b2'];alpha=info['policies']['C']['modes']['centered_gain_bias']['alpha'];expected_b2=info['policies']['C']['modes']['centered_gain_bias']['b2_sha256'];rr,rrinfo=part(rr_path,rr_meta)
        if (rrinfo.get('step'),rrinfo.get('block'))!=(step,block) or rrinfo.get('weight_sha256')!=original_w2_sha: raise ValueError('Residual coefficient position/weight differs')
        expected_rr=rrinfo['parameter_sha256']
    expected=(54,108,108,162)[step]
    if groups.shape!=(expected,) or not np.issubdtype(groups.dtype,np.integer) or len(set(map(int,groups)))!=expected or int(groups.min())<0 or int(groups.max())>=216: raise ValueError('Frozen support mismatch')
    groups=np.sort(groups)
    if set(rr)!={'first','second','bias_first','bias_second'}: raise ValueError('Residual parameters differ')
    ids=torch.tensor([int(g)*64+i for g in groups for i in range(64)],device=device)
    b2=torch.as_tensor(b2,device=device,dtype=torch.bfloat16)
    params={k:torch.as_tensor(v,device=device,dtype=torch.bfloat16) for k,v in rr.items()}
    shapes={'first':(256,5120),'second':(5120,256),'bias_first':(256,),'bias_second':(5120,)}
    if b2.shape!=(5120,) or any(tuple(params[k].shape)!=v for k,v in shapes.items()) or not all(bool(torch.isfinite(v).all()) for v in [b2,*params.values()]): raise ValueError('Frozen parameter shape/finite mismatch')
    if tensor_sha(b2)!=expected_b2 or {k:tensor_sha(v) for k,v in params.items()}!=expected_rr: raise ValueError('BF16 coefficient tensor SHA differs')
    if not np.isfinite(alpha) or alpha<0: raise ValueError('Invalid alpha')
    return ids,b2,float(alpha),params


def run(args):
    plan=json.loads(args.protocol.read_text());digest=sha(args.protocol)
    authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()))
    hf=args.hf_root.resolve();weights=args.weight_hf_root.resolve()
    if not plan.get('assets_finalized'): raise ValueError('Asset views not finalized')
    required={'tools/e0_ub1_native_consistency.py','tests/test_e0_ub1_native_consistency.py','tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py','tools/step_budget_runtime.py','tools/trained_packed_runtime.py','tools/reduced_rank_residual.py','.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py','.runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py'}
    if not required <= {x['path'] for x in plan['repo_sources']} or not any(x['path'].endswith('.so') and 'hifxg_quant' in x['path'] for x in plan['repo_sources']):raise ValueError('Required source bindings missing')
    verify_bindings(plan['repo_sources'],ROOT);verify_bindings(plan['hf_assets'],hf);verify_bindings(plan['weight_assets'],weights)
    for c in plan['cases']:
        validate_rows(c['rows']);expected=('high_noise_model' if c['step']<2 else 'low_noise_model')
        if c['expert']!=expected: raise ValueError('Expert mismatch')
        for key in ('panel','run','main_npz','main_json','rr_npz','rr_json'):
            if c[key] not in {v['path'] for v in plan['hf_assets']}: raise ValueError('Unbound case asset')
        if c['shard'] not in {v['path'] for v in plan['weight_assets']}: raise ValueError('Unbound shard')
        bindings={v['path']:v['sha256'] for v in plan['hf_assets']}
        if bindings[c['panel']]!=c['panel_sha256'] or bindings[c['run']]!=c['run_sha256']: raise ValueError('Case run/panel SHA mismatch')
        validate_scope(c,json.loads((hf/c['run']).read_text()))
    sys.path.append(str(ROOT/'tools'))
    from worker_identity import require_worker
    require_worker();out=(hf/plan['output_HF_relative']).resolve()
    if out==hf or not out.is_relative_to(hf): raise ValueError('Output escaped HF')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest,'attempt':1},f)
    report={'id':ID,'status':'STARTED','protocol_sha256':digest,'counts':dict.fromkeys(LIMITS,0),'cases':[],
            'scope':'160 observed rows placed at original IDs in otherwise zero M31668; no full original state, video, fit or speed claim'}
    monitor=None;start=time.monotonic()
    def check():
        if monitor:monitor.check()
        if shutil.disk_usage(out).free<10<<30:raise RuntimeError('Disk reserve')
        if sum(x.stat().st_size for x in out.rglob('*') if x.is_file() and not x.is_symlink())>1<<30:raise RuntimeError('Output budget')
    def alarm(*_):raise TimeoutError('Wall budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(int(plan['wall_seconds']))
    try:
        import torch
        import torch.nn.functional as F
        from safetensors.torch import load_file
        from startup_gpu_reservation import reserve_before_assets
        reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt
        if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0): raise ValueError('Exactly one visible A100 required')
        torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
        report['runtime']={'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,'device_name':torch.cuda.get_device_name(0),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'physical_gpu_index':os.environ.get('FFNDS_PHYSICAL_GPU'),'float32_matmul_precision':torch.get_float32_matmul_precision(),'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'NVIDIA_TF32_OVERRIDE':os.environ.get('NVIDIA_TF32_OVERRIDE'),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')}
        report['scope'] += '; gate-only diagnostic, no residual writeback'
        snap=out/'EXECUTED_SOURCES';snap.mkdir()
        for item in plan['repo_sources']:
            dst=snap/item['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/item['path'],dst)
        shutil.copyfile(args.protocol,snap/'protocol.json');check()
        sys.path.insert(0,str(ROOT/'.runtime/code/Wan2.2-I2V-A14B-W4A4'));sys.path.insert(0,str(ROOT/'.runtime/code/HiF4_Sp'))
        from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
        ext=importlib.import_module('HiFloat4.hif4_gpu.quant_cy.base.cusrc.hifxg_quant')
        allowed={str((ROOT/x['path']).resolve()) for x in plan['repo_sources'] if x['path'].endswith('.so') and 'hifxg_quant' in x['path']}
        if str(Path(ext.__file__).resolve()) not in allowed: raise ValueError('Loaded QDQ extension is not bound')
        def gemm(x,w,b):count_call(report['counts'],'BF16_GEMM');return F.linear(x,w,b)
        def quant(x):count_call(report['counts'],'native_Q2');return qdq(x,force_fp32=True).to(torch.bfloat16)
        def rr(x,p):return gemm(gemm(x,p['first'],p['bias_first']),p['second'],p['bias_second'])
        for c in plan['cases']:
            check();step=c['step'];block=c['block'];raw=torch.load(weights/c['shard'],map_location='cpu',weights_only=True,mmap=True)[c['expert']]
            r0=raw[f'blocks.{block}.ffn.0'];r2=raw[f'blocks.{block}.ffn.2']
            original_w2_sha=tensor_sha(r2['weight_qdq'])
            w0=r0['weight_qdq'].to(device='cuda',dtype=torch.bfloat16);w2=r2['weight_qdq'].to(device='cuda',dtype=torch.bfloat16);smooth=r2['smooth_scale'].to(device='cuda',dtype=torch.float32)
            if tuple(w0.shape)!=(13824,5120) or tuple(w2.shape)!=(5120,13824) or smooth.numel()!=13824 or not bool(torch.isfinite(smooth).all()) or not bool((smooth>0).all()) or not bool(torch.isfinite(w0).all()) or not bool(torch.isfinite(w2).all()): raise ValueError('Native weight/scale mismatch')
            d=load_file(str(hf/c['panel']),device='cpu');shapes={'q1_post':(160,5120),'z':(160,13824),'gelu':(160,13824),'q2_post':(160,13824),'ffn_output':(160,5120),'gate':(1,5120),'b1':(13824,),'b2':(5120,)}
            for name,shape in shapes.items():
                if tuple(d[name].shape)!=shape or d[name].dtype!=torch.bfloat16 or not bool(torch.isfinite(d[name]).all()): raise ValueError('Panel shape/dtype/finite mismatch: '+name)
            d={k:d[k].cuda() for k in shapes};ids,bias,alpha,params=load_coeff(hf/c['main_npz'],hf/c['main_json'],hf/c['rr_npz'],hf/c['rr_json'],step,block,'cuda',original_w2_sha,tensor_sha(d['b2']))
            full,rows=padded(d['q1_post'],c['rows'],31668)
            def baseline(boundary,actual,expected):
                diff=compare(actual,expected);report['baseline_progress']={'source':c['source'],'step':step,'block':block,'boundary':boundary,'comparison':diff};write(out/'RUN.json',report)
                if not diff['bitwise_equal']: raise ValueError('Strict dense '+boundary+' baseline failed')
            z=gemm(full,w0,d['b1']);zs=z.index_select(0,rows)
            baseline('W1',zs,d['z'])
            h=F.gelu(z,approximate='tanh');hs=h.index_select(0,rows)
            baseline('GELU',hs,d['gelu'])
            a=quant((h.float()/smooth.reshape(1,-1)).to(torch.bfloat16));a_s=a.index_select(0,rows)
            baseline('Q2',a_s,d['q2_post'])
            y=gemm(a,w2,d['b2']).index_select(0,rows)
            baseline('W2',y,d['ffn_output'])
            del z,h,a
            row_plan=validate_scope(c,json.loads((hf/c['run']).read_text()))
            center=set(row_plan['contiguous_tiles'][0]); uniform=set(row_plan['uniform_rows'])
            groups={'center128':[i for i,r in enumerate(c['rows']) if r in center],'uniform32':[i for i,r in enumerate(c['rows']) if r in uniform]}
            if len(groups['center128'])!=128 or len(groups['uniform32'])!=32: raise ValueError('Panel row plan differs')
            result={'source':c['source'],'step':step,'block':block,'baseline_bitwise':True,'arms':{}}; saved_shapes={}
            w0s=w0.index_select(0,ids).contiguous();w2s=w2.index_select(1,ids).contiguous();b1s=d['b1'].index_select(0,ids);s2=smooth.index_select(0,ids); selected=d['q2_post'].index_select(1,ids)
            for label,x,chosen in [('M31668',full,rows),('M160',d['q1_post'],None)]:
                zn=gemm(x,w0s,b1s);hn=F.gelu(zn,approximate='tanh');an=quant((hn.float()/s2.reshape(1,-1)).to(torch.bfloat16))
                pred=rr(x,params)
                sf=selected if chosen is None else padded(selected,c['rows'],31668)[0]
                selected_main=gemm((sf.float()*alpha).to(torch.bfloat16),w2s,bias)
                narrow_main=gemm((an.float()*alpha).to(torch.bfloat16),w2s,bias)
                if chosen is not None:
                    zn=zn.index_select(0,chosen);hn=hn.index_select(0,chosen);an=an.index_select(0,chosen);pred=pred.index_select(0,chosen);selected_main=selected_main.index_select(0,chosen);narrow_main=narrow_main.index_select(0,chosen)
                combined_selected=selected_main+pred;combined_narrow=narrow_main+pred
                if not all(bool(torch.isfinite(t).all()) for t in [zn,hn,an,pred,combined_selected,combined_narrow]):raise ValueError('Nonfinite output')
                gate=d['gate']; gated_dense=d['ffn_output']*gate;gated_selected=combined_selected*gate;gated_narrow=combined_narrow*gate
                saved_shapes[label]={'z':zn,'gelu':hn,'q2':an,'main_selected':selected_main,'main_narrow':narrow_main,'pred_rr':pred,'combined_selected':combined_selected,'combined_narrow':combined_narrow,'gated_selected':gated_selected,'gated_narrow':gated_narrow}
                error=(gated_narrow.float()-gated_dense.float()).double().norm().item();gap=(gated_narrow.float()-gated_selected.float()).double().norm().item()
                grouped={group:{'path_gap':compare(gated_narrow[ix],gated_selected[ix]),'candidate_error':compare(gated_narrow[ix],gated_dense[ix])} for group,ix in groups.items()}
                per_row=[{'row_id':row,'path_gap':compare(gated_narrow[j:j+1],gated_selected[j:j+1]),'candidate_error':compare(gated_narrow[j:j+1],gated_dense[j:j+1])} for j,row in enumerate(c['rows'])]
                result['arms'][label]={'grouped':grouped,'rows':per_row,'path_gap_over_candidate_error':gap/error if error else None, 'z_vs_dense_selected':compare(zn,d['z'].index_select(1,ids)), 'gelu_vs_dense_selected':compare(hn,d['gelu'].index_select(1,ids)), 'q2_vs_dense_selected':compare(an,selected), 'main_narrow_vs_selected':compare(narrow_main,selected_main),'gated_narrow_vs_selected':compare(gated_narrow,gated_selected),'gated_narrow_vs_dense':compare(gated_narrow,gated_dense),'gated_selected_vs_dense':compare(gated_selected,gated_dense)}
                del sf
            cross={}
            for name,left in saved_shapes['M31668'].items():
                right=saved_shapes['M160'][name]
                cross[name]={'all':compare(left,right),'groups':{group:compare(left[ix],right[ix]) for group,ix in groups.items()},'rows':[{'row_id':row,**compare(left[j:j+1],right[j:j+1])} for j,row in enumerate(c['rows'])]}
            result['cross_shape']=cross
            report['cases'].append(result);write(out/'RUN.json',report);check()
            del raw,r0,r2,w0,w2,smooth,d,full,ids,bias,params,w0s,w2s,b1s,s2,selected
        if report['counts']!=LIMITS:raise RuntimeError('Incomplete budget')
        report.update(status='COMPLETE_UB1_LOCAL_CONSISTENCY',elapsed_seconds=time.monotonic()-start)
    except BaseException as e:
        report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0)
        if monitor:
            monitor.close();report['assigned_gpu_resources']=monitor.report()
            if report['assigned_gpu_resources']['violations']:report['status']='FAILED_GPU_MONITOR'
        write(out/'RUN.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--hf-root',type=Path,required=True);parser.add_argument('--weight-hf-root',type=Path,required=True);run(parser.parse_args())
