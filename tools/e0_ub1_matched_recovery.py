"""Fixed source-disjoint UB1 recovery diagnostic; no full-model execution."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ID='E0_UB1_MATCHED_RECOVERY_20261027'
POSITIONS={(s,b) for s in (0,3) for b in (0,39)}
ARMS=('cmean_scalar','cmean_ridge','flap_scalar','flap_ridge')
LIMITS={'group_weight_gram':4,'group_activation_gram':8,'scalar_signal_FP32_linear':192,'dense_FP32_linear':96,'consumer_ridge_fit':16,'RR_fit':32,'BF16_forward_linear':2304,'optimizer_steps':0,'W1_calls':0,'QDQ_calls':0,'full_model_calls':0,'video_generations':0}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def folds():
    even=tuple(range(0,24,2));odd=tuple(range(1,24,2))
    return ((even,odd),(odd,even))


def authorize(plan,digest,active):
    if plan.get('id')!=ID or plan.get('limits')!=LIMITS or plan.get('positions')!=[[0,0],[0,39],[3,0],[3,39]]: raise ValueError('Fixed scope/budget differs')
    if not plan.get('assets_finalized') or not plan.get('math_sha256') or len(plan['math_sha256'])!=64: raise ValueError('Asset or math binding unfinished')
    if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in active['experiment_authority']['current_user_authorized_scopes']): raise ValueError('Exact protocol SHA not authorized')


def verify_bindings(items,root):
    root=Path(root).resolve()
    if len({v['path'] for v in items})!=len(items): raise ValueError('Duplicate asset binding')
    for item in items:
        p=(root/item['path']).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p)!=item['sha256']: raise ValueError('Asset SHA/path mismatch')


def validate_cases(plan):
    cases=plan['cases'];expected={(i,s,b) for i in range(24) for s,b in POSITIONS}
    if len(cases)!=96 or {(v['index'],v['step'],v['block']) for v in cases}!=expected: raise ValueError('Expected exact 96 payloads')
    sources=plan['sources']
    if len(sources)!=24 or {x['index'] for x in sources}!=set(range(24)) or len({x['source_id'] for x in sources})!=24: raise ValueError('Source identities differ')
    for train,held in folds():
        if len(train)!=12 or len(held)!=12 or set(train)&set(held): raise ValueError('Fold leakage')
    source={x['index']:x for x in sources}
    for group in (range(0,24,2),range(1,24,2)):
        strata=[source[i]['dimension'] for i in group]
        if strata.count('i2v_subject')!=6 or strata.count('i2v_background')!=6: raise ValueError('Stratification differs')
    for c in cases:
        if c['expert']!=('high_noise_model' if c['step']==0 else 'low_noise_model') or len(c['rows'])!=256 or len(set(c['rows']))!=256 or any(type(r) is not int or r<0 or r>=31668 for r in c['rows']): raise ValueError('Case expert/row geometry differs')


def training_subset(dataset,train,held):
    if len(train)!=12 or len(held)!=12 or set(train)&set(held) or set(train)|set(held)!=set(range(24)): raise ValueError('Invalid source partition')
    return [dataset[i] for i in train]


def sourcewise_forward(x,operation,rows_per_source=256):
    import torch
    if x.ndim!=2 or rows_per_source<=0 or len(x)==0 or len(x)%rows_per_source: raise ValueError('Invalid sourcewise geometry')
    return torch.cat([operation(x[start:start+rows_per_source]) for start in range(0,len(x),rows_per_source)],dim=0)


def validate_input_scope(case, record, source):
    if record.get('status')!='PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP' or record.get('index')!=case['index']:
        raise ValueError('Input report identity differs')
    value=record['input']; provenance=value['input_provenance']
    if provenance['image_sha256']!=source['source_id'] or value['source_group']!=source['source_group'] or value['calibration_seed']!=source['seed']:
        raise ValueError('Source identity or seed differs')
    found=[x for x in record['scopes'] if (x['step'],x['block'])==(case['step'],case['block'])]
    if len(found)!=1: raise ValueError('Original scope mismatch')
    scope=found[0]
    if scope['sha256']!=case['sha256'] or scope['expert']!=case['expert'] or scope['rows']!=256 or scope['selected_rows']!=case['rows']:
        raise ValueError('Original payload/rows differ')


def charge(counts,key,limit=LIMITS):
    if counts.get(key,0)>=limit[key]: raise RuntimeError('Budget exceeded before operation: '+key)
    counts[key]=counts.get(key,0)+1


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');os.replace(tmp,path)


def run(args):
    plan=json.loads(args.protocol.read_text());digest=sha(args.protocol)
    authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()));validate_cases(plan)
    hf=args.hf_root.resolve();weights=args.weight_hf_root.resolve()
    required={'tools/e0_ub1_matched_recovery.py','tests/test_e0_ub1_matched_recovery.py','tools/ub1_recovery_math.py','tools/centered_signal_fit.py','tools/reduced_rank_residual.py','tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py'}
    if not required <= {x['path'] for x in plan['repo_sources']}:raise ValueError('Required source identity missing')
    verify_bindings(plan['repo_sources'],ROOT);verify_bindings(plan['hf_assets'],hf);verify_bindings(plan['weight_assets'],weights)
    if sha(ROOT/'tools/ub1_recovery_math.py')!=plan['math_sha256']:raise ValueError('Math implementation changed')
    hfpaths={x['path']:x['sha256'] for x in plan['hf_assets']}; wpaths={x['path']:x['sha256'] for x in plan['weight_assets']}
    source_by_id={v['index']:v for v in plan['sources']}
    for c in plan['cases']:
        if hfpaths.get(c['path'])!=c['sha256'] or hfpaths.get(c['input_report'])!=c['input_report_sha256']:raise ValueError('Payload/input report not bound')
        validate_input_scope(c,json.loads((hf/c['input_report']).read_text()),source_by_id[c['index']])
    if len(plan['weights'])!=4 or {(v['step'],v['block']) for v in plan['weights']}!=POSITIONS: raise ValueError('Weight position mismatch')
    for pos in plan['weights']:
        if wpaths.get(pos['path'])!=pos['sha256']:raise ValueError('Weight shard not bound')
    sys.path.append(str(ROOT/'tools'))
    from worker_identity import require_worker
    require_worker();out=(hf/plan['output_HF_relative']).resolve()
    if out==hf or not out.is_relative_to(hf):raise ValueError('Output escaped HF')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest,'attempt':1},f)
    report={'id':ID,'status':'STARTED','protocol_sha256':digest,'counts':dict.fromkeys(LIMITS,0),'positions':[],
            'scope':'24 historically exposed sources, 12/12 source-disjoint exploratory CV; no video, new model or deployable speed claim'}
    monitor=None;started=time.monotonic()
    def check():
        if monitor:monitor.check()
        if shutil.disk_usage(out).free<10<<30:raise RuntimeError('Disk reserve')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())>256<<20:raise RuntimeError('Output limit')
    def timeout(*_):raise TimeoutError('Wall limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(3600)
    try:
        import torch
        from startup_gpu_reservation import reserve_before_assets
        reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt
        if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('Require one visible A100')
        if os.environ.get('NVIDIA_TF32_OVERRIDE')=='1':raise ValueError('TF32 override conflicts with frozen FP32 comparison')
        torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False)
        report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'float32_matmul_precision':torch.get_float32_matmul_precision(),'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'NVIDIA_TF32_OVERRIDE':os.environ.get('NVIDIA_TF32_OVERRIDE')}
        snap=out/'EXECUTED_SOURCES';snap.mkdir()
        for item in plan['repo_sources']:
            target=snap/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/item['path'],target)
        shutil.copyfile(args.protocol,snap/'protocol.json');check()
        from safetensors.torch import load_file
        import torch.nn.functional as F
        import ub1_recovery_math as math_api
        def tensor_sha(value): return hashlib.sha256(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()
        def counted(name): charge(report['counts'],name)
        def linear(x,w,b): counted('BF16_forward_linear');return F.linear(x,w,b)
        def residual(x,p): return linear(linear(x,p['first'],p['bias_first']),p['second'],p['bias_second'])
        source_map={v['index']:v for v in plan['sources']}
        for step,block in sorted(POSITIONS):
            weight_binding=next(v for v in plan['weights'] if (v['step'],v['block'])==(step,block))
            expert='high_noise_model' if step==0 else 'low_noise_model'
            raw=torch.load(weights/weight_binding['path'],map_location='cpu',weights_only=True,mmap=True)[expert][f'blocks.{block}.ffn.2']
            if tensor_sha(raw['weight_qdq'])!=weight_binding['weight_tensor_sha256']:raise ValueError('Original W2 tensor identity differs')
            w=raw['weight_qdq'].to(device='cuda',dtype=torch.bfloat16)
            if tuple(w.shape)!=(5120,13824) or not bool(torch.isfinite(w).all()):raise ValueError('Native W2 layout differs')
            dataset={};b2=None;gate=None
            for index in range(24):
                c=next(v for v in plan['cases'] if (v['index'],v['step'],v['block'])==(index,step,block))
                d=load_file(str(hf/c['path']),device='cpu')
                shape={'V1':(256,5120),'V2':(256,13824),'Y':(256,5120),'gate':(1,5120),'b2':(5120,)}
                if set(d)!=set(shape) or any(tuple(d[k].shape)!=v or d[k].dtype!=torch.bfloat16 or not bool(torch.isfinite(d[k]).all()) for k,v in shape.items()):raise ValueError('Saved tensor shape/dtype/finite mismatch')
                d={k:v.cuda() for k,v in d.items()}
                if tensor_sha(d['gate'])!=weight_binding['gate_tensor_sha256'] or tensor_sha(d['b2'])!=weight_binding['b2_tensor_sha256']:raise ValueError('Original gate/b2 identity differs')
                if gate is None:gate=d['gate'];b2=d['b2']
                elif not torch.equal(gate.view(torch.int16),d['gate'].view(torch.int16)) or not torch.equal(b2.view(torch.int16),d['b2'].view(torch.int16)):raise ValueError('Calibration gate/b2 differs')
                counted('dense_FP32_linear');d['dense_no_bias']=F.linear(d['V2'].float(),w.float(),None)
                dataset[index]=d;check()
            gram=math_api.group_weight_gram(w,gate,counted)
            result={'step':step,'block':block,'expert':expert,'folds':[]}
            for fold_id,(train,held) in enumerate(folds()):
                selected_training=training_subset(dataset,train,held)
                cat=lambda field:torch.cat([v[field] for v in selected_training],dim=0)
                x=cat('V1');a=cat('V2');y=cat('Y');dense=cat('dense_no_bias')
                supports=math_api.select_supports(a,gram,54 if step==0 else 162,counted)
                fold={'fold':fold_id,'train_indices':list(train),'held_indices':list(held),'support_scores':supports['scores'],'arms':{}}
                for support in ('cmean','flap'):
                    group=supports[support]
                    if group.ndim!=1 or len(group)!=(54 if step==0 else 162) or not torch.equal(group,group.sort().values) or len(group.unique())!=len(group) or int(group.min())<0 or int(group.max())>=216:raise ValueError('Invalid whole64 support')
                    channels=(group[:,None]*64+torch.arange(64,device=group.device)).reshape(-1).to(device=w.device,dtype=torch.long)
                    w_selected=w.index_select(1,channels).contiguous();selected=a.index_select(1,channels)
                    scalar=math_api.scalar_parameters(selected,w_selected,b2,gate,dense,counted,rows_per_source=256)
                    alpha=scalar['alpha'];bias=scalar['bias']
                    if not math.isfinite(alpha) or alpha<0 or bias.dtype!=torch.bfloat16 or tuple(bias.shape)!=(5120,) or not bool(torch.isfinite(bias).all()):raise ValueError('Scalar fit output invalid')
                    features=(selected.float()*alpha).to(torch.bfloat16)
                    for consumer in ('scalar','ridge'):
                        if consumer=='ridge': fitted=math_api.ridge_consumer(features,y,w_selected,bias,counted,fraction=.01);cw,cb=fitted['weight'],fitted['bias'];consumer_diag=fitted['diagnostics']
                        else: cw,cb=w_selected,bias;consumer_diag=scalar['diagnostics']
                        if tuple(cw.shape)!=(5120,len(channels)) or tuple(cb.shape)!=(5120,) or cw.dtype!=torch.bfloat16 or cb.dtype!=torch.bfloat16 or not bool(torch.isfinite(cw).all()) or not bool(torch.isfinite(cb).all()):raise ValueError('Consumer output mismatch')
                        main=sourcewise_forward(features,lambda batch:linear(batch,cw,cb));rr=math_api.fit_residual(x,y,main,gate,counted)
                        prediction=main+sourcewise_forward(x,lambda batch:residual(batch,rr))
                        arm={'groups':[int(v) for v in group.cpu().tolist()],'alpha':alpha,'consumer_diagnostics':consumer_diag,'train':[],'held':[]}
                        for ordinal,index in enumerate(train):
                            section=slice(ordinal*256,(ordinal+1)*256)
                            arm['train'].append({'index':index,'source_id':source_map[index]['source_id'],'seed':source_map[index]['seed'],'main':math_api.gated_metrics(main[section],y[section],gate),'combined':math_api.gated_metrics(prediction[section],y[section],gate)})
                        for index in held:
                            entry=dataset[index];feat=(entry['V2'].index_select(1,channels).float()*alpha).to(torch.bfloat16)
                            hm=linear(feat,cw,cb);hp=hm+residual(entry['V1'],rr)
                            arm['held'].append({'index':index,'source_id':source_map[index]['source_id'],'seed':source_map[index]['seed'],'main':math_api.gated_metrics(hm,entry['Y'],gate),'combined':math_api.gated_metrics(hp,entry['Y'],gate)})
                        fold['arms'][support+'_'+consumer]=arm;check()
                        del main,rr,prediction,cw,cb
                result['folds'].append(fold);check()
            report['positions'].append(result);write(out/'RUN.json',report);check()
            del raw,w,dataset,b2,gate,gram
        if report['counts']!=LIMITS:raise RuntimeError('Exact count closure failed')
        report.update(status='COMPLETE_EXPOSED_SOURCE_CV_ONLY',elapsed_seconds=time.monotonic()-started)
    except BaseException as e:
        report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0)
        if monitor:
            monitor.close();report['assigned_gpu_resources']=monitor.report()
            if report['assigned_gpu_resources']['violations']:report['status']='FAILED_GPU_MONITOR'
        write(out/'RUN.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--hf-root',type=Path,required=True);p.add_argument('--weight-hf-root',type=Path,required=True);run(p.parse_args())
