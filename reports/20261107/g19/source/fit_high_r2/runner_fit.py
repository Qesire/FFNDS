"""Guarded G19 layer runner. Runs only in an allocated CUDA Slurm job."""
import argparse, hashlib, json, os, resource, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from fit_core import Problem

def pair_rows(source, target, expected):
    """Exact identity and token-THW join; metadata checked before tensor use."""
    for obj,step in ((source,expected['source_step']),(target,expected['target_step'])):
        meta=obj['metadata']
        for key in ('source','seed','layer','expert','checkpoint_sha256'):
            if meta[key]!=expected[key]: raise ValueError('metadata mismatch: '+key)
        if meta['step']!=step: raise ValueError('step mismatch')
        ids=obj['token_indices']; coords=obj['token_thw']
        if ids.unique().numel()!=ids.numel() or coords.shape!=(ids.numel(),3): raise ValueError('invalid coordinates')
    ids0=source['token_indices']; ids1=target['token_indices']
    order=torch.argsort(ids1); pos=torch.searchsorted(ids1[order],ids0); valid=pos<ids1.numel()
    valid &= ids1[order[pos.clamp(max=ids1.numel()-1)]]==ids0
    i=valid.nonzero().flatten(); j=order[pos[valid]]
    if not i.numel() or not torch.equal(source['token_thw'][i],target['token_thw'][j]): raise ValueError('no matching coordinates')
    return i,j


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()

def load_pair(assets,source,seed,expert,steps,checkpoint_sha,layer):
    objects=[]
    for step in steps:
        hits=[x for x in assets['captures'] if (x['metadata']['source'],x['metadata']['seed'],x['metadata']['step'],x['metadata']['layer'])==(source,seed,step,layer)]
        if len(hits)!=1: raise ValueError('missing/duplicate capture')
        e=hits[0]; m=e['metadata']
        if sha(e['path'])!=e['sha256'] or sha(e['sidecar'])!=e['sidecar_sha256'] or json.loads(Path(e['sidecar']).read_text())!=m: raise ValueError('capture identity')
        data=torch.load(e['path'],map_location='cpu',weights_only=True)
        for k in ('source','seed','step','layer','expert','checkpoint_sha256'):
            if data['metadata'][k]!=m[k]: raise ValueError('payload identity '+k)
        if m['grid']!=[21,29,52] or m['checkpoint_sha256']!=checkpoint_sha or m['expert']!=expert: raise ValueError('checkpoint/geometry')
        ids=data['token_indices']
        if ids.dtype!=torch.int64 or ids.shape!=(2048,) or ids.min()<0 or ids.max()>=31668: raise ValueError('invalid token ids')
        expected=torch.stack((ids//1508,(ids//52)%29,ids%52),1)
        if not torch.equal(data['token_thw'],expected): raise ValueError('token geometry')
        if any(data[k].shape!=(2048,5120) or data[k].dtype!=torch.bfloat16 or not torch.isfinite(data[k]).all() for k in ('z','w2_output_before_gate')): raise ValueError('tensor shape/dtype/finiteness')
        if not torch.isfinite(data['gate']).all(): raise ValueError('nonfinite gate')
        objects.append(data)
    expected={'source':source,'seed':seed,'source_step':steps[0],'target_step':steps[1],'layer':layer,'expert':expert,'checkpoint_sha256':checkpoint_sha}
    i,j=pair_rows(*objects,expected); a,b=objects
    def gate(x,idx):
        g=x['gate'].squeeze()
        if g.shape==(5120,): return g[None,:].expand(len(idx),-1).cuda().float()
        if g.shape==(31668,5120): return g.index_select(0,x['token_indices'][idx]).cuda().float()
        raise ValueError('gate shape')
    return {'source':source,'ids':a['token_indices'][i].tolist(),'u0':a['z'][i].cuda(),'u1':b['z'][j].cuda(),'y0':a['w2_output_before_gate'][i].cuda(),'y1':b['w2_output_before_gate'][j].cuda(),'g':gate(b,j)}


def selfcheck(cfg):
    torch.manual_seed(19); device='cuda'; d=4; m=7
    w1=torch.randn(m,d,device=device); w2=torch.randn(d,m,device=device); a0=torch.randn(d,d,device=device)
    ss=[dict(x=torch.randn(9,d,device=device),y=torch.randn(9,d,device=device),g=torch.randn(9,d,device=device),dh=torch.randn(9,m,device=device))]
    p=Problem(w1,w2,a0,ss); alpha,_=p.fit(torch.zeros(m,device=device),100,1e-5)
    eye=torch.eye(m,device=device); normal=torch.stack([p.normal(eye[:,i]) for i in range(m)],1)
    base=p.ridge*p.k0/p.za+(ss[0]['h']*((ss[0]['y']*ss[0]['g'].square())@p.w2)).mean(0)/p.z
    k=torch.stack([torch.outer(w1[i],w2[:,i]).reshape(-1) for i in range(m)],1)
    if not torch.allclose(k.T@k,p.gram,rtol=2e-5,atol=2e-5): raise RuntimeError('synthetic Gram mismatch')
    design=torch.stack([((ss[0]['x']@k[:,i].reshape(d,d))*ss[0]['g']).reshape(-1) for i in range(m)],1)
    dense=design.T@design/(ss[0]['x'].shape[0]*p.z)+p.ridge*(k.T@k)/p.za
    dense_rhs=design.T@(ss[0]['y']*ss[0]['g']).reshape(-1)/(ss[0]['x'].shape[0]*p.z)+p.ridge*(k.T@a0.reshape(-1))/p.za
    if not torch.allclose(normal,dense,rtol=3e-5,atol=3e-5) or not torch.allclose(base,dense_rhs,rtol=3e-5,atol=3e-5): raise RuntimeError('synthetic normal equations mismatch')
    direct=torch.linalg.solve(dense,dense_rhs)
    if (alpha-direct).norm()>2e-3*direct.norm().clamp_min(1e-8): raise RuntimeError('synthetic PCG solve mismatch')
    if (normal@alpha-base).norm() > 2e-4*base.norm().clamp_min(1e-10): raise RuntimeError('synthetic normal residual')
    kg=k.T@p.gradient(p.matrix(alpha)).reshape(-1)
    if not torch.allclose(kg,2*(normal@alpha-base),rtol=1e-3,atol=1e-5): raise RuntimeError('synthetic restricted gradient mismatch')
    a=p.matrix(alpha); b,info,_=p.rank1(a,20)
    if not torch.isfinite(b).all() or info['after']>info['before']+1e-6: raise RuntimeError('synthetic selfcheck')
    del p,ss,a,b
    torch.cuda.empty_cache()

def replay(s,a):
    a=a.to(torch.bfloat16); u0=s['u0']; u1=s['u1']; y0=s['y0']; y1=s['y1']; g=s['g']
    q0=u0@a; q1=u1@a
    c=(y0.float()-q0.float()).bfloat16()
    pred=(q1.float()+c.float()).bfloat16()
    sse=(((pred.float()-y1.float())*g).square().sum().item())
    return {'source':s['source'],'gated_sse':sse,'gated_mse':sse/y1.numel(),'tokens':u0.shape[0],'token_indices':s['ids'],'tau_coverage':sorted(set(i//1508 for i in s['ids']))}

def legacy(sources):
    num=sum((s['h']*s['dh']).mean(0) for s in sources)/len(sources)
    den=sum(s['h'].square().mean(0) for s in sources)/len(sources)
    return torch.where(den>0,num/den.clamp_min(torch.finfo(den.dtype).tiny),torch.zeros_like(den))

def main():
    ap=argparse.ArgumentParser()
    for name in ('assets','output','config','old-fit-manifest'): ap.add_argument('--'+name,required=True)
    ap.add_argument('--expert',choices=['high','low'],required=True); args=ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('requires Slurm CUDA allocation')
    for x in (args.assets,args.output,args.config,args.old_fit_manifest):
        if not os.path.isabs(x): raise ValueError('absolute paths required')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    cfg=json.loads(Path(args.config).read_text()); assets=json.loads(Path(args.assets).read_text()); old_manifest=json.loads(Path(args.old_fit_manifest).read_text())
    if old_manifest['expert']!=args.expert or set(old_manifest['layers'])!=set(map(str,range(40))) or old_manifest['shared_assets_sha256']!=sha(args.assets): raise ValueError('old manifest identity/assets SHA')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False); started=time.monotonic(); complete=[]
    try:
        selfcheck(cfg)
        ck=assets['checkpoints'][args.expert]
        if sha(ck['path'])!=ck['sha256']: raise ValueError('checkpoint SHA')
        steps=(0,1) if args.expert=='high' else (2,3)
        for layer in range(40):
            if time.monotonic()-started>cfg['soft_seconds']: raise TimeoutError('soft time budget')
            torch.cuda.reset_peak_memory_stats(); print('layer',layer,flush=True)
            with safe_open(ck['path'],framework='pt',device='cpu') as f:
                weights=[f.get_tensor('blocks.%d.ffn.%s'%(layer,n)).cuda() for n in ('0.weight','0.bias','2.weight','2.bias')]
            w1,b1,w2,b2=weights
            if [tuple(t.shape) for t in weights]!=[(13824,5120),(13824,),(5120,13824),(5120,)] or any(t.dtype!=torch.bfloat16 or not torch.isfinite(t).all() for t in weights): raise ValueError('weight shape/dtype/finiteness')
            entry=assets['matrices'][args.expert+'/'+str(layer)]
            if sha(entry['path'])!=entry['sha256']: raise ValueError('baseline SHA')
            raw=np.load(entry['path'],allow_pickle=False)
            if raw.dtype!=np.uint16 or raw.shape!=(5120,5120): raise ValueError('baseline geometry')
            a0=torch.from_numpy(raw.copy()).view(torch.bfloat16).cuda().float()
            if not torch.isfinite(a0).all(): raise ValueError('baseline nonfinite')
            sources=[]
            for name in assets['cal_sources']:
                s=load_pair(assets,name,42,args.expert,steps,ck['sha256'],layer)
                s['x']=s['u1'].float()-s['u0'].float(); s['y']=s['y1'].float()-s['y0'].float()
                z0=F.linear(s['u0'].float(),w1.float(),b1.float()); z1=F.linear(s['u1'].float(),w1.float(),b1.float())
                s['dh']=(F.gelu(z1,approximate='tanh').float()-F.gelu(z0,approximate='tanh').float()); del z0,z1
                sources.append(s)
            p=Problem(w1,w2,a0,sources); initial=legacy(sources)
            candidate,fit=p.fit(initial,cfg['maxiter'],cfg['tol']); ah,hm=p.fit(initial,cfg['maxiter'],cfg['tol'],hidden=True)
            old=old_manifest['layers'][str(layer)]
            for key in ('alpha','metrics','q1'):
                item=old[key]
                if not os.path.isabs(item['path']) or sha(item['path'])!=item['sha256']: raise ValueError('old fit identity '+key)
            old_metrics=json.loads(Path(old['metrics']['path']).read_text())
            if old_metrics['expert']!=args.expert or old_metrics['layer']!=layer: raise ValueError('old fit layer identity')
            alpha_old=torch.from_numpy(np.load(old['alpha']['path'],allow_pickle=False)['q1'].copy()).cuda().float()
            if alpha_old.shape!=initial.shape or not torch.isfinite(alpha_old).all(): raise ValueError('old alpha identity')
            old_bits=np.load(old['q1']['path'],allow_pickle=False)
            if old_bits.dtype!=np.uint16 or old_bits.shape!=(5120,5120): raise ValueError('old q1 geometry')
            old_bf=torch.from_numpy(old_bits.copy()).view(torch.bfloat16).cuda()
            old_rebuilt=((w1.double().T*alpha_old.double())@w2.double().T).float().bfloat16()
            if not torch.equal(old_bf,old_rebuilt): raise ValueError('old alpha does not rebuild old Q1')
            candidate_bf=((w1.double().T*candidate.double())@w2.double().T).float().bfloat16()
            old_obj=p.objective(old_bf.float()).item(); candidate_obj=p.objective(candidate_bf.float()).item()
            if not np.isfinite(old_obj) or not np.isfinite(candidate_obj): raise ValueError('nonfinite selection objective')
            _,old_res=p.fit(alpha_old,0,cfg['tol']); r0=fit['initial_residual']
            selection={'rule':'minimum same-data BF16 objective; tie selects old',
                'selected':'OLD' if old_obj<=candidate_obj else 'NEW',
                'old_bf16_objective':old_obj,'candidate_bf16_objective':candidate_obj,
                'old_fp32_objective':p.objective(p.matrix(alpha_old)).item(),
                'candidate_fp32_objective':p.objective(p.matrix(candidate)).item(),
                'old_true_normal_residual':old_res['true_normal_residual'],
                'old_true_relative_to_legacy_r0':old_res['true_normal_residual']/max(r0,1e-30),
                'candidate_true_normal_residual':fit['true_normal_residual'],
                'candidate_true_relative_to_legacy_r0':fit['true_relative_residual'],
                'old_q1_sha256':old['q1']['sha256']}
            alpha=alpha_old if selection['selected']=='OLD' else candidate
            if selection['selected']=='OLD':
                selected_fit={'origin':'OLD','initial_residual':r0,
                    'true_normal_residual':old_res['true_normal_residual'],
                    'true_relative_residual':selection['old_true_relative_to_legacy_r0'],
                    'status':'CONVERGED' if old_res['true_normal_residual']<=cfg['tol']*max(r0,1e-30) else 'FINITE_ITERATE_NOT_OPTIMUM',
                    'certificate':'CERTIFICATE_UNRESOLVED'}
            else:
                selected_fit={**fit,'origin':'NEW'}
            del old_bf,old_rebuilt,candidate_bf,old_bits

            def hidden_objective(v):
                data=sum(((s['h']*v-s['dh']).square().sum()/s['h'].shape[0]) for s in sources)/len(sources)/p.hz
                return (data+p.ridge*(p.matrix(v)-p.a0).square().sum()/p.za).item()
            hidden_metrics={'legacy':hidden_objective(initial),'h_match':hidden_objective(ah),'q1':hidden_objective(alpha)}
            a1=p.matrix(alpha); a2,rank,direction=p.rank1(a1,cfg['power_iterations'])
            folds=[]
            for hold in range(4):
                train=[s for i,s in enumerate(sources) if i!=hold]; fp=Problem(w1,w2,a0,train,shared=p)
                fa,fi=fp.fit(legacy(train),cfg['maxiter'],cfg['tol']); f1=fp.matrix(fa); f2,fr,_=fp.rank1(f1,cfg['power_iterations'])
                held=sources[hold]
                def error(a): return (((held['x']@a-held['y'])*held['g']).square().mean()).item()
                folds.append({'held_source':assets['cal_sources'][hold],'q1_mse':error(f1),'q2_mse':error(f2),'fit':fi,'rank1':fr,'scope':'NEW_PCG_FIXED_PRIOR_LOSO_REFIT_ONLY','metric_precision':'FP32 real-matrix gated mean squared error'})
                del fp,fa,f1,f2
            # Merge in F64 before conversion, matching production fold recipe.
            real1=(w1.double().T*alpha.double())@w2.double().T
            real2=real1-rank['eta']*direction.double()
            def obj64(a):
                z64=sum(((s['y'].double()*s['g'].double()).square().sum()/s['y'].shape[0]) for s in sources)/len(sources)+1e-12
                za64=p.a0.double().square().sum()+1e-12
                data=sum((((s['x'].double()@a-s['y'].double())*s['g'].double()).square().sum()/s['x'].shape[0]) for s in sources)/len(sources)/z64
                return (data+p.ridge*(a-p.a0.double()).square().sum()/za64).item()
            real_before=obj64(real1); real_after=obj64(real2)
            if real_after>real_before+1e-9*max(1.,abs(real_before)): raise RuntimeError('F64 merged objective increased')
            rank['f64_merged_before']=real_before; rank['f64_merged_after']=real_after
            q1=real1.float().bfloat16(); q2=real2.float().bfloat16()
            changed=(q1.view(torch.int16)!=q2.view(torch.int16)).sum().item()
            legacy_q=((w1.double().T*initial.double())@w2.double().T).float().bfloat16()
            hmatch_q=((w1.double().T*ah.double())@w2.double().T).float().bfloat16()
            def parts(a):
                data=sum((((s['x']@a-s['y'])*s['g']).square().sum()/s['x'].shape[0]) for s in sources)/len(sources)/p.z
                prior=p.ridge*(a-p.a0).square().sum()/p.za
                return {'data':data.item(),'prior':prior.item(),'total':(data+prior).item()}
            metrics={'objective_parts':{'G0':parts(a0),'Hlegacy':parts(legacy_q.float()),'Hmatch':parts(hmatch_q.float()),'Q1':parts(q1.float()),'Q2':parts(q2.float())},'hidden_objective':hidden_metrics,'fp32_q1_objective':p.objective(real1.float()).item(),'fp32_q2_objective':p.objective(real2.float()).item(),'bf16_q1_objective':p.objective(q1.float()).item(),'bf16_q2_objective':p.objective(q2.float()).item(),'expert':args.expert,'layer':layer,'q1_fit':selected_fit,'q1_candidate_fit':fit,'selection':selection,'h_match_fit':hm,'rank1':rank,'folds':folds,'q1_q2_bf16_changed_entries':changed,'replay':{'G0':[replay(s,a0) for s in sources],'Hlegacy':[replay(s,legacy_q) for s in sources],'Hmatch':[replay(s,hmatch_q) for s in sources],'Q1':[replay(s,q1) for s in sources],'Q2':[replay(s,q2) for s in sources]},'numeric':'FP32 restarted PCG with true residual; certificate unresolved; F64 merge to F32 to BF16','rss_max_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            if metrics['rss_max_kib']>192*1024**2: raise RuntimeError('RSS budget exceeded')
            if metrics['gpu_peak_bytes']>78*1024**3 or metrics['gpu_peak_reserved_bytes']>78*1024**3: raise RuntimeError('GPU budget exceeded')
            layerdir=out/('layer_%02d'%layer); layerdir.mkdir()
            for name,q in [('Q1',q1),('Q2',q2)]: np.save(layerdir/(name+'.npy'),q.view(torch.int16).cpu().numpy().view(np.uint16),allow_pickle=False)
            np.savez(layerdir/'alpha.npz',q1=alpha.cpu().numpy(),h_match=ah.cpu().numpy(),h_legacy=initial.cpu().numpy())
            (layerdir/'METRICS.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
            size=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
            if size>cfg['output_budget_bytes']: raise RuntimeError('output budget exceeded')
            (layerdir/'COMPLETE.json').write_text(json.dumps({'expert':args.expert,'layer':layer,'output_bytes':size,'sha256':{name:sha(layerdir/name) for name in ('Q1.npy','Q2.npy','alpha.npz','METRICS.json')}}))
            complete.append(layer)
            del weights,w1,b1,w2,b2,a0,sources,p,alpha,ah,initial,a1,a2,direction,real1,real2,q1,q2,legacy_q,hmatch_q,s,held,train
            torch.cuda.empty_cache()
        (out/'SUMMARY.json').write_text(json.dumps({'expert':args.expert,'completed_layers':complete},indent=2))
    except BaseException as e:
        (out/'FAILURE.json').write_text(json.dumps({'expert':args.expert,'completed_layers':complete,'error':str(e),'traceback':traceback.format_exc()},indent=2))
        raise

if __name__=='__main__': main()
