"""One bounded W2 shape control on existing C02 saved rows; no model execution."""
import argparse, hashlib, json, os, shutil, signal, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ID='E0_W2_SHAPE_CONTROL_20261027'
LIMITS={'bf16_linear':16,'qdq':0,'fp32_projection':0,'w1':0,'full_model':0}


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''): h.update(chunk)
    return h.hexdigest()


def validate_rows(rows, total, count):
    if len(rows)!=count or any(type(x) is not int or x<0 or x>=total for x in rows) or len(set(rows))!=count:
        raise ValueError('Row IDs must be distinct, in range, and match panel shape')


def authorize(plan, digest, active):
    if plan['id']!=ID or plan['limits']!=LIMITS or plan['full_rows']!=31668 or plan['saved_rows']!=160: raise ValueError('Scope or budget mismatch')
    if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in active['experiment_authority']['current_user_authorized_scopes']): raise ValueError('Exact SHA authorization required')


def check_bindings(plan, hf, weight_hf):
    oldpath=(ROOT/plan['old_protocol']['path']).resolve()
    if not oldpath.is_relative_to(ROOT) or sha(oldpath)!=plan['old_protocol']['sha256']: raise ValueError('Old protocol SHA mismatch')
    old=json.loads(oldpath.read_text())
    if old['id']!='E0_Q2_DIRECT_COLLATERAL_20261027' or len(old['cases'])!=8 or len(old['shards'])!=4: raise ValueError('Old protocol scope mismatch')
    if {(x['source'],x['step'],x['block']) for x in old['cases']}!={(s,t,b) for s in (0,12) for t in (0,2) for b in (0,39)}: raise ValueError('Old case coverage mismatch')
    for item,base in [(x,hf) for x in old['cases']]+[(x,weight_hf) for x in old['shards']]:
        p=(base/item['path']).resolve()
        if not p.is_relative_to(base) or not p.is_file() or sha(p)!=item['sha256']: raise ValueError('Asset SHA mismatch')
    for x in old['cases']: validate_rows(x['row_plan']['rows'],31668,160)
    for item in old['sources']+plan['sources']:
        p=(ROOT/item['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or sha(p)!=item['sha256']: raise ValueError('Source SHA mismatch')
    required={'tools/e0_w2_shape_control.py','tests/test_e0_w2_shape_control.py'}
    if not required <= {x['path'] for x in plan['sources']}: raise ValueError('New source bindings missing')
    previous=(hf/plan['old_run']['path']).resolve()
    if not previous.is_relative_to(hf) or sha(previous)!=plan['old_run']['sha256']: raise ValueError('Old RUN SHA mismatch')
    snapshot=previous.parent/'EXECUTED_SOURCES'
    if sha(snapshot/'protocol.json')!=plan['old_protocol']['sha256']: raise ValueError('Old snapshot protocol mismatch')
    for item in old['sources']:
        if sha(snapshot/item['path'])!=item['sha256']: raise ValueError('Old frozen source snapshot mismatch')
    return old,json.loads(previous.read_text())


def write(path, obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n'); os.replace(tmp,path)


def compare(a,b):
    import torch
    if a.shape!=b.shape or a.ndim!=2: raise ValueError('Comparison shape mismatch')
    da=a.double(); db=b.double(); diff=da-db; denom=db.norm().item()
    if a.dtype != b.dtype: raise ValueError('Comparison dtype mismatch')
    same=(a.contiguous().view(torch.uint8).reshape(*a.shape,a.element_size()) == b.contiguous().view(torch.uint8).reshape(*b.shape,b.element_size())).all(-1)
    return {'bitwise_equal_elements':int(same.sum().item()),'elements':a.numel(),
            'unequal_elements':int((~same).sum().item()),
            'relative_l2':diff.norm().item()/denom if denom else None,'max_abs':diff.abs().max().item()}


def run(args):
    plan=json.loads(args.protocol.read_text()); digest=sha(args.protocol)
    authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()))
    hf=args.hf_root.resolve(); whf=args.weight_hf_root.resolve(); out=(hf/plan['output']).resolve()
    if out==hf or not out.is_relative_to(hf): raise ValueError('Output must be in HF root')
    old, prior=check_bindings(plan,hf,whf)
    from worker_identity import require_worker
    require_worker(); out.mkdir(parents=True,exist_ok=True)
    with (out/'ATTEMPT.json').open('x') as f: json.dump({'id':ID,'protocol_sha256':digest,'attempt':1},f)
    report={'id':ID,'status':'STARTED','protocol_sha256':digest,'counts':dict.fromkeys(LIMITS,0),'cases':[],
            'scope':'Only W2 shape control: zero-fill unknown rows; selected rows alone are observed. No full model, residual, video or speed claim.'}
    monitor=None; start=time.monotonic()
    def check():
        if monitor: monitor.check()
        if shutil.disk_usage(out).free < 10<<30: raise RuntimeError('Free space reserve')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink()) > 1<<20: raise RuntimeError('Output budget')
    def alarm(*_): raise TimeoutError('15 minute limit')
    signal.signal(signal.SIGALRM,alarm); signal.alarm(900)
    try:
        import torch
        import torch.nn.functional as F
        from safetensors.torch import load_file
        from startup_gpu_reservation import reserve_before_assets
        reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt
        torch.set_num_threads(2); torch.set_grad_enabled(False)
        # Match old C02's one explicit setting; leave all other controls at their defaults.
        torch.backends.cuda.matmul.allow_tf32=False
        report['matmul_flags']={'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'float32_matmul_precision':torch.get_float32_matmul_precision(),'default_dtype':str(torch.get_default_dtype())}
        report['runtime']={'torch':torch.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),
                           'overrides':{key:os.environ.get(key) for key in ('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG')}}
        check(); snap=out/'EXECUTED_SOURCES';snap.mkdir()
        for item in plan['sources']:
            target=snap/Path(item['path']).name; shutil.copyfile(ROOT/item['path'],target)
        shutil.copyfile(args.protocol,snap/'protocol.json'); check()
        for c in old['cases']:
            expert='high_noise_model' if c['step']==0 else 'low_noise_model'
            if c['expert']!=expert: raise ValueError('Expert mismatch')
            shard=next(x for x in old['shards'] if x['expert']==expert and x['block']==c['block'])
            record=torch.load(whf/shard['path'],map_location='cpu',weights_only=False)[expert][f"blocks.{c['block']}.ffn.2"]
            w=record['weight_qdq']; original_dtype=str(w.dtype)
            if tuple(w.shape)!=(5120,13824) or not w.is_floating_point() or not bool(torch.isfinite(w).all()): raise ValueError('Weight shape or values')
            w=w.to(device='cuda',dtype=torch.bfloat16); d=load_file(str(hf/c['path']),device='cpu')
            for k,shape in {'q2_post':(160,13824),'ffn_output':(160,5120),'b2':(5120,)}.items():
                if tuple(d[k].shape)!=shape or d[k].dtype!=torch.bfloat16: raise ValueError('Panel shape or dtype')
            a=d['q2_post'].cuda(); bias=d['b2'].cuda(); saved=d['ffn_output'].cuda(); rows=c['row_plan']['rows']; ids=torch.tensor(rows,device='cuda')
            check(); report['counts']['bf16_linear']+=1
            small=F.linear(a,w,bias)
            full=torch.zeros((31668,13824),device='cuda',dtype=torch.bfloat16);full.index_copy_(0,ids,a)
            check(); report['counts']['bf16_linear']+=1
            padded=F.linear(full,w,bias).index_select(0,ids)
            check(); oldcase=next(x for x in prior['cases'] if (x['source'],x['step'],x['block'])==(c['source'],c['step'],c['block']))
            tile=set(c['row_plan']['contiguous_tiles'][0]);groups={'center128':[j for j,r in enumerate(rows) if r in tile],'uniform_only':[j for j,r in enumerate(rows) if r not in tile]}
            item={'source':c['source'],'step':c['step'],'block':c['block'],'expert':expert,'weight_storage_dtype':original_dtype,
                  'small_vs_saved':compare(small,saved),'padded_vs_saved':compare(padded,saved),'padded_vs_small':compare(padded,small),
                  'old_c02_small_vs_saved_relative_l2':oldcase['base_replay']['x_over_y'],'groups':{},'rows':[]}
            for key,ix in groups.items(): item['groups'][key]={'small_vs_saved':compare(small[ix],saved[ix]),'padded_vs_saved':compare(padded[ix],saved[ix]),'padded_vs_small':compare(padded[ix],small[ix])} if ix else None
            for j,row in enumerate(rows): item['rows'].append({'row_id':row,'group':'center128' if row in tile else 'uniform_only','small_vs_saved':compare(small[j:j+1],saved[j:j+1]),'padded_vs_saved':compare(padded[j:j+1],saved[j:j+1]),'padded_vs_small':compare(padded[j:j+1],small[j:j+1])})
            report['cases'].append(item); del full,padded,small,w,record,d,a,bias,saved; check()
        if report['counts']!=LIMITS: raise RuntimeError('Incomplete budget')
        report['status']='COMPLETE_W2_SHAPE_CONTROL';report['elapsed_seconds']=time.monotonic()-start
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
