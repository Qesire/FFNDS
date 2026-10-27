"""Fixed UB1 support/alpha/R256; fit one folded consumer per site.

All 24 existing calibration sources are training data; no held-out claim.
"""
import argparse,hashlib,json,os,resource,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='UB1_FIXED_RR_CONSUMER_BANK_20261027'
LIMITS={'BF16_forward_linear':3840,'consumer_delta_fit':40,'persisted_consumer_pack':40,'RR_fit':0,'W1_calls':0,'QDQ_calls':0,'optimizer_steps':0,'video_generations':0,'full_model_calls':0}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(4<<20),b''):h.update(x)
    return h.hexdigest()


def authorize(plan,digest,active,part):
    if plan['id']!=ID or plan['limits_per_part']!=LIMITS or part not in (0,1,2,3):raise ValueError('Fixed part budget mismatch')
    if not plan.get('assets_finalized') or len(plan.get('math_sha256',''))!=64:raise ValueError('Unfrozen asset/math identities')
    if not any(s.get('id')==ID and s.get('protocol_sha256')==digest and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for s in active['experiment_authority']['current_user_authorized_scopes']):raise ValueError('Exact SHA authorization required')


def validate_cases(plan,part):
    if part not in range(4):raise ValueError('Invalid partition')
    blocks=range(part*10,(part+1)*10);expected={(s,b) for s in range(4) for b in blocks}
    chosen=[x for x in plan['positions'] if x['block'] in blocks]
    if len(chosen)!=40 or {(x['step'],x['block']) for x in chosen}!=expected:raise ValueError('Part position coverage mismatch')
    inputs=plan['inputs']
    if len(inputs)!=24 or {x['index'] for x in inputs}!=set(range(24)) or len({x['path'] for x in inputs})!=24:raise ValueError('Expected 24 distinct input manifests')
    return chosen


def expand_inputs(plan,hf,part):
    hf=Path(hf).resolve(); cases=[];source={x['index']:x for x in plan['sources']}
    if len(source)!=24 or set(source)!=set(range(24)) or len({x['source_id'] for x in source.values()})!=24:raise ValueError('Source identities differ')
    for item in plan['inputs']:
        p=(hf/item['path']).resolve()
        if not p.is_relative_to(hf) or sha(p)!=item['sha256']:raise ValueError('Input report SHA/path differs')
        record=json.loads(p.read_text());idx=item['index']
        if record.get('index')!=idx or record.get('status')!='PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP':raise ValueError('Input report identity differs')
        x=record['input']; expected=source[idx]
        if x['input_provenance']['image_sha256']!=expected['source_id'] or x['source_group']!=expected['source_group'] or x['calibration_seed']!=expected['seed']:raise ValueError('Source identity differs')
        scopes=record['scopes']
        if len(scopes)!=160 or {(v['step'],v['block']) for v in scopes}!={(s,b) for s in range(4) for b in range(40)}:raise ValueError('Input scope coverage differs')
        for v in scopes:
            if v['block']//10!=part:continue
            rows=v['selected_rows'];expert='high_noise_model' if v['step']<2 else 'low_noise_model'
            if v['expert']!=expert or v['rows']!=256 or len(rows)!=256 or len(set(rows))!=256 or any(type(i) is not int or i<0 or i>=31668 for i in rows):raise ValueError('Original scope/rows mismatch')
            path=Path(v['path']).resolve()
            if not path.is_relative_to(hf):raise ValueError('Payload escaped HF root')
            cases.append({'index':idx,'step':v['step'],'block':v['block'],'expert':expert,'path':str(path.relative_to(hf)),'sha256':v['sha256'],'bytes':v['bytes']})
    if len(cases)!=960 or len({(c['index'],c['step'],c['block']) for c in cases})!=960:raise ValueError('Part payload coverage differs')
    return cases


def verify_bindings(items,root):
    root=Path(root).resolve()
    if len({x['path'] for x in items})!=len(items):raise ValueError('Duplicate asset binding')
    for item in items:
        p=(root/item['path']).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p)!=item['sha256']:raise ValueError('Bound asset differs')


def validate_position_bindings(pos, private_map):
    for name in ('main_npz','main_json','rr_npz','rr_json','shard'):
        alias=pos[name]
        expected=pos[name+'_sha256']
        actual=private_map.get(alias)
        if actual is None or actual.get('sha256')!=expected:
            raise ValueError('Frozen position asset identity differs: '+name)


def charge(counts,key):
    if counts.get(key,0)>=LIMITS[key]:raise RuntimeError('Budget exceeded before '+key)
    counts[key]=counts.get(key,0)+1


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(memoryview(tensor.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n');os.replace(tmp,path)


def validate_input_scope(c,record,source):
    if record.get('index')!=c['index'] or record.get('status')!='PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP':raise ValueError('Input report identity differs')
    x=record['input']
    if x['input_provenance']['image_sha256']!=source['source_id'] or x['source_group']!=source['source_group'] or x['calibration_seed']!=source['seed']:raise ValueError('Source identity differs')
    scopes=[s for s in record['scopes'] if (s['step'],s['block'])==(c['step'],c['block'])]
    if len(scopes)!=1 or scopes[0]['sha256']!=c['sha256'] or scopes[0]['selected_rows']!=c['rows'] or scopes[0]['rows']!=256 or scopes[0]['expert']!=c['expert']:raise ValueError('Original calibration scope differs')


def baseline_for_source(v1,v2,ids,alpha,weight,bias,rr,linear):
    features=(v2.index_select(1,ids).float()*alpha).bfloat16()
    main=linear(features,weight,bias)
    residual=linear(linear(v1,rr['first'],rr['bias_first']),rr['second'],rr['bias_second'])
    return features,main,residual,main+residual


def run(args):
    plan=json.loads(args.protocol.read_text());digest=sha(args.protocol);authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()),args.part)
    positions=validate_cases(plan,args.part);hf=args.hf_root.resolve();private=args.private_hf_root.resolve()
    verify_bindings(plan['repo_sources'],ROOT);verify_bindings(plan['hf_assets'],hf)
    private_map={x['path']:x for x in plan['private_assets']}
    keys=('main_npz','main_json','main_identity','main_report','rr_npz','rr_json','rr_identity','rr_report','shard')
    needed=set(plan.get('private_common',[]))|{pos[k] for pos in positions for k in keys}
    if not needed<=set(private_map):raise ValueError('Missing part private bindings')
    verify_bindings([private_map[k] for k in sorted(needed)],private)
    if sha(ROOT/'tools/ub1_fixed_rr_consumer_math.py')!=plan['math_sha256']:raise ValueError('Math source differs')
    repo={x['path'] for x in plan['repo_sources']};required={'tools/build_ub1_fixed_rr_consumer_bank.py','tests/test_build_ub1_fixed_rr_consumer_bank.py','tools/ub1_fixed_rr_consumer_math.py','tools/e0_ub1_native_consistency.py','tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py','reports/worker_20261020/UB1_policy.json'}
    if not required<=repo:raise ValueError('Required source binding missing')
    pp={k:v['sha256'] for k,v in private_map.items()}
    source={x['index']:x for x in plan['sources']}
    cases=expand_inputs(plan,hf,args.part)
    for pos in positions:
        validate_position_bindings(pos,private_map)
        for k in ('main_npz','main_json','main_identity','main_report','rr_npz','rr_json','rr_identity','rr_report','shard'):
            if pos[k] not in pp:raise ValueError('Unbound private asset: '+k)
        for prefix in ('main','rr'):
            identity=json.loads((private/pos[prefix+'_identity']).read_text());source_report=json.loads((private/pos[prefix+'_report']).read_text());sidecar=json.loads((private/pos[prefix+'_json']).read_text());expected=pos[prefix+'_fingerprint']
            if identity.get('fingerprint')!=expected or source_report.get('fingerprint')!=expected or sidecar.get('fingerprint')!=expected or sidecar.get('payload_sha256')!=pp[pos[prefix+'_npz']]:raise ValueError('Coefficient report/identity/payload chain differs')
    sys.path.append(str(ROOT/'tools'))
    from worker_identity import require_worker
    require_worker();out=(hf/plan['output_HF_relative']/f'part{args.part}').resolve()
    if out==hf or not out.is_relative_to(hf):raise ValueError('Output escaped HF')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'part':args.part,'protocol_sha256':digest,'attempt':1},f)
    report={'id':ID,'part':args.part,'protocol_sha256':digest,'status':'STARTED','counts':dict.fromkeys(LIMITS,0),'sites':[],'scope':'All 24 exposed sources are training data; gate-only training scores, no held-out, video or speed claim. Preflight outside compute timer; external launcher enforces total wall.'}
    started=time.monotonic();monitor=None
    def check():
        if monitor:monitor.check()
        if shutil.disk_usage(out).free<80<<30:raise RuntimeError('HF free-space reserve')
        if sum(f.stat().st_size for f in out.rglob('*') if f.is_file() and not f.is_symlink())>3<<30:raise RuntimeError('Part output budget')
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        if available<128<<30:raise RuntimeError('Host available memory below reserve')
    def timeout(*_):raise TimeoutError('Part wall budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    try:
        import torch
        import torch.nn.functional as F
        from safetensors.torch import load_file,save_file
        from startup_gpu_reservation import reserve_before_assets
        from e0_ub1_native_consistency import load_coeff
        from ub1_fixed_rr_consumer_math import fit_delta_consumer
        reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt
        if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('One visible A100 required')
        if os.environ.get('NVIDIA_TF32_OVERRIDE')=='1':raise ValueError('TF32 override conflicts with fit')
        torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.set_grad_enabled(False)
        report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'bf16_reduced_precision':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'float32_matmul_precision':torch.get_float32_matmul_precision()}
        snap=out/'EXECUTED_SOURCES';snap.mkdir()
        for x in plan['repo_sources']:
            dst=snap/x['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/x['path'],dst)
        shutil.copyfile(args.protocol,snap/'protocol.json');check()
        def counted(name):charge(report['counts'],name)
        def linear(x,w,b):counted('BF16_forward_linear');return F.linear(x,w,b)
        def gate_stats(pred,target,gate):
            a=(pred*gate).double();b=(target*gate).double();diff=a-b;den=float(b.square().sum());sse=float(diff.square().sum())
            return {'SSE':sse,'target_energy':den,'relative_SSE':sse/den if den else None}
        for pos in sorted(positions,key=lambda x:(x['step'],x['block'])):
            check();step,block=pos['step'],pos['block'];expert='high_noise_model' if step<2 else 'low_noise_model'
            if pos['expert']!=expert:raise ValueError('Expert mismatch')
            raw=torch.load(private/pos['shard'],map_location='cpu',weights_only=True,mmap=True)[expert][f'blocks.{block}.ffn.2'];original=raw['weight_qdq'];original_sha=tensor_sha(original)
            if original_sha!=pos['original_w2_tensor_sha256'] or tuple(original.shape)!=(5120,13824):raise ValueError('Original W2 identity mismatch')
            w=original.to(device='cuda',dtype=torch.bfloat16);data={};gate=None;b2=None
            for index in range(24):
                c=next(v for v in cases if (v['index'],v['step'],v['block'])==(index,step,block));path=hf/c['path']
                if path.stat().st_size!=c['bytes'] or sha(path)!=c['sha256']:raise ValueError('Payload SHA/size differs before load')
                d=load_file(str(path),device='cpu')
                shapes={'V1':(256,5120),'V2':(256,13824),'Y':(256,5120),'gate':(1,5120),'b2':(5120,)}
                if set(d)!=set(shapes) or any(tuple(d[k].shape)!=shape or d[k].dtype!=torch.bfloat16 or not bool(torch.isfinite(d[k]).all()) for k,shape in shapes.items()):raise ValueError('Calibration tensor mismatch')
                d={k:v.cuda() for k,v in d.items()}
                if tensor_sha(d['gate'])!=pos['gate_sha256'] or tensor_sha(d['b2'])!=pos['original_b2_sha256']:raise ValueError('Original gate/b2 differs')
                if gate is None:gate=d['gate'];b2=d['b2']
                data[index]=d
            ids,bias,alpha,rr=load_coeff(private/pos['main_npz'],private/pos['main_json'],private/pos['rr_npz'],private/pos['rr_json'],step,block,'cuda',original_sha,tensor_sha(b2));selected_w=w.index_select(1,ids).contiguous()
            feats=[];errors=[];baseline=[];residuals=[]
            for index in range(24):
                d=data[index];x,base,res,out0=baseline_for_source(d['V1'],d['V2'],ids,alpha,selected_w,bias,rr,linear)
                feats.append(x);errors.append(d['Y'].float()-out0.float());baseline.append(out0);residuals.append(res)
            fitted=fit_delta_consumer(torch.cat(feats),torch.cat(errors),selected_w,bias,counted,fraction=.01);cw,cb=fitted['weight'],fitted['bias']
            if cw.dtype!=torch.bfloat16 or cb.dtype!=torch.bfloat16 or cw.shape!=selected_w.shape or cb.shape!=(5120,) or not bool(torch.isfinite(cw).all()) or not bool(torch.isfinite(cb).all()):raise ValueError('Fitted consumer mismatch')
            metrics=[]
            for index in range(24):
                d=data[index];candidate=linear(feats[index],cw,cb)+residuals[index]
                metrics.append({'index':index,'source_id':source[index]['source_id'],'seed':source[index]['seed'],'baseline':gate_stats(baseline[index],d['Y'],gate),'candidate':gate_stats(candidate,d['Y'],gate)})
            check();projected=(cw.numel()+cb.numel())*2+1048576
            used=sum(f.stat().st_size for f in out.rglob('*') if f.is_file() and not f.is_symlink())
            if used+projected>3<<30 or shutil.disk_usage(out).free-projected<80<<30:raise RuntimeError('Projected output exceeds storage limit')
            counted('persisted_consumer_pack');dest=out/f'step{step}_block{block:02d}.safetensors'
            if dest.exists():raise FileExistsError('Refusing replacement')
            save_file({'weight':cw.detach().cpu().contiguous(),'bias':cb.detach().cpu().contiguous()},str(dest))
            saved=load_file(str(dest),device='cpu')
            if set(saved)!={'weight','bias'} or tensor_sha(saved['weight'])!=tensor_sha(cw) or tensor_sha(saved['bias'])!=tensor_sha(cb):raise ValueError('Persisted consumer bytes differ')
            del saved
            result={'step':step,'block':block,'expert':expert,'source_shard_sha256':pp[pos['shard']],'original_w2_tensor_sha256':original_sha,'original_b2_sha256':tensor_sha(b2),'gate_sha256':tensor_sha(gate),'alpha':alpha,'group_ids':(ids.reshape(-1,64)[:,0]//64).tolist(),'original_RR_tensor_sha256':{k:tensor_sha(v) for k,v in rr.items()},'weight_tensor_sha256':tensor_sha(cw),'bias_tensor_sha256':tensor_sha(cb),'file':dest.name,'file_sha256':sha(dest),'bytes':dest.stat().st_size,'diagnostics':fitted['diagnostics'],'training_sources':metrics}
            report['sites'].append(result);write(out/'RUN.json',report);check()
            del raw,original,w,data,gate,b2,ids,bias,rr,selected_w,feats,errors,baseline,residuals,fitted,cw,cb
        if report['counts']!=LIMITS:raise RuntimeError('Part call ledger incomplete')
        report.update(status='COMPLETE_TRAINING_CONSUMER_BANK_PART',elapsed_seconds=time.monotonic()-started)
        write(out/'MANIFEST.json',{'id':ID,'part':args.part,'protocol_sha256':digest,'sites':report['sites']})
    except BaseException as e:
        report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0)
        if monitor:
            monitor.close();report['assigned_gpu_resources']=monitor.report()
            if report['assigned_gpu_resources']['violations']:report['status']='FAILED_GPU_MONITOR'
        write(out/'RUN.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--part',type=int,required=True);p.add_argument('--hf-root',type=Path,required=True);p.add_argument('--private-hf-root',type=Path,required=True);run(p.parse_args())
