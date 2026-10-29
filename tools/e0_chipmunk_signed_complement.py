"""Draft: one true-raster full-M source/expert pair, separately registered."""
import argparse,hashlib,importlib.metadata,importlib.util,json,os,pathlib,resource,shutil,signal,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]

def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def fp(p):
    s=p.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
def atomic(path,data):
    b=json.dumps(data,indent=2,allow_nan=False).encode();need(len(b)<(16<<20),'report limit');tmp=path.with_suffix('.tmp');tmp.write_bytes(b);os.replace(tmp,path)
def reserve_counts(existing,counts,limits):
    need(all(k in limits and isinstance(v,int) and v>=0 and existing.get(k,0)+v<=limits[k] for k,v in counts.items()),'operation budget')
    return {**existing,**{k:existing.get(k,0)+v for k,v in counts.items()}}

def authorize(p,digest,cpu=False):
    need(p['assets_finalized'] is True and p['id'].startswith('E0_CHIPMUNK_SIGNED_COMPLEMENT_20261030_PART'),'unfinalized draft')
    a=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text());s=[v for v in a['experiment_authority']['current_user_authorized_scopes'] if v.get('id')==(p['id']+'_CPU_PREFLIGHT' if cpu else p['id'])];need(a['user_continuation_authorization_20261027']['new_scope_registration_allowed'] and len(s)==1,'authority');s=s[0]
    need(s.get('execution_status')=='RUNNING_FIXED_SCOPE' and s.get('protocol_sha256')==digest and s.get('max_gpu_cards')==(0 if cpu else 1) and s.get('resources')==p['cpu_preflight_resources' if cpu else 'resources'] and s.get('limits')==p['cpu_preflight_limits' if cpu else 'operation_limits'],'scope mismatch')
    ref=s['review'];rp=(ROOT/ref['path']).resolve();need(rp.is_relative_to(ROOT) and rp.stat().st_size==ref['bytes'] and sha(rp)==ref['sha256'],'review');r=json.loads(rp.read_text());need(r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('scope_id')==p['id'] and r.get('protocol_sha256')==digest and r.get('cpu_job_spec_sha256' if cpu else 'job_spec_sha256')==s.get('launch_spec_sha256') and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']},'review identity');return s

def bound_hf(hf,ref):
    q=(hf/ref['path']).resolve();need(q.is_relative_to(hf.resolve()) and q.stat().st_size==ref['bytes'] and sha(q)==ref['sha256'],'receipt identity');return json.loads(q.read_text())

def verify_cpu_receipts(a,p,digest,scope):
    refs=scope['cpu_preflight'];r=bound_hf(a.hf_root,refs['RUN']);e=bound_hf(a.hf_root,refs['EXIT']);review=json.loads((ROOT/scope['review']['path']).read_text())
    need(refs['job_spec_sha256']==review['cpu_job_spec_sha256'] and r.get('id')==p['id']+'_CPU_PREFLIGHT' and r.get('status')=='PASS_ISOLATED_ENTRY_CPU_PREFLIGHT' and r.get('protocol_sha256')==digest and r.get('cuda_initialized') is False and r.get('gpu_job_sha256')==review['job_spec_sha256'] and e.get('id')==p['id']+'_CPU_PREFLIGHT' and e.get('protocol_sha256')==digest and e.get('child_returncode')==0 and e.get('process_exited') is True and e.get('run_sha256')==refs['RUN']['sha256'] and e.get('spec_sha256')==r.get('job_spec_sha256')==refs['job_spec_sha256'],'CPU preflight not closed')

def cpu_preflight(a,p,digest,scope,job,profile):
    out=(a.hf_root/p['cpu_output_HF_relative']).resolve();need(out.is_relative_to(a.hf_root.resolve()) and not out.exists(),'CPU output identity')
    expected=['env','FFNDS_WORKER_CONFIG='+profile,'CUDA_VISIBLE_DEVICES=','taskset','-c',','.join(map(str,job['cpu_affinity'])),str(ROOT/'.venv/bin/python'),'-I','-B']+sys.argv
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and job['argv']==expected and pathlib.Path(job['output_dir']).resolve()==out.parent and pathlib.Path(job['run_json']).resolve()==out/'RUN.json' and job['external_timeout_seconds']==p['cpu_preflight_resources']['outer_seconds'],'CPU job boundary')
    out.mkdir(parents=True);r={'id':p['id']+'_CPU_PREFLIGHT','protocol_sha256':digest,'job_spec_sha256':sha(a.job_spec),'status':'ATTEMPT','automatic_retry':False};atomic(out/'ATTEMPT.json',r)
    old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU preflight')));signal.alarm(p['cpu_preflight_resources']['wall_seconds'])
    try:
        r.update(source_hash_predebit_bytes=sum(x['bytes'] for x in p['repo_sources']),source_hash_completed_bytes=0,source_hash_inflight='UNKNOWN');atomic(out/'RUN.json',r)
        for b in p['repo_sources']:
            q=(ROOT/b['path']).resolve();need(q.is_relative_to(ROOT) and q.stat().st_size==b['bytes'] and sha(q)==b['sha256'],'source');r['source_hash_completed_bytes']+=b['bytes'];atomic(out/'RUN.json',r)
        r['source_hash_inflight']='NONE';review=json.loads((ROOT/scope['review']['path']).read_text());gp=(ROOT/p['gpu_job_private_path']).resolve();need(gp.is_relative_to(ROOT) and sha(gp)==review['job_spec_sha256'],'GPU job SHA');g=json.loads(gp.read_text())
        driver=[str(ROOT/'tools/e0_chipmunk_signed_complement.py'),'--protocol',str(a.protocol.resolve()),'--hf-root',str(a.hf_root.resolve()),'--job-spec',str(gp)];expected_gpu=['env','FFNDS_WORKER_CONFIG='+profile,'bash',str(ROOT/g['gpu_launcher']),str(g['physical_gpu']),g['gpu_uuid'],'taskset','-c',','.join(map(str,g['cpu_affinity'])),str(ROOT/'.venv/bin/python'),'-I','-B']+driver
        need(g['argv']==expected_gpu and g['driver_argv']==driver and g['protocol_sha256']==digest and g['physical_gpu']==p['physical_gpu'] and g['external_timeout_seconds']==p['resources']['outer_seconds'],'GPU job boundary')
        r.update(environment_predebit_bytes=p['read_budgets']['environment_logical_bytes'],environment_completed_bytes=0,environment_inflight='UNKNOWN');atomic(out/'RUN.json',r)
        sys.path.insert(0,str(ROOT/'tools'));spec=importlib.util.spec_from_file_location('packed_cpu_environment',ROOT/'tools/e0_chipmunk_paid_proxy_probe.py');legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy);need(legacy.environment_snapshot(sys.prefix)==p['prepared_environment'],'CPU environment mismatch')
        nv=pathlib.Path(shutil.which('nvidia-smi')).resolve();need(str(nv)==p['monitor_binary']['path'] and nv.stat().st_size==p['monitor_binary']['bytes'] and sha(nv)==p['monitor_binary']['sha256'],'CPU monitor binary');r.update(environment_completed_bytes=p['read_budgets']['environment_logical_bytes'],environment_inflight='NONE',metadata_discovery_bytes='UNKNOWN_ADDITIONAL');atomic(out/'RUN.json',r)
        import torch
        need(not torch.cuda.is_initialized() and 'out_dtype' in (torch.mm.__doc__ or ''),'torch mm API or CUDA state');r.update(status='PASS_ISOLATED_ENTRY_CPU_PREFLIGHT',gpu_job_sha256=review['job_spec_sha256'],torch_imported=True,cuda_initialized=False,torch_version=torch.__version__,out_dtype_api_documented=True,device_execution='NOT_TESTED');atomic(out/'RUN.json',r)
    except BaseException as e:
        r.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());atomic(out/'RUN.json',r);raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)

def run(a):
    p=json.loads(a.protocol.read_text());digest=sha(a.protocol);scope=authorize(p,digest,a.preflight);need(sha(a.job_spec)==scope['launch_spec_sha256'] and sys.flags.isolated==1,'job/isolated');job=json.loads(a.job_spec.read_text());need(job['driver_argv']==sys.argv and job['protocol_sha256']==digest and job['id']==(p['id']+'_CPU_PREFLIGHT' if a.preflight else p['id']) and pathlib.Path(job['cwd']).resolve()==ROOT and pathlib.Path(job['hf_root']).resolve()==a.hf_root.resolve() and sorted(os.sched_getaffinity(0))==job['cpu_affinity'],'job identity');need(all(importlib.metadata.version(k)==v for k,v in p['runtime_versions'].items()),'runtime versions')
    def module(name,path):
        s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
    ref=next(z for z in p['repo_sources'] if z['path']=='tools/worker_identity.py');need(sha(ROOT/ref['path'])==ref['sha256'],'worker source');worker=module('worker_identity','tools/worker_identity.py');worker.require_worker();need(sha(worker.private_config_path())==p['worker_config_sha256'],'worker profile')
    profile=str(worker.private_config_path().resolve())
    if a.preflight:return cpu_preflight(a,p,digest,scope,job,profile)
    verify_cpu_receipts(a,p,digest,scope)
    expected=['env','FFNDS_WORKER_CONFIG='+profile,'bash',str(ROOT/job['gpu_launcher']),str(job['physical_gpu']),job['gpu_uuid'],'taskset','-c',','.join(map(str,job['cpu_affinity'])),str(ROOT/'.venv/bin/python'),'-I','-B']+sys.argv;need(job['argv']==expected and job['gpu_launcher']=='scripts/worker_teal_assigned_uuid_exec.sh' and job['physical_gpu']==p['physical_gpu'] and os.environ.get('CUDA_VISIBLE_DEVICES')==job['gpu_uuid'] and os.environ.get('FFNDS_PHYSICAL_GPU')==str(job['physical_gpu']),'GPU launcher')
    out=(a.hf_root/p['output_HF_relative']).resolve();need(out.is_relative_to(a.hf_root.resolve()) and not out.exists() and pathlib.Path(job['output_dir']).resolve()==out.parent and pathlib.Path(job['run_json']).resolve()==out/'RUN.json' and job['external_timeout_seconds']==p['resources']['outer_seconds'],'output/job');out.mkdir(parents=True);need(shutil.disk_usage(out).free>=p['resources']['disk_free_min_bytes'],'disk')
    r={'id':p['id'],'protocol_sha256':digest,'status':'ATTEMPT','predebited':{},'completed':{},'inflight':None,'timings':{},'automatic_retry':False};atomic(out/'ATTEMPT.json',r);start=time.monotonic();monitor=None;hold=None
    def persist():atomic(out/'RUN.json',r)
    def predebit(counts):
        authorize(p,digest);need(time.monotonic()-start<p['resources']['wall_seconds'],'deadline');monitor.check();need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<p['resources']['host_peak_rss_bytes'] and torch.cuda.max_memory_reserved()<p['resources']['gpu_reserved_bytes'],'memory');need(sum(f.stat().st_size for f in out.parent.rglob('*') if f.is_file() and not f.is_symlink())<=p['resources']['output_bytes_max'],'output budget')
        r['predebited']=reserve_counts(r['predebited'],counts,p['operation_limits'])
        r['inflight']=counts;persist()
    def complete(counts):
        for k,v in counts.items():r['completed'][k]=r['completed'].get(k,0)+v
        r['inflight']=None;persist()
    def call(name,counts,fn):
        predebit(counts);t=time.monotonic();value=fn();torch.cuda.synchronize();r['timings'][name]={'wall_seconds_including_sync':time.monotonic()-t};complete(counts);return value
    def measured(name,counts,fn):
        predebit(counts);torch.cuda.synchronize();a0=torch.cuda.Event(enable_timing=True);a1=torch.cuda.Event(enable_timing=True);start_cpu=time.perf_counter();a0.record();value=fn();a1.record();a1.synchronize();wall=time.perf_counter()-start_cpu;r['timings'][name]={'cuda_event_ms':a0.elapsed_time(a1),'synchronized_host_wall_seconds':wall,'ledger_and_persist_outside_interval':True};complete(counts);return value
    old_handler=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('scope timeout')));signal.alarm(p['resources']['wall_seconds'])
    try:
        r['source_hash_predebit_bytes']=sum(z['bytes'] for z in p['repo_sources']);r['source_hash_completed_bytes']=0;r['source_hash_inflight']='UNKNOWN';persist()
        for z in p['repo_sources']:
            q=(ROOT/z['path']).resolve();need(q.is_relative_to(ROOT) and q.stat().st_size==z['bytes'] and sha(q)==z['sha256'],'source');r['source_hash_completed_bytes']+=z['bytes'];persist()
        r['source_hash_inflight']='NONE';r['backend_hash_predebit_bytes']=sum(z['bytes'] for z in p['backend_files']);r['backend_hash_completed_bytes']=0;r['backend_hash_inflight']='UNKNOWN';persist();backend_stats={}
        for z in p['backend_files']:
            q=(ROOT/z['path']).resolve();st=fp(q);need(q.is_relative_to(ROOT) and q.stat().st_size==z['bytes'] and sha(q)==z['sha256'] and fp(q)==st,'backend');backend_stats[q]=st;r['backend_hash_completed_bytes']+=z['bytes'];persist()
        r['backend_hash_inflight']='NONE';r['environment_predebit_bytes']=p['read_budgets']['environment_logical_bytes'];r['environment_completed_bytes']=0;r['environment_inflight']='UNKNOWN';persist();sys.path.insert(0,str(ROOT/'tools'));legacy=module('fullm_environment_helper','tools/e0_chipmunk_paid_proxy_probe.py');need(legacy.environment_snapshot(sys.prefix)==p['prepared_environment'],'environment');nv=pathlib.Path(shutil.which('nvidia-smi')).resolve();need(str(nv)==p['monitor_binary']['path'] and nv.stat().st_size==p['monitor_binary']['bytes'] and sha(nv)==p['monitor_binary']['sha256'],'monitor executable');r['environment_completed_bytes']=p['read_budgets']['environment_logical_bytes'];r['environment_inflight']='NONE';r['environment_metadata_discovery_bytes']='UNKNOWN_ADDITIONAL';persist()
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TRITON_CACHE_DIR=str(out/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'))
        import socket
        def deny(*args,**kwargs):raise RuntimeError('network denied')
        socket.socket.connect=deny;socket.create_connection=deny
        import torch
        from safetensors import safe_open
        sys.path.insert(0,str(ROOT/'tools'));import startup_gpu_reservation
        from chipmunk_signed_complement_math import p64_observer,checked_mask,packed,split_dc_ac
        from chipmunk_packed_native_math import packed_action
        from chipmunk_fullm_proxy_math import fixed_prefix_permutation
        import torch.nn.functional as F
        torch.set_num_threads(p['resources']['cpu_threads']);torch.backends.cuda.matmul.allow_tf32=False;need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0) and 'A100' in torch.cuda.get_device_name(0),'A100 only');hold,monitor,r['reservation']=startup_gpu_reservation.reserve_before_assets(torch);r['precision']={'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0),'tf32':torch.backends.cuda.matmul.allow_tf32,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'float32_matmul_precision':torch.get_float32_matmul_precision()};persist()
        prior={}
        for ref in p['prior_receipts']:
            prior[ref['kind']]=call('prior_'+ref['kind'],{'prior_hash':1},lambda ref=ref:bound_hf(a.hf_root,ref))
        need(prior['RUN']['id']==p['prior_id'] and prior['RUN']['status']=='COMPLETE_DIAGNOSTIC' and prior['exit']['id']==p['prior_id'] and prior['exit']['protocol_sha256']==prior['RUN']['protocol_sha256'] and prior['exit']['child_returncode']==0 and prior['exit']['process_exited'] and prior['exit']['run_sha256']==next(z['sha256'] for z in p['prior_receipts'] if z['kind']=='RUN') and prior['RUN']['metrics']['arms']['p64']['frozen_indices_sha256']==p['p64_mask_sha256'],'prior P64 identity')
        parent={}
        for ref in p['parent_receipts']:
            q=(a.hf_root/ref['path']).resolve();call('parent_'+ref['kind'],{'parent_hash':1},lambda q=q,ref=ref:need(q.is_relative_to(a.hf_root.resolve()) and q.stat().st_size==ref['bytes'] and sha(q)==ref['sha256'],'parent identity'));parent[ref['kind']]=json.loads(q.read_text())
        need(parent['RUN']['id']==parent['exit']['id']==p['parent_id'] and parent['RUN']['source_ordinal']==p['source'] and parent['RUN']['status']=='COMPLETE_CHIPMUNK_FULLM_RASTER_EXIT_PENDING' and parent['RUN']['protocol_sha256']==parent['exit']['protocol_sha256']==p['parent_protocol_sha256'] and parent['verification_RUN']['parent_scope_id']==p['parent_id'] and parent['verification_RUN']['predebited_bytes']==parent['verification_RUN']['completed_bytes'] and parent['verification_RUN']['inflight']=='NONE' and parent['exit']['child_returncode']==0 and parent['exit']['process_exited'] and parent['exit']['run_sha256']==next(z['sha256'] for z in p['parent_receipts'] if z['kind']=='RUN') and parent['verification_exit']['child_returncode']==0 and parent['verification_exit']['process_exited'] and parent['verification_exit']['run_sha256']==next(z['sha256'] for z in p['parent_receipts'] if z['kind']=='verification_RUN') and parent['verification_RUN']['status']=='PASS_CLOSED_CHIPMUNK_FULLM_RASTER','parent completion')
        manifest=json.loads((ROOT/p['manifest']['path']).read_text());parentpart=next(z for z in manifest['parts'] if z['source_ordinal']==p['source']);expected_captures=[v for v in parentpart['captures'] if v['step'] in p['steps']];need(expected_captures==p['captures'] and [v['step'] for v in p['captures']]==p['steps'] and all(v['block']==20 and v['expert']==p['expert'] for v in p['captures']),'capture manifest')
        captures={}
        for item in p['captures']:
            hdr=item['header'];q=(a.hf_root/hdr['path']).resolve();call('header_'+str(item['step']),{'header_hash':1},lambda q=q,hdr=hdr:need(q.is_relative_to(a.hf_root.resolve()) and q.stat().st_size==hdr['bytes'] and sha(q)==hdr['sha256'],'header'))
            for key,b in item['operands'].items():
                q=(a.hf_root/b['file']['path']).resolve();st=fp(q);call('hash_'+str(item['step'])+'_'+key,{'operand_file_hash':1},lambda q=q,b=b,st=st:need(q.is_relative_to(a.hf_root.resolve()) and q.stat().st_size==b['file']['bytes'] and sha(q)==b['file']['sha256'] and fp(q)==st,'operand identity'));captures[(item['step'],key)]=(q,st,b)
        cp=pathlib.Path(job['checkpoint']).resolve();cst=fp(cp);call('checkpoint_hash',{'checkpoint_hash':1},lambda:need(cp.stat().st_size==p['checkpoint']['bytes'] and sha(cp)==p['checkpoint']['sha256'] and fp(cp)==cst,'checkpoint'))
        def weights():
            with safe_open(str(cp),framework='pt',device='cpu') as f:return [f.get_tensor('blocks.20.ffn.'+k).contiguous().cuda() for k in ('0.weight','0.bias','2.weight','2.bias')]
        w1,b1,w2,b2=call('weights_load',{'checkpoint_tensor_load':1},weights);need(fp(cp)==cst and [tuple(t.shape) for t in (w1,b1,w2,b2)]==[(13824,5120),(13824,),(5120,13824),(5120,)] and all(t.dtype==torch.bfloat16 for t in (w1,b1,w2,b2)),'weight identity')
        call('weights_finite',{'checkpoint_finite':1},lambda:need(all(bool(torch.isfinite(t).all()) for t in (w1,b1,w2,b2)),'nonfinite weights'))
        def load(step,key):
            q,st,b=captures[step,key];need(fp(q)==st,'operand changed');
            with safe_open(str(q),framework='pt',device='cpu') as f:t=f.get_tensor(key);need(list(t.shape)==b['shape'] and t.dtype==torch.bfloat16,'operand shape');v=t.contiguous().cuda()
            need(fp(q)==st,'operand changed in read');return v
        step0,step1=p['steps'];x0=call('previous_input',{'operand_tensor_load':1},lambda:load(step0,'V1'));x1=call('current_input',{'operand_tensor_load':1},lambda:load(step1,'V1'));gate=call('current_gate',{'operand_tensor_load':1},lambda:load(step1,'gate')).reshape(5120);call('inputs_finite',{'operand_finite':1},lambda:need(all(bool(torch.isfinite(v).all()) for v in (x0,x1,gate)),'nonfinite input'))
        perm=call('fixed_seeds',{'fixed_selection_init':1},lambda:fixed_prefix_permutation(31668,42));need(perm[:4].tolist()==p['sample_prefix4'] and perm.tolist()==p['sample_prefix64'],'sample seed mismatch')
        def dense(x):
            pre=F.linear(x,w1,b1);h=F.gelu(pre,approximate='tanh');y=F.linear(h,w2,b2);return h,y,(y*gate).to(torch.bfloat16)
        old_h,old_y,_=measured('old_cache',{'dense_w1':1,'gelu':1,'dense_w2':1,'gate_product':1},lambda:dense(x0))
        guards=[];masks=[]
        def action(corrected):
            mask,mean=p64_observer(x1,w1,b1,old_h,perm,5120);guards.append(checked_mask(mask,p['p64_mask_sha256']));masks.append(mask)
            return packed(x1,w1,b1,w2,old_h,old_y,gate,mask,mean if corrected else None)
        base={'observer_w1':1,'observer_gelu':1,'observer_rms_mean':1,'selector':1,'mask_guard':1,'clone_h':1,'clone_y':1,'gather_w1':1,'gather_bias':1,'gather_w2':1,'gather_h':1,'selected_w1':1,'selected_gelu':1,'delta_round':1,'update_h':1,'selected_w2_fp32':1,'accumulate_y':1,'gate_product':1}
        outs={}
        for mode in ('baseline','signed'):
            debit=dict(base)
            if mode=='signed':debit.update(complement_zero=1,gemv_fp32=1,broadcast_add=1)
            measured('cold_'+mode,debit,lambda mode=mode:action(mode=='signed'))
            outs[mode]=[measured('online_'+mode+'_'+str(i),debit,lambda mode=mode:action(mode=='signed')) for i in range(3)]
        reference_debit={'baseline_reference_action':1,'clone_h':1,'clone_y':1,'gather_w1':1,'gather_bias':1,'gather_w2':1,'gather_h':1,'selected_w1':1,'selected_gelu':1,'delta_round':1,'update_h':1,'selected_w2_fp32':1,'accumulate_y':1,'gate_product':1}
        baseline_reference=measured('baseline_same_mask_reference',reference_debit,lambda:packed_action(x1,w1,b1,w2,old_h,old_y,gate,masks[-1]))
        measured('dense_cold',{'dense_w1':1,'gelu':1,'dense_w2':1,'gate_product':1},lambda:dense(x1))
        dense_runs=[measured('dense_'+str(i),{'dense_w1':1,'gelu':1,'dense_w2':1,'gate_product':1},lambda:dense(x1)) for i in range(3)]
        current_h,current_y,current_gated=dense_runs[-1]
        def hindsight():
            mean=(current_h.float()-old_h.float()).mean(0);return packed(x1,w1,b1,w2,old_h,old_y,gate,masks[-1],mean)
        oracle_debit={'full_mean':1,'clone_h':1,'clone_y':1,'gather_w1':1,'gather_bias':1,'gather_w2':1,'gather_h':1,'selected_w1':1,'selected_gelu':1,'delta_round':1,'update_h':1,'selected_w2_fp32':1,'accumulate_y':1,'gate_product':1,'complement_zero':1,'gemv_fp32':1,'broadcast_add':1}
        outs['hindsight']=[measured('hindsight_paid',oracle_debit,hindsight)]
        def stats():
            den=float(torch.linalg.vector_norm(current_gated.float()));need(den>0,'zero dense reference');arms={}
            for name,ys in outs.items():
                arms[name]=[float(torch.linalg.vector_norm(y[1].float()-current_gated.float()))/den for y in ys]
            dr=outs['baseline'][-1][1].float()-baseline_reference.float();rn=float(torch.linalg.vector_norm(baseline_reference.float()));direct={'relative_l2':float(torch.linalg.vector_norm(dr))/rn if rn else None,'absolute_max':float(dr.abs().max()),'numeric_equal':bool(torch.equal(outs['baseline'][-1][1],baseline_reference))};r['partial_direct_baseline_reference']=direct;persist();need(direct['relative_l2'] is not None and direct['relative_l2']<=p['baseline_reference_rtol'],'same-mask baseline direct mismatch')
            priorv=prior['RUN']['metrics']['arms']['p64']['gated_relative_l2_repeats'];r['partial_baseline_replay']={'previous':priorv,'observed':arms['baseline'],'absolute_tolerance':p['prior_baseline_rtol']};persist();need(len(priorv)==3 and max(abs(a-b) for a,b in zip(priorv,arms['baseline']))<=p['prior_baseline_rtol'],'prior P64 numerical replay')
            decomp=split_dc_ac(outs['baseline'][-1][1].float()-current_gated.float())
            return {'gated_relative_l2':arms,'baseline_error_dc_ac_fp64':decomp,'same_mask_baseline_reference':direct,'mask_sha256':p['p64_mask_sha256'],'prior_p64_gated_relative_l2':priorv,'mask_guard_host_seconds':guards,'recipe':'sample BF16 GELU minus cached BF16 h in FP32; 64-row FP32 mean; selected columns zero; cast complement BF16; BF16 GEMV FP32 output; add old Y and selected delta in FP32; one BF16 output cast then current BF16 gate'}
        r['metrics']=call('metrics',{'metrics':1},stats);need(r['predebited']==r['completed']==p['operation_limits'] and r['inflight'] is None,'ledger incomplete');maps=pathlib.Path('/proc/self/maps').read_text().splitlines();need(all(any(str(q) in line and not line.endswith(' (deleted)') for line in maps) and fp(q)==st for q,st in backend_stats.items()),'mapped backend');r['backend_mapping_verified']=[b['path'] for b in p['backend_files']];r.update(status='COMPLETE_DIAGNOSTIC',wall_seconds=time.monotonic()-start,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    except BaseException as e:
        r.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());persist();raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler);primary=sys.exc_info()[0] is not None;err=None
        if monitor:
            try:
                monitor.close();r['monitor']=monitor.report();need(not r['monitor']['violations'],'monitor violation')
            except BaseException as e:err=e;r.update(status='FAILED_GPU_MONITOR',monitor_error=type(e).__name__)
        persist()
        if err and not primary:raise err

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=pathlib.Path,required=True);p.add_argument('--hf-root',type=pathlib.Path,required=True);p.add_argument('--job-spec',type=pathlib.Path,required=True);p.add_argument('--preflight',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
