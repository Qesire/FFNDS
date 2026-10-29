"""P4 draft: four fixed causal-proxy pairs. Not authorized until independent review."""
import argparse, hashlib, importlib.metadata, importlib.util, json, os, pathlib, resource, shutil, signal, sys, time, traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
ID='E0_CHIPMUNK_P4_CAUSAL_PROXY_V2_20261030'
OUTPUT='ffnds-derived/e0_chipmunk_p4_causal_proxy_v2_20261030/attempt1/worker/results'

def need(x,msg):
    if not x: raise ValueError(msg)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def fingerprint(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def write(path,obj):
    b=json.dumps(obj,indent=2,allow_nan=False).encode();need(len(b)<=16<<20,'report limit');tmp=path.with_suffix('.tmp');tmp.write_bytes(b);os.replace(tmp,path)
def authorize(plan,digest,cpu=False):
    need(plan.get('id')==ID and plan.get('assets_finalized') is True,'P4 draft not executable')
    a=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text());scopes=[s for s in a['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==(ID+'_CPU_PREFLIGHT' if cpu else ID)]
    need(a['user_continuation_authorization_20261027']['new_scope_registration_allowed'] and len(scopes)==1,'authorization')
    s=scopes[0];need(s.get('protocol_sha256')==digest and s.get('execution_status')=='RUNNING_FIXED_SCOPE' and s.get('max_gpu_cards')==(0 if cpu else 1) and ((s.get('limits')==plan['cpu_preflight_limits'] and s.get('resources')==plan['cpu_preflight_resources']) if cpu else (s.get('limits')==plan['operation_limits'] and s.get('resources')==plan['resources'])),'registered scope mismatch')
    ref=s['review'];p=(ROOT/ref['path']).resolve();need(p.is_relative_to(ROOT) and p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],'review mismatch');r=json.loads(p.read_text())
    need(r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('scope_id')==ID and r.get('protocol_sha256')==digest and r.get('cpu_job_spec_sha256' if cpu else 'job_spec_sha256')==s.get('launch_spec_sha256') and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('critical_source_hashes')=={x['path']:x['sha256'] for x in plan['repo_sources']},'review gate')
    return s

def bound_hf(hf, ref):
    p=(hf/ref['path']).resolve();need(p.is_relative_to(hf.resolve()) and p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],'preflight receipt identity');return json.loads(p.read_text())

def verify_cpu_receipts(scope,plan,digest):
    refs=scope['cpu_preflight'];hf=pathlib.Path('/home/xiang/.cache/huggingface');r=bound_hf(hf,refs['RUN']);e=bound_hf(hf,refs['EXIT']);review=json.loads((ROOT/scope['review']['path']).read_text());need(refs['job_spec_sha256']==review['cpu_job_spec_sha256'],'CPU review job identity')
    need(r.get('id')==ID+'_CPU_PREFLIGHT' and r.get('status')=='PASS_ISOLATED_ENTRY_CPU_PREFLIGHT' and r.get('protocol_sha256')==digest and r.get('torch_imported') is False and e.get('child_returncode')==0 and e.get('process_exited') is True and e.get('run_sha256')==refs['RUN']['sha256'] and e.get('spec_sha256')==r.get('job_spec_sha256')==refs['job_spec_sha256'] and e.get('protocol_sha256')==digest and e.get('id')==ID+'_CPU_PREFLIGHT','CPU preflight not closed')

def cpu_preflight(a,plan,digest,scope,job,profile,module):
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight requires CUDA hidden');soft,hard=resource.getrlimit(resource.RLIMIT_FSIZE);need(hard==resource.RLIM_INFINITY or hard>=4194304,'file size hard limit');resource.setrlimit(resource.RLIMIT_FSIZE,(4194304,hard))
    out=(a.hf_root/'ffnds-derived/e0_chipmunk_p4_causal_proxy_v2_20261030/cpu_preflight/worker/results').resolve()
    expected=['env','FFNDS_WORKER_CONFIG='+profile,'CUDA_VISIBLE_DEVICES=','taskset','-c','0,1,2,3',str(ROOT/'.venv/bin/python'),'-I','-B']+sys.argv
    need(job['argv']==expected and job['cpu_affinity']==[0,1,2,3] and pathlib.Path(job['output_dir']).resolve()==out.parent and pathlib.Path(job['run_json']).resolve()==out/'RUN.json' and job['external_timeout_seconds']==300,'CPU job boundary')
    need(not out.exists(),'exclusive CPU preflight');out.mkdir(parents=True)
    report={'id':ID+'_CPU_PREFLIGHT','protocol_sha256':digest,'job_spec_sha256':sha(a.job_spec),'status':'ATTEMPT','torch_imported':('torch' in sys.modules),'automatic_retry':False,'predebited':{'preflight':1},'completed':{'preflight':0}};write(out/'ATTEMPT.json',report)
    old_handler=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU preflight deadline')));signal.alarm(plan['cpu_preflight_resources']['wall_seconds'])
    try:
        need('torch' not in sys.modules,'unexpected preflight torch import')
        review=json.loads((ROOT/scope['review']['path']).read_text());gpu_job_path=ROOT/'.runtime/chipmunk_p4_causal_proxy_v2_20261030/GPU_JOB.json';report['preflight_bound_metadata_predebit_bytes']=plan['previous_failure_exit']['bytes']+gpu_job_path.stat().st_size;report['preflight_bound_metadata_completed_bytes']=0;report['preflight_bound_metadata_inflight']='UNKNOWN';write(out/'RUN.json',report);need(sha(gpu_job_path)==review['job_spec_sha256'],'GPU job SHA');gpu=json.loads(gpu_job_path.read_text());gpu_driver=[str(ROOT/'tools/e0_chipmunk_p4_causal_proxy_v2.py'),'--protocol',str(a.protocol.resolve()),'--hf-root',str(a.hf_root.resolve()),'--job-spec',str(gpu_job_path)];expected_gpu=['env','FFNDS_WORKER_CONFIG='+profile,'bash',str(ROOT/'scripts/worker_teal_gpu0_uuid_exec.sh'),'taskset','-c','0,1,2,3',str(ROOT/'.venv/bin/python'),'-I','-B']+gpu_driver;need(gpu['argv']==expected_gpu and gpu['driver_argv']==gpu_driver and gpu['protocol_sha256']==digest and gpu['cpu_affinity']==[0,1,2,3] and gpu['external_timeout_seconds']==plan['resources']['outer_seconds'],'GPU job shape');report['gpu_job_sha256']=review['job_spec_sha256'];write(out/'RUN.json',report)
        old=bound_hf(a.hf_root,plan['previous_failure_exit']);need(old.get('id')=='E0_CHIPMUNK_P4_CAUSAL_PROXY_20261030' and old.get('protocol_sha256')==plan['previous_failure_protocol_sha256'] and old.get('child_returncode')==1 and old.get('process_exited') and old.get('run_sha256') is None,'old failure identity');report['previous_failure_exit_verified']=True;report['preflight_bound_metadata_completed_bytes']=report['preflight_bound_metadata_predebit_bytes'];report['preflight_bound_metadata_inflight']='NONE';write(out/'RUN.json',report)
        report['source_hash_predebit_bytes']=sum(b['bytes'] for b in plan['repo_sources']);report['source_hash_completed_bytes']=0;report['source_hash_inflight']='UNKNOWN';write(out/'RUN.json',report)
        for b in plan['repo_sources']:
            p=(ROOT/b['path']).resolve();need(p.is_relative_to(ROOT) and p.stat().st_size==b['bytes'] and sha(p)==b['sha256'],'source mismatch');report['source_hash_completed_bytes']+=b['bytes'];write(out/'RUN.json',report)
        report['source_hash_inflight']='NONE';report['environment_predebit_bytes']=plan['read_budgets']['environment_logical_bytes'];report['environment_completed_bytes']=0;report['environment_inflight']='UNKNOWN';write(out/'RUN.json',report)
        sys.path.insert(0,str(ROOT/'tools'));legacy=module('p4_v2_cpu_legacy','tools/e0_chipmunk_paid_proxy_probe.py');need(legacy.environment_snapshot(sys.prefix)==plan['prepared_environment'],'environment mismatch');n=pathlib.Path(shutil.which('nvidia-smi')).resolve();need(str(n)==plan['monitor_binary']['path'] and n.stat().st_size==plan['monitor_binary']['bytes'] and sha(n)==plan['monitor_binary']['sha256'],'monitor binary mismatch')
        report['completed']['preflight']=1;report.update(status='PASS_ISOLATED_ENTRY_CPU_PREFLIGHT',environment_completed_bytes=plan['read_budgets']['environment_logical_bytes'],environment_inflight='NONE',torch_imported=('torch' in sys.modules),cuda_initialized='NOT_IMPORTED',metadata_discovery_bytes='UNKNOWN_ADDITIONAL');need(not report['torch_imported'],'preflight imported torch');write(out/'RUN.json',report)
    except BaseException as e:
        report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e),torch_imported=('torch' in sys.modules));write(out/'RUN.json',report);raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)


def run(a):
    plan=json.loads(a.protocol.read_text());digest=sha(a.protocol);scope=authorize(plan,digest,a.preflight);need(sha(a.job_spec)==scope['launch_spec_sha256'],'job identity');job=json.loads(a.job_spec.read_text());need(job['protocol_sha256']==digest and job['id']==(ID+'_CPU_PREFLIGHT' if a.preflight else ID) and job['driver_argv']==sys.argv and sys.flags.isolated==1,'job scope / isolated argv');need(pathlib.Path(job['hf_root']).resolve()==a.hf_root.resolve() and pathlib.Path(job['cwd']).resolve()==ROOT and sorted(os.sched_getaffinity(0))==job['cpu_affinity'],'job runtime identity');need(all(importlib.metadata.version(k)==v for k,v in plan['runtime_versions'].items()),'runtime version mismatch')
    def module(name,path):
        s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
    worker_ref=next(b for b in plan['repo_sources'] if b['path']=='tools/worker_identity.py');worker_path=ROOT/worker_ref['path'];need(worker_path.stat().st_size==worker_ref['bytes'] and sha(worker_path)==worker_ref['sha256'],'startup worker source identity');worker=module('worker_identity','tools/worker_identity.py');worker.require_worker();need(sha(worker.private_config_path())==plan['worker_config_sha256'],'worker profile')
    profile=str(worker.private_config_path().resolve())
    if a.preflight:
        return cpu_preflight(a,plan,digest,scope,job,profile,module)
    verify_cpu_receipts(scope,plan,digest)
    need(os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and os.environ.get('FFNDS_PHYSICAL_GPU')=='0','UUID assignment');expected=['env','FFNDS_WORKER_CONFIG='+profile,'bash',str(ROOT/'scripts/worker_teal_gpu0_uuid_exec.sh'),'taskset','-c','0,1,2,3',str(ROOT/'.venv/bin/python'),'-I','-B']+sys.argv;need(job['argv']==expected and job['cpu_affinity']==[0,1,2,3] and pathlib.Path(job['output_dir']).resolve()==(a.hf_root/OUTPUT).resolve().parent and pathlib.Path(job['run_json']).resolve()==(a.hf_root/OUTPUT/'RUN.json').resolve() and job['external_timeout_seconds']==plan['resources']['outer_seconds'],'job launch boundary')
    out=(a.hf_root/OUTPUT).resolve();need(out.is_relative_to(a.hf_root.resolve()) and not out.exists(),'exclusive output');out.mkdir(parents=True);need(shutil.disk_usage(out).free>=plan['resources']['disk_free_min_bytes'],'disk free')
    report={'id':ID,'status':'ATTEMPT','counts':{},'completed':{},'inflight':[],'cases':[],'automatic_retry':False,'startup_worker_source_bytes':worker_ref['bytes']};write(out/'ATTEMPT.json',report);start=time.monotonic();monitor=None
    def persist():write(out/'RUN.json',report)
    def charge(name,fn):
        authorize(plan,digest);need(time.monotonic()-start<plan['resources']['wall_seconds'],'deadline');need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=plan['resources']['host_peak_rss_bytes'],'host memory');need(torch.cuda.max_memory_reserved()<=plan['resources']['gpu_reserved_bytes'],'GPU memory');need(sum(f.stat().st_size for f in out.parent.rglob('*') if f.is_file() and not f.is_symlink())<=plan['resources']['output_bytes_max'],'output limit');need(report['counts'].get(name,0)<plan['operation_limits'][name],'budget exceeded');monitor.check();report['counts'][name]=report['counts'].get(name,0)+1;report['inflight'].append(name);persist();t=time.monotonic();value=fn();torch.cuda.synchronize();report.setdefault('wall_seconds_by_operation',{}).setdefault(name,[]).append(time.monotonic()-t);report['completed'][name]=report['completed'].get(name,0)+1;need(report['inflight'][-1]==name,'ledger stack');report['inflight'].pop();persist();return value
    old_handler=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('P4 deadline')));signal.alarm(plan['resources']['wall_seconds'])
    try:
        report['source_hash_predebit_bytes']=sum(b['bytes'] for b in plan['repo_sources']);report['source_hash_completed_bytes']=0;report['source_hash_inflight']='UNKNOWN';persist()
        for b in plan['repo_sources']:
            p=(ROOT/b['path']).resolve();need(p.is_relative_to(ROOT) and p.stat().st_size==b['bytes'] and sha(p)==b['sha256'],'source identity');report['source_hash_completed_bytes']+=b['bytes'];persist()
        report['source_hash_inflight']='NONE';report['backend_hash_predebited_bytes']=sum(b['bytes'] for b in plan['backend_files']);report['backend_hash_completed_bytes']=0;report['backend_hash_inflight']='UNKNOWN';persist()
        backend_stats={}
        for b in plan['backend_files']:
            q=(ROOT/b['path']).resolve();st=fingerprint(q);need(q.is_relative_to(ROOT) and q.stat().st_size==b['bytes'] and sha(q)==b['sha256'] and fingerprint(q)==st,'backend identity');backend_stats[q]=st;report['backend_hash_completed_bytes']+=b['bytes'];persist()
        report['backend_hash_inflight']='NONE';persist()
        report['environment_logical_predebit_bytes']=plan['read_budgets']['environment_logical_bytes'];report['environment_logical_completed_bytes']=0;report['environment_logical_inflight']='UNKNOWN';persist()
        sys.path.insert(0,str(ROOT/'tools'));legacy=module('p4_original_scope_helper','tools/e0_chipmunk_paid_proxy_probe.py');need(str(pathlib.Path(sys.prefix).resolve())==plan['prepared_environment']['venv_path'] and legacy.environment_snapshot(sys.prefix)==plan['prepared_environment'],'environment files');nvidia=pathlib.Path(shutil.which('nvidia-smi')).resolve();need(str(nvidia)==plan['monitor_binary']['path'] and nvidia.stat().st_size==plan['monitor_binary']['bytes'] and sha(nvidia)==plan['monitor_binary']['sha256'],'monitor binary');report['environment_logical_completed_bytes']=plan['read_budgets']['environment_logical_bytes'];report['environment_logical_inflight']='NONE';report['environment_metadata_discovery_bytes']='UNKNOWN_ADDITIONAL';report['environment_bound']=True;persist()
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TRITON_CACHE_DIR=str(out/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'))
        import socket
        def deny(*args,**kwargs):raise RuntimeError('network forbidden')
        socket.socket.connect=deny;socket.create_connection=deny
        import torch
        from safetensors import safe_open
        sys.path.insert(0,str(ROOT/'tools'))
        import startup_gpu_reservation
        import chipmunk_a100_selected_delta_kernels as kernels
        from chipmunk_p4_proxy_math import pair_core
        from chipmunk_p4_sparse_action import selected_action
        torch.set_num_threads(plan['resources']['cpu_threads']);torch.backends.cuda.matmul.allow_tf32=False;report['precision']={'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'tf32':torch.backends.cuda.matmul.allow_tf32,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'float32_matmul_precision':torch.get_float32_matmul_precision()};need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0),'one A100 required')
        hold,monitor,report['reservation']=startup_gpu_reservation.reserve_before_assets(torch)
        author=(a.hf_root/plan['author_mlp_source']['path']).resolve();charge('author_source_hash',lambda:need(author.is_relative_to(a.hf_root.resolve()) and author.stat().st_size==plan['author_mlp_source']['bytes'] and sha(author)==plan['author_mlp_source']['sha256'],'author source'));parents={}
        for ref in plan['parent_bindings']:
            path=(a.hf_root/ref['path']).resolve();need(path.is_relative_to(a.hf_root.resolve()),'parent path');charge('parent_hash',lambda:need(path.stat().st_size==ref['bytes'] and sha(path)==ref['sha256'],'parent identity'));parents[(ref['source'],ref['type'])]=json.loads(path.read_text())
        for source in (898,96):
            ex=parents[source,'exit'];pr=parents[source,'run'];pg=parents[source,'generation'];need(ex['child_returncode']==0 and ex['process_exited'] and not ex['timeout'] and ex['run_sha256']==next(b['sha256'] for b in plan['parent_bindings'] if b['source']==source and b['type']=='run') and pr['status']=='COMPLETE_ONE_SOURCE_THREE_ARMS_FULL160_SPATIAL_STUDENT_CAPTURE' and pg['status']=='COMPLETE_ALL160_SPATIAL_TEACHER_OR_STUDENT_CAPTURE_TEN_BYTE_NEUTRALS' and pr['numeric_path']==pg['numeric_path']=='BF16_ORIGINAL','parent exit/status')
        tensors={}
        for asset in plan['assets']:
            mp=(a.hf_root/asset['metadata']['path']).resolve();pp=(a.hf_root/asset['payload']['path']).resolve();need(mp.is_relative_to(a.hf_root.resolve()) and pp.is_relative_to(a.hf_root.resolve()),'payload path');pst=fingerprint(pp);charge('metadata_hash',lambda mp=mp:need(mp.stat().st_size==asset['metadata']['bytes'] and sha(mp)==asset['metadata']['sha256'],'metadata mismatch'));meta=json.loads(mp.read_text());charge('payload_hash',lambda pp=pp:need(pp.stat().st_size==asset['payload']['bytes'] and sha(pp)==asset['payload']['sha256'],'payload mismatch'))
            need(len(meta.get('selected_rows',[]))==1092 and len(set(meta['selected_rows']))==1092 and all(0<=i<31668 for i in meta['selected_rows']),'metadata rows');need(meta.get('native_source_ordinal')==asset['source'] and meta.get('step')==asset['step'] and meta.get('block')==20 and meta.get('seed')==42 and meta.get('source_state')=='teacher_input' and meta.get('mode')=='dense' and meta.get('expert')==('high' if asset['step']<2 else 'low') and meta.get('payload',{}).get('sha256')==asset['payload']['sha256'],'packet identity');parent=parents[asset['source'],'generation'];need(all(meta.get(k)==parent.get(k) for k in ('source_id','image_sha256','prompt_sha256','protocol_sha256')),'parent source identity');entries=[q for q in parent['packets'] if q.get('step')==asset['step'] and q.get('block')==20];need(len(entries)==1 and entries[0]['metadata']==asset['metadata'],'generation packet binding')
            def load():
                with safe_open(str(pp),framework='pt',device='cpu') as f:
                    rows=f.get_tensor('rows');v=f.get_tensor('V1');gate=f.get_tensor('gate');need(rows.tolist()==meta['selected_rows'] and v.shape==(1092,5120) and gate.shape==(5120,) and v.dtype==torch.bfloat16 and gate.dtype==torch.bfloat16,'operand shape');return {'rows':rows[:128].tolist(),'x':v[:128].contiguous().cuda(),'gate':gate.cuda()}
            need(fingerprint(pp)==pst,'payload changed during SHA');tensors[(asset['source'],asset['step'])]=charge('payload_load',load);charge('payload_finite',lambda:need(bool(torch.isfinite(tensors[(asset['source'],asset['step'])]['x']).all()) and bool(torch.isfinite(tensors[(asset['source'],asset['step'])]['gate']).all()),'nonfinite operand'));tensors[(asset['source'],asset['step'])]['payload_path']=pp;tensors[(asset['source'],asset['step'])]['payload_stat']=pst;need(fingerprint(pp)==pst,'payload changed during read')
        weights={}
        for item in plan['checkpoints']:
            cp=pathlib.Path(job['checkpoints'][item['kind']]).resolve();cst=fingerprint(cp);charge('checkpoint_hash',lambda:need(cp.stat().st_size==item['bytes'] and sha(cp)==item['sha256'],'checkpoint mismatch'))
            def loadweights():
                with safe_open(str(cp),framework='pt',device='cpu') as f:return [f.get_tensor('blocks.20.ffn.'+key).contiguous().cuda() for key in ('0.weight','0.bias','2.weight','2.bias')]
            need(fingerprint(cp)==cst,'checkpoint changed during SHA');weights[item['kind']]=charge('checkpoint_tensor_load',loadweights);need(fingerprint(cp)==cst,'checkpoint changed during read');w1,b1,w2,b2=weights[item['kind']];need([tuple(t.shape) for t in (w1,b1,w2,b2)]==[(13824,5120),(13824,),(5120,13824),(5120,)] and all(t.dtype==torch.bfloat16 for t in (w1,b1,w2,b2)),'checkpoint tensor shape');charge('checkpoint_finite',lambda:need(all(bool(torch.isfinite(t).all()) for t in (w1,b1,w2,b2)),'checkpoint tensor nonfinite'))
        def initialize_selections():
            gen=torch.Generator(device='cpu').manual_seed(42);rows=torch.randperm(128,generator=gen)[:4].tolist();gen=torch.Generator(device='cpu').manual_seed(101);random_ids=torch.randperm(13824,generator=gen)[:5120].tolist();return rows,random_ids
        rows,random_ids=charge('selection_init',initialize_selections);need(rows==plan['scope']['sample_row_positions'],'sample RNG mismatch')
        for source in (898,96):
            for old,new in ((0,1),(2,3)):
                prev,cur=tensors[source,old],tensors[source,new];need(prev['rows']==cur['rows'],'matched rows');w1,b1,w2,b2=weights['high' if old==0 else 'low'];x0,x1=prev['x'],cur['x']
                action=lambda mask,h,y:selected_action(x1,w1,b1,w2,h,y,mask,charged=charge,kernels=kernels)
                result=pair_core(x0,x1,w1,b1,w2,b2,rows,random_ids,5120,charged=charge,action=action)
                def sample_shape():
                    delta=result['current_h_reference'][rows].float()-result['previous_h'][rows].float();ref=(delta.square().mean(dim=0)).sqrt();obs=result['sample_scores'];den=float(torch.linalg.vector_norm(ref));return {'sample_score_relative_l2':float(torch.linalg.vector_norm(obs-ref))/den if den else None,'sample_score_max_abs':float((obs-ref).abs().max())}
                sample_shape_result=charge('sample_shape_reference',sample_shape)
                def channel_reference():
                    delta=result['current_h_reference'].float()-result['previous_h'].float();truth=delta.square().mean(dim=0).sqrt();sample=result['sample_scores'];den=float(torch.linalg.vector_norm(truth));sample_norm=float(torch.linalg.vector_norm(sample));need(den>0,'zero change RMS');total=float(truth.square().sum());return {'sample_RMS_relative_l2':float(torch.linalg.vector_norm(sample-truth))/den,'sample_RMS_cosine':float((sample*truth).sum())/(den*sample_norm) if sample_norm else None,'true_delta_h_energy_capture':{name:float(truth[mask].square().sum())/total for name,mask in result['masks'].items()}}
                channel_stats=charge('channel_reference_stats',channel_reference)
                def saved_reference(packet):
                    path=packet['payload_path'];need(fingerprint(path)==packet['payload_stat'],'payload changed before reference')
                    with safe_open(str(path),framework='pt',device='cpu') as f:
                        h=f.get_tensor('h');y=f.get_tensor('Y');need(tuple(h.shape)==(1092,13824) and tuple(y.shape)==(1092,5120) and h.dtype==torch.bfloat16 and y.dtype==torch.bfloat16,'saved reference shape');result=(h[:128].contiguous().cuda(),y[:128].contiguous().cuda())
                    need(fingerprint(path)==packet['payload_stat'],'payload changed in reference');return result
                old_saved=charge('saved_reference_read',lambda:saved_reference(prev));new_saved=charge('saved_reference_read',lambda:saved_reference(cur))
                def metrics():
                    target=(result['current_y_reference']*cur['gate']).to(torch.bfloat16).float();need(torch.isfinite(target).all() and all(torch.isfinite(v).all() for v in result['actions'].values()),'nonfinite output');den=float(torch.linalg.vector_norm(target));need(den>0,'zero reference')
                    shape_control={}
                    for side,saved,h,y in (('previous',old_saved,result['previous_h'],result['previous_y']),('current',new_saved,result['current_h_reference'],result['current_y_reference'])):
                        for key,a,b in (('h',h,saved[0]),('Y',y,saved[1])):
                            denom=float(torch.linalg.vector_norm(b.float()));shape_control[side+'_'+key+'_relative_l2']=float(torch.linalg.vector_norm(a.float()-b.float()))/denom if denom else None
                    arms={name:{'gated_relative_l2':float(torch.linalg.vector_norm((v*cur['gate']).to(torch.bfloat16).float()-target))/den,'indices_sha256':hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()} for (name,v),mask in zip(result['actions'].items(),result['masks'].values())};return {'arms':arms,'M128_vs_saved':shape_control,'masks':{k:v.tolist() for k,v in result['masks'].items()}}
                report['cases'].append({'source':source,'steps':[old,new],'rows':cur['rows'],'sample_row_positions':rows,'M4_vs_M128_sample_score':sample_shape_result,'channel_reference':channel_stats,'metrics':charge('metrics',metrics)});persist()
        need(report['counts']==report['completed']==plan['operation_limits'] and not report['inflight'],'incomplete budget');maps=pathlib.Path('/proc/self/maps').read_text().splitlines();need(all(any(str(q) in line and not line.endswith(' (deleted)') for line in maps) and fingerprint(q)==st for q,st in backend_stats.items()),'loaded backend mismatch');report['backend_mapping_verified']=[b['path'] for b in plan['backend_files']];report.update(status='COMPLETE_DIAGNOSTIC',wall_seconds=time.monotonic()-start,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,output_worker_bytes_sampled=sum(f.stat().st_size for f in out.parent.rglob('*') if f.is_file() and not f.is_symlink()),compilation_cost='INCLUDED_IN_OBSERVED_COLD_ACTION_WALL_NOT_SEPARATELY_MEASURED')
    except BaseException as e:
        report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());persist();raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        primary_error=sys.exc_info()[0] is not None;monitor_error=None
        if monitor:
            try:
                monitor.close();report['monitor']=monitor.report()
                if report['monitor']['violations']:raise RuntimeError('GPU monitor violation')
            except BaseException as e:
                monitor_error=e;report.update(status='FAILED_GPU_MONITOR',monitor_error=type(e).__name__)
        persist()
        if monitor_error and not primary_error:raise monitor_error

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True,type=pathlib.Path);p.add_argument('--hf-root',required=True,type=pathlib.Path);p.add_argument('--job-spec',required=True,type=pathlib.Path);p.add_argument('--preflight',action='store_true');run(p.parse_args())
if __name__=='__main__':main()
