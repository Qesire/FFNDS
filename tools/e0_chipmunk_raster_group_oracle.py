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
def authorize(p,digest):
    need(p['assets_finalized'] is True and p['id'].startswith('E0_CHIPMUNK_RASTER_GROUP_ORACLE_20261030_PART'),'unfinalized draft')
    a=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text());s=[v for v in a['experiment_authority']['current_user_authorized_scopes'] if v.get('id')==p['id']];need(a['user_continuation_authorization_20261027']['new_scope_registration_allowed'] and len(s)==1,'authority');s=s[0]
    need(s.get('execution_status')=='RUNNING_FIXED_SCOPE' and s.get('protocol_sha256')==digest and s.get('max_gpu_cards')==1 and s.get('resources')==p['resources'] and s.get('limits')==p['operation_limits'],'scope mismatch')
    ref=s['review'];rp=(ROOT/ref['path']).resolve();need(rp.is_relative_to(ROOT) and rp.stat().st_size==ref['bytes'] and sha(rp)==ref['sha256'],'review');r=json.loads(rp.read_text());need(r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('scope_id')==p['id'] and r.get('protocol_sha256')==digest and r.get('job_spec_sha256')==s.get('launch_spec_sha256') and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']},'review identity');return s

def reserve_counts(current, limits, counts):
    need(all(k in limits and type(v) is int and v >= 0 and current.get(k,0)+v <= limits[k] for k,v in counts.items()), "operation budget")
    for k,v in counts.items(): current[k]=current.get(k,0)+v

def perform_charged(counts, pre, fn, sync, done):
    pre(counts)
    result=fn()
    sync()
    done(counts)
    return result

def run(a):
    p=json.loads(a.protocol.read_text());digest=sha(a.protocol);scope=authorize(p,digest);need(sha(a.job_spec)==scope['launch_spec_sha256'] and sys.flags.isolated==1,'job/isolated');job=json.loads(a.job_spec.read_text());need(job['driver_argv']==sys.argv and job['protocol_sha256']==digest and job['id']==p['id'] and pathlib.Path(job['cwd']).resolve()==ROOT and pathlib.Path(job['hf_root']).resolve()==a.hf_root.resolve() and sorted(os.sched_getaffinity(0))==job['cpu_affinity'],'job identity');need(all(importlib.metadata.version(k)==v for k,v in p['runtime_versions'].items()),'runtime versions')
    def module(name,path):
        s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
    ref=next(z for z in p['repo_sources'] if z['path']=='tools/worker_identity.py');need(sha(ROOT/ref['path'])==ref['sha256'],'worker source');worker=module('worker_identity','tools/worker_identity.py');worker.require_worker();need(sha(worker.private_config_path())==p['worker_config_sha256'],'worker profile')
    profile=str(worker.private_config_path().resolve());expected=['env','FFNDS_WORKER_CONFIG='+profile,'bash',str(ROOT/job['gpu_launcher']),str(job['physical_gpu']),job['gpu_uuid'],'taskset','-c',','.join(map(str,job['cpu_affinity'])),str(ROOT/'.venv/bin/python'),'-I','-B']+sys.argv;need(job['argv']==expected and job['gpu_launcher']=='scripts/worker_teal_assigned_uuid_exec.sh' and job['physical_gpu']==p['physical_gpu'] and os.environ.get('CUDA_VISIBLE_DEVICES')==job['gpu_uuid'] and os.environ.get('FFNDS_PHYSICAL_GPU')==str(job['physical_gpu']),'GPU launcher')
    out=(a.hf_root/p['output_HF_relative']).resolve();need(out.is_relative_to(a.hf_root.resolve()) and not out.exists() and pathlib.Path(job['output_dir']).resolve()==out.parent and pathlib.Path(job['run_json']).resolve()==out/'RUN.json' and job['external_timeout_seconds']==p['resources']['outer_seconds'],'output/job');out.mkdir(parents=True);need(shutil.disk_usage(out).free>=p['resources']['disk_free_min_bytes'],'disk')
    r={'id':p['id'],'protocol_sha256':digest,'status':'ATTEMPT','predebited':{},'completed':{},'inflight':None,'timings':{},'automatic_retry':False};atomic(out/'ATTEMPT.json',r);start=time.monotonic();monitor=None;hold=None
    def persist():atomic(out/'RUN.json',r)
    def predebit(counts):
        authorize(p,digest);need(time.monotonic()-start<p['resources']['wall_seconds'],'deadline');monitor.check();need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<p['resources']['host_peak_rss_bytes'] and torch.cuda.max_memory_reserved()<p['resources']['gpu_reserved_bytes'],'memory');need(sum(f.stat().st_size for f in out.parent.rglob('*') if f.is_file() and not f.is_symlink())<=p['resources']['output_bytes_max'],'output budget')
        reserve_counts(r['predebited'],p['operation_limits'],counts)
        r['inflight']=counts;persist()
    def complete(counts):
        for k,v in counts.items():r['completed'][k]=r['completed'].get(k,0)+v
        r['inflight']=None;persist()
    def call(name,counts,fn):
        started=[None]
        def pre(c):
            predebit(c);started[0]=time.monotonic()
        def done(c):
            r['timings'][name]={'wall_seconds_including_sync':time.monotonic()-started[0]};complete(c)
        return perform_charged(counts,pre,fn,torch.cuda.synchronize,done)
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
        from chipmunk_raster_group_oracle_math import selected_delta,relative,project,selection_debits
        torch.set_num_threads(p['resources']['cpu_threads']);torch.backends.cuda.matmul.allow_tf32=False;need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0) and 'A100' in torch.cuda.get_device_name(0),'A100 only');hold,monitor,r['reservation']=startup_gpu_reservation.reserve_before_assets(torch);r['precision']={'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0),'tf32':torch.backends.cuda.matmul.allow_tf32,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'float32_matmul_precision':torch.get_float32_matmul_precision()};persist()
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
            with safe_open(str(cp),framework='pt',device='cpu') as f:return [f.get_tensor('blocks.20.ffn.'+k).contiguous().cuda() for k in ('2.weight','2.bias')]
        w2,b2=call('weights_load',{'checkpoint_tensor_load':1},weights);need(fp(cp)==cst and [tuple(t.shape) for t in (w2,b2)]==[(5120,13824),(5120,)] and all(t.dtype==torch.bfloat16 for t in (w2,b2)),'weight identity')
        call('weights_finite',{'checkpoint_finite':1},lambda:need(all(bool(torch.isfinite(t).all()) for t in (w2,b2)),'nonfinite weights'))
        def load(step,key):
            q,st,b=captures[step,key];need(fp(q)==st,'operand changed')
            with safe_open(str(q),framework='pt',device='cpu') as f:t=f.get_tensor(key);need(list(t.shape)==b['shape'] and t.dtype==torch.bfloat16,'operand shape');v=t.contiguous().cuda()
            need(fp(q)==st,'operand changed in read');return v
        step0,step1=p['steps'];h0=call('old_h',{'operand_tensor_load':1},lambda:load(step0,'h'));h1=call('new_h',{'operand_tensor_load':1},lambda:load(step1,'h'));y0=call('old_Y',{'operand_tensor_load':1},lambda:load(step0,'Y'));y1=call('new_Y',{'operand_tensor_load':1},lambda:load(step1,'Y'));gate=call('gate',{'operand_tensor_load':1},lambda:load(step1,'gate')).reshape(5120)
        call('inputs_finite',{'operand_finite':1},lambda:need(all(bool(torch.isfinite(v).all()) for v in (h0,h1,y0,y1,gate)),'nonfinite input'))
        import torch.nn.functional as F
        y0r=call('replay_old_W2',{'dense_w2':1},lambda:F.linear(h0,w2,b2));y1r=call('replay_new_W2',{'dense_w2':1},lambda:F.linear(h1,w2,b2))
        ref=call('target_gate',{'gate_reference':1},lambda:(y1r*gate).to(torch.bfloat16))
        r['replay']=call('replay_metrics',{'metrics':1},lambda:{'old_y_relative_l2_to_captured':relative(y0r,y0),'new_y_relative_l2_to_captured':relative(y1r,y1)})
        full=call('full_delta',{'full_delta':1},lambda:(h1.float()-h0.float()).to(torch.bfloat16))
        total=call('total_energy',{'energy':1},lambda:float((h1.float()-h0.float()).double().square().sum()))
        need(total>0,'zero delta energy')
        yf=call('full_delta_projection',{'dense_w2':1},lambda:project(y0r,full,w2,gate));r['full_delta_gated_relative_l2']=call('full_metrics',{'metrics':1},lambda:relative(yf,ref));del full,yf
        r['arms']={}
        for kind in ('global','raster','random'):
            masked,info=call('selection_'+kind,selection_debits(31668,5120,kind,128),lambda kind=kind:selected_delta(h0,h1,5120,kind,128,101))
            need(info['selected_elements']==31668*5120 and info['group_count']==(1 if kind=='global' else 248),'group budget')
            candidate=call('projection_'+kind,{'dense_w2':1},lambda:project(y0r,masked,w2,gate))
            info['gated_relative_l2']=call('metrics_'+kind,{'metrics':1},lambda:relative(candidate,ref));info['delta_h_energy_capture']=info.pop('selected_energy')/total;r['arms'][kind]=info;persist();del masked,candidate
        r['zero_update_gated_relative_l2']=call('zero_metrics',{'metrics':1},lambda:relative((y0r*gate).to(torch.bfloat16),ref))
        r['method']={'identity':'POSTHOC_TRUE_H_NO_SPARSE_PERFORMANCE','m':31668,'h':13824,'d':5120,'k':5120,'group_size':128,'tail':52,'random_seed':101,'delta':'FP32 score; selected/full delta cast BF16 before dense W2','projection':'BF16 dense W2; FP32 add to BF16 replay Y0 then BF16; BF16 current gate','baseline':'same-device recomputed BF16 dense W2 with bias on captured old/current h; compare Y to captured','original_h_current_is_posthoc':True}
        need(r['predebited']==r['completed']==p['operation_limits'] and r['inflight'] is None,'ledger incomplete');maps=pathlib.Path('/proc/self/maps').read_text().splitlines();need(all(any(str(q) in line and not line.endswith(' (deleted)') for line in maps) and fp(q)==st for q,st in backend_stats.items()),'mapped backend');r['backend_mapping_verified']=[b['path'] for b in p['backend_files']];r.update(status='COMPLETE_DIAGNOSTIC',wall_seconds=time.monotonic()-start,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
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
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=pathlib.Path,required=True);p.add_argument('--hf-root',type=pathlib.Path,required=True);p.add_argument('--job-spec',type=pathlib.Path,required=True);run(p.parse_args())
if __name__=='__main__':main()
