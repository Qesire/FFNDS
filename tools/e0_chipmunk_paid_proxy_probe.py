"""Review-pending, single-group A100 BF16 selected-delta operator probe.

Imports and tensor/model access occur only after exact independent-review and
ACTIVE gates. This is a non-raster source-specific controlled port, not author
Hopper reproduction, and cannot authorize itself.
"""
import argparse,hashlib,importlib.metadata,json,math,os,resource,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import chipmunk_paid_proxy_selection as selection
HF=Path('/home/xiang/.cache/huggingface')
ID='E0_CHIPMUNK_PAID_PROXY_EQUAL_K_20261029'
SELF='tools/e0_chipmunk_paid_proxy_probe.py'
SELECTOR='tools/chipmunk_paid_proxy_selection.py'
PRIOR_SOURCES=('tools/e0_chipmunk_a100_delta_probe.py','tests/test_e0_chipmunk_a100_delta_probe.py','config/e0_chipmunk_a100_delta_probe_20261029_pending_review_v1.json')
KERNEL='tools/chipmunk_a100_selected_delta_kernels.py'
TEST='tests/test_e0_chipmunk_paid_proxy_probe.py'
PROFILE_HELPERS=('tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py',
                 'tools/durable_worker_job.py','scripts/worker_teal_gpu0_uuid_exec.sh')
BANK='ffnds-derived/e0_s2_spatial_student_full160_portable_v2_20261029/part0/attempt1/worker'
BANK_RUN=BANK+'/results/spatial52_ord898_seed42_dense'
OUTPUT='ffnds-derived/e0_chipmunk_paid_proxy_equal_k_20261029/attempt1/worker/results'
META_SHA=('be8f1401edb64b63e958bc52f4cbf03b01a2e8dd86916a604633b107d3ade9f0','d9d70142cb0c68fbd3da1997fdd8870efd1683b1eb7e66e5d71c85a26fdde99a')
PAYLOAD_SHA=('a9a5679622c4e8949874e34a83a537598f7e129ce18e97b559a4d91eac79bab6','f4a408ffc6202040eebbbb60273750a9addd42ee45ea5a15aa11636056babeaf')
AUTHOR_MLP='ffnds-research-sources/p0_chipmunk_teal_20261029/chipmunk_mlp.py'
AUTHOR_MLP_SHA='28bb218e7cb474ee5da0d07d9ffa773c94fbc48f86203baf202d53e191759944'
WEIGHT_SHA='39d3af74e46e59e989db5cc36f401e666f3ae77c7648eb55d3aff9fee47716fa'
PARENT_RUN_SHA='2c47447ea11e4c608dbd3e5185815db86b5b1ec8ae3e87104b8ae994840c647a'
M,D,H=128,5120,13824
DESIGN={'identity':'A100_BF16_AUTHOR_INSPIRED_SAFE_PROXY_NOT_HOPPER_NOT_EXACT_CURAND',
 'source_ordinal':898,'seed':42,'expert':'high','block':20,'steps':[0,1],'M':M,'D':D,'H':H,
 'initializer':'same_M128_dense','cases':['proxy','random_101','random_202','random_303','fixed','full','empty'],
 'proxy':'BF16_FC1_MEAN_X1_MINUS_MEAN_BF16_PRE_GELU_MID0',
 'sample':1024,'quantile':0.7,'random_probability':0.05,'padding':256,'proxy_seed':46829,'random_seeds':[101,202,303],
 'rng':'PYTHON_MT19937_NOT_CURAND','padding_order':'SAFE_ASCENDING_INVALID_NOT_AUTHOR_ATOMIC',
 'delta_dtype':'BF16','accumulation':'FP32_TENSOR_CORE','bias2':'ONLY_INITIAL_DENSE_Y',
 'gate':'CURRENT_STEP_BF16_PRODUCT','selected_h_relative_l2_max':0.015,
 'independent_delta_relative_l2_max':0.05,'independent_Y_relative_l2_max':0.015,
 'full_same_m_relative_l2_max':0.03,'cpu_affinity':[0,1,2,3],'physical_gpu_index':0,
 'timing_scope':'SINGLE_SHOT_DIAGNOSTIC_WALL_NOT_FAIR_PERFORMANCE',
 'selector_visibility':'NO_CURRENT_DENSE_H_OR_ORACLE_BEFORE_MASK_FROZEN',
 'source_scope':'ONE_TEACHER_NONRASTER_GROUP_NO_VIDEO','degenerate_K':'KEEP_AND_LABEL_NOT_IDENTIFIABLE_NO_RETRY',
 'weight_layout':'CONTIGUOUS_FULL_WEIGHTS_DIRECT_GATHER_NO_PREPACKED_SELECTED_COLUMNS'}
RESOURCES={'GPU':1,'session_max_GPUs':4,'CPU_threads':4,'wall_seconds':900,'outer_seconds':1020,'GPU_reserved_bytes':24<<30,
 'host_peak_RSS_bytes':64<<30,'logical_input_bytes':30<<30,'new_HF_bytes':512<<20,'automatic_retry':False}
LIMITS={'payload_hash':2,'parent_packet_load':2,'checkpoint_hash':1,'high_checkpoint_load':1,
 'dense_m128':2,'group_mean':2,'proxy_w1':1,'safe_selector':1,'proxy_cache_update':1,'control_mask_build':1,
 'selection_dispatches':7,'scatter_calls':7,'independent_references':7,
 'kernel_a':6,'kernel_b':6,'video_generations':0,'model_forward':0,'training':0}


def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def binding(p,root):
    p=Path(p).resolve();root=Path(root).resolve();need(p.is_relative_to(root) and p.is_file(),'Bad bound path')
    return {'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)}
def check_binding(root,b):
    p=(Path(root)/b['path']).resolve();need(p.is_relative_to(Path(root).resolve()) and p.is_file() and p.stat().st_size==b['bytes'] and sha(p)==b['sha256'],'Bound input changed: '+b['path']);return p

def validate_indices(indices,h=H):
    need(isinstance(indices,list) and all(type(i) is int and 0<=i<h for i in indices),'Invalid selected index')
    need(len(indices)==len(set(indices)),'Duplicate selected index creates write race')
    return indices

def case_indices(name,h=H):
    if name=='empty':return []
    if name=='full':return list(range(h))
    if name=='fixed_first_4096':return list(range(min(4096,h)))
    raise ValueError('Unknown selection')

def tile_counts(m,d,h,s,bm=16,bn=32,bk=32):
    need(m>0 and d>0 and h>0 and 0<=s<=h,'Invalid dimensions')
    div=lambda a,b:(a+b-1)//b
    return {'selected':s,'padded_selected':div(s,bn)*bn if s else 0,
            'A_programs':div(m,bm)*div(s,bn) if s else 0,
            'B_programs':div(m,bm)*div(d,bn) if s else 0,
            'A_k_tiles':div(d,bk),'B_k_tiles':div(s,bk) if s else 0}

class Ledger:
    def __init__(self,report,persist):self.r=report;self.persist=persist
    def run(self,key,fn,sync=lambda:None):
        pre=self.r['predebited'];obs=self.r['completed']
        need(pre[key]<LIMITS[key],'Budget exceeded '+key);pre[key]+=1;self.r['inflight']=key;self.persist()
        result=fn();sync();obs[key]+=1;self.r['inflight']=None;self.persist();return result

def stable_file_stat(path):
    s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def durable_hash_loss_bound():
    # Progress is persisted at most every 256 MiB; allow another 8 MiB read
    # that may be in progress when the process is killed without finally.
    return (256+8)<<20

def resource_violations(host_peak,output_bytes):
    v=[]
    if host_peak>RESOURCES['host_peak_RSS_bytes']:v.append('host_peak_RSS')
    if output_bytes>RESOURCES['new_HF_bytes']-65536:v.append('output_bytes_with_receipt_reserve')
    return v

def case_closed(c):
    try:
        if c['status']!='COMPLETE':return False
        ref=c['operator_reference'];s=c['selected_count']
        if not ref['unselected_h_elementwise_equal'] or not ref['delta_elementwise_equal']:return False
        if s:
            for key,limit in (('selected_h_vs_M128_dense',DESIGN['selected_h_relative_l2_max']),
                              ('delta_vs_independent_M128_reference',DESIGN['independent_delta_relative_l2_max'])):
                v=ref[key]['relative_l2']
                if v is None or not math.isfinite(v) or not 0<=v<=limit:return False
        for k in ('Y_vs_actual_delta_FP32_reference','Y_vs_independent_delta_reference','gated_Y_vs_independent_delta_reference'):
            v=ref[k]['relative_l2']
            if v is None or not math.isfinite(v) or not 0<=v<=DESIGN['independent_Y_relative_l2_max']:return False
        return True
    except (KeyError,TypeError):return False

def masks_valid(records,k):
    try:
        if type(k) is not int or not 0<=k<=H or set(records)!=set(DESIGN['cases']):return False
        expected=selection.masks(k,H)
        for name in DESIGN['cases']:
            row=records[name];ids=validate_indices(row['indices'])
            if len(ids)!=row['count'] or row['sha256']!=selection.digest(ids):return False
            if name=='proxy' and len(ids)!=k:return False
            if name!='proxy' and ids!=expected[name]:return False
        return True
    except (ValueError,KeyError,TypeError):return False

def charged_file_hash(path,report,persist,key):
    """Track issued sequential bytes even if interrupted; caller predebits first."""
    before=stable_file_stat(path);h=hashlib.sha256();n=0;started=time.monotonic();next_mark=256<<20
    report[key]={'logical_bytes_read':0,'completed':False,'durable_unrecorded_read_bytes_max':durable_hash_loss_bound(),
                 'logical_bytes_semantics':'DURABLE_COMPLETED_READ_BYTES_NOT_PHYSICAL_DISK_IO'};persist()
    try:
        with Path(path).open('rb') as f:
            need(stable_file_stat(path)==before,'Checkpoint changed before hash')
            for block in iter(lambda:f.read(8<<20),b''):
                h.update(block);n+=len(block)
                if n>=next_mark:
                    report[key]['logical_bytes_read']=n;persist();next_mark=n+(256<<20)
        need(stable_file_stat(path)==before and n==before[2],'Checkpoint changed during hash')
        report[key].update(logical_bytes_read=n,completed=True,sha256=h.hexdigest(),stable_stat=True,durable_unrecorded_read_bytes_max=0)
        return h.hexdigest(),before
    finally:
        report[key]['logical_bytes_read']=n
        report[key]['wall_seconds']=time.monotonic()-started
        persist()

def environment_snapshot(venv):
    venv=Path(venv).resolve();site=venv/'lib/python3.11/site-packages'
    need(site.is_dir() and (venv/'bin/python').is_file() and (venv/'pyvenv.cfg').is_file(),'Expected Python 3.11 venv')
    d={x.metadata['Name'].lower():x for x in importlib.metadata.distributions(path=[str(site)])}
    need(all(name in d for name in ('torch','triton','safetensors')),'Missing Torch/Triton/safetensors')
    rows={}
    for name in ('torch','triton','safetensors'):
        record=d[name].read_text('RECORD');need(record is not None,'Package RECORD missing')
        rows[name]={'version':d[name].version,'record_sha256':hashlib.sha256(record.encode()).hexdigest()}
    files={}
    for name in ('torch/__init__.py','triton/__init__.py','triton/language/__init__.py','safetensors/__init__.py'):
        f=site/name;need(f.is_file(),'Runtime file missing');files[name]={'bytes':f.stat().st_size,'sha256':sha(f)}
    return {'venv_path':str(venv),'python_sha256':sha(venv/'bin/python'),'pyvenv_sha256':sha(venv/'pyvenv.cfg'),
            'distributions':rows,'files':files}

def draft(venv=None):
    sources=[binding(ROOT/p,ROOT) for p in (SELF,KERNEL,TEST,SELECTOR,*PRIOR_SOURCES,*PROFILE_HELPERS,'config/e0_s2_new12_dense_baseline_part0_20261029_v2.json')]
    parent=[binding(HF/p,HF) for p in (BANK+'/EXIT_RECEIPT.json',BANK+'/results/RUN.json',BANK_RUN+'/GENERATION_OBSERVATION.json')]
    metadata=[binding(HF/(BANK_RUN+f'/step{i}_block20.json'),HF) for i in (0,1)]
    author=binding(HF/AUTHOR_MLP,HF);need(author['sha256']==AUTHOR_MLP_SHA,'Author source differs')
    monitor=Path(shutil.which('nvidia-smi')).resolve()
    need(tuple(x['sha256'] for x in metadata)==META_SHA and parent[1]['sha256']==PARENT_RUN_SHA,'Parent binding differs')
    return {'id':ID,'status':'DRAFT_REVIEW_REQUESTED','design':DESIGN,'resources':RESOURCES,'limits':LIMITS,
      'repo_sources':sources,'parent_bindings':parent,'metadata':metadata,
      'payloads':[{'path':BANK_RUN+f'/step{i}_block20.safetensors','sha256':PAYLOAD_SHA[i]} for i in (0,1)],
      'checkpoint_sha256':WEIGHT_SHA,'output_HF_relative':OUTPUT,'worker_config_sha256':None,
      'prepared_environment':environment_snapshot(venv) if venv else None,'author_mlp_source':author,
      'monitor_binary':{'path':str(monitor),'bytes':monitor.stat().st_size,'sha256':sha(monitor)},
      'selector_sources':[binding(HF/'ffnds-research-sources/p0_chipmunk_teal_20261029'/x,HF) for x in ('chipmunk_topk_indices_177fdef.cu','chipmunk_copy_indices_177fdef.cu')],
      'prior_probe':[binding(HF/'ffnds-derived/e0_chipmunk_a100_128row_selected_delta_probe_20261029/attempt1/worker'/x,HF) for x in ('EXIT_RECEIPT.json','results/RUN.json')],
      'review':None,'automatic_retry':False}

def fingerprint(p):
    return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate(p,frozen=False):
    need(p['id']==ID and p['design']==DESIGN and p['resources']==RESOURCES and p['limits']==LIMITS
         and p['output_HF_relative']==OUTPUT and p['automatic_retry'] is False,'Scope differs')
    need({x['path'] for x in p['repo_sources']}=={SELF,KERNEL,TEST,SELECTOR,*PRIOR_SOURCES,*PROFILE_HELPERS,'config/e0_s2_new12_dense_baseline_part0_20261029_v2.json'},'Source closure differs')
    need(tuple(x['sha256'] for x in p['metadata'])==META_SHA and tuple(x['sha256'] for x in p['payloads'])==PAYLOAD_SHA
         and p['checkpoint_sha256']==WEIGHT_SHA and p['parent_bindings'][1]['sha256']==PARENT_RUN_SHA
         and p['author_mlp_source']['sha256']==AUTHOR_MLP_SHA and p['author_mlp_source']['path']==AUTHOR_MLP,'Parent/author identity differs')
    need([b['sha256'] for b in p['selector_sources']]==['05407a642dcb5f201b605547d53b91effb15bb9b38fce3344207c8f4204ae251','7b3725a28082dfff98f18ad70e29537ad68bf6f93265f518328798386542128f'] and p['prior_probe'][1]['sha256']=='5cf6b138d5c1f385e253aa8d3cf9377211aeb62b03c495ca14430cff0ec2131f','Author selector/prior probe differs')
    if frozen:need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['review'] is None and isinstance(p['worker_config_sha256'],str) and len(p['worker_config_sha256'])==64 and isinstance(p['prepared_environment'],dict),'Frozen worker/environment binding missing')

def authorize(p,digest):
    validate(p,True);a=read(ROOT/'research/ACTIVE_RESEARCH.json')
    scopes=[s for s in a['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==ID]
    need(a.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True and len(scopes)==1,'Exact ACTIVE scope required')
    s=scopes[0];need(s.get('protocol_sha256')==digest and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
       and s.get('resources')==RESOURCES and s.get('limits')==LIMITS and s.get('max_gpu_cards')==1,'ACTIVE budget differs')
    r=read(check_binding(ROOT,s['review']))
    need(r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('scope_id')==ID
         and r.get('reviewer_disposition') in ('APPROVE','REVISE') and r.get('state')=='REVIEWER_RECHECK_APPROVED'
         and r.get('recheck_disposition')=='APPROVE' and r.get('author_agent_id') and r.get('reviewer_agent_id')
         and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('responses_to_review') and r.get('adversarial_counterexample')
         and r.get('approval_scope_fingerprint')==fingerprint(p),'Independent review/recheck required')
    need(r.get('critical_source_hashes')=={b['path']:b['sha256'] for b in p['repo_sources']},'Review source closure differs')
    for key in ('author_review','independent_review','reviewer_recheck'):check_binding(ROOT,r[key])
    need(isinstance(s.get('launch_spec_sha256'),str) and len(s['launch_spec_sha256'])==64,'Supervisor identity required')
    return s

def run(args):
    need(not {'torch','triton','safetensors'} & set(sys.modules),'Fresh numerical process required')
    p=read(args.protocol);digest=sha(args.protocol);scope=authorize(p,digest)
    need(args.job_spec and sha(args.job_spec)==scope['launch_spec_sha256'],'Supervisor spec binding differs')
    job=read(args.job_spec);out=(HF/OUTPUT).resolve()
    argv=job.get('argv');need(isinstance(argv,list) and len(argv)==17 and argv[1].startswith('FFNDS_WORKER_CONFIG='),'Launch argv differs')
    profile=Path(argv[1].split('=',1)[1]);need(profile.is_absolute() and sha(profile)==p['worker_config_sha256'],'Profile differs')
    expected=['env',argv[1],'bash',str(ROOT/'scripts/worker_teal_gpu0_uuid_exec.sh'),'taskset','-c','0,1,2,3',
              str(Path(p['prepared_environment']['venv_path'])/'bin/python'),'-B',str(ROOT/SELF),'run',
              '--protocol',str(args.protocol.resolve()),'--hf-root',str(HF),'--job-spec',str(args.job_spec.resolve())]
    need(argv==expected and Path(job['cwd']).resolve()==ROOT and job.get('hf_root')==str(HF)
         and job.get('id')==ID and job.get('protocol_sha256')==digest and Path(job['protocol_path']).resolve()==args.protocol.resolve()
         and Path(job['output_dir']).resolve()==out.parent and Path(job['run_json']).resolve()==out/'RUN.json'
         and job['external_timeout_seconds']==RESOURCES['outer_seconds'],'Supervisor receipt/spec differs')
    need(os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and os.environ.get('FFNDS_PHYSICAL_GPU')=='0'
         and sorted(os.sched_getaffinity(0))==DESIGN['cpu_affinity'],'Expected GPU0 UUID assignment and CPU affinity')
    for b in p['repo_sources']:check_binding(ROOT,b)
    for b in p['parent_bindings']+p['metadata']+p['selector_sources']+p['prior_probe']:check_binding(HF,b)
    check_binding(HF,p['author_mlp_source'])
    monitor_path=Path(shutil.which('nvidia-smi')).resolve();monitor_binding=p['monitor_binary']
    need(str(monitor_path)==monitor_binding['path'] and monitor_path.stat().st_size==monitor_binding['bytes']
         and sha(monitor_path)==monitor_binding['sha256'],'Monitor binary differs')
    import worker_identity
    worker_identity.require_worker();need(sha(worker_identity.private_config_path())==p['worker_config_sha256'],'Private worker differs')
    need(str(Path(sys.prefix).resolve())==p['prepared_environment']['venv_path']
         and environment_snapshot(sys.prefix)==p['prepared_environment'],'Environment identity differs')
    need(not out.exists(),'Fresh attempt required');out.mkdir(parents=True)
    report={'id':ID,'protocol_sha256':digest,'status':'STARTED_NO_RETRY','design':DESIGN,
            'predebited':dict.fromkeys(LIMITS,0),'completed':dict.fromkeys(LIMITS,0),
            'inflight':None,'physical_GPU_launches':'UNKNOWN','cases':[],
            'launch_spec_sha256':scope['launch_spec_sha256'],'automatic_retry':False}
    def persist():write(out/'RUN.json',report)
    ledger=Ledger(report,persist);start=time.monotonic();monitor=None
    persist()
    def measured(name,key,fn,synchronize=lambda:None):
        t=time.monotonic()
        try:return ledger.run(key,fn,synchronize)
        finally:
            report.setdefault('input_loading_wall_seconds_including_ledger',{})[name]=time.monotonic()-t;persist()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Worker deadline')));signal.alarm(RESOURCES['wall_seconds'])
    try:
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                          TRITON_CACHE_DIR=str(out/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'))
        import socket
        def no_network(*a,**k):raise RuntimeError('Downloads forbidden')
        socket.socket.connect=no_network;socket.create_connection=no_network
        import torch
        import torch.nn.functional as F
        import startup_gpu_reservation
        import chipmunk_a100_selected_delta_kernels as kernels
        from safetensors import safe_open
        torch.set_num_threads(4);torch.set_num_interop_threads(4)
        need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0) and 'A100' in torch.cuda.get_device_name(0),'A100 SM80 required')
        _,monitor,report['GPU_reservation']=startup_gpu_reservation.reserve_before_assets(torch)
        def check():
            authorize(p,digest);monitor.check();need(time.monotonic()-start<=RESOURCES['wall_seconds'],'Deadline exceeded')
            need(torch.cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_bytes'],'GPU budget exceeded')
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RESOURCES['host_peak_RSS_bytes'],'Host RSS budget exceeded')
            need(sum(f.stat().st_size for f in out.rglob('*') if f.is_file() and not f.is_symlink())<=RESOURCES['new_HF_bytes'],'Output budget exceeded')
        sync=torch.cuda.synchronize
        parent=read(HF/p['parent_bindings'][0]['path']);generation=read(HF/p['parent_bindings'][2]['path'])
        prior_exit=read(HF/p['prior_probe'][0]['path']);prior_run=read(HF/p['prior_probe'][1]['path'])
        need(prior_exit.get('child_returncode')==0 and prior_exit.get('process_exited') is True
             and prior_exit.get('run_sha256')==p['prior_probe'][1]['sha256']
             and prior_run.get('status')=='COMPLETE_CONTROLLED_PORT_NOT_VIDEO','Prior operator run not closed')
        need(parent.get('child_returncode')==0 and parent.get('process_exited') is True and parent.get('run_sha256')==PARENT_RUN_SHA
          and generation.get('native_source_ordinal')==898 and generation.get('seed')==42
          and generation.get('mode')=='dense' and generation.get('all_ten_prior_video_state_bytes_equal') is True
          and generation.get('completed_native_flow_steps')==4,'Parent closure differs')
        metas=[read(HF/b['path']) for b in p['metadata']]
        for i,meta in enumerate(metas):
            need(meta.get('step')==i and meta.get('block')==20 and meta.get('expert')=='high' and meta.get('native_source_ordinal')==898
                 and meta.get('seed')==42 and meta.get('source_state')=='teacher_input' and meta.get('mode')=='dense'
                 and meta.get('population_rows')==31668 and meta.get('payload',{}).get('sha256')==PAYLOAD_SHA[i]
                 and all(meta.get(k)==generation.get(k) for k in ('source_id','image_sha256','prompt_sha256','protocol_sha256')),'Packet identity differs')
            entries=[z for z in generation['packets'] if z.get('step')==i and z.get('block')==20]
            need(len(entries)==1 and entries[0]['metadata']==p['metadata'][i],'Parent generation metadata binding differs')
        rows=metas[0]['selected_rows'];need(rows==metas[1]['selected_rows'] and len(rows)==1092 and len(set(rows))==1092 and all(0<=n<31668 for n in rows),'Coordinates mismatch')
        row_hash=hashlib.sha256(json.dumps(rows[:M],separators=(',',':')).encode()).hexdigest()
        report['row_binding']={'selected_rows':rows[:M],'row_list_sha256':row_hash,'geometry':'NONRASTER_SELECTED_SUBSET_NOT_FULL_PERMUTATION'};persist()
        cfg=read(ROOT/'config/e0_s2_new12_dense_baseline_part0_20261029_v2.json');model=next(x for x in cfg['models'] if x['kind']=='high');weight=Path(model['path'])
        need(model['sha256']==WEIGHT_SHA and weight.stat().st_size==model['bytes'],'High checkpoint metadata differs')
        selected_tensor_bytes=2*(H*D+D*H+H+D)
        packet_tensor_bound=2*(1092*(D+H+D)*2+D*2+1092*8)
        planned_read=model['bytes']+sum(x['payload']['bytes'] for x in metas)+packet_tensor_bound+selected_tensor_bytes
        need(planned_read<=RESOURCES['logical_input_bytes'],'Logical input byte budget exceeded');report['planned_logical_input_bytes']=planned_read;persist()
        payload_stats=[]
        for i in (0,1):
            payload=HF/p['payloads'][i]['path'];need(payload.stat().st_size==metas[i]['payload']['bytes'],'Payload size differs')
            payload_sha,payload_stat=ledger.run('payload_hash',lambda i=i,payload=payload:charged_file_hash(payload,report,persist,'payload_hash_'+str(i)))
            need(payload_sha==PAYLOAD_SHA[i],'Payload bytes changed');payload_stats.append(payload_stat)
        def load_packet(i):
            payload=HF/p['payloads'][i]['path'];need(stable_file_stat(payload)==payload_stats[i],'Payload changed before tensor read')
            key='packet_load_'+str(i);report[key]={'completed_logical_tensor_bytes':0,'physical_disk_read_bytes':'UNKNOWN','stable_stat':False};persist()
            with safe_open(str(payload),framework='pt',device='cpu') as f:
                vals={}
                for k in ('V1','h','Y','gate','rows'):
                    report[key]['inflight_tensor']=k;persist()
                    vals[k]=f.get_tensor(k);report[key]['completed_logical_tensor_bytes']+=vals[k].numel()*vals[k].element_size()
                    report[key]['inflight_tensor']=None;persist()
            need(vals['rows'].tolist()==rows,'Payload coordinates mismatch')
            need(vals['V1'].shape==(1092,D) and vals['h'].shape==(1092,H) and vals['Y'].shape==(1092,D) and vals['gate'].shape==(D,),'Payload shape mismatch')
            need(all(vals[k].dtype==torch.bfloat16 for k in ('V1','h','Y','gate')),'Packet dtype differs')
            result={k:(v[:M] if k in ('V1','h','Y') else v).contiguous().to('cuda') for k,v in vals.items() if k!='rows'}
            sync();need(stable_file_stat(payload)==payload_stats[i],'Payload changed during tensor read')
            report[key]['stable_stat']=True;persist();return result
        packets=[measured('packet_'+str(i),'parent_packet_load',lambda i=i:load_packet(i),sync) for i in (0,1)]
        check()
        actual_sha,before=ledger.run('checkpoint_hash',lambda:charged_file_hash(weight,report,persist,'checkpoint_hash_progress'))
        need(actual_sha==WEIGHT_SHA,'High checkpoint bytes differ')
        def load_weights():
            need(stable_file_stat(weight)==before,'Checkpoint changed before tensor load')
            report['checkpoint_tensor_load']={'completed_logical_bytes':0,'physical_disk_read_bytes':'UNKNOWN','stable_stat':False};persist()
            with safe_open(str(weight),framework='pt',device='cpu') as f:
                keys=['blocks.20.ffn.0.weight','blocks.20.ffn.0.bias','blocks.20.ffn.2.weight','blocks.20.ffn.2.bias']
                need(all(k in f.keys() for k in keys),'Weight header missing keys')
                vals=[]
                for k in keys:
                    report['checkpoint_tensor_load']['inflight_tensor']=k;persist()
                    val=f.get_tensor(k).contiguous().to('cuda');sync();vals.append(val)
                    report['checkpoint_tensor_load']['completed_logical_bytes']+=val.numel()*val.element_size()
                    report['checkpoint_tensor_load']['inflight_tensor']=None;persist()
            need(stable_file_stat(weight)==before,'Checkpoint changed during tensor load')
            report['checkpoint_tensor_load']['stable_stat']=True;persist()
            need([tuple(x.shape) for x in vals]==[(H,D),(H,),(D,H),(D,)] and all(x.dtype==torch.bfloat16 for x in vals),'Weight header shape/dtype differs')
            return vals
        w1,b1,w2,b2=measured('checkpoint_selected_tensors','high_checkpoint_load',load_weights,sync)
        check()
        torch.backends.cuda.matmul.allow_tf32=False
        def dense(x):
            mid=F.linear(x,w1,b1);h=F.gelu(mid,approximate='tanh');y=F.linear(h,w2,b2);return mid,h,y
        init_start=time.monotonic();old_mid,old_h,old_y=ledger.run('dense_m128',lambda:dense(packets[0]['V1']),sync)
        report['initial_dense_wall_seconds_including_ledger']=time.monotonic()-init_start
        proxy_start=time.monotonic()
        stage_cost={}
        def timed(name,key,fn):
            t=time.monotonic();value=ledger.run(key,fn,sync);stage_cost[name]=time.monotonic()-t;return value
        old_mean=timed('initial_preGELU_mean','group_mean',lambda:old_mid.mean(dim=0))
        mean_x=timed('current_input_mean','group_mean',lambda:packets[1]['V1'].mean(dim=0))
        proxy_mid=timed('full_W1_bias_proxy','proxy_w1',lambda:F.linear(mean_x,w1,b1))
        def compute_selection():
            scores=(proxy_mid-old_mean).abs().float().cpu().tolist()
            return selection.selector(scores),scores
        (sel,scores)=timed('score_copy_selector_padding','safe_selector',compute_selection);K=sel['K']
        # Cache update pays selected indices and changes only selected historical pre-GELU means.
        def update_proxy_cache():
            ix=torch.tensor(sel['indices'],dtype=torch.long,device='cuda');updated=old_mean.clone()
            updated.index_copy_(0,ix,proxy_mid.index_select(0,ix))
            keep=torch.ones(H,dtype=torch.bool,device='cuda');keep[ix]=False
            need(torch.equal(updated[keep],old_mean[keep]) and torch.equal(updated.index_select(0,ix),proxy_mid.index_select(0,ix)),
                 'Mean cache selected/unselected invariant failed')
            return updated
        proxy_cache=timed('proxy_selected_mean_cache_update_and_check','proxy_cache_update',update_proxy_cache)
        proxy_end=time.monotonic();report['proxy_paid_wall_seconds_including_ledger']=proxy_end-proxy_start
        report['compile_cost']='UNKNOWN_SEPARATE_AMOUNT_INCLUDED_IN_COLD_SINGLE_SHOT_KERNEL_WALL'
        def make_controls():
            sets=selection.masks(K,H);sets['proxy']=sel['indices']
            for indices in sets.values():validate_indices(indices)
            return sets
        t=time.monotonic();sets=ledger.run('control_mask_build',make_controls)
        report['control_masks_wall_seconds_including_ledger']=time.monotonic()-t
        report['proxy_selection']={k:v for k,v in sel.items() if k!='indices'}
        report['proxy_scores']={'values':scores,'sha256':hashlib.sha256(json.dumps(scores,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                                'identity':'BF16_PRE_GELU_MEAN_PROXY_SCORE_NO_FUTURE_STATE'}
        report['proxy_selection'].update(indices_sha256=selection.digest(sel['indices']),
           identifiability='DEGENERATE_K_ZERO_OR_H' if K in (0,H) else 'ELIGIBLE_SINGLE_GROUP_ONLY',
           author_random_and_atomic_semantics='NOT_REPRODUCED',proxy_cache_update_actual=True)
        report['mask_bindings']={k:{'count':len(v),'sha256':selection.digest(v),'indices':v} for k,v in sets.items()}
        need(masks_valid(report['mask_bindings'],K),'Mask identities/counts invalid')
        report['proxy_stage_wall_seconds_including_ledger']=stage_cost
        persist()
        # The current full teacher is first computed only after every mask is frozen.
        current_mid,current_h,current_y=ledger.run('dense_m128',lambda:dense(packets[1]['V1']),sync)
        def stats(a,b):
            da=a.float();db=b.float();delta=da-db;den=float(torch.linalg.vector_norm(db));num=float(torch.linalg.vector_norm(delta));mx=float(delta.abs().max())
            need(all(math.isfinite(x) for x in (den,num,mx)),'Nonfinite comparison')
            return {'max_abs':mx,'relative_l2':num/den if den else (0.0 if num==0 else None),'finite':True}
        def gated(y):return (y*packets[1]['gate']).to(torch.bfloat16)
        fullref={'h':packets[1]['h'],'Y':packets[1]['Y'],'gated_Y':gated(packets[1]['Y'])}
        localref={'h':current_h,'Y':current_y,'gated_Y':gated(current_y)}
        report['M128_dense_vs_fullM_capture']={k:stats(localref[k],fullref[k]) for k in localref};persist()
        for case in DESIGN['cases']:
            check();action_start=time.monotonic();pack_start=action_start
            indices=validate_indices(sets[case]);s=len(indices);tiles=tile_counts(M,D,H,s)
            hcache=old_h.clone();ycache=old_y.clone();hbefore=hcache.clone();ybefore=ycache.clone()
            sync();pack_wall=time.monotonic()-pack_start
            row={'initializer':'same_M128_dense','case':case,'selected_count':s,'tile_counts':tiles,
                 'selection_sha256':selection.digest(indices),'pack_cache_clone_wall_seconds':pack_wall,'status':'STARTED'}
            report['cases'].append(row);persist()
            selection_start=time.monotonic()
            idx=ledger.run('selection_dispatches',lambda:torch.tensor(indices,dtype=torch.int32,device='cuda'),sync)
            row['selection_wall_seconds_including_ledger']=time.monotonic()-selection_start
            ptrs=[t.untyped_storage().data_ptr() for t in (packets[1]['V1'],w1,b1,w2,hcache,ycache,idx)]
            need(len(ptrs)==len(set(ptrs)) or not s,'Aliasing would corrupt cache or inputs')
            delta=None
            if s:
                delta=torch.empty((M,s),dtype=torch.bfloat16,device='cuda');need(delta.untyped_storage().data_ptr() not in ptrs,'Delta alias')
                grid_a=((M+15)//16,(s+31)//32);grid_b=((M+15)//16,(D+31)//32)
                t=time.monotonic();ledger.run('kernel_a',lambda:kernels.selected_w1_gelu_delta[grid_a](packets[1]['V1'],w1,b1,hcache,idx,delta,M,D,H,s,16,32,32),sync)
                row['kernel_a_wall_seconds_including_ledger_and_possible_compile']=time.monotonic()-t
                t=time.monotonic();ledger.run('kernel_b',lambda:kernels.selected_w2_add_cache[grid_b](delta,w2,idx,ycache,M,D,H,s,16,32,32),sync)
                row['kernel_b_wall_seconds_including_ledger_and_possible_compile']=time.monotonic()-t
            scatter_start=time.monotonic();row_index=torch.tensor(rows[:M],dtype=torch.int64,device='cuda')
            scatter_output=torch.empty((31668,D),dtype=torch.bfloat16,device='cuda')
            ledger.run('scatter_calls',lambda:scatter_output.index_copy_(0,row_index,ycache),sync)
            row['scatter_wall_seconds_including_allocation_and_ledger']=time.monotonic()-scatter_start
            row['scatter_scope']='ONLY_128_SELECTED_ROWS_OTHER_ROWS_UNINITIALIZED_NOT_FULL_PERMUTATION'
            row['scatter_output_logical_bytes']=31668*D*2;del scatter_output,row_index
            sync();row['single_shot_action_wall_seconds_including_ledger']=time.monotonic()-action_start
            if case=='proxy':row['proxy_plus_action_wall_sum_seconds']=report['proxy_paid_wall_seconds_including_ledger']+row['single_shot_action_wall_seconds_including_ledger']
            # Reference is outside action interval and uses already-frozen mask.
            ref_start=time.monotonic()
            def independent_reference():
                selected=torch.tensor(indices,dtype=torch.long,device='cuda');mask=torch.ones(H,dtype=torch.bool,device='cuda');mask[selected]=False
                unchanged=torch.equal(hcache[:,mask],hbefore[:,mask])
                if s:
                    chosen=hcache.index_select(1,selected);old_chosen=hbefore.index_select(1,selected);expected_h=current_h.index_select(1,selected)
                    actual_delta=(chosen.float()-old_chosen.float()).to(torch.bfloat16);delta_match=torch.equal(delta,actual_delta)
                    ref_delta=(expected_h.float()-old_chosen.float()).to(torch.bfloat16);selected_w=w2.index_select(1,selected).float()
                    expected_y=(ybefore.float()+torch.matmul(ref_delta.float(),selected_w.T)).to(torch.bfloat16)
                    stage_y=(ybefore.float()+torch.matmul(actual_delta.float(),selected_w.T)).to(torch.bfloat16)
                    delta_error=stats(delta,ref_delta);h_error=stats(chosen,expected_h)
                else:delta_match=True;expected_y=ybefore;stage_y=ybefore;h_error=None;delta_error=None
                return {'unselected_h_elementwise_equal':unchanged,'delta_elementwise_equal':delta_match,'selected_h_vs_M128_dense':h_error,
                   'delta_vs_independent_M128_reference':delta_error,'Y_vs_actual_delta_FP32_reference':stats(ycache,stage_y),
                   'Y_vs_independent_delta_reference':stats(ycache,expected_y),'gated_Y_vs_independent_delta_reference':stats(gated(ycache),gated(expected_y)),
                   'reference':'M128_DENSE_H_MINUS_OLD_H_ROUND_BF16_FP32_IEEE_MATMUL_PLUS_OLD_Y_ROUND_BF16'}
            row['operator_reference']=ledger.run('independent_references',independent_reference,sync)
            row['independent_reference_wall_seconds_including_ledger']=time.monotonic()-ref_start
            need(case_closed(dict(row,status='COMPLETE')),'Independent operator reference failed')
            actual={'h':hcache,'Y':ycache,'gated_Y':gated(ycache)}
            row['vs_same_M128_dense']={k:stats(actual[k],localref[k]) for k in actual}
            row['vs_fullM_capture']={k:stats(actual[k],fullref[k]) for k in actual}
            row['signed_cross']={k:float(((actual[k].float()-localref[k].float())*(localref[k].float()-fullref[k].float())).sum()) for k in actual}
            if case=='empty':
                row['empty_cache_elementwise_equal']=bool(torch.equal(hcache,hbefore) and torch.equal(ycache,ybefore))
                need(row['empty_cache_elementwise_equal'],'Empty control changed cache')
            if case=='full':need(all(row['vs_same_M128_dense'][k]['relative_l2'] is not None and row['vs_same_M128_dense'][k]['relative_l2']<=DESIGN['full_same_m_relative_l2_max'] for k in actual),'Full control failed')
            row['status']='COMPLETE';persist()
        need(report['predebited']==report['completed'],'Operation accounting differs')
        expected_dispatches=1+(5 if K else 0)
        need(report['completed']['kernel_a']==report['completed']['kernel_b']==expected_dispatches,'Dynamic kernel dispatch coverage differs')
        need(all(report['completed'][k]==v for k,v in LIMITS.items() if k not in ('kernel_a','kernel_b')),'Exact operation coverage differs');check()
        errors={c['case']:c['vs_same_M128_dense']['gated_Y']['relative_l2'] for c in report['cases']}
        report['same_group_gate_error_comparison']={'errors':errors,'K':K,
          'interpretation':'DEGENERATE_NOT_IDENTIFIABLE' if K in (0,H) else 'DESCRIPTIVE_ONE_GROUP_NOT_SOURCE_POPULATION'}
        report.update(status='COMPLETE_PAID_PROXY_FIXED_SINGLE_GROUP_NOT_VIDEO',peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as e:
        report.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-start
        monitor_failure=None
        if monitor:
            try:
                monitor.close();report['assigned_GPU_resources']=monitor.report()
                need(not report['assigned_GPU_resources']['violations'],'GPU monitor violation')
            except BaseException as e:report.update(status='FAILED_GPU_MONITOR_PRESERVED',monitor_error=str(e));monitor_failure=e
        report['resource_accounting']={'host_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          'output_bytes_before_final_RUN':sum(f.stat().st_size for f in out.rglob('*') if f.is_file() and not f.is_symlink()),
          'host_limit_enforcement':'BOUNDARY_AND_FINAL_PEAK_CHECK_NOT_CONTINUOUS_HARD_RSS_LIMIT'}
        violations=resource_violations(report['resource_accounting']['host_peak_RSS_bytes'],report['resource_accounting']['output_bytes_before_final_RUN'])
        report['resource_accounting']['violations']=violations
        if violations:report.update(status='FAILED_RESOURCE_BUDGET_PRESERVED',resource_error=','.join(violations))
        persist()
        if violations:raise ValueError('Resource budget exceeded: '+','.join(violations))
        if monitor_failure is not None:raise monitor_failure

def verify_closed(args):
    p=read(args.protocol);validate(p,True);d=sha(args.protocol);out=HF/OUTPUT
    need(args.run.resolve()==(out/'RUN.json').resolve() and args.exit.resolve()==(out.parent/'EXIT_RECEIPT.json').resolve(),'Receipt location differs')
    r=read(args.run);e=read(args.exit);rb=binding(args.run,HF)
    scores=r['proxy_scores']['values'];replayed=selection.selector(scores)
    need(len(scores)==H and r['proxy_scores']['sha256']==hashlib.sha256(json.dumps(scores,separators=(',',':'),allow_nan=False).encode()).hexdigest()
         and all(r['proxy_selection'].get(k)==v for k,v in replayed.items() if k!='indices')
         and replayed['indices']==r['mask_bindings']['proxy']['indices'],'Proxy score/mask replay differs')
    need(e.get('id')==ID and e.get('protocol_sha256')==d and e.get('run_sha256')==rb['sha256'] and e.get('process_exited') is True
         and e.get('child_returncode')==0 and e.get('timeout') is False and e.get('automatic_retry') is False
         and e.get('spec_sha256')==r.get('launch_spec_sha256'),'Actual EXIT0 required')
    need(r.get('status')=='COMPLETE_PAID_PROXY_FIXED_SINGLE_GROUP_NOT_VIDEO' and r.get('protocol_sha256')==d and r.get('design')==DESIGN
         and r.get('predebited')==r.get('completed') and all(r['completed'][k]==v for k,v in LIMITS.items() if k not in ('kernel_a','kernel_b')) and r['completed']['kernel_a']==r['completed']['kernel_b']==1+(5 if r['proxy_selection']['K'] else 0) and r.get('inflight') is None and len(r.get('cases',[]))==7
         and [c.get('case') for c in r['cases']]==DESIGN['cases'] and all(c.get('initializer')==DESIGN['initializer'] for c in r['cases'])
         and masks_valid(r['mask_bindings'],r['proxy_selection']['K'])
         and all(c['selected_count']==r['mask_bindings'][c['case']]['count'] and c['selection_sha256']==r['mask_bindings'][c['case']]['sha256'] for c in r['cases'])
         and next(c for c in r['cases'] if c['case']=='empty').get('empty_cache_elementwise_equal') is True
         and all(next(c for c in r['cases'] if c['case']=='full')['vs_same_M128_dense'][k]['relative_l2']<=DESIGN['full_same_m_relative_l2_max'] for k in ('h','Y','gated_Y'))
         and all(case_closed(c) for c in r['cases']) and r.get('assigned_GPU_resources',{}).get('sampled_isolation_pass') is True
         and not r['assigned_GPU_resources'].get('violations') and 0<=r['wall_seconds']<=RESOURCES['wall_seconds']
         and r['peak_GPU_reserved_bytes']<=RESOURCES['GPU_reserved_bytes'] and not r.get('resource_accounting',{}).get('violations')
         and not resource_violations(r['resource_accounting']['host_peak_RSS_bytes'],r['resource_accounting']['output_bytes_before_final_RUN'])
         and r['checkpoint_hash_progress']['completed'] is True and r['checkpoint_hash_progress']['sha256']==WEIGHT_SHA
         and r['planned_logical_input_bytes']<=RESOURCES['logical_input_bytes']
         and r['checkpoint_tensor_load']['stable_stat'] is True and r['checkpoint_tensor_load']['completed_logical_bytes']==2*(H*D+D*H+H+D)
         and all(r['packet_load_'+str(i)]['stable_stat'] is True and r['payload_hash_'+str(i)]['completed'] is True
              and r['payload_hash_'+str(i)]['sha256']==PAYLOAD_SHA[i] for i in (0,1)),'Run incomplete')
    for b in p['repo_sources']:check_binding(ROOT,b)
    for b in p['parent_bindings']+p['metadata']+p['selector_sources']+p['prior_probe']:check_binding(HF,b)
    check_binding(HF,p['author_mlp_source'])
    return {'id':ID,'status':'PASS_CLOSED_PAID_PROXY_SINGLE_GROUP','RUN':rb,'exit':binding(args.exit,HF),'verification_gpu_calls':0,'verification_model_calls':0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','inspect','run','verify-closed'));ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path);ap.add_argument('--job-spec',type=Path);ap.add_argument('--environment-venv',type=Path);ap.add_argument('--hf-root',type=Path,default=HF);a=ap.parse_args()
    if a.action=='draft':
        need(a.out and not a.out.exists(),'Fresh output required');a.out.parent.mkdir(parents=True,exist_ok=True);write(a.out,draft(a.environment_venv))
    elif a.action=='inspect':
        p=read(a.protocol);validate(p);print(json.dumps({'id':ID,'fingerprint':fingerprint(p),'design':DESIGN,'limits':LIMITS},indent=2))
    elif a.action=='run':run(a)
    else:print(json.dumps(verify_closed(a),indent=2))
if __name__=='__main__':main()
