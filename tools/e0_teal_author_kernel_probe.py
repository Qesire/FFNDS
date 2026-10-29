"""Reviewed, registered author-TEAL single-token FP16 kernel probe.

Static draft is not execution authorization. This bounded microbenchmark adapts
TEAL's source benchmark; it is not full Llama decoding, paper reproduction, or
Wan transfer. No compilation, numeric imports or GPU work occur in draft/inspect.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import shutil
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
ID='E0_TEAL_AUTHOR_FP16_KERNEL_PROBE_20261029'
UPSTREAM='fb7373c93ac3594817c9ee64d4e08b47430a1822'
SOURCE='.runtime/teal_'+UPSTREAM
SELF='tools/e0_teal_author_kernel_probe.py'
TEST='tests/test_e0_teal_author_kernel_probe.py'
REVIEW_ROOT='research/experiment_reviews/E0_TEAL_AUTHOR_FP16_KERNEL_PROBE_20261029'
INDEPENDENT_REVIEW=REVIEW_ROOT+'/INDEPENDENT_REVIEW_V1.md'
AUTHOR_RESPONSE=REVIEW_ROOT+'/AUTHOR_RESPONSE_V1.md'
REVIEWER_RECHECK=REVIEW_ROOT+'/REVIEWER_RECHECK_V2.md'
HELPERS=('tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py',
         'tools/durable_worker_job.py','scripts/worker_wait_uuid_idle_exec.sh','scripts/worker_teal_gpu0_uuid_exec.sh')
OUTPUT='ffnds-derived/e0_teal_author_fp16_kernel_probe_20261029/attempt1/worker/results'
SOURCES={
 'kernels/__init__.py':'01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b',
 'kernels/sparse_gemv.py':'21183499ac9eb06c57365023388920721636b5bc0d2acf5d91c95fd5e48c37a3',
 'kernels/compile_wrapper.py':'7e1b86358171923d3ac658476b483c5b1dc199601b84976264d5814933239d7c',
 'kernels/kernel_utils.py':'c47ca739752a4012b027493f9ad987a1b27cf2f4b0739d714b0ae1f881efd44f',
 'scripts/benchmark_gemv.py':'27dddfe933e5359cbdcbffe246a41a5be541cddd59e4698043f02bd2e8d111af',
 'pyproject.toml':'b04eb6ea56b66b5cf6a342e769795046341f6d73e3edb5a7be53688f738aac81'}
DESIGN={'claim':'AUTHOR_KERNEL_PROBE_NOT_LLM_REPRO_OR_WAN', 'source_commit':UPSTREAM,
 'gpu_compute_capability':[8,0], 'gpu_name_contains':'A100', 'dtype':'float16',
 'input_shape':[1,1,4096], 'weight_shape':[14336,4096], 'weight_layout':'TRANSPOSE_CONTIGUOUS_TRANSPOSE',
 'seed':42,'input_distribution':'torch.rand_minus_half','thresholds':[0.0,0.25],
 'sparsity_bins':[0,9],'batch':1,'fallback_sequence_length':2,
 'warmup_iterations':10,'timed_iterations':20, 'latency':'BALANCED_INTERLEAVED_PER_CALL_CUDA_EVENTS_AND_SYNCHRONIZED_CPU_WALL',
 'decision':'DESCRIPTIVE_PAIRED_MEDIAN_NO_CAUSAL_TRAFFIC_CLAIM;_NOISE_UNCHARACTERIZED',
 'autotune':'TWO_FIXED_KEYS_16_AUTHOR_CONFIGS_INTERNAL_LAUNCH_COUNT_UNKNOWN_SEPARATE_SETUP_WALL',
 'launcher':'DURABLE_WORKER_GPU0_UUID_FLOCK_CPU0_1_ACTUAL_EXIT_OUTER_1020S_SINGLE_ATTEMPT',
 'cpu_affinity':[0,1], 'physical_gpu_index':0,
 'execution':'AUTHOR_EAGER_FORWARD_BOUNDED_ADAPTED_HARNESS_NOT_TORCH_COMPILE',
 'comparison':'DENSE_ZERO_THRESHOLD_AND_DENSE_EXPLICIT_MAGNITUDE_MASK_SAME_FP16_OPERANDS',
 'fallback':'SAME_AUTHOR_FORWARD_SEQ_LEN_2_DENSE_MATMUL_NO_THRESHOLD_MASK',
 'relative_l2_max':0.015,'new_training':False}
ENV_FILES=('torch/__init__.py','torch/version.py','torch/_C.cpython-311-x86_64-linux-gnu.so',
           'triton/__init__.py','triton/runtime/autotuner.py','triton/runtime/jit.py',
           'triton/backends/nvidia/compiler.py','msgspec/__init__.py')
ENV_DISTRIBUTIONS=('torch','triton','msgspec','numpy','nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12',
                   'nvidia-cublas-cu12','nvidia-nvjitlink-cu12')
RESOURCES={'GPU':1,'CPU_threads':2,'wall_seconds':900,'outer_seconds':1020,
 'GPU_reserved_bytes':4<<30,'new_HF_bytes':1<<30,'free_disk_bytes':10<<30,'automatic_retry':False}
LIMITS={'sparse_forward':62,'sparse_gemv_dispatch':62,'author_triton_launch_entry':62,'fallback_forward':1,'fallback_matmul':1,
 'dense_reference':2,'fallback_reference':1,'fallback_masked_reference':1,'dense_benchmark':60,'video_generations':0,'model_loads':0,'training_updates':0}


def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def canonical(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def bind(path,root):
    path=Path(path).resolve();root=Path(root).resolve();need(path.is_relative_to(root) and path.is_file(),'Binding outside root')
    return {'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':sha(path)}
def bound(root,b):
    path=(Path(root)/b['path']).resolve();need(path.is_relative_to(Path(root).resolve()) and path.is_file()
        and path.stat().st_size==b['bytes'] and sha(path)==b['sha256'],'Bound file changed: '+b['path']);return path
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def environment_snapshot(venv):
    venv=Path(venv).resolve();site=venv/'lib/python3.11/site-packages';exe=venv/'bin/python'
    need(site.is_dir() and exe.is_file() and (venv/'pyvenv.cfg').is_file(),'Prepared environment missing')
    distributions={d.metadata['Name'].lower():d for d in importlib.metadata.distributions(path=[str(site)])}
    need(set(ENV_DISTRIBUTIONS)<=set(distributions),'Required distribution missing')
    rows={}
    for name,d in sorted(distributions.items()):
        record=d.read_text('RECORD');need(record is not None,'Distribution RECORD missing')
        rows[name]={'version':d.version,'RECORD_sha256':hashlib.sha256(record.encode()).hexdigest()}
    files={name:{'bytes':(site/name).stat().st_size,'sha256':sha(site/name)} for name in ENV_FILES}
    return {'venv_path':str(venv),'python_executable_sha256':sha(exe),'pyvenv_cfg_sha256':sha(venv/'pyvenv.cfg'),
            'distributions':rows,'runtime_files':files}


def draft(venv=None):
    bindings=[bind(ROOT/SOURCE/p,ROOT) for p in sorted(SOURCES)]
    need(all(b['sha256']==SOURCES[b['path'][len(SOURCE)+1:]] for b in bindings),'Pinned author source differs')
    return {'id':ID,'status':'DRAFT_REVIEW_REQUESTED','assets_finalized':False,'upstream_repository':'https://github.com/FasterDecoding/TEAL',
      'design':DESIGN,'resources':RESOURCES,'limits':LIMITS,'source_files':bindings,
      'repo_sources':[bind(ROOT/p,ROOT) for p in (SELF,TEST,*HELPERS)],'output_HF_relative':OUTPUT,
      'environment':{'python_min':'3.11','torch_required':'2.4.0','triton_required':'3.0.0',
                     'distribution_policy':'AUTHOR_PINNED_TORCH_Triton_GPU_TARGET_SM80','worker_config_sha256':None,
                     'prepared_environment':environment_snapshot(venv) if venv else None,'cuda_required':'12.1','driver_version':None,
                     'monitor_binary':None},
      'review':None,'automatic_retry':False}


def fingerprint(plan):
    return canonical({k:plan[k] for k in ('id','design','resources','limits','source_files','repo_sources','output_HF_relative','environment')})


def validate(plan,frozen=False):
    need(plan['id']==ID and plan['design']==DESIGN and plan['resources']==RESOURCES and plan['limits']==LIMITS
         and plan['output_HF_relative']==OUTPUT and plan['automatic_retry'] is False,'Fixed experiment differs')
    need({b['path'] for b in plan['repo_sources']}=={SELF,TEST,*HELPERS} and len(plan['repo_sources'])==2+len(HELPERS),'Source closure differs')
    need({b['path'] for b in plan['source_files']}=={SOURCE+'/'+p for p in SOURCES} and len(plan['source_files'])==len(SOURCES)
         and all(b['sha256']==SOURCES[b['path'][len(SOURCE)+1:]] for b in plan['source_files']), 'Pinned upstream source differs')
    need(plan['environment']['python_min']=='3.11' and plan['environment']['torch_required']=='2.4.0'
         and plan['environment']['triton_required']=='3.0.0' and plan['environment']['cuda_required']=='12.1'
         and plan['environment']['distribution_policy']=='AUTHOR_PINNED_TORCH_Triton_GPU_TARGET_SM80','Author compiler environment differs')
    if frozen:
        worker=plan['environment']['worker_config_sha256']
        need(plan['status']=='FROZEN_PENDING_REGISTRATION' and plan['assets_finalized'] is True and isinstance(worker,str)
             and len(worker)==64 and set(worker)<=set('0123456789abcdef') and plan['review'] is None
             and isinstance(plan['environment']['prepared_environment'],dict) and isinstance(plan['environment']['driver_version'],str)
             and plan['environment']['driver_version'] and isinstance(plan['environment']['monitor_binary'],dict),'Review/worker/environment missing')


def authorize(plan,digest):
    validate(plan,True)
    active=read(ROOT/'research/ACTIVE_RESEARCH.json')
    scopes=[s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id')==ID]
    need(active.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True
         and len(scopes)==1 and scopes[0].get('protocol_sha256')==digest
         and scopes[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE')
         and scopes[0].get('resources')==RESOURCES and scopes[0].get('limits')==LIMITS
         and scopes[0].get('max_gpu_cards')==1,'Exact ACTIVE scope required')
    review=scopes[0].get('review');need(isinstance(review,dict),'Registered independent review binding required')
    r=read(bound(ROOT,review))
    need(r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('scope_id')==ID
         and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('reviewer_disposition')=='REVISE'
         and r.get('recheck_disposition')=='APPROVE' and r.get('author_agent_id')
         and r.get('reviewer_agent_id') and r['reviewer_agent_id']!=r['author_agent_id']
         and r.get('responses_to_review') and r.get('adversarial_counterexample')
         and r.get('author_review') and r.get('independent_review') and r.get('reviewer_recheck')
         and r.get('approval_scope_fingerprint')==fingerprint(plan),'Actual independent review/recheck required')
    expected_hashes={b['path']:b['sha256'] for b in plan['repo_sources']+plan['source_files']}
    need(r.get('critical_source_hashes')==expected_hashes,'Review source closure differs')
    need(r['author_review']['path']==AUTHOR_RESPONSE and r['independent_review']['path']==INDEPENDENT_REVIEW
         and r['reviewer_recheck']['path']==REVIEWER_RECHECK, 'Review evidence paths differ')
    for key in ('author_review','independent_review','reviewer_recheck'):bound(ROOT,r[key])
    spec=scopes[0].get('launch_spec_sha256')
    need(isinstance(spec,str) and len(spec)==64 and set(spec)<=set('0123456789abcdef'), 'Registered supervisor spec SHA required')
    return scopes[0]


def paired_order(index):return ('sparse','dense') if index%2==0 else ('dense','sparse')

def record_then_require(report,key,value,persist,condition,message):
    report[key]=value;persist();need(condition,message)

def verify_job_spec(job,args,plan,digest):
    hf=args.hf_root.resolve();out=(hf/OUTPUT).resolve()
    argv=job['argv'];need(isinstance(argv,list) and len(argv)==17,'Supervisor argv length differs')
    need(argv[0]=='env' and argv[1].startswith('FFNDS_WORKER_CONFIG='),'Private worker profile argument missing')
    profile=Path(argv[1].split('=',1)[1]);need(profile.is_absolute() and sha(profile)==plan['environment']['worker_config_sha256'],'Spec worker profile differs')
    expected=['env',argv[1],'bash',str(ROOT/'scripts/worker_teal_gpu0_uuid_exec.sh'),
              'taskset','-c','0,1',str(Path(plan['environment']['prepared_environment']['venv_path'])/'bin/python'),
              '-B',str(ROOT/SELF),'run','--protocol',str(args.protocol.resolve()),
              '--hf-root',str(hf),'--job-spec',str(args.job_spec.resolve())]
    need(argv==expected and job['id']==ID and job['protocol_sha256']==digest
         and Path(job['protocol_path']).resolve()==args.protocol.resolve() and Path(job['cwd']).resolve()==ROOT
         and Path(job['hf_root']).resolve()==hf and Path(job['output_dir']).resolve()==out.parent
         and Path(job['run_json']).resolve()==out/'RUN.json'
         and job['external_timeout_seconds']==RESOURCES['outer_seconds'],'Supervisor executable chain differs')


def run(args):
    need(not {'torch','numpy','triton'} & set(sys.modules),'Fresh numerical worker required')
    plan=read(args.protocol);digest=sha(args.protocol);scope=authorize(plan,digest)
    need(args.job_spec is not None and sha(args.job_spec)==scope['launch_spec_sha256'],'Supervisor spec SHA differs')
    job=read(args.job_spec);verify_job_spec(job,args,plan,digest)
    need(sorted(os.sched_getaffinity(0))==DESIGN['cpu_affinity'] and os.environ.get('FFNDS_PHYSICAL_GPU')=='0'
         and os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'),'GPU0 UUID lock/CPU affinity differs')
    for b in plan['source_files']+plan['repo_sources']:bound(ROOT,b)
    import worker_identity
    need(Path(worker_identity.__file__).resolve()==ROOT/'tools/worker_identity.py','Worker helper origin differs')
    worker_identity.require_worker();need(sha(worker_identity.private_config_path())==plan['environment']['worker_config_sha256'],'Worker differs')
    binary=shutil.which('nvidia-smi');monitor_binding=plan['environment']['monitor_binary']
    need(binary is not None and str(Path(binary).resolve())==monitor_binding['path']
         and Path(binary).stat().st_size==monitor_binding['bytes'] and sha(binary)==monitor_binding['sha256'],'Monitor binary changed')
    env=plan['environment']['prepared_environment']
    need(sys.version_info[:2]>=(3,11) and str(Path(sys.prefix).resolve())==env['venv_path']
         and environment_snapshot(sys.prefix)==env,'Prepared environment identity differs')
    need(importlib.metadata.version('torch').split('+')[0]=='2.4.0' and importlib.metadata.version('triton')=='3.0.0',
         'Author pinned torch/triton unavailable')
    hf=args.hf_root.resolve();out=(hf/OUTPUT).resolve()
    need(out.is_relative_to(hf) and not out.exists(),'Fresh confined attempt required');out.mkdir(parents=True)
    report={'id':ID,'protocol_sha256':digest,'status':'STARTED_NO_RETRY','design':DESIGN,'limits':LIMITS,
      'predebited':dict.fromkeys(LIMITS,0),'observed':dict.fromkeys(LIMITS,0),
      'autotune_internal_GPU_launches':'UNKNOWN','launch_spec_sha256':scope['launch_spec_sha256'],'automatic_retry':False,'LLM_reproduction':False,'WAN_transfer':False}
    def persist():write(out/'RUN.json',report)
    batch={'active':False,'used':{}}
    def call(key,fn):
        if batch['active']:
            used=batch['used'].get(key,0)+1;need(used<=batch['reserved'].get(key,0),'Batch reservation exceeded '+key)
            batch['used'][key]=used
            v=fn();report['observed'][key]+=1;return v
        need(report['predebited'][key]+1<=LIMITS[key],'Operation budget exceeded '+key)
        report['predebited'][key]+=1;persist();v=fn();report['observed'][key]+=1;persist();return v
    started=time.monotonic();monitor=None
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Probe wall budget exceeded')));signal.alarm(RESOURCES['wall_seconds'])
    try:
        for name in ('tmp','xdg','torch','hf','cuda_cache'):(out/name).mkdir(exist_ok=True)
        os.environ.update(TMPDIR=str(out/'tmp'), XDG_CACHE_HOME=str(out/'xdg'), TORCH_HOME=str(out/'torch'),
                          HF_HOME=str(out/'hf'), CUDA_CACHE_PATH=str(out/'cuda_cache'), HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TRITON_CACHE_DIR=str(out/'triton_cache'),
                          TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        import socket
        def no_network(*_a,**_k):raise RuntimeError('network forbidden')
        socket.socket.connect=no_network;socket.create_connection=no_network
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(ROOT/SOURCE));torch=importlib.import_module('torch');triton=importlib.import_module('triton')
        import startup_gpu_reservation,assigned_gpu_resources
        need(Path(startup_gpu_reservation.__file__).resolve()==ROOT/'tools/startup_gpu_reservation.py'
             and Path(assigned_gpu_resources.__file__).resolve()==ROOT/'tools/assigned_gpu_resources.py','GPU helper origin differs')
        need(torch.version.cuda==plan['environment']['cuda_required'] and
             Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()) and
             Path(triton.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), 'Runtime numerical backend differs')
        report['environment']={'python_version':sys.version,'prepared_environment':env,'cuda_runtime':torch.version.cuda}
        _,monitor,report['GPU_reservation']=startup_gpu_reservation.reserve_before_assets(torch)
        driver=monitor.samples[-1]['hardware'][0][1].strip()
        need(driver==plan['environment']['driver_version'],'GPU driver version differs')
        report['environment']['GPU_driver_version']=driver
        need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0)
             and 'A100' in torch.cuda.get_device_name(0),'One A100 SM80 required')
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(DESIGN['seed'])
        sparse=importlib.import_module('kernels.sparse_gemv');need(Path(sparse.__file__).resolve()==(ROOT/SOURCE/'kernels/sparse_gemv.py').resolve(),'Author import differs')
        original_sparse=sparse.splitk_sparse_gemv;original_kernel=sparse.splitk_sparse_gemv_kernel;original_matmul=torch.matmul
        need(len(original_kernel.configs)==16,'Author autotune config set differs')
        class KernelObserver:
            def __getitem__(self,grid):
                launch=original_kernel[grid]
                def invoke(*a,**kw):return call('author_triton_launch_entry',lambda:launch(*a,**kw))
                return invoke
        sparse.splitk_sparse_gemv_kernel=KernelObserver()
        def observed_sparse(*a,**k):return call('sparse_gemv_dispatch',lambda:original_sparse(*a,**k))
        sparse.splitk_sparse_gemv=observed_sparse
        op=sparse.SparseGEMV.initialize('ffnds_teal_author_probe','cuda')
        x=torch.rand(DESIGN['input_shape'],device='cuda',dtype=torch.float16)-0.5
        w=(torch.rand(DESIGN['weight_shape'],device='cuda',dtype=torch.float16)-0.5).T.contiguous().T
        need(w.stride()==(1,14336) and x.shape==(1,1,4096),'Author shape/layout differs')
        def check():
            authorize(plan,digest);monitor.check();need(time.monotonic()-started<=RESOURCES['wall_seconds'],'Wall budget exceeded')
            need(torch.cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_bytes'],'GPU memory budget exceeded')
            need(shutil.disk_usage(out).free>=RESOURCES['free_disk_bytes'],'Disk reserve exceeded')
            need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())<=RESOURCES['new_HF_bytes'],'Output budget exceeded')
        def sparse_call(threshold):return call('sparse_forward',lambda:op.forward(x,w,threshold,DESIGN['sparsity_bins'][DESIGN['thresholds'].index(threshold)]))
        def dense_call(key,input_,threshold):
            return call(key,lambda:original_matmul(input_ * (input_.abs()>threshold),w.T))
        def error(y,ref):
            a=y.float();b=ref.float();den=float(torch.linalg.vector_norm(b));diff=float(torch.linalg.vector_norm(a-b));maximum=float((a-b).abs().max())
            valid=math.isfinite(den) and math.isfinite(diff) and math.isfinite(maximum)
            return {'finite':valid,'relative_l2':diff/den if valid and den else None,'max_abs':maximum if valid else None}
        check();setup_start=time.monotonic();results={};report['results']=results
        for threshold in DESIGN['thresholds']:
            y=sparse_call(threshold);ref=dense_call('dense_reference',x,threshold);torch.cuda.synchronize()
            stats=error(y,ref)
            config=getattr(original_kernel,'best_config',None)
            results[str(threshold)]={'correctness':stats,'actual_input_zero_fraction':float((x.abs()<=threshold).float().mean()),
                 'selected_autotune_config':str(config) if config is not None else 'UNAVAILABLE'}
            record_then_require(report,'results',results,persist,stats['relative_l2'] is not None and stats['relative_l2']<=DESIGN['relative_l2_max'],'Sparse vs masked reference mismatch');check()
        report['compile_autotune_and_correctness_wall_seconds']=time.monotonic()-setup_start
        x2=torch.cat((x,x),dim=1)
        def counted_matmul(*a,**k):return call('fallback_matmul',lambda:original_matmul(*a,**k))
        torch.matmul=counted_matmul
        try:y2=call('fallback_forward',lambda:op.forward(x2,w,0.25,DESIGN['sparsity_bins'][1]))
        finally:torch.matmul=original_matmul
        reference=call('fallback_reference',lambda:original_matmul(x2,w.T))
        masked=call('fallback_masked_reference',lambda:original_matmul(x2*(x2.abs()>0.25),w.T))
        torch.cuda.synchronize();fallback=error(y2,reference);contrast=error(reference,masked)
        report['fallback']={'sequence_length':2,'comparison_to_unmasked_dense':fallback,
            'masked_vs_unmasked_reference':contrast,'actual_fallback_matmul_count':report['observed']['fallback_matmul']}
        record_then_require(report,'fallback',report['fallback'],persist,fallback['max_abs']==0.0 and contrast['max_abs'] is not None and contrast['max_abs']>0,'Fallback/threshold contrast differs')
        batch['reserved']={'sparse_forward':60,'sparse_gemv_dispatch':60,'author_triton_launch_entry':60,'dense_benchmark':60}
        for key,n in batch['reserved'].items():
            need(report['predebited'][key]+n<=LIMITS[key],'Batch budget exceeded '+key);report['predebited'][key]+=n
        persist();batch['active']=True
        for threshold in DESIGN['thresholds']:
            row=results[str(threshold)];fns={'sparse':lambda:sparse_call(threshold),
                'dense':lambda:call('dense_benchmark',lambda:original_matmul(x,w.T))}
            times={name:[] for name in fns};walls={name:[] for name in fns};order=[]
            for i in range(DESIGN['warmup_iterations']):
                for name in paired_order(i):fns[name]()
            torch.cuda.synchronize()
            for i in range(DESIGN['timed_iterations']):
                check()  # Outside both event and synchronized-call wall intervals.
                current=paired_order(i);order.append(list(current))
                for name in current:
                    torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    wall=time.monotonic();a.record();fns[name]();b.record();b.synchronize()
                    walls[name].append(time.monotonic()-wall);times[name].append(a.elapsed_time(b))
            for name in fns:
                row[name]={'warmup_calls':10,'measured_calls':20,'median_cuda_ms':statistics.median(times[name]),
                           'all_cuda_ms':times[name],'all_synchronized_call_wall_seconds':walls[name]}
            paired=[a-b for a,b in zip(times['sparse'],times['dense'])]
            row['paired_event_difference_ms']={'sparse_minus_dense':paired,'median':statistics.median(paired),
                'interpretation':'DESCRIPTIVE_EVENT_CALL_INTERVAL_NOT_KERNEL_ONLY_CAUSAL_TRAFFIC_OR_ROBUST_SPEEDUP'}
            row['timed_order']=order;row['decision']='INDETERMINATE_NO_PREDECLARED_NOISE_MODEL'
        batch['active']=False;persist()
        need(batch['used']==batch['reserved'],'Actual batch call coverage differs')
        torch.cuda.synchronize();check()
        need(report['predebited']==report['observed']==LIMITS,'Exact direct-call ledger incomplete')
        report.update(status='COMPLETE_AUTHOR_KERNEL_PROBE_NOT_LLM_REPRO',results=results,
          GPU_name=torch.cuda.get_device_name(0),GPU_compute_capability=[8,0],peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
          torch_version=torch.__version__,triton_version=triton.__version__)
    except BaseException as e:
        report.update(status='FAILED_AUTHOR_KERNEL_PROBE_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-started
        if monitor:
            try:
                monitor.close();report['assigned_GPU_resources']=monitor.report()
                need(not report['assigned_GPU_resources']['violations'],'Assigned GPU violation')
            except BaseException as e:
                report.update(status='FAILED_GPU_MONITOR_PRESERVED_NO_RETRY',monitor_error=str(e));persist();raise
        persist()


def verify_closed(args):
    hf=args.hf_root.resolve();plan=read(args.protocol);validate(plan,True);digest=sha(args.protocol)
    out=hf/OUTPUT;need(args.run.resolve()==(out/'RUN.json').resolve() and args.exit.resolve()==(out.parent/'EXIT_RECEIPT.json').resolve(),'Receipt path differs')
    binding=bind(args.run,hf);report=read(args.run);ex=read(args.exit)
    need(ex.get('id')==ID and ex.get('protocol_sha256')==digest and ex.get('run_sha256')==binding['sha256']
         and ex.get('spec_sha256')==report.get('launch_spec_sha256') and isinstance(ex.get('spec_sha256'),str)
         and ex.get('process_exited') is True and type(ex.get('child_returncode')) is int and ex['child_returncode']==0
         and ex.get('timeout') is False and ex.get('automatic_retry') is False,'Actual EXIT0 required')
    need(report.get('id')==ID and report.get('protocol_sha256')==digest
         and report.get('status')=='COMPLETE_AUTHOR_KERNEL_PROBE_NOT_LLM_REPRO'
         and report.get('predebited')==report.get('observed')==LIMITS and report.get('design')==DESIGN
         and report.get('environment',{}).get('prepared_environment')==plan['environment']['prepared_environment']
         and report.get('environment',{}).get('GPU_driver_version')==plan['environment']['driver_version']
         and report.get('environment',{}).get('cuda_runtime')==plan['environment']['cuda_required']
         and report.get('GPU_compute_capability')==[8,0] and 'A100' in report.get('GPU_name','')
         and 0<=report.get('wall_seconds',-1)<=RESOURCES['wall_seconds']
         and report.get('peak_GPU_reserved_bytes',RESOURCES['GPU_reserved_bytes']+1)<=RESOURCES['GPU_reserved_bytes']
         and not report.get('assigned_GPU_resources',{}).get('violations')
         and report.get('assigned_GPU_resources',{}).get('sampled_isolation_pass') is True,'Actual run closure differs')
    need(set(report['results'])=={'0.0','0.25'} and report['fallback']['comparison_to_unmasked_dense']['max_abs']==0.0
         and all(0<=row['correctness']['relative_l2']<=DESIGN['relative_l2_max'] and 0<=row['actual_input_zero_fraction']<=1
                 and all(len(row[arm]['all_cuda_ms'])==20 for arm in ('sparse','dense')) for row in report['results'].values()),'Result coverage differs')
    for b in plan['source_files']+plan['repo_sources']:bound(ROOT,b)
    return {'id':ID,'status':'PASS_CLOSED_AUTHOR_KERNEL_PROBE','RUN':binding,'exit':bind(args.exit,hf),'model_calls':0,'GPU_calls':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('draft','inspect','run','verify-closed'))
    p.add_argument('--environment-venv',type=Path);p.add_argument('--job-spec',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--out',type=Path);p.add_argument('--run',type=Path);p.add_argument('--exit',type=Path);p.add_argument('--hf-root',type=Path,default=Path('/home/xiang/.cache/huggingface'))
    a=p.parse_args()
    if a.action=='draft':
        need(a.out is not None and not a.out.exists(),'Fresh draft required');a.out.parent.mkdir(parents=True,exist_ok=True);write(a.out,draft(a.environment_venv))
    elif a.action=='inspect':
        plan=read(a.protocol);validate(plan);print(json.dumps({'id':ID,'fingerprint':fingerprint(plan),'design':DESIGN,'limits':LIMITS},indent=2))
    elif a.action=='run':run(a)
    else:print(json.dumps(verify_closed(a),indent=2))

if __name__=='__main__':main()
