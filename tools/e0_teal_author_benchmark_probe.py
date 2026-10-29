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
ID='E0_TEAL_AUTHOR_BENCHMARK_PROBE_20261029'
UPSTREAM='fb7373c93ac3594817c9ee64d4e08b47430a1822'
SOURCE='.runtime/teal_'+UPSTREAM
SELF='tools/e0_teal_author_benchmark_probe.py'
TEST='tests/test_e0_teal_author_benchmark_probe.py'
REVIEW_ROOT='research/experiment_reviews/E0_TEAL_AUTHOR_BENCHMARK_PROBE_20261029'
INDEPENDENT_REVIEW=REVIEW_ROOT+'/INDEPENDENT_REVIEW_V1.md'
AUTHOR_RESPONSE=REVIEW_ROOT+'/AUTHOR_RESPONSE_V1.md'
REVIEWER_RECHECK=REVIEW_ROOT+'/REVIEWER_RECHECK_V1.md'
HELPERS=('tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py',
         'tools/durable_worker_job.py','scripts/worker_wait_uuid_idle_exec.sh','scripts/worker_teal_gpu0_uuid_exec.sh')
OUTPUT='ffnds-derived/e0_teal_author_benchmark_probe_20261029/attempt1/worker/results'
SOURCES={
 'kernels/__init__.py':'01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b',
 'kernels/sparse_gemv.py':'21183499ac9eb06c57365023388920721636b5bc0d2acf5d91c95fd5e48c37a3',
 'kernels/compile_wrapper.py':'7e1b86358171923d3ac658476b483c5b1dc199601b84976264d5814933239d7c',
 'kernels/kernel_utils.py':'c47ca739752a4012b027493f9ad987a1b27cf2f4b0739d714b0ae1f881efd44f',
 'scripts/benchmark_gemv.py':'27dddfe933e5359cbdcbffe246a41a5be541cddd59e4698043f02bd2e8d111af',
 'pyproject.toml':'b04eb6ea56b66b5cf6a342e769795046341f6d73e3edb5a7be53688f738aac81'}
DESIGN={'claim':'BOUNDED_AUTHOR_BENCHMARK_ADAPTED_NOT_LLM_REPRO_OR_WAN', 'source_commit':UPSTREAM,
 'gpu_compute_capability':[8,0], 'gpu_name_contains':'A100', 'dtype':'float16',
 'input_shape':[1,1,4096], 'weight_shape':[14336,4096], 'seed':42,
 'sparsity_levels':[0.0,0.5], 'thresholds':[0.0,0.25], 'sparsity_bins':[0,9],
 'cpu_affinity':[0,1], 'physical_gpu_index':0, 'quantiles':[0.5,0.2,0.8],
 'warmup_ms':25,'sparse_rep_ms':1000,'dense_rep_ms':100,'flush_bytes':256000000,
 'measurement':'TRITON_3_0_DO_BENCH_EQUIVALENT_EVENT_FLUSH_GUARDED_ADAPTATION',
 'scope':'PAIRED_SAME_INPUT_TWO_LEVELS_NOT_FULL_AUTHOR_SWEEP',
 'no_compile_no_cuda_graph':True,'relative_l2_max':0.015,'new_training':False}
ENV_FILES=('torch/__init__.py','torch/version.py','torch/_C.cpython-311-x86_64-linux-gnu.so',
           'triton/__init__.py','triton/runtime/autotuner.py','triton/runtime/jit.py',
           'triton/backends/nvidia/compiler.py','triton/runtime/build.py','setuptools/__init__.py','wheel/__init__.py','msgspec/__init__.py','triton/testing.py')
ENV_DISTRIBUTIONS=('torch','triton','msgspec','numpy','nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12',
                   'nvidia-cublas-cu12','nvidia-nvjitlink-cu12','setuptools','wheel')
RESOURCES={'GPU':1,'CPU_threads':2,'wall_seconds':900,'outer_seconds':1020,
 'GPU_reserved_bytes':4<<30,'new_HF_bytes':1<<30,'free_disk_bytes':10<<30,'automatic_retry':False}
LIMITS={'benchmark_arms':4,'max_repeats_per_arm':30000,'max_warmups_per_arm':5000,
 'max_python_fn_calls_per_arm':35006,'correctness_sparse':2,'correctness_dense':2,
 'video_generations':0,'model_loads':0,'training_updates':0}



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


V1_HF=Path('/home/xiang/.cache/huggingface')
V1_WORKER=V1_HF/'ffnds-derived/e0_teal_author_fp16_kernel_probe_20261029/attempt1/worker'
SMOKE=V1_HF/'ffnds-derived/teal_author_env_v2_20261029/IMPORT_SMOKE.json'
V1_REPO=('tools/e0_teal_author_kernel_probe.py','tests/test_e0_teal_author_kernel_probe.py',
         'config/e0_teal_author_kernel_probe_20261029_pending_review_v1.json',
         'tools/e0_teal_author_kernel_probe_v2.py','tests/test_e0_teal_author_kernel_probe_v2.py',
         'config/e0_teal_author_kernel_probe_v2_20261029_pending_review_v1.json')
def failure_snapshot():
    files=[V1_WORKER/'EXIT_RECEIPT.json',V1_WORKER/'results/RUN.json',V1_WORKER/'child.stderr.log']
    ex=read(files[0]);run=read(files[1]);stderr=files[2].read_text()
    need(ex.get('child_returncode')==1 and ex.get('process_exited') is True and ex.get('timeout') is False
         and ex.get('run_sha256')==sha(files[1])=='0e7e6bb9213ceafa2ea07c3bf1ec3f3a395d0ea2bc152bc18644d25884ad5a2f'
         and run.get('error_type')=='ModuleNotFoundError' and run.get('error')=="No module named 'setuptools'"
         and all(v==0 for v in run['observed'].values()) and all(v==0 for v in run['predebited'].values())
         and "No module named 'setuptools'" in stderr,'V1 failure evidence mismatch')
    return {'cause':'TRITON_IMPORT_MISSING_SETUPTOOLS','retry':'NEW_SCOPE_SINGLE_ATTEMPT',
            'files':[bind(f,V1_HF) for f in files]}

V2_WORKER=V1_HF/'ffnds-derived/e0_teal_author_fp16_kernel_probe_v2_20261029/attempt1/worker'
def v2_snapshot():
    files=[V2_WORKER/'EXIT_RECEIPT.json',V2_WORKER/'results/RUN.json']
    ex=read(files[0]);run=read(files[1]);need(ex.get('child_returncode')==0 and ex.get('process_exited') is True
        and ex.get('run_sha256')==sha(files[1])=='fd83d1ebb9d8d5e82068ea8a3149558a3fcecbac55fca628bdc99ad90873280a'
        and run.get('status')=='COMPLETE_AUTHOR_KERNEL_PROBE_NOT_LLM_REPRO'
        and run.get('observed',{}).get('sparse_forward')==62,'V2 actual closure differs')
    return {'identity':'EAGER_INSTRUMENTED_DO_NOT_INTERPRET_AS_AUTHOR_BENCHMARK',
            'files':[bind(f,V1_HF) for f in files]}

def draft(venv=None):
    bindings=[bind(ROOT/SOURCE/p,ROOT) for p in sorted(SOURCES)]
    need(all(b['sha256']==SOURCES[b['path'][len(SOURCE)+1:]] for b in bindings),'Pinned author source differs')
    return {'id':ID,'status':'DRAFT_REVIEW_REQUESTED','assets_finalized':False,'upstream_repository':'https://github.com/FasterDecoding/TEAL',
      'design':DESIGN,'resources':RESOURCES,'limits':LIMITS,'source_files':bindings,
      'repo_sources':[bind(ROOT/p,ROOT) for p in (SELF,TEST,*V1_REPO,*HELPERS)],'output_HF_relative':OUTPUT,
      'v1_failure':failure_snapshot(),'engineering_import_smoke':bind(SMOKE,V1_HF),'v2_closed':v2_snapshot(),
      'environment':{'python_min':'3.11','torch_required':'2.4.0','triton_required':'3.0.0',
                     'distribution_policy':'AUTHOR_PINNED_TORCH_Triton_GPU_TARGET_SM80','worker_config_sha256':None,
                     'prepared_environment':environment_snapshot(venv) if venv else None,'cuda_required':'12.1','driver_version':None,
                     'monitor_binary':None},
      'review':None,'automatic_retry':False}


def fingerprint(plan):
    return canonical({k:plan[k] for k in ('id','design','resources','limits','source_files','repo_sources','output_HF_relative','environment','v1_failure','engineering_import_smoke','v2_closed')})


def validate(plan,frozen=False):
    need(plan['id']==ID and plan['design']==DESIGN and plan['resources']==RESOURCES and plan['limits']==LIMITS
         and plan['output_HF_relative']==OUTPUT and plan['automatic_retry'] is False,'Fixed experiment differs')
    need({b['path'] for b in plan['repo_sources']}=={SELF,TEST,*V1_REPO,*HELPERS} and len(plan['repo_sources'])==2+len(V1_REPO)+len(HELPERS),'Source closure differs')
    need({b['path'] for b in plan['source_files']}=={SOURCE+'/'+p for p in SOURCES} and len(plan['source_files'])==len(SOURCES)
         and all(b['sha256']==SOURCES[b['path'][len(SOURCE)+1:]] for b in plan['source_files']), 'Pinned upstream source differs')
    need(plan['environment']['python_min']=='3.11' and plan['environment']['torch_required']=='2.4.0'
         and plan['environment']['triton_required']=='3.0.0' and plan['environment']['cuda_required']=='12.1'
         and plan['environment']['distribution_policy']=='AUTHOR_PINNED_TORCH_Triton_GPU_TARGET_SM80','Author compiler environment differs')
    need(plan['v1_failure']==failure_snapshot(),'V1 actual failure evidence changed')
    need(plan['v2_closed']==v2_snapshot(),'V2 actual closed evidence changed')
    need(plan['engineering_import_smoke']==bind(SMOKE,V1_HF),'Engineering import smoke changed')
    smoke=json.loads(SMOKE.read_text())
    need(smoke.get('kind')=='ENGINEERING_IMPORT_SMOKE_NO_TENSORS_NO_MODEL_NO_CUDA_COMPILATION'
         and smoke.get('cuda_visible_devices')=='' and smoke.get('returncode')==0,'Engineering import smoke missing')
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
         and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('reviewer_disposition') in ('APPROVE','REVISE')
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

def correctness_call(row,key,fn,synchronize,persist):
    row[key]={'predebited':1,'completed':'UNKNOWN'};persist()
    value=fn();synchronize()
    row[key]['completed']=1;persist()
    return value

def guarded_do_bench(torch,fn,row,persist,rep):
    """Triton 3.0 do_bench's numerical/timing sequence with a bounded loop guard.

    The six estimate calls and the remaining loops are predebited separately.
    Loop completion is recorded after synchronization, not instrumented inside
    the measured event interval. Interrupted batches have UNKNOWN actual calls.
    """
    fn();torch.cuda.synchronize()
    cache=torch.empty(int(256e6//4),dtype=torch.int,device='cuda')
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        cache.zero_();fn()
    end.record();torch.cuda.synchronize()
    estimate=start.elapsed_time(end)/5
    row.update(completed_python_calls=6,estimate_ms=estimate);persist()
    need(math.isfinite(estimate) and estimate>0,'Invalid benchmark estimate')
    n_warmup=max(1,int(DESIGN['warmup_ms']/estimate));n_repeat=max(1,int(rep/estimate))
    row.update(warmup_calls=n_warmup,repeat_calls=n_repeat,rep_ms=rep)
    need(n_warmup<=LIMITS['max_warmups_per_arm'] and n_repeat<=LIMITS['max_repeats_per_arm']
         and 6+n_warmup+n_repeat<=LIMITS['max_python_fn_calls_per_arm'],'Dynamic benchmark budget exceeded')
    row.update(predebited_python_calls=6+n_warmup+n_repeat,partial_batch_actual_calls='UNKNOWN_UNTIL_SYNC');persist()
    starts=[torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    ends=[torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for _ in range(n_warmup):fn()
    for i in range(n_repeat):
        cache.zero_();starts[i].record();fn();ends[i].record()
    torch.cuda.synchronize()
    times=torch.tensor([s.elapsed_time(e) for s,e in zip(starts,ends)],dtype=torch.float)
    values=torch.quantile(times,torch.tensor(DESIGN['quantiles'],dtype=torch.float)).tolist()
    need(len(values)==3 and all(math.isfinite(v) and v>=0 for v in values),'Invalid timing quantiles')
    row.update(completed_python_calls=6+n_warmup+n_repeat,partial_batch_actual_calls='COMPLETE',quantiles_ms=values)
    persist()

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
      'benchmark_arms':[],'correctness':[],'autotune_internal_GPU_launches':'UNKNOWN',
      'physical_kernel_launches':'UNKNOWN','launch_spec_sha256':scope['launch_spec_sha256'],
      'automatic_retry':False,'LLM_reproduction':False,'WAN_transfer':False}
    def persist():write(out/'RUN.json',report)
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
        need(len(sparse.splitk_sparse_gemv_kernel.configs)==16,'Author autotune configuration differs')
        x=torch.rand(DESIGN['input_shape'],device='cuda',dtype=torch.float16)-0.5
        w=(torch.rand(DESIGN['weight_shape'],device='cuda',dtype=torch.float16)-0.5).T.contiguous().T
        wt=w.T.contiguous();need(w.stride()==(1,14336),'Author layout differs')
        def check():
            authorize(plan,digest);monitor.check();need(time.monotonic()-started<=RESOURCES['wall_seconds'],'Wall budget exceeded')
            need(torch.cuda.max_memory_reserved()<=RESOURCES['GPU_reserved_bytes'],'GPU memory budget exceeded')
            need(shutil.disk_usage(out).free>=RESOURCES['free_disk_bytes'],'Disk reserve exceeded')
            need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())<=RESOURCES['new_HF_bytes'],'Output budget exceeded')
        for level,threshold,bin_id in zip(DESIGN['sparsity_levels'],DESIGN['thresholds'],DESIGN['sparsity_bins']):
            check();need(int(abs(level)//0.05)==bin_id,'Author sparsity bin differs')
            row={'level':level,'threshold':threshold};report['correctness'].append(row);persist()
            y=correctness_call(row,'sparse_call',lambda:sparse.splitk_sparse_gemv(x,w,threshold,bin_id),torch.cuda.synchronize,persist)
            ref=correctness_call(row,'dense_call',lambda:torch.matmul(x*(x.abs()>threshold),wt),torch.cuda.synchronize,persist)
            den=float(torch.linalg.vector_norm(ref.float()));diff=float(torch.linalg.vector_norm(y.float()-ref.float()))
            rel=diff/den if den else None
            row.update(zero_fraction=float((x.abs()<=threshold).float().mean()),relative_l2=rel,
                 config=str(sparse.splitk_sparse_gemv_kernel.best_config));persist()
            need(rel is not None and math.isfinite(rel) and rel<=DESIGN['relative_l2_max'],'Correctness mismatch')
        # Four independent author-style do_bench arms. The guarded adaptation changes
        # only budget checks and persistent accounting outside measured event intervals.
        arms=[('sparse',0.0,0.0,0),('dense',0.0,0.0,0),('dense',0.5,0.25,9),('sparse',0.5,0.25,9)]
        for provider,level,threshold,bin_id in arms:
            check()
            fn=(lambda t=threshold,b=bin_id:sparse.splitk_sparse_gemv(x,w,t,b)) if provider=='sparse' else (lambda:torch.matmul(x,wt))
            row={'provider':provider,'level':level,'state':'STARTED','predebited_python_calls':6,
                 'completed_python_calls':'UNKNOWN','physical_launches':'UNKNOWN'}
            report['benchmark_arms'].append(row);persist()
            guarded_do_bench(torch,fn,row,persist,DESIGN['sparse_rep_ms'] if provider=='sparse' else DESIGN['dense_rep_ms'])
            row['state']='COMPLETE';persist();check()
        need(len(report['benchmark_arms'])==4 and all(a['state']=='COMPLETE' and a['completed_python_calls']==a['predebited_python_calls'] for a in report['benchmark_arms']),'Benchmark accounting incomplete')
        report.update(status='COMPLETE_AUTHOR_BENCHMARK_ADAPTED_NOT_LLM_REPRO',
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
         and report.get('status')=='COMPLETE_AUTHOR_BENCHMARK_ADAPTED_NOT_LLM_REPRO'
         and report.get('design')==DESIGN and report.get('environment',{}).get('prepared_environment')==plan['environment']['prepared_environment']
         and report.get('environment',{}).get('GPU_driver_version')==plan['environment']['driver_version']
         and report.get('environment',{}).get('cuda_runtime')==plan['environment']['cuda_required']
         and report.get('GPU_compute_capability')==[8,0] and 'A100' in report.get('GPU_name','')
         and 0<=report.get('wall_seconds',-1)<=RESOURCES['wall_seconds']
         and report.get('peak_GPU_reserved_bytes',RESOURCES['GPU_reserved_bytes']+1)<=RESOURCES['GPU_reserved_bytes']
         and not report.get('assigned_GPU_resources',{}).get('violations')
         and report.get('assigned_GPU_resources',{}).get('sampled_isolation_pass') is True,'Actual run closure differs')
    rows=report.get('benchmark_arms',[])
    need(len(rows)==4 and [(r['provider'],r['level']) for r in rows]==[('sparse',0.0),('dense',0.0),('dense',0.5),('sparse',0.5)]
         and all(r['state']=='COMPLETE' and r['predebited_python_calls']==r['completed_python_calls']==6+r['warmup_calls']+r['repeat_calls']
                 and 1<=r['repeat_calls']<=LIMITS['max_repeats_per_arm'] and 1<=r['warmup_calls']<=LIMITS['max_warmups_per_arm']
                 and len(r['quantiles_ms'])==3 for r in rows)
         and len(report.get('correctness',[]))==2 and all(0<=r['relative_l2']<=DESIGN['relative_l2_max']
             and r['sparse_call']==r['dense_call']=={'predebited':1,'completed':1} for r in report['correctness']), 'Result coverage differs')
    for b in plan['source_files']+plan['repo_sources']:bound(ROOT,b)
    return {'id':ID,'status':'PASS_CLOSED_AUTHOR_BENCHMARK_PROBE','RUN':binding,'exit':bind(args.exit,hf),'model_calls':0,'GPU_calls':0}


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
