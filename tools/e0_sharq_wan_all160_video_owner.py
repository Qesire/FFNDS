"""Independently reviewed owner for same-master Wan SharQ four-video parts.

The draft command is metadata only. A GPU run requires an actual exited CPU
preflight, exact reviewer and ACTIVE rows, private job, source and worker
identity. No model or numerical library is imported on module import.
"""
import argparse,copy,hashlib,importlib,importlib.metadata,json,math,os,resource,shutil,signal,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_teal_wan_all160_video_v3 as prior
import e0_sharq_wan_all160_video_plan as source_plan
import e0_sharq_wan_all160_runtime as rt
import e0_sharq_wan_all160_video_driver as driver
import e0_sharq_wan_all160_video_evidence as evidence
import e0_teal_llama3_author_checkpoint_convert as output_support
native,vm,io=prior.native,prior.vm,prior.io
need=native.need
HF=Path('/home/xiang/.cache/huggingface')
SELF='tools/e0_sharq_wan_all160_video_owner.py';TEST='tests/test_e0_sharq_wan_all160_video_owner.py'
AUTHOR=HF/'ffnds-derived/sharq_sm120_author_build_20261030/source'
ELF=HF/'ffnds-derived/sharq_sm120_author_build_20261030/build_sm120a/sharq_ops.so'
PYTHON=HF/'ffnds-derived/sharq_sm120_author_build_20261030/venv/bin/python'
BASE='config/e0_s2_new12_sm120_fixed_depth_part0_20261029_v1.json'
KERNEL_PROTOCOL='config/e0_sharq_sm120_wan_ffn_pair_probe_v2_20261030_pending_review_v2.json'
KERNEL_OUT='ffnds-derived/e0_sharq_sm120_wan_ffn_pair_probe_v2_20261030/attempt1/worker/results'
GROUP='E0_SHARQ_WAN_ALL160_SM120_GROUP_20261030'
RES={'GPU':1,'CPU_threads':4,'wall_seconds':7200,'outer_seconds':7320,'new_HF_bytes':2<<30,'video_bytes_max':32<<20,'GPU_reserved_max_bytes':80<<30,'host_available_min_bytes':128<<30,'free_disk_min_bytes':8<<30,'max_concurrent_SM120_models':2,'session_max_GPUs':4,'automatic_retry':False}
CPU={'GPU':0,'CPU_threads':4,'wall_seconds':900,'outer_seconds':1020,'new_HF_bytes':16<<20,'automatic_retry':False}
VERIFY={'GPU':0,'CPU_threads':1,'wall_seconds':1800,'outer_seconds':1920,'read_payload_bytes_max':2<<30,'new_HF_bytes':8<<20,'automatic_retry':False}
ASSET_READ_MAX=512<<20
DESIGN={'sources':[1040,979],'seed':42,'modes':['bf16','bf16_cache','dense32','sharq'],'steps':4,'blocks_per_step':40,'linear_calls_per_video':320,'neutral':'same-device two BF16 actual videos, all 10 states and video byte identity','Q2':'current arm post-GELU','source_role':'EXPOSED_DEVELOPMENT_SINGLE_SEED','asset_hash_read_max_bytes_per_CPU_and_GPU_phase':ASSET_READ_MAX,'cache':'160 static shared W packed in second BF16 neutral from original buffer on GPU; both candidates reuse cache','original_W1_W2_prefetch':'UNCHANGED_AND_PAID','quality_acceptance':False,'automatic_retry':False}

def scope_id(part):need(type(part) is int and part in (0,1),'invalid part');return f'E0_SHARQ_WAN_ALL160_VIDEO_20261030_PART{part}'
def outpath(part):return f'ffnds-derived/e0_sharq_wan_all160_video_20261030/part{part}/attempt1/worker/results'
def run_ids(part):return [f'sharq_full160_ord{DESIGN["sources"][part]}_seed42_{x}' for x in ('bf16_old','bf16_new','dense32','sharq')]
def limits():
 result={k:(v if k in prior.base.acquisition.SETUP else v//2*4) for k,v in vm.native_original_limits.items()}
 result.update(core_CPU_reports=0,prefetch_transfer_observations=648,original_FFN_prefetch_observations=648,stage_CUDA_event_pairs=28,
 state_tensor_D2H=40,state_tensor_D2H_bytes=4*math.prod(native.VIDEO['latent_shape'])*30,state_output_writes=40,video_geometry_checks=4,
  authority_checks=4,neutral_state_checks=10,neutral_video_checks=1,noise_pair_checks=4,
  sharq_static_W_materializations=160,sharq_static_W_materialization_bytes=160*5120*13824*2,sharq_static_W_quantizations=160,
  sharq_static_W_cache_predebit_bytes=80*sum(n*(k//2)+2*((n//128+1)*128*(k//16))+4 for k,n in rt.LINEARS.values()),
  dense32_prepare=320,dense32_gemm=320,sharq_prepare=320,sparse_gemm=320,dense_residual_gemm=320,output_scale=640,bias_add=640)
 return result

def fingerprint(p):return native.canonical({k:v for k,v in p.items() if k not in ('status','assets_finalized','review')})
def kernel_parent():
 p=native.read(ROOT/KERNEL_PROTOCOL);folder=HF/KERNEL_OUT;run=folder/'RUN.json';ex=folder.parent/'EXIT_RECEIPT.json';r=native.read(run);e=native.read(ex)
 need(r['id']=='E0_SHARQ_SM120_WAN_FFN_PAIR_PROBE_V2_20261030' and r['status']=='COMPLETE_WAN_FFN_OPERATOR_EXIT_PENDING' and e['process_exited'] is True and e['child_returncode']==0 and e['timeout'] is False and e['automatic_retry'] is False and e['run_sha256']==native.sha(run) and e['protocol_sha256']==r['protocol_sha256']==native.sha(ROOT/KERNEL_PROTOCOL),'actual V2 operator EXIT0 required')
 import e0_sharq_wan_ffn_probe_v2 as operator
 closed=operator.inspect_closed(ROOT/KERNEL_PROTOCOL);need(closed['status']=='PASS_CLOSED_SINGLE_SOURCE_PAIRED_FFN','V2 operator closed proof required')
 return {'protocol':native.binding(ROOT/KERNEL_PROTOCOL,ROOT),'RUN':native.binding(run,HF),'exit':native.binding(ex,HF),'closed':closed}

def draft(part,profile,affinity):
 scope_id(part);need(profile.is_file(),'actual private worker profile required');template=native.read(ROOT/BASE);sources,pool=prior.source_inputs();case=copy.deepcopy(pool[DESIGN['sources'][part]]);kp=kernel_parent()
 need(len(affinity)==4 and len(set(affinity))==4 and all(type(x) is int and x>=0 for x in affinity),'four CPU affinity indices required')
 need(native.sha(profile)==template['execution_private_worker_config_sha256'],'worker profile must match actual raw-master proof')
 keys=('models','model_view_root','model_view_metadata','runtime_versions','local_transformers','private_worker_config_sha256','runtime_overlay','execution_raw_master_CPU_proof')
 p={k:copy.deepcopy(template[k]) for k in keys};p['execution_private_worker_config_sha256']=native.sha(profile)
 paths={x['path'] for x in template['repo_sources']}|{b['path'] for s in sources for b in s['repo_sources']}|set(source_plan.CODE)|set(source_plan.CONFIGS)|{SELF,TEST,BASE,KERNEL_PROTOCOL,'tools/e0_teal_wan_all160_video_v3.py','tools/e0_s2_new12_sm120_fixed_depth_videos.py','tools/e0_s2_causal_worker_proof_v5.py','tools/worker_identity.py','tools/durable_worker_job.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py','scripts/worker_teal_assigned_uuid_exec.sh'}
 old=native.read(ROOT/KERNEL_PROTOCOL);paths|={b['path'] for b in old['repo_sources']};paths.add(old['assets']['cpu_protocol']);monitor=Path('/usr/bin/nvidia-smi');p.update(id=scope_id(part),part=part,status='DRAFT_REVIEW_REQUESTED',assets_finalized=False,review=None,case=case,source_bank=sources[0]['source_bank'],resources=RES,CPU_preflight_resources=CPU,verification_resources=VERIFY,limits=limits(),design=DESIGN,output_HF_relative=outpath(part),run_ids=run_ids(part),cpu_affinity=affinity,kernel_parent=kp,author_software=old['software'],execution_monitor_binary={'path':str(monitor),'bytes':monitor.stat().st_size,'sha256':native.sha(monitor)},repo_sources=[native.binding(ROOT/x,ROOT) for x in sorted(paths)],automatic_retry=False)
 return p

def validate(p,frozen=False):
 part=p.get('part');need(p.get('id')==scope_id(part) and p.get('resources')==RES and p.get('CPU_preflight_resources')==CPU and p.get('verification_resources')==VERIFY and p.get('limits')==limits() and p.get('design')==DESIGN and p.get('output_HF_relative')==outpath(part) and p.get('run_ids')==run_ids(part) and p['case']['native_source_ordinal']==DESIGN['sources'][part] and p['kernel_parent']==kernel_parent(),'fixed scope differs')
 template=native.read(ROOT/BASE);sources,pool=prior.source_inputs();old=native.read(ROOT/KERNEL_PROTOCOL)
 keys=('models','model_view_root','model_view_metadata','runtime_versions','local_transformers','private_worker_config_sha256','runtime_overlay','execution_raw_master_CPU_proof','execution_private_worker_config_sha256')
 need(all(p[k]==template[k] for k in keys) and p['case']==pool[DESIGN['sources'][part]] and p['source_bank']==sources[0]['source_bank'] and p['author_software']==old['software'],'actual master/source/software identity differs')
 paths={x['path'] for x in template['repo_sources']}|{b['path'] for s in sources for b in s['repo_sources']}|set(source_plan.CODE)|set(source_plan.CONFIGS)|{SELF,TEST,BASE,KERNEL_PROTOCOL,'tools/e0_teal_wan_all160_video_v3.py','tools/e0_s2_new12_sm120_fixed_depth_videos.py','tools/e0_s2_causal_worker_proof_v5.py','tools/worker_identity.py','tools/durable_worker_job.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py','scripts/worker_teal_assigned_uuid_exec.sh'}|{b['path'] for b in old['repo_sources']}|{old['assets']['cpu_protocol']}
 need({x['path'] for x in p['repo_sources']}==paths and len(p['repo_sources'])==len(paths) and p.get('automatic_retry') is False,'source closure differs')
 need(len(p.get('cpu_affinity',[]))==4 and len(set(p['cpu_affinity']))==4 and all(type(x) is int and x>=0 for x in p['cpu_affinity']),'CPU affinity differs')
 if frozen:need(p.get('status')=='FROZEN_PENDING_REGISTRATION' and p.get('assets_finalized') is True and p.get('review') is None,'scope not frozen')

def authorize(p,digest):
 validate(p,True);a=native.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==p['id']]
 need(a['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True and len(rows)==1,'ACTIVE required');s=rows[0]
 need(s.get('protocol_sha256')==digest and s.get('resources')==RES and s.get('limits')==limits() and s.get('run_ids')==p['run_ids'] and s.get('max_gpu_cards')==1 and s.get('session_max_GPUs')==4 and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('native_group_id')==GROUP and s.get('native_group_active_slots') in (0,1),'ACTIVE scope differs')
 active=a['experiment_authority']['current_user_authorized_scopes'];need(sum(x.get('native_group_active_slots',0) for x in active if x.get('native_group_id')==GROUP and x.get('execution_status')=='RUNNING_FIXED_SCOPE')<=2,'SM120 model group limit exceeded')
 review=native.read(native.bound(ROOT,s['review']));need(review.get('scope_id')==p['id'] and review.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and review.get('state')=='REVIEWER_RECHECK_APPROVED' and review.get('recheck_disposition')=='APPROVE' and review.get('author_agent_id') and review.get('reviewer_agent_id') and review['author_agent_id']!=review['reviewer_agent_id'] and review.get('responses_to_review') and review.get('adversarial_counterexample') and review.get('approval_scope_fingerprint')==fingerprint(p) and review.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']} and review.get('cpu_preflight_spec_sha256')==s.get('cpu_preflight_spec_sha256') and review.get('gpu_launch_spec_sha256')==s.get('gpu_launch_spec_sha256'),'independent exact review required')
 for k in ('author_review','independent_review','reviewer_recheck'):native.bound(ROOT,review[k])
 return s

def verify_job(a,p,s,kind):
 digest=native.sha(a.protocol);key='cpu_preflight_spec_sha256' if kind=='preflight' else 'gpu_launch_spec_sha256';need(a.job_spec and native.sha(a.job_spec)==s[key],'private job SHA differs');j=native.read(a.job_spec);argv=j['argv'];need(len(argv)>1 and argv[0]=='env' and argv[1].startswith('FFNDS_WORKER_CONFIG='),'worker profile absent');profile=Path(argv[1].split('=',1)[1]);need(profile.is_absolute() and native.sha(profile)==p['execution_private_worker_config_sha256'],'profile differs')
 aff=','.join(map(str,p['cpu_affinity']));env=['env',argv[1],'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin'];child=['taskset','-c',aff,str(PYTHON),'-I','-B',str(ROOT/SELF)];tail=['--protocol',str(a.protocol.resolve()),'--hf-root',str(HF),'--local-root',str(a.local_root.resolve()),'--job-spec',str(a.job_spec.resolve())];out=HF/p['output_HF_relative']
 if kind=='preflight':
  dest=out.parent.parent/'cpu_preflight/worker/results';expected=env+['CUDA_VISIBLE_DEVICES=']+child+['preflight','--out',str(dest)]+tail
  need(a.out.resolve()==dest and Path(j['output_dir']).resolve()==dest.parent and Path(j['run_json']).resolve()==dest/'RUN.json' and j['external_timeout_seconds']==CPU['outer_seconds'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU job differs')
 else:
  gpu=os.environ.get('FFNDS_PHYSICAL_GPU','');uuid=os.environ.get('CUDA_VISIBLE_DEVICES','');need(gpu in ('0','1','2','3') and uuid.startswith('GPU-'),'one assigned UUID required')
  expected=env+['bash',str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),gpu,uuid]+child+['run']+tail
  need(Path(j['output_dir']).resolve()==out.parent and Path(j['run_json']).resolve()==out/'RUN.json' and j['external_timeout_seconds']==RES['outer_seconds'],'GPU job differs')
  b=p['execution_monitor_binary'];monitor=Path(b['path']);need(shutil.which('nvidia-smi')==str(monitor) and monitor.stat().st_size==b['bytes'] and native.sha(monitor)==b['sha256'],'monitor differs')
 need(argv==expected and j['id']==p['id'] and j['protocol_sha256']==digest and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and Path(j['hf_root']).resolve()==HF and sorted(os.sched_getaffinity(0))==sorted(p['cpu_affinity']) and sys.flags.isolated,'exact worker/job/affinity differs')
 import worker_identity;worker_identity.require_worker()

class AssetHashBudget:
 """Bound full SHA reads and preserve conservative in-flight status on failure."""
 def __init__(self,report,persist):self.report,self.persist=report,persist;self.used=0
 def __enter__(self):
  import local_python_package_mirror as mirror
  self.mirror=mirror;self.old_sha=native.sha;self.old_digest=mirror.digest
  self.report['asset_hash_reads']={'reserved_bytes':ASSET_READ_MAX,'completed_bytes':0,'inflight':'UNKNOWN_UNTIL_PHASE_COMPLETES','metadata_read_bytes':'UNKNOWN_SMALL_METADATA_EXCLUDING_HASHED_FILES'};self.persist()
  native.sha=self.hash;mirror.digest=self.hash;return self
 def hash(self,path):
  p=Path(path);size=p.stat().st_size;need(size>=0 and self.used+size<=ASSET_READ_MAX,'asset SHA read budget exceeded');h=hashlib.sha256();read=0
  with p.open('rb') as f:
   while True:
    chunk=f.read(min(8<<20,size-read+1));read+=len(chunk);need(read<=size,'asset grew during SHA read')
    if not chunk:break
    h.update(chunk)
  need(read==size and p.stat().st_size==size,'asset SHA size changed');self.used+=read;return h.hexdigest()
 def __exit__(self,typ,value,tb):
  native.sha=self.old_sha;self.mirror.digest=self.old_digest
  self.report['asset_hash_reads'].update(completed_bytes=self.used,inflight='NONE' if typ is None else 'UNKNOWN_PARTIAL_FILE');self.persist()

def charged_assets(p,local,report,persist):
 with AssetHashBudget(report,persist):return check_assets(p,local)

def check_assets(p,local_root):
 for b in p['repo_sources']:native.bound(ROOT,b)
 for b in p['author_software']['author_modules']:native.bound(HF,b)
 native.bound(HF,p['author_software']['sharq_elf']);native.bound(HF,p['case']['image'])
 import e0_s2_sm120_bf16_worker_proof as proof
 proof.require_closed_proof(p,HF);native.model_metadata(p);return native.verify_local_package(p,{'repo':ROOT,'hf':HF},local_root)

def preflight(a):
 p=native.read(a.protocol);digest=native.sha(a.protocol);s=authorize(p,digest);verify_job(a,p,s,'preflight');need(not {'torch','numpy'}&set(sys.modules) and not a.out.exists() and a.out.resolve().is_relative_to(HF),'fresh numerical-free preflight required');a.out.mkdir(parents=True);started=time.monotonic();r={'id':p['id'],'protocol_sha256':digest,'job_spec_sha256':s['cpu_preflight_spec_sha256'],'status':'STARTED_CPU_NO_RETRY','model_calls':0,'GPU_calls':0,'automatic_retry':False};native.write(a.out/'RUN.json',r);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU preflight wall')));signal.alarm(CPU['wall_seconds'])
 try:
  charged_assets(p,a.local_root,r,lambda:native.write(a.out/'RUN.json',r));need(not {'torch','numpy'}&set(sys.modules),'numerical import during preflight');r.update(status='PASS_SHARQ_WAN_BYTE_PREFLIGHT',source_closure_sha256=native.canonical(p['repo_sources']),numerical_imports=False)
 except BaseException as e:r.update(status='FAILED_CPU_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:signal.alarm(0);r['wall_seconds']=time.monotonic()-started;native.write(a.out/'RUN.json',r)

def require_cpu(p,digest,s):
 proof=s['cpu_preflight'];r=native.read(native.bound(HF,proof['RUN']));e=native.read(native.bound(HF,proof['exit']));vm.exited(e,proof['RUN'],digest,p['id']);need(e.get('spec_sha256')==r.get('job_spec_sha256')==s['cpu_preflight_spec_sha256'] and r.get('status')=='PASS_SHARQ_WAN_BYTE_PREFLIGHT' and r.get('source_closure_sha256')==native.canonical(p['repo_sources']) and r.get('model_calls')==r.get('GPU_calls')==0 and r.get('numerical_imports') is False and r.get('wall_seconds',CPU['wall_seconds']+1)<=CPU['wall_seconds'],'actual CPU preflight EXIT0 required');return proof

def run(a):
 started=time.monotonic();p=native.read(a.protocol);digest=native.sha(a.protocol);s=authorize(p,digest);verify_job(a,p,s,'run');need(s.get('execution_status')=='RUNNING_FIXED_SCOPE' and s.get('native_group_active_slots')==1,'registered running model slot required');cpu=require_cpu(p,digest,s);need(not {'torch','numpy'}&set(sys.modules),'fresh numerical process required')
 need({k:importlib.metadata.version(k) for k in p['runtime_versions']}==p['runtime_versions'],'runtime versions differ');out=HF/p['output_HF_relative'];need(not out.exists(),'fresh GPU attempt required');out.mkdir(parents=True)
 native.ID,native.LIMITS,native.RUN_IDS,native.RESOURCE=p['id'],limits(),tuple(p['run_ids']),RES
 report={'id':p['id'],'protocol_sha256':digest,'job_spec_sha256':s['gpu_launch_spec_sha256'],'status':'STARTED_NO_RETRY','source_ordinal':p['case']['native_source_ordinal'],'seed':42,'quality_acceptance':False,'CPU_preflight':cpu,'kernel_parent':p['kernel_parent'],'automatic_retry':False,'phases':[],'operation_ledger':{},'checkpoint_input_identity':'two original BF16 masters; actual loader logical tensor bytes and physical mmap/page-cache read bytes are not directly observed','checkpoint_file_bytes':sum(x['bytes'] for x in p['models'] if x['kind'] in ('high','low'))}
 emergency=output_support.reserve_emergency(out.parent);monitor=None;counter=io.WrittenBytes(out,RES['new_HF_bytes'],lambda:monitor.check() if monitor else None)
 for rid in p['run_ids']:counter.hold(str(counter.path(out/rid/'output.mp4')),RES['video_bytes_max'])
 journal=counter.open_journal(out/'OPERATIONS.jsonl');events=counter.open_journal(out/'ACTIONS.jsonl');ledger=vm.Ledger(report['operation_ledger'],lambda e:journal.write(json.dumps(e,separators=(',',':'))+'\n'))
 def emit(e):events.write(json.dumps(e,separators=(',',':'))+'\n')
 def persist():counter.write_json(out/'RUN.json',report)
 def check():
  if monitor:monitor.check()
  counter.check();usage=output_support.worker_usage(out.parent);summary=counter.summary();need(usage['apparent_bytes']+summary['unconsumed_reserved_bytes']+(1<<20)<=RES['new_HF_bytes'],'whole worker output soft budget exceeded');available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
  need(available>=RES['host_available_min_bytes'] and shutil.disk_usage(out).free>=RES['free_disk_min_bytes'] and time.monotonic()-started<=RES['wall_seconds'],'host/disk/wall budget exceeded')
  if 'torch' in sys.modules:need(sys.modules['torch'].cuda.max_memory_reserved()<=RES['GPU_reserved_max_bytes'],'GPU reserved budget exceeded')
 persist();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall limit')));signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('output monitor failure')));signal.alarm(RES['wall_seconds']);output_monitor=output_support.UsageMonitor(out.parent,RES['new_HF_bytes'],RES['free_disk_min_bytes']);output_monitor.start()
 from e0_b_final_block_capture_local_package import CheckedPackageFinder
 finder=None;no_hif4=native.NoHiF4();file_guard=io.single_file_bound();file_guard_entered=False
 try:
  local,package_rows=charged_assets(p,a.local_root,report,persist);finder=CheckedPackageFinder(local,package_rows);file_guard.__enter__();file_guard_entered=True
  sys.dont_write_bytecode=True;sys.path[:0]=[str(local),str(ROOT/'.runtime/code/LightX2V'),str(AUTHOR),str(AUTHOR/'model')];sys.meta_path[:0]=[no_hif4,finder]
  os.environ.update(DTYPE='BF16',SENSITIVE_LAYER_DTYPE='BF16',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SHARQ_KERNEL_BUILD=str(ELF.parent))
  import torch
  from safetensors.torch import save_file
  from ub1_consumer_runtime import tensor_sha
  from run_ub1_consumer_videos import inspect_video
  from e0_sharq_wan_ffn_probe_v2 import AuthorAPI
  reservation,monitor,report['CUDA_reservation']=vm.reserved_startup.reserve_before_assets(torch)
  need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(12,0) and 'RTX PRO 6000' in torch.cuda.get_device_name(0),'SM120 device differs');torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
  quant=importlib.import_module('quantize');extension=quant.load_sharq_ops();need(Path(quant.__file__).resolve()==(AUTHOR/'model/quantize.py').resolve() and Path(sys.modules['hif4'].__file__).resolve()==(AUTHOR/'model/hif4.py').resolve() and Path(sys.modules['sharq_loader'].__file__).resolve()==(AUTHOR/'sharq_loader.py').resolve() and Path(extension.__file__).resolve()==ELF.resolve(),'author import differs');api=AuthorAPI(torch,quant,extension,None);check()
  with native.OriginalGuard(torch,p,ledger) as guard:
   setup=time.perf_counter();pipe=native.build_pipeline(p);torch.cuda.synchronize();report.update(raw_model_setup_seconds=time.perf_counter()-setup,original_loader_paths=guard.loaded);persist();check()
   runtime=rt.SameMasterFFNRuntime(pipe,guard,api,quant,ledger,native.operand_check,sync=torch.cuda.synchronize,emit=emit)
   def capture_factory(rid):
    folder=out/rid;folder.mkdir();states={}
    def save(value,name,key):
     def write():
      dtype=torch.float32 if key in ('raw_FP32_noise_files','native_FP32_flow_files') else torch.bfloat16;need(name not in states and value.dtype==dtype and list(value.shape)==native.VIDEO['latent_shape'],'state geometry differs');path=folder/(name+'.safetensors');counter.reserve_payload(value.numel()*value.element_size())
      def d2h():
       cpu=value.detach().cpu().contiguous();need(bool(torch.isfinite(cpu).all()),'nonfinite state');return cpu
      cpu=ledger.call(rid,{'state_tensor_D2H':1,'state_tensor_D2H_bytes':value.numel()*value.element_size()},d2h)
      try:ledger.call(rid,{'state_output_writes':1},save_file,{'flow' if key=='native_FP32_flow_files' else 'latents':cpu},str(path))
      finally:
       if path.exists():counter.observe_external(path)
      states[name]={'root':'hf',**counter.binding(path,HF),'raw_tensor_sha256':tensor_sha(cpu),'shape':list(cpu.shape),'dtype':str(cpu.dtype).removeprefix('torch.')};counter.write_json(folder/'STATES.json',states);return states[name]
     return ledger.call(rid,{key:1},write)
    return save,states
   def inspect(rid,path):
    inspected=ledger.call(rid,{'video_geometry_checks':1},inspect_video,path,native.VIDEO);counter.observe_external(path,hold=str(counter.path(path)),known=inspected);return inspected
   auth_index={'value':0}
   def phase_authorize():
    i=auth_index['value'];need(i<4,'authority check budget exceeded');auth_index['value']+=1;at=time.perf_counter();result=ledger.call(p['run_ids'][i],{'authority_checks':1},authorize,p,digest);emit({'event':'authority_check','run_id':p['run_ids'][i],'wall_seconds':time.perf_counter()-at});return result
   result=driver.execute_part(torch=torch,pipe=pipe,guard=guard,runtime=runtime,vm=vm,run_ids=p['run_ids'],image_path=HF/p['case']['image']['path'],prompt=p['case']['prompt'],output_paths=[out/r/'output.mp4' for r in p['run_ids']],capture_factory=capture_factory,inspect_video=inspect,check=check,authorize=phase_authorize,emit=emit,gate=lambda rid,cost,fn:ledger.call(rid,cost,fn))
   report.update(phases=result['phases'],same_device_BF16_neutral=result['same_device_BF16_neutral'],paired_noise=result['paired_noise']);ledger.finish();need(result['phases'][-1]['weight_cache']['cache_entries']==160,'shared W cache incomplete');check();report['status']='COMPLETE_SHARQ_WAN_ALL160_EXIT_PENDING';persist()
 except BaseException as e:report.update(status='FAILED_SHARQ_WAN_ALL160_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:
  signal.alarm(0);monitor_error=None
  if monitor:
   try:monitor.close();report['GPU_monitor']=monitor.report();need(report['GPU_monitor']['sampled_isolation_pass'] and not report['GPU_monitor']['violations'],'GPU monitor violation')
   except BaseException as e:monitor_error=e;report.update(status='FAILED_SHARQ_WAN_ALL160_PRESERVED_NO_RETRY',resource_error=str(e))
  for x in (finder,no_hif4):
   if x in sys.meta_path:sys.meta_path.remove(x)
  for rid in p['run_ids']:
   path=counter.path(out/rid/'output.mp4')
   if path not in counter.files and path.exists():
    try:counter.observe_external(path,hold=str(path))
    except BaseException as e:counter.accounting_errors.append(str(e))
  output_status=output_monitor.stop();report.update(wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,output_monitor=output_status,output_accounting=counter.summary(),operation_journal=counter.binding(out/'OPERATIONS.jsonl',HF),action_journal=counter.binding(out/'ACTIONS.jsonl',HF))
  if output_status['fault'] or not output_status['thread_stopped']:report['status']='FAILED_SHARQ_WAN_ALL160_PRESERVED_NO_RETRY';monitor_error=RuntimeError('output monitor failure')
  try:persist()
  except BaseException as error:
   output_support.emergency_failure(emergency,{'id':p['id'],'status':report['status'],'predebited':report['operation_ledger'].get('predebited'),'completed':report['operation_ledger'].get('observed')},'final RUN persist failed '+type(error).__name__);raise
  finally:
   journal.close();events.close()
   if file_guard_entered:file_guard.__exit__(None,None,None)
  if monitor_error:raise monitor_error

def authorize_verification(p,digest,a):
 verify_id=p['id']+'_VERIFY';active=native.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==verify_id];need(len(rows)==1,'independent verifier ACTIVE required');row=rows[0]
 need(row.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and row.get('protocol_sha256')==digest and row.get('resources')==VERIFY and row.get('max_gpu_cards')==0 and row.get('parent_scope_id')==p['id'],'verifier registration differs')
 review=native.read(native.bound(ROOT,row['review']));need(review.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and review.get('scope_id')==verify_id and review.get('state')=='REVIEWER_RECHECK_APPROVED' and review.get('recheck_disposition')=='APPROVE' and review.get('author_agent_id') and review.get('reviewer_agent_id') and review['author_agent_id']!=review['reviewer_agent_id'] and review.get('approval_scope_fingerprint')==fingerprint(p) and review.get('verification_spec_sha256')==row.get('verification_spec_sha256'),'verifier independent review differs')
 for key in ('author_review','independent_review','reviewer_recheck'):native.bound(ROOT,review[key])
 need(a.job_spec and native.sha(a.job_spec)==row['verification_spec_sha256'],'verifier spec SHA differs');j=native.read(a.job_spec);argv=j['argv'];need(len(argv)>1 and argv[0]=='env' and argv[1].startswith('FFNDS_WORKER_CONFIG='),'verifier profile absent');profile=Path(argv[1].split('=',1)[1]);need(profile.is_absolute() and native.sha(profile)==p['execution_private_worker_config_sha256'],'verifier profile differs')
 cpus=sorted(os.sched_getaffinity(0));need(len(cpus)==1 and cpus[0] in p['cpu_affinity'] and sys.flags.isolated and os.environ.get('CUDA_VISIBLE_DEVICES')=='','verifier isolation/CPU differs');out=HF/p['output_HF_relative'];dest=out.parent.parent/'verification/worker/results';expected=['env',argv[1],'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PATH=/usr/bin:/bin','CUDA_VISIBLE_DEVICES=','taskset','-c',str(cpus[0]),str(PYTHON),'-I','-B',str(ROOT/SELF),'verify-closed','--protocol',str(a.protocol.resolve()),'--hf-root',str(HF),'--job-spec',str(a.job_spec.resolve()),'--run',str(out/'RUN.json'),'--exit',str(out.parent/'EXIT_RECEIPT.json'),'--verify-out',str(dest/'RUN.json')]
 need(argv==expected and j['id']==verify_id and j['protocol_sha256']==digest and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and Path(j['hf_root']).resolve()==HF and Path(j['output_dir']).resolve()==dest.parent and Path(j['run_json']).resolve()==dest/'RUN.json' and j['external_timeout_seconds']==VERIFY['outer_seconds'] and a.verify_out.resolve()==dest/'RUN.json','verifier exact job differs')
 import worker_identity;worker_identity.require_worker();return row

def verify_closed(a):
 p=native.read(a.protocol);validate(p,True);digest=native.sha(a.protocol);scope=authorize_verification(p,digest,a);out=HF/p['output_HF_relative'];need(a.run.resolve()==(out/'RUN.json').resolve() and a.exit.resolve()==(out.parent/'EXIT_RECEIPT.json').resolve(),'actual RUN/EXIT paths differ');r=native.read(a.run);e=native.read(a.exit);record=evidence.verify_record(p,r,e,native.sha(a.run),digest)
 need(r['wall_seconds']<=RES['wall_seconds'] and r['output_monitor']['samples']>0 and not r['output_monitor']['fault'] and r['output_monitor']['thread_stopped'] and r['output_accounting']['peak_tracked_bytes_plus_reserved_and_atomic_temp']<=RES['new_HF_bytes'] and r['output_accounting']['unconsumed_reserved_bytes']==0 and not r['output_accounting']['accounting_errors'] and r['same_device_BF16_neutral'] is True and r['paired_noise'] is True,'resource/neutral proof differs')
 # Rehash the exact 40 states and four videos once; input bytes are bounded by
 # the registered output cap. This verifier is read-only and its caller must
 # account for this additional CPU/I/O separately from the GPU run.
 assets=[]
 for phase in r['phases']:
  assets.extend(phase['states'].values());v=phase['video'];path=Path(v['path']).resolve();need(path.is_relative_to(HF),'video escaped HF');assets.append({'path':str(path.relative_to(HF)),'bytes':v['bytes'],'sha256':v['sha256']})
 assets.extend((r['operation_journal'],r['action_journal']));expected=out.parent.parent/'verification/worker/results/RUN.json';need(len(assets)==46 and sum(x['bytes'] for x in assets)<=VERIFY['read_payload_bytes_max'] and a.verify_out and a.verify_out.resolve()==expected and not a.verify_out.exists() and len(os.sched_getaffinity(0))==1,'verification read budget/output/CPU differs')
 a.verify_out.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic();v={'id':p['id']+'_VERIFY','protocol_sha256':digest,'job_spec_sha256':scope['verification_spec_sha256'],'parent_scope_id':p['id'],'status':'STARTED_READ_ONLY_VERIFY','predebited_bytes':0,'completed_bytes':0,'inflight':'NONE','automatic_retry':False,'read_accounting':'46 output artifact SHA reads only; small protocol/RUN/EXIT/source identity reads additional'};native.write(a.verify_out,v);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('verification wall')));signal.alarm(VERIFY['wall_seconds'])
 try:
  for b in assets:
   v['predebited_bytes']+=b['bytes'];v['inflight']={'path':b['path'],'bytes':b['bytes'],'completed':'UNKNOWN'};native.write(a.verify_out,v);native.bound(HF,b);v['completed_bytes']+=b['bytes'];v['inflight']='NONE';native.write(a.verify_out,v)
  v.update(status='PASS_CLOSED_SHARQ_WAN_ALL160_VIDEO',RUN=native.binding(a.run,HF),exit=native.binding(a.exit,HF),record=record,quality_acceptance=False)
 except BaseException as error:v.update(status='FAILED_READ_ONLY_VERIFY_NO_RETRY',error_type=type(error).__name__,error=str(error)[:500]);raise
 finally:signal.alarm(0);v['wall_seconds']=time.monotonic()-started;native.write(a.verify_out,v)
 return v

def inspect_verification(a):
 p=native.read(a.protocol);validate(p,True);digest=native.sha(a.protocol);out=HF/p['output_HF_relative'];run=out.parent.parent/'verification/worker/results/RUN.json';ex=run.parent.parent/'EXIT_RECEIPT.json';r=native.read(run);e=native.read(ex);need(r['id']==p['id']+'_VERIFY' and r['status']=='PASS_CLOSED_SHARQ_WAN_ALL160_VIDEO' and r['protocol_sha256']==digest and e['id']==r['id'] and e['process_exited'] is True and e['child_returncode']==0 and e['timeout'] is False and e['automatic_retry'] is False and e['protocol_sha256']==digest and e['run_sha256']==native.sha(run) and e['spec_sha256']==r['job_spec_sha256'],'actual verifier EXIT0 required');return {'status':'PASS_ACTUAL_EXIT_CLOSED_SHARQ_WAN_ALL160_VIDEO','id':p['id'],'verification_RUN':native.binding(run,HF),'verification_exit':native.binding(ex,HF),'quality_acceptance':False}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','freeze','inspect','preflight','run','verify-closed','inspect-verification'));ap.add_argument('--part',type=int);ap.add_argument('--worker-profile',type=Path);ap.add_argument('--cpu-affinity',default='0,1,2,3');ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--verify-out',type=Path);ap.add_argument('--hf-root',type=Path,default=HF);ap.add_argument('--local-root',type=Path);ap.add_argument('--job-spec',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path);a=ap.parse_args();need(a.hf_root.resolve()==HF,'HF root differs')
 if a.action=='draft':need(a.out and a.worker_profile and not a.out.exists(),'fresh draft/profile required');native.write(a.out,draft(a.part,a.worker_profile,[int(x) for x in a.cpu_affinity.split(',')]))
 elif a.action=='freeze':
  need(a.out and not a.out.exists(),'fresh frozen file required');p=native.read(a.protocol);validate(p);need(p['status']=='DRAFT_REVIEW_REQUESTED' and a.local_root,'review draft and local source root required');check_assets(p,a.local_root);p['status']='FROZEN_PENDING_REGISTRATION';p['assets_finalized']=True;native.write(a.out,p)
 elif a.action=='inspect':p=native.read(a.protocol);validate(p);print(json.dumps({'id':p['id'],'status':p['status'],'fingerprint':fingerprint(p),'requires':'independent review and ACTIVE'}))
 elif a.action=='preflight':preflight(a)
 elif a.action=='run':run(a)
 elif a.action=='verify-closed':print(json.dumps(verify_closed(a),indent=2))
 else:print(json.dumps(inspect_verification(a),indent=2))
if __name__=='__main__':main()
