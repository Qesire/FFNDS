"""Four-source, five-arm A100 BF16 full-video owner; review and ACTIVE required."""
import argparse,copy,hashlib,importlib.metadata,json,os,resource,shutil,signal,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_sharq_wan_all160_video_owner_v2 as prior
import chipmunk_input_video_tap as capture
import chipmunk_input_video_driver as driver
import e0_chipmunk_wan_global_p64_owner_v2 as previous
import e0_teal_llama3_author_checkpoint_convert as support
native,vm,io=prior.native,prior.vm,prior.io;need=native.need;HF=prior.HF;PYTHON=ROOT/'.venv/bin/python'
SELF='tools/e0_chipmunk_input_video_owner.py';TEST='tests/test_e0_chipmunk_input_video_owner.py';BASE='config/e0_teal_wan_all160_video_v3_part0_20261029_pending_review_v1.json';SPATIAL='config/e0_s2_spatial_student_collect_part{}_20261029_v2.json';SOURCE_PLAN='config/e0_chipmunk_input_video_sources_20261030_v1.json';GROUP='E0_CHIPMUNK_INPUT_VIDEO_GROUP_20261030'
RES={'GPU':1,'CPU_threads':4,'wall_seconds':7200,'outer_seconds':7320,'new_HF_bytes':512<<20,'video_bytes_max':32<<20,'GPU_reserved_max_bytes':76<<30,'host_available_min_bytes':128<<30,'max_RSS_bytes':128<<30,'free_disk_min_bytes':12<<30,'max_concurrent_A100_models':2,'session_max_GPUs':4,'single_file_bytes_max':1<<30,'automatic_retry':False}
CPU={'GPU':0,'CPU_threads':4,'wall_seconds':900,'outer_seconds':1020,'new_HF_bytes':16<<20,'automatic_retry':False}
VERIFY={'GPU':0,'CPU_threads':1,'wall_seconds':3600,'outer_seconds':3720,'read_payload_bytes_max':512<<20,'new_HF_bytes':8<<20,'automatic_retry':False}
DESIGN={'source_ordinals':[898,96,1040,979],'seed':42,'steps':4,'blocks':40,'grid':[21,29,52],'modes':list(driver.MODES),'reference':'both cache Dense match NONE; candidates match NONE through high step0; old-source H matches prior HIGH_ONLY','actions':'high step1 P64 readonly old h or selected old input recomputation; current gate; other 120 sites dense','H_Y_cache_logical_bytes':40*31668*(13824+5120)*2,'X_shadow_Y_cache_logical_bytes':40*(2*31668*5120*2+64*13824*2),'original_prefetch':'PRESERVED','source_role':'EXPOSED_DEVELOPMENT_SINGLE_SEED_NO_BLIND_CLAIM','quality_acceptance':False,'allocator_policy':'after high step1 release, synchronize, empty unused cache, synchronize; no peak reset'}
SOURCES={SELF,TEST,'tools/e0_chipmunk_input_video_jobs.py','tools/chipmunk_input_video_tap.py','tests/test_chipmunk_input_video_tap.py','tools/chipmunk_input_video_driver.py','tests/test_chipmunk_input_video_driver.py','tools/chipmunk_input_state_math.py','tools/chipmunk_fullm_proxy_math.py','tools/e0_sharq_wan_all160_video_owner_v2.py','tools/e0_teal_wan_fullm_capture.py','tools/e0_s2_causal_worker_proof_v5.py','tools/worker_identity.py','tools/durable_worker_job.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py','scripts/worker_teal_assigned_uuid_exec.sh',BASE}

def sid(part):need(type(part) is int and part in (0,1,2,3),'invalid part');return f'E0_CHIPMUNK_INPUT_VIDEO_20261030_PART{part}'
def outpath(part):return f'ffnds-derived/e0_chipmunk_input_video_20261030/part{part}/attempt1/worker/results'
def run_ids(part):return [f'chipmunk_input_video_ord{DESIGN["source_ordinals"][part]}_seed42_{x}' for x in driver.MODES]
def compare_reference(q,path,expected_bytes,expected_sha,actual_sha):
 q=Path(q).resolve();path=Path(path).resolve();need(q.is_relative_to(HF.resolve()) and q.is_file() and q.stat().st_size==expected_bytes and actual_sha==expected_sha,'reference SHA/size differs')
 identity=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
 need(q!=path,'reference and output path identical')
 with q.open('rb') as a,path.open('rb') as b:
  before_a=os.fstat(a.fileno());before_b=os.fstat(b.fileno())
  need((before_a.st_dev,before_a.st_ino)!=(before_b.st_dev,before_b.st_ino),'reference and output inode identical')
  need(identity(q.stat())==identity(before_a) and identity(path.stat())==identity(before_b),'path changed before comparison')
  same=True
  while True:
   left=a.read(4<<20);right=b.read(4<<20)
   if left!=right:same=False;break
   if not left:break
  return bool(same and identity(os.fstat(a.fileno()))==identity(before_a) and identity(os.fstat(b.fileno()))==identity(before_b) and identity(q.stat())==identity(before_a) and identity(path.stat())==identity(before_b))

def verify_reference_metadata(phases, controls, old_high=None):
 prefix={'raw_noise_FP32','initial_noise','flow_step0','latent_step0'}
 need([x['mode'] for x in phases]==list(driver.MODES),'phase identity')
 dense=phases[0];expected={(mode,name) for mode in driver.MODES[1:] for name in (prefix if mode.startswith('high_') else set(dense['states']))}
 if old_high is not None:expected|={('old_high',name) for name in dense['states']}
 need(len(controls)==len(expected) and {(x['mode'],x['state']) for x in controls}==expected and all(x['equal'] is True for x in controls),'byte controls missing')
 for phase in phases[1:]:
  names=prefix if phase['mode'].startswith('high_') else set(dense['states'])
  for name in names:need(all(phase['states'][name][k]==dense['states'][name][k] for k in ('sha256','bytes','raw_tensor_sha256','shape','dtype')),'state neutral differs')
  if phase['mode'] in ('h_cache_dense','x_cache_dense'):need(all(phase['video'][k]==dense['video'][k] for k in ('sha256','bytes')),'cache dense video differs')
 if old_high is not None:
  h=phases[3]
  for name in dense['states']:need(all(h['states'][name][k]==old_high['states'][name][k] for k in ('sha256','bytes','raw_tensor_sha256','shape','dtype')),'old H state differs')
  need(all(h['video'][k]==old_high['video'][k] for k in ('sha256','bytes')),'old H video differs')

def limits(part):
 fixed={'original_expert_loads','raw_Default_MM_loads','buffer_Default_MM_loads','T5_raw_Default_MM_loads','T5_buffer_Default_MM_loads'}
 d={key:(v if key in fixed else v*5//2) for key,v in vm.native_original_limits.items()}
 m,dmodel,h,k=31668,5120,13824,5120;state=__import__('math').prod(native.VIDEO['latent_shape'])*30
 d.update(core_CPU_reports=0,prefetch_transfer_observations=810,stage_CUDA_event_pairs=35,state_tensor_D2H=50,state_tensor_D2H_bytes=5*state,state_output_writes=50,video_geometry_checks=5,video_file_hash_calls=5,video_file_hash_reserved_bytes=5*(32<<20),authority_checks=5,state_file_hash_calls=50,state_file_hash_reserved_bytes=5*state+(50<<20),state_raw_sha_calls=50,state_raw_sha_bytes=5*state,p64_row_seed_and_H2D=1,cache_h_clone=80,cache_h_clone_bytes=80*m*h*2,cache_x_clone=80,cache_x_clone_bytes=80*m*dmodel*2,cache_shadow=80,cache_shadow_bytes=80*64*h*2,cache_y_clone=160,cache_y_clone_bytes=160*m*dmodel*2,cache_release_sync=160,cache_boundary_memory_observer=160,expert_boundary_cleanup=4,weight_materializations=160,weight_materialization_logical_bytes=80*2*dmodel*h*2,p64_score=80,p64_gather_input=80,p64_gather_old_h=40,p64_shadow_read=40,p64_W1_gemm=80,p64_gelu=80,p64_rms=80,p64_select=80,p64_input_gather_logical_bytes=80*64*dmodel*2,p64_old_h_gather_logical_bytes=40*64*h*2,mask_sha_calls=80,mask_sha_logical_bytes=80*k*8,selected_old_h=80,extra_old_W1=40,old_selected_logical_bytes=80*m*k*2,old_weight_gather_logical_bytes=40*(k*dmodel+k)*2,packed_action=80,packed_W1_gemm=80,packed_W2_gemm_fp32=80,packed_clone_y=80,packed_gather_w1=80,packed_gather_w2=80,packed_gather_bias=80,packed_selected_gelu=80,packed_delta_round=80,packed_accumulate_y=80,packed_y_clone_bytes=80*m*dmodel*2,packed_delta_bytes=80*m*k*2,packed_weight_gather_bytes=80*(k*dmodel+dmodel*k+k)*2,reference_state_checks=38 if part<2 else 28,reference_video_checks=3 if part<2 else 2,reference_state_read_reservation=(38 if part<2 else 28)*(18<<20),reference_video_read_reservation=(3 if part<2 else 2)*(64<<20))
 return d

def fingerprint(p):return native.canonical({k:v for k,v in p.items() if k not in ('status','assets_finalized','review')})
def source_entry(part):
 bank=native.read(ROOT/SOURCE_PLAN);need(bank['parts'][part]['part']==part and bank['parts'][part]['case']['native_source_ordinal']==DESIGN['source_ordinals'][part],'source identity');return bank['parts'][part]
def parent_proof(part):
 entry=source_entry(part);proto=native.read(native.bound(ROOT,entry['parent_protocol']));refs=entry['parent_completion'];run=native.read(native.bound(HF,refs['actual_RUN']));ex=native.read(native.bound(HF,refs['actual_EXIT']));vr=native.read(native.bound(HF,refs['verification_RUN']));ve=native.read(native.bound(HF,refs['verification_EXIT']));need(run['id']==ex['id']==entry['parent_scope_id'] and ex['child_returncode']==ve['child_returncode']==0 and ex['process_exited'] and ve['process_exited'] and ex['run_sha256']==refs['actual_RUN']['sha256'] and ve['run_sha256']==refs['verification_RUN']['sha256'] and run['source_ordinal']==entry['case']['native_source_ordinal'] and vr['parent_scope_id']==entry['parent_scope_id'] and vr['status']==('PASS_CLOSED_CHIPMUNK_P64_FACTORIAL_VIDEO' if part<2 else 'PASS_CLOSED_SHARQ_WAN_ALL160_VIDEO') and vr['protocol_sha256']==run['protocol_sha256']==ex['protocol_sha256']==ve['protocol_sha256']==entry['parent_protocol']['sha256'] and ve['id']==vr['id']==entry['parent_scope_id']+'_VERIFY' and vr['predebited_bytes']==vr['completed_bytes'] and vr['inflight']=='NONE','source parent closure');need(part>=2 or (run['phases'][1]['run_id']==entry['legacy_high_only_run_id'] and run['phases'][1]['mode']=='high_only'),'legacy high identity');return entry
def source_paths(part):
 entry=source_entry(part);b=native.read(ROOT/BASE);q=native.read(native.bound(ROOT,entry['parent_protocol']));return {x['path'] for x in b['repo_sources']}|{x['path'] for x in q['repo_sources']}|SOURCES|{SOURCE_PLAN,entry['parent_protocol']['path']}
def draft(part,profile,affinity):
 sid(part);base=native.read(ROOT/BASE);entry=parent_proof(part);need(profile.is_file() and native.sha(profile)==base['execution_private_worker_config_sha256'] and len(affinity)==4 and len(set(affinity))==4,'profile/affinity differs');keys=('models','model_view_root','model_view_metadata','runtime_versions','local_transformers','private_worker_config_sha256','runtime_overlay','execution_raw_master_CPU_proof');p={k:copy.deepcopy(base[k]) for k in keys};p.update(id=sid(part),part=part,status='DRAFT_REVIEW_REQUESTED',assets_finalized=False,review=None,case=copy.deepcopy(entry['case']),source_parent=entry,execution_private_worker_config_sha256=native.sha(profile),cpu_affinity=affinity,resources=RES,CPU_preflight_resources=CPU,verification_resources=VERIFY,limits=limits(part),design=DESIGN,run_ids=run_ids(part),output_HF_relative=outpath(part),repo_sources=[native.binding(ROOT/x,ROOT) for x in sorted(source_paths(part))],execution_monitor_binary=prior.native.binding(Path('/usr/bin/nvidia-smi'),Path('/usr/bin')),automatic_retry=False);p['execution_monitor_binary']['path']='/usr/bin/nvidia-smi';return p
def validate(p,frozen=False):
 part=p['part'];base=native.read(ROOT/BASE);entry=parent_proof(part);need(p['id']==sid(part) and p['resources']==RES and p['CPU_preflight_resources']==CPU and p['verification_resources']==VERIFY and p['limits']==limits(part) and p['design']==DESIGN and p['run_ids']==run_ids(part) and p['output_HF_relative']==outpath(part) and p['case']==entry['case'] and p['source_parent']==entry and p['automatic_retry'] is False,'fixed scope differs');keys=('models','model_view_root','model_view_metadata','runtime_versions','local_transformers','private_worker_config_sha256','runtime_overlay','execution_raw_master_CPU_proof','execution_private_worker_config_sha256');need(all(p[k]==base[k] for k in keys) and {x['path'] for x in p['repo_sources']}==source_paths(part) and len(p['repo_sources'])==len(source_paths(part)) and len(p['cpu_affinity'])==len(set(p['cpu_affinity']))==4,'source/master/affinity differs')
 if frozen:need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['assets_finalized'] is True and p['review'] is None,'frozen protocol required')

def authorize(p,d):
 validate(p,True);a=native.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==p['id']];need(a['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True and len(rows)==1,'ACTIVE required');s=rows[0];need(s.get('protocol_sha256')==d and s.get('resources')==RES and s.get('limits')==limits(p['part']) and s.get('run_ids')==p['run_ids'] and s.get('max_gpu_cards')==1 and s.get('session_max_GPUs')==4 and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('native_group_id')==GROUP and s.get('native_group_active_slots') in (0,1),'ACTIVE differs');need(sum(x.get('native_group_active_slots',0) for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('native_group_id')==GROUP and x.get('execution_status')=='RUNNING_FIXED_SCOPE')<=2,'two model slot limit');r=native.read(native.bound(ROOT,s['review']));need(r.get('scope_id')==p['id'] and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE' and r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('approval_scope_fingerprint')==fingerprint(p) and r.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']} and r.get('cpu_preflight_spec_sha256')==s.get('cpu_preflight_spec_sha256') and r.get('gpu_launch_spec_sha256')==s.get('gpu_launch_spec_sha256') and r.get('responses_to_review') and r.get('adversarial_counterexample'),'independent review differs')
 for k in ('author_review','independent_review','reviewer_recheck'):native.bound(ROOT,r[k])
 return s

def verify_job(a,p,s,kind):
 key='cpu_preflight_spec_sha256' if kind=='preflight' else 'gpu_launch_spec_sha256';need(a.job_spec and native.sha(a.job_spec)==s[key],'job SHA differs');j=native.read(a.job_spec);v=j['argv'];need(len(v)>1 and v[0]=='env' and v[1].startswith('FFNDS_WORKER_CONFIG='),'profile argument missing');profile=Path(v[1].split('=',1)[1]);need(profile.is_absolute() and native.sha(profile)==p['execution_private_worker_config_sha256'],'profile differs');aff=','.join(map(str,p['cpu_affinity']));env=['env',v[1],'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','PATH=/usr/bin:/bin'];child=['taskset','-c',aff,str(PYTHON),'-I','-B',str(ROOT/SELF)];tail=['--protocol',str(a.protocol.resolve()),'--hf-root',str(HF),'--local-root',str(a.local_root.resolve()),'--job-spec',str(a.job_spec.resolve())];out=HF/p['output_HF_relative']
 if kind=='preflight':
  dest=out.parent.parent/'cpu_preflight/worker/results';expected=env+['CUDA_VISIBLE_DEVICES=']+child+['preflight','--out',str(dest)]+tail;need(a.out.resolve()==dest and Path(j['output_dir']).resolve()==dest.parent and Path(j['run_json']).resolve()==dest/'RUN.json' and j['external_timeout_seconds']==CPU['outer_seconds'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='','preflight job differs')
 else:
  gpu=os.environ.get('FFNDS_PHYSICAL_GPU','');uuid=os.environ.get('CUDA_VISIBLE_DEVICES','');need(gpu in ('0','1','2','3') and uuid.startswith('GPU-'),'one UUID required');expected=env+['bash',str(ROOT/'scripts/worker_teal_assigned_uuid_exec.sh'),gpu,uuid]+child+['run']+tail;need(Path(j['output_dir']).resolve()==out.parent and Path(j['run_json']).resolve()==out/'RUN.json' and j['external_timeout_seconds']==RES['outer_seconds'],'GPU paths differ');b=p['execution_monitor_binary'];monitor=Path(b['path']);need(shutil.which('nvidia-smi')==str(monitor) and monitor.stat().st_size==b['bytes'] and native.sha(monitor)==b['sha256'],'monitor differs')
 need(v==expected and j['id']==p['id'] and j['protocol_sha256']==native.sha(a.protocol) and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and Path(j['hf_root']).resolve()==HF and sorted(os.sched_getaffinity(0))==sorted(p['cpu_affinity']) and sys.flags.isolated,'exact job differs');import worker_identity;worker_identity.require_worker()
def check_assets(p,local):
 for b in p['repo_sources']:native.bound(ROOT,b)
 native.bound(HF,p['case']['image']);import e0_s2_causal_worker_proof_v5 as proof
 proof.require_closed_proof(p,HF);native.model_metadata(p);return native.verify_local_package(p,{'repo':ROOT,'hf':HF},local)
def charged_assets(p,local,r,persist):
 with prior.AssetHashBudget(r,persist):return check_assets(p,local)
def preflight(a):
 p=native.read(a.protocol);digest=native.sha(a.protocol);s=authorize(p,digest);verify_job(a,p,s,'preflight');need(not {'torch','numpy'}&set(sys.modules) and not a.out.exists() and a.out.resolve().is_relative_to(HF),'fresh numerical-free CPU preflight');a.out.mkdir(parents=True);start=time.monotonic();r={'id':p['id'],'protocol_sha256':digest,'job_spec_sha256':s['cpu_preflight_spec_sha256'],'status':'STARTED_CPU_NO_RETRY','model_calls':0,'GPU_calls':0,'automatic_retry':False};native.write(a.out/'RUN.json',r);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU wall')));signal.alarm(CPU['wall_seconds'])
 try:charged_assets(p,a.local_root,r,lambda:native.write(a.out/'RUN.json',r));need(not {'torch','numpy'}&set(sys.modules),'numerical import');r.update(status='PASS_CHIPMUNK_INPUT_VIDEO_BYTE_PREFLIGHT',source_closure_sha256=native.canonical(p['repo_sources']),numerical_imports=False)
 except BaseException as e:r.update(status='FAILED_CPU_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:signal.alarm(0);r['wall_seconds']=time.monotonic()-start;native.write(a.out/'RUN.json',r)
def require_cpu(p,d,s):
 proof=s['cpu_preflight'];r=native.read(native.bound(HF,proof['RUN']));e=native.read(native.bound(HF,proof['exit']));vm.exited(e,proof['RUN'],d,p['id']);need(e.get('spec_sha256')==r.get('job_spec_sha256')==s['cpu_preflight_spec_sha256'] and r.get('status')=='PASS_CHIPMUNK_INPUT_VIDEO_BYTE_PREFLIGHT' and r.get('source_closure_sha256')==native.canonical(p['repo_sources']) and r.get('model_calls')==r.get('GPU_calls')==0 and r.get('numerical_imports') is False and r.get('wall_seconds',CPU['wall_seconds']+1)<=CPU['wall_seconds'],'actual CPU preflight EXIT0 required');return proof

def final_memory(torch):
 try:
  return capture.memory_snapshot(torch) if torch.cuda.is_initialized() else {'status':'UNKNOWN_CUDA_NOT_INITIALIZED'}
 except BaseException as e:return {'status':'UNKNOWN','error_type':type(e).__name__}

def run(a):
 started=time.monotonic();p=native.read(a.protocol);digest=native.sha(a.protocol);s=authorize(p,digest);verify_job(a,p,s,'run');need(s.get('execution_status')=='RUNNING_FIXED_SCOPE' and s.get('native_group_active_slots')==1,'running slot required');cpu=require_cpu(p,digest,s);need(not {'torch','numpy'}&set(sys.modules),'fresh numerical process');need({k:importlib.metadata.version(k) for k in p['runtime_versions']}==p['runtime_versions'],'runtime versions differ');out=HF/p['output_HF_relative'];need(not out.exists(),'fresh output');out.mkdir(parents=True);native.ID,native.LIMITS,native.RUN_IDS,native.RESOURCE=p['id'],limits(p['part']),tuple(p['run_ids']),RES
 report={'id':p['id'],'protocol_sha256':digest,'job_spec_sha256':s['gpu_launch_spec_sha256'],'status':'STARTED_NO_RETRY','source_ordinal':p['case']['native_source_ordinal'],'seed':42,'quality_acceptance':False,'CPU_preflight':cpu,'automatic_retry':False,'phases':[],'captures':[],'current_phase':None,'current_state':None,'operation_ledger':{},'checkpoint_input_identity':'original BF16 two masters; physical mmap read bytes UNKNOWN'};emergency=support.reserve_emergency(out.parent);monitor=None;counter=io.WrittenBytes(out,RES['new_HF_bytes'],lambda:monitor.check() if monitor else None)
 for rid in p['run_ids']:counter.hold(str(counter.path(out/rid/'output.mp4')),RES['video_bytes_max'])
 journal=counter.open_journal(out/'OPERATIONS.jsonl');events=counter.open_journal(out/'ACTIONS.jsonl');ledger=vm.Ledger(report['operation_ledger'],lambda x:journal.write(json.dumps(x,separators=(',',':'))+'\n'));persist=lambda:counter.write_json(out/'RUN.json',report)
 def check():
  if monitor:monitor.check()
  counter.check();usage=support.worker_usage(out.parent);need(usage['apparent_bytes']+counter.summary()['unconsumed_reserved_bytes']+(1<<20)<=RES['new_HF_bytes'],'worker output limit');available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RES['max_RSS_bytes'] and available>=RES['host_available_min_bytes'] and shutil.disk_usage(out).free>=RES['free_disk_min_bytes'] and time.monotonic()-started<=RES['wall_seconds'],'host/disk/wall limit')
  if 'torch' in sys.modules:need(sys.modules['torch'].cuda.max_memory_reserved()<=RES['GPU_reserved_max_bytes'],'GPU memory limit')
 persist();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall limit')));signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('output monitor')));signal.alarm(RES['wall_seconds']);out_monitor=support.UsageMonitor(out.parent,RES['new_HF_bytes'],RES['free_disk_min_bytes']);out_monitor.start();finder=None;no_hif4=native.NoHiF4();import e0_teal_wan_fullm_capture as fullm_guard
 file_guard=fullm_guard.single_file_bound(RES['single_file_bytes_max']);file_guard_entered=False
 try:
  from e0_b_final_block_capture_local_package import CheckedPackageFinder
  local,rows=charged_assets(p,a.local_root,report,persist);finder=CheckedPackageFinder(local,rows);file_guard.__enter__();file_guard_entered=True;sys.dont_write_bytecode=True;sys.path[:0]=[str(local),str(ROOT/'.runtime/code/LightX2V')];sys.meta_path[:0]=[no_hif4,finder];os.environ.update(DTYPE='BF16',SENSITIVE_LAYER_DTYPE='BF16',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  import torch
  from safetensors.torch import save_file
  kernels=None
  from ub1_consumer_runtime import tensor_sha
  from run_ub1_consumer_videos import inspect_video
  reservation,monitor,report['CUDA_reservation']=vm.reserved_startup.reserve_before_assets(torch);need(torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0) and 'A100' in torch.cuda.get_device_name(0),'A100 required');torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
  report['precision']={'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0),'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'float32_matmul_precision':torch.get_float32_matmul_precision()};persist()
  with native.OriginalGuard(torch,p,ledger) as guard:
   begin=time.perf_counter();pipe=native.build_pipeline(p);torch.cuda.synchronize();report.update(raw_model_setup_seconds=time.perf_counter()-begin,original_loader_paths=guard.loaded);persist();check()
   def progress(event,data):events.write(json.dumps({'event':event,'run_id':data['run_id']},separators=(',',':'))+'\n');prior.update_progress(report,event,data,persist)
   old_high=native.read(native.bound(HF,p['source_parent']['parent_completion']['actual_RUN']))['phases'][1] if p['part']<2 else None;mode_for_id=dict(zip(p['run_ids'],driver.MODES));report['prefix_byte_controls']=[]
   def capture_factory(rid):
    folder=out/rid;folder.mkdir();states={}
    def save(value,name,key):
     def write():
      prior.mark_state(report,rid,name,'STARTED',persist);dtype=torch.float32 if key in ('raw_FP32_noise_files','native_FP32_flow_files') else torch.bfloat16;need(name not in states and value.dtype==dtype and list(value.shape)==native.VIDEO['latent_shape'],'state differs');path=folder/(name+'.safetensors');counter.reserve_payload(value.numel()*value.element_size())
      def d2h():
       cpu=value.detach().cpu().contiguous();need(bool(torch.isfinite(cpu).all()),'nonfinite state');return cpu
      cpu=ledger.call(rid,{'state_tensor_D2H':1,'state_tensor_D2H_bytes':value.numel()*value.element_size()},d2h)
      try:ledger.call(rid,{'state_output_writes':1},save_file,{'flow' if key=='native_FP32_flow_files' else 'latents':cpu},str(path))
      finally:
       if path.exists():ledger.call(rid,{'state_file_hash_calls':1,'state_file_hash_reserved_bytes':value.numel()*value.element_size()+(1<<20)},counter.observe_external,path)
      states[name]={'root':'hf',**counter.binding(path,HF),'raw_tensor_sha256':ledger.call(rid,{'state_raw_sha_calls':1,'state_raw_sha_bytes':cpu.numel()*cpu.element_size()},tensor_sha,cpu),'shape':list(cpu.shape),'dtype':str(cpu.dtype).removeprefix('torch.')};counter.write_json(folder/'STATES.json',states)
      mode=mode_for_id[rid];prefix={'raw_noise_FP32','initial_noise','flow_step0','latent_step0'}
      refs=[]
      if mode in ('h_cache_dense','x_cache_dense') or (mode.startswith('high_') and name in prefix):refs.append((mode,report['phases'][0]['states'][name]))
      if mode=='high_h_readonly_p64' and old_high is not None:refs.append(('old_high',old_high['states'][name]))
      for label,old in refs:
       q=(HF/old['path']).resolve();same=ledger.call(rid,{'reference_state_checks':1,'reference_state_read_reservation':18<<20},lambda:compare_reference(q,path,old['bytes'],old['sha256'],states[name]['sha256']));report['prefix_byte_controls'].append({'mode':label,'state':name,'equal':same});persist();need(same,'reference state byte differs')
      prior.mark_state(report,rid,name,'COMPLETE',persist);return states[name]
     return ledger.call(rid,{key:1},write)
    return save,states
   def inspect(rid,path):
    value=ledger.call(rid,{'video_geometry_checks':1,'video_file_hash_calls':1,'video_file_hash_reserved_bytes':32<<20},inspect_video,path,native.VIDEO);counter.observe_external(path,hold=str(counter.path(path)),known=value);mode=mode_for_id[rid]
    refs=[]
    if mode in ('h_cache_dense','x_cache_dense'):refs.append((mode,report['phases'][0]['video']))
    if mode=='high_h_readonly_p64' and old_high is not None:refs.append(('old_high',old_high['video']))
    for label,old in refs:
     q=Path(old['path']).resolve();same=ledger.call(rid,{'reference_video_checks':1,'reference_video_read_reservation':64<<20},lambda:compare_reference(q,path,old['bytes'],old['sha256'],value['sha256']));report.setdefault('neutral_video_checks',{})[label]=same;persist();need(same,'reference video byte differs')
    return value
   def auth(rid):return ledger.call(rid,{'authority_checks':1},authorize,p,digest)
   def emit_action(row):
    events.write(json.dumps(row,separators=(',',':'))+'\n')
    if row.get('event')=='cache_boundary_memory':report.setdefault('cache_boundary_memory',[]).append(row);persist()
    if row.get('event')=='candidate_packed_action':report.setdefault('mask_observations',[]).append(row);persist()
    if row.get('event') in ('phase_memory_before','phase_memory_after','expert_boundary_pre_cleanup','expert_boundary_cleanup'):report.setdefault('memory_observations',[]).append(row);persist()
   result=driver.execute(torch=torch,pipe=pipe,guard=guard,vm=vm,native=native,run_ids=p['run_ids'],image_path=HF/p['case']['image']['path'],prompt=p['case']['prompt'],output_paths=[out/r/'output.mp4' for r in p['run_ids']],capture_factory=capture_factory,inspect_video=inspect,check=check,authorize=auth,gate=lambda rid,cost,fn:ledger.call(rid,cost,fn),progress=progress,kernels=kernels,emit=emit_action)
   report.update(phases=result['phases'],cache_dense_matches_current_none=True,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved());ledger.finish();check();report['status']='COMPLETE_CHIPMUNK_INPUT_VIDEO_EXIT_PENDING';persist()
 except BaseException as e:report.update(status='FAILED_CHIPMUNK_INPUT_VIDEO_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500],error_notes=[str(x)[:1000] for x in getattr(e,'__notes__',[])]);raise
 finally:
  signal.alarm(0);monitor_error=None
  if 'torch' in sys.modules:report['final_memory']=final_memory(sys.modules['torch'])
  if monitor:
   try:monitor.close();report['GPU_monitor']=monitor.report();need(report['GPU_monitor']['sampled_isolation_pass'] and not report['GPU_monitor']['violations'],'GPU monitor violation')
   except BaseException as e:monitor_error=e;report.update(status='FAILED_MONITOR',resource_error=str(e))
  for x in (finder,no_hif4):
   if x in sys.meta_path:sys.meta_path.remove(x)
  for rid in p['run_ids']:
   path=counter.path(out/rid/'output.mp4')
   if path not in counter.files and path.exists():
    try:ledger.call(rid,{'video_file_hash_calls':1,'video_file_hash_reserved_bytes':32<<20},counter.observe_external,path,hold=str(path))
    except BaseException as e:counter.accounting_errors.append(str(e))
  status=out_monitor.stop();report.update(wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,output_monitor=status,output_accounting=counter.summary(),operation_journal=counter.binding(out/'OPERATIONS.jsonl',HF),action_journal=counter.binding(out/'ACTIONS.jsonl',HF))
  if status['fault'] or not status['thread_stopped']:report['status']='FAILED_MONITOR';monitor_error=RuntimeError('output monitor failure')
  try:persist()
  except BaseException as e:support.emergency_failure(emergency,{'id':p['id'],'status':report['status']},'final RUN persist failed '+type(e).__name__);raise
  finally:
   journal.close();events.close()
   if file_guard_entered:file_guard.__exit__(None,None,None)
  if monitor_error:raise monitor_error

def authorize_verification(p,d,a):
 vid=p['id']+'_VERIFY';active=native.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==vid];need(len(rows)==1,'independent verifier ACTIVE required');s=rows[0];need(s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('protocol_sha256')==d and s.get('resources')==VERIFY and s.get('max_gpu_cards')==0 and s.get('parent_scope_id')==p['id'],'verifier scope differs');r=native.read(native.bound(ROOT,s['review']));need(r.get('scope_id')==vid and r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE' and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('approval_scope_fingerprint')==fingerprint(p) and r.get('verification_spec_sha256')==s.get('verification_spec_sha256'),'verifier review differs')
 for k in ('author_review','independent_review','reviewer_recheck'):native.bound(ROOT,r[k])
 need(a.job_spec and native.sha(a.job_spec)==s['verification_spec_sha256'],'verifier spec differs');j=native.read(a.job_spec);v=j['argv'];need(len(v)>1 and v[0]=='env' and v[1].startswith('FFNDS_WORKER_CONFIG='),'profile missing');profile=Path(v[1].split('=',1)[1]);need(native.sha(profile)==p['execution_private_worker_config_sha256'],'verifier profile differs');cpus=sorted(os.sched_getaffinity(0));need(len(cpus)==1 and cpus[0] in p['cpu_affinity'] and sys.flags.isolated and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU isolation differs');out=HF/p['output_HF_relative'];dest=out.parent.parent/'verification/worker/results';expected=['env',v[1],'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PATH=/usr/bin:/bin','CUDA_VISIBLE_DEVICES=','taskset','-c',str(cpus[0]),str(PYTHON),'-I','-B',str(ROOT/SELF),'verify-closed','--protocol',str(a.protocol.resolve()),'--hf-root',str(HF),'--job-spec',str(a.job_spec.resolve()),'--run',str(out/'RUN.json'),'--exit',str(out.parent/'EXIT_RECEIPT.json'),'--verify-out',str(dest/'RUN.json')];need(v==expected and j['id']==vid and j['protocol_sha256']==d and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and Path(j['hf_root']).resolve()==HF and Path(j['output_dir']).resolve()==dest.parent and Path(j['run_json']).resolve()==dest/'RUN.json' and j['external_timeout_seconds']==VERIFY['outer_seconds'] and a.verify_out.resolve()==dest/'RUN.json','verifier exact job differs');import worker_identity;worker_identity.require_worker();return s

def inspect_payload(path,name,shape,dtype,expected_file_sha,expected_raw_sha):
 filehash=hashlib.sha256();rawhash=hashlib.sha256();total=0;before=path.stat();identity=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
 with path.open('rb') as f:
  prefix=f.read(8);need(len(prefix)==8,'safetensors header truncated');n=struct.unpack('<Q',prefix)[0];need(0<n<=16<<20,'safetensors header size');header=f.read(n);need(len(header)==n,'safetensors header truncated');filehash.update(prefix);filehash.update(header);total=8+n;doc=json.loads(header);need(set(doc)=={name} and doc[name]['dtype']==dtype and dtype in ('BF16','F32') and doc[name]['shape']==shape and doc[name]['data_offsets']==[0,(2 if dtype=='BF16' else 4)*__import__('math').prod(shape)],'typed payload header differs')
  while True:
   block=f.read(8<<20)
   if not block:break
   total+=len(block);filehash.update(block);rawhash.update(block)
 need(identity(before)==identity(path.stat()) and total==before.st_size==8+n+(2 if dtype=='BF16' else 4)*__import__('math').prod(shape) and filehash.hexdigest()==expected_file_sha and rawhash.hexdigest()==expected_raw_sha,'payload bytes or hashes differ')

def verify_closed(a):
 p=native.read(a.protocol);validate(p,True);digest=native.sha(a.protocol);scope=authorize_verification(p,digest,a);out=HF/p['output_HF_relative'];need(a.run.resolve()==(out/'RUN.json').resolve() and a.exit.resolve()==(out.parent/'EXIT_RECEIPT.json').resolve(),'RUN/EXIT paths differ');r=native.read(a.run);e=native.read(a.exit);rb=native.binding(a.run,HF);vm.exited(e,rb,digest,p['id']);need(e.get('spec_sha256')==r.get('job_spec_sha256') and r.get('status')=='COMPLETE_CHIPMUNK_INPUT_VIDEO_EXIT_PENDING' and r.get('cache_dense_matches_current_none') is True and r.get('neutral_video_checks')==({'h_cache_dense':True,'x_cache_dense':True,'old_high':True} if p['part']<2 else {'h_cache_dense':True,'x_cache_dense':True}) and r.get('current_phase',{}).get('status')=='EVIDENCE_COMPLETE' and r.get('current_state') is None and r.get('source_ordinal')==DESIGN['source_ordinals'][p['part']],'parent completed identity differs')
 need(len([x for x in r.get('memory_observations',[]) if x['event']=='expert_boundary_pre_cleanup'])==4 and len([x for x in r.get('memory_observations',[]) if x['event']=='expert_boundary_cleanup'])==4 and len([x for x in r.get('memory_observations',[]) if x['event'] in ('phase_memory_before','phase_memory_after')])==10 and r.get('final_memory',{}).get('peak_stats_reset') is False,'allocator observation coverage');need(r['operation_ledger']['predebited']==r['operation_ledger']['observed']=={k:v for k,v in limits(p['part']).items() if v} and len(r['phases'])==5 and [x['mode'] for x in r['phases']]==list(driver.MODES),'physical ledger/coverage differs');need(r['wall_seconds']<=RES['wall_seconds'] and r['peak_RSS_bytes']<=RES['max_RSS_bytes'] and r['GPU_monitor']['sampled_isolation_pass'] and not r['GPU_monitor']['violations'] and r['output_monitor']['thread_stopped'] and not r['output_monitor']['fault'] and r['output_accounting']['peak_tracked_bytes_plus_reserved_and_atomic_temp']<=RES['new_HF_bytes'] and r['output_accounting']['unconsumed_reserved_bytes']==0 and not r['output_accounting']['accounting_errors'],'resources differ')
 observations=r.get('cache_boundary_memory',[]);need(len(observations)==160 and {(x['mode'],x['block']) for x in observations}=={(mode,block) for mode in driver.MODES[1:] for block in range(40)} and all(x['step']==0 and x['expert']==0 and x['run_id']==p['run_ids'][driver.MODES.index(x['mode'])] and type(x['cache_bytes']) is int and x['cache_bytes']==(x['block']+1)*(31668*(13824+5120)*2 if x['mode'] in ('h_cache_dense','high_h_readonly_p64') else (2*31668*5120*2+64*13824*2)) and all(type(x['memory'][k]) is int and x['memory'][k]>=0 for k in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes')) and x['memory']['peak_stats_reset'] is False for x in observations),'cache boundary observation coverage')
 need(len(r.get('mask_observations',[]))==80 and {(x['mode'],x['block']) for x in r['mask_observations']}=={(mode,block) for mode in driver.MODES[3:] for block in range(40)} and all(x['step']==1 and x['expert']==0 and x['run_id']==p['run_ids'][driver.MODES.index(x['mode'])] and len(x['mask_indices_sha256'])==64 for x in r['mask_observations']),'mask observation count')
 need(all(x.get('actual_full_FFN_sites')==160 and x.get('logical_FFN_linear_dispatches')==320 and x.get('native_transfer_observations')==162 for x in r['phases']) and all(x.get('cache_creations')==x.get('cache_releases')==40 for x in r['phases'][1:]) and [x.get('actual_selected_W1_calls') for x in r['phases']]==[0,0,0,40,40] and sum(x['actual_dense_FFN_GEMMs'] for x in r['phases'])==1440,'coverage/transaction count differs')
 old_high=native.read(native.bound(HF,p['source_parent']['parent_completion']['actual_RUN']))['phases'][1] if p['part']<2 else None;verify_reference_metadata(r['phases'],r.get('prefix_byte_controls',[]),old_high)
 assets=[]
 for phase in r['phases']:
  assets.extend({'binding':b,'kind':'state','name':name} for name,b in phase['states'].items());v=phase['video'];vp=Path(v['path']).resolve();need(vp.is_relative_to(HF),'video escaped HF');assets.append({'binding':{'path':str(vp.relative_to(HF)),'bytes':v['bytes'],'sha256':v['sha256']},'kind':'file'})
 assets.extend({'binding':r[k],'kind':'file'} for k in ('operation_journal','action_journal'));need(len(assets)==57 and len({x['binding']['path'] for x in assets})==57 and sum(x['binding']['bytes'] for x in assets)<=VERIFY['read_payload_bytes_max'] and not a.verify_out.exists(),'verification budget/output differs');a.verify_out.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic();v={'id':p['id']+'_VERIFY','protocol_sha256':digest,'job_spec_sha256':scope['verification_spec_sha256'],'parent_scope_id':p['id'],'status':'STARTED_READ_ONLY_VERIFY','predebited_bytes':0,'completed_bytes':0,'inflight':'NONE','automatic_retry':False,'read_accounting':'57 output artifact hashes; protocol/source/receipt identity metadata separately UNKNOWN'};native.write(a.verify_out,v);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('verification wall')));signal.alarm(VERIFY['wall_seconds'])
 try:
  for x in assets:
   b=x['binding'];path=(HF/b['path']).resolve();need(path.is_relative_to(out.resolve()) and path.is_file() and path.stat().st_size==b['bytes'],'artifact identity/size differs');v['predebited_bytes']+=b['bytes'];v['inflight']={'path':b['path'],'bytes':b['bytes'],'completed':'UNKNOWN'};native.write(a.verify_out,v)
   if x['kind']=='state':
    name=x['name'];dtype='F32' if name=='raw_noise_FP32' or name.startswith('flow_step') else 'BF16';key='flow' if name.startswith('flow_step') else 'latents';need(b['dtype']==('float32' if dtype=='F32' else 'bfloat16') and b['shape']==native.VIDEO['latent_shape'],'state metadata dtype/shape differs');inspect_payload(path,key,b['shape'],dtype,b['sha256'],b['raw_tensor_sha256'])
   else:native.bound(HF,b)
   v['completed_bytes']+=b['bytes'];v['inflight']='NONE';native.write(a.verify_out,v)
  v.update(status='PASS_CLOSED_CHIPMUNK_INPUT_VIDEO',RUN=rb,exit=native.binding(a.exit,HF),capture_count=0,artifact_count=len(assets),quality_acceptance=False)
 except BaseException as e:v.update(status='FAILED_READ_ONLY_VERIFY_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:signal.alarm(0);v['wall_seconds']=time.monotonic()-started;native.write(a.verify_out,v)
 return v

def inspect_verification(a):
 p=native.read(a.protocol);validate(p,True);digest=native.sha(a.protocol);out=HF/p['output_HF_relative'];run=out.parent.parent/'verification/worker/results/RUN.json';ex=run.parent.parent/'EXIT_RECEIPT.json';r=native.read(run);e=native.read(ex);need(r['id']==p['id']+'_VERIFY' and r['status']=='PASS_CLOSED_CHIPMUNK_INPUT_VIDEO' and r['protocol_sha256']==digest and e['id']==r['id'] and e['process_exited'] is True and e['child_returncode']==0 and e['timeout'] is False and e['automatic_retry'] is False and e['protocol_sha256']==digest and e['run_sha256']==native.sha(run) and e['spec_sha256']==r['job_spec_sha256'] and r['predebited_bytes']==r['completed_bytes'] and r['inflight']=='NONE','actual verifier EXIT0 required');return {'status':'PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_INPUT_VIDEO','id':p['id'],'verification_RUN':native.binding(run,HF),'verification_exit':native.binding(ex,HF),'quality_acceptance':False}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','freeze','inspect','preflight','run','verify-closed','inspect-verification'));ap.add_argument('--part',type=int);ap.add_argument('--worker-profile',type=Path);ap.add_argument('--cpu-affinity',default='0,1,2,3');ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--verify-out',type=Path);ap.add_argument('--hf-root',type=Path,default=HF);ap.add_argument('--local-root',type=Path);ap.add_argument('--job-spec',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path);a=ap.parse_args();need(a.hf_root.resolve()==HF,'HF root differs')
 if a.action=='draft':need(a.out and a.worker_profile and not a.out.exists(),'fresh draft/profile required');native.write(a.out,draft(a.part,a.worker_profile,[int(x) for x in a.cpu_affinity.split(',')]))
 elif a.action=='freeze':
  need(a.out and not a.out.exists(),'fresh frozen output required');p=native.read(a.protocol);validate(p);need(p['status']=='DRAFT_REVIEW_REQUESTED','draft required');p['status']='FROZEN_PENDING_REGISTRATION';p['assets_finalized']=True;native.write(a.out,p)
 elif a.action=='inspect':p=native.read(a.protocol);validate(p);print(json.dumps({'id':p['id'],'status':p['status'],'fingerprint':fingerprint(p),'requires':'independent review and ACTIVE'}))
 elif a.action=='preflight':preflight(a)
 elif a.action=='run':run(a)
 elif a.action=='verify-closed':print(json.dumps(verify_closed(a)))
 else:print(json.dumps(inspect_verification(a)))
if __name__=='__main__':main()
