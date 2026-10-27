"""Fixed eight-dimensional official VBench evaluation of 60 UB1 videos."""
import argparse, csv, gc, hashlib, importlib, importlib.util, json, os, shutil, signal, sys, time
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='E0_UB1_CONSUMER_DISJOINT_QUALITY_HISTORY_20261027'
DIMENSIONS=('imaging_quality','aesthetic_quality','motion_smoothness','dynamic_degree','subject_consistency','background_consistency','i2v_subject','i2v_background')
COSTS={'imaging_quality':{'MUSIQ':2430},'aesthetic_quality':{'CLIP_L14':90,'aesthetic_head':90},'motion_smoothness':{'AMT':1200},'dynamic_degree':{'RAFT':1200,'RAFT_iterations':24000},'subject_consistency':{'DINO':2430},'background_consistency':{'CLIP_B32':30},'i2v_subject':{'DINO':2460},'i2v_background':{'DreamSim_embed':2460}}
LIMITS={'GPU':1,'CPU':2,'wall_seconds':1800,'outer_wall_seconds':2100,'new_HF_bytes':256<<20,'max_simultaneous_gpu':2,'max_total_video':60,'total_HF_bytes':512<<20}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)

def bind(root,item):
 p=(root/item['path']).resolve()
 if not p.is_relative_to(root.resolve()) or not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Asset binding mismatch: '+item['path'])
 return p

def debit(report,dimension):
 if dimension in report['predebited_work'] or dimension not in DIMENSIONS:raise ValueError('Duplicate or unknown dimension')
 report['predebited_work'][dimension]=COSTS[dimension].copy()

def validate_results(details,expected):
 if not isinstance(details,list) or len(details)!=len(expected):raise ValueError('Missing official per-video result')
 found=[x.get('video_path') for x in details]
 if len(set(found))!=len(found) or set(found)!=set(expected):raise ValueError('Wrong or duplicate result video')
 return details

def plain(v):
 if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [plain(x) for x in v]
 if hasattr(v,'item'):return plain(v.item())
 return v

def observe_score(original,scores):
 def wrapped(self,*a,**kw):
  value=original(self,*a,**kw);scores.append(float(value));return value
 return wrapped

def make_local_hub_loader(oldhub, local_repo, permitted_repo):
 def load(*args, **kwargs):
  if len(args)>2: raise ValueError('Unexpected hub positional arguments')
  repo=args[0] if args else kwargs.pop('repo_or_dir',None)
  model=args[1] if len(args)==2 else kwargs.pop('model',None)
  if repo not in permitted_repo or model!='dino_vitb16':raise ValueError('Unregistered DINO source/model')
  for key in ('source','trust_repo','force_reload','path','read_frame','resolution'):kwargs.pop(key,None)
  return oldhub(str(local_repo),model,source='local',**kwargs)
 return load

def dreamsim_adapter_loader(original, config_factory, adapter_dir, expected_sha):
 adapter_dir=Path(adapter_dir).resolve()
 def load(model, model_id, *args, **kwargs):
  if Path(model_id).resolve()!=adapter_dir or 'config' in kwargs:raise ValueError('DreamSim adapter directory/config differs')
  path=adapter_dir/'adapter_config.json'
  if sha(path)!=expected_sha:raise ValueError('DreamSim adapter config SHA differs')
  values=json.loads(path.read_text())
  defaults={'alpha_pattern':{},'rank_pattern':{},'layer_replication':None,'loftq_config':{},'megatron_config':None,'megatron_core':'megatron.core','use_dora':False,'use_rslora':False}
  if any(values.get(k)!=v for k,v in defaults.items()):raise ValueError('Unsupported active DreamSim adapter extension')
  keys=('r','lora_alpha','lora_dropout','bias','target_modules')
  config=config_factory(**{k:values[k] for k in keys})
  return original(model,model_id,*args,config=config,**kwargs)
 return load

@contextmanager
def dreamsim_adapter_compat(peft_class, config_factory, directory, digest):
 descriptor=peft_class.__dict__['from_pretrained']; original=peft_class.from_pretrained
 peft_class.from_pretrained=staticmethod(dreamsim_adapter_loader(original,config_factory,directory,digest))
 try:yield
 finally:peft_class.from_pretrained=descriptor

def observed_raft_forward(original, observed):
 def forward(self,*args,**kwargs):
  if len(args)!=2 or kwargs.get('iters')!=20 or kwargs.get('test_mode') is not True:raise ValueError('Unexpected RAFT invocation')
  if observed['forwards']>=1200:raise RuntimeError('RAFT budget exceeded')
  observed['forwards']+=1;observed['iterations']+=20
  return original(self,*args,**kwargs)
 return forward

def validate_pairs(videos,hf,producer_by_source=None,expected_bank_sha=None):
 import filecmp
 if len(videos)!=40 or {(x['source'],x['mode']) for x in videos}!={(i,m) for i in range(20) for m in ('dense','candidate')}:raise ValueError('Wrong 20-source paired cohort')
 for i in range(20):
  dense=next(x for x in videos if x['source']==i and x['mode']=='dense');candidate=next(x for x in videos if x['source']==i and x['mode']=='candidate')
  if any(dense[k]!=candidate[k] for k in ('source_id','seed','prompt_sha256','scheduler_id','image_sha256')) or dense['seed']!=42:raise ValueError('Paired source/seed/condition differs')
  if not filecmp.cmp(bind(hf,dense['initial_noise']),bind(hf,candidate['initial_noise']),shallow=False):raise ValueError('Actual initial noise bytes differ')
  for item in (dense,candidate):
   bind(hf,item['video']);d=json.loads(bind(hf,item['generation']).read_text())
   if producer_by_source is not None:
    expected=producer_by_source[item['source']]
    if d.get('protocol_sha256')!=expected['sha256'] or d.get('id')!=expected['id'] or not d.get('registered_run_id') or (item['mode']=='candidate' and d.get('bank_manifest_sha256')!=expected_bank_sha):raise ValueError('Generation producer/bank identity differs')
   if d['status']!='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE' or d['video']['size']!=[832,464] or d['source_index']!=item['source'] or d['mode']!=item['mode'] or d['source_id']!=item['source_id'] or d['seed']!=42 or d['prompt_sha256']!=item['prompt_sha256'] or d['initial_noise']['sha256']!=item['initial_noise']['sha256'] or d['initial_noise']['path']!=item['initial_noise']['path'] or d['initial_noise']['bytes']!=item['initial_noise']['bytes'] or d['video']['sha256']!=item['video']['sha256'] or d['video']['frames']!=81 or d['video']['fps']!=16:raise ValueError('Generation identity differs')
 return True

def validate_historical(videos,hf,generation_protocol,history_run):
 if len(videos)!=20 or {x['source'] for x in videos}!=set(range(20)):raise ValueError('Historical 20 source coverage differs')
 by_index={x['index']:x for x in history_run['completed']}
 if history_run.get('status')!='PASS_FORMAL_GENERATION_SLICE_ONLY' or set(by_index)!=set(range(20)):raise ValueError('Historical RUN coverage differs')
 for item in videos:
  i=item['source'];case=generation_protocol['cases'][i];reference=by_index[i];d=json.loads(bind(hf,item['historical_input']).read_text());v=d['video'];x=d['input']
  if item['mode']!='historical_neutral' or item['exposure']!='HISTORICAL_SAME_SOURCE_SEED_NO_SAVED_INITIAL_NOISE' or reference['case_report_sha256']!=item['historical_input']['sha256'] or d['index']!=i or d['status']!='PASS_FORMAL_INPUT_GENERATION_ONLY' or d['arm']!='D-Q-PACKED-TRAINED-REFERENCE':raise ValueError('Historical receipt identity differs')
  if hashlib.sha256(item['prompt'].encode()).hexdigest()!=item['prompt_sha256'] or x['prompt']!=item['prompt'] or x['seed']!=42 or x['source_group']!=case['source_group'] or x['prompt_sha256']!=case['prompt_sha256'] or x['input_provenance']['image_sha256']!=case['source_id'] or item['source_id']!=case['source_id'] or item['image_sha256']!=case['image']['sha256'] or item['prompt_sha256']!=case['prompt_sha256']:raise ValueError('Historical source/seed differs')
  if v['sha256']!=item['video']['sha256'] or v['frames']!=81 or v['fps']!=16 or v['size']!=[832,464]:raise ValueError('Historical video metadata differs')
  bind(hf,item['video'])
 return True

def validate_producer_method(producer,original,required_sources):
 for key in ('bank_manifest','bank_protocol','cache_manifest','original_recipe'):
  if producer[key]['sha256']!=original[key]['sha256']:raise ValueError('Producer bank or method binding differs')
 actual={x['path']:x['sha256'] for x in producer['repo_sources']};baseline={x['path']:x['sha256'] for x in original['repo_sources']}
 if any(k not in baseline or actual.get(k)!=baseline[k] for k in required_sources):raise ValueError('Producer native method sources differ')
 return True

def validate_clip_paths(plan,hf):
 items=plan['videos']+plan['historical_videos']
 if len(items)!=60:raise ValueError('Exactly sixty input clips required')
 paths=[bind(hf,x['video']) for x in items]
 if len(set(paths))!=60 or len({(p.stat().st_dev,p.stat().st_ino) for p in paths})!=60:raise ValueError('Sixty distinct video files required; path/inode aliases are not new samples')
 # Distinct real generation runs may produce equal bytes. Their identities are
 # checked against producers separately; retain equal-content observations.
 hashes={}
 for item in items:hashes.setdefault(item['video']['sha256'],[]).append({'source':item['source'],'mode':item['mode']})
 return [v for v in hashes.values() if len(v)>1]

def validate_producers(plan,hf,generation_protocol):
 cases=generation_protocol['cases']
 producer_by_source={}; observed_run_ids=set()
 required_sources=('tools/ub1_consumer_runtime.py','tools/e0_ub1_native_consistency.py','tools/step_budget_runtime.py','tools/trained_packed_runtime.py','tools/reduced_rank_residual.py','.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py','.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json','.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py','.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py','.runtime/code/LightX2V/lightx2v/models/schedulers/wan/scheduler.py','.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py','.runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py')
 baseline_sources={x['path']:x['sha256'] for x in generation_protocol['repo_sources']}
 if any(k not in baseline_sources for k in required_sources):raise ValueError('Original method sources missing')
 for receipt in plan['part_receipts']:
  producer=json.loads(bind(ROOT,receipt['producer_protocol']).read_text());producer_sha=receipt['producer_protocol']['sha256'];indices=receipt['source_indices']
  if receipt['part'] not in range(4) or indices!=list(range(5*receipt['part'],5*receipt['part']+5)) or receipt['producer_id']!=producer['id'] or receipt['producer_status']!='PASS_FIXED_DISJOINT20_PART_FULL_DEPTH_CORE_METRICS_QUALITY_PENDING' or (receipt['part']==0 and producer_sha==plan['generation_protocol']['sha256']) or (receipt['part']!=0 and producer_sha!=plan['generation_protocol']['sha256']):raise ValueError('Producer source partition differs')
  validate_producer_method(producer,generation_protocol,required_sources)
  producer_cases={x['index']:x for x in producer['cases']}
  for i in indices:
   if i in producer_by_source or i not in producer_cases:raise ValueError('Duplicate or missing producer source')
   if any(producer_cases[i][k]!=cases[i][k] for k in ('source_id','prompt_sha256','seed')):raise ValueError('Producer source identity differs')
   producer_by_source[i]={'sha256':producer_sha,'id':producer['id']}
  run=json.loads(bind(hf,receipt['run']).read_text());launch=json.loads(bind(hf,receipt['launch']).read_text())
  if run['status']!=receipt['producer_status'] or run['id']!=producer['id'] or run['part']!=receipt['producer_part'] or run['source_indices']!=indices or run['protocol_sha256']!=producer_sha or launch['protocol_sha256']!=producer_sha or launch['launcher_returncode']!=0 or launch.get('process_exited') is not True:raise ValueError('Generation producer part incomplete')
  phases=run.get('phases',[])
  if len(phases)!=10 or {(x['source_index'],x['mode']) for x in phases}!={(i,m) for i in indices for m in ('dense','candidate')}:raise ValueError('Producer phase coverage differs')
  for phase in phases:
   item=next((x for x in plan['videos'] if x['source']==phase['source_index'] and x['mode']==phase['mode']),None)
   if item is None or phase.get('status')!='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE' or phase['generation_receipt']['sha256']!=item['generation']['sha256']:raise ValueError('Video receipt outside producer RUN')
   data=json.loads(bind(hf,item['generation']).read_text());run_id=data.get('registered_run_id')
   if not run_id or run_id!=phase.get('registered_run_id') or run_id in observed_run_ids:raise ValueError('Duplicate producer run ID')
   observed_run_ids.add(run_id)
 if set(producer_by_source)!=set(range(20)) or len(observed_run_ids)!=40:raise ValueError('Composite producer coverage incomplete')
 validate_pairs(plan['videos'],hf,producer_by_source,generation_protocol['bank_manifest']['sha256'])
 return True

def preflight(plan,hf):
 if plan['id']!=ID or plan['dimensions']!=list(DIMENSIONS) or plan['costs']!=COSTS or not plan['assets_finalized']:raise ValueError('Unfrozen scope')
 for x in plan['repo_sources']:bind(ROOT,x)
 for x in plan['hf_bindings']:bind(hf,x)
 validate_clip_paths(plan,hf)
 if len(plan['images'])!=20 or {x['source'] for x in plan['images']}!=set(range(20)) or len({x['name'] for x in plan['images']})!=20:raise ValueError('Image staging coverage differs')
 if len(plan['part_receipts'])!=4 or {x['part'] for x in plan['part_receipts']}!=set(range(4)):raise ValueError('Four completed generation parts required')
 generation_protocol=json.loads(bind(ROOT,plan['generation_protocol']).read_text())
 cases=generation_protocol['cases']
 if len(cases)!=20 or len({x['source_id'] for x in cases})!=20:raise ValueError('Frozen generation sources differ')
 for item in plan['videos']:
  case=cases[item['source']]
  image=next(x for x in plan['images'] if x['source']==item['source'])
  if hashlib.sha256(item['prompt'].encode()).hexdigest()!=item['prompt_sha256']:raise ValueError('Prompt bytes differ')
  if case['index']!=item['source'] or case['source_id']!=item['source_id'] or case['prompt_sha256']!=item['prompt_sha256'] or case['image']['sha256']!=item['image_sha256'] or image['binding']['sha256']!=item['image_sha256'] or image['name']!=item['image_name']:raise ValueError('Video source differs from generation protocol')
  bind(hf,image['binding'])
 validate_producers(plan,hf,generation_protocol)
 view=json.loads(bind(hf,plan['historical_view']).read_text())
 if view['status']!='COMPLETE_BYTE_VERIFIED_HISTORICAL_CONTROLS_VIEW' or len(view['arms']['neutral']['sources'])!=20:raise ValueError('Historical view incomplete')
 for item,source in zip(sorted(plan['historical_videos'],key=lambda x:x['source']),view['arms']['neutral']['sources']):
  if item['source']!=source['index'] or item['video']!=source['video'] or item['historical_input']!=source['input']:raise ValueError('Historical view binding differs')
 history_run=json.loads(bind(ROOT,plan['historical_run']).read_text())
 validate_historical(plan['historical_videos'],hf,generation_protocol,history_run)
 receipt=json.loads(bind(hf,plan['v8_receipt']).read_text())
 if receipt.get('status')!='PASS_IMPORTS_ONLY' or receipt.get('protocol_sha256')!=sha(ROOT/plan['v8_protocol']):raise ValueError('V8 receipt differs')
 return True

def cluster_rows(plan,part_runs,hf,protocol_sha):
 if len(part_runs)!=2 or {r.get('part') for r in part_runs}!={0,1} or any(r.get('status')!='COMPLETE_SEVEN_PART_DIMENSIONS' or r.get('protocol_sha256')!=protocol_sha for r in part_runs):raise ValueError('Both completed scoring parts required')
 generation=json.loads(bind(ROOT,plan['generation_protocol']).read_text());cases=generation['cases'];bank=generation['source_bank']['sources']
 lookup={str(bind(hf,x['video'])):(x['source'],x['mode']) for x in plan['videos']+plan['historical_videos']}
 if len(lookup)!=60:raise ValueError('Expected 60 distinct video paths')
 values={}
 for run in part_runs:
  for entry in run['completed']:
   dimension=entry['dimension']
   for row in entry['result'][-1]:
    source,mode=lookup[row['video_path']];key=(source,mode,dimension)
    if source//10!=run['part'] or key in values:raise ValueError('Duplicate/misassigned metric')
    values[key]=float(row['video_results'])
   if dimension=='dynamic_degree':
    for row in entry['RAFT_observations']:
     source,mode=lookup[row['video_path']];key=(source,mode,'raft_continuous_mean')
     if len(row['scores'])!=40 or key in values:raise ValueError('RAFT pair data differ')
     values[key]=sum(row['scores'])/40
 result={'dense':[],'historical_neutral':[]}
 for source in range(20):
  case=cases[source];entry=next(x for x in bank if x['source_id']==case['semantic_source_id'])
  metrics=list(DIMENSIONS[:6])+['raft_continuous_mean',('i2v_subject' if source<10 else 'i2v_background')]
  for metric in metrics:
   for baseline in result:
    keys=[(source,baseline,metric),(source,'candidate',metric)]
    if any(k not in values for k in keys):raise ValueError('Missing source/arm/metric')
    result[baseline].append({'source_id':entry['source_id'],'image_id':entry['image_id'],'seed':42,'metric':metric,'dense':values[keys[0]],'candidate':values[keys[1]]})
 return result

def export_clusters(plan,part_runs,hf,protocol_sha,out):
 from ffnds_cluster_metric_report import paired_report
 rows=cluster_rows(plan,part_runs,hf,protocol_sha);out.mkdir(parents=True,exist_ok=True)
 for label,records in rows.items():
  with (out/(label+'_pairs.csv')).open('x',newline='') as f:
   w=csv.DictWriter(f,fieldnames=('source_id','image_id','seed','metric','dense','candidate'));w.writeheader();w.writerows(records)
  result=paired_report(records,replicates=2000,seed=42);result['baseline_label']=label;result['source_role']='PREVIOUSLY_EXPOSED_DEVELOPMENT';result['historical_noise']='NOT_SAVED' if label=='historical_neutral' else 'ACTUAL_NOISE_BYTES_VERIFIED';result['formal_quality_acceptance']=False
  with (out/(label+'_cluster_report.json')).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')

def audit_scope(scope, query_code, binary, index):
 allowed={('--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'),('--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu','--format=csv,noheader,nounits'),('--query-gpu=memory.used','--format=csv,noheader,nounits')}
 def audit(event,values):
  if event=='subprocess.Popen':
   exe,argv=values[:2];frame=sys._getframe();trusted=False
   while frame:
    if frame.f_code is query_code:trusted=True;break
    frame=frame.f_back
   if trusted and isinstance(argv,(list,tuple)) and len(argv)==5 and argv[0]=='nvidia-smi' and argv[1]=='-i' and argv[2]==index and tuple(argv[3:]) in allowed and Path(shutil.which(exe) or '').resolve()==binary:return
   raise RuntimeError('Subprocess denied')
  if event.startswith('socket.') or event in ('os.system','os.exec','os.posix_spawn'):raise RuntimeError('Network/process denied')
  if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
   path=Path(os.fsdecode(values[0])).resolve();mode=values[1] or '';flags=values[2] or 0
   if (any(c in mode for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)) and not path.is_relative_to(scope):raise RuntimeError('Write outside scope')
  if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.symlink','os.link','os.chmod','os.truncate'):
   vals=values[1:2] if event=='os.symlink' else (values[:2] if event in ('os.rename','os.link') else values[:1])
   for v in vals:
    if isinstance(v,(str,bytes,os.PathLike)) and not Path(os.fsdecode(v)).resolve().is_relative_to(scope):raise RuntimeError('Mutation outside scope')
 return audit

def authorize(plan,digest,active,part):
 if type(part) is not int or part not in (0,1) or plan.get('id')!=ID or plan.get('assets_finalized') is not True or plan.get('limits')!=LIMITS or plan.get('automatic_retry') is not False:raise ValueError('Unfrozen fixed two-part scoring scope')
 user=active.get('user_continuation_authorization_20261027',{})
 if user.get('id')!='USER_RESUME_OFFLINE_IFC_20261027' or user.get('status')!='ACTIVE_CURRENT_USER_DIRECTED_SESSION':raise ValueError('Current user continuation inactive')
 if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in active.get('experiment_authority',{}).get('current_user_authorized_scopes',[])):raise ValueError('Exact scope not authorized')

def run(args):
 plan=json.loads(args.protocol.read_text());digest=sha(args.protocol);hf=args.hf_root.resolve()
 authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()),args.part)
 scope=(hf/plan['output_HF_relative']/f'part{args.part}').resolve()
 if scope==hf or not scope.is_relative_to(hf):raise ValueError('Output path')
 scope.mkdir(parents=True,exist_ok=True)
 with (scope/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest},f)
 def alarm(*_):raise TimeoutError('1800 second bound including input and environment preflight')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(LIMITS['wall_seconds'])
 try:
  preflight(plan,hf)
  run_scoring(args,plan,digest,hf,scope)
 except BaseException as e:
  path=scope/'RUN.json';report=json.loads(path.read_text()) if path.exists() else {'id':ID,'protocol_sha256':digest,'part':args.part,'predebited_work':{},'completed':[]}
  if not str(report.get('status','')).startswith('FAILED'):report['status']='FAILED_NO_RETRY'
  report.update(error_type=type(e).__name__,error=str(e));write(path,report);raise
 finally:signal.alarm(0)

def run_scoring(args,plan,digest,hf,scope):
 site=(hf/plan['site']).resolve();overlay=(hf/plan['overlay']).resolve();sys.path.append(str(site));sys.path.insert(0,str(overlay));sys.dont_write_bytecode=True
 spec=importlib.util.spec_from_file_location('ub1_v8',ROOT/plan['helper']);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 v8=json.loads((ROOT/plan['v8_protocol']).read_text());lock=json.loads((ROOT/v8['lock_path']).read_text())
 if sha(ROOT/v8['lock_path'])!=v8['lock_sha256'] or sha(Path(sys.executable))!=v8['installed_environment']['python_sha256']:raise ValueError('Interpreter identity differs')
 helper.verify_original_sources(hf,lock);helper.verify_shared(lock['shared_runtime'])
 if helper.verify_installed_environment(hf,v8,lock).resolve()!=site or helper.verify_overlay(hf/plan['v8_directory'],v8,hf).resolve()!=overlay:raise ValueError('Metric environment differs')
 helper.two_cpus();sys.path.append(str(ROOT/'tools'));from worker_identity import require_worker;require_worker()
 raft_scores=[]
 report={'id':ID,'status':'STARTED','protocol_sha256':digest,'predebited_work':{},'completed':[], 'RAFT_observed':{'forwards':0,'iterations':0},'opaque_internal_calls':'UNKNOWN','checkpoint_loading':'Original torch.load arguments; TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, bound local checkpoints','temporal_flickering':'NOT_RUN_STATIC_ELIGIBILITY_UNESTABLISHED','camera_motion':'NOT_RUN_NOT_IN_SCOPE','part':args.part,'source_role':'PREVIOUSLY_EXPOSED_DEVELOPMENT','historical_neutral':'HISTORICAL_SAME_SOURCE_SEED_NO_SAVED_INITIAL_NOISE','quality_acceptance':False};monitor=None
 def check():
  if monitor:monitor.check()
  if shutil.disk_usage(scope).free<10<<30 or sum(p.stat().st_size for p in scope.rglob('*') if p.is_file() and not p.is_symlink())>256<<20:raise RuntimeError('Resource bound')
 try:
  for x in ('tmp','xdg','mpl','torch','hf','inductor','debug'): (scope/x).mkdir()
  os.environ.update(TMPDIR=str(scope/'tmp'),XDG_CACHE_HOME=str(scope/'xdg'),MPLCONFIGDIR=str(scope/'mpl'),TORCH_HOME=str(scope/'torch'),HF_HOME=str(scope/'hf'),TORCHINDUCTOR_CACHE_DIR=str(scope/'inductor'),TORCH_COMPILE_DEBUG_DIR=str(scope/'debug'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
  helper.disable_local_ipv6_probe(site,v8)
  import torch
  from startup_gpu_reservation import reserve_before_assets
  from assigned_gpu_resources import AssignedGPUResources
  reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt;torch.set_num_threads(2)
  torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
  report['runtime_precision']={'matmul_TF32':torch.backends.cuda.matmul.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
  if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):raise RuntimeError('Single assigned A100 required')
  snapshot=scope/'EXECUTED_SOURCES';snapshot.mkdir()
  for item in plan['repo_sources']:
   dst=snapshot/item['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/item['path'],dst)
  shutil.copyfile(args.protocol,snapshot/'protocol.json')
  binary=Path(shutil.which('nvidia-smi') or '').resolve()
  if str(binary)!=plan['nvidia_smi']['path'] or sha(binary)!=plan['nvidia_smi']['sha256']:raise ValueError('Monitor binary differs')
  sys.addaudithook(audit_scope(scope,AssignedGPUResources.query.__code__,binary,os.environ['FFNDS_PHYSICAL_GPU']))
  source=hf/plan['source_root'];sys.path.insert(0,str(source));device=torch.device('cuda:0')
  from offline_i2v_metric_loaders import submodules
  assets=json.loads((ROOT/plan['asset_manifest']).read_text());base=hf/assets['HF_relative_root']; oldhub=torch.hub.load;oldurl=torch.hub.load_state_dict_from_url
  def no_download(*a,**k):raise RuntimeError('Network download forbidden')
  hub=make_local_hub_loader(oldhub,base/'source/dino',{'facebookresearch/dino:main',str(base/'source/dino')})
  def url(url,*a,**kw):
   if url!='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth':raise RuntimeError('Unregistered weight URL')
   return torch.load(base/'dino_model/dino_vitbase16_pretrain.pth',map_location=kw.get('map_location'),weights_only=kw.get('weights_only',False))
  torch.hub.load=hub;torch.hub.load_state_dict_from_url=url;torch.hub.download_url_to_file=no_download
  import socket
  socket.socket.connect=no_download;socket.create_connection=no_download
  import cv2
  report['decoded_video_checks']=[]
  for video in [x for x in plan['videos']+plan['historical_videos'] if x['source']//10==args.part]:
   cap=cv2.VideoCapture(str(bind(hf,video['video'])));count=0
   try:
    while True:
     ok,frame=cap.read()
     if not ok:break
     if frame.shape[:2]!=(464,832):raise ValueError('Decoded frame shape differs')
     count+=1
   finally:cap.release()
   if count!=81:raise ValueError('Decoded frame count differs')
   report['decoded_video_checks'].append({'source':video['source'],'mode':video['mode'],'frames':count})
  write(scope/'RUN.json',report)
  work=scope/'work';images=work/'vbench2_beta_i2v/data/crop/16-9';images.mkdir(parents=True)
  for image in plan['images']:
   if image['source']//10==args.part:(images/image['name']).symlink_to(bind(hf,image['binding']))
  os.chdir(work);raft_scores=[];oldinfer=None
  for dimension in [x for x in DIMENSIONS if x not in ('i2v_subject','i2v_background') or x==('i2v_subject' if args.part==0 else 'i2v_background')]:
   check();debit(report,dimension);write(scope/'RUN.json',report)
   selected=[x for x in plan['videos']+plan['historical_videos'] if x['source']//10==args.part]
   doc=[{'prompt_en':x['prompt'],'dimension':[dimension],'image_name':x['image_name'],'video_list':[str(bind(hf,x['video']))]} for x in selected]
   inp=scope/(dimension+'_INPUT.json');write(inp,doc)
   mod=importlib.import_module(('vbench2_beta_i2v.' if dimension.startswith('i2v_') else 'vbench.')+dimension)
   if dimension=='i2v_background':
    original=mod.dreamsim
    def dream(*a,**kw):kw.update(cache_dir=str(base/'dreamsim'),device=device);return original(*a,**kw)
    mod.dreamsim=dream
   if dimension=='dynamic_degree':
    original_score=mod.DynamicDegree.get_score;original_infer=mod.DynamicDegree.infer
    mod.RAFT.forward=observed_raft_forward(mod.RAFT.forward,report['RAFT_observed'])
    def score(self,*a,**kw):return observe_score(original_score,raft_scores[-1]['scores'])(self,*a,**kw)
    def infer(self,path,*a,**kw):
     row={'video_path':path,'scores':[]};raft_scores.append(row);value=original_infer(self,path,*a,**kw);row['params']=dict(self.params);row['official_binary']=bool(value);return value
    mod.DynamicDegree.get_score=score;mod.DynamicDegree.infer=infer
   if dimension=='i2v_background':
    from peft import PeftModel,LoraConfig
    adapter=base/'dreamsim/ensemble_lora';expected=next(x['sha256'] for x in plan['hf_bindings'] if x['path'].endswith('/dreamsim/ensemble_lora/adapter_config.json'))
    with dreamsim_adapter_compat(PeftModel,LoraConfig,adapter,expected),torch.inference_mode():
     result=getattr(mod,'compute_'+dimension)(str(inp),device,submodules(dimension,assets,hf))
    report['dreamsim_adapter_compatibility']={'config_sha256':expected,'injected_original_five_lora_keys':True,'unsupported_extensions_verified_inactive':True,'checkpoint_bytes_changed':False}
   else:
    with torch.inference_mode():result=getattr(mod,'compute_'+dimension)(str(inp),device,submodules(dimension,assets,hf))
   torch.cuda.synchronize();values=plain(result);validate_results(values[-1],[v['video_list'][0] for v in doc]);entry={'dimension':dimension,'result':values,'input_sha256':sha(inp)}
   if dimension=='dynamic_degree':
    if report['RAFT_observed']!={'forwards':1200,'iterations':24000} or len(raft_scores)!=30 or any(len(x['scores'])!=40 for x in raft_scores):raise ValueError('RAFT observed count differs')
    entry['RAFT_observations']=raft_scores
   report['completed'].append(entry);write(scope/'RUN.json',report);gc.collect();torch.cuda.empty_cache()
  report['status']='COMPLETE_SEVEN_PART_DIMENSIONS';report['device_argument']='torch.device(cuda:0); original motion_smoothness else branch';check()
 except BaseException as e:report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e),RAFT_partial_observations=raft_scores);raise
 finally:
  original_error_active=sys.exc_info()[0] is not None;monitor_error=None
  if monitor:
   try:
    monitor.close();report['assigned_gpu_resources']=monitor.report()
    if report['assigned_gpu_resources'].get('violations'):
     report['status']='FAILED_GPU_MONITOR';monitor_error=RuntimeError('Assigned GPU monitor reported violations')
   except BaseException as e:
    report['status']='FAILED_GPU_MONITOR';report['monitor_error']={'type':type(e).__name__,'message':str(e)};monitor_error=e
  write(scope/'RUN.json',report)
  if monitor_error is not None and not original_error_active:raise monitor_error

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--hf-root',type=Path,required=True);p.add_argument('--part',type=int,choices=(0,1));p.add_argument('--aggregate-runs',type=Path,nargs=2);p.add_argument('--aggregate-out',type=Path);args=p.parse_args()
 if args.aggregate_runs:
  if args.part is not None or args.aggregate_out is None:raise ValueError('Aggregate needs two runs and output only')
  sys.path.append(str(ROOT/'tools'));plan=json.loads(args.protocol.read_text());hf=args.hf_root.resolve();preflight(plan,hf);runs=[json.loads(x.read_text()) for x in args.aggregate_runs];export_clusters(plan,runs,hf,sha(args.protocol),args.aggregate_out)
 else:
  if args.part is None or args.aggregate_out is not None:raise ValueError('Scoring requires part only')
  run(args)
