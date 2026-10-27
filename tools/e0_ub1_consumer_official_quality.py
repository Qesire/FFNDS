"""Fixed eight-dimensional official VBench evaluation of six UB1 videos."""
import argparse, gc, hashlib, importlib, importlib.util, json, os, shutil, signal, sys, time
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='E0_UB1_CONSUMER_OFFICIAL_QUALITY_20261027'
DIMENSIONS=('imaging_quality','aesthetic_quality','motion_smoothness','dynamic_degree','subject_consistency','background_consistency','i2v_subject','i2v_background')
COSTS={'imaging_quality':{'MUSIQ':486},'aesthetic_quality':{'CLIP_L14':18,'aesthetic_head':18},'motion_smoothness':{'AMT':240},'dynamic_degree':{'RAFT':240,'RAFT_iterations':4800},'subject_consistency':{'DINO':486},'background_consistency':{'CLIP_B32':6},'i2v_subject':{'DINO':246},'i2v_background':{'DreamSim_embed':246}}

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
  if observed['forwards']>=240:raise RuntimeError('RAFT budget exceeded')
  observed['forwards']+=1;observed['iterations']+=20
  return original(self,*args,**kwargs)
 return forward

def preflight(plan,hf):
 if plan['id']!=ID or plan['dimensions']!=list(DIMENSIONS) or plan['costs']!=COSTS or not plan['assets_finalized']:raise ValueError('Unfrozen scope')
 for x in plan['repo_sources']:bind(ROOT,x)
 for x in plan['hf_bindings']:bind(hf,x)
 if len(plan['videos'])!=6 or {(x['source'],x['mode']) for x in plan['videos']}!={(s,m) for s in (0,10) for m in ('dense','neutral','candidate')}:raise ValueError('Wrong cohort')
 for x in plan['videos']:
  bind(hf,x['video']);d=json.loads(bind(hf,x['generation']).read_text())
  if d['status']!='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE' or d['video']['size']!=[832,464] or d['mode']!=x['mode'] or d['source_id']!=x['source_id'] or d['seed']!=42 or d['video']['sha256']!=x['video']['sha256'] or d['video']['frames']!=81 or d['video']['fps']!=16:raise ValueError('Generation identity differs')
 receipt=json.loads(bind(hf,plan['v8_receipt']).read_text())
 if receipt.get('status')!='PASS_IMPORTS_ONLY' or receipt.get('protocol_sha256')!=sha(ROOT/plan['v8_protocol']):raise ValueError('V8 receipt differs')
 return True

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

def run(args):
 plan=json.loads(args.protocol.read_text());digest=sha(args.protocol);hf=args.hf_root.resolve();preflight(plan,hf)
 active=json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()); scopes=active['experiment_authority']['current_user_authorized_scopes']
 if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in scopes):raise ValueError('Exact scope not authorized')
 site=(hf/plan['site']).resolve();overlay=(hf/plan['overlay']).resolve();sys.path.append(str(site));sys.path.insert(0,str(overlay));sys.dont_write_bytecode=True
 spec=importlib.util.spec_from_file_location('ub1_v8',ROOT/plan['helper']);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 v8=json.loads((ROOT/plan['v8_protocol']).read_text());lock=json.loads((ROOT/v8['lock_path']).read_text())
 if sha(ROOT/v8['lock_path'])!=v8['lock_sha256'] or sha(Path(sys.executable))!=v8['installed_environment']['python_sha256']:raise ValueError('Interpreter identity differs')
 helper.verify_original_sources(hf,lock);helper.verify_shared(lock['shared_runtime'])
 if helper.verify_installed_environment(hf,v8,lock).resolve()!=site or helper.verify_overlay(hf/plan['v8_directory'],v8,hf).resolve()!=overlay:raise ValueError('Metric environment differs')
 helper.two_cpus();sys.path.append(str(ROOT/'tools'));from worker_identity import require_worker;require_worker()
 scope=(hf/plan['output_HF_relative']).resolve()
 if scope==hf or not scope.is_relative_to(hf):raise ValueError('Output path')
 scope.mkdir(parents=True,exist_ok=True)
 with (scope/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest},f)
 raft_scores=[]
 report={'id':ID,'status':'STARTED','protocol_sha256':digest,'predebited_work':{},'completed':[], 'RAFT_observed':{'forwards':0,'iterations':0},'opaque_internal_calls':'UNKNOWN','checkpoint_loading':'Original torch.load arguments; TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, bound local checkpoints','temporal_flickering':'NOT_RUN_STATIC_ELIGIBILITY_UNESTABLISHED','quality_acceptance':False};monitor=None
 def check():
  if monitor:monitor.check()
  if shutil.disk_usage(scope).free<10<<30 or sum(p.stat().st_size for p in scope.rglob('*') if p.is_file() and not p.is_symlink())>256<<20:raise RuntimeError('Resource bound')
 def alarm(*_):raise TimeoutError('1800 second bound')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(1800)
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
  for video in plan['videos']:
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
  for image in plan['images']:(images/image['name']).symlink_to(bind(hf,image['binding']))
  os.chdir(work);raft_scores=[];oldinfer=None
  for dimension in DIMENSIONS:
   check();debit(report,dimension);write(scope/'RUN.json',report)
   selected=[x for x in plan['videos'] if dimension not in ('i2v_subject','i2v_background') or x['source']==(0 if dimension=='i2v_subject' else 10)]
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
    if report['RAFT_observed']!={'forwards':240,'iterations':4800} or len(raft_scores)!=6 or any(len(x['scores'])!=40 for x in raft_scores):raise ValueError('RAFT observed count differs')
    entry['RAFT_observations']=raft_scores
   report['completed'].append(entry);write(scope/'RUN.json',report);gc.collect();torch.cuda.empty_cache()
  report['status']='COMPLETE_EIGHT_OFFICIAL_DIMENSIONS';report['device_argument']='torch.device(cuda:0); original motion_smoothness else branch';check()
 except BaseException as e:report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e),RAFT_partial_observations=raft_scores);raise
 finally:
  signal.alarm(0)
  if monitor:
   monitor.close();report['assigned_gpu_resources']=monitor.report()
   if report['assigned_gpu_resources'].get('violations'):report['status']='FAILED_GPU_MONITOR'
  write(scope/'RUN.json',report)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--hf-root',type=Path,required=True);run(p.parse_args())
