"""TEAL author grab_acts adapter. REVIEW_REQUESTED; execution requires ACTIVE.

Pinned source, assets, environment, exact spec and independent recheck are
mandatory. Calibration has never run under this producer.
"""
import argparse, hashlib, importlib, importlib.metadata, json, os, resource, runpy, signal, socket, subprocess, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_teal_llama3_tokenizer_byte_bridge as base
import e0_teal_alpaca_acquisition as alpaca
import e0_teal_llama3_author_checkpoint_convert as support
HF=Path('/home/xiang/.cache/huggingface')
ID='E0_TEAL_LLAMA3_AUTHOR_GRAB_ACTS_20261030'
AUTHOR=HF/'ffnds-research-sources/p0_teal_full_llm_20261029'
SNAP=HF/'ffnds-derived/e0_teal_llama3_public_weight_acquisition_20261029/attempt1/worker/scoped_hf_cache/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061'
PARQUET=HF/alpaca.CACHE/'datasets--tatsu-lab--alpaca'/'blobs'/alpaca.DIGEST
OUT=HF/'ffnds-derived/e0_teal_llama3_author_grab_acts_20261030/attempt1/worker/results'
ENV=HF/'ffnds-derived/teal_author_calibration_env_20261029/venv'
RES={'GPU':1,'CPU_threads':2,'wall_seconds':14400,'outer_seconds':14520,'host_RSS_max_bytes':34359738368,'GPU_memory_max_bytes':85899345920,'new_HF_bytes':17179869184,'minimum_free_disk_bytes':21474836480,'max_asset_hash_bytes':16094013218,'automatic_retry':False}
AUTHOR_FILES=('teal/grab_acts.py','teal/model.py','teal/mlp.py','teal/self_attn.py','utils/__init__.py','utils/data.py','utils/eval_ppl.py','utils/utils.py')
LIMITS={'source_rows_per_pass':300,'dataset_passes':2,'segments':10,'sequence':2048,'layers':32,'histograms':64,'sorts':128,'layer_inputs':32,'model_loads':1,'layer_forwards':32,'activation_finite_reductions':64,'activation_finite_elements':5368709120,'training_updates':0,'histogram_validation_files':64,'histogram_validation_max_logical_read_bytes':268435456}
need=base.need

def fingerprint(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate(p):
 need(p.get('id')==ID and p.get('status')=='FROZEN_PENDING_REGISTRATION' and p.get('resources')==RES and p.get('limits')==LIMITS and p.get('model_snapshot')==str(SNAP) and p.get('parquet')==str(PARQUET) and p.get('output')==str(OUT) and p.get('review') is None,'protocol differs')
 need(isinstance(p.get('source_bindings'),list) and p['source_bindings'] and isinstance(p.get('asset_receipts'),list) and len(p['asset_receipts'])>=3,'identity missing')
 need(p.get('model_closed')=='.runtime/teal_llama3_public_weight_acquisition_20261029/CLOSED_VERIFICATION.json','model proof identity')
 need(isinstance(p.get('environment_binding'),dict) and isinstance(p.get('worker_config_sha256'),str),'runtime binding missing')
 need({b['path'] for b in p.get('hf_source_bindings',[])}=={str((AUTHOR/x).relative_to(HF)) for x in AUTHOR_FILES},'author source closure')
 need({b['path'] for b in p['source_bindings']}=={'tools/e0_teal_llama3_grab_acts_launcher.py','tests/test_e0_teal_llama3_grab_acts_launcher.py','tools/e0_teal_alpaca_acquisition.py','tools/e0_teal_llama3_author_checkpoint_convert.py','tools/e0_teal_llama3_tokenizer_byte_bridge.py','tools/worker_identity.py','tools/durable_worker_job.py','scripts/worker_teal_gpu0_uuid_exec.sh'},'launcher source closure')

def authorize(p,d):
 validate(p);active=base.read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in active['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID]
 need(active.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True and len(rows)==1,'ACTIVE scope missing');s=rows[0]
 need(s.get('protocol_sha256')==d and s.get('resources')==RES and s.get('limits')==LIMITS and s.get('max_gpu_cards')==1 and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE'),'ACTIVE differs')
 rev=base.read(base.bound(ROOT,s['review']))
 need(rev.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and rev.get('scope_id')==ID and rev.get('state')=='REVIEWER_RECHECK_APPROVED' and rev.get('recheck_disposition')=='APPROVE' and rev.get('reviewer_disposition') in ('APPROVE','REVISE') and rev.get('author_agent_id') and rev.get('reviewer_agent_id') and rev['author_agent_id']!=rev['reviewer_agent_id'] and rev.get('adversarial_counterexample') and rev.get('responses_to_review') and rev.get('approval_scope_fingerprint')==fingerprint(p),'independent review missing')
 need(rev.get('critical_source_hashes')=={b['path']:b['sha256'] for b in p['source_bindings']},'review source mismatch')
 for k in ('author_review','independent_review','reviewer_recheck'):base.bound(ROOT,rev[k])
 return s

def runtime_binding():
 packages=('torch','triton','transformers','datasets','accelerate','flash-attn','pyarrow','numpy','safetensors','huggingface-hub','psutil')
 records=[]
 for name in packages:
  d=importlib.metadata.distribution(name);files=[]
  for f in d.files or []:
   q=str(f)
   if q.endswith(('.dist-info/METADATA','.dist-info/RECORD')):
    path=Path(d.locate_file(f)).resolve()
    if path.is_file():files.append([q,base.sha(path)])
  need(len(files)>=2,'runtime package incomplete '+name);records.append({'name':name,'version':d.version,'files':sorted(files)})
 return {'prefix':str(Path(sys.prefix).resolve()),'python_sha256':base.sha(Path(sys.executable).resolve()),'bridge_sha256':base.sha(ENV/'lib/python3.11/site-packages/ffnds_author_packages_readonly.pth'),'coverage':'metadata and RECORD only; full packages and native ELF bytes not independently hashed','packages':records}

def check_counts(counts):
 need(counts=={'sort':128,'histogram':128,'layer_inputs':32,'histogram_files':64,'layer_forward_pre':32,'layer_forward_post':32,'activation_finite_reductions':64},'author operation counts differ')

def read_prefix(dataset):
 rows=list(dataset.skip(0).take(300));need(len(rows)==300 and all(isinstance(r.get('text'),str) for r in rows),'300 text rows required');return ''.join(r['text']+'\n\n' for r in rows)

def validate_segments(ids,mask):
 need(tuple(ids.shape)==(10,2048) and tuple(mask.shape)==(10,2048),'author position_ids requires exactly ten 2048 segments')
 return {'shape':[10,2048],'input_ids_sha256':hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),'attention_mask_sha256':hashlib.sha256(mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),'padded_positions':int((mask==0).sum().item())}

def check_resources(rss,disk,free,gpu):
 need(rss<=RES['host_RSS_max_bytes'] and disk<=RES['new_HF_bytes']-(1<<20) and free>=RES['minimum_free_disk_bytes'] and gpu<=RES['GPU_memory_max_bytes'],'resource monitor limit')

class Monitor:
 def __init__(self,root,uuid):self.root=root;self.uuid=uuid;self.stop_event=threading.Event();self.fault=None;self.samples=[];self.thread=threading.Thread(target=self.loop,daemon=True)
 def loop(self):
  while not self.stop_event.is_set():
   try:
    import psutil
    rss=psutil.Process().memory_info().rss; disk=support.worker_usage(self.root)['apparent_bytes'];free=__import__('shutil').disk_usage(self.root).free
    data=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
    matches=[line.split(',') for line in data.splitlines() if line.split(',')[0].strip()==self.uuid];need(len(matches)==1,'GPU monitor identity');gpu=int(matches[0][1].strip())*1048576
    apps=subprocess.check_output(['/usr/bin/nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    foreign=[line for line in apps.splitlines() if len(line.split(','))==2 and line.split(',')[0].strip()==self.uuid and int(line.split(',')[1].strip())!=os.getpid()]
    need(not foreign,'foreign GPU compute process');self.samples.append([rss,disk,free,gpu]);check_resources(rss,disk,free,gpu)
   except BaseException as e:self.fault=type(e).__name__+': '+str(e)[:300];os.kill(os.getpid(),signal.SIGUSR1);return
   self.stop_event.wait(5)
 def start(self):self.thread.start()
 def stop(self):self.stop_event.set();self.thread.join(15);return {'samples':len(self.samples),'peak_rss':max((x[0] for x in self.samples),default=0),'peak_disk':max((x[1] for x in self.samples),default=0),'minimum_free':min((x[2] for x in self.samples),default=0),'peak_gpu':max((x[3] for x in self.samples),default=0),'fault':self.fault,'stopped':not self.thread.is_alive(),'sampling_seconds':5,'unsampled_peak':'UNKNOWN'}

def run(a):
 need(not {'torch','transformers','datasets'} & set(sys.modules),'fresh process required');p=base.read(a.protocol);digest=base.sha(a.protocol);scope=authorize(p,digest)
 need(a.job_spec and base.sha(a.job_spec)==scope['launch_spec_sha256'],'spec mismatch');job=base.read(a.job_spec);argv=job['argv'];need(len(argv)==15 and argv[1].startswith('FFNDS_WORKER_CONFIG='),'job argv')
 profile=Path(argv[1].split('=',1)[1]);need(base.sha(profile)==p['worker_config_sha256'],'worker profile')
 expected=['env',argv[1],'bash',str(ROOT/'scripts/worker_teal_gpu0_uuid_exec.sh'),'taskset','-c','0,1',str(ENV/'bin/python'),'-B',str(ROOT/'tools/e0_teal_llama3_grab_acts_launcher.py'),'run','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())]
 need(argv==expected and job['id']==ID and job['protocol_sha256']==digest and Path(job['protocol_path']).resolve()==a.protocol.resolve() and Path(job['cwd']).resolve()==ROOT and job['hf_root']==str(HF) and Path(job['output_dir']).resolve()==OUT.parent and Path(job['run_json']).resolve()==OUT/'RUN.json' and job['external_timeout_seconds']==RES['outer_seconds'],'job mismatch')
 need(sorted(os.sched_getaffinity(0))==[0,1] and os.environ.get('FFNDS_PHYSICAL_GPU')=='0' and os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and Path(sys.prefix).resolve()==ENV.resolve(),'worker/GPU binding')
 for b in p['source_bindings']:base.bound(ROOT,b)
 for b in p['hf_source_bindings']+p['asset_receipts']:base.bound(HF,b)
 need(runtime_binding()==p['environment_binding'],'calibration environment changed')
 need(PARQUET.is_file() and not PARQUET.is_symlink() and PARQUET.stat().st_size==alpaca.SIZE and SNAP.is_dir(),'asset identity')
 ac=p['alpaca_closed'];alpaca.validate_closed(base.read(ROOT/ac['protocol']),base.sha(ROOT/ac['protocol']),base.read(HF/ac['run']),base.read(HF/ac['exit']),base.sha(HF/ac['run']),base.sha(ROOT/ac['spec']))
 model=base.read(ROOT/p['model_closed']);need(model.get('id')=='E0_TEAL_LLAMA3_PUBLIC_WEIGHT_ACQUISITION_20261029' and model.get('status')=='PASS_CLOSED_PUBLIC_ASSETS_NOT_MODEL_EXECUTED','model receipt incomplete')
 for k in ('verification_RUN','verification_exit'):base.bound(HF,model[k])
 vr=base.read(base.bound(HF,model['verification_RUN']));ve=base.read(base.bound(HF,model['verification_exit']))
 need(vr.get('status')=='COMPLETE_VERIFIED_PUBLIC_BYTES' and ve.get('process_exited') is True and ve.get('child_returncode')==0 and ve.get('run_sha256')==model['verification_RUN']['sha256'] and vr.get('completed_bytes')==16069766580,'model verify actual exit differs')
 need(base.sha(ROOT/p['model_closed'])==p['model_closed_sha256'],'model closure changed')
 import worker_identity;worker_identity.require_worker();need(not OUT.exists(),'attempt already exists');OUT.mkdir(parents=True,mode=0o700)
 start=time.monotonic();cpu=time.process_time();counts={'sort':0,'histogram':0,'layer_inputs':0,'histogram_files':0,'layer_forward_pre':0,'layer_forward_post':0,'activation_finite_reductions':0};r={'id':ID,'status':'STARTED','protocol_sha256':digest,'launch_spec_sha256':scope['launch_spec_sha256'],'automatic_retry':False,'predebit':LIMITS,'counts':counts,'sort_path':'GPU FP32; int64 indices; CPU FP32 capture and CPU histogram','actual_physical_kernel_launches':'UNKNOWN','attempts':{'sort':0,'histogram':0,'layer_inputs':0,'histogram_files':0},'layer_events':[],'network_bytes':'UNKNOWN; offline flags and socket connect denial','model_snapshot':str(SNAP),'model_closed_sha256':p['model_closed_sha256'],'alpaca_revision':alpaca.REV,'alpaca_sha256':alpaca.DIGEST,'finite_observer':{'predebit_reductions':64,'predebit_elements':5368709120,'completed_reductions':0,'completed_elements':0,'synchronized_wall_seconds':0.0}}
 def persist():base.write(OUT/'RUN.json',r)
 persist();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall limit')));signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('resource monitor failure')));signal.alarm(RES['wall_seconds']);mon=Monitor(OUT.parent,os.environ['CUDA_VISIBLE_DEVICES']);mon.start()
 try:
  cache=OUT/'cache';cache.mkdir(mode=0o700);tmp=cache/'tmp';tmp.mkdir(mode=0o700)
  os.environ.update(HF_HOME=str(cache/'hf'),HF_DATASETS_CACHE=str(cache/'datasets'),XDG_CACHE_HOME=str(cache/'xdg'),TMPDIR=str(tmp),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),CUDA_CACHE_PATH=str(cache/'cuda'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
  def no_network(*args,**kwargs):raise RuntimeError('network forbidden')
  socket.socket.connect=no_network
  import torch, datasets, transformers, flash_attn
  torch.cuda.init(); context=torch.empty(1,device='cuda');del context
  r['gpu_context_initialized']=True;persist()
  need(torch.__version__.split('+')[0]=='2.4.0' and transformers.__version__=='4.44.1' and flash_attn.__version__.split('+')[0]=='2.6.1' and torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(8,0) and 'A100' in torch.cuda.get_device_name(0),'author runtime/FA2/GPU mismatch')
  torch.set_num_threads(2);torch.set_num_interop_threads(2)
  acquisition_path=HF/'ffnds-derived/e0_teal_llama3_public_weight_acquisition_20261029/attempt1/worker/results/RUN.json'
  need(vr.get('parent_RUN',{}).get('sha256')==base.sha(acquisition_path),'acquisition linkage differs')
  acquisition=base.read(acquisition_path)
  need(acquisition.get('status')=='COMPLETE_PUBLIC_BYTES_NOT_MODEL_EXECUTED' and len(acquisition.get('files',[]))==13,'model acquisition RUN differs')
  r['model_hash_reads']=[];expected_bytes=0;asset_states={}
  for row in acquisition['files']:
   path=SNAP/row['filename'];blob=(HF/row['cache_path']);need(path.resolve()==blob.resolve() and blob.is_file() and not blob.is_symlink() and row['status']=='COMPLETE','snapshot mapping differs')
   item={'filename':row['filename'],'predebited_bytes':row['result']['bytes'],'completed_bytes':0,'progress':'UNKNOWN_UNTIL_COMPLETE'};r['model_hash_reads'].append(item);expected_bytes+=item['predebited_bytes'];need(expected_bytes<=16069766580,'model hash budget');persist()
   need(alpaca.common.verify_file(blob,{'bytes':item['predebited_bytes'],'sha256':row['result']['sha256'],'git_blob':None})['sha256']==row['result']['sha256'],'model hash mismatch');asset_states[str(blob)]=support.statid(blob.stat());item.update(completed_bytes=item['predebited_bytes'],progress='COMPLETE');persist()
  need(expected_bytes==16069766580,'model hash total differs')
  r['parquet_hash_predebit_bytes']=alpaca.SIZE;r['parquet_hash_progress']='UNKNOWN_UNTIL_COMPLETE';persist();need(alpaca.common.verify_file(PARQUET,{'bytes':alpaca.SIZE,'sha256':alpaca.DIGEST,'git_blob':None})['sha256']==alpaca.DIGEST,'parquet hash differs');parquet_state=support.statid(PARQUET.stat());r['parquet_hash_completed_bytes']=alpaca.SIZE;r['parquet_hash_progress']='COMPLETE';persist()
  original_load=datasets.load_dataset
  def local_dataset(*args,**kwargs):
   need(args and args[0]=='tatsu-lab/alpaca' and kwargs.get('streaming') is True,'unexpected dataset request')
   return original_load('parquet',data_files={'train':str(PARQUET)},streaming=True)
  def source():return original_load('parquet',data_files={'train':str(PARQUET)},streaming=True)['train']
  tokenizer=transformers.AutoTokenizer.from_pretrained(str(SNAP),use_fast=True,trust_remote_code=True,local_files_only=True)
  if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
  r['preflight_dataset_attempt']=1;persist();text=read_prefix(source());r['preflight_dataset_rows']=300;r['source_text_sha256']=hashlib.sha256(text.encode()).hexdigest()
  r['preflight_tokenize_attempt']=1;persist();encoded=tokenizer(text,truncation=True,return_tensors='pt',max_length=2048,return_overflowing_tokens=True,padding='max_length');need(encoded.input_ids.shape[0]>=10,'fewer than ten segments');r['total_segments']=int(encoded.input_ids.shape[0]);r['segments']=validate_segments(encoded.input_ids[:10,:],encoded.attention_mask[:10,:]);persist()
  del encoded,text,tokenizer
  datasets.load_dataset=local_dataset
  oldsort,oldhist,oldsave=torch.sort,torch.histogram,torch.save
  def invoke(kind,limit,fn,*args,**kwargs):
   r['attempts'][kind]+=1;need(r['attempts'][kind]<=limit,kind+' attempt limit');persist();value=fn(*args,**kwargs);counts[kind]+=1;persist();return value
  def sort(*args,**kwargs):
   need(args[0].is_cuda,'sort device');return invoke('sort',128,oldsort,*args,**kwargs)
  def hist(*args,**kwargs):
   need(not args[0].is_cuda,'hist device');return invoke('histogram',128,oldhist,*args,**kwargs)
  def save(obj,path,*args,**kwargs):
   path=Path(path);need(path.is_relative_to(OUT/'author'),'author write outside scope')
   if path.name.startswith('act_') and path.suffix=='.pt':kind,limit='layer_inputs',32
   elif path.name=='histograms.pt':kind,limit='histogram_files',64
   else:raise ValueError('unexpected author save')
   return invoke(kind,limit,oldsave,obj,path,*args,**kwargs)
  sys.path.insert(0,str(AUTHOR))
  import utils.utils as author_utils
  import utils.data as author_data
  need(Path(author_utils.__file__).resolve()==(AUTHOR/'utils/utils.py').resolve() and Path(author_data.__file__).resolve()==(AUTHOR/'utils/data.py').resolve(),'author utils import origin differs')
  for module_name,source in (('teal.model','teal/model.py'),('teal.mlp','teal/mlp.py'),('teal.self_attn','teal/self_attn.py'),('utils.eval_ppl','utils/eval_ppl.py')):
   module=importlib.import_module(module_name);need(Path(module.__file__).resolve()==(AUTHOR/source).resolve(),'author module origin differs')
  old_get_dataset=author_data.get_dataset
  def measured_dataset(*args,**kwargs):
   r['author_dataset_attempt']=1;persist();original=old_get_dataset(*args,**kwargs);rows=list(original);need(len(rows)==300,'author prefix count changed');text=''.join(row['text']+'\n\n' for row in rows);need(hashlib.sha256(text.encode()).hexdigest()==r['source_text_sha256'],'author source differs from preflight');r['author_dataset_rows']=len(rows);persist();return rows
  author_data.get_dataset=measured_dataset
  old_tokenizer=author_utils.get_tokenizer
  class CheckedTokenizer:
   def __init__(self,inner):self.inner=inner
   def __getattr__(self,name):return getattr(self.inner,name)
   def __call__(self,*args,**kwargs):
    r['author_tokenize_attempt']=1;persist();result=self.inner(*args,**kwargs);r['author_total_segments']=int(result.input_ids.shape[0]);need(result.input_ids.shape[0]>=10,'author tokenizer has fewer than ten segments');actual=validate_segments(result.input_ids[:10,:],result.attention_mask[:10,:]);need(actual==r['segments'],'author tokenization differs from preflight');r['author_segments']=actual;r['author_tokenization_verified']=True;persist();return result
  def checked_tokenizer(*args,**kwargs):return CheckedTokenizer(old_tokenizer(*args,**kwargs))
  author_utils.get_tokenizer=checked_tokenizer
  old_load=author_utils.get_sparse_model
  def checked_model(*args,**kwargs):
   need(r.get('model_load_calls',0)==0,'model load limit');r['model_load_calls']=1;persist();model=old_load(*args,**kwargs)
   need(all(str(device) in ('0','cuda:0') for device in getattr(model,'hf_device_map',{}).values()),'model device map differs')
   need(len(model.model.layers)==32 and model.config.hidden_size==4096 and model.config.intermediate_size==14336 and all(param.device.type=='cuda' and param.device.index==0 and param.dtype==torch.float16 for param in model.parameters()),'model shape/device/dtype differs')
   def check_tensor(x):
    need(r.get('author_tokenization_verified') is True and tuple(x.shape)==(10,2048,4096),'layer input/tokenization mismatch')
    counts['activation_finite_reductions']+=1;need(counts['activation_finite_reductions']<=64,'finite observer budget');t0=time.perf_counter();finite=torch.isfinite(x).all().item();elapsed=time.perf_counter()-t0
    observer=r['finite_observer'];observer['completed_reductions']+=1;observer['completed_elements']+=x.numel();observer['synchronized_wall_seconds']+=elapsed;persist();need(finite,'nonfinite layer activation')
   for index,layer in enumerate(model.model.layers):
    def before(module,inputs,index=index):
     need(counts['layer_forward_pre']==index and counts['layer_forward_post']==index,'layer forward order mismatch');r['layer_events'].append({'layer':index,'phase':'PREDEBIT','input_shape':list(inputs[0].shape)});counts['layer_forward_pre']+=1;check_tensor(inputs[0]);persist()
    def after(module,inputs,output,index=index):
     need(counts['layer_forward_pre']==index+1 and counts['layer_forward_post']==index,'layer forward completion order mismatch');check_tensor(output[0]);counts['layer_forward_post']+=1;r['layer_events'].append({'layer':index,'phase':'COMPLETE','output_shape':list(output[0].shape)});persist()
    layer.register_forward_pre_hook(before);layer.register_forward_hook(after)
   r['model_loaded']={'layers':32,'hidden_size':4096,'intermediate_size':14336,'dtype':'float16','device':'cuda'};persist();return model
  author_utils.get_sparse_model=checked_model
  oldfind=author_utils.ActivationModule.find_histogram
  def measured_find(self,*args,**kwargs):
   torch.sort=sort;torch.histogram=hist
   try:return oldfind(self,*args,**kwargs)
   finally:torch.sort=oldsort;torch.histogram=oldhist
  author_utils.ActivationModule.find_histogram=measured_find;torch.save=save
  sys.argv=[str(AUTHOR/'teal/grab_acts.py'),'--model_name',str(SNAP),'--output_path',str(OUT/'author')]
  need(all(support.statid(Path(path).stat())==identity for path,identity in asset_states.items()) and support.statid(PARQUET.stat())==parquet_state,'input asset changed before execution')
  runpy.run_path(str(AUTHOR/'teal/grab_acts.py'),run_name='__main__');check_counts(counts)
  need(all(support.statid(Path(path).stat())==identity for path,identity in asset_states.items()) and support.statid(PARQUET.stat())==parquet_state,'input asset changed during execution')
  files=[OUT/'author'/'histograms'/f'layer-{i}'/module/'histograms.pt' for i in range(32) for module in ('mlp','self_attn')]
  inputs=[OUT/'author'/'activations'/f'act_{i}.pt' for i in range(32)]
  need(all(x.is_file() and not x.is_symlink() for x in files+inputs),'author artifacts incomplete')
  r['histogram_validation_predebit_files']=64;r['histogram_validation_predebit_bytes']=268435456;r['histogram_validation_completed_bytes']=0;r['histogram_validation_predebit_cpu_reductions']=640;persist();summaries=[];hist_started=time.perf_counter()
  for file in files:
   size=file.stat().st_size;need(0<size<=2097152,'histogram file size limit')
   d=torch.load(file,map_location='cpu',weights_only=True);need(set(d)=={'h1','h2','h1_centers','h2_centers'},'hist keys differ')
   need(all(tuple(v.shape)==(10000,) and torch.isfinite(v).all().item() for v in d.values()),'hist shape or finite')
   for key in ('h1','h2'):need((d[key]>=0).all().item() and d[key].sum().item()>0 and (d[key+'_centers'][1:]>=d[key+'_centers'][:-1]).all().item(),'invalid histogram CDF')
   summaries.append({'path':str(file.relative_to(OUT)),'bytes':size,'sha256':base.sha(file)});r['histogram_validation_completed_bytes']+=size*2;persist()
  r['histogram_validation_wall_seconds']=time.perf_counter()-hist_started;r['histogram_validation_completed_files']=64;r['histogram_files']=summaries;r['layer_input_files']=[{'path':str(f.relative_to(OUT)),'bytes':f.stat().st_size} for f in inputs];r['status']='COMPLETE_AUTHOR_CALIBRATION_EXIT_PENDING'
 except BaseException as e:r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:
  signal.alarm(0);r.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,monitor=mon.stop())
  try:
   r['final_worker_usage']=support.worker_usage(OUT.parent)
   if r['final_worker_usage']['apparent_bytes']>RES['new_HF_bytes'] or r['peak_RSS_bytes']>RES['host_RSS_max_bytes']:r['status']='FAILED_MONITOR'
  except BaseException as e:r.update(status='FAILED_MONITOR',final_usage_error=type(e).__name__)
  if r['monitor']['fault'] or not r['monitor']['stopped'] or not r['monitor']['samples']:r['status']='FAILED_MONITOR'
  persist()
  if r['status']=='FAILED_MONITOR':raise RuntimeError('monitor failure')

def draft(profile):
 alp_protocol=ROOT/'config/e0_teal_alpaca_parquet_acquisition_20261030_pending_review_v3.json'
 alp_spec=ROOT/'.runtime/teal_alpaca_parquet_acquisition_20261030/CPU_JOB_V3.json'
 alp_run=HF/alpaca.OUT/'RUN.json';alp_exit=(HF/alpaca.OUT).parent/'EXIT_RECEIPT.json'
 alpaca.validate_closed(base.read(alp_protocol),base.sha(alp_protocol),base.read(alp_run),base.read(alp_exit),base.sha(alp_run),base.sha(alp_spec))
 model_path=ROOT/'.runtime/teal_llama3_public_weight_acquisition_20261029/CLOSED_VERIFICATION.json';model=base.read(model_path)
 need(model.get('status')=='PASS_CLOSED_PUBLIC_ASSETS_NOT_MODEL_EXECUTED','model closure required')
 for k in ('verification_RUN','verification_exit'):base.bound(HF,model[k])
 source_names=('tools/e0_teal_llama3_grab_acts_launcher.py','tests/test_e0_teal_llama3_grab_acts_launcher.py','tools/e0_teal_alpaca_acquisition.py','tools/e0_teal_llama3_author_checkpoint_convert.py','tools/e0_teal_llama3_tokenizer_byte_bridge.py','tools/worker_identity.py','tools/durable_worker_job.py','scripts/worker_teal_gpu0_uuid_exec.sh')
 return {'id':ID,'status':'FROZEN_PENDING_REGISTRATION','resources':RES,'limits':LIMITS,'model_snapshot':str(SNAP),'parquet':str(PARQUET),'output':str(OUT),'model_closed':str(model_path.relative_to(ROOT)),'model_closed_sha256':base.sha(model_path),'alpaca_closed':{'protocol':str(alp_protocol.relative_to(ROOT)),'spec':str(alp_spec.relative_to(ROOT)),'run':str(alp_run.relative_to(HF)),'exit':str(alp_exit.relative_to(HF))},'asset_receipts':[base.binding(alp_run,HF),base.binding(alp_exit,HF),model['verification_RUN'],model['verification_exit']],'source_bindings':[base.binding(ROOT/x,ROOT) for x in source_names],'hf_source_bindings':[base.binding(AUTHOR/x,HF) for x in AUTHOR_FILES],'environment_binding':runtime_binding(),'worker_config_sha256':base.sha(profile),'review':None}

def inspect_closed(protocol,spec):
 p=base.read(protocol);validate(p);runpath=OUT/'RUN.json';exitpath=OUT.parent/'EXIT_RECEIPT.json';r=base.read(runpath);e=base.read(exitpath);d=base.sha(protocol);specsha=base.sha(spec)
 need(r.get('id')==ID and r.get('protocol_sha256')==d and r.get('launch_spec_sha256')==specsha and r.get('status')=='COMPLETE_AUTHOR_CALIBRATION_EXIT_PENDING','calibration RUN incomplete')
 need(e.get('id')==ID and e.get('process_exited') is True and e.get('child_returncode')==0 and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('run_sha256')==base.sha(runpath) and e.get('protocol_sha256')==d and e.get('spec_sha256')==specsha,'actual EXIT0 missing')
 check_counts(r['counts']);need(r.get('author_tokenization_verified') is True and r.get('author_segments')==r.get('segments') and len(r.get('layer_events',[]))==64,'author token or layer evidence missing');need(r.get('finite_observer',{}).get('completed_elements')==5368709120 and r.get('finite_observer',{}).get('completed_reductions')==64 and r.get('model_load_calls')==1 and r.get('model_loaded',{}).get('layers')==32 and r.get('histogram_validation_completed_files')==64 and r.get('histogram_validation_completed_bytes',268435457)<=268435456 and r.get('parquet_hash_completed_bytes')==alpaca.SIZE and sum(x.get('completed_bytes',0) for x in r.get('model_hash_reads',[]))==16069766580,'hash/count incomplete')
 mon=r['monitor'];need(mon['samples']>0 and mon['stopped'] and not mon['fault'] and mon['peak_rss']<=RES['host_RSS_max_bytes'] and mon['peak_gpu']<=RES['GPU_memory_max_bytes'] and mon['peak_disk']<=RES['new_HF_bytes'],'resource monitoring incomplete')
 expected={f'author/histograms/layer-{i}/{module}/histograms.pt' for i in range(32) for module in ('mlp','self_attn')};rows=r['histogram_files'];need(len(rows)==64 and {x['path'] for x in rows}==expected,'hist names differ')
 for row in rows:
  file=OUT/row['path'];need(file.is_file() and not file.is_symlink() and file.stat().st_size==row['bytes'] and len(row['sha256'])==64,'hist artifact stat differs')
 need(len(r.get('layer_input_files',[]))==32,'layer inputs missing')
 return {'id':ID,'status':'PASS_ACTUAL_EXIT_AUTHOR_CALIBRATION','RUN':base.binding(runpath,HF),'exit':base.binding(exitpath,HF),'protocol_sha256':d,'spec_sha256':specsha,'histogram_files':rows,'histogram_sha_identity':'recorded in producer, current stat only; no unbudgeted reread'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['draft','inspect','run','inspect-closed']);ap.add_argument('--protocol',type=Path);ap.add_argument('--job-spec',type=Path);ap.add_argument('--worker-profile',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
 if a.action=='draft':need(a.worker_profile and a.out and not a.out.exists(),'fresh output and profile required');base.write(a.out,draft(a.worker_profile))
 elif a.action=='inspect':p=base.read(a.protocol);validate(p);print(json.dumps({'fingerprint':fingerprint(p),'resources':RES,'limits':LIMITS},indent=2))
 elif a.action=='run':run(a)
 else:print(json.dumps(inspect_closed(a.protocol,a.job_spec),indent=2))
if __name__=='__main__':main()
