"""Reviewed, bounded wrapper around the unmodified author HF->gpt-fast converter.

No assets are inferred from a draft. Actual RUN requires separately reviewed
ACTIVE, exact asset-closure receipts, source/runtime bindings, and a fresh job.
"""
import argparse, fcntl, hashlib, stat, importlib, importlib.util, importlib.metadata, json, os, resource, shutil, signal, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_teal_llama3_tokenizer_byte_bridge as helper
HF=Path('/home/xiang/.cache/huggingface');ID='E0_TEAL_LLAMA3_AUTHOR_CHECKPOINT_CONVERSION_20261029'
OUTPUT='ffnds-derived/e0_teal_llama3_author_checkpoint_conversion_20261029/attempt1/worker/results'
VERIFY='ffnds-derived/e0_teal_llama3_author_checkpoint_conversion_20261029/attempt1/verification'
VERIFY_RESOURCES={'GPU':0,'CPU_threads':1,'wall_seconds':3600,'outer_seconds':3720,'max_file_read_bytes':21474836480,'new_HF_bytes':67108864,'automatic_retry':False}
VERIFY_LIMITS={'output_full_hash_passes':1,'model_forward_calls':0,'GPU_calls':0}
AUTHOR='ffnds-research-sources/p0_teal_full_llm_20261029/gpt-fast/scripts/convert_hf_checkpoint.py'
AUTHOR_SHA='b5878af3722eae1d060e7d10820451b883ae0cd13ef8f9f91c356a28ca7a2d7c'
AUTHOR_MODEL='ffnds-research-sources/p0_teal_full_llm_20261029/gpt-fast/model.py'
AUTHOR_MODEL_SHA='a1ce36c8d432a7978684f0610bdecafc6b1f9e3fff7cc4527baef362251fe176'
KERNEL='ffnds-research-sources/p0_teal_full_llm_20261029/kernels/sparse_gemv.py'
KERNEL_SHA='21183499ac9eb06c57365023388920721636b5bc0d2acf5d91c95fd5e48c37a3'
TOKEN='ffnds-derived/e0_teal_llama3_tokenizer_byte_bridge_20261029/attempt1/worker/results/tokenizer.model'
TOKEN_SHA='82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55'
SELF='tools/e0_teal_llama3_author_checkpoint_convert.py';TEST='tests/test_e0_teal_llama3_author_checkpoint_convert.py'
COMPILE_WRAPPER='.runtime/teal_fb7373c93ac3594817c9ee64d4e08b47430a1822/kernels/compile_wrapper.py'
COMPILE_SHA='7e1b86358171923d3ac658476b483c5b1dc199601b84976264d5814933239d7c'
SOURCES=(SELF,TEST,COMPILE_WRAPPER,'tools/e0_teal_llama3_tokenizer_byte_bridge.py','tools/worker_identity.py','tools/durable_worker_job.py','research/literature/20261029/P0_TEAL_FULL_LLM_STATIC_SOURCE_MANIFEST_20261029.json','research/method_discussion/discoveries/E0/p0-teal-llama3-author-full32-execution-design-20261029.md')
HF_SOURCES={AUTHOR:AUTHOR_SHA,AUTHOR_MODEL:AUTHOR_MODEL_SHA,KERNEL:KERNEL_SHA}
RESOURCES={'GPU':0,'CPU_threads':2,'wall_seconds':5400,'outer_seconds':5520,'address_space_max_bytes':51539607552,'minimum_free_disk_bytes':25769803776,'output_file_max_bytes':21474836480,'new_HF_bytes':22548578304,'automatic_retry':False}
LIMITS={'original_converter_calls':1,'input_shard_loads':4,'checkpoint_writes':1,'model_forward_calls':0,'GPU_calls':0,'training_updates':0,'output_full_hash_passes':1,'input_full_hash_passes_per_file':2}
ENV='/home/xiang/.cache/huggingface/ffnds-derived/teal_author_llm_env_20261029/venv/bin/python'
need=helper.need;read=helper.read;sha=helper.sha;binding=helper.binding;bound=helper.bound;write=helper.write

def source_expected():
 d={'model.embed_tokens.weight':(128256,4096),'model.norm.weight':(4096,),'lm_head.weight':(128256,4096)}
 for i in range(32):
  p=f'model.layers.{i}.'
  for k,shape in {'self_attn.q_proj.weight':(4096,4096),'self_attn.k_proj.weight':(1024,4096),'self_attn.v_proj.weight':(1024,4096),'self_attn.o_proj.weight':(4096,4096),'mlp.gate_proj.weight':(14336,4096),'mlp.up_proj.weight':(14336,4096),'mlp.down_proj.weight':(4096,14336),'input_layernorm.weight':(4096,),'post_attention_layernorm.weight':(4096,)}.items():d[p+k]=shape
 return d

def output_expected():
 d={'tok_embeddings.weight':(128256,4096),'norm.weight':(4096,),'output.weight':(128256,4096)}
 for i in range(32):
  p=f'layers.{i}.'
  for k,shape in {'attention.wqkv.weight':(6144,4096),'attention.wo.weight':(4096,4096),'feed_forward.w1.weight':(14336,4096),'feed_forward.w3.weight':(14336,4096),'feed_forward.w2.weight':(4096,14336),'attention_norm.weight':(4096,),'ffn_norm.weight':(4096,)}.items():d[p+k]=shape
 return d

def check_shapes(actual,expected):
 need(set(actual)==set(expected),'key set mismatch');need(all(tuple(actual[k])==v for k,v in expected.items()),'tensor shape mismatch')

def source_module():
 src=HF/AUTHOR;need(sha(src)==AUTHOR_SHA and sha(HF/AUTHOR_MODEL)==AUTHOR_MODEL_SHA and sha(HF/KERNEL)==KERNEL_SHA and sha(ROOT/COMPILE_WRAPPER)==COMPILE_SHA,'author source mismatch')
 author_root=HF/'ffnds-research-sources/p0_teal_full_llm_20261029';sys.path.insert(0,str((ROOT/COMPILE_WRAPPER).parent));sys.path.insert(0,str(author_root));sys.path.insert(0,str(author_root/'gpt-fast'))
 spec=importlib.util.spec_from_file_location('ffnds_pinned_author_converter',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def runtime():
 ps=('torch','triton','safetensors','transformers','tokenizers','tiktoken','sentencepiece','blobfile','numpy','msgspec','setuptools','wheel')
 records=[]
 for name in ps:
  d=importlib.metadata.distribution(name);entries=[]
  for f in d.files or []:
   text=str(f)
   if text.endswith(('.dist-info/METADATA','.dist-info/RECORD')) or text in ('torch/__init__.py','torch/version.py','torch/_C.cpython-311-x86_64-linux-gnu.so','triton/__init__.py','safetensors/__init__.py','safetensors/torch.py','msgspec/__init__.py'):
    path=Path(d.locate_file(f)).resolve()
    if path.is_file():entries.append([text,path.stat().st_size,sha(path)])
  entries.sort();need(entries,'package files missing: '+name);records.append({'name':name,'version':d.version,'files':entries})
 pth=Path(sys.prefix)/'lib/python3.11/site-packages/ffnds_teal_kernel_v2_readonly.pth'
 return {'python':str(Path(sys.executable).resolve()),'prefix':sys.prefix,'python_sha256':sha(Path(sys.executable).resolve()),'version':sys.version.split()[0],'site_bridge_sha256':sha(pth),'packages':records,'native_shared_libraries':'distribution RECORD identity; individual ELF bytes NOT_HASHED'}

def fingerprint(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def draft(asset_run,asset_exit,verify_run,verify_exit):
 # The acquisition and independently registered verifier must actually have exited.
 ar=read(asset_run);ae=read(asset_exit);vr=read(verify_run);ve=read(verify_exit)
 need(ae.get('process_exited') is True and ae.get('timeout') is False and ae.get('automatic_retry') is False and ae.get('id')=='E0_TEAL_LLAMA3_PUBLIC_WEIGHT_ACQUISITION_20261029' and ae.get('child_returncode')==0 and ae.get('run_sha256')==sha(asset_run) and ar.get('status')=='COMPLETE_PUBLIC_BYTES_NOT_MODEL_EXECUTED','asset acquisition EXIT0 required')
 need(ve.get('process_exited') is True and ve.get('timeout') is False and ve.get('automatic_retry') is False and ve.get('id')==ae['id']+'_VERIFY' and ve.get('child_returncode')==0 and ve.get('run_sha256')==sha(verify_run) and vr.get('status')=='COMPLETE_VERIFIED_PUBLIC_BYTES' and vr.get('parent_RUN',{}).get('sha256')==sha(asset_run) and vr.get('parent_exit',{}).get('sha256')==sha(asset_exit),'asset verification EXIT0 required')
 weights=[x for x in ar['files'] if x['filename'].endswith('.safetensors')];index=[x for x in ar['files'] if x['filename']=='model.safetensors.index.json'];config=[x for x in ar['files'] if x['filename']=='config.json'];need(len(weights)==4 and len(index)==len(config)==1 and all(x['status']=='COMPLETE' for x in weights+index+config),'asset list differs');need({x['filename'] for x in weights}=={f'model-{i:05d}-of-00004.safetensors' for i in range(1,5)},'shard names differ');need(len({x['cache_path'] for x in weights+index+config})==6,'duplicate asset paths')
 need(all(x['cache_path'].startswith('ffnds-derived/e0_teal_llama3_public_weight_acquisition_20261029/attempt1/worker/scoped_hf_cache/models--NousResearch--Meta-Llama-3-8B/blobs/') and '..' not in Path(x['cache_path']).parts for x in weights+index+config),'asset cache provenance')
 assets=[{'filename':x['filename'],'path':x['cache_path'],'bytes':x['result']['bytes'],'sha256':x['result']['sha256']} for x in weights+index+config]
 return {'id':ID,'status':'DRAFT_REVIEW_REQUESTED','resources':RESOURCES,'limits':LIMITS,'output_HF_relative':OUTPUT,'verification_HF_relative':VERIFY,'verification_resources':VERIFY_RESOURCES,'verification_limits':VERIFY_LIMITS,'author_function':AUTHOR,'author_model_name':'llama-3-8b','source_bindings':[binding(ROOT/x,ROOT) for x in SOURCES],'hf_source_bindings':[binding(HF/x,HF) for x in HF_SOURCES],'tokenizer':{'path':TOKEN,'bytes':2183982,'sha256':TOKEN_SHA},'asset_receipts':[binding(x,HF) for x in [asset_run,asset_exit,verify_run,verify_exit]],'assets':assets,'runtime':runtime(),'worker_config_sha256':None,'review':None,'automatic_retry':False}

def validate(p,frozen=False):
 need(p['id']==ID and p['resources']==RESOURCES and p['limits']==LIMITS and p['output_HF_relative']==OUTPUT and p['verification_HF_relative']==VERIFY and p['verification_resources']==VERIFY_RESOURCES and p['verification_limits']==VERIFY_LIMITS and p['author_function']==AUTHOR and p['author_model_name']=='llama-3-8b' and p['automatic_retry'] is False,'protocol mismatch')
 need({x['path'] for x in p['source_bindings']}==set(SOURCES) and {x['path']:x['sha256'] for x in p['hf_source_bindings']}==HF_SOURCES,'source closure')
 need(p['tokenizer']=={'path':TOKEN,'bytes':2183982,'sha256':TOKEN_SHA} and len(p['asset_receipts'])==4 and len(p['assets'])==6 and len({x['path'] for x in p['assets']})==6 and {x['filename'] for x in p['assets']}=={*(f'model-{i:05d}-of-00004.safetensors' for i in range(1,5)),'model.safetensors.index.json','config.json'},'actual assets missing')
 if frozen:need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['review'] is None and isinstance(p['worker_config_sha256'],str) and len(p['worker_config_sha256'])==64,'not frozen')

def authorize(p,digest,verify=False):
 validate(p,True);scope_id=ID+'_VERIFY' if verify else ID;resources=VERIFY_RESOURCES if verify else RESOURCES;limits=VERIFY_LIMITS if verify else LIMITS;a=read(ROOT/'research/ACTIVE_RESEARCH.json');sc=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==scope_id]
 need(a.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') and len(sc)==1,'ACTIVE missing');s=sc[0];need(s.get('protocol_sha256')==digest and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('resources')==resources and s.get('limits')==limits and s.get('max_gpu_cards')==0,'ACTIVE mismatch')
 v=read(bound(ROOT,s['review']));need(v.get('scope_id')==scope_id and v.get('state')=='REVIEWER_RECHECK_APPROVED' and v.get('recheck_disposition')=='APPROVE' and v.get('reviewer_disposition') in ('REVISE','APPROVE') and v.get('reviewer_agent_id') and v.get('author_agent_id') and v['reviewer_agent_id']!=v['author_agent_id'] and v.get('responses_to_review') and v.get('adversarial_counterexample') and v.get('approval_scope_fingerprint')==fingerprint(p),'review missing')
 need(v.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['source_bindings']},'review closure');[bound(ROOT,v[k]) for k in ('author_review','independent_review','reviewer_recheck')];return s

def statid(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
def worker_usage(path):
 root=Path(os.path.abspath(path));root_stat=root.lstat();need(stat.S_ISDIR(root_stat.st_mode),'worker root is not a directory');root_identity=(root_stat.st_dev,root_stat.st_ino);total=0;count=0;missing=0
 def onerror(exc):
  nonlocal missing
  if isinstance(exc,FileNotFoundError) and exc.filename:
   vanished=Path(os.path.abspath(exc.filename))
   if vanished!=root and vanished.is_relative_to(root):missing+=1;return
  raise exc
 for parent,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
  for name in files:
   p=Path(parent)/name
   try:st=p.lstat()
   except FileNotFoundError:missing+=1;continue
   if not stat.S_ISLNK(st.st_mode):total+=st.st_size;count+=1
 end=root.lstat();need(stat.S_ISDIR(end.st_mode) and (end.st_dev,end.st_ino)==root_identity,'worker root disappeared or changed during scan')
 return {'apparent_bytes':total,'regular_files':count,'missing_entries':missing}

def checked_persist(path,report,worker,budget):
 # Includes supervisor logs and atomic temp allowance; output file has its own 20 GiB limit.
 size=worker_usage(worker);report['worker_usage_before_persist']=size
 need(size['apparent_bytes']<=budget-(1<<20),'whole worker output budget exceeded')
 data=json.dumps(report,indent=2,allow_nan=False).encode()+b'\n';need(len(data)<(1<<20),'RUN exceeds reserved atomic size');tmp=path.with_suffix('.tmp');tmp.write_bytes(data);os.replace(tmp,path)
class UsageMonitor:
 def __init__(self,worker,budget,min_free):
  self.worker=worker;self.budget=budget;self.min_free=min_free;self.stop_event=threading.Event();self.fault=None;self.peak=0;self.minimum_free=None;self.samples=0;self.missing_entries=0;self.thread=threading.Thread(target=self._loop,daemon=True)
 def _loop(self):
  while not self.stop_event.is_set():
   try:
    observation=worker_usage(self.worker);used=observation['apparent_bytes'];self.missing_entries+=observation['missing_entries'];free=shutil.disk_usage(self.worker).free;self.peak=max(self.peak,used);self.minimum_free=free if self.minimum_free is None else min(self.minimum_free,free);self.samples+=1
    if used>self.budget-(1<<20) or free<self.min_free:self.fault='worker budget or free disk exceeded';os.kill(os.getpid(),signal.SIGUSR1);return
   except BaseException as e:self.fault='monitor error '+type(e).__name__;os.kill(os.getpid(),signal.SIGUSR1);return
   self.stop_event.wait(5)
 def start(self):self.thread.start()
 def stop(self):self.stop_event.set();self.thread.join(timeout=10);return {'samples':self.samples,'missing_entries':self.missing_entries,'peak_apparent_bytes':self.peak,'minimum_free_disk_bytes':self.minimum_free,'fault':self.fault,'thread_stopped':not self.thread.is_alive()}
EMERGENCY_BYTES=65536
def reserve_emergency(directory):
 path=Path(directory)/'EMERGENCY_FAILURE.reserve';fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
 try:os.write(fd,b' '*EMERGENCY_BYTES);os.fsync(fd)
 finally:os.close(fd)
 return path
def emergency_failure(path,report,reason):
 obj={'id':report.get('id'),'status':'FAILED_EMERGENCY_RECEIPT','reason':reason,'prior_status':report.get('status'),'predebited':report.get('predebited'),'completed':report.get('completed'),'hash_reads':report.get('hash_reads'),'worker_usage_on_exit':report.get('worker_usage_on_exit'),'monitor':report.get('whole_worker_monitor')};data=json.dumps(obj,allow_nan=False).encode();need(len(data)<EMERGENCY_BYTES,'emergency receipt too large');fd=os.open(path,os.O_WRONLY|os.O_NOFOLLOW)
 try:need(os.pwrite(fd,data+b' '*(EMERGENCY_BYTES-len(data)),0)==EMERGENCY_BYTES,'emergency short write');os.fsync(fd)
 finally:os.close(fd)
def stage_pinned(checkpoint,shards,index,token,pins):
 checkpoint.mkdir(mode=0o700);(checkpoint/'original').mkdir(mode=0o700)
 for item in shards+[index]:os.symlink('/proc/self/fd/'+str(pins[item['path']]),checkpoint/item['filename'])
 os.symlink('/proc/self/fd/'+str(pins[token['path']]),checkpoint/'original'/'tokenizer.model')
def check_pins(pins,states):need(all(statid(os.fstat(fd))==states[path] for path,fd in pins.items()),'input asset changed during conversion')
def pin_file(path):
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
 except BaseException:os.close(fd);raise
 return fd

def tracked_hash(path,expected,report,persist,label):
 path=Path(path);row={'label':label,'predebited_bytes':expected,'completed_bytes':0,'progress_bytes':0,'max_unrecorded_bytes':(65<<20),'status':'INFLIGHT'};report.setdefault('hash_reads',[]).append(row);persist();h=hashlib.sha256();n=0;mark=64<<20
 with path.open('rb') as f:
  st=os.fstat(f.fileno());need(st.st_size==expected,'file size changed')
  while True:
   b=f.read(1<<20)
   if not b:break
   h.update(b);n+=len(b)
   if n>=mark:row['progress_bytes']=n;persist();mark=n+(64<<20)
  need(statid(os.fstat(f.fileno()))==statid(st) and statid(path.stat())==statid(st),'file changed during read')
 row.update(completed_bytes=n,progress_bytes=n,max_unrecorded_bytes=0,status='COMPLETE');persist();return h.hexdigest()

def asset_source_headers(assets,pins):
 from safetensors import safe_open
 mapping={};types={};owner={}
 for a in assets:
  with safe_open('/proc/self/fd/'+str(pins[a['path']]),framework='pt',device='cpu') as f:
   for k in f.keys():
    need(k not in mapping,'duplicate source tensor');v=f.get_slice(k);mapping[k]=tuple(v.get_shape());types[k]=v.get_dtype();owner[k]=a['filename']
 check_shapes(mapping,source_expected());need(set(types.values())=={'BF16'},'source dtype differs');return mapping,types,owner

def run(a):
 need(not {'torch','triton','safetensors'} & set(sys.modules),'fresh process required');p=read(a.protocol);d=sha(a.protocol);scope=authorize(p,d);need(a.job_spec and sha(a.job_spec)==scope['launch_spec_sha256'],'job differs');job=read(a.job_spec);argv=job['argv'];need(len(argv)==15 and argv[1].startswith('FFNDS_WORKER_CONFIG='),'job argv');profile=Path(argv[1].split('=',1)[1]);need(sha(profile)==p['worker_config_sha256'],'profile differs')
 expected=['env',argv[1],'CUDA_VISIBLE_DEVICES=','taskset','-c','0,1',ENV,'-I','-B',str(ROOT/SELF),'run','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())];out=HF/OUTPUT
 need(argv==expected and job['id']==ID and job['protocol_sha256']==d and Path(job['protocol_path']).resolve()==a.protocol.resolve() and Path(job['cwd']).resolve()==ROOT and job['hf_root']==str(HF) and Path(job['output_dir']).resolve()==out.parent and Path(job['run_json']).resolve()==out/'RUN.json' and job['external_timeout_seconds']==RESOURCES['outer_seconds'],'supervisor identity')
 need(sys.flags.isolated and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sorted(os.sched_getaffinity(0))==[0,1] and runtime()==p['runtime'],'runtime differs')
 for x in p['source_bindings']:bound(ROOT,x)
 for x in p['hf_source_bindings']+p['asset_receipts']:bound(HF,x)
 import worker_identity;worker_identity.require_worker();need(shutil.disk_usage(HF).free>=RESOURCES['minimum_free_disk_bytes'],'free disk')
 need(not out.exists(),'fresh attempt required');out.mkdir(parents=True,mode=0o700);t=time.monotonic();cpu=time.process_time();r={'id':ID,'protocol_sha256':d,'launch_spec_sha256':scope['launch_spec_sha256'],'status':'STARTED_NO_RETRY','predebited':dict.fromkeys(LIMITS,0),'completed':dict.fromkeys(LIMITS,0),'automatic_retry':False,'source_numeric_path':'BF16','author_converter_unmodified':True,'hash_reads':[],'io_semantics':'logical application operations; kernel page-cache/block IO UNKNOWN','unknowns':'native torch.save read/write detail and allocator transient RSS sampled only'}
 emergency=reserve_emergency(out)
 def persist():checked_persist(out/'RUN.json',r,out.parent,RESOURCES['new_HF_bytes'])
 persist();pins={};signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('worker output monitor failure')));monitor=UsageMonitor(out.parent,RESOURCES['new_HF_bytes'],2<<30);monitor.start()
 try:
  resource.setrlimit(resource.RLIMIT_AS,(RESOURCES['address_space_max_bytes'],)*2);oldsoft,oldhard=resource.getrlimit(resource.RLIMIT_FSIZE);need(oldhard==resource.RLIM_INFINITY or oldhard>=RESOURCES['output_file_max_bytes'],'file limit');resource.setrlimit(resource.RLIMIT_FSIZE,(RESOURCES['output_file_max_bytes'],oldhard))
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('conversion wall limit')));signal.alarm(RESOURCES['wall_seconds'])
  os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
  assets={x['filename']:x for x in p['assets']};shards=sorted([x for x in p['assets'] if x['filename'].endswith('.safetensors')],key=lambda x:x['filename']);need(len(shards)==4,'four shards required')
  for x in p['assets']+[p['tokenizer']]:
   path=HF/x['path'];need(path.is_file() and not path.is_symlink(),'asset is not a regular blob');pins[x['path']]=pin_file(path)
   actual=tracked_hash('/proc/self/fd/'+str(pins[x['path']]),x['bytes'],r,persist,'input:'+x.get('filename','tokenizer'));need(actual==x['sha256'],'asset hash mismatch')
  input_stats={path:statid(os.fstat(fd)) for path,fd in pins.items()}
  tokenizer=p['tokenizer']
  import torch;torch.set_num_threads(2);torch.set_num_interop_threads(1);need(torch.__version__.startswith('2.4.0'),'torch version');author=source_module();model_config=author.ModelArgs.from_name('llama-3-8b');need((model_config.n_layer,model_config.dim,model_config.intermediate_size,model_config.n_head,model_config.n_local_heads,model_config.head_dim,model_config.vocab_size,model_config.rope_base)==(32,4096,14336,32,8,128,128256,500000),'author model config differs');r['actual_author_model_config']=model_config.__dict__;persist()
  r['source_header_scans_predebited']=4;r['source_header_read_bound_bytes']=sum(x['bytes'] for x in shards);persist();shape,dtypes,owner=asset_source_headers(shards,pins);r['source_header_scans_completed']=4;index=read('/proc/self/fd/'+str(pins[assets['model.safetensors.index.json']['path']]));need(index['weight_map']==owner,'index and header disagree')
  config=read('/proc/self/fd/'+str(pins[assets['config.json']['path']]));need(config['hidden_size']==4096 and config['intermediate_size']==14336 and config['num_hidden_layers']==32 and config['num_attention_heads']==32 and config['num_key_value_heads']==8 and config['vocab_size']==128256,'model config differs')
  r['source_headers']={'keys':len(shape),'dtypes':sorted(set(dtypes.values())),'indexed_shards':4};persist()
  checkpoint=out/'llama-3-8b';stage_pinned(checkpoint,shards,assets['model.safetensors.index.json'],tokenizer,pins)
  native_load=author.load_safetensors_file;native_save=torch.save
  def load_wrapper(path,*args,**kw):
   need(r['predebited']['input_shard_loads']<4,'extra shard');name=Path(path).name;need(name in assets and name.endswith('.safetensors'),'unexpected shard');r.setdefault('author_shard_loads',[]).append({'filename':name,'predebited_bytes':assets[name]['bytes'],'completed':False});r['predebited']['input_shard_loads']+=1;persist();result=native_load(path,*args,**kw);r['completed']['input_shard_loads']+=1;r['author_shard_loads'][-1]['completed']=True;persist();return result
  def save_wrapper(obj,path,*args,**kw):
   need(Path(path)==checkpoint/'model.pth' and r['predebited']['checkpoint_writes']==0,'unexpected checkpoint write');check_shapes({k:tuple(v.shape) for k,v in obj.items()},output_expected());need({str(v.dtype) for v in obj.values()}=={'torch.bfloat16'},'unexpected output dtype');r['output_keys_before_save']=len(obj);r['predebited']['checkpoint_writes']=1;r['checkpoint_write_predebited_max_bytes']=RESOURCES['output_file_max_bytes'];persist();x=native_save(obj,path,*args,**kw);r['completed']['checkpoint_writes']=1;r['checkpoint_actual_bytes']=Path(path).stat().st_size;persist();return x
  author.load_safetensors_file=load_wrapper;torch.save=save_wrapper
  try:
   r['predebited']['original_converter_calls']=1;r['author_tokenizer_copy_predebited_bytes']=tokenizer['bytes'];persist();author.convert_hf_checkpoint(checkpoint_dir=checkpoint,model_name='llama-3-8b');r['completed']['original_converter_calls']=1;r['author_tokenizer_copy_completed_bytes']=tokenizer['bytes'];persist()
  finally:author.load_safetensors_file=native_load;torch.save=native_save
  result=checkpoint/'model.pth';need(result.is_file() and not result.is_symlink() and 0<result.stat().st_size<=RESOURCES['output_file_max_bytes'],'output invalid');need((checkpoint/'tokenizer.model').is_file() and tracked_hash(checkpoint/'tokenizer.model',tokenizer['bytes'],r,persist,'author_tokenizer_copy')==TOKEN_SHA,'author tokenizer copy differs');check_pins(pins,input_stats)
  for x in p['assets']+[p['tokenizer']]:
   actual=tracked_hash('/proc/self/fd/'+str(pins[x['path']]),x['bytes'],r,persist,'post_conversion:'+x.get('filename','tokenizer'));need(actual==x['sha256'],'input asset changed during conversion')
  r['predebited']['output_full_hash_passes']=1;persist();r['output']={'path':str(result.relative_to(HF)),'bytes':result.stat().st_size,'sha256':tracked_hash(result,result.stat().st_size,r,persist,'output:model.pth')};r['completed']['output_full_hash_passes']=1;persist()
  r['checkpoint_inspection_predebited']=1;r['checkpoint_inspection_read_bound_bytes']=result.stat().st_size;persist();saved=torch.load(str(result),map_location='cpu',mmap=True,weights_only=True);check_shapes({k:tuple(v.shape) for k,v in saved.items()},output_expected());need({str(v.dtype) for v in saved.values()}=={'torch.bfloat16'},'saved dtype differs');r['finite_samples_predebited']=32*len(saved);persist();bad=[]
  for key,value in saved.items():
   sample=value.reshape(-1)[:32];
   if not bool(torch.isfinite(sample).all().item()):bad.append(key)
  r.update(checkpoint_inspection_completed=1,finite_samples_checked=32*len(saved),nonfinite_sample_keys=bad);need(not bad,'nonfinite sample');del saved
  need(r['predebited']==r['completed'] and r['completed']['input_shard_loads']==4,'operation ledger incomplete');r['status']='COMPLETE_AUTHOR_CONVERTED_BF16_NOT_LLM'
 except BaseException as e:r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:
  signal.alarm(0);r.update(wall_seconds=time.monotonic()-t,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,worker_usage_on_exit=worker_usage(out.parent))
  for fd in pins.values():os.close(fd)
  r['whole_worker_monitor']=monitor.stop()
  monitor_bad=bool(r['whole_worker_monitor']['fault'] or not r['whole_worker_monitor']['thread_stopped'])
  if monitor_bad:r['status']='FAILED_MONITOR_PRESERVED'
  try:persist()
  except BaseException as exc:emergency_failure(emergency,r,'final persist failed: '+type(exc).__name__);raise
  if monitor_bad:raise RuntimeError('worker output monitor failure')

def verify_closed(a):
 p=read(a.protocol);d=sha(a.protocol);s=authorize(p,d,True);need(a.job_spec and sha(a.job_spec)==s['launch_spec_sha256'],'verification job mismatch');job=read(a.job_spec);argv=job['argv'];need(len(argv)==19 and argv[1].startswith('FFNDS_WORKER_CONFIG='),'verification argv');profile=Path(argv[1].split('=',1)[1]);need(sha(profile)==p['worker_config_sha256'],'verification profile');vdir=HF/VERIFY;out=HF/OUTPUT
 expected=['env',argv[1],'CUDA_VISIBLE_DEVICES=','taskset','-c','0',ENV,'-I','-B',str(ROOT/SELF),'verify-closed','--protocol',str(a.protocol.resolve()),'--run',str(a.run.resolve()),'--exit',str(a.exit.resolve()),'--job-spec',str(a.job_spec.resolve())]
 need(argv==expected and job['id']==ID+'_VERIFY' and job['protocol_sha256']==d and Path(job['protocol_path']).resolve()==a.protocol.resolve() and Path(job['cwd']).resolve()==ROOT and job['hf_root']==str(HF) and Path(job['output_dir']).resolve()==vdir and Path(job['run_json']).resolve()==vdir/'results/RUN.json' and job['external_timeout_seconds']==VERIFY_RESOURCES['outer_seconds'],'verification supervisor identity')
 need(sys.flags.isolated and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sorted(os.sched_getaffinity(0))==[0] and runtime()==p['runtime'],'verification runtime');import worker_identity;worker_identity.require_worker()
 need(a.run.resolve()==(out/'RUN.json').resolve() and a.exit.resolve()==(out.parent/'EXIT_RECEIPT.json').resolve(),'parent paths');r=read(a.run);e=read(a.exit);need(e.get('id')==ID and e.get('process_exited') is True and e.get('child_returncode')==0 and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('run_sha256')==sha(a.run) and e.get('protocol_sha256')==d and e.get('spec_sha256')==r.get('launch_spec_sha256') and r.get('status')=='COMPLETE_AUTHOR_CONVERTED_BF16_NOT_LLM','conversion actual EXIT0 required')
 output=r['output'];target=HF/output['path'];need(target==out/'llama-3-8b/model.pth' and output['bytes']<=VERIFY_RESOURCES['max_file_read_bytes'],'output identity');dest=vdir/'results';need(not dest.exists(),'fresh verify required');dest.mkdir(parents=True,mode=0o700);t=time.monotonic();cpu=time.process_time();v={'id':ID+'_VERIFY','protocol_sha256':d,'launch_spec_sha256':s['launch_spec_sha256'],'status':'STARTED','parent_RUN':binding(a.run,HF),'parent_exit':binding(a.exit,HF),'model_calls':0,'GPU_calls':0,'automatic_retry':False,'hash_reads':[]}
 emergency=reserve_emergency(dest)
 def persist():checked_persist(dest/'RUN.json',v,vdir,VERIFY_RESOURCES['new_HF_bytes'])
 persist();signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('verification output monitor failure')));monitor=UsageMonitor(vdir,VERIFY_RESOURCES['new_HF_bytes'],2<<30);monitor.start()
 try:
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('verification wall limit')));signal.alarm(VERIFY_RESOURCES['wall_seconds']);fd=pin_file(target)
  try:
   st=statid(os.fstat(fd));actual=tracked_hash('/proc/self/fd/'+str(fd),output['bytes'],v,persist,'checkpoint:model.pth');need(actual==output['sha256'] and statid(target.stat())==st,'output hash or identity changed')
  finally:os.close(fd)
  v['output']=output;v['status']='COMPLETE_VERIFIED_AUTHOR_CHECKPOINT_NOT_LLM'
 except BaseException as e:v.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:
  signal.alarm(0);v.update(wall_seconds=time.monotonic()-t,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,worker_usage_on_exit=worker_usage(vdir));v['whole_worker_monitor']=monitor.stop()
  monitor_bad=bool(v['whole_worker_monitor']['fault'] or not v['whole_worker_monitor']['thread_stopped'])
  if monitor_bad:v['status']='FAILED_MONITOR_PRESERVED'
  try:persist()
  except BaseException as exc:emergency_failure(emergency,v,'verification final persist failed: '+type(exc).__name__);raise
  if monitor_bad:raise RuntimeError('verification output monitor failure')

def inspect_closed(a):
 p=read(a.protocol);validate(p,True);vpath=HF/VERIFY/'results/RUN.json';epath=HF/VERIFY/'EXIT_RECEIPT.json';v=read(vpath);e=read(epath);need(v['status']=='COMPLETE_VERIFIED_AUTHOR_CHECKPOINT_NOT_LLM' and e['id']==ID+'_VERIFY' and e['process_exited'] is True and e['child_returncode']==0 and e['timeout'] is False and e['automatic_retry'] is False and e['protocol_sha256']==sha(a.protocol) and e['run_sha256']==sha(vpath) and e['spec_sha256']==v['launch_spec_sha256'] and v['wall_seconds']<=VERIFY_RESOURCES['wall_seconds'],'verification actual EXIT0 required');return {'id':ID,'status':'PASS_CLOSED_AUTHOR_CONVERSION_NOT_LLM','verification_RUN':binding(vpath,HF),'verification_exit':binding(epath,HF),'output':v['output']}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','inspect','run','verify-closed','inspect-closed'));ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--job-spec',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path);ap.add_argument('--asset-run',type=Path);ap.add_argument('--asset-exit',type=Path);ap.add_argument('--asset-verify-run',type=Path);ap.add_argument('--asset-verify-exit',type=Path);a=ap.parse_args()
 if a.action=='draft':
  need(a.out and not a.out.exists() and all([a.asset_run,a.asset_exit,a.asset_verify_run,a.asset_verify_exit]),'fresh draft and actual parent receipts required');p=draft(a.asset_run,a.asset_exit,a.asset_verify_run,a.asset_verify_exit);a.out.parent.mkdir(parents=True,exist_ok=True);write(a.out,p)
 elif a.action=='inspect':p=read(a.protocol);validate(p);print(json.dumps({'id':ID,'fingerprint':fingerprint(p),'resources':RESOURCES,'limits':LIMITS},indent=2))
 elif a.action=='run':run(a)
 elif a.action=='verify-closed':verify_closed(a)
 else:print(json.dumps(inspect_closed(a),indent=2))
if __name__=='__main__':main()
