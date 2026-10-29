"""Single reviewed anonymous, pinned Llama3 asset acquisition; no model execution."""
import argparse, fcntl, hashlib, importlib.metadata, json, os, resource, shutil, signal, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'tools'))
import e0_teal_llama3_tokenizer_byte_bridge as base
HF=Path('/home/xiang/.cache/huggingface'); ID='E0_TEAL_LLAMA3_PUBLIC_WEIGHT_ACQUISITION_20261029'
SELF='tools/e0_teal_llama3_public_weight_acquisition.py'; TEST='tests/test_e0_teal_llama3_public_weight_acquisition.py'
REV='315b20096dc791d381d514deb5f8bd9c8d6d3061'; REPO='NousResearch/Meta-Llama-3-8B'
OUTPUT='ffnds-derived/e0_teal_llama3_public_weight_acquisition_20261029/attempt1/worker/results'
CACHE='ffnds-derived/e0_teal_llama3_public_weight_acquisition_20261029/attempt1/worker/scoped_hf_cache'
VERIFY='ffnds-derived/e0_teal_llama3_public_weight_acquisition_20261029/attempt1/verification'
VERIFY_RESOURCES={'CPU_threads':1,'GPU':0,'wall_seconds':3600,'outer_seconds':3720,'max_asset_read_bytes':16069766580,'new_HF_bytes':67108864,'automatic_retry':False}
VERIFY_LIMITS={'asset_files':13,'hash_read_bytes':16069766580,'model_calls':0,'GPU_calls':0,'network_calls':0}
TOKEN='ffnds-derived/e0_teal_llama3_tokenizer_byte_bridge_20261029/attempt1/worker/results/tokenizer.model'
TOKEN_SHA='82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55'
BRIDGE_RUN='ffnds-derived/e0_teal_llama3_tokenizer_byte_bridge_20261029/attempt1/worker/results/RUN.json'
BRIDGE_EXIT='ffnds-derived/e0_teal_llama3_tokenizer_byte_bridge_20261029/attempt1/worker/EXIT_RECEIPT.json'
META='research/literature/20261029/P0_TEAL_PUBLIC_REDISTRIBUTION_METADATA_20261029.json'
HF_SOURCES=('ffnds-research-sources/p0_teal_full_llm_20261029/redistribution_metadata/nous3_api.json','ffnds-research-sources/p0_teal_full_llm_20261029/redistribution_metadata/official3_api.json','ffnds-research-sources/p0_teal_full_llm_20261029/redistribution_metadata/nous3_LICENSE','ffnds-research-sources/p0_teal_full_llm_20261029/redistribution_metadata/nous3_README.md')
SOURCES=(SELF,TEST,'tools/e0_teal_llama3_tokenizer_byte_bridge.py','tools/worker_identity.py','tools/durable_worker_job.py',META,'research/method_discussion/discoveries/E0/p0-teal-author-full-llm-feasibility-20261029.md')
NOTICES={'README.md':(36589,'485890de16fe610f2718992ce930e0ca96f4a997'),'USE_POLICY.md':(4696,'6f8e5f6e3243c92046a133c92eb62b1ed7975a0b')}
RESOURCES={'GPU':0,'CPU_threads':1,'wall_seconds':7200,'outer_seconds':7320,'peak_address_space_bytes':4294967296,'peak_RSS_bytes':4294967296,'minimum_free_disk_bytes':25769803776,'new_HF_bytes':17179869184,'max_logical_hash_bytes':48211483722,'automatic_retry':False}
LIMITS={'asset_files':13,'hub_api_calls_max':13,'model_calls':0,'GPU_calls':0,'torch_imports':0,'checkpoint_conversion':0}
PACKAGES=('huggingface-hub','httpx','httpcore','certifi','anyio','h11','idna','filelock','fsspec','packaging','pyyaml','tqdm','typing-extensions','click','typer','hf-xet','rich','shellingham')
need=base.need; read=base.read; sha=base.sha; write=base.write; binding=base.binding; bound=base.bound

def manifest():
 d=read(ROOT/META)['results']['llama3']; need(d['public_repo']==REPO and d['public_revision']==REV,'metadata identity')
 names=['LICENSE','README.md','USE_POLICY.md','config.json','generation_config.json','model.safetensors.index.json','model-00001-of-00004.safetensors','model-00002-of-00004.safetensors','model-00003-of-00004.safetensors','model-00004-of-00004.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json']
 rows={x['file']:x for x in d['files']}; out=[]
 for name in names:
  if name in NOTICES:
   size,git=NOTICES[name];out.append({'filename':name,'bytes':size,'git_blob':git,'sha256':None,'official_match':'NOT_ASSERTED'});continue
  x=rows[name]; a=x['official'];b=x['public'];need(a==b and a is not None,'metadata differs');out.append({'filename':name,'bytes':b['bytes'],'git_blob':b['git_blob'],'sha256':b['lfs_sha256'],'official_match':'PUBLIC_METADATA_MATCH'})
 return out

def runtime():
 rows=[]
 for pkg in PACKAGES:
  d=importlib.metadata.distribution(pkg); records=[]
  for f in d.files or []:
   if str(f).endswith(('.py','.so','.json','.pem')):
    p=Path(d.locate_file(f)).resolve()
    if p.is_file(): records.append([str(f),p.stat().st_size,sha(p)])
  records.sort(); need(records,'empty package closure');rows.append({'name':pkg,'version':d.version,'files_sha256':hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest(),'files_count':len(records)})
 return {'python':str(Path(sys.executable).resolve()),'python_sha256':sha(Path(sys.executable).resolve()),'version':sys.version.split()[0],'prefix':sys.prefix,'packages':rows}

def draft():
 return {'id':ID,'status':'DRAFT_REVIEW_REQUESTED','repo':REPO,'revision':REV,'files':manifest(),'resources':RESOURCES,'limits':LIMITS,'output_HF_relative':OUTPUT,'cache_HF_relative':CACHE,'verification_resources':VERIFY_RESOURCES,'verification_limits':VERIFY_LIMITS,'verification_HF_relative':VERIFY,'tokenizer':{'path':TOKEN,'bytes':2183982,'sha256':TOKEN_SHA},'hf_source_bindings':[binding(HF/x,HF) for x in HF_SOURCES],'bridge_evidence':[binding(HF/BRIDGE_RUN,HF),binding(HF/BRIDGE_EXIT,HF)],'source_bindings':[binding(ROOT/x,ROOT) for x in SOURCES],'runtime':runtime(),'worker_config_sha256':None,'review':None,'automatic_retry':False}
def fingerprint(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate(p,frozen=False):
 need(p['id']==ID and p['repo']==REPO and p['revision']==REV and p['files']==manifest() and p['resources']==RESOURCES and p['limits']==LIMITS and p['output_HF_relative']==OUTPUT and p['cache_HF_relative']==CACHE and p['verification_resources']==VERIFY_RESOURCES and p['verification_limits']==VERIFY_LIMITS and p['verification_HF_relative']==VERIFY and p['automatic_retry'] is False,'scope differs')
 need(p['tokenizer']=={'path':TOKEN,'bytes':2183982,'sha256':TOKEN_SHA} and {x['path'] for x in p['source_bindings']}==set(SOURCES) and {x['path'] for x in p['hf_source_bindings']}==set(HF_SOURCES),'closure differs')
 if frozen: need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['review'] is None and isinstance(p['worker_config_sha256'],str) and len(p['worker_config_sha256'])==64,'freeze differs')
def file_size_limit(item):return max(item['bytes'],16<<20)
def statid(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
def usage(path):
 total=0;files=0
 for parent,dirs,names in os.walk(path,followlinks=False):
  for name in names:
   p=Path(parent)/name
   if not p.is_symlink():total+=p.stat().st_size;files+=1
 return {'apparent_bytes':total,'regular_files':files}
def clone_fresh(src,dst):
 fd=os.open(dst,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 try:
  with open(src,'rb') as f:fcntl.ioctl(fd,0x40049409,f.fileno())
  return True
 except OSError:
  os.close(fd);fd=None;os.unlink(dst);return False
 finally:
  if fd is not None:os.close(fd)
def call_backend(item,items,backend,**kwargs):
 maximum=max(file_size_limit(x) for x in items);oldsoft,oldhard=resource.getrlimit(resource.RLIMIT_FSIZE);need(oldhard==resource.RLIM_INFINITY or oldhard>=maximum,'file limit unavailable')
 resource.setrlimit(resource.RLIMIT_FSIZE,(file_size_limit(item),oldhard))
 try:return backend(**kwargs)
 finally:resource.setrlimit(resource.RLIMIT_FSIZE,(oldsoft,oldhard))
def verify_file(path,item,progress=lambda n:None):
 path=Path(path); need(path.is_file() and not path.is_symlink(),'regular file required'); h=hashlib.sha256(); g=hashlib.sha1(b'blob '+str(item['bytes']).encode()+b'\0'); count=0
 with path.open('rb') as f:
  st=os.fstat(f.fileno());need(st.st_size==item['bytes'],'size differs')
  while True:
   block=f.read(1<<20)
   if not block:break
   h.update(block);g.update(block);count+=len(block);progress(count)
  need(statid(os.fstat(f.fileno()))==statid(st) and statid(path.stat())==statid(st),'asset changed during hash')
 need(count==item['bytes'] and (h.hexdigest()==item['sha256'] if item['sha256'] else g.hexdigest()==item['git_blob']),'hash differs')
 return {'bytes':count,'sha256':h.hexdigest(),'git_blob':g.hexdigest()}

def select_scope(a,p,digest,verification=False):
 scope_id=ID+'_VERIFY' if verification else ID;resources=VERIFY_RESOURCES if verification else RESOURCES;limits=VERIFY_LIMITS if verification else LIMITS
 sc=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==scope_id]
 need(a.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True and len(sc)==1,'ACTIVE scope required');s=sc[0]
 need(s.get('protocol_sha256')==digest and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('resources')==resources and s.get('limits')==limits and s.get('max_gpu_cards')==0,'ACTIVE mismatch')
 return scope_id,s

def authorize(p,digest,verification=False):
 validate(p,True);a=read(ROOT/'research/ACTIVE_RESEARCH.json');scope_id,s=select_scope(a,p,digest,verification)
 rev=read(bound(ROOT,s['review']));need(rev.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and rev.get('scope_id')==scope_id and rev.get('state')=='REVIEWER_RECHECK_APPROVED' and rev.get('recheck_disposition')=='APPROVE' and rev.get('reviewer_disposition') in ('APPROVE','REVISE') and rev.get('author_agent_id') and rev.get('reviewer_agent_id') and rev['author_agent_id']!=rev['reviewer_agent_id'] and rev.get('responses_to_review') and rev.get('adversarial_counterexample') and rev.get('approval_scope_fingerprint')==fingerprint(p),'review mismatch')
 need(rev.get('critical_source_hashes')=={b['path']:b['sha256'] for b in p['source_bindings']},'review source mismatch')
 for k in ('author_review','independent_review','reviewer_recheck'):bound(ROOT,rev[k])
 return s

def validate_parent_receipt(r,e,d,run_sha):
 need(e.get('id')==ID and e.get('child_returncode')==0 and e.get('process_exited') is True and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('run_sha256')==run_sha and e.get('protocol_sha256')==d and e.get('spec_sha256')==r.get('launch_spec_sha256') and r.get('status')=='COMPLETE_PUBLIC_BYTES_NOT_MODEL_EXECUTED','actual acquisition EXIT0 required')

def run(a):
 need(not {'torch','triton','transformers'} & set(sys.modules),'numerical library imported');p=read(a.protocol);d=sha(a.protocol);scope=authorize(p,d)
 need(a.job_spec and sha(a.job_spec)==scope['launch_spec_sha256'],'job spec');job=read(a.job_spec);argv=job['argv'];need(len(argv)==15 and argv[1].startswith('FFNDS_WORKER_CONFIG='),'job argv')
 profile=Path(argv[1].split('=',1)[1]);need(sha(profile)==p['worker_config_sha256'],'profile mismatch')
 expected=['env',argv[1],'CUDA_VISIBLE_DEVICES=','taskset','-c','0',str(ROOT/'.venv/bin/python'),'-I','-B',str(ROOT/SELF),'run','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())]
 out=HF/OUTPUT;need(argv==expected and job['id']==ID and job['protocol_sha256']==d and Path(job['protocol_path']).resolve()==a.protocol.resolve() and Path(job['cwd']).resolve()==ROOT and job['hf_root']==str(HF) and Path(job['output_dir']).resolve()==out.parent and Path(job['run_json']).resolve()==out/'RUN.json' and job['external_timeout_seconds']==RESOURCES['outer_seconds'],'job identity')
 need(sys.flags.isolated and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sorted(os.sched_getaffinity(0))==[0] and runtime()==p['runtime'],'environment differs')
 for b in p['source_bindings']:bound(ROOT,b)
 for b in p['hf_source_bindings']:bound(HF,b)
 need(len(p['bridge_evidence'])==2 and {b['path'] for b in p['bridge_evidence']}=={BRIDGE_RUN,BRIDGE_EXIT},'bridge proof identity')
 for b in p['bridge_evidence']:bound(HF,b)
 br=read(HF/BRIDGE_RUN);be=read(HF/BRIDGE_EXIT);need(br['status']=='COMPLETE_BYTE_IDENTICAL_NOT_TOKENIZER_RUNTIME' and br['output']['sha256']==TOKEN_SHA and be['child_returncode']==0 and be['run_sha256']==sha(HF/BRIDGE_RUN),'bridge actual exit')
 import worker_identity;worker_identity.require_worker()
 cache_root=HF/CACHE;need(not out.exists() and not cache_root.exists(),'fresh attempt required');out.mkdir(parents=True,mode=0o700);cache_root.mkdir(mode=0o700);start=time.monotonic();cpu=time.process_time()
 r={'id':ID,'protocol_sha256':d,'launch_spec_sha256':scope['launch_spec_sha256'],'status':'STARTED','automatic_retry':False,'model_calls':0,'gpu_calls':0,'hub_api_calls':0,'files':[],'hash_reads':[],'max_payload_bytes':sum(x['bytes'] for x in p['files']),'write_budget_semantics':'fresh exclusive scope cache, single sequential blob partial capped to max(expected size,16 MiB); aggregate limits plus 64 MiB reserve, cache scanned between calls and on exit; backend wire bytes unknown','actual_network_bytes':'UNKNOWN_BACKEND_NOT_EXPOSED','network_retry':'hub HEAD backoff and GET stream retries; 5 consecutive no-progress retries, reset after any progress, overall wall bounded','transport':'http; XET disabled; token=False','output_HF_relative':OUTPUT}
 def persist():write(out/'RUN.json',r)
 def checked(path,item):
  need(sum(x['predebited_bytes'] for x in r['hash_reads'])+item['bytes']<=RESOURCES['max_logical_hash_bytes'],'hash read budget');row={'filename':item['filename'],'predebited_bytes':item['bytes'],'completed_bytes':0,'progress_bytes':0,'unrecorded_read_bytes_max':(64<<20)+(1<<20),'status':'INFLIGHT'};r['hash_reads'].append(row);persist();mark=[64<<20]
  def prog(n):
   if n>=mark[0]:row['progress_bytes']=n;persist();mark[0]=n+(64<<20)
  val=verify_file(path,item,prog);row.update(completed_bytes=val['bytes'],progress_bytes=val['bytes'],unrecorded_read_bytes_max=0,status='COMPLETE');persist();return val
 persist()
 try:
  need(sum(file_size_limit(x) for x in p['files'])+(64<<20)<=RESOURCES['new_HF_bytes'],'aggregate file limit exceeds budget');need(shutil.disk_usage(HF).free>=RESOURCES['minimum_free_disk_bytes'],'insufficient disk');resource.setrlimit(resource.RLIMIT_AS,(RESOURCES['peak_address_space_bytes'],)*2)
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('scope wall limit')));signal.alarm(RESOURCES['wall_seconds'])
  tb={'filename':'tokenizer.model','bytes':2183982,'sha256':TOKEN_SHA,'git_blob':None};checked(HF/TOKEN,tb)
  for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):
   os.environ.pop(key,None)
  os.environ.update(HF_ENDPOINT='https://huggingface.co',HF_HUB_OFFLINE='0',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  from huggingface_hub import hf_hub_download
  need('torch' not in sys.modules,'torch imported');cache=cache_root/'models--NousResearch--Meta-Llama-3-8B';blobs=cache/'blobs';blobs.mkdir(parents=True,mode=0o700);shared=HF/'hub'/'models--NousResearch--Meta-Llama-3-8B'/'blobs'
  for item in p['files']:
   # Cache is exclusively created for this attempt. Reflink (never hardlink) may reuse a verified standard-cache blob.
   key=item['sha256'] or item['git_blob'];blob=blobs/key;source=shared/key;existing=source.exists();need(not blob.exists() and not blob.is_symlink(),'private destination collision')
   row={'filename':item['filename'],'bytes':item['bytes'],'reused':False,'status':'PREDEBITED','download_payload_max':item['bytes']};r['files'].append(row);persist()
   if existing:
    checked(source,item);row['clone_attempt_predebited_bytes']=item['bytes'];persist()
    if clone_fresh(source,blob):
     checked(blob,item);row['reused']=True;row['download_payload_max']=0;row['clone_status']='COMPLETE_INDEPENDENT_REFLINK'
    else:row['clone_status']='UNSUPPORTED_FALL_BACK_TO_DOWNLOAD'
    persist()
   need(shutil.disk_usage(HF).free>=RESOURCES['minimum_free_disk_bytes']-sum(f['bytes'] for f in r['files'] if not f['reused']),'disk reserve')
   row['cache_usage_before']=usage(cache_root);need(usage(out.parent)['apparent_bytes']<=RESOURCES['new_HF_bytes']-(64<<20),'whole worker budget');r['hub_api_calls']+=1;persist()
   got=Path(call_backend(item,p['files'],hf_hub_download,repo_id=REPO,filename=item['filename'],revision=REV,cache_dir=str(cache_root),token=False,force_download=False,local_files_only=False)).resolve()
   need(got==blob.resolve(),'unexpected cache blob');result=checked(got,item);row.update(result=result,status='COMPLETE',cache_path=str(got.relative_to(HF)),cache_usage_after=usage(cache_root));need(row['cache_usage_after']['apparent_bytes']<=RESOURCES['new_HF_bytes']-(64<<20),'cache budget');persist()
  r['status']='COMPLETE_PUBLIC_BYTES_NOT_MODEL_EXECUTED'
 except BaseException as e:
  r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=(str(e)[:500] if isinstance(e,(ValueError,TimeoutError)) else 'BACKEND_ERROR_DETAILS_REDACTED'));raise
 finally:
  signal.alarm(0);r.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cache_usage_on_exit=usage(cache_root),worker_usage_before_final_RUN=usage(out.parent));
  if r['worker_usage_before_final_RUN']['apparent_bytes']>RESOURCES['new_HF_bytes']-(64<<20):r['status']='FAILED_WORKER_BUDGET_PRESERVED'
  persist()

def verify_closed(a):
 p=read(a.protocol);d=sha(a.protocol);scope=authorize(p,d,True);out=HF/OUTPUT;verify=HF/VERIFY;need(a.job_spec and sha(a.job_spec)==scope.get('launch_spec_sha256'),'verification spec required');job=read(a.job_spec);argv=job['argv'];need(len(argv)==19 and argv[1].startswith('FFNDS_WORKER_CONFIG='),'verification argv')
 profile=Path(argv[1].split('=',1)[1]);need(sha(profile)==p['worker_config_sha256'],'verification worker profile');expected=['env',argv[1],'CUDA_VISIBLE_DEVICES=','taskset','-c','0',str(ROOT/'.venv/bin/python'),'-I','-B',str(ROOT/SELF),'verify-closed','--protocol',str(a.protocol.resolve()),'--run',str(a.run.resolve()),'--exit',str(a.exit.resolve()),'--job-spec',str(a.job_spec.resolve())]
 need(argv==expected and job['id']==ID+'_VERIFY' and job['protocol_sha256']==d and Path(job['protocol_path']).resolve()==a.protocol.resolve() and Path(job['cwd']).resolve()==ROOT and job['hf_root']==str(HF) and Path(job['output_dir']).resolve()==verify and Path(job['run_json']).resolve()==verify/'results/RUN.json' and job['external_timeout_seconds']==VERIFY_RESOURCES['outer_seconds'],'verification job identity')
 need(sys.flags.isolated and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sorted(os.sched_getaffinity(0))==[0] and runtime()==p['runtime'],'verification runtime')
 for b in p['source_bindings']:bound(ROOT,b)
 for b in p['hf_source_bindings']:bound(HF,b)
 import worker_identity;worker_identity.require_worker()
 need(a.run.resolve()==(out/'RUN.json').resolve() and a.exit.resolve()==(out.parent/'EXIT_RECEIPT.json').resolve(),'receipt paths');r=read(a.run);e=read(a.exit)
 validate_parent_receipt(r,e,d,sha(a.run))
 need(len(r['files'])==LIMITS['asset_files'] and r['hub_api_calls']==LIMITS['asset_files'] and r['model_calls']==r['gpu_calls']==0 and not r['automatic_retry'] and r['wall_seconds']<=RESOURCES['wall_seconds'] and r['peak_RSS_bytes']<=RESOURCES['peak_RSS_bytes'] and all(x['status']=='COMPLETE' for x in r['files']) and all(x['status']=='COMPLETE' and x['completed_bytes']==x['predebited_bytes'] for x in r['hash_reads']),'acquisition ledger incomplete')
 target=verify/'results';need(not target.exists(),'fresh verification required');target.mkdir(parents=True,mode=0o700);t=time.monotonic();cpu=time.process_time();v={'id':ID+'_VERIFY','protocol_sha256':d,'launch_spec_sha256':scope['launch_spec_sha256'],'parent_RUN':binding(a.run,HF),'parent_exit':binding(a.exit,HF),'status':'STARTED','predebited_bytes':0,'completed_bytes':0,'hash_reads':[],'model_calls':0,'GPU_calls':0,'automatic_retry':False}
 def persist():write(target/'RUN.json',v)
 persist()
 try:
  resource.setrlimit(resource.RLIMIT_AS,(RESOURCES['peak_address_space_bytes'],)*2);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('verification wall limit')));signal.alarm(VERIFY_RESOURCES['wall_seconds'])
  for item,row in zip(p['files'],r['files']):
   need(item['filename']==row['filename'],'file order');path=HF/row['cache_path'];need(path.is_relative_to(HF/CACHE),'cache path outside scope')
   h={'filename':item['filename'],'predebited_bytes':item['bytes'],'progress_bytes':0,'completed_bytes':0,'unrecorded_read_bytes_max':(64<<20)+(1<<20),'status':'INFLIGHT'};v['hash_reads'].append(h);v['predebited_bytes']+=item['bytes'];need(v['predebited_bytes']<=VERIFY_RESOURCES['max_asset_read_bytes'],'verification read budget');persist();mark=[64<<20]
   def progress(n):
    if n>=mark[0]:h['progress_bytes']=n;persist();mark[0]=n+(64<<20)
   actual=verify_file(path,item,progress);need(actual==row['result'],'previous result differs');h.update(status='COMPLETE',completed_bytes=item['bytes'],progress_bytes=item['bytes'],unrecorded_read_bytes_max=0);v['completed_bytes']+=item['bytes'];persist()
  need(v['completed_bytes']==v['predebited_bytes']==VERIFY_RESOURCES['max_asset_read_bytes'],'verification incomplete');v['status']='COMPLETE_VERIFIED_PUBLIC_BYTES'
 except BaseException as ex:v.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(ex).__name__,error=str(ex)[:500] if isinstance(ex,(ValueError,TimeoutError)) else 'REDACTED');raise
 finally:
  signal.alarm(0);v.update(wall_seconds=time.monotonic()-t,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,worker_usage_before_final_RUN=usage(verify))
  if v['worker_usage_before_final_RUN']['apparent_bytes']>VERIFY_RESOURCES['new_HF_bytes']-(1<<20):v['status']='FAILED_VERIFICATION_OUTPUT_BUDGET'
  persist()

def inspect_closed(a):
 p=read(a.protocol);validate(p,True);vpath=HF/VERIFY/'results/RUN.json';epath=HF/VERIFY/'EXIT_RECEIPT.json';v=read(vpath);e=read(epath)
 need(v['status']=='COMPLETE_VERIFIED_PUBLIC_BYTES' and e['id']==ID+'_VERIFY' and e['process_exited'] and e['child_returncode']==0 and not e['timeout'] and e['protocol_sha256']==sha(a.protocol) and e['run_sha256']==sha(vpath) and e['spec_sha256']==v['launch_spec_sha256'] and v['completed_bytes']==VERIFY_RESOURCES['max_asset_read_bytes'] and v['wall_seconds']<=VERIFY_RESOURCES['wall_seconds'],'verification EXIT0 required')
 return {'id':ID,'status':'PASS_CLOSED_PUBLIC_ASSETS_NOT_MODEL_EXECUTED','verification_RUN':binding(vpath,HF),'verification_exit':binding(epath,HF),'asset_read_bytes':v['completed_bytes']}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','inspect','run','verify-closed','inspect-closed'));ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path);ap.add_argument('--job-spec',type=Path);a=ap.parse_args()
 if a.action=='draft':need(a.out and not a.out.exists(),'fresh draft');write(a.out,draft())
 elif a.action=='inspect':p=read(a.protocol);validate(p);print(json.dumps({'fingerprint':fingerprint(p),'files':p['files'],'resources':RESOURCES},indent=2))
 elif a.action=='run':run(a)
 elif a.action=='verify-closed':verify_closed(a)
 else:print(json.dumps(inspect_closed(a),indent=2))
if __name__=='__main__':main()
