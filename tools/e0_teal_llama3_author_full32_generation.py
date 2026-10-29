"""Finite author gpt-fast 32-layer generation. Requires actual reviewed parents.

Drafts without 64 actually closed calibration histograms cannot be frozen/run.
No numerical library is imported by draft, inspect, preflight, or this import.
"""
import argparse, hashlib, json, math, os, resource, signal, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_teal_llama3_tokenizer_byte_bridge as base
import e0_teal_llama3_author_checkpoint_convert as envutil
HF=Path('/home/xiang/.cache/huggingface'); ID='E0_TEAL_LLAMA3_AUTHOR_FULL32_GENERATION_20261030'
SELF='tools/e0_teal_llama3_author_full32_generation.py';TEST='tests/test_e0_teal_llama3_author_full32_generation.py'
ENV='/home/xiang/.cache/huggingface/ffnds-derived/teal_author_llm_env_20261029/venv/bin/python'
OUT='ffnds-derived/e0_teal_llama3_author_full32_generation_20261030/attempt1/worker/results'
CPU='ffnds-derived/e0_teal_llama3_author_full32_generation_20261030/attempt1/cpu_preflight/worker/results'
REPORT='research/20261029/TEAL_LLAMA3_AUTHOR_CHECKPOINT_CONVERSION_RESULTS_20261029.json'
HROOT='ffnds-derived/e0_teal_llama3_author_grab_acts_v2_20261030/attempt1/worker/results'
HIST_ID='E0_TEAL_LLAMA3_AUTHOR_GRAB_ACTS_V2_20261030'
SRC='ffnds-research-sources/p0_teal_full_llm_20261029';SUP='ffnds-research-sources/p0_teal_gptfast_support_20261030'
AUTHOR_FILES=[SRC+'/gpt-fast/'+x for x in ('generate.py','model.py','quantize.py','tokenizer.py')]+[SRC+'/utils/__init__.py',SRC+'/utils/utils.py',SRC+'/kernels/sparse_gemv.py',SUP+'/distribution.py',SUP+'/tp.py']
KERNEL='.runtime/teal_fb7373c93ac3594817c9ee64d4e08b47430a1822/kernels/compile_wrapper.py'
SOURCES=(SELF,TEST,'tools/e0_teal_llama3_tokenizer_byte_bridge.py','tools/e0_teal_llama3_author_checkpoint_convert.py','tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py','tools/durable_worker_job.py','scripts/worker_teal_gpu0_uuid_exec.sh',KERNEL,REPORT,'research/method_discussion/discoveries/E0/p0-teal-llama3-author-full32-execution-design-20261029.md','research/literature/20261029/P0_TEAL_FULL_LLM_STATIC_SOURCE_MANIFEST_20261029.json')
MODEL_CLOSED='.runtime/teal_llama3_public_weight_acquisition_20261029/CLOSED_VERIFICATION.json'
SNAPSHOT=HF/'ffnds-derived/e0_teal_llama3_public_weight_acquisition_20261029/attempt1/worker/scoped_hf_cache/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061'
ALPACA_SHA='06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073'
MODEL_SHA='d420e39d07864fd3b1ee8006e2b064b38370825bb369191e0c363affa1196887';MODEL_SIZE=16060590926
TOKEN_SHA='82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55'
PROMPT='The history of science is full of examples in which careful observation changed how people understood the world.'
DESIGN={'arms':['dense','truezero','uniform50'],'sequence':['dense','truezero','uniform50','dense','truezero','uniform50','uniform50','truezero','dense'],'warmup_per_arm':1,'measured_per_arm':2,'max_new_tokens':64,'seed':1234,'top_k':200,'temperature':0.8,'prompt':PROMPT,'compile':'AUTHOR_TORCH_COMPILE_DECODE_ONLY_FULLGRAPH_REDUCE_OVERHEAD','neutral_forced_decode':4,'neutral_rel_l2_max':0.03,'neutral_require_top1_match':True,'neutral_finite_required':True,'neutral_scope':'EAGER_LOGITS_PREFILL_DEFAULT_SDPA_DECODE_MATH_ONLY_NOT_COMPILED','cache_scope':'ALL_ARMS_IDENTICAL_PREALLOCATED_T_PLUS_68_ROUNDED_8','eos':'NO_EARLY_STOP_64_FIXED_RECORD_POSITIONS','observer':'SEPARATE_AFTER_TIMED_RUNS_ONE_FORCED_DECODE_PER_PATCHED_ARM','source_scope':'AUTHOR_ALL_ATTENTION_AND_FFN_NOT_FFN_ONLY','physical_launches':'UNKNOWN_COMPILE_AUTOTUNE_GRAPH_REPLAY','intervention':'truezero forces all seven thresholds to exact 0 after author patch'}
RESOURCES={'GPU':1,'CPU_threads':2,'wall_seconds':7200,'outer_seconds':7320,'address_space_max_bytes':103079215104,'GPU_reserved_max_bytes':81604378624,'minimum_free_disk_bytes':10737418240,'new_HF_bytes':4294967296,'output_file_max_bytes':2147483648,'automatic_retry':False}
CPU_RESOURCES={'GPU':0,'CPU_threads':1,'wall_seconds':900,'outer_seconds':1020,'new_HF_bytes':16777216,'max_hist_hash_bytes':268435456,'tokenizer_hash_bytes':2183982,'other_input_reads':'source/receipt/package metadata; bytes UNKNOWN; hist bound is not total I/O','automatic_retry':False}
LIMITS={'model_loads':3,'cache_preparations':3,'checkpoint_hash_passes':2,'histogram_hash_passes':1,'histogram_file_loads':256,'compile_calls':3,'generations':9,'generation_prefills':9,'generation_decode_calls':567,'teacher_forced_model_calls':10,'observer_model_calls':4,'observer_mask_counts':448,'observer_zero_counts':448,'GPU':1,'training_updates':0,'automatic_retry':False}
need=base.need;read=base.read;sha=base.sha;write=base.write;binding=base.binding;bound=base.bound

def hist_keyset():return {f'author/histograms/layer-{i}/{group}/histograms.pt' for i in range(32) for group in ('mlp','self_attn')}
def check_hist_parent(run,receipt,run_sha):
 need(run.get('id')==HIST_ID and run.get('status')=='COMPLETE_AUTHOR_CALIBRATION_EXIT_PENDING' and receipt.get('id')==run.get('id') and receipt.get('process_exited') is True and receipt.get('child_returncode')==0 and receipt.get('timeout') is False and receipt.get('automatic_retry') is False and receipt.get('run_sha256')==run_sha and receipt.get('protocol_sha256')==run.get('protocol_sha256') and receipt.get('spec_sha256')==run.get('launch_spec_sha256'),'hist actual EXIT0 required')
 need(run.get('model_snapshot')==str(SNAPSHOT) and run.get('model_closed_sha256')==sha(ROOT/MODEL_CLOSED) and run.get('alpaca_revision')=='dce01c9b08f87459cf36a430d809084718273017' and run.get('alpaca_sha256')==ALPACA_SHA,'calibration source lineage differs');need(run.get('histogram_validation_completed_files')==64 and run.get('author_tokenization_verified') is True and run.get('author_segments')==run.get('segments') and len(run.get('layer_events',[]))==64 and run.get('finite_observer',{}).get('completed_elements')==5368709120 and not run.get('monitor',{}).get('fault') and run.get('monitor',{}).get('stopped') is True,'hist validation incomplete');counts=run.get('counts',{});need(all(counts.get(k)==v for k,v in {'sort':128,'histogram':128,'layer_inputs':32,'histogram_files':64,'layer_forward_pre':32,'layer_forward_post':32,'activation_finite_reductions':64}.items()),'hist ledger incomplete')
 rows=run.get('histogram_files');need(isinstance(rows,list) and len(rows)==64 and {r.get('path') for r in rows}==hist_keyset(),'hist identity mismatch');need(all(type(r.get('bytes')) is int and 0<r['bytes']<=4<<20 and isinstance(r.get('sha256'),str) and len(r['sha256'])==64 for r in rows) and sum(r['bytes'] for r in rows)<=CPU_RESOURCES['max_hist_hash_bytes'],'hist byte limit');return rows

def parent_model():
 rep=read(ROOT/REPORT);need(rep.get('status')=='PASS_CLOSED_AUTHOR_CONVERSION_NOT_LLM' and rep.get('output',{}).get('sha256')==MODEL_SHA and rep['output'].get('bytes')==MODEL_SIZE,'converted model parent mismatch')
 for key in ('conversion_RUN','conversion_exit','verification_RUN','verification_exit'):bound(HF,rep[key])
 r=read(HF/rep['conversion_RUN']['path']);e=read(HF/rep['conversion_exit']['path']);v=read(HF/rep['verification_RUN']['path']);ve=read(HF/rep['verification_exit']['path'])
 need(e.get('child_returncode')==0 and e.get('process_exited') is True and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('protocol_sha256')==r.get('protocol_sha256') and e.get('spec_sha256')==r.get('launch_spec_sha256') and e.get('run_sha256')==rep['conversion_RUN']['sha256'] and r.get('status')=='COMPLETE_AUTHOR_CONVERTED_BF16_NOT_LLM' and ve.get('child_returncode')==0 and ve.get('process_exited') is True and ve.get('timeout') is False and ve.get('automatic_retry') is False and ve.get('protocol_sha256')==v.get('protocol_sha256') and ve.get('spec_sha256')==v.get('launch_spec_sha256') and ve.get('run_sha256')==rep['verification_RUN']['sha256'] and v.get('status')=='COMPLETE_VERIFIED_AUTHOR_CHECKPOINT_NOT_LLM' and v.get('output')==rep['output'],'parent receipts mismatch')
 return rep

def draft(hist_run=None,hist_exit=None):
 parent=parent_model();sources=[binding(ROOT/x,ROOT) for x in SOURCES];author=[binding(HF/x,HF) for x in AUTHOR_FILES];need(sha(ROOT/KERNEL)=='7e1b86358171923d3ac658476b483c5b1dc199601b84976264d5814933239d7c','kernel helper mismatch')
 manifest=read(ROOT/'research/literature/20261029/P0_TEAL_FULL_LLM_STATIC_SOURCE_MANIFEST_20261029.json');lookup={x['path']:x['sha256'] for x in manifest['files']}
 for x in author:
  if x['path']==SRC+'/utils/__init__.py':need(x['sha256']==hashlib.sha256(b'').hexdigest(),'author package init mismatch')
  elif x['path'].startswith(SRC+'/'):need(x['sha256']==lookup[x['path'][len(SRC)+1:]],'author source mismatch')
  else:need(x['sha256']=={'distribution.py':'94100f32155358451b1b121b0340f94b5d19c271dd8e8e41e038d3bcc7c89a16','tp.py':'d1d88eb60d2b985341ea64bf92c55a4d689b379decbf4baaedc6d787ddfc3be4'}[Path(x['path']).name],'support source mismatch')
 h=None
 if hist_run is not None or hist_exit is not None:
  need(hist_run and hist_exit,'both hist receipts required');hr=read(hist_run);he=read(hist_exit);rows=check_hist_parent(hr,he,sha(hist_run));need(Path(hist_run).resolve()==(HF/HROOT/'RUN.json').resolve() and Path(hist_exit).resolve()==(HF/HROOT).parent.joinpath('EXIT_RECEIPT.json').resolve(),'hist receipt path mismatch')
  h={'RUN':binding(hist_run,HF),'exit':binding(hist_exit,HF),'files':rows}
 return {'id':ID,'status':'DRAFT_REVIEW_REQUESTED','design':DESIGN,'resources':RESOURCES,'cpu_resources':CPU_RESOURCES,'limits':LIMITS,'output':OUT,'cpu_output':CPU,'source_bindings':sources,'author_sources':author,'parent_model':parent['output'],'parent_receipts':[parent[k] for k in ('conversion_RUN','conversion_exit','verification_RUN','verification_exit')],'histograms':h,'runtime':envutil.runtime(),'worker_config_sha256':None,'review':None}

def fingerprint(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def job_spec(protocol,job,profile,cpu):
 protocol=Path(protocol).resolve();job=Path(job).resolve();profile=Path(profile).resolve();out=HF/(CPU if cpu else OUT)
 prefix=['env','FFNDS_WORKER_CONFIG='+str(profile)]
 if cpu:prefix+=['CUDA_VISIBLE_DEVICES=','taskset','-c','0']
 else:prefix+=[str(ROOT/'scripts/worker_teal_gpu0_uuid_exec.sh'),'taskset','-c','0,1']
 return {'id':ID,'protocol_path':str(protocol),'protocol_sha256':sha(protocol),'cwd':str(ROOT),'argv':prefix+[ENV,'-I','-B',str(ROOT/SELF),'preflight' if cpu else 'run','--protocol',str(protocol),'--job-spec',str(job)],'output_dir':str(out.parent),'hf_root':str(HF),'run_json':str(out/'RUN.json'),'external_timeout_seconds':CPU_RESOURCES['outer_seconds'] if cpu else RESOURCES['outer_seconds']}

def prepare(a):
 need(a.out and a.worker_profile and a.hist_run and a.hist_exit and a.cpu_job_spec and a.gpu_job_spec,'closed parent, profile and all three fresh output paths required')
 paths=[a.out,a.cpu_job_spec,a.gpu_job_spec];need(len({str(x.resolve()) for x in paths})==3 and all(not x.exists() for x in paths),'fresh distinct outputs required')
 need(a.worker_profile.is_file() and a.cpu_job_spec.parent.is_dir() and a.gpu_job_spec.parent.is_dir() and a.out.parent.is_dir(),'private spec/output parent missing')
 p=draft(a.hist_run,a.hist_exit);p['status']='FROZEN_PENDING_REGISTRATION';p['worker_config_sha256']=sha(a.worker_profile);validate(p,True)
 write(a.out,p);old_umask=os.umask(0o077)
 try:write(a.cpu_job_spec,job_spec(a.out,a.cpu_job_spec,a.worker_profile,True));write(a.gpu_job_spec,job_spec(a.out,a.gpu_job_spec,a.worker_profile,False))
 finally:os.umask(old_umask)
 print(json.dumps({'fingerprint':fingerprint(p),'protocol_sha256':sha(a.out),'cpu_job_sha256':sha(a.cpu_job_spec),'gpu_job_sha256':sha(a.gpu_job_spec),'status':'PENDING_INDEPENDENT_RECHECK_AND_ACTIVE'},indent=2))
def validate(p,frozen=False):
 need(p['id']==ID and p['design']==DESIGN and p['resources']==RESOURCES and p['cpu_resources']==CPU_RESOURCES and p['limits']==LIMITS and p['output']==OUT and p['cpu_output']==CPU,'scope differs')
 need({x['path'] for x in p['source_bindings']}==set(SOURCES) and {x['path'] for x in p['author_sources']}==set(AUTHOR_FILES),'source closure differs');need(p['parent_model']['sha256']==MODEL_SHA and p['parent_model']['bytes']==MODEL_SIZE and len(p['parent_receipts'])==4,'model parent differs')
 if frozen:
  need(p['status']=='FROZEN_PENDING_REGISTRATION' and p['review'] is None and isinstance(p['worker_config_sha256'],str) and len(p['worker_config_sha256'])==64 and p['histograms'] is not None,'actual closed hist required')
  need({x['path'] for x in p['histograms']['files']}==hist_keyset(),'hist paths differ')

def authorize(p,d):
 validate(p,True);a=read(ROOT/'research/ACTIVE_RESEARCH.json');sc=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID];need(a.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True and len(sc)==1,'ACTIVE required');s=sc[0];need(s.get('protocol_sha256')==d and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('resources')==RESOURCES and s.get('CPU_preflight_resources')==CPU_RESOURCES and s.get('limits')==LIMITS and s.get('max_gpu_cards')==1,'ACTIVE scope mismatch')
 r=read(bound(ROOT,s['review']));need(r.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and r.get('scope_id')==ID and r.get('state')=='REVIEWER_RECHECK_APPROVED' and r.get('recheck_disposition')=='APPROVE' and r.get('reviewer_disposition') in ('APPROVE','REVISE') and r.get('author_agent_id') and r.get('reviewer_agent_id') and r['author_agent_id']!=r['reviewer_agent_id'] and r.get('responses_to_review') and r.get('adversarial_counterexample') and r.get('approval_scope_fingerprint')==fingerprint(p),'independent review required');need(r.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['source_bindings']},'review closure mismatch')
 for k in ('author_review','independent_review','reviewer_recheck'):bound(ROOT,r[k])
 return s

def verify_inputs(p,full_hist):
 parent_model()
 for x in p['source_bindings']:bound(ROOT,x)
 for x in p['author_sources']+p['parent_receipts']:bound(HF,x)
 h=p['histograms'];hr=read(bound(HF,h['RUN']));he=read(bound(HF,h['exit']));need(check_hist_parent(hr,he,h['RUN']['sha256'])==h['files'],'hist parent changed')
 model=HF/p['parent_model']['path'];need(model.is_file() and not model.is_symlink() and model.stat().st_size==MODEL_SIZE and 'llama-3-8b' in model.parent.name.lower(),'model path/stat mismatch');token=model.parent/'tokenizer.model';need(token.is_file() and not token.is_symlink() and token.stat().st_size==CPU_RESOURCES['tokenizer_hash_bytes'] and sha(token)==TOKEN_SHA,'tokenizer mismatch')
 if full_hist:
  for x in h['files']:
   path=HF/HROOT/x['path'];need(path.is_file() and not path.is_symlink() and path.stat().st_size==x['bytes'] and sha(path)==x['sha256'],'hist bytes changed')
 return model,token,HF/HROOT/'author/histograms'

def verify_job(a,p,s,cpu):
 spec_key='cpu_preflight_spec_sha256' if cpu else 'launch_spec_sha256';need(a.job_spec and sha(a.job_spec)==s[spec_key],'job spec differs');j=read(a.job_spec);argv=j['argv'];need(argv[0]=='env' and argv[1].startswith('FFNDS_WORKER_CONFIG='),'worker spec');profile=Path(argv[1].split('=',1)[1]);need(sha(profile)==p['worker_config_sha256'],'private profile differs')
 if cpu:
  out=HF/CPU;expected=['env',argv[1],'CUDA_VISIBLE_DEVICES=','taskset','-c','0',ENV,'-I','-B',str(ROOT/SELF),'preflight','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())];outer=CPU_RESOURCES['outer_seconds']
 else:
  out=HF/OUT;expected=['env',argv[1],str(ROOT/'scripts/worker_teal_gpu0_uuid_exec.sh'),'taskset','-c','0,1',ENV,'-I','-B',str(ROOT/SELF),'run','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())];outer=RESOURCES['outer_seconds']
 need(argv==expected and j['id']==ID and j['protocol_sha256']==sha(a.protocol) and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and j['hf_root']==str(HF) and Path(j['output_dir']).resolve()==out.parent and Path(j['run_json']).resolve()==out/'RUN.json' and j['external_timeout_seconds']==outer,'exact supervisor differs')
 need(sys.flags.isolated and sorted(os.sched_getaffinity(0))==([0] if cpu else [0,1]) and envutil.runtime()==p['runtime'],'environment differs');import worker_identity;worker_identity.require_worker();return out

def preflight(a):
 p=read(a.protocol);d=sha(a.protocol);s=authorize(p,d);out=verify_job(a,p,s,True);need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not {'torch','triton','transformers'} & set(sys.modules),'preflight is CPU only');need(not out.exists(),'fresh CPU preflight required');out.mkdir(parents=True,mode=0o700);t=time.monotonic();cpu=time.process_time();r={'id':ID,'protocol_sha256':d,'spec_sha256':s['cpu_preflight_spec_sha256'],'status':'STARTED','hist_hash_bytes_predebited':sum(x['bytes'] for x in p['histograms']['files']),'hist_hash_bytes_completed':0,'tokenizer_hash_bytes_predebited':CPU_RESOURCES['tokenizer_hash_bytes'],'other_input_bytes':'UNKNOWN: source, receipts and package metadata; histogram limit is not total I/O','model_calls':0,'GPU_calls':0,'automatic_retry':False}
 def persist():write(out/'RUN.json',r)
 persist()
 try:
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU preflight timeout')));signal.alarm(CPU_RESOURCES['wall_seconds']);verify_inputs(p,True);r['hist_hash_bytes_completed']=r['hist_hash_bytes_predebited'];r['tokenizer_hash_bytes_completed']=CPU_RESOURCES['tokenizer_hash_bytes'];r['status']='PASS_CPU_PREFLIGHT_NO_MODEL_NO_GPU'
 except BaseException as e:r.update(status='FAILED_CPU_PREFLIGHT_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:signal.alarm(0);r.update(wall_seconds=time.monotonic()-t,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);persist()

def require_cpu(p,d,s):
 rr=HF/CPU/'RUN.json';ee=(HF/CPU).parent/'EXIT_RECEIPT.json';r=read(rr);e=read(ee);need(r.get('status')=='PASS_CPU_PREFLIGHT_NO_MODEL_NO_GPU' and r.get('protocol_sha256')==d and r.get('hist_hash_bytes_predebited')==r.get('hist_hash_bytes_completed') and r.get('model_calls')==r.get('GPU_calls')==0 and e.get('id')==ID and e.get('process_exited') is True and e.get('child_returncode')==0 and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('run_sha256')==sha(rr) and e.get('protocol_sha256')==d and e.get('spec_sha256')==s.get('cpu_preflight_spec_sha256'),'actual CPU preflight EXIT0 required');return {'RUN':binding(rr,HF),'exit':binding(ee,HF)}

def set_truezero(model):
 need(len(model.layers)==32,'32 layers required')
 for layer in model.layers:
  for name in ('thresh_q','thresh_k','thresh_v','thresh_o'):setattr(layer.attention,name,0.0)
  for name in ('thresh_up','thresh_gate','thresh_down'):setattr(layer.feed_forward,name,0.0)

def thresholds(model):
 out=[]
 for i,layer in enumerate(model.layers):
  vals={name:float(getattr(layer.attention,'thresh_'+name)) for name in ('q','k','v','o')};vals.update({name:float(getattr(layer.feed_forward,'thresh_'+name)) for name in ('up','gate','down')});need(all(math.isfinite(x) and x>=0 for x in vals.values()),'nonfinite or negative thresholds');out.append({'layer':i,'values':vals})
 return out

def budget_before(r,persist,key,n=1):
 need(r['predebited'][key]+n<=LIMITS[key],'operation exceeds '+key);r['predebited'][key]+=n;persist()
def budget_completed(r,key,n=1):r['completed'][key]+=n

def cache_identity(model,T):
 capacity=((T+68+7)//8)*8;need(model.max_seq_length==capacity and model.max_batch_size==1 and tuple(model.causal_mask.shape)==(capacity,capacity),'cache capacity/mask differs')
 shapes=[]
 for layer in model.layers:
  cache=layer.attention.kv_cache;ks=tuple(cache.k_cache.shape);vs=tuple(cache.v_cache.shape);need(ks==vs==(1,8,capacity,128),'per-layer KV layout differs');shapes.append(list(ks))
 return {'capacity':capacity,'per_layer_kv_shapes':shapes,'causal_mask_shape':list(model.causal_mask.shape),'mask_semantics':'author lower triangular; input_pos indexes rows','generation_positions':{'prefill':[0,T-1],'decode':[T,T+62]}}

def neutral_compare(torch,dense,zero,prompt,report,persist):
 # Same fixed teacher tokens and positions; not a free-generation equality claim.
 device=prompt.device;T=int(prompt.numel());seq=prompt[-1:].reshape(1,1);tokens=[int(x) for x in prompt[-4:].tolist()];tokens=(tokens*4)[:4]
 observations=[]
 for step in range(5):
  x=prompt.view(1,-1) if step==0 else torch.tensor([[tokens[step-1]]],device=device,dtype=torch.int);pos=torch.arange(T,device=device) if step==0 else torch.tensor([T+step-1],device=device,dtype=torch.int)
  pair=[]
  for model in (dense,zero):
   budget_before(report,persist,'teacher_forced_model_calls')
   if step==0:v=model(x,pos).float()
   else:
    with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True):v=model(x,pos).float()
   torch.cuda.synchronize();budget_completed(report,'teacher_forced_model_calls');pair.append(v[0,-1].detach())
  a,b=pair;finite=bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item());den=torch.linalg.vector_norm(a).item();rel=torch.linalg.vector_norm(b-a).item()/max(den,1e-12);topa=int(torch.argmax(a).item());topb=int(torch.argmax(b).item());observations.append({'step':step,'relative_l2':rel,'dense_top1':topa,'zero_top1':topb,'finite':finite})
  report['neutral']=observations;persist();need(finite and rel<=DESIGN['neutral_rel_l2_max'] and topa==topb,'true-zero neutral failed')
 return observations

def add_observer(model,rows,torch):
 restore=[]
 def wrap(owner,name,layer,kind):
  old=getattr(owner,name);restore.append((owner,name,old))
  def call(x,w,*args):
   ts=args[:3] if kind=='qkv' else args[:1]
   names={'qkv':['q','k','v'],'attn_o':['o'],'ff1':['gate' if w is owner.w1.weight else 'up' if w is owner.w3.weight else 'UNKNOWN'],'ff2':['down']}[kind];need('UNKNOWN' not in names,'unexpected FFN input weight')
   for label,t in zip(names,ts):
    size=x.numel();zero=int((x==0).sum().item());masked=int((x.abs()<=float(t)).sum().item());rows.append({'layer':layer,'projection':label,'threshold':float(t),'total':size,'preexisting_zero':zero,'masked':masked})
   return old(x,w,*args)
  setattr(owner,name,call)
 for i,l in enumerate(model.layers):
  wrap(l.attention,'gemv1',i,'qkv');wrap(l.attention,'gemv2',i,'attn_o');wrap(l.feed_forward,'gemv1',i,'ff1');wrap(l.feed_forward,'gemv2',i,'ff2')
 return restore

def check_observer(rows):
 need(len(rows)==224 and {(x['layer'],x['projection']) for x in rows}=={(i,p) for i in range(32) for p in ('q','k','v','o','up','gate','down')},'observer projection coverage differs')

def author_import():
 root=HF/SRC;support=HF/SUP;sys.path.insert(0,str((ROOT/KERNEL).parent));sys.path.insert(0,str(root));sys.path.insert(0,str(root/'gpt-fast'));sys.path.insert(0,str(support))
 import generate, importlib
 for name,path in (('generate',root/'gpt-fast/generate.py'),('model',root/'gpt-fast/model.py'),('tokenizer',root/'gpt-fast/tokenizer.py'),('kernels.sparse_gemv',root/'kernels/sparse_gemv.py'),('compile_wrapper',ROOT/KERNEL),('utils.utils',root/'utils/utils.py'),('distribution',support/'distribution.py'),('tp',support/'tp.py')):
  need(Path(importlib.import_module(name).__file__).resolve()==path.resolve(),'wrong author module '+name)
 return generate

def run(a):
 need(not {'torch','triton','transformers'} & set(sys.modules),'fresh process required');p=read(a.protocol);d=sha(a.protocol);s=authorize(p,d);out=verify_job(a,p,s,False);proof=require_cpu(p,d,s);need(not out.exists(),'fresh GPU attempt required');out.mkdir(parents=True,mode=0o700);t=time.monotonic();cpu=time.process_time();ops=('model_loads','cache_preparations','checkpoint_hash_passes','histogram_hash_passes','histogram_file_loads','compile_calls','generations','generation_prefills','generation_decode_calls','teacher_forced_model_calls','observer_model_calls','observer_mask_counts','observer_zero_counts')
 r={'id':ID,'status':'STARTED_NO_RETRY','protocol_sha256':d,'launch_spec_sha256':s['launch_spec_sha256'],'cpu_preflight':proof,'predebited':dict.fromkeys(ops,0),'completed':dict.fromkeys(ops,0),'design':DESIGN,'limits':LIMITS,'maximum_model_calls':590,'model_call_breakdown':'9 prefill + 567 decode + 10 teacher + 4 observer; logical only','mmap_actual_read_bytes':'UNKNOWN for 3 model loads; two explicit full hashes separately predebited','other_input_reads':'tokenizer, source, receipts, environment metadata and author histogram loads; bytes outside explicit hash UNKNOWN','automatic_retry':False,'physical_launches':'UNKNOWN_TRITON_AUTOTUNE_AND_CUDA_GRAPHS','partial_generation_counts':'IN_MEMORY_COMPLETIONS_PERSISTED_IN_FINALLY_PREDEBITED_PER_GENERATION','model_or_tensor_execution_preflight':False,'source_evidence':'AUTHOR_GPT_FAST_ALL_MATRIX_FINITE_DEVELOPMENT_PROBE'}
 emergency=envutil.reserve_emergency(out)
 def persist():envutil.checked_persist(out/'RUN.json',r,out.parent,RESOURCES['new_HF_bytes'])
 persist();signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('output monitor failed')));diskmon=envutil.UsageMonitor(out.parent,RESOURCES['new_HF_bytes'],2<<30);diskmon.start();gpumon=None;fd=None
 try:
  need(os.environ.get('FFNDS_PHYSICAL_GPU')=='0' and len(os.environ.get('CUDA_VISIBLE_DEVICES',''))>4,'GPU launcher identity');need(shutil_disk_free(HF)>=RESOURCES['minimum_free_disk_bytes'],'disk reserve');resource.setrlimit(resource.RLIMIT_AS,(RESOURCES['address_space_max_bytes'],)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(RESOURCES['output_file_max_bytes'],)*2)
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('GPU scope wall limit')));signal.alarm(RESOURCES['wall_seconds']);os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TRITON_CACHE_DIR=str(out/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache'))
  import torch;torch.set_num_threads(2);torch.set_num_interop_threads(1);need(torch.__version__.startswith('2.4.0'),'torch version');from startup_gpu_reservation import reserve_before_assets
  reserve,gpumon,receipt=reserve_before_assets(torch);r['CUDA_reservation']=receipt;name=torch.cuda.get_device_name(0);cap=list(torch.cuda.get_device_capability(0));need('A100' in name and cap==[8,0],'A100 SM80 required');r['device']={'name':name,'capability':cap};persist()
  budget_before(r,persist,'histogram_hash_passes');model_path,token_path,histroot=verify_inputs(p,True);budget_completed(r,'histogram_hash_passes');persist()
  fd=envutil.pin_file(model_path);stat0=envutil.statid(os.fstat(fd));budget_before(r,persist,'checkpoint_hash_passes');need(envutil.tracked_hash('/proc/self/fd/'+str(fd),MODEL_SIZE,r,persist,'model_before')==MODEL_SHA,'model SHA mismatch');budget_completed(r,'checkpoint_hash_passes');persist()
  staged=out/'llama-3-8b';staged.mkdir(mode=0o700);os.symlink('/proc/self/fd/'+str(fd),staged/'model.pth');os.symlink(token_path,staged/'tokenizer.model');g=author_import();tokenizer=g.get_tokenizer(staged/'tokenizer.model',staged/'model.pth');prompt=g.encode_tokens(tokenizer,PROMPT,bos=True,device='cuda');need(1<int(prompt.numel())<=256,'invalid prompt length');r['prompt']={'text':PROMPT,'token_ids':prompt.tolist(),'token_count':int(prompt.numel())};persist()
  original_load=torch.load;histpaths={str((HF/HROOT/x['path']).resolve()) for x in p['histograms']['files']}
  def counted_load(path,*args,**kwargs):
   resolved=str(Path(path).resolve())
   if resolved in histpaths:
    budget_before(r,persist,'histogram_file_loads');value=original_load(path,*args,**kwargs);budget_completed(r,'histogram_file_loads');persist();return value
   need(resolved==str(model_path.resolve()),'unbound torch.load asset');return original_load(path,*args,**kwargs)
  models={};torch.load=counted_load
  try:
   for arm in DESIGN['arms']:
    budget_before(r,persist,'model_loads');st=time.monotonic();models[arm]=g._load_model(staged/'model.pth','cuda',torch.float16,False,None if arm=='dense' else str(histroot),0.5 if arm!='dense' else 0.0);torch.cuda.synchronize();budget_completed(r,'model_loads');model=models[arm];need(len(model.layers)==32 and model.config.dim==4096 and model.config.intermediate_size==14336 and all(v.device.type=='cuda' and v.device.index==0 and v.dtype==torch.float16 for v in model.parameters()),'model shape/device/dtype mismatch');r.setdefault('setup',{})[arm]={'model_load_seconds':time.monotonic()-st,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'layers':32,'dimension':4096,'intermediate_size':14336,'all_parameters_cuda0_fp16':True};need(torch.cuda.memory_reserved()<=RESOURCES['GPU_reserved_max_bytes'],'GPU reservation exceeded');gpumon.check();persist()
  finally:torch.load=original_load
  set_truezero(models['truezero']);r['thresholds']={'truezero':thresholds(models['truezero']),'uniform50':thresholds(models['uniform50'])};need(all(v==0.0 for x in r['thresholds']['truezero'] for v in x['values'].values()),'zero threshold mismatch');persist()
  cache_started=time.monotonic();T=int(prompt.numel())
  for model in models.values():
   budget_before(r,persist,'cache_preparations')
   with torch.device('cuda'):model.setup_caches(max_batch_size=1,max_seq_length=T+68)
   torch.cuda.synchronize();budget_completed(r,'cache_preparations');persist()
  torch.cuda.synchronize();r['cache_setup_wall_seconds']=time.monotonic()-cache_started;r['cache_identity']={arm:cache_identity(model,T) for arm,model in models.items()};need(len({json.dumps(x,sort_keys=True) for x in r['cache_identity'].values()})==1,'arm cache layouts differ');persist()
  with torch.inference_mode():neutral_compare(torch,models['dense'],models['truezero'],prompt,r,persist)
  original_decode=g.decode_one_token;original_prefill=g.prefill;compiled={}
  for arm in DESIGN['arms']:
   budget_before(r,persist,'compile_calls');st=time.monotonic();compiled[arm]=torch.compile(original_decode,mode='reduce-overhead',fullgraph=True,dynamic=False);budget_completed(r,'compile_calls');r['setup'][arm]['compile_wrapper_seconds']=time.monotonic()-st;persist()
  r['generations']=[];occurrences=dict.fromkeys(DESIGN['arms'],0)
  for arm in DESIGN['sequence']:
   model=models[arm];k=occurrences[arm];occurrences[arm]+=1;need(cache_identity(model,T)==r['cache_identity'][arm],'cache layout changed before generation');entry={'arm':arm,'ordinal':k,'phase':'warmup' if k==0 else 'measured','cache_layout':r['cache_identity'][arm],'logical_prefill_completed':0,'logical_decode_completed':0,'status':'INFLIGHT','timing_identity':'PER_TOKEN_CUDA_EVENT_INSTRUMENTED_THREE_MODELS_RESIDENT_NOT_PAPER_SPEED'};r['generations'].append(entry)
   budget_before(r,persist,'generations');budget_before(r,persist,'generation_prefills');budget_before(r,persist,'generation_decode_calls',63);events=[]
   def measured_prefill(*args,**kw):
    a1=torch.cuda.Event(enable_timing=True);b1=torch.cuda.Event(enable_timing=True);a1.record();v=original_prefill(*args,**kw);b1.record();events.append(('prefill',a1,b1));entry['logical_prefill_completed']+=1;return v
   def measured_decode(*args,**kw):
    a1=torch.cuda.Event(enable_timing=True);b1=torch.cuda.Event(enable_timing=True);a1.record();v=compiled[arm](*args,**kw);b1.record();events.append(('decode',a1,b1));entry['logical_decode_completed']+=1;return v
   g.prefill=measured_prefill;g.decode_one_token=measured_decode;torch.manual_seed(DESIGN['seed']);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();start.record();wall=time.monotonic()
   try:
    with torch.inference_mode():y,_=g.generate(model,prompt,64,interactive=False,draft_model=None,temperature=DESIGN['temperature'],top_k=DESIGN['top_k'])
    end.record();torch.cuda.synchronize();wall=time.monotonic()-wall;ids=y.tolist();generated=ids[int(prompt.numel()):];eos=int(tokenizer.eos_id());need(len(ids)==int(prompt.numel())+64 and entry['logical_prefill_completed']==1 and entry['logical_decode_completed']==63 and cache_identity(model,T)==r['cache_identity'][arm],'author generation count/cache differs');entry.update(status='COMPLETE',generated_token_ids=generated,eos_id=eos,eos_positions=[i for i,v in enumerate(generated) if v==eos],wall_seconds=wall,current_stream_cuda_seconds=start.elapsed_time(end)/1000,prefill_cuda_seconds=sum(a.elapsed_time(b) for n,a,b in events if n=='prefill')/1000,decode_cuda_seconds=sum(a.elapsed_time(b) for n,a,b in events if n=='decode')/1000,peak_reserved_bytes=torch.cuda.max_memory_reserved());budget_completed(r,'generations');budget_completed(r,'generation_prefills');budget_completed(r,'generation_decode_calls',63);gpumon.check();need(torch.cuda.memory_reserved()<=RESOURCES['GPU_reserved_max_bytes'],'GPU reservation exceeded');persist()
   finally:g.prefill=original_prefill;g.decode_one_token=original_decode
  r['observer']={}
  for arm in ('truezero','uniform50'):
   model=models[arm];rows=[];r['observer'][arm]={'rows':rows,'status':'INFLIGHT'};st=time.monotonic();T=int(prompt.numel())
   with torch.inference_mode():
    budget_before(r,persist,'observer_model_calls');model(prompt.view(1,-1),torch.arange(T,device='cuda'));torch.cuda.synchronize();budget_completed(r,'observer_model_calls');persist()
    restore=add_observer(model,rows,torch)
    try:
     budget_before(r,persist,'observer_mask_counts',224);budget_before(r,persist,'observer_zero_counts',224);budget_before(r,persist,'observer_model_calls');model(prompt[-1:].view(1,1),torch.tensor([T],device='cuda',dtype=torch.int));torch.cuda.synchronize();check_observer(rows);budget_completed(r,'observer_mask_counts',224);budget_completed(r,'observer_zero_counts',224);budget_completed(r,'observer_model_calls');r['observer'][arm].update(status='COMPLETE',wall_seconds=time.monotonic()-st);persist()
    finally:
     for owner,name,old in restore:setattr(owner,name,old)
  need(envutil.statid(os.fstat(fd))==stat0,'model changed');budget_before(r,persist,'checkpoint_hash_passes');need(envutil.tracked_hash('/proc/self/fd/'+str(fd),MODEL_SIZE,r,persist,'model_after')==MODEL_SHA,'model changed after generation');budget_completed(r,'checkpoint_hash_passes');need(all(r['predebited'][k]==r['completed'][k] for k in ops),'incomplete ledger');gpumon.check();r['status']='COMPLETE_AUTHOR_FULL32_FINITE_DIAGNOSTIC_NOT_PAPER_BENCHMARK'
 except BaseException as e:r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:
  signal.alarm(0)
  if fd is not None:os.close(fd)
  if gpumon is not None:gpumon.close();r['gpu_monitor']=gpumon.report()
  r.update(wall_seconds=time.monotonic()-t,cpu_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,whole_worker_monitor=diskmon.stop())
  if r['whole_worker_monitor']['fault'] or not r['whole_worker_monitor']['thread_stopped'] or (gpumon is not None and gpumon.violations):r['status']='FAILED_MONITOR_PRESERVED'
  try:persist()
  except BaseException as e:envutil.emergency_failure(emergency,r,'final persist failed: '+type(e).__name__);raise
  if r['status']=='FAILED_MONITOR_PRESERVED':raise RuntimeError('monitor failure')

def shutil_disk_free(path):
 import shutil
 return shutil.disk_usage(path).free

def verify_closed(a):
 p=read(a.protocol);validate(p,True);d=sha(a.protocol);out=HF/OUT;rr=out/'RUN.json';ee=out.parent/'EXIT_RECEIPT.json';need(a.run and a.exit and a.run.resolve()==rr.resolve() and a.exit.resolve()==ee.resolve(),'receipt paths differ');r=read(rr);e=read(ee)
 need(e.get('id')==ID and e.get('process_exited') is True and e.get('child_returncode')==0 and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('run_sha256')==sha(rr) and e.get('protocol_sha256')==d and e.get('spec_sha256')==r.get('launch_spec_sha256') and r.get('status')=='COMPLETE_AUTHOR_FULL32_FINITE_DIAGNOSTIC_NOT_PAPER_BENCHMARK','actual EXIT0 required')
 need(r['design']==DESIGN and len(r.get('generations',[]))==9 and all(x.get('status')=='COMPLETE' and len(x.get('generated_token_ids',[]))==64 for x in r['generations']) and [x['arm'] for x in r['generations']]==DESIGN['sequence'],'generation identity')
 need(len(r.get('neutral',[]))==5 and all(x['finite'] and x['relative_l2']<=DESIGN['neutral_rel_l2_max'] and x['dense_top1']==x['zero_top1'] for x in r['neutral']),'neutral missing')
 keys=('model_loads','cache_preparations','checkpoint_hash_passes','histogram_hash_passes','histogram_file_loads','compile_calls','generations','generation_prefills','generation_decode_calls','teacher_forced_model_calls','observer_model_calls','observer_mask_counts','observer_zero_counts');need(all(r['predebited'][k]==r['completed'][k]==LIMITS[k] for k in keys),'ledger incomplete')
 for arm in ('truezero','uniform50'):need(r['observer'][arm]['status']=='COMPLETE','observer incomplete');check_observer(r['observer'][arm]['rows'])
 need(len({json.dumps(x,sort_keys=True) for x in r['cache_identity'].values()})==1 and set(r['cache_identity'])==set(DESIGN['arms']),'cache identity differs');need(r.get('gpu_monitor',{}).get('sampled_isolation_pass') is True and r.get('whole_worker_monitor',{}).get('thread_stopped') is True and not r['whole_worker_monitor'].get('fault') and r.get('wall_seconds',RESOURCES['wall_seconds']+1)<=RESOURCES['wall_seconds'],'resource evidence incomplete')
 return {'id':ID,'status':'PASS_CLOSED_AUTHOR_FULL32_FINITE_DIAGNOSTIC','RUN':binding(rr,HF),'exit':binding(ee,HF),'quality_or_speed_acceptance':False,'full_paper_benchmark':False}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('draft','prepare','inspect','preflight','run','verify-closed'));ap.add_argument('--out',type=Path);ap.add_argument('--protocol',type=Path);ap.add_argument('--job-spec',type=Path);ap.add_argument('--hist-run',type=Path);ap.add_argument('--hist-exit',type=Path);ap.add_argument('--worker-profile',type=Path);ap.add_argument('--cpu-job-spec',type=Path);ap.add_argument('--gpu-job-spec',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--exit',type=Path);a=ap.parse_args()
 if a.action=='draft':need(a.out and not a.out.exists(),'fresh draft');write(a.out,draft(a.hist_run,a.hist_exit))
 elif a.action=='prepare':prepare(a)
 elif a.action=='inspect':p=read(a.protocol);validate(p);print(json.dumps({'fingerprint':fingerprint(p),'hist_status':'ACTUAL_PARENT_BOUND' if p['histograms'] else 'MISSING_NOT_FREEZABLE','resources':RESOURCES,'limits':LIMITS},indent=2))
 elif a.action=='preflight':preflight(a)
 elif a.action=='run':run(a)
 else:print(json.dumps(verify_closed(a),indent=2))
if __name__=='__main__':main()
