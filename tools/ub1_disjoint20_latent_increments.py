"""Source-cluster signed increments from saved native latents; NumPy CPU only."""
import argparse,filecmp,hashlib,importlib.util,importlib.metadata,json,math,os,resource,shutil,signal,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='UB1_DISJOINT20_LATENT_INCREMENTS_20261027'
SHAPE=(16,21,58,104)
LIMITS={'cpu_threads':2,'wall_seconds':300,'external_wall_seconds':360,'memory_bytes':1<<30,'output_bytes':32<<20,'attempts':1,'gpu':0,'model_calls':0}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def bind(root,item):
 p=(root/item['path']).resolve()
 if not p.is_relative_to(root.resolve()) or not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Input binding differs: '+item['path'])
 return p

def decode_bf16(path,shape=SHAPE,expected_raw_sha=None):
 import numpy as np
 with open(path,'rb') as f:
  n=int.from_bytes(f.read(8),'little')
  if n<=0 or n>1<<20:raise ValueError('Invalid safetensor header length')
  header=json.loads(f.read(n));info=header.get('latents');data=f.read()
 if set(header)-{'latents','__metadata__'} or not info or info['dtype']!='BF16' or tuple(info['shape'])!=tuple(shape) or info['data_offsets']!=[0,len(data)] or len(data)!=math.prod(shape)*2:raise ValueError('BF16 tensor header or length differs')
 if expected_raw_sha is not None and hashlib.sha256(data).hexdigest()!=expected_raw_sha:raise ValueError('BF16 raw tensor SHA differs')
 words=np.frombuffer(data,dtype='<u2').astype(np.uint32);out=(words<<16).view(np.float32).reshape(shape)
 if not np.isfinite(out).all():raise ValueError('Nonfinite BF16 initial state')
 return out

def verify_bf16_post(value,shape=SHAPE):
 import numpy as np
 a=np.asarray(value)
 if a.dtype!=np.float32 or a.shape!=tuple(shape) or not np.isfinite(a).all() or np.any((a.view(np.uint32)&np.uint32(0xffff))!=0):raise ValueError('Post state is not finite BF16-exact FP32')
 return a

def increment_stats(dprev,cprev,dpost,cpost,h):
 import numpy as np
 if not math.isfinite(h) or h<=0:raise ValueError('Invalid sigma step')
 arrays=[np.asarray(x,dtype=np.float64) for x in (dprev,cprev,dpost,cpost)]
 if any(a.shape!=arrays[0].shape or not np.isfinite(a).all() for a in arrays):raise ValueError('Invalid state')
 dp,cp,dn,cn=arrays;e0=cp-dp;delta=(cn-cp)-(dn-dp);e1=cn-dn
 def one(a,b,c):
  inner=lambda x,y:float(np.dot(x.ravel(),y.ravel()))
  prev=inner(a,a);inc=inner(b,b);cross=2*inner(a,b);curr=inner(c,c);closure=curr-prev-inc-cross;scale=max(1.,curr,prev,inc,abs(cross));tolerance=64*np.finfo(np.float64).eps*a.size*scale
  if abs(closure)>tolerance:raise ValueError('Energy decomposition closure failed')
  return {'previous_error_energy':prev,'increment_difference_energy':inc,'twice_previous_increment_dot':cross,'current_error_energy':curr,'closure_error':closure,'closure_tolerance':float(tolerance),'cos_previous_increment':cross/(2*math.sqrt(prev*inc)) if prev and inc else None,'increment_difference_norm':math.sqrt(inc),'effective_velocity_from_state_difference_norm':math.sqrt(inc)/h}
 return {'identity':one(e0,delta,e1),'latent_time_first_difference':one(np.diff(e0,axis=1),np.diff(delta,axis=1),np.diff(e1,axis=1))}

def summarize(cases):
 metrics=('previous_error_energy','increment_difference_energy','twice_previous_increment_dot','current_error_energy','increment_difference_norm','effective_velocity_from_state_difference_norm')
 out={}
 for group,subset in [('all',cases),('subject',[x for x in cases if x['index']<10]),('background',[x for x in cases if x['index']>=10])]:
  if not subset:raise ValueError('Empty source group')
  out[group]={'source_count':len(subset),'steps':[]}
  for step in range(4):
   row={'step':step,'source_count':len(subset)}
   for projection in ('identity','latent_time_first_difference'):
    row[projection]={k:sum(x['steps'][step]['statistics'][projection][k] for x in subset)/len(subset) for k in metrics}
   out[group]['steps'].append(row)
 return out

def authorize(plan,digest,active):
 if plan.get('id')!=ID or plan.get('limits')!=LIMITS or plan.get('assets_finalized') is not True:raise ValueError('Unfrozen scope')
 u=active.get('user_continuation_authorization_20261027',{})
 if u.get('id')!='USER_RESUME_OFFLINE_IFC_20261027' or u.get('status')!='ACTIVE_CURRENT_USER_DIRECTED_SESSION':raise ValueError('User authorization inactive')
 if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in active['experiment_authority']['current_user_authorized_scopes']):raise ValueError('Exact scope not registered')

def preflight(plan,hf):
 for x in plan['repo_sources']:bind(ROOT,x)
 for x in plan['inputs']:bind(hf,x)
 cases=plan['cases'];producers=plan['producers']
 if len(cases)!=20 or {x['index'] for x in cases}!=set(range(20)) or len(producers)!=4 or {x['part'] for x in producers}!=set(range(4)) or len(plan['inputs'])!=244:raise ValueError('Required 200 tensors, 40 generations, four producer RUNs')
 indexed={x['path']:x for x in plan['inputs']}
 if len(indexed)!=244:raise ValueError('Repeated input path')
 expected_paths={x['run_path'] for x in producers}
 for case in cases:
  for mode in ('dense','candidate'):
   arm=case['arms'][mode]
   if len(arm['post'])!=4:raise ValueError('Expected four post states')
   expected_paths.update([arm['generation'],arm['initial'],*arm['post']])
 if len(expected_paths)!=244 or set(indexed)!=expected_paths:raise ValueError('Input references do not cover exact assets')
 canonical=json.loads(bind(ROOT,plan['canonical_manifest']).read_text())
 if len(canonical)!=20:raise ValueError('Canonical source count differs')
 original=json.loads(bind(ROOT,plan['original_protocol']).read_text());phase_map={}
 for producer in producers:
  if producer['source_indices']!=list(range(5*producer['part'],5*producer['part']+5)):raise ValueError('Producer source partition differs')
  protocol=json.loads(bind(ROOT,producer['protocol']).read_text());doc=json.loads(bind(hf,indexed[producer['run_path']]).read_text())
  if protocol['id']!=producer['id'] or producer['protocol']['sha256']!=producer['protocol_sha256'] or doc.get('id')!=producer['id'] or doc.get('part')!=producer['producer_part'] or (producer['part']==0 and producer['protocol_sha256']==plan['original_protocol']['sha256']) or (producer['part']!=0 and producer['protocol_sha256']!=plan['original_protocol']['sha256']):raise ValueError('Producer protocol identity differs')
  if doc['status']!='PASS_FIXED_DISJOINT20_PART_FULL_DEPTH_CORE_METRICS_QUALITY_PENDING' or doc['protocol_sha256']!=producer['protocol_sha256'] or doc['source_indices']!=producer['source_indices']:raise ValueError('Producer RUN not complete')
  if any(protocol[k]['sha256']!=original[k]['sha256'] for k in ('bank_manifest','bank_protocol','cache_manifest','original_recipe')):raise ValueError('Producer bank/recipe differs')
  phases=doc.get('phases',[])
  if len(phases)!=10 or {(x['source_index'],x['mode']) for x in phases}!={(i,m) for i in producer['source_indices'] for m in ('dense','candidate')}:raise ValueError('Producer phase coverage differs')
  for phase in phases:phase_map[phase['source_index'],phase['mode']]=phase
 for c in cases:
  prod=next(x for x in producers if c['index'] in x['source_indices']);source=canonical[c['index']]
  if source['input_provenance']['image_sha256']!=c['source_id'] or source['prompt_sha256']!=c['prompt_sha256'] or source['source_group']!=c['source_group'] or source['seed']!=42:raise ValueError('Canonical source identity differs')
  for mode in ('dense','candidate'):
   arm=c['arms'][mode];g=json.loads(bind(hf,indexed[arm['generation']]).read_text());phase=phase_map[c['index'],mode]
   if phase.get('status')!='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE' or phase['generation_receipt']['sha256']!=indexed[arm['generation']]['sha256'] or phase.get('registered_run_id')!=g.get('registered_run_id'):raise ValueError('Generation outside producer RUN')
   if g['status']!='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE' or g['protocol_sha256']!=prod['protocol_sha256'] or g['source_index']!=c['index'] or g['source_id']!=c['source_id'] or g['seed']!=42 or g['mode']!=mode or g['prompt_sha256']!=c['prompt_sha256'] or g.get('image_sha256')!=c['source_id'] or g.get('scheduler_id')!=c['scheduler_id'] or not g.get('registered_run_id') or (mode=='candidate' and g.get('bank_manifest_sha256')!=original['bank_manifest']['sha256']) or len(g['trajectory'])!=4:raise ValueError('Generation provenance differs')
   if g['initial_noise']['path']!=arm['initial'] or g['initial_noise']['sha256']!=indexed[arm['initial']]['sha256'] or g['initial_noise']['dtype']!='bfloat16':raise ValueError('Initial state identity differs')
   for i,t in enumerate(g['trajectory']):
    if t['step']!=i or Path(arm['post'][i]).name!=t['path'] or t['file_sha256']!=indexed[arm['post'][i]]['sha256'] or t['dtype']!='float32' or t['shape']!=list(SHAPE):raise ValueError('Post-state identity differs')
  if not filecmp.cmp(hf/c['arms']['dense']['initial'],hf/c['arms']['candidate']['initial'],shallow=False):raise ValueError('Actual initial tensors differ')
 return True

def validate_address_limit(soft,hard):
 if soft<0 or hard<0 or soft>LIMITS['memory_bytes'] or hard>LIMITS['memory_bytes']:raise ValueError('External RLIMIT_AS must be at most 1 GiB for soft and hard limits')

def guarded_attempt(hf,plan,digest,work):
 out=(hf/plan['output_HF_relative']).resolve()
 if out==hf or not out.is_relative_to(hf):raise ValueError('Output path escaped HF')
 out.mkdir(parents=True,exist_ok=True)
 with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest},f)
 result={'id':ID,'status':'STARTED','protocol_sha256':digest,'cases':[],'native_noise_pred_saved':False,'native_state_dtype':'bfloat16','serialized_post_dtype':'float32','new_native_flows':0,'metric':'actual state difference / actual sigma difference, not recovered noise prediction','RSS_enforcement':'sampled peak RSS watchdog plus external RLIMIT_AS hard 1 GiB virtual address space limit','external_wall_seconds':LIMITS['external_wall_seconds'],'hard_kill_note':'External SIGKILL or cgroup OOM may leave STARTED; launcher exit is required and the attempt is consumed'};started=time.monotonic()
 def timeout(*_):raise TimeoutError('CPU wall limit')
 def memory_alarm(*_):raise MemoryError('RSS limit exceeded')
 def check():
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMITS['memory_bytes']:raise MemoryError('RSS limit exceeded')
  if sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())>LIMITS['output_bytes']:raise RuntimeError('Output limit exceeded')
 def save():
  tmp=out/'RUN.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');os.replace(tmp,out/'RUN.json')
 old_alarm=signal.signal(signal.SIGALRM,timeout);old_mem=signal.signal(signal.SIGUSR1,memory_alarm);signal.alarm(LIMITS['wall_seconds']);stop=threading.Event()
 def watch():
  while not stop.wait(.1):
   if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMITS['memory_bytes']:
    os.kill(os.getpid(),signal.SIGUSR1);return
 thread=threading.Thread(target=watch,daemon=True);thread.start()
 try:
  save();check();work(result,out,check,save,started)
 except BaseException as e:
  result.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
 finally:
  stop.set();thread.join();signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGUSR1,old_mem);save()
 return result

def main():
 for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='2'
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--hf-root',type=Path,required=True);args=ap.parse_args();plan=json.loads(args.protocol.read_text());digest=sha(args.protocol);hf=args.hf_root.resolve();authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()))
 def work(result,out,check,save,started):
  try:os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
  except AttributeError:pass
  validate_address_limit(*resource.getrlimit(resource.RLIMIT_AS))
  if sha(Path(sys.executable))!=plan['python_sha256'] or {n:importlib.metadata.version(n) for n in ('numpy','safetensors')}!=plan['versions']:raise ValueError('CPU environment identity differs')
  preflight(plan,hf);check()
  spec=importlib.util.spec_from_file_location('ub1_geometry',ROOT/'tools/ub1_consumer_latent_geometry.py');geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
  if geometry.SHAPE!=SHAPE:raise ValueError('Legacy reader shape differs')
  import numpy as np
  from safetensors.numpy import load_file
  snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir();shutil.copyfile(args.protocol,snapshot/'protocol.json')
  for item in plan['repo_sources']:
   target=snapshot/item['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/item['path'],target)
  check()
  for case in sorted(plan['cases'],key=lambda x:x['index']):
   states={};sigmas={}
   for mode in ('dense','candidate'):
    arm=case['arms'][mode];g=json.loads((hf/arm['generation']).read_text());initial=decode_bf16(hf/arm['initial'],expected_raw_sha=g['initial_noise']['raw_tensor_sha256']);seq=[initial]
    for step,path in enumerate(arm['post']):
     raw=load_file(str(hf/path));info=g['trajectory'][step]
     if set(raw)!={'latents'}:raise ValueError('Unexpected tensor key')
     a=verify_bf16_post(raw['latents']);
     if hashlib.sha256(a.tobytes(order='C')).hexdigest()!=info['raw_tensor_sha256']:raise ValueError('Raw FP32 tensor SHA differs')
     seq.append(a)
    states[mode]=seq;sigmas[mode]=[float(x['sigma']) for x in g['trajectory']]+[0.]
   if sigmas['dense']!=sigmas['candidate'] or any(not math.isfinite(x) for x in sigmas['dense']):raise ValueError('Sigma mismatch')
   row={'index':case['index'],'source_id':case['source_id'],'seed':42,'source_group':case['source_group'],'steps':[]}
   for i in range(4):
    h=sigmas['dense'][i]-sigmas['dense'][i+1];row['steps'].append({'step':i,'sigma':sigmas['dense'][i],'next_sigma':sigmas['dense'][i+1],'h':h,'statistics':increment_stats(states['dense'][i],states['candidate'][i],states['dense'][i+1],states['candidate'][i+1],h)});check()
   result['cases'].append(row);save()
  result['source_weighted_summary']=summarize(result['cases']);result.update(status='COMPLETE_SAVED_STATE_INCREMENTS',elapsed_seconds=time.monotonic()-started)
 
 guarded_attempt(hf,plan,digest,work)
if __name__=='__main__':main()
