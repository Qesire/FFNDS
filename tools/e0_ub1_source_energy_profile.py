"""Legacy CAL24 source-level C_mean contributions at four fixed native positions."""
import argparse,hashlib,json,os,shutil,signal,sys,time,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='E0_UB1_SOURCE_ENERGY_PROFILE_20261027'
POSITIONS=((0,0),(0,39),(3,0),(3,39))
LIMITS={'source_calls':96,'FP32_batched_bmm':2592,'logical_group_products':20736,'QDQ':0,'W1':0,'new_FFN':0,'model':0,'video':0,'fit':0,'optimizer':0,'decomposition':0,'official_scores':0}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def bound(root,item):
 p=(root/item['path']).resolve()
 if not p.is_relative_to(root.resolve()) or not p.is_file() or sha(p)!=item['sha256'] or ('bytes' in item and p.stat().st_size!=item['bytes']):raise ValueError('Bound asset differs: '+item['path'])
 return p

def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)

def reserve(counts):
 cost={'source_calls':1,'FP32_batched_bmm':27,'logical_group_products':216}
 if any(counts[k]+v>LIMITS[k] for k,v in cost.items()):raise ValueError('Budget exceeded before source')
 for k,v in cost.items():counts[k]+=v

def instrument_bmm(original,observed):
 def wrapped(*a,**kw):
  if observed['FP32_batched_bmm']>=LIMITS['FP32_batched_bmm']:raise RuntimeError('Actual bmm limit exceeded')
  observed['FP32_batched_bmm']+=1
  return original(*a,**kw)
 return wrapped

def reference_compare(computed,reference,rtol=5e-5,atol=1e-10):
 import numpy as np
 a=np.asarray(computed,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
 if a.shape!=(216,) or b.shape!=(216,) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid reference C_mean')
 diff=np.abs(a-b);bound=atol+rtol*np.abs(b)
 return {'max_abs':float(diff.max()),'max_scaled_error':float((diff/bound).max()),'within_declared_tolerance':bool(np.all(diff<=bound)),'rtol':rtol,'atol':atol,'bitwise_equal':bool(np.array_equal(a.view(np.uint64),b.view(np.uint64)))}

def save_position_and_require_reference(out,report,profile,position):
 report['positions'].append(position)
 write(out/'PROFILE.json',profile);write(out/'RUN.json',report)
 if not position['reference_comparison']['within_declared_tolerance']:
  raise ValueError('Original C_mean reference reproduction failed at step%d block%d' % (position['step'],position['block']))

def authorize(plan,digest,active):
 if plan.get('id')!=ID or plan.get('limits')!=LIMITS or plan.get('positions')!=[list(x) for x in POSITIONS] or plan.get('assets_finalized') is not True or plan.get('comparison_tolerance')!={'rtol':5e-5,'atol':1e-10}:raise ValueError('Unfrozen or altered scope')
 resource=plan.get('resource_limits',{})
 if resource!={'GPU':1,'global_max_GPU':4,'CPU_threads':2,'compute_wall_seconds':300,'external_wall_seconds':600,'new_HF_bytes':64<<20,'free_disk_bytes':10<<30} or plan.get('diagnostic',{}).get('keep_groups_by_step')!=[54,108,108,162] or plan.get('diagnostic',{}).get('bootstrap_resamples')!=200 or plan.get('diagnostic',{}).get('seed')!=42:raise ValueError('Frozen resource/diagnostic contract differs')
 u=active.get('user_continuation_authorization_20261027',{})
 if u.get('id')!='USER_RESUME_OFFLINE_IFC_20261027' or u.get('status')!='ACTIVE_CURRENT_USER_DIRECTED_SESSION' or u.get('max_simultaneously_assigned_idle_GPUs')!=4 or u.get('new_scope_registration_allowed') is not True:raise ValueError('User authority inactive')
 if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in active['experiment_authority']['current_user_authorized_scopes']):raise ValueError('Exact scope not authorized')

def verify_runtime_versions(plan):
 expected={'torch':'2.10.0+cu128','numpy':'2.4.6','safetensors':'0.8.0'}
 if plan.get('runtime_versions')!=expected:raise ValueError('Runtime version contract differs')
 actual={name:importlib.metadata.version(name) for name in expected}
 if actual!=expected:raise ValueError('Installed runtime versions differ')
 return actual

def validate_plan(plan):
 cases=plan['cases'];sources=plan['sources'];expected={(i,s,b) for i in range(24) for s,b in POSITIONS}
 if len(cases)!=96 or {(x['index'],x['step'],x['block']) for x in cases}!=expected or len(sources)!=24 or {x['index'] for x in sources}!=set(range(24)) or len({x['source_id'] for x in sources})!=24:raise ValueError('Fixed 96 case/source coverage differs')
 bank=plan.get('source_bank_metadata',{});records=bank.get('records',[])
 if bank.get('schema')!='ffnds.source.bank.v1' or bank.get('source_count')!=64 or len(records)!=64 or len({x['source_id'] for x in records})!=64 or sum(x['role']=='CALIBRATION' for x in records)!=24 or sum(x['role']=='DEVELOPMENT' for x in records)!=40:raise ValueError('Legacy 64-source role metadata differs')
 calibration={x['legacy_image_fingerprint']:x['source_id'] for x in records if x['role']=='CALIBRATION'}
 if any(calibration.get(x['source_id'])!=x['semantic_source_id'] for x in sources):raise ValueError('CAL source bank identity differs')
 for c in cases:
  if c['expert']!=('high_noise_model' if c['step']==0 else 'low_noise_model') or len(c['rows'])!=256 or len(set(c['rows']))!=256 or any(type(r)!=int or r<0 or r>=31668 for r in c['rows']):raise ValueError('Original rows/expert differ')
 if len(plan['weights'])!=4 or {(x['step'],x['block']) for x in plan['weights']}!=set(POSITIONS) or len(plan['reference_cmean'])!=4 or {(x['step'],x['block']) for x in plan['reference_cmean']}!=set(POSITIONS):raise ValueError('Missing original W2 or C_mean references')

def run(args):
 plan=json.loads(args.protocol.read_text());digest=sha(args.protocol);authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()));validate_plan(plan)
 hf=args.hf_root.resolve();weights=args.weight_hf_root.resolve();refs=args.reference_hf_root.resolve();sys.path.insert(0,str(ROOT/'tools'))
 from worker_identity import require_worker
 require_worker();out=(hf/plan['output_HF_relative']).resolve()
 if out==hf or not out.is_relative_to(hf):raise ValueError('Output escaped HF')
 out.mkdir(parents=True,exist_ok=True)
 with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest},f)
 report={'id':ID,'status':'STARTED','protocol_sha256':digest,'predebited':dict.fromkeys(LIMITS,0),'observed':{'FP32_batched_bmm':0},'positions':[],'scope':'legacy 24 exposed CAL sources x four sites x 256 rows; not new V2 collection, full160, video or quality'};monitor=None;started=time.monotonic()
 def check():
  if monitor:monitor.check()
  if shutil.disk_usage(out).free<10<<30 or sum(x.stat().st_size for x in out.rglob('*') if x.is_file() and not x.is_symlink())>64<<20:raise RuntimeError('Disk/output limit')
 def timeout(*_):raise TimeoutError('300 second compute wall')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
 try:
  report['runtime_package_versions']=verify_runtime_versions(plan)
  required={'tools/e0_ub1_source_energy_profile.py','tests/test_e0_ub1_source_energy_profile.py','tools/generate_proxy_pr2.py','tools/fit_cstatic_centered.py','tools/e0_ub1_matched_recovery.py','tools/ffnds_calibration_design.py','tools/ar2_core.py','tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py','config/worker.env','scripts/worker_wait_uuid_idle_exec.sh'}
  if not required <= {x['path'] for x in plan['repo_sources']}:raise ValueError('Required source bindings missing')
  for x in plan['repo_sources']:bound(ROOT,x)
  for x in plan['hf_assets']:bound(hf,x)
  for x in plan['weight_assets']:bound(weights,x)
  for x in plan['reference_cmean']:bound(refs,x)
  paths={x['path']:x['sha256'] for x in plan['hf_assets']};wpaths={x['path']:x['sha256'] for x in plan['weight_assets']};source={x['index']:x for x in plan['sources']}
  from e0_ub1_matched_recovery import validate_input_scope
  for c in plan['cases']:
   if paths.get(c['path'])!=c['sha256'] or paths.get(c['input_report'])!=c['input_report_sha256']:raise ValueError('Payload/input binding differs')
   validate_input_scope(c,json.loads((hf/c['input_report']).read_text()),source[c['index']])
  for w in plan['weights']:
   if wpaths.get(w['path'])!=w['sha256']:raise ValueError('Shard binding differs')
  check();snapshot=out/'EXECUTED_SOURCES';snapshot.mkdir();shutil.copyfile(args.protocol,snapshot/'protocol.json')
  for x in plan['repo_sources']:
   p=snapshot/x['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/x['path'],p)
  import numpy as np
  import torch
  from safetensors.torch import load_file
  from startup_gpu_reservation import reserve_before_assets
  from generate_proxy_pr2 import contribution_scores
  from ffnds_calibration_design import diagnose
  from ar2_core import top_groups
  reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt
  if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('Exactly one idle A100 required')
  torch.set_num_threads(2);torch.set_grad_enabled(False);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
  report['runtime']={'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,'device_name':torch.cuda.get_device_name(0),'float32_matmul_precision':torch.get_float32_matmul_precision(),'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32}
  try:os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
  except AttributeError:pass
  def tensor_sha(t):return hashlib.sha256(memoryview(t.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()
  profile={'schema':'ffnds.calibration.source_statistics.v2','evidence_kind':'ACTUAL_NATIVE_OBSERVATION','records':[]};device=torch.device('cuda')
  for step,block in POSITIONS:
   wref=next(x for x in plan['weights'] if (x['step'],x['block'])==(step,block));expert=wref['expert'];raw=torch.load(weights/wref['path'],map_location='cpu',weights_only=True,mmap=True)[expert][f'blocks.{block}.ffn.2'];w=raw['weight_qdq']
   if tuple(w.shape)!=(5120,13824) or tensor_sha(w)!=wref['weight_tensor_sha256'] or not w.is_floating_point() or not bool(torch.isfinite(w).all()):raise ValueError('Original W2 differs')
   storage_dtype=str(w.dtype);w=w.to(device);pooled=np.zeros(216,dtype=np.float64);begin=len(profile['records'])
   for index in range(24):
    check();c=next(x for x in plan['cases'] if (x['index'],x['step'],x['block'])==(index,step,block));d=load_file(str(hf/c['path']),device='cpu')
    if set(d)!={'V1','V2','Y','gate','b2'} or tuple(d['V2'].shape)!=(256,13824) or d['V2'].dtype!=torch.bfloat16 or tuple(d['gate'].shape)!=(1,5120) or d['gate'].dtype!=torch.bfloat16 or tuple(d['b2'].shape)!=(5120,) or d['b2'].dtype!=torch.bfloat16 or not bool(torch.isfinite(d['V2']).all()) or not bool(torch.isfinite(d['gate']).all()) or not bool(torch.isfinite(d['b2']).all()) or tensor_sha(d['gate'])!=wref['gate_tensor_sha256'] or tensor_sha(d['b2'])!=wref['b2_tensor_sha256']:raise ValueError('Original operand/gate differs')
    a=d['V2'].to(device);gate=d['gate'].to(device);reserve(report['predebited']);write(out/'RUN.json',report)
    original=torch.bmm;before=report['observed']['FP32_batched_bmm'];torch.bmm=instrument_bmm(original,report['observed'])
    try:q=contribution_scores(a,w,gate)
    finally:torch.bmm=original
    if report['observed']['FP32_batched_bmm']-before!=27 or q.shape!=(256,216) or not np.isfinite(q).all():raise ValueError('Original 27-BMM result differs')
    vector=q.mean(0);pooled+=vector/24.;src=source[index];profile['records'].append({'source_id':src.get('semantic_source_id',src['source_id']),'image_id':src['source_id'],'index':index,'seed':src['seed'],'step':step,'block':block,'source_state':'teacher_input','statistic_semantics':'native_gate_times_W2_group_contribution_l2_squared','normalization':'mean_over_256_saved_rows','group_energy':vector.tolist(),'payload_sha256':c['sha256']});write(out/'PROFILE.json',profile);write(out/'RUN.json',report);del d,a,gate,q,vector
   ref=next(x for x in plan['reference_cmean'] if (x['step'],x['block'])==(step,block))
   with np.load(refs/ref['path'],allow_pickle=False) as f:
    reference=f['C_mean'];comparison=reference_compare(pooled,reference);keep=54 if step==0 else 162
    actual_support=set(top_groups(pooled,keep).tolist());reference_support=set(top_groups(reference,keep).tolist());comparison['support_count']=keep;comparison['support_equal_to_original_C_mean']=actual_support==reference_support;comparison['support_intersection']=len(actual_support & reference_support)
    if 'original_ub1_groups' in ref:
     expected=set(ref['original_ub1_groups']);comparison['support_equal_to_frozen_ub1']=actual_support==expected
   save_position_and_require_reference(out,report,profile,{'step':step,'block':block,'original_W2_stored_dtype':storage_dtype,'reference_comparison':comparison,'source_records':len(profile['records'])-begin})
   del raw,w,pooled
  if report['predebited']!=LIMITS or report['observed']['FP32_batched_bmm']!=LIMITS['FP32_batched_bmm']:raise RuntimeError('Incomplete budget')
  diagnostic=diagnose(profile,keep_groups_by_step=[54,108,108,162],bootstrap=200,seed=42);write(out/'DIAGNOSTIC.json',diagnostic);report.update(status='COMPLETE_LEGACY_CAL24_SOURCE_PROFILE',elapsed_seconds=time.monotonic()-started);check()
 except BaseException as e:report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
 finally:
  original_error=sys.exc_info()[0] is not None;signal.alarm(0);monitor_error=None
  if monitor:
   try:
    monitor.close();report['assigned_gpu_resources']=monitor.report()
    if report['assigned_gpu_resources'].get('violations'):report['status']='FAILED_GPU_MONITOR';monitor_error=RuntimeError('GPU monitor violations')
   except BaseException as e:report['status']='FAILED_GPU_MONITOR';monitor_error=e
  write(out/'RUN.json',report)
  if monitor_error is not None and not original_error:raise monitor_error

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--hf-root',type=Path,required=True);p.add_argument('--weight-hf-root',type=Path,required=True);p.add_argument('--reference-hf-root',type=Path,required=True);run(p.parse_args())
