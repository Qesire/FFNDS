"""Bounded synthetic 2:4 CUTLASS versus cuSPARSELt backend probe; no model execution."""
import argparse,hashlib,json,os,shutil,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='E0_ACTIVATION24_CUTLASS_BACKEND_PROBE_20261027'
SHAPES=((31668,5120,13824),(31668,13824,5120))
LIMITS={'random_tensor_init':4,'stable_select':2,'triton_select':30,'cutlass_compress':30,'cusparselt_compress':30,'cutlass_reconstruct':2,'dense_gemm':16,'cutlass_sparse_gemm':30,'cusparselt_sparse_gemm':30,'search_api':0,'model_calls':0,'fits':0,'video_generations':0}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def charge(counts,cost):
 if any(k not in LIMITS or type(v)!=int or v<0 or counts[k]+v>LIMITS[k] for k,v in cost.items()):raise ValueError('Budget exceeded before dispatch')
 for k,v in cost.items():counts[k]+=v

def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,allow_nan=False,indent=2)+'\n');os.replace(t,p)

def authorize(plan,digest,active):
 if plan.get('id')!=ID or plan.get('limits')!=LIMITS or plan.get('shapes')!=[list(x) for x in SHAPES] or not plan.get('assets_finalized') or plan.get('algorithm')!={'alg_id':1,'split_k':1,'split_k_mode':1,'transpose_result':False}:raise ValueError('Unfrozen or changed scope')
 resources=plan.get('resource_limits',{})
 if (plan.get('warmups')!=2 or plan.get('repetitions')!=5 or plan.get('wall_seconds')!=480 or resources.get('simultaneously_assigned_idle_GPUs')!=1 or resources.get('global_user_GPU_max')!=4 or resources.get('cpu_threads')!=2 or resources.get('HF_free_min_bytes')!=10<<30 or resources.get('HF_output_max_bytes')!=100<<20 or resources.get('new_HF_bytes')!=100<<20 or resources.get('compute_wall_seconds')!=480 or resources.get('external_total_wall_seconds')!=600 or plan.get('search_internal_kernel_count')!='NOT_CALLED' or plan.get('search_cost')!='NOT_CALLED'):raise ValueError('Resource or sampling contract differs')
 user=active.get('user_continuation_authorization_20261027',{})
 if user.get('id')!='USER_RESUME_OFFLINE_IFC_20261027' or user.get('status')!='ACTIVE_CURRENT_USER_DIRECTED_SESSION' or user.get('new_scope_registration_allowed') is not True or user.get('max_simultaneously_assigned_idle_GPUs')!=4:raise ValueError('Current user continuation authorization inactive or changed')
 if not any(x.get('id')==ID and x.get('protocol_sha256')==digest and x.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for x in active['experiment_authority']['current_user_authorized_scopes']):raise ValueError('Exact scope not registered')

def finish_monitor(report,monitor):
 try:
  monitor.close();result=monitor.report();report['assigned_gpu_resources']=result
  if result.get('violations'):
   report['status']='FAILED_GPU_MONITOR';return RuntimeError('Assigned GPU monitor reported violations')
 except BaseException as e:
  report['status']='FAILED_GPU_MONITOR';report['monitor_error']={'type':type(e).__name__,'message':str(e)};return e
 return None

def raise_monitor_if_unmasked(error,original_error_active):
 if error is not None and not original_error_active:raise error

def tie_rank(values):
 if len(values)!=4:raise ValueError('Expected four values')
 return [sum(abs(values[j])>abs(values[i]) or (abs(values[j])==abs(values[i]) and j<i) for j in range(4)) for i in range(4)]

def run(args):
 plan=json.loads(args.protocol.read_text());digest=sha(args.protocol);authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()));sys.path.insert(0,str(ROOT/'tools'))
 from e0_activation24_spmm_probe import select_top2,padded_rows,verify,verify_loaded_backends
 from worker_identity import require_worker
 required={'tools/e0_activation24_cutlass_backend_probe.py','tests/test_e0_activation24_cutlass_backend_probe.py','tools/e0_activation24_online_components_probe.py','tools/e0_activation24_spmm_probe.py','tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py','config/worker.env','scripts/worker_wait_uuid_idle_exec.sh','.venv/lib/python3.11/site-packages/torch/sparse/semi_structured.py','.venv/lib/python3.11/site-packages/torch/sparse/_semi_structured_ops.py','.venv/lib/python3.11/site-packages/torch/sparse/_semi_structured_conversions.py','.venv/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so','.venv/lib/python3.11/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0','.venv/lib/python3.11/site-packages/triton/__init__.py','.venv/lib/python3.11/site-packages/triton/runtime/jit.py','.venv/lib/python3.11/site-packages/triton/language/core.py','.venv/lib/python3.11/site-packages/triton/compiler/compiler.py','.venv/lib/python3.11/site-packages/triton/_C/libtriton.so'}
 names={x['path'] for x in plan['bindings']}
 if not required<=names:raise ValueError('Missing bound backend or launcher source')
 verify(plan['bindings']);require_worker();hf=args.hf_root.resolve()
 for item in plan['prior_runs']:
  source=(hf/item['HF_relative']).resolve()
  if not source.is_relative_to(hf) or not source.is_file() or sha(source)!=item['sha256']:raise ValueError('Historical RUN differs')
 out=(hf/plan['output_HF_relative']).resolve()
 if out==hf or not out.is_relative_to(hf):raise ValueError('Output escaped HF')
 out.mkdir(parents=True,exist_ok=True)
 with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest},f)
 report={'id':ID,'status':'STARTED','protocol_sha256':digest,'counts':dict.fromkeys(LIMITS,0),'cases':[],'search_internal_kernel_count':'NOT_CALLED','scope':'synthetic BF16, no model or video','hard_kill_note':'in-flight completion after uncatchable kill unknown'};monitor=None;started=time.monotonic()
 def reserve(cost):charge(report['counts'],cost);write(out/'RUN.json',report)
 def check():
  if monitor:monitor.check()
  if shutil.disk_usage(out).free<plan['resource_limits']['HF_free_min_bytes'] or sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())>plan['resource_limits']['HF_output_max_bytes']:raise RuntimeError('Storage limit')
 def alarm(*_):raise TimeoutError('480s compute limit')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(plan['resource_limits']['compute_wall_seconds'])
 try:
  snap=out/'EXECUTED_SOURCES';snap.mkdir();shutil.copyfile(args.protocol,snap/'protocol.json')
  for i,x in enumerate(plan['bindings']):
   p=(ROOT/x['path']).resolve()
   if p.suffix in ('.py','.sh'):
    if p.stat().st_size>1<<20:raise ValueError('Snapshot too large')
    shutil.copyfile(p,snap/(str(i)+'_'+p.name))
  os.environ['TRITON_CACHE_DIR']=str(out/'triton_cache')
  import torch
  import torch.nn.functional as F
  import torch.sparse.semi_structured as sm
  import triton
  import triton.language as tl
  from startup_gpu_reservation import reserve_before_assets
  bound={str((ROOT/x['path']).resolve()) for x in plan['bindings']}
  if str(Path(triton.__file__).resolve()) not in bound or str(Path(sm.__file__).resolve()) not in bound or str(Path(sys.modules['torch.sparse._semi_structured_ops'].__file__).resolve()) not in bound or str(Path(sys.modules['torch.sparse._semi_structured_conversions'].__file__).resolve()) not in bound or triton.__version__!=plan['triton_version']:raise ValueError('Imported package identity differs')
  reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt
  if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0) or not torch.backends.cusparselt.is_available() or sm.SparseSemiStructuredTensor._FORCE_CUTLASS:raise ValueError('A100 cuSPARSELt required')
  torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
  report['runtime']={'torch':torch.__version__,'triton':triton.__version__,'cuda':torch.version.cuda,'cusparselt':torch.backends.cusparselt.version(),'gpu':torch.cuda.get_device_name(0)}
  @triton.jit
  def select_kernel(src,dst,M:tl.constexpr,K:tl.constexpr,MP:tl.constexpr,B:tl.constexpr):
   g=tl.program_id(0)*B+tl.arange(0,B);groups=K//4;row=g//groups;col=(g%groups)*4;valid=g<MP*groups;real=valid&(row<M)
   x0=tl.load(src+row*K+col+0,mask=real,other=0).to(tl.float32);x1=tl.load(src+row*K+col+1,mask=real,other=0).to(tl.float32);x2=tl.load(src+row*K+col+2,mask=real,other=0).to(tl.float32);x3=tl.load(src+row*K+col+3,mask=real,other=0).to(tl.float32)
   a0=tl.abs(x0);a1=tl.abs(x1);a2=tl.abs(x2);a3=tl.abs(x3)
   r0=(a1>a0).to(tl.int32)+(a2>a0).to(tl.int32)+(a3>a0).to(tl.int32)
   r1=(a0>=a1).to(tl.int32)+(a2>a1).to(tl.int32)+(a3>a1).to(tl.int32)
   r2=(a0>=a2).to(tl.int32)+(a1>=a2).to(tl.int32)+(a3>a2).to(tl.int32)
   r3=(a0>=a3).to(tl.int32)+(a1>=a3).to(tl.int32)+(a2>=a3).to(tl.int32)
   tl.store(dst+row*K+col+0,tl.where(real&(r0<2),x0,0.),mask=valid);tl.store(dst+row*K+col+1,tl.where(real&(r1<2),x1,0.),mask=valid);tl.store(dst+row*K+col+2,tl.where(real&(r2<2),x2,0.),mask=valid);tl.store(dst+row*K+col+3,tl.where(real&(r3<2),x3,0.),mask=valid)
  def rnd(shape,seed):
   reserve({'random_tensor_init':1});g=torch.Generator(device='cuda');g.manual_seed(seed);return torch.randn(shape,dtype=torch.bfloat16,device='cuda',generator=g)
  def fast(a,mp):
   out0=torch.empty((mp,a.shape[1]),dtype=a.dtype,device=a.device);select_kernel[(triton.cdiv(mp*a.shape[1]//4,256),)](a,out0,a.shape[0],a.shape[1],mp,256);return out0
  def cs_compress(a):
   x=sm.SparseSemiStructuredTensorCUSPARSELT.from_dense(a)
   if not isinstance(x,sm.SparseSemiStructuredTensorCUSPARSELT):raise RuntimeError('Wrong sparse backend')
   return x
  def cutlass_compress(a):return sm.SparseSemiStructuredTensorCUTLASS.from_dense(a)
  def cutlass_spmm(packed,b,m):return torch.mm(packed,b)[:m].contiguous()
  def cs_spmm(packed,b,m):return torch._cslt_sparse_mm(packed.packed,b,transpose_result=False,alg_id=1,split_k=1,split_k_mode=1)[:m].contiguous()
  def timed(cost,fn):
   check();reserve(cost);torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);wall=time.monotonic();start.record();value=fn();end.record();torch.cuda.synchronize();return value,{'cuda_ms':start.elapsed_time(end),'wall_ms':1000*(time.monotonic()-wall)}
  for i,(m,k,n) in enumerate(SHAPES):
   a=rnd((m,k),20261027+2*i);w=rnd((n,k),20261028+2*i);mp=padded_rows(m);b=w.t()
   reserve({'triton_select':1});torch.cuda.synchronize();setup_started=time.monotonic();masked=fast(a,mp);torch.cuda.synchronize();setup_seconds=time.monotonic()-setup_started
   reserve({'stable_select':1});stable=F.pad(select_top2(a),(0,0,0,mp-m))
   if not torch.equal(masked.view(torch.uint8),stable.view(torch.uint8)):raise RuntimeError('Triton selector differs bytewise from original stable top2')
   del stable
   reserve({'cusparselt_compress':1});cs=cs_compress(masked)
   if i==0:
    report['loaded_backends']=verify_loaded_backends(plan)
    report['cutlass_dispatch_library']='mapped and SHA-bound libtorch_cuda.so; installed PyTorch CUTLASS dispatch'
    mapped=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
     fields=line.split(maxsplit=5)
     if len(fields)!=6:continue
     raw=fields[5];deleted=raw.endswith(' (deleted)');path=raw[:-10] if deleted else raw
     if path.endswith('/libtriton.so'):
      if deleted:raise ValueError('Deleted Triton library mapping')
      mapped.add(str(Path(path).resolve()))
    expected={x for x in bound if x.endswith('/triton/_C/libtriton.so')}
    if len(expected)!=1 or mapped!=expected:raise ValueError('Triton loaded library differs')
    report['triton_library_binding_verified']=True
   reserve({'cutlass_compress':1});cutlass=cutlass_compress(masked)
   reserve({'cutlass_reconstruct':1});reconstructed=cutlass.to_dense()
   if reconstructed.dtype!=masked.dtype or reconstructed.shape!=masked.shape or not torch.equal(reconstructed,masked):raise RuntimeError('CUTLASS reconstruction differs from masked BF16 values')
   del reconstructed
   reserve({'dense_gemm':1});ref=F.linear(masked[:m],w);correctness={}
   for name,packed,fn,key in [('cutlass',cutlass,cutlass_spmm,'cutlass_sparse_gemm'),('cusparselt',cs,cs_spmm,'cusparselt_sparse_gemm')]:
    reserve({key:1});value=fn(packed,b,m);torch.cuda.synchronize();diff=value.float()-ref.float();den=ref.float().norm().item();err=diff.norm().item()/den if den else None
    if err is None or not bool(torch.isfinite(value).all()) or err>plan['relative_l2_tolerance'] or not value.is_contiguous() or tuple(value.shape)!=(m,n):raise RuntimeError(name+' functional correctness failed')
    correctness[name]={'relative_l2':err,'max_abs':diff.abs().max().item(),'output_shape':[m,n],'output_contiguous':True};del diff,value
   del ref
   case={'M':m,'M_padded':mp,'K':k,'N':n,'algorithm':plan['algorithm'],'triton_stable_mask_byte_equal':True,'cutlass_reconstruction_bf16_value_equal':True,'cutlass_packed_shape':list(cutlass.packed.shape),'cutlass_meta_shape':list(cutlass.meta.shape),'cutlass_meta_dtype':str(cutlass.meta.dtype),'triton_initial_dispatch_including_compilation_seconds':setup_seconds,'correctness':correctness,'timings':{}};report['cases'].append(case);write(out/'RUN.json',report)
   operations=[('dense',{'dense_gemm':1},lambda:F.linear(a,w)),('cutlass_compress',{'cutlass_compress':1},lambda:cutlass_compress(masked)),('cusparselt_compress',{'cusparselt_compress':1},lambda:cs_compress(masked)),('cutlass_prepacked_functional',{'cutlass_sparse_gemm':1},lambda:cutlass_spmm(cutlass,b,m)),('cusparselt_prepacked_functional',{'cusparselt_sparse_gemm':1},lambda:cs_spmm(cs,b,m)),('cutlass_online',{'triton_select':1,'cutlass_compress':1,'cutlass_sparse_gemm':1},lambda:cutlass_spmm(cutlass_compress(fast(a,mp)),b,m)),('cusparselt_online',{'triton_select':1,'cusparselt_compress':1,'cusparselt_sparse_gemm':1},lambda:cs_spmm(cs_compress(fast(a,mp)),b,m))]
   for name,cost,fn in operations:
    vals=[]
    for j in range(7):
     x,tm=timed(cost,fn);del x
     if j>=2:vals.append(tm)
    case['timings'][name]=vals;write(out/'RUN.json',report)
   del a,w,b,masked,cutlass,cs
  if report['counts']!=LIMITS:raise RuntimeError('Incomplete ledger')
  report.update(status='COMPLETE_SYNTHETIC_DUAL_BACKEND_ONLY',elapsed_seconds=time.monotonic()-started)
 except BaseException as e:report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
 finally:
  original_error_active=sys.exc_info()[0] is not None
  signal.alarm(0)
  monitor_error=finish_monitor(report,monitor) if monitor else None
  write(out/'RUN.json',report)
  raise_monitor_if_unmasked(monitor_error,original_error_active)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--hf-root',type=Path,required=True);run(p.parse_args())
