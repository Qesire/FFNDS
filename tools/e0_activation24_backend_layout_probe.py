"""Bounded synthetic cuSPARSELt algorithm/layout diagnostic, separate from C1."""
import argparse, hashlib, json, os, shutil, signal, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ID='E0_ACTIVATION24_BACKEND_LAYOUT_PROBE_20261027'
SHAPES=((31668,5120,13824),(31668,13824,5120))
LIMITS={'random_tensor_init':4,'top2_select':2,'compress':2,'search_api':8,'sparse_gemm':128,'dense_gemm':16,'model_calls':0,'video_generations':0,'fits':0,'official_scores':0}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def charge(counts,key):
 if key not in LIMITS or counts.get(key,0)>=LIMITS[key]:raise RuntimeError('Budget exceeded before '+key)
 counts[key]=counts.get(key,0)+1

def authorize(plan,digest,active):
 if plan.get('id')!=ID or plan.get('limits')!=LIMITS or plan.get('shapes')!=[list(x) for x in SHAPES] or plan.get('warmups')!=2 or plan.get('repetitions')!=5 or plan.get('wall_seconds')!=300 or not plan.get('assets_finalized'):raise ValueError('Scope not frozen')
 if not any(v.get('id')==ID and v.get('protocol_sha256')==digest and v.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') for v in active['experiment_authority']['current_user_authorized_scopes']):raise ValueError('Exact protocol authorization missing')

def search_parameters(result):
 if not isinstance(result,tuple) or len(result)!=4 or any(type(x) is not int for x in result):raise ValueError('Unexpected mm_search tuple')
 alg,split,mode,max_id=result
 if alg<0 or split<1 or max_id<0 or alg>max_id:raise ValueError('Invalid search result')
 return {'alg_id':alg,'split_k':split,'split_k_mode':mode,'max_alg_id':max_id}

def normalize_output(raw,transpose,m):
 if transpose:raw=raw.t()
 return raw[:m].contiguous()

def write(path,obj):
 tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:
  f.write(json.dumps(obj,allow_nan=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def run(args):
 plan=json.loads(args.protocol.read_text());digest=sha(args.protocol);authorize(plan,digest,json.loads((ROOT/'research/ACTIVE_RESEARCH.json').read_text()))
 sys.path.insert(0,str(ROOT/'tools'))
 from e0_activation24_spmm_probe import select_top2,padded_rows,verify,verify_loaded_backends
 from worker_identity import require_worker
 required={'tools/e0_activation24_backend_layout_probe.py','tests/test_e0_activation24_backend_layout_probe.py','tools/e0_activation24_spmm_probe.py','tools/worker_identity.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py'}
 paths={v['path'] for v in plan['bindings']}
 if not required<=paths or not any(v.endswith('libtorch_cuda.so') for v in paths) or not any(v.endswith('libcusparseLt.so.0') for v in paths) or not any(v.endswith('torch/sparse/semi_structured.py') for v in paths) or not any(v.endswith('torch/sparse/_semi_structured_ops.py') for v in paths) or not any(v.endswith('torch/sparse/_semi_structured_conversions.py') for v in paths):raise ValueError('Required bindings missing')
 verify(plan['bindings']);require_worker();hf=args.hf_root.resolve();out=(hf/plan['output_HF_relative']).resolve()
 if out==hf or not out.is_relative_to(hf):raise ValueError('Output escaped HF')
 out.mkdir(parents=True,exist_ok=True)
 with (out/'ATTEMPT.json').open('x') as f:json.dump({'id':ID,'protocol_sha256':digest},f)
 report={'id':ID,'status':'STARTED','protocol_sha256':digest,'counts':dict.fromkeys(LIMITS,0),'search_internal_kernel_count':'UNKNOWN','hard_kill_note':'After external hard kill explicit call completion since last persisted checkpoint is UNKNOWN.','cases':[],'scope':'synthetic only; not model speed or video quality'}
 monitor=None;start=time.monotonic()
 def storage():
  if shutil.disk_usage(out).free<10<<30:raise RuntimeError('Insufficient HF space')
  if sum(x.stat().st_size for x in out.rglob('*') if x.is_file() and not x.is_symlink())>32<<20:raise RuntimeError('Output exceeds 32MiB')
 def timeout(*_):raise TimeoutError('Fixed 300 second limit')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
 try:
  storage();snap=out/'EXECUTED_SOURCES';snap.mkdir();shutil.copyfile(args.protocol,snap/'protocol.json')
  for i,v in enumerate(plan['bindings']):
   p=(ROOT/v['path']).resolve()
   if p.suffix not in ('.py','.sh'):continue
   if p.stat().st_size>1<<20:raise ValueError('Snapshot too large')
   shutil.copyfile(p,snap/(str(i)+'_'+p.name));storage()
  import torch
  import torch.nn.functional as F
  import torch.sparse.semi_structured as sm
  from startup_gpu_reservation import reserve_before_assets
  bound={str((ROOT/v['path']).resolve()) for v in plan['bindings']}
  if str(Path(sm.__file__).resolve()) not in bound:raise RuntimeError('Imported sparse module differs')
  reservation,monitor,receipt=reserve_before_assets(torch);report['reservation']=receipt
  if torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0) or not torch.backends.cusparselt.is_available() or sm.SparseSemiStructuredTensor._FORCE_CUTLASS:raise RuntimeError('Need A100 default cuSPARSELt')
  torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
  report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'cusparselt':torch.backends.cusparselt.version(),'gpu':torch.cuda.get_device_name(0),'bf16_reduced_precision':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
  device=torch.device('cuda')
  def rand(shape,seed):
   charge(report['counts'],'random_tensor_init');g=torch.Generator(device=device);g.manual_seed(seed);return torch.randn(shape,device=device,dtype=torch.bfloat16,generator=g)
  def event_start():torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);a.record();return a,time.monotonic()
  def event_end(a,wall):
   b=torch.cuda.Event(enable_timing=True);b.record();torch.cuda.synchronize();return {'cuda_ms':a.elapsed_time(b),'wall_ms':(time.monotonic()-wall)*1000}
  for idx,(m,k,n) in enumerate(SHAPES):
   monitor.check();storage();mp=padded_rows(m);a=rand((m,k),20261027+idx*2);w=rand((n,k),20261028+idx*2)
   charge(report['counts'],'top2_select');masked=select_top2(a);charge(report['counts'],'compress');packed=torch.sparse.to_sparse_semi_structured(F.pad(masked,(0,0,0,mp-m)))
   if not isinstance(packed,sm.SparseSemiStructuredTensorCUSPARSELT):raise RuntimeError('Wrong sparse backend')
   if idx==0:report['loaded_backends']=verify_loaded_backends(plan);write(out/'RUN.json',report)
   charge(report['counts'],'dense_gemm');ref=F.linear(masked,w)
   dense_times=[]
   for it in range(7):
    monitor.check();ev,wall=event_start();charge(report['counts'],'dense_gemm');value=F.linear(a,w);t=event_end(ev,wall);del value
    if it>=2:dense_times.append(t)
   ev,wall=event_start();before=torch.cuda.memory_allocated();copy=w.t().contiguous();copy_time=event_end(ev,wall);copy_time.update({'bytes':copy.numel()*copy.element_size(),'allocated_before':before,'allocated_after':torch.cuda.memory_allocated()})
   case={'M':m,'M_padded':mp,'K':k,'N':n,'dense':dense_times,'weight_copy':copy_time,'settings':[]}
   report['cases'].append(case);write(out/'RUN.json',report)
   for layout,b in [('view',w.t()),('contiguous',copy)]:
    for transpose in (False,True):
     monitor.check();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();search_before=torch.cuda.memory_allocated();charge(report['counts'],'search_api')
     report['inflight_search']={'M':m,'K':k,'N':n,'B_layout':layout,'transpose_result':transpose,'started_unix_seconds':time.time(),'internal_kernel_count':'UNKNOWN'}
     write(out/'RUN.json',report)
     ev,wall=event_start();rawparams=torch._C._cusparselt.mm_search(packed.packed,b,None,None,None,transpose);search_time=event_end(ev,wall);search_time.update({'allocated_before':search_before,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()});param=search_parameters(rawparams)
     item={'B_layout':layout,'B_stride':list(b.stride()),'transpose_result':transpose,'search':{'parameters':param,'time':search_time,'internal_kernel_count':'UNKNOWN'},'variants':{}}
     case['settings'].append(item);report.pop('inflight_search');write(out/'RUN.json',report)
     for label,params in [('default',{'alg_id':0,'split_k':1,'split_k_mode':-1}),('searched',param)]:
      metrics=[];correct=None
      for it in range(8):
       monitor.check();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();ev,wall=event_start();charge(report['counts'],'sparse_gemm')
       raw=torch._cslt_sparse_mm(packed.packed,b,transpose_result=transpose,alg_id=params['alg_id'],split_k=params['split_k'],split_k_mode=params['split_k_mode'])
       dispatch_ms=(time.monotonic()-wall)*1000
       middle=torch.cuda.Event(enable_timing=True);middle.record();value=normalize_output(raw,transpose,m);tm=event_end(ev,wall);tm['raw_spmm_cuda_ms']=ev.elapsed_time(middle);tm['raw_spmm_dispatch_wall_ms']=dispatch_ms;tm['normalization_cuda_ms']=tm['cuda_ms']-tm['raw_spmm_cuda_ms'];tm['allocated_before']=before;tm['peak_allocated']=torch.cuda.max_memory_allocated();tm['peak_reserved']=torch.cuda.max_memory_reserved()
       if it==0:
        diff=value.float()-ref.float();den=ref.float().norm().item();err=diff.norm().item()/den if den else None
        if err is None or not bool(torch.isfinite(value).all()) or err>plan['relative_l2_tolerance']:raise RuntimeError('Sparse correctness failed')
        correct={'relative_l2':err,'max_abs':diff.abs().max().item(),'raw_stride':list(raw.stride()),'normalized_contiguous':value.is_contiguous()};del diff
       if it>=3:metrics.append(tm)
       del raw,value
      item['variants'][label]={'parameters':params,'correctness':correct,'timing':metrics}
     write(out/'RUN.json',report);storage()
   write(out/'RUN.json',report);storage();del a,w,masked,packed,ref,copy
  if report['counts']!=LIMITS:raise RuntimeError('Incomplete count ledger')
  report.update(status='COMPLETE_SYNTHETIC_ONLY',elapsed_seconds=time.monotonic()-start)
 except BaseException as e:
  report.update(status='FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
 finally:
  signal.alarm(0)
  if monitor:
   monitor.close();report['assigned_gpu_resources']=monitor.report()
   if report['assigned_gpu_resources']['violations']:report['status']='FAILED_GPU_MONITOR'
  write(out/'RUN.json',report);storage()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--hf-root',type=Path,required=True);run(p.parse_args())
