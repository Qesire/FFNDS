"""Pending-review Wan FFN native SharQ producer.

Import and synthetic tests do not touch torch, tensors, models or GPU. A future
CPU asset producer must first create and independently close the exact paired
bundle, including original BF16 weights/bias and gate plus NVFP4 checkpoint
operands. An absent binding is deliberately not executable.
"""
import argparse, hashlib, importlib, importlib.util, json, os, resource, signal, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import e0_sharq_wan_ffn_ops as ops
import e0_teal_llama3_tokenizer_byte_bridge as base
import e0_teal_llama3_author_checkpoint_convert as support
import e0_sharq_wan_ffn_assets as asset_parent
import nvfp4_real_ffn_probe as decoder
from types import SimpleNamespace
ID='E0_SHARQ_SM120_WAN_FFN_PAIR_PROBE_V2_20261030'
HF=Path('/home/xiang/.cache/huggingface')
OUTPUT='ffnds-derived/e0_sharq_sm120_wan_ffn_pair_probe_v2_20261030/attempt1/worker/results'
RES={'GPU':1,'CPU_threads':4,'wall_seconds':3600,'outer_seconds':3720,'new_HF_bytes':512<<20,'host_RSS_max_bytes':16<<30,'GPU_reserved_max_bytes':72<<30,'bundle_hash_bytes_max':1073741824,'automatic_retry':False}
DESIGN={'experts':['high','low'],'M':[32,31668],'source':911,'block':20,'teacher_steps':[1,2],'fullM':'repeat input rows i mod 32; cost-only, not real full native state',
 'arms':list(ops.ARMS),'warmup':2,'measured':5,'correctness':1,'uninstrumented_warmup':2,'uninstrumented_measured':5,'group32':'same raw BF16 W shared by dense32 and SharQ',
 'dense16':'separate NVFP4 checkpoint packed weight and fixed checkpoint global A scale',
 'bfloat16':'original BF16 master W1/W2+b1/b2 from matching expert/block','gate':'real matched gate from same source/step/block',
 'Q2':'actual arm W1+bias+GELU output','sparse_residual':'R=X-dequant(Q_sparse(mask*X)); dense residual GEMM spans full K and N',
 'bundle_full_hash_bytes':730509120,'bundle_io':'one full byte hash; subsequent mmap/page-cache physical reads UNKNOWN; H2D logical bytes completed after CUDA synchronization',
 'neutral_rel_l2_max':0.03,'same_encoding_linear_rel_l2_max':0.03,'reference_scope':'M32 debug sparse logical payload compared to actual E-path; debug shares author selector and is not independent selector proof',
 'reference_max_calls':{'native_reprepare':12,'debug_sparse':4,'duplicate_sparse_scales':4,'fp32_reference_gemm':16,'linear_comparisons':12},
 'retry_identity':'new independently reviewed scope after actual V1 EXIT1; all 240 FFN calls are newly charged; V1 partial work remains separate',
 'author_scale_layout':'author dense SF allocates (rows//128+1)*128*(K//16) bytes; drop only the unused trailing aligned-row tile; LightX uses ceil(rows/128) tile count',
 'activation_scale_identity':'compare exact bytes only at logical row/group coordinates; allocator padding in dense activation SF is unspecified and excluded',
 'scope':'FFN operator only; single exposed development source; no video or system acceptance'}
SRC=HF/'ffnds-derived/sharq_sm120_author_build_20261030/source';ELF=HF/'ffnds-derived/sharq_sm120_author_build_20261030/build_sm120a/sharq_ops.so';ENV=HF/'ffnds-derived/sharq_sm120_author_build_20261030/venv/bin/python'
AUTHOR_MODULES=('model/quantize.py','model/hif4.py','sharq_loader.py','kernels/src/bindings.cpp','kernels/include/reorder.cuh','kernels/src/reorder.cu')
SELF='tools/e0_sharq_wan_ffn_probe_v2.py';TEST='tests/test_e0_sharq_wan_ffn_probe_v2.py'
SOURCES=(SELF,TEST,'tools/e0_sharq_wan_ffn_probe.py','tests/test_e0_sharq_wan_ffn_probe.py','config/e0_sharq_sm120_wan_ffn_pair_probe_20261030_pending_review_v2.json','tools/e0_sharq_wan_ffn_ops.py','tests/test_e0_sharq_wan_ffn_ops.py','tools/nvfp4_real_ffn_probe.py','tools/e0_sharq_wan_ffn_assets.py','tools/e0_sharq_wan_ffn_asset_plan.py','tools/e0_sharq_wan_ffn_asset_extract.py','tools/e0_teal_llama3_tokenizer_byte_bridge.py','tools/e0_teal_llama3_author_checkpoint_convert.py','tools/worker_identity.py','tools/durable_worker_job.py','tools/startup_gpu_reservation.py','tools/assigned_gpu_resources.py','scripts/worker_teal_gpu0_uuid_exec.sh','research/20261029/SHARQ_SM120_CPU_ENGINEERING_BUILD_20261030.json','research/20261029/SHARQ_CPU_AUTHOR_SOURCE_IMPORT_ENGINEERING_20261030.json','research/20261029/NATIVE_NVFP4_REAL_FFN_PROBE_RESULTS_20261029.json')
FAILED=HF/'ffnds-derived/e0_sharq_sm120_wan_ffn_pair_probe_20261030/attempt1/worker'
LIMITS={}
for _arm in ops.ARMS:
 for _key,_n in ops.expected_calls(_arm).items():LIMITS[_key]=LIMITS.get(_key,0)+60*_n
LIMITS.update(static_group32_weight_quant=4,checkpoint_scale_validation=4,fullM_row_repeat=2,correctness_comparisons=16,bundle_hashes=1,reference_linear=12,reference_prepare=12,reference_debug_sparse=4,reference_duplicate_scales=4,reference_decode=28,reference_fp32_gemm=16)
need=base.need

def fingerprint(p):return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='review'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return base.sha(p)
def read(p):return base.read(p)

def validate(p,frozen=False):
 need(p.get('id')==ID and p.get('resources')==RES and p.get('design')==DESIGN and p.get('output')==OUTPUT,'design/resources differ')
 if frozen:
  need(p.get('status')=='FROZEN_PENDING_REGISTRATION' and p.get('review') is None and isinstance(p.get('assets'),dict) and isinstance(p.get('software'),dict) and isinstance(p.get('worker_config_sha256'),str) and p.get('limits')==LIMITS and isinstance(p.get('failure'),dict),'actual closed assets/software/failure and worker required')
  need({'bundle','closed_run','closed_exit','cpu_protocol','cpu_protocol_binding'}<=set(p['assets']) and {'sharq_elf','lightx_files','lightx_package','author_modules'}<=set(p['software']),'asset/source closure missing')
  need({x['path'] for x in p['repo_sources']}==set(SOURCES),'repo source closure differs')

def authorize(p,d):
 validate(p,True);a=read(ROOT/'research/ACTIVE_RESEARCH.json');rows=[x for x in a['experiment_authority']['current_user_authorized_scopes'] if x.get('id')==ID];need(len(rows)==1 and a.get('user_continuation_authorization_20261027',{}).get('new_scope_registration_allowed') is True,'ACTIVE scope missing');s=rows[0]
 need(s.get('protocol_sha256')==d and s.get('resources')==RES and s.get('limits')==LIMITS and s.get('execution_status') in ('AUTHORIZED_NOT_STARTED','RUNNING_FIXED_SCOPE') and s.get('max_gpu_cards')==1,'ACTIVE scope differs')
 review=read(base.bound(ROOT,s['review']));need(review.get('review_schema')=='FFNDS_PREEXEC_REVIEW_V1' and review.get('scope_id')==ID and review.get('state')=='REVIEWER_RECHECK_APPROVED' and review.get('recheck_disposition')=='APPROVE' and review.get('author_agent_id')!=review.get('reviewer_agent_id') and review.get('approval_scope_fingerprint')==fingerprint(p),'exact independent review required')
 for k in ('author_review','independent_review','reviewer_recheck'):base.bound(ROOT,review[k])
 need(review.get('critical_source_hashes')=={x['path']:x['sha256'] for x in p['repo_sources']},'review source closure differs')
 return s

class Ledger:
 def __init__(self,report,persist,torch=None):self.r=report;self.persist=persist;self.budget=None;self.returned={};self.torch=torch;self.events=[];self.instrument=True
 def reserve(self,arm):
  need(self.budget is None,'overlapping work');self.budget=ops.expected_calls(arm).copy()
  for k,n in self.budget.items():need(self.r['predebited'].get(k,0)+n<=LIMITS[k],'operation budget '+k)
  for k,n in self.budget.items():self.r['predebited'][k]=self.r['predebited'].get(k,0)+n
  self.persist()
 def call(self,key,fn,*args):
  need(self.budget is not None and self.budget.get(key,0)>0,'call without predebit '+key);self.budget[key]-=1;self.r['invoked'][key]=self.r['invoked'].get(key,0)+1
  start=end=None
  if self.torch is not None and self.instrument:start=self.torch.cuda.Event(enable_timing=True);end=self.torch.cuda.Event(enable_timing=True);start.record()
  out=fn(*args)
  if end is not None:end.record();self.events.append((key,start,end))
  self.r['returned'][key]=self.r['returned'].get(key,0)+1;return out
 def finish(self):
  need(self.budget is not None and not any(self.budget.values()),'incomplete operation ledger');self.budget=None;self.persist()
 def extra(self,key,fn):
  need(self.budget is None and self.r['predebited'].get(key,0)<LIMITS[key],'extra operation budget '+key)
  self.r['predebited'][key]=self.r['predebited'].get(key,0)+1;self.r.setdefault('inflight_extra',{})[key]='UNKNOWN_UNTIL_RETURN';self.persist()
  self.r['invoked'][key]=self.r['invoked'].get(key,0)+1;start=time.monotonic();out=fn();self.r.setdefault('extra_wall_seconds',{}).setdefault(key,[]).append(time.monotonic()-start);self.r['returned'][key]=self.r['returned'].get(key,0)+1;self.r['inflight_extra'][key]='RETURNED';self.persist();return out

class AuthorAPI:
 def __init__(self,torch,quant,extension,lightx):self.torch,self.quant,self.extension,self.lightx=torch,quant,extension,lightx;self.capture=False;self.prepared=[]
 def keep(self,result):
  if self.capture:self.prepared.append(result)
  return result
 def sharq_prepare(self,x,n):return self.keep(self.quant.quantize_activation_sparse_residual_nvfp4(x,n))
 def dense_prepare(self,x):return self.keep(self.quant.quantize_activation_nvfp4(x))
 def output_scale(self,a,w):return (a*w).contiguous()
 def sparse_gemm(self,a,w,e,sa,sw,m,n,k,alpha):return self.extension.sparse_matmul_tensor_scale(a,w,e,sa,sw,m,n,k,alpha=alpha)
 def dense_accum(self,a,w,sa,sw,alpha,sparse):return self.extension.matmul_accum_tensor_scale(a,w,sa,sw,alpha,sparse,1.0)
 def dense_gemm(self,a,w,sa,sw,alpha):return self.extension.matmul_tensor_scale(a,w,sa,sw,alpha)
 def bias_add(self,y,b):return y+b
 def gelu_tanh(self,x):return self.torch.nn.functional.gelu(x,approximate='tanh')
 def gate_multiply(self,y,g):return y*g
 def bf16_linear(self,x,layer):return self.torch.addmm(layer.bias,x,layer.weight.T)
 def dense16_prepare(self,x,scale):return self.keep(self.lightx.scaled_nvfp4_quant(x,scale))
 def dense16_gemm(self,q,layer,s):return self.lightx.cutlass_scaled_nvfp4_mm(q,layer.dense16_weight,s,layer.dense16_scale,layer.dense16_alpha,layer.dense16_bias)


def run_case(torch,api,x,first,second,gate,arm,ledger,check):
 results=[];plain=[];correctness=None;api.prepared=[];instrumented=DESIGN['correctness']+DESIGN['warmup']+DESIGN['measured'];total=instrumented+DESIGN['uninstrumented_warmup']+DESIGN['uninstrumented_measured']
 for index in range(total):
  ledger.instrument=index<instrumented;api.capture=index==0 and x.shape[0]==32
  check();ledger.reserve(arm);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();wall=time.monotonic();start.record()
  result=ops.forward(arm,x,first,second,gate,api,ledger);end.record();torch.cuda.synchronize();elapsed=time.monotonic()-wall;per_call=[{'operation':k,'cuda_ms':a.elapsed_time(b)} for k,a,b in ledger.events];ledger.events=[];ledger.finish()
  if index==0:correctness=result
  if DESIGN['correctness']+DESIGN['warmup']<=index<instrumented:results.append({'cuda_ms':start.elapsed_time(end),'wall_seconds':elapsed,'operation_cuda_ms':per_call})
  if index>=instrumented+DESIGN['uninstrumented_warmup']:plain.append({'cuda_ms':start.elapsed_time(end),'wall_seconds':elapsed})
 api.capture=False;return correctness,{'per_operation_events':results,'overall_event_only':plain}

def relative(torch,a,b):
 a=a.float();b=b.float();den=torch.linalg.vector_norm(b);value=torch.linalg.vector_norm(a-b)/torch.clamp(den,min=1e-20)
 return {'relative_l2':float(value.item()),'finite':bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())}

def make_layer(t,expert,number,quant,ledger,torch):
 def get(name):return t[expert+'_'+name]
 w=get(f'raw_{number}_weight');b=get(f'raw_{number}_bias')
 def validate_scale():
  for v in (get(f'nv_{number}_alpha'),get(f'nv_{number}_input_global_scale')):need(bool(torch.isfinite(v).all().item()) and bool((v>0).all().item()),'invalid checkpoint scale')
 ledger.extra('checkpoint_scale_validation',validate_scale)
 def pack():
  result=quant.quantize_weight_shared_nvfp4(w);torch.cuda.synchronize();return result
 q,ss,sd,sw=ledger.extra('static_group32_weight_quant',pack)
 return SimpleNamespace(weight=w,bias=b,in_features=w.shape[1],out_features=w.shape[0],weight_q=q,scale_w_sparse=ss,scale_w_dense=sd,weight_scale=sw,
  dense16_weight=get(f'nv_{number}_weight'),dense16_scale=get(f'nv_{number}_weight_scale'),dense16_bias=get(f'nv_{number}_bias'),dense16_alpha=get(f'nv_{number}_alpha').reshape(1),input_global_scale=get(f'nv_{number}_input_global_scale').reshape(1))

def check_parent(p):
 cpu=ROOT/p['assets']['cpu_protocol'];need(base.bound(ROOT,p['assets']['cpu_protocol_binding'])==cpu.resolve(),'CPU protocol binding differs');closed=asset_parent.inspect_closed(cpu)
 need(closed['RUN']==p['assets']['closed_run'] and closed['exit']==p['assets']['closed_exit'] and closed['bundle']==p['assets']['bundle'],'CPU parent closed identity differs')
 return closed

def check_failure(p):
 f=p['failure'];rr=base.bound(HF,f['RUN']);ee=base.bound(HF,f['exit']);base.bound(HF,f['stderr']);r=read(rr);e=read(ee)
 need(r['id']=='E0_SHARQ_SM120_WAN_FFN_PAIR_PROBE_20261030' and r['status']=='FAILED_PRESERVED_NO_RETRY' and r['error_type']=='ValueError' and r['error']=='scale buffer size differs' and e['process_exited'] is True and e['child_returncode']==1 and e['timeout'] is False and e['automatic_retry'] is False and e['run_sha256']==sha(rr) and e['id']==r['id'] and e['protocol_sha256']==r['protocol_sha256']==sha(ROOT/'config/e0_sharq_sm120_wan_ffn_pair_probe_20261030_pending_review_v2.json') and e['spec_sha256']==r['launch_spec_sha256'],'V1 actual failure identity differs')
 need([(c['expert'],c['M'],c['arm']) for c in r['cases']]==[('high',32,'original_bf16'),('high',32,'dense_group16')] and r['returned'].get('dense32_gemm')==30 and not r['invoked'].get('sparse_gemm',0),'V1 partial accounting differs')

def validate_tensors(t,torch):
 expected={}
 for expert in DESIGN['experts']:
  for name,shape,dtype in (('V1',(32,5120),torch.bfloat16),('a',(32,13824),torch.bfloat16),('h',(32,13824),torch.bfloat16),('Y',(32,5120),torch.bfloat16),('gate',(5120,),torch.bfloat16),('rows',(32,),torch.int64)):
   expected[expert+'_'+name]=(shape,dtype)
  for number,n,k in ((0,13824,5120),(2,5120,13824)):
   for name,shape,dtype in (('raw_weight',(n,k),torch.bfloat16),('raw_bias',(n,),torch.bfloat16),('nv_weight',(n,k//2),torch.uint8),('nv_weight_scale',(n,k//16),torch.float8_e4m3fn),('nv_bias',(n,),torch.bfloat16),('nv_alpha',(),torch.float32),('nv_input_global_scale',(),torch.float32)):
    prefix,field=name.split('_',1);expected[f'{expert}_{prefix}_{number}_{field}']=(shape,dtype)
 need(set(t)==set(expected),'bundle tensor keys differ')
 for key,(shape,dtype) in expected.items():need(tuple(t[key].shape)==shape and t[key].dtype==dtype,'bundle tensor shape/dtype differs: '+key)

def scale_buffer_view(torch,s,rows,k,layout):
 mr=(rows+127)//128*128;groups=(k//16+3)//4*4
 need(layout in ('lightx','author') and k%16==0,'unknown scale layout')
 expected=mr*groups
 if layout=='author':
  need(k%64==0,'author K must have complete scale tiles')
  expected=(rows//128+1)*128*(k//16)
 need(s.numel()==expected,'scale buffer size differs for '+layout)
 # Author bindings overallocate one trailing 128-row tile when rows are
 # aligned. CUTLASS row tiles are consecutive; only that tail is discarded.
 return s.reshape(-1)[:mr*groups].view(torch.uint8).reshape(mr,groups)

def logical_scale_bytes(torch,s,rows,k,layout):
 sf=scale_buffer_view(torch,s,rows,k,layout);mr,groups=sf.shape
 return sf.reshape(mr//128,groups//4,32,4,4).permute(0,3,2,1,4).reshape(mr,groups)[:rows,:k//16].contiguous()

def decode_actual(torch,q,s,layout):
 rows,k=q.shape[0],q.shape[1]*2;sf=scale_buffer_view(torch,s,rows,k,layout).view(torch.float8_e4m3fn)
 return decoder.decode_unscaled(q,sf)

def linear_reference(torch,api,arm,x,layer,actual,ledger,captured):
 # Each observer reserves one bounded linear bundle before this function.
 # It re-prepares actual A once, decodes the actual packed W/A, and uses at
 # most two FP32 GEMMs. Debug shares author selector code, not an independent
 # mathematical proof of selection; compare its payload with normal prepare.
 for v in (layer.dense16_alpha,layer.input_global_scale):need(bool(torch.isfinite(v).all().item()) and bool((v>0).all().item()),'invalid checkpoint scale')
 if arm=='dense_group16':
  q,sa=ledger.extra('reference_prepare',lambda:api.dense16_prepare(x,layer.input_global_scale));same=(q,sa);a=ledger.extra('reference_decode',lambda:decode_actual(torch,q,sa,'lightx'));w=ledger.extra('reference_decode',lambda:decode_actual(torch,layer.dense16_weight,layer.dense16_scale,'lightx'))
  reference=ledger.extra('reference_fp32_gemm',lambda:torch.addmm(layer.dense16_bias.float(),a,w.T,alpha=float(layer.dense16_alpha.item())))
 elif arm=='dense_shared32':
  q,sa,scale=ledger.extra('reference_prepare',lambda:api.dense_prepare(x));same=(q,sa,scale);a=ledger.extra('reference_decode',lambda:decode_actual(torch,q,sa,'author'));w=ledger.extra('reference_decode',lambda:decode_actual(torch,layer.weight_q,layer.scale_w_dense,'author'))
  reference=ledger.extra('reference_fp32_gemm',lambda:torch.addmm(layer.bias.float(),a,w.T,alpha=float((scale*layer.weight_scale).item())))
 else:
  ac,e,ss,qr,sr,scale=ledger.extra('reference_prepare',lambda:api.sharq_prepare(x,layer.out_features));same=(ac,e,ss,qr,sr,scale)
  normalized=(x/scale).to(torch.bfloat16);qs,ss_debug,qr_debug,sr_debug=ledger.extra('reference_debug_sparse',lambda:api.extension.fused_sparse_residual_quantize_x_debug(normalized,layer.out_features))
  need(torch.equal(qr,qr_debug) and torch.equal(sr,sr_debug) and torch.equal(ss,ss_debug),'debug/normal sparse preparation differs')
  decoded_scale=ledger.extra('reference_duplicate_scales',lambda:api.extension.duplicate_sfa32_to_sfa16(ss_debug,x.shape[0],layer.out_features,layer.in_features))
  sparse=ledger.extra('reference_decode',lambda:decode_actual(torch,qs,decoded_scale,'author'));residual=ledger.extra('reference_decode',lambda:decode_actual(torch,qr,sr,'author'));w=ledger.extra('reference_decode',lambda:decode_actual(torch,layer.weight_q,layer.scale_w_dense,'author'))
  nonzero=int(torch.count_nonzero(sparse).item());need(0<nonzero<sparse.numel(),'degenerate sparse logical payload')
  alpha=float((scale*layer.weight_scale).item());part=ledger.extra('reference_fp32_gemm',lambda:torch.mm(sparse,w.T)).mul_(alpha).to(torch.bfloat16)
  combined=ledger.extra('reference_fp32_gemm',lambda:torch.addmm(part.float(),residual,w.T,alpha=alpha)).to(torch.bfloat16);reference=combined+layer.bias
 def same_bits(a,b):
  if a is None or b is None:return a is None and b is None
  return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
 need(len(same)==len(captured),'reprepared operands differ from actual correctness call')
 for i,(a,b) in enumerate(zip(same,captured)):
  if arm in ('dense_group16','dense_shared32') and i==1:
   layout='lightx' if arm=='dense_group16' else 'author';a=logical_scale_bytes(torch,a,x.shape[0],x.shape[1],layout);b=logical_scale_bytes(torch,b,x.shape[0],x.shape[1],layout)
  need(same_bits(a,b),'reprepared operands differ from actual correctness call')
 result=relative(torch,actual,reference);result['same_encoding']=True;need(result['finite'] and result['relative_l2']<=DESIGN['same_encoding_linear_rel_l2_max'],'same-encoding reference mismatch');return result

def verify_job(a,p,s):
 need(a.job_spec and sha(a.job_spec)==s['launch_spec_sha256'],'job SHA differs');j=read(a.job_spec);v=j['argv'];need(len(v)==18 and v[1].startswith('FFNDS_WORKER_CONFIG='),'job argv malformed')
 profile=Path(v[1].split('=',1)[1]);need(sha(profile)==p['worker_config_sha256'],'worker profile changed')
 expected=['env',v[1],'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',str(ROOT/'scripts/worker_teal_gpu0_uuid_exec.sh'),'taskset','-c','0,1,2,3',str(ENV),'-I','-B',str(ROOT/SELF),'run','--protocol',str(a.protocol.resolve()),'--job-spec',str(a.job_spec.resolve())]
 need(v==expected and j['id']==ID and j['protocol_sha256']==sha(a.protocol) and Path(j['protocol_path']).resolve()==a.protocol.resolve() and Path(j['cwd']).resolve()==ROOT and j['hf_root']==str(HF) and Path(j['output_dir']).resolve()==(HF/OUTPUT).parent and Path(j['run_json']).resolve()==HF/OUTPUT/'RUN.json' and j['external_timeout_seconds']==RES['outer_seconds'],'job identity differs')
 need(sys.flags.isolated and sorted(os.sched_getaffinity(0))==[0,1,2,3] and not {'torch','numpy','safetensors'}&set(sys.modules) and asset_parent.runtime_identity()==p['runtime'],'fresh isolated worker or runtime differs')
 import worker_identity;worker_identity.require_worker()

def verify_software(p):
 for b in p['repo_sources']:base.bound(ROOT,b)
 for b in p['software']['author_modules']:base.bound(HF,b)
 base.bound(HF,p['software']['sharq_elf'])
 for b in p['software']['lightx_files']:
  path=Path(b['path']);need(path.is_file() and path.stat().st_size==b['bytes'] and sha(path)==b['sha256'],'lightx2v source differs')


def run(a):
 p=read(a.protocol);d=sha(a.protocol);s=authorize(p,d);verify_job(a,p,s);verify_software(p);check_parent(p);check_failure(p)
 out=HF/OUTPUT;need(not out.exists(),'fresh output required');out.mkdir(parents=True,mode=0o700);started=time.monotonic();cpu=time.process_time()
 r={'id':ID,'status':'STARTED_NO_RETRY','protocol_sha256':d,'launch_spec_sha256':s['launch_spec_sha256'],'predebited':{},'invoked':{},'returned':{},'cases':[], 'automatic_retry':False,'model_calls':0,'counter_semantics':'Python entries and returns; CUDA asynchronous work and internal kernel launches UNKNOWN on interruption','fullM_identity':DESIGN['fullM'],'bundle_io':DESIGN['bundle_io']}
 emergency=support.reserve_emergency(out)
 def persist():support.checked_persist(out/'RUN.json',r,out.parent,RES['new_HF_bytes'])
 persist();signal.signal(signal.SIGUSR1,lambda *_:(_ for _ in ()).throw(RuntimeError('output monitor failure')));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(RES['wall_seconds']);monitor=support.UsageMonitor(out.parent,RES['new_HF_bytes'],4<<30);monitor.start();gpu=None
 try:
  import torch
  from safetensors.torch import load_file
  import startup_gpu_reservation
  torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;hold,gpu,reservation=startup_gpu_reservation.reserve_before_assets(torch);r['reservation']=reservation
  need(torch.cuda.get_device_capability()==(12,0) and 'RTX PRO 6000' in torch.cuda.get_device_name(),'unexpected GPU architecture or model')
  ledger=Ledger(r,persist,torch);b=p['assets']['bundle'];need(b['bytes']==DESIGN['bundle_full_hash_bytes'] and b['bytes']<=RES['bundle_hash_bytes_max'],'bundle size budget or identity differs')
  def load_bound():return base.bound(HF,b)
  bundle_path=HF/b['path'];st=asset_parent.extract.stat_id(bundle_path);r['bundle_hash_predebit_bytes']=b['bytes'];persist();bundle=ledger.extra('bundle_hashes',load_bound);need(asset_parent.extract.stat_id(bundle)==st,'bundle changed during hash');r['bundle_hash_completed_bytes']=b['bytes'];persist()
  t=load_file(str(bundle),device='cpu');need(asset_parent.extract.stat_id(bundle)==st,'bundle changed during load');validate_tensors(t,torch);r['bundle_keys']=sorted(t);r['H2D_predebit_bytes']=sum(x.numel()*x.element_size() for x in t.values());need(r['H2D_predebit_bytes']<=RES['bundle_hash_bytes_max'],'H2D budget exceeded');persist()
  h2d_start=time.monotonic();t={k:v.to('cuda') for k,v in t.items()};torch.cuda.synchronize();r['H2D_completed_bytes']=r['H2D_predebit_bytes'];r['H2D_wall_seconds']=time.monotonic()-h2d_start;persist()
  os.environ['SHARQ_KERNEL_BUILD']=str(ELF.parent);sys.path.insert(0,str(SRC));sys.path.insert(0,str(SRC/'model'));sys.path.insert(0,str(Path(p['software']['lightx_package']).parent))
  quant=importlib.import_module('quantize');ext=quant.load_sharq_ops();lightx=importlib.import_module('lightx2v_kernel.gemm')
  need(Path(quant.__file__).resolve()==(SRC/'model/quantize.py').resolve() and Path(sys.modules['hif4'].__file__).resolve()==(SRC/'model/hif4.py').resolve() and Path(sys.modules['sharq_loader'].__file__).resolve()==(SRC/'sharq_loader.py').resolve() and Path(ext.__file__).resolve()==ELF.resolve() and Path(lightx.__file__).resolve()==(Path(p['software']['lightx_package'])/'gemm.py').resolve(),'loaded module origin differs')
  api=AuthorAPI(torch,quant,ext,lightx)
  def check():
   gpu.check();need(time.monotonic()-started<RES['wall_seconds'],'wall budget exceeded');need(torch.cuda.max_memory_reserved()<=RES['GPU_reserved_max_bytes'],'GPU reserved budget exceeded');need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=RES['host_RSS_max_bytes'],'host RSS budget exceeded')
  for expert in DESIGN['experts']:
   first=make_layer(t,expert,0,quant,ledger,torch);second=make_layer(t,expert,2,quant,ledger,torch);gate=t[expert+'_gate']
   for m in DESIGN['M']:
    x=t[expert+'_V1'] if m==32 else ledger.extra('fullM_row_repeat',lambda:t[expert+'_V1'].repeat((m+31)//32,1)[:m].contiguous())
    reference=None
    for arm in ops.ARMS:
     result,times=run_case(torch,api,x,first,second,gate,arm,ledger,check)
     if reference is None:reference={k:v.clone() for k,v in result.items()}
     def compare():
      metrics={k:relative(torch,result[k],reference[k]) for k in ('pre_gelu','h','y','gated')};neutral={}
      if arm=='original_bf16' and m==32:
       neutral={k:relative(torch,result[k],t[expert+'_'+source]) for k,source in (('pre_gelu','a'),('h','h'),('y','Y'))}
       neutral['gated']=relative(torch,result['gated'],t[expert+'_Y']*gate)
      return metrics,neutral
     metrics,neutral=ledger.extra('correctness_comparisons',compare)
     need(all(v['finite'] for v in metrics.values()) and all(v['finite'] and v['relative_l2']<=DESIGN['neutral_rel_l2_max'] for v in neutral.values()),'nonfinite or BF16 neutral mismatch')
     same_encoding={}
     if m==32 and arm!='original_bf16':
      need(len(api.prepared)==2,'missing actual quantization capture')
      for pos,(name,value,layer,actual) in enumerate((('W1',x,first,result['pre_gelu']),('W2',result['h'],second,result['y']))):
       def observe():
        result_ref=linear_reference(torch,api,arm,value,layer,actual,ledger,api.prepared[pos]);torch.cuda.synchronize();return result_ref
       same_encoding[name]=ledger.extra('reference_linear',observe)
     r['cases'].append({'expert':expert,'M':m,'arm':arm,'relative_to_original_bf16':metrics,'captured_bf16_neutral':neutral,'same_encoding_reference':same_encoding,'timings':times});persist();check()
    del reference,x
   del first,second
  need(r['predebited']==LIMITS and r['invoked']==LIMITS and r['returned']==LIMITS and len(r['cases'])==16,'incomplete physical entry ledger')
  r['status']='COMPLETE_WAN_FFN_OPERATOR_EXIT_PENDING'
 except BaseException as e:r.update(status='FAILED_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500]);raise
 finally:
  signal.alarm(0)
  if gpu is not None:gpu.close();r['GPU_monitor']=gpu.report();r['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved();r['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
  r['output_monitor']=monitor.stop();r['wall_seconds']=time.monotonic()-started;r['cpu_seconds']=time.process_time()-cpu;r['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  if r['output_monitor']['fault'] or not r['output_monitor']['thread_stopped'] or (gpu is not None and not r['GPU_monitor']['sampled_isolation_pass']):r['status']='FAILED_MONITOR_PRESERVED'
  try:persist()
  except BaseException as e:support.emergency_failure(emergency,r,'final persist failed '+type(e).__name__);raise
  if r['status']=='FAILED_MONITOR_PRESERVED':raise RuntimeError('monitor failure')

def draft(cpu_protocol,profile):
 closed=asset_parent.inspect_closed(cpu_protocol);report=read(ROOT/'research/20261029/NATIVE_NVFP4_REAL_FFN_PROBE_RESULTS_20261029.json');prior=read(base.bound(HF,report['CPU_assets']['RUN']));prior_exit=read(base.bound(HF,report['CPU_assets']['exit']));asset_parent.verify_parent(prior,prior_exit,report['CPU_assets']['RUN'])
 need(cpu_protocol.resolve().is_relative_to(ROOT) and profile.is_file(),'protocol/profile required')
 failure={'RUN':base.binding(FAILED/'results/RUN.json',HF),'exit':base.binding(FAILED/'EXIT_RECEIPT.json',HF),'stderr':base.binding(FAILED/'child.stderr.log',HF)};check_failure({'failure':failure})
 return {'id':ID,'status':'FROZEN_PENDING_REGISTRATION','design':DESIGN,'resources':RES,'limits':LIMITS,'output':OUTPUT,'review':None,
  'assets':{'cpu_protocol':str(cpu_protocol.resolve().relative_to(ROOT)),'cpu_protocol_binding':base.binding(cpu_protocol,ROOT),'closed_run':closed['RUN'],'closed_exit':closed['exit'],'bundle':closed['bundle']},'failure':failure,
  'software':{'author_modules':[base.binding(SRC/x,HF) for x in AUTHOR_MODULES],'sharq_elf':base.binding(ELF,HF),'lightx_package':prior['software_package'],'lightx_files':prior['software_files']},
  'repo_sources':[base.binding(ROOT/x,ROOT) for x in SOURCES],'worker_config_sha256':sha(profile),'runtime':asset_parent.runtime_identity()}

def inspect_closed(protocol):
 p=read(protocol);validate(p,True);out=HF/OUTPUT;rr=out/'RUN.json';ee=out.parent/'EXIT_RECEIPT.json';r=read(rr);e=read(ee);d=sha(protocol)
 need(r.get('id')==ID and r.get('status')=='COMPLETE_WAN_FFN_OPERATOR_EXIT_PENDING' and r.get('protocol_sha256')==d and e.get('id')==ID and e.get('process_exited') is True and e.get('child_returncode')==0 and e.get('timeout') is False and e.get('automatic_retry') is False and e.get('run_sha256')==sha(rr) and e.get('protocol_sha256')==d and e.get('spec_sha256')==r['launch_spec_sha256'],'actual GPU EXIT0 required')
 need(r['predebited']==r['invoked']==r['returned']==LIMITS and len(r['cases'])==16 and {(x['expert'],x['M'],x['arm']) for x in r['cases']}=={(e,m,a) for e in DESIGN['experts'] for m in DESIGN['M'] for a in ops.ARMS} and all(len(x['same_encoding_reference'])==(2 if x['M']==32 and x['arm']!='original_bf16' else 0) for x in r['cases']) and r['wall_seconds']<=RES['wall_seconds'] and r['peak_RSS_bytes']<=RES['host_RSS_max_bytes'] and r['GPU_peak_reserved_bytes']<=RES['GPU_reserved_max_bytes'] and r['GPU_monitor']['sampled_isolation_pass'] and not r['output_monitor']['fault'] and r['output_monitor']['thread_stopped'],'GPU accounting/resources incomplete')
 need(r['bundle_hash_predebit_bytes']==r['bundle_hash_completed_bytes']==DESIGN['bundle_full_hash_bytes'] and r['H2D_predebit_bytes']==r['H2D_completed_bytes'] and bool(r['GPU_monitor']['samples']) and all(len(x['timings']['per_operation_events'])==DESIGN['measured'] and len(x['timings']['overall_event_only'])==DESIGN['uninstrumented_measured'] for x in r['cases']),'GPU timing or transfer proof incomplete')
 check_parent(p);check_failure(p);return {'id':ID,'status':'PASS_CLOSED_SINGLE_SOURCE_PAIRED_FFN','RUN':base.binding(rr,HF),'exit':base.binding(ee,HF),'parent':p['assets']['closed_run']}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['draft','inspect','run','inspect-closed']);ap.add_argument('--protocol',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--cpu-protocol',type=Path);ap.add_argument('--worker-profile',type=Path);ap.add_argument('--job-spec',type=Path);a=ap.parse_args()
 if a.action=='draft':
  need(a.out and a.cpu_protocol and a.worker_profile and not a.out.exists(),'fresh draft, closed CPU protocol and worker profile required');base.write(a.out,draft(a.cpu_protocol,a.worker_profile))
 elif a.action=='inspect':p=read(a.protocol);validate(p,True);print(json.dumps({'fingerprint':fingerprint(p),'status':p['status'],'execution_requires':'independent recheck and ACTIVE'},indent=2))
 elif a.action=='run':run(a)
 else:print(json.dumps(inspect_closed(a.protocol),indent=2))
if __name__=='__main__':main()
