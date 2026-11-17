"""Candidate six-call true-Wan author shape/action oracle; no standalone launch.

Caller must separately validate reviewed scope, real allocation and resources.
Not a scientific similarity pass/fail threshold. Original author functions are
extracted verbatim from the identity-pinned source, never reimplemented.
"""
import ast,hashlib,json,math,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from icc_stage3_assets import checked_path,load_weights,verify_live_weights
from icc_stage3_grid import exclusive_json

AUTHOR_SHA256='f1bf94d28a65b3efe3df9157cbf8f9439e546182e05140699afb1b154d34d316'
AUTHOR_GIT_BLOB='d053db79cfffba9b2c054a7e6b195066c77a4f10'
FAMILIES=('SVD','CA','CD')

def author_source(path):
 raw=Path(path).read_bytes();digest=hashlib.sha256(raw).hexdigest();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
 if digest!=AUTHOR_SHA256 or blob!=AUTHOR_GIT_BLOB:raise ValueError('pinned author source identity mismatch')
 tree=ast.parse(raw,filename=str(path));names={'wsvd','wsvd_merge_lowrank','decomp_check'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 if len(nodes)!=len(names) or {n.name for n in nodes}!=names:raise ValueError('author source functions mismatch')
 namespace={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace['wsvd_merge_lowrank']

def probe(rows,channels,device):
 r=torch.arange(rows,device=device,dtype=torch.int64)[:,None];c=torch.arange(channels,device=device,dtype=torch.int64)[None,:]
 return (((c*17+r*31)%257)-128).to(torch.float32)/128

def metrics(x,y):
 delta=(x.double()-y.double());a=x.double();b=y.double()
 rms=lambda v:float(v.square().mean().sqrt().item())
 out={'difference_rms':rms(delta),'author_rms':rms(a),'reference_rms':rms(b),'difference_max_abs':float(delta.abs().max().item())}
 out['difference_over_reference_rms']=out['difference_rms']/out['reference_rms'] if out['reference_rms']>0 else None
 if any(not math.isfinite(v) for v in out.values() if v is not None):raise ValueError('nonfinite action diagnostic')
 return out

def run(scope,*,authorize,check_resources,device):
 authorize(scope)
 if scope['author_source']['sha256']!=AUTHOR_SHA256 or scope['expert']!='high' or scope['layer']!=4 or scope['rank']!=1408 or scope['max_author_calls']!=6 or scope['probe_rows']!=8 or scope['retries']!=0:
  raise ValueError('shape acceptance scope differs')
 if not torch.cuda.is_available() or torch.device(device).type!='cuda' or torch.backends.cuda.matmul.allow_tf32:raise ValueError('CUDA with TF32 disabled required')
 author=author_source(checked_path(scope['author_source']))
 factor_index={(r['identity']['family'],r['identity']['expert'],r['identity']['layer']):r for r in scope['factors']}
 if len(factor_index)!=3 or set(factor_index)!={(f,'high',4) for f in FAMILIES}:raise ValueError('three matched high04 factors required')
 scale_index={r['family']:r for r in scope['scales']}
 if len(scale_index)!=2 or set(scale_index)!={'CA','CD'}:raise ValueError('matched CA/CD statistics required')
 weights=load_weights(scope['checkpoints']['high'],4,device);w1,b1,w2,b2=weights
 verify_live_weights(scope['live_weight_manifest'],'high',4,w1,b1,w2,b2)
 out=Path(scope['output']);out.mkdir(parents=True,exist_ok=False);journal=out/'AUTHOR_CALLS.jsonl'
 def event(value):
  with journal.open('a',encoding='utf8') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 records=[]
 for family in FAMILIES:
  factor=factor_index[(family,'high',4)];artifact=torch.load(checked_path(factor),map_location='cpu',weights_only=True);ident=artifact['identity']
  if ident!=factor['identity'] or ident['family']!=family or ident['rank']!=1408 or ident['checkpoint_sha256']!=scope['checkpoints']['high']['sha256'] or ident['svd_driver']!='gesvd' or ident['execution_factor_dtype']!='bfloat16':raise ValueError('matched factor identity mismatch')
  if family=='SVD':
   if ident['calibration_manifest_sha256'] is not None:raise ValueError('unexpected SVD calibration')
   scales=None
  else:
   spec=scale_index[family]
   if spec['calibration_manifest_sha256']!=scope['clean_cal11_manifest_sha256'] or ident['calibration_manifest_sha256']!=scope['clean_cal11_manifest_sha256'] or ident['scale_sha256']!=spec['file']['sha256']:raise ValueError('calibration ancestry mismatch')
   raw=torch.load(checked_path(spec['file']),map_location='cpu',weights_only=True)
   if set(raw)!={'u','z','h','y'}:raise ValueError('scale fields')
   scales={k:raw[k]['capped'].to(device=device,dtype=torch.float32) for k in raw}
  for index,(name,w,left,right) in enumerate((('W1',w1,'z','u'),('W2',w2,'y','h'))):
   check_resources();event({'event':'AUTHOR_CALL_START','family':family,'matrix':name,'call':len(records)+1})
   ls=None if scales is None else scales[left].clone();rs=None if scales is None else scales[right].clone()
   torch.cuda.synchronize();started=time.monotonic();torch.cuda.reset_peak_memory_stats()
   a,b,energy,abs_error,rel_error=author(w,1408,ls,rs)
   torch.cuda.synchronize();seconds=time.monotonic()-started
   if a.shape!=(w.shape[0],1408) or b.shape!=(1408,w.shape[1]) or a.dtype!=torch.float32 or b.dtype!=torch.float32 or not all(math.isfinite(float(x)) for x in (energy,abs_error,rel_error)) or not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():raise ValueError('author outputs nonfinite or wrong geometry')
   prefix='first' if index==0 else 'second';af=artifact[prefix+'_left'].to(device);bf=artifact[prefix+'_right'].to(device)
   if af.shape!=a.shape or bf.shape!=b.shape or af.dtype!=torch.bfloat16 or bf.dtype!=torch.bfloat16:raise ValueError('reference factor shape/dtype')
   x=probe(8,w.shape[1],device);ya=F.linear(F.linear(x,b),a);yr=F.linear(F.linear(x,bf.float()),af.float())
   ya_b=F.linear(F.linear(x.bfloat16(),b.bfloat16()),a.bfloat16()).float();yr_b=F.linear(F.linear(x.bfloat16(),bf),af).float()
   row={'family':family,'matrix':name,'shape':list(w.shape),'author_factor_shapes':[list(a.shape),list(b.shape)],'reference_factor_sha256':factor['sha256'],'scale_sha256':None if scales is None else scale_index[family]['file']['sha256'],'elapsed_seconds':seconds,'peak_gpu_allocated':torch.cuda.max_memory_allocated(),'peak_gpu_reserved':torch.cuda.max_memory_reserved(),'author_energy_sum_fraction':float(energy),'author_full_matrix_abs_error_mean':float(abs_error),'author_full_matrix_rel_error':float(rel_error),'action_fp32_author_vs_bf16_factor_lifted':metrics(ya,yr),'action_bf16_author_vs_bf16_reference':metrics(ya_b,yr_b)}
   records.append(row);exclusive_json(out/f'{family}__{name}.json',row);event({'event':'AUTHOR_CALL_COMPLETE','family':family,'matrix':name,'call':len(records)});check_resources()
   del a,b,af,bf,x,ya,yr,ya_b,yr_b;torch.cuda.empty_cache()
  del artifact,scales
 if len(records)!=6:raise RuntimeError('six-call oracle incomplete')
 summary={'status':'SIX_AUTHOR_CALLS_COMPLETE_NUMERICS_DESCRIPTIVE','author_source_sha256':AUTHOR_SHA256,'rank':1408,'site':['high',4],'author_calls':6,'probe_formula':'((17*column+31*row)%257-128)/128; rows 0..7','author_svd':'torch.linalg.svd default driver, full_matrices=True','reference_svd':'gesvd, full_matrices=False','comparison':'matched scales and BF16 reference; RMS descriptive, no similarity acceptance threshold','rows':records}
 exclusive_json(out/'AUTHOR_WAN_SHAPE_SUMMARY.json',summary);return summary
