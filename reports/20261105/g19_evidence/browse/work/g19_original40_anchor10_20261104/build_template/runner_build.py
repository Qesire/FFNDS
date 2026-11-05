"""Original40 secant fitting and BF16 matrix build. No work without reviewed Slurm.

CAL training diagnostics are descriptive; they are not held-out or trajectory quality.
"""
import argparse,hashlib,importlib.metadata,json,math,os,re,sys,time
from pathlib import Path
from runtime_guard import require_compute_allocation,verify_frozen_scope,RunReceipt
from cal_contract import SOURCES,BRANCHES,EXPERTS,BUILD_CONFIG,read,sha,write_json,verify_inputs
from build_assets import save_matrix

def append(path,value):
 with path.open('a',encoding='utf8') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()

def weight_index(runs,identities,receipt):
 index={};stats={}
 specs={x['path']:x for x in runs[SOURCES[0]]['scope']['original_weight_files']}
 for name,ident in sorted(identities.items()):
  p=Path(name);before=p.stat();spec=specs[name]
  if (before.st_size,before.st_mtime_ns)!=(ident['size'],ident['mtime_ns']) or sha(p)!=ident['sha256']:raise ValueError('original weight payload changed')
  with p.open('rb') as f:
   n=int.from_bytes(f.read(8),'little')
   if not 0<n<16*1024**2 or n!=spec['header_bytes']:raise ValueError('safetensors header bytes')
   header=f.read(n)
  if hashlib.sha256(header).hexdigest()!=ident['header_sha256']:raise ValueError('original tensor header')
  data=json.loads(header);expert='high' if p.parent.name=='high_noise_model' else 'low' if p.parent.name=='low_noise_model' else None
  if expert is None:raise ValueError('original shard expert')
  for key,item in data.items():
   m=re.fullmatch(r'blocks\.(\d+)\.ffn\.(0|2)\.(weight|bias)',key)
   if not m:continue
   layer,which,kind=int(m[1]),m[2],m[3];expected=([13824,5120] if which=='0' else [5120,13824]) if kind=='weight' else ([13824] if which=='0' else [5120])
   identity=(expert,layer,which+'.'+kind)
   if not 0<=layer<40 or identity in index or item['dtype']!='F32' or item['shape']!=expected:raise ValueError('original F32 tensor geometry')
   lo,hi=item['data_offsets']
   if not 0<=lo<hi<=before.st_size-8-n or hi-lo!=math.prod(expected)*4:raise ValueError('tensor data offsets')
   index[identity]={'path':name,'key':key,'shape':expected,'header_entry':item}
  after=p.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('weight changed during verification')
  stats[name]=(after.st_size,after.st_mtime_ns);receipt.check_deadline()
 expected={(e,l,s) for e in EXPERTS for l in range(40) for s in ('0.weight','0.bias','2.weight','2.bias')}
 if set(index)!=expected:raise ValueError('original FFN tensor inventory incomplete')
 return index,stats

def tensor_sha(x,torch):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def load_native_weight(entry,torch,safe_open):
 with safe_open(entry['path'],framework='pt',device='cpu') as f:
  raw=f.get_tensor(entry['key'])
  if list(raw.shape)!=entry['shape'] or raw.dtype!=torch.float32 or not bool(torch.isfinite(raw).all()):raise ValueError('original weight tensor')
  raw_hash=tensor_sha(raw,torch);native=raw.to(torch.bfloat16).contiguous();native_hash=tensor_sha(native,torch)
  if not bool(torch.isfinite(native).all()):raise ValueError('native BF16 weight overflow')
  result=native.to('cuda')
 return result,{'shard':entry['path'],'tensor':entry['key'],'storage_dtype':'F32','native_dtype':'BF16','raw_payload_sha256':raw_hash,'native_bf16_payload_sha256':native_hash,'shape':entry['shape']}

def load_sample(run,expert,branch,layer,step,torch,verified):
 entry=run['captures'][expert,branch,layer,step];p=Path(entry['file']);before=p.stat()
 if str(p) not in verified:
  if sha(p)!=entry['sha256']:raise ValueError('CAL sample payload SHA')
  verified[str(p)]=(before.st_size,before.st_mtime_ns)
 elif verified[str(p)]!=(before.st_size,before.st_mtime_ns):raise ValueError('CAL sample mutated')
 d=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
 expected={'u':(64,5120),'preactivation':(64,13824),'postactivation':(64,13824),'F':(64,5120)}
 if not set(expected).union({'ids','gate','grid','coordinates','row_mean','row_variance','row_norm2'}).issubset(d):raise ValueError('CAL operands missing')
 for name,shape in expected.items():
  x=d[name]
  if tuple(x.shape)!=shape or x.dtype!=torch.bfloat16 or x.device.type!='cpu' or not bool(torch.isfinite(x).all()):raise ValueError('CAL tensor schema '+name)
 grid=tuple(d['grid']);ids=d['ids'];g=d['gate'].squeeze()
 if list(grid)!=entry['token_grid'] or ids.dtype!=torch.int64 or ids.shape!=(64,) or ids.unique().numel()!=64 or int(ids.min())<0 or int(ids.max())>=math.prod(grid) or g.shape!=(5120,) or g.dtype not in (torch.float32,torch.bfloat16) or not bool(torch.isfinite(g).all()):raise ValueError('CAL gate/ids/grid')
 seed=int.from_bytes(hashlib.sha256(f'42|{expert}|{layer}|{grid}'.encode()).digest()[:8],'little')&((1<<63)-1)
 gen=torch.Generator(device='cpu').manual_seed(seed);expected_ids=torch.randperm(math.prod(grid),generator=gen)[:64].sort().values
 if not torch.equal(ids,expected_ids):raise ValueError('CAL uniform sample identity')
 T,H,W=grid;coords=[(i//(H*W),(i//W)%H,i%W) for i in ids.tolist()]
 if [tuple(x) for x in d['coordinates']]!=coords:raise ValueError('CAL token coordinates')
 after=p.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('CAL sample changed while reading')
 return d

def moments_for_site(runs,expert,layer,torch,np,receipt,verified):
 numerator=torch.zeros(13824,device='cuda',dtype=torch.float64);denominator=numerator.clone();postenergy=numerator.clone()
 records=[];raw_num=[];raw_den=[];raw_energy=[]
 offset_num=np.zeros((2,2,10,13824),dtype=np.float64);offset_den=np.zeros_like(offset_num);offset_energy=np.zeros_like(offset_num);offset_counts=np.zeros((2,2,10),dtype=np.int64)
 for si,source in enumerate(SOURCES):
  run=runs[source];pairs=[p for p in run['pairs'] if p['expert']==expert]
  for bi,branch in enumerate(BRANCHES):
   anchor_step=None;anchor=None;weight=1/(len(SOURCES)*len(BRANCHES)*len(pairs))
   for pair in pairs:
    if pair['anchor_step']!=anchor_step:anchor_step=pair['anchor_step'];anchor=load_sample(run,expert,branch,layer,anchor_step,torch,verified)
    current=load_sample(run,expert,branch,layer,pair['step'],torch,verified)
    if tuple(anchor['grid'])!=tuple(current['grid']) or not torch.equal(anchor['ids'],current['ids']):raise ValueError('anchor pair token mismatch')
    dz=current['preactivation'].to('cuda',torch.float64)-anchor['preactivation'].to('cuda',torch.float64)
    dh=current['postactivation'].to('cuda',torch.float64)-anchor['postactivation'].to('cuda',torch.float64)
    n=(dz*dh).mean(0);d=(dz*dz).mean(0);h=(dh*dh).mean(0)
    if not all(bool(torch.isfinite(v).all()) for v in (n,d,h)):raise ValueError('nonfinite secant moments')
    numerator.add_(n,alpha=weight);denominator.add_(d,alpha=weight);postenergy.add_(h,alpha=weight)
    nn,dd,hh=(x.cpu().numpy() for x in (n,d,h));raw_num.append(nn);raw_den.append(dd);raw_energy.append(hh)
    offset=pair['offset'];offset_num[si,bi,offset]+=nn;offset_den[si,bi,offset]+=dd;offset_energy[si,bi,offset]+=hh;offset_counts[si,bi,offset]+=1
    records.append({'source':source,'branch':branch,'layer':layer,**pair,'weight':weight,'rows':64,'actual_nonanchor_steps_in_expert':len(pairs)})
    receipt.check_deadline()
 zero=denominator==0
 if bool((numerator[zero]!=0).any()):raise ValueError('zero denominator with nonzero numerator')
 alpha=torch.zeros_like(numerator);alpha[~zero]=numerator[~zero]/denominator[~zero]
 if not bool(torch.isfinite(alpha).all()):raise ValueError('nonfinite alpha; no clipping/ridge fallback')
 residual=postenergy-2*alpha*numerator+alpha*alpha*denominator
 scale=postenergy.abs()+2*(alpha*numerator).abs()+(alpha*alpha*denominator).abs();tol=1e-10*torch.maximum(scale,torch.ones_like(scale))
 if bool((residual < -tol).any()) or bool((residual-postenergy>tol).any()):raise ValueError('secant least-squares certificate')
 normal=numerator-alpha*denominator
 certificate={'zero_denominator_channels':int(zero.sum()),'normal_equation_max_abs':float(normal.abs().max()),'weighted_hidden_increment_energy':float(postenergy.sum()),'weighted_hidden_residual_energy':float(residual.sum()),'alpha_min':float(alpha.min()),'alpha_max':float(alpha.max()),'pair_count':len(records),'training_measure_only':True}
 arrays={'numerator':numerator.cpu().numpy(),'denominator':denominator.cpu().numpy(),'postactivation_increment_energy':postenergy.cpu().numpy(),'zero_denominator':zero.cpu().numpy(),'pair_numerator':np.stack(raw_num),'pair_denominator':np.stack(raw_den),'pair_postactivation_increment_energy':np.stack(raw_energy),'source_branch_offset_numerator_sum':offset_num,'source_branch_offset_denominator_sum':offset_den,'source_branch_offset_postactivation_energy_sum':offset_energy,'source_branch_offset_step_counts':offset_counts}
 return alpha,arrays,records,certificate

def diagnose_site(runs,expert,layer,A,alpha,records,torch,receipt,verified):
 for record in records:
  run=runs[record['source']];branch=record['branch'];a=load_sample(run,expert,branch,layer,record['anchor_step'],torch,verified);t=load_sample(run,expert,branch,layer,record['step'],torch,verified)
  u0,u1=a['u'].cuda(),t['u'].cuda();f0,f1=a['F'].cuda(),t['F'].cuda();g=t['gate'].squeeze().cuda().double()
  l0=(u0@A).float();l1=(u1@A).float();pred=(l1+(f0.float()-l0)).to(torch.bfloat16)
  d=(f1.double()-f0.double())*g;p=(l1.double()-l0.double())*g
  N=(((pred.double()-f1.double())*g)**2).sum(1);D=(d*d).sum(1);S=((f1.double()*g)**2).sum(1);E=(p*p).sum(1);C=(d*p).sum(1)
  dz=t['preactivation'].cuda().double()-a['preactivation'].cuda().double();dh=t['postactivation'].cuda().double()-a['postactivation'].cuda().double()
  u0d,u1d=u0.double(),u1.double();du=u1d-u0d
  vectors={'N_anchored':N,'N_reuse':D,'S':S,'linear_increment_energy':E,'signed_cross':C,'input_mean_anchor':u0d.mean(1),'input_mean_target':u1d.mean(1),'input_norm2_anchor':(u0d*u0d).sum(1),'input_norm2_target':(u1d*u1d).sum(1),'input_delta_norm2':(du*du).sum(1),'input_delta_mean':du.mean(1),'input_anchor_target_inner_product':(u0d*u1d).sum(1),'preactivation_delta_norm2':(dz*dz).sum(1),'postactivation_delta_norm2':(dh*dh).sum(1),'hidden_secant_residual_norm2':((dh-dz*alpha)**2).sum(1)}
  if not all(bool(torch.isfinite(v).all()) for v in vectors.values()):raise ValueError('nonfinite diagnostic')
  values={k:v.cpu().tolist() for k,v in vectors.items()};grid=tuple(t['grid']);T,H,W=grid
  groups={f'{tt},{hh},{ww}':{'count':0,'N_anchored':0.,'N_reuse':0.,'S':0.,'input_norm2_target':0.,'input_delta_norm2':0.,'input_delta_mean_sum':0.,'input_delta_mean_squared_sum':0.,'input_mean_target_sum':0.,'input_mean_target_squared_sum':0.} for tt in range(3) for hh in range(4) for ww in range(4)}
  for i,(tt,hh,ww) in enumerate(t['coordinates']):
   group=groups[f'{tt*3//T},{hh*4//H},{ww*4//W}'];group['count']+=1
   for k in ('N_anchored','N_reuse','S','input_norm2_target','input_delta_norm2'):group[k]+=values[k][i]
   mu=values['input_mean_target'][i];group['input_mean_target_sum']+=mu;group['input_mean_target_squared_sum']+=mu*mu
   dmu=values['input_delta_mean'][i];group['input_delta_mean_sum']+=dmu;group['input_delta_mean_squared_sum']+=dmu*dmu
  totals={k:float(v.sum()) for k,v in vectors.items() if k in ('N_anchored','N_reuse','S','linear_increment_energy','signed_cross')}
  totals['anchored_finite_precision_residual']=totals['N_anchored']-(totals['N_reuse']+totals['linear_increment_energy']-2*totals['signed_cross'])
  append(receipt.path/'pair_diagnostics.jsonl',{'expert':expert,**record,'grid':list(grid),'token_ids':t['ids'].tolist(),'coordinates':t['coordinates'],'current_gate_sha256':tensor_sha(t['gate'],torch),'anchor_gate_sha256':tensor_sha(a['gate'],torch),'current_gate_energy':float((g*g).sum()),'token_statistics':values,'groups':groups,'totals':totals,'native_F':'CAPTURED_FULL_BATCH_BEFORE_GATE_INCLUDES_B2','matrix_batch_rows':64,'training_measure_only':True,'not_closed_loop_trajectory':True})
  receipt.check_deadline()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scope',required=True);ap.add_argument('--scope-sha',required=True);args=ap.parse_args()
 allocation=require_compute_allocation();root=Path(__file__).resolve().parent;scope=verify_frozen_scope(root,args.scope,args.scope_sha)
 if scope.get('phase')!='ORIGINAL40_MATRIX_BUILD' or scope.get('mode')!='CAL_SECANT_MATRIX_BUILD' or scope['required_runtime_config']!=BUILD_CONFIG:raise ValueError('build contract')
 b=scope['budget'];job=dict(re.findall(r'(\w+)=([^\s]+)',allocation['scheduler_job_record']));match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)',job.get('TimeLimit',''))
 if not match:raise ValueError('scheduler wall limit')
 days,h,m,s=(int(x or 0) for x in match.groups())
 if days*86400+h*3600+m*60+s>b['wall_seconds'] or int(job['NumCPUs'])>b['cpus'] or allocation['memory_mib']>b['host_memory_gib']*1024 or Path(sys.executable).resolve()!=Path(scope['python_executable']).resolve():raise ValueError('allocation/python identity')
 if not 0<b['gpu_memory_gib']<=32 or not 0<b['output_bytes']<=16*1024**3:raise ValueError('build budget')
 with RunReceipt(scope['output_dir'],scope,allocation) as receipt:
  inputs_path=root/scope['input_manifest_file'];inputs=read(inputs_path);runs,identities=verify_inputs(inputs)
  if any(scope['runtime_versions'].get(k)!=runs[s]['scope']['runtime_versions'][k] for s in SOURCES for k in ('torch','numpy')):raise ValueError('build/capture numeric runtime version mismatch')
  # Every true tensor/model operation follows allocation, exact review and START.
  import torch,numpy as np
  from safetensors import safe_open
  if torch.__version__!=scope['runtime_versions']['torch'] or np.__version__!=scope['runtime_versions']['numpy'] or importlib.metadata.version('safetensors')!=scope['runtime_versions']['safetensors']:raise ValueError('build runtime versions')
  if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='NVIDIA A100-SXM4-80GB':raise ValueError('one A100 required')
  limit=b['gpu_memory_gib']*1024**3
  if limit>=torch.cuda.get_device_properties(0).total_memory:raise ValueError('GPU memory limit')
  torch.cuda.set_per_process_memory_fraction(limit/torch.cuda.get_device_properties(0).total_memory);torch.set_num_threads(b['cpus'])
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
  write_json(receipt.path/'ENVIRONMENT.json',{'torch':torch.__version__,'numpy':np.__version__,'safetensors':importlib.metadata.version('safetensors'),'gpu':torch.cuda.get_device_name(0),'tf32':False,'allow_bf16_reduced_precision_reduction':True,'weights':'F32_STORAGE_TO_NATIVE_BF16_BEFORE_F64_SYNTHESIS'})
  # Preserve the actual CAL manifest bytes, so this identity is the independently
  # observed original payload manifest SHA, not a reserialized surrogate.
  original_manifest=Path(inputs['sources'][SOURCES[0]]['receipts']['weight_identities']['path'])
  with (receipt.path/'original_weight_manifest.json').open('xb') as f:f.write(original_manifest.read_bytes())
  weight_manifest_sha=sha(receipt.path/'original_weight_manifest.json')
  if weight_manifest_sha!=scope['original_weight_manifest_sha256']:raise ValueError('scope original weight manifest identity')
  index,weight_stats=weight_index(runs,identities,receipt)
  scope_hashes={s:inputs['sources'][s]['receipts']['scope']['sha256'] for s in SOURCES}
  if scope_hashes!=scope['cal_scope_sha256s']:raise ValueError('scope CAL identity')
  plans={s:{'anchors':runs[s]['anchors'],'pairs':runs[s]['pairs']} for s in SOURCES};write_json(receipt.path/'pair_plans.json',plans)
  verified={};manifest={};alpha_manifest={};sites=[];started=time.monotonic()
  with torch.no_grad():
   for expert in EXPERTS:
    for layer in range(40):
     receipt.check_deadline();alpha,arrays,records,certificate=moments_for_site(runs,expert,layer,torch,np,receipt,verified)
     alpha_path=receipt.path/f'alpha_{expert}_{layer:02d}.npy';moments_path=receipt.path/f'moments_{expert}_{layer:02d}.npz';records_path=receipt.path/f'pair_records_{expert}_{layer:02d}.json'
     with alpha_path.open('xb') as f:np.save(f,alpha.cpu().numpy().astype('<f8',copy=False),allow_pickle=False)
     with moments_path.open('xb') as f:np.savez(f,**arrays)
     write_json(records_path,{'records':records,'offset_moment_semantics':'sum of per-step token means; divide by source_branch_offset_step_counts only for conditional offset descriptions; fitting weights are in records','certificate':certificate})
     w1,id1=load_native_weight(index[expert,layer,'0.weight'],torch,safe_open);w2,id2=load_native_weight(index[expert,layer,'2.weight'],torch,safe_open)
     provenance={'original_weight_manifest_sha256':weight_manifest_sha,'cal_scope_sha256s':scope_hashes,'alpha_sha256':sha(alpha_path),'alpha_path':str(alpha_path),'moments_sha256':sha(moments_path),'pair_records_sha256':sha(records_path),'sources':list(SOURCES),'fit':BUILD_CONFIG['fit'],'expert':expert,'layer':layer,'weight_role':'ORIGINAL','native_weight_conversion':'F32_TO_BF16_RNE','w1':id1,'w2':id2}
     A,asset=save_matrix(receipt.path/f'A_{expert}_{layer:02d}.bf16',w1,w2,alpha,provenance,require_compute_allocation);del w1,w2
     diagnose_site(runs,expert,layer,A,alpha,records,torch,receipt,verified)
     manifest[f'{expert}/{layer}']=asset
     alpha_manifest[f'{expert}/{layer}']={'path':str(alpha_path),'dtype':'F64_NPY','shape':[13824],'sha256':sha(alpha_path),'moments_path':str(moments_path),'moments_sha256':sha(moments_path),'pair_records_path':str(records_path),'pair_records_sha256':sha(records_path)}
     sites.append({'site':f'{expert}/{layer}','alpha_sha256':sha(alpha_path),'moments_sha256':sha(moments_path),'pair_records_sha256':sha(records_path),**certificate})
     append(receipt.path/'BUILD_PROGRESS.jsonl',{'event':'SITE_COMPLETE','site':f'{expert}/{layer}','A_sha256':asset['sha256'],'completed_sites':len(manifest),'certificate':certificate})
     receipt.result={'status':'IN_PROGRESS','sites':len(manifest)}
     if torch.cuda.max_memory_allocated()>limit or sum(p.stat().st_size for p in receipt.path.iterdir() if p.is_file())>b['output_bytes']:raise MemoryError('build budget exceeded')
     del A,alpha,arrays,records;torch.cuda.empty_cache()
  for name,stat in weight_stats.items():
   p=Path(name);now=p.stat()
   if (now.st_size,now.st_mtime_ns)!=stat:raise ValueError('original shard changed during build')
  if len(verified)!=6400 or len(manifest)!=80:raise ValueError('complete sample/matrix coverage required')
  write_json(receipt.path/'matrix_manifest.json',manifest);write_json(receipt.path/'alpha_manifest.json',alpha_manifest)
  write_json(receipt.path/'build_provenance.json',{'input_manifest_sha256':sha(inputs_path),'original_weight_manifest_sha256':weight_manifest_sha,'cal_scope_sha256s':scope_hashes,'pair_plan_sha256':sha(receipt.path/'pair_plans.json'),'sites':sites,'claims':'CAL TRAINING MEASURE ONLY; no original40 generated-candidate quality, low rank or speed claim','diagnostic_batch_boundary':'F0/F1 captured native full batch; A evaluated on 64 sampled rows, batch rounding may differ','finite_precision_boundary':'anchored difference from D+E-2C includes F32 cache arithmetic and BF16 final rounding; separate from hidden secant residual'})
  torch.cuda.synchronize();receipt.check_deadline()
  receipt.result={'status':'ORIGINAL40_MATRIX_BUILD_COMPLETE_NEEDS_INDEPENDENT_REVIEW','sites':80,'verified_capture_files':len(verified),'matrix_manifest_sha256':sha(receipt.path/'matrix_manifest.json'),'alpha_manifest_sha256':sha(receipt.path/'alpha_manifest.json'),'provenance_sha256':sha(receipt.path/'build_provenance.json'),'original_weight_manifest_sha256':weight_manifest_sha,'pair_diagnostics_sha256':sha(receipt.path/'pair_diagnostics.jsonl'),'pair_plans_sha256':sha(receipt.path/'pair_plans.json'),'build_progress_sha256':sha(receipt.path/'BUILD_PROGRESS.jsonl'),'fit_and_matrix_seconds':time.monotonic()-started,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'output_bytes':sum(p.stat().st_size for p in receipt.path.iterdir() if p.is_file())}
  if receipt.result['output_bytes']>b['output_bytes']:raise RuntimeError('final output budget')

if __name__=='__main__':main()
