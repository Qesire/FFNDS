"""Bounded, separately guarded selector/copy stream diagnostics (no Wan NFE).

For each 247-row and one-row layout, run default/default/nondefault for both
quantile=0, aligned binary support, and production 0.7/0.05 settings.
18 selector/copy invocations maximum; persist diagnostic evidence before failure.
"""
import hashlib,json,os,sys,time
from pathlib import Path
import torch
from chip_provider import load_author_provider

def digest_tensor(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def compare_prefixes(left,right):
 """Pure Python ordered-prefix and set comparison; never changes kernel data."""
 need_counts=len(left)==len(right);order_rows=mask_rows=0;max_symdiff=0;samples=[]
 if not need_counts:raise ValueError('row count mismatch')
 for i,(a,b) in enumerate(zip(left,right)):
  if a!=b:order_rows+=1
  diff=set(a)^set(b)
  if diff:
   mask_rows+=1;max_symdiff=max(max_symdiff,len(diff))
   if len(samples)<4:samples.append({'row':i,'symmetric_difference_count':len(diff),'only_left_sample':sorted(set(a)-set(b))[:8],'only_right_sample':sorted(set(b)-set(a))[:8]})
 return {'ordered_prefix_equal':order_rows==0,'selected_mask_equal':mask_rows==0,'ordered_prefix_mismatch_rows':order_rows,'mask_mismatch_rows':mask_rows,'max_mask_symmetric_difference':max_symdiff,'mask_mismatch_examples':samples}

def run(scope,*,authorize,check_resources):
 authorize(scope)
 if scope['max_selector_calls']!=18 or scope['max_copy_calls']!=18 or scope['wan_nfe']!=0 or scope['videos']!=0 or scope['retries']!=0:raise ValueError('bounded fixture scope mismatch')
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):raise RuntimeError('one reviewed A100 required')
 bridge_path=Path(sys.modules['chip_provider'].__file__).resolve();bridge_sha=hashlib.sha256(bridge_path.read_bytes()).hexdigest()
 if bridge_sha!=scope['stream_bridge_sha256']:raise ValueError('bridge source identity mismatch')
 load_author_provider(scope['provider']);ops=sys.modules['chipmunk.ops'];device=torch.device('cuda',torch.cuda.current_device());out=Path(scope['output']);out.mkdir(parents=True,exist_ok=False)
 data={'status':'RUNNING','selector_sha256':scope['provider']['library_sha256'],'stream_bridge_sha256':bridge_sha,'selector_attempts':0,'copy_attempts':0,'wan_nfe':0,'videos':0,'production_default_repeatability':'INCONCLUSIVE','production_cross_stream_mask_parity':'INCONCLUSIVE','production_mask_repeatability':'INCONCLUSIVE','cases':[],'exceptions':[],'interpretation':'synthetic selector/copy mask and copy diagnostics; raw index order is descriptive and full Wan numerical equivalence is not established'}
 def one(rows,stream,case,quantile,random_probability):
  check_resources();host_start=time.monotonic()
  with torch.cuda.stream(stream):
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record(stream)
   columns=torch.arange(13824,device=device,dtype=torch.int32).view(1,1,-1);rows_id=torch.arange(rows,device=device,dtype=torch.int32).view(1,-1,1)
   if case=='aligned_support':
    x=((columns<512)|((columns>=1024)&(columns<4608))).expand(1,rows,-1).to(torch.bfloat16).contiguous()
   else:
    x=(((columns*17+rows_id*31)%257)-128).to(torch.bfloat16).contiguous()
   indices=torch.empty_like(x,dtype=torch.int32);counts=torch.empty((1,rows),device=device,dtype=torch.int32);cache=torch.full_like(x,-300)
   data['selector_attempts']+=1;ops.topk_indices(x,indices,counts,quantile,256,random_probability)
   data['copy_attempts']+=1;ops.copy_indices(x,cache,indices,counts)
   clones=tuple(v.clone() for v in (x,indices,counts,cache));end.record(stream)
  stream.synchronize();originals=tuple(v.cpu() for v in (x,indices,counts,cache));copies=tuple(v.cpu() for v in clones)
  caller_clone_equal=all(torch.equal(a,b) for a,b in zip(originals,copies));xx,ii,cc,yy=originals;prefixes=[];expected=torch.full_like(xx,-300);tail_sentinel=True;unique_valid=True
  for i in range(rows):
   count=int(cc[0,i].item())
   if count<0 or count>13824 or count%256:raise ValueError('invalid count')
   prefix=ii[0,i,:count].tolist();prefixes.append(prefix)
   if len(set(prefix))!=count or any(v<0 or v>=13824 for v in prefix):unique_valid=False
   if not torch.all(ii[0,i,count:]==-1).item():tail_sentinel=False
   idx=torch.tensor(prefix,dtype=torch.int64);expected[0,i,idx]=xx[0,i,idx]
  check_resources();return {'prefixes':prefixes,'counts':cc.flatten().tolist(),'counts_values':cc.flatten().tolist(),'seed_first_two_bf16_as_int32':xx[0,:,:2].contiguous().view(torch.int32).flatten().tolist(),'x':xx,'cache':yy,'caller_clone_equal':caller_clone_equal,'unused_indices_sentinel_intact':tail_sentinel,'valid_unique_indices':unique_valid,'cache_matches_prefix_scatter':torch.equal(expected,yy),'input_sha256':digest_tensor(xx),'indices_sha256':digest_tensor(ii),'counts_sha256':digest_tensor(cc),'cache_sha256':digest_tensor(yy),'caller_stream_elapsed_ms':float(start.elapsed_time(end)),'host_elapsed_seconds':time.monotonic()-host_start}
 def compact(v):return {k:x for k,x in v.items() if k not in ('prefixes','counts','x','cache')}
 def compare(a,b):
  result=compare_prefixes(a['prefixes'],b['prefixes']);result.update(input_exact=torch.equal(a['x'],b['x']),counts_exact=a['counts']==b['counts'],copied_cache_exact=torch.equal(a['cache'],b['cache']))
  return result
 error=None
 try:
  for rows in (247,1):
   for case,quantile,rnd in (('deterministic_all',0.,0.),('aligned_support',0.7,0.),('production',0.7,0.05)):
    control=one(rows,torch.cuda.default_stream(device=device),case,quantile,rnd);repeat=one(rows,torch.cuda.default_stream(device=device),case,quantile,rnd);alternate=one(rows,torch.cuda.Stream(device=device),case,quantile,rnd)
    c1=compare(control,repeat);c2=compare(control,alternate);checks=(control,repeat,alternate)
    basic=all(x['caller_clone_equal'] and x['unused_indices_sentinel_intact'] and x['valid_unique_indices'] and x['cache_matches_prefix_scatter'] for x in checks) and all(v['input_exact'] and v['counts_exact'] for v in (c1,c2))
    analytic=True
    if case!='production':
     expected=(set(range(512))|set(range(1024,4608))) if case=='aligned_support' else set(range(13824))
     analytic=all(all(v==len(expected) for v in x['counts']) and all(set(row)==expected for row in x['prefixes']) for x in checks)
    else:
     for x in checks:
      for vector,prefix in zip(x['x'][0],x['prefixes'],strict=True):
       threshold=vector[:1024].sort().values[716]
       qualified=set((vector>=threshold).nonzero().flatten().tolist())
       analytic=analytic and qualified.issubset(prefix)
    bounded=all(v['max_mask_symmetric_difference']<=510 for v in (c1,c2)) if case=='production' else True
    masks_equal=all(v['selected_mask_equal'] and v['copied_cache_exact'] for v in (c1,c2))
    data['cases'].append({'selector_rows':rows,'layout':'247 full blocks' if rows==247 else 'valid52 single block','case':case,'quantile':quantile,'random_probability':rnd,'default_first':compact(control),'default_repeat':compact(repeat),'nondefault':compact(alternate),'default_vs_repeat':c1,'default_vs_nondefault':c2,'basic_valid':basic,'analytic_support_exact':analytic,'padding_symdiff_within_510':bounded,'mask_and_copy_parity':masks_equal,'ordered_prefix_parity_is_descriptive':True})
    del control,repeat,alternate
  if data['selector_attempts']!=18 or data['copy_attempts']!=18:raise RuntimeError('fixture operator budget differs')
  if not all(c['basic_valid'] and c['analytic_support_exact'] and c['padding_symdiff_within_510'] for c in data['cases']):raise AssertionError('invalid selector/copy, analytic support, or padding bound; see saved diagnostics')
  if not all(c['mask_and_copy_parity'] for c in data['cases'] if c['case']!='production'):raise AssertionError('deterministic masks or copy differ across streams')
  prod=[c for c in data['cases'] if c['case']=='production']
  data['production_default_repeatability']='IDENTICAL' if all(c['default_vs_repeat']['selected_mask_equal'] for c in prod) else 'VARIATION_OBSERVED'
  data['production_cross_stream_mask_parity']='IDENTICAL' if all(c['default_vs_nondefault']['selected_mask_equal'] for c in prod) else 'VARIATION_OBSERVED'
  data['production_mask_repeatability']='IDENTICAL' if all(c['mask_and_copy_parity'] for c in prod) else 'VARIATION_OBSERVED'
  data['status']='PASS_STREAM_TRANSPORT'
 except Exception as exc:
  error=exc;data['status']='FAIL' if data['status']=='RUNNING' else data['status'];data['exceptions'].append({'type':type(exc).__name__,'message':str(exc)})
 finally:
  with (out/'SUMMARY.json').open('x',encoding='utf8') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 if error is not None:raise error
 return data
