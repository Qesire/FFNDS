"""Prospective guarded GPU engineering fixture; never dispatches itself.

Uses only synthetic selector/cache arrays; no Wan model, generation or NFE.
Two geometries x (default, nondefault) => four selector and copy calls.
"""
import hashlib,json,os,sys
from pathlib import Path
import torch
from chip_provider import load_author_provider

def run(scope,*,authorize,check_resources):
 authorize(scope)
 if scope['max_selector_calls']!=4 or scope['max_copy_calls']!=4 or scope['wan_nfe']!=0 or scope['videos']!=0 or scope['retries']!=0:raise ValueError('bounded fixture scope mismatch')
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0):raise RuntimeError('one reviewed A100 required')
 bridge_path=Path(sys.modules['chip_provider'].__file__).resolve()
 bridge_sha=hashlib.sha256(bridge_path.read_bytes()).hexdigest()
 if bridge_sha!=scope['stream_bridge_sha256']:raise ValueError('bridge source identity mismatch')
 load_author_provider(scope['provider']);ops=sys.modules['chipmunk.ops'];device=torch.device('cuda',torch.cuda.current_device())
 out=Path(scope['output']);out.mkdir(parents=True,exist_ok=False);results=[];calls=0
 def one(rows,stream):
  nonlocal calls
  check_resources()
  with torch.cuda.stream(stream):
   # Generated on the actual caller stream; no presynchronization before the bridge.
   columns=torch.arange(13824,device=device,dtype=torch.int32).view(1,1,-1)
   rows_id=torch.arange(rows,device=device,dtype=torch.int32).view(1,-1,1)
   x=(((columns*17+rows_id*31)%257)-128).to(torch.bfloat16).contiguous()
   indices=torch.empty_like(x,dtype=torch.int32);counts=torch.empty((1,rows),device=device,dtype=torch.int32);cache=torch.full_like(x,-3)
   ops.topk_indices(x,indices,counts,0.7,256,0.05);ops.copy_indices(x,cache,indices,counts);calls+=1
   # Consumer clones are queued on caller; inspect only after completion.
   consumed=(indices.clone(),counts.clone(),cache.clone())
  stream.synchronize()
  return consumed
 for rows in (247,1):
  baseline=one(rows,torch.cuda.default_stream(device=device));alternate=one(rows,torch.cuda.Stream(device=device))
  if not all(torch.equal(a,b) for a,b in zip(baseline,alternate)):raise AssertionError('default/nondefault compiled results differ')
  results.append({'selector_rows':rows,'layout':'247 full blocks' if rows==247 else 'valid52 single block','indices_exact':True,'counts_exact':True,'copy_cache_exact':True,'caller_consumer_clones_exact':True,'counts_min':int(alternate[1].min().item()),'counts_max':int(alternate[1].max().item())})
  del baseline,alternate
 if calls!=4:raise RuntimeError('fixture operator budget differs')
 data={'status':'PASS','selector_sha256':scope['provider']['library_sha256'],'stream_bridge_sha256':bridge_sha,'selector_calls':4,'copy_calls':4,'wan_nfe':0,'videos':0,'exceptions':[],'rows':results,'interpretation':'synthetic indexed-operator stream ordering only; does not establish full Wan or E6 numerical equivalence'}
 with (out/'SUMMARY.json').open('x',encoding='utf8') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 return data
