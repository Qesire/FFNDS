"""Bounded GPU cache preflight using a frozen real full-field capture; compute node only."""
import time
import cache_codec
from gpu_slab import Slab
from pathlib import Path

def run(torch, field_path, field_spec, sha256_file, A, slab_dtype):
 start=time.monotonic();p=Path(field_path);st=p.stat()
 if st.st_size!=field_spec['bytes'] or sha256_file(p)!=field_spec['sha256']:raise RuntimeError('preflight field identity')
 field=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
 if (p.stat().st_size,p.stat().st_mtime_ns)!=(st.st_size,st.st_mtime_ns):raise RuntimeError('preflight field changed')
 u=field['u'].to('cuda');f=field['F'].to('cuda');saved=field['Au'].to('cuda')
 if u.shape!=(31668,5120) or f.shape!=u.shape or saved.shape!=u.shape or any(x.dtype!=torch.bfloat16 for x in (u,f,saved)):raise RuntimeError('preflight BF16 full field')
 if not all(bool(torch.isfinite(x).all()) for x in (u,f,saved,A)):raise RuntimeError('preflight nonfinite')
 pred=u@A
 if not torch.equal(pred,saved):raise RuntimeError('preflight XA differs from frozen capture')
 raw=cache_codec.encode(f,pred,torch.float32); cpu=raw.cpu(); device=cpu.to('cuda')
 if not torch.equal(raw,device):raise RuntimeError('GPU and CPU FP32 storage differ')
 r0=cache_codec.decode(pred,device,torch.float32,torch.bfloat16);r0ref=(saved.float()+raw).to(torch.bfloat16)
 if not torch.equal(r0,r0ref):raise RuntimeError('R0 decoder mismatch')
 r16=cache_codec.encode(f,pred,torch.bfloat16);out=cache_codec.decode(pred,r16,torch.bfloat16,torch.bfloat16);ref=(saved.float()+(f.float()-saved.float()).to(torch.bfloat16).float()).to(torch.bfloat16)
 if not torch.equal(out,ref):raise RuntimeError('R16 decoder mismatch')
 delta=(out.float()-r0.float()).double(); sse=float(delta.square().sum().item()); maximum=float(delta.abs().max().item()); del delta,ref,out,r0,r0ref,cpu,device
 evidence=[]
 if slab_dtype not in (torch.bfloat16,torch.float32):raise ValueError('slab dtype')
 for dtype,payload in ((slab_dtype,r16 if slab_dtype==torch.bfloat16 else raw),):
  slab=Slab(torch,dtype,u.device); slab.bind('high'); slab.put('high','conditional',0,payload); changed=payload.neg(); slab.put('high','unconditional',0,changed)
  if slab.get('high','conditional',0).data_ptr()==slab.get('high','unconditional',0).data_ptr():raise RuntimeError('slab slots alias')
  if not torch.equal(slab.get('high','conditional',0),payload) or not torch.equal(slab.get('high','unconditional',0),changed):raise RuntimeError('slab write/nonalias mismatch')
  slab.put('high','conditional',0,changed)
  if not torch.equal(slab.get('high','conditional',0),changed):raise RuntimeError('slab refresh mismatch')
  del changed; slab.put('high','conditional',0,payload); actual=cache_codec.decode(pred,slab.get('high','conditional',0),dtype,torch.bfloat16)
  expected=(saved.float()+(f.float()-saved.float()).to(dtype).float()).to(torch.bfloat16)
  if not torch.equal(actual,expected):raise RuntimeError('slab refresh/decode mismatch')
  del actual,expected;slab.bind('low')
  try:slab.get('high','conditional',0)
  except RuntimeError:pass
  else:raise RuntimeError('old expert slot remained valid')
  try:slab.get('low','conditional',0)
  except RuntimeError:pass
  else:raise RuntimeError('new expert inherited old slot validity')
  slab.put('low','conditional',0,payload)
  if not torch.equal(slab.get('low','conditional',0),payload):raise RuntimeError('new expert write mismatch')
  evidence.append({'dtype':str(dtype),'slab_bytes':slab.physical_bytes,'nonalias':True,'refresh':True,'expert_invalidation':True});slab.release();del slab;torch.cuda.synchronize();torch.cuda.empty_cache()
 torch.cuda.synchronize();return {'status':'PASS','field_sha256':field_spec['sha256'],'comparison':'elementwise exact; signed zero not distinguished','r16_vs_r0_sse':sse,'r16_vs_r0_max_abs':maximum,'slab_checks':evidence,'seconds':time.monotonic()-start}
