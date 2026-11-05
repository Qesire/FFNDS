"""Draft full-depth history ablation hook; requires independent review before use."""
import time
import torch

class HistoryAblation:
 def __init__(self, mode, load_matrix, load_c, verify_phase, emit, *, expected_layers=40, host_cap_bytes=32*1024**3, shadow_indices=None, neutral=False, anchor_relative_sse_max=1e-12):
  if mode not in ('reuse','static_affine','anchored'): raise ValueError('mode')
  self.mode=mode; self.load_matrix=load_matrix; self.load_c=load_c; self.verify_phase=verify_phase; self.emit=emit
  self.layers=expected_layers; self.cap=host_cap_bytes; self.shadow_indices=shadow_indices; self.neutral=neutral
  self.anchor_tol=float(anchor_relative_sse_max)
  if not 0<=self.anchor_tol<=1e-8: raise ValueError('invalid anchor tolerance')
  self.cache={}; self.seen=set(); self.dispatch=None; self.owner=None; self.live=0; self.peak=0; self.written=0; self.read=0
 def install(self, owner):
  if self.owner is not None: raise RuntimeError('already installed')
  original=owner.infer_ffn; h=self
  def ffn(model,phase,x,attn_out,shift,scale,gate=None):
   step=int(model.scheduler.step_index); layer=int(model.block_idx); expert='high' if step<2 else 'low'; key=(expert,layer)
   if step not in range(4) or layer not in range(h.layers) or h.dispatch!=(0 if step<2 else 1): raise RuntimeError('invalid site')
   if (step,layer) in h.seen: raise RuntimeError('duplicate site')
   h.seen.add((step,layer))
   if step in (0,2) and h.mode=='static_affine':
    result=original(model,phase,x,attn_out,shift,scale,gate); h.emit({'event':'EXACT','step':step,'expert':expert,'layer':layer}); return result
   if gate is None or hasattr(phase,'smooth_norm2_weight') or hasattr(phase,'smooth_norm2_bias') or model._use_mxfp8_quant_fuse(): raise RuntimeError('unsupported FFN')
   h.verify_phase(model,phase,expert,layer,step)
   first0,second0=phase.ffn_0.apply,phase.ffn_2.apply; state={}
   def first(z):
    if 'z' in state or z.ndim!=2 or z.dtype!=torch.bfloat16 or not bool(torch.isfinite(z).all()): raise RuntimeError('invalid W1 input')
    state['z']=z.detach().clone()
    return first0(z) if step in (0,2) or h.neutral else z
   def second(v):
    if state.get('second') or 'z' not in state: raise RuntimeError('invalid W2 order')
    state['second']=True; z=state['z']; start=time.monotonic(); A=None
    if h.mode!='reuse':
     A=h.load_matrix(expert,layer,z.device)
     if A.dtype!=torch.bfloat16 or A.shape!=(z.shape[1],z.shape[1]) or not bool(torch.isfinite(A).all()): raise ValueError('invalid A')
    if step in (0,2):
     y=second0(v)
     if y.shape!=z.shape or y.dtype!=torch.bfloat16 or not bool(torch.isfinite(y).all()): raise ValueError('invalid anchor output')
     diagnostic={}
     if h.mode=='reuse': value=y
     else:
      predicted=(z@A).float(); value=y.float()-predicted; reconstruction=predicted+value
      if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(reconstruction).all()): raise FloatingPointError('nonfinite anchor')
      mismatch=int((reconstruction.to(y.dtype)!=y).sum().item()); maxerr=float((reconstruction-y.float()).abs().max().item())
      delta=reconstruction.double()-y.double(); sse=float((delta*delta).sum()); energy=float((y.double()*y.double()).sum()); relative=sse/energy if energy>0 else None
      import math
      if not all(math.isfinite(v) for v in (sse,energy,maxerr)): raise FloatingPointError('nonfinite anchor diagnostic')
      if (energy==0 and sse!=0) or (relative is not None and relative>h.anchor_tol): raise RuntimeError('ANCHOR_RELATIVE_SSE_EXCEEDED')
      diagnostic={'bf16_mismatch':mismatch,'max_abs_f32_error':maxerr,'anchor_sse':sse,'anchor_energy':energy,'anchor_relative_sse':relative,'anchor_relative_sse_limit':h.anchor_tol}
     size=value.numel()*value.element_size()
     if key in h.cache or h.live+size>h.cap: raise RuntimeError('cache identity or cap')
     h.cache[key]=value.detach().cpu().clone(); h.live+=size; h.written+=size; h.peak=max(h.peak,h.live)
     h.emit({'event':'ANCHOR','step':step,'expert':expert,'layer':layer,'cache_bytes':size,'cache_live_bytes':h.live,'cache_peak_bytes':h.peak,'cache_dtype':str(value.dtype),'cache_device':'cpu','write_payload_bytes':size,'seconds':time.monotonic()-start,**diagnostic}); return y
    dense=second0(v) if h.neutral else None
    if dense is not None and (dense.shape!=z.shape or dense.dtype!=torch.bfloat16 or not bool(torch.isfinite(dense).all())): raise ValueError('invalid neutral output')
    if h.mode=='static_affine':
     c=h.load_c(expert,layer,z.device)
     if c.dtype!=torch.float32 or c.shape!=(z.shape[1],) or not bool(torch.isfinite(c).all()): raise ValueError('invalid c')
     y=((z@A).float()+c).to(torch.bfloat16); size=0
    else:
     if key not in h.cache: raise RuntimeError('missing cache')
     value=h.cache.pop(key); size=value.numel()*value.element_size()
     if value.shape!=z.shape: raise RuntimeError('token grid changed')
     h.live-=size; h.read+=size
     y=value.to(z.device) if h.mode=='reuse' else ((z@A).float()+value.to(z.device)).to(torch.bfloat16)
    if not bool(torch.isfinite(y).all()): raise RuntimeError('nonfinite prediction')
    if h.shadow_indices is not None:
     shadow_start=time.monotonic()
     idx=torch.as_tensor(h.shadow_indices(expert,layer,z.shape[0]),dtype=torch.long,device=z.device)
     if idx.ndim!=1 or idx.numel()!=64 or idx.unique().numel()!=64 or idx.min()<0 or idx.max()>=z.shape[0]: raise ValueError('shadow selection')
     sample=z.index_select(0,idx); reference=second0(torch.nn.functional.gelu(first0(sample),approximate='tanh'))
     g=gate.squeeze()
     if g.ndim!=1 or g.shape[0]!=sample.shape[1] or not bool(torch.isfinite(g).all()) or not bool(torch.isfinite(reference).all()): raise RuntimeError('unsupported gate or shadow output')
     diff=(y.index_select(0,idx).double()-reference.double())*g.double(); ref=reference.double()*g.double(); s=sample.double()
     h.emit({'event':'SHADOW_QUALITY_ONLY','step':step,'expert':expert,'layer':layer,'token_ids':idx.cpu().tolist(),'token_N':(diff*diff).sum(1).cpu().tolist(),'token_S':(ref*ref).sum(1).cpu().tolist(),'token_input_mean':s.mean(1).cpu().tolist(),'token_input_norm2':(s*s).sum(1).cpu().tolist(),'instrumented_seconds':time.monotonic()-shadow_start})
    h.emit({'event':'REPLACEMENT','step':step,'expert':expert,'layer':layer,'cache_live_bytes':h.live,'cache_peak_bytes':h.peak,'read_payload_bytes':size,'neutral':h.neutral,'seconds':time.monotonic()-start})
    return dense if h.neutral else y
   phase.ffn_0.apply=first; phase.ffn_2.apply=second
   try: result=original(model,phase,x,attn_out,shift,scale,gate)
   finally: phase.ffn_0.apply=first0; phase.ffn_2.apply=second0
   if not state.get('second'): raise RuntimeError('missing W2')
   return result
  owner.infer_ffn=ffn; self.owner=owner; self.original=original
 def finish(self):
  if self.cache or self.live or self.seen!={(s,l) for s in range(4) for l in range(self.layers)}: raise RuntimeError('incomplete coverage/cache')
  return {'cache_peak_bytes':self.peak,'write_payload_bytes':self.written,'read_payload_bytes':self.read}
 def uninstall(self):
  if self.owner is not None: self.owner.infer_ffn=self.original; self.owner=None


