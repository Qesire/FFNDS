"""High expert pre-gate Y-cache-only Wan FFN; no execution at import."""
import threading,time
from chipmunk_p64_factorial_tap import need,memory_snapshot,validate_active_compute_buffer
_LOCK=threading.Lock()
class YCache:
 def __init__(self,limit):self.limit=limit;self.items={};self.bytes=0;self.peak=0
 def put(self,block,y):
  need(type(block) is int and 0<=block<40 and block not in self.items,'duplicate Y cache');n=y.numel()*y.element_size();need(self.bytes+n<=self.limit,'Y cache capacity');self.items[block]=y;self.bytes+=n;self.peak=max(self.peak,self.bytes)
 def get(self,block):need(block in self.items,'missing high-expert Y');return self.items[block]
 def release(self,block):
  y=self.items.pop(block);self.bytes-=y.numel()*y.element_size()
 def empty(self):need(not self.items and self.bytes==0,'Y cache not empty')

def make_tap(vm,native,*,mode,emit,geometry=(31668,5120,13824)):
 m,d,h=geometry;need(mode in ('none','high_y_cache_dense','high_y_reuse') and min(geometry)>0,'mode/geometry')
 class Tap(vm.LightTap):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw);self.book=YCache(40*m*d*2);self.pending=None;self.cache_puts=0;self.cache_releases=0;self.reused=0;self.thread_id=threading.get_ident()
  def charged(self,cost,fn):
   def call():value=fn();self.guard.torch.cuda.synchronize();return value
   return self.guard.ledger.call(self.rid,cost,call)
  def mm_call(self,mm,value,meta,original):
   slot=self.mapping.get(id(mm));suffix=slot[2] if slot is not None and slot[0]=='block' else None;p=self.pending
   if p is None or suffix not in ('ffn.0','ffn.2'):return super().mm_call(mm,value,meta,original)
   need(self.block==(p['expert'],p['block']) and self.step(p['expert'])==p['step'],'FFN transaction')
   torch=self.guard.torch;step=p['step'];expert=p['expert'];block=p['block'];active=expert==0 and mode!='none'
   if active and step==1 and mode=='high_y_reuse' and suffix=='ffn.0':
    need(mm is p['phase'].ffn_0 and p.get('sentinel') is None and tuple(value.shape)==(m,d),'W1 transaction')
    old=self.book.get(block);out=self.charged({'reuse_y_clone':1,'reuse_y_clone_bytes':m*d*2},lambda:old.detach().clone());need(tuple(out.shape)==(m,d) and out.dtype==torch.bfloat16,'reuse Y identity')
    p['out']=out;p['sentinel']=out[:,0:1].expand(-1,h)
    def bypass(_mm,_x):return p['sentinel']
    return super().mm_call(mm,value,meta,bypass)
   if active and step==1 and mode=='high_y_reuse' and suffix=='ffn.2':
    need(mm is p['phase'].ffn_2 and value is p.get('sentinel') and p.get('out') is not None,'W2 sentinel')
    out=super().mm_call(mm,value,meta,lambda _mm,_x:p['out']);self.charged({'cache_release_sync':1},lambda:None);self.book.release(block);self.cache_releases+=1;self.reused+=1;return out
   if active and step==0 and suffix=='ffn.2':
    out=super().mm_call(mm,value,meta,original);need(tuple(out.shape)==(m,d) and out.dtype==torch.bfloat16 and (geometry!=(31668,5120,13824) or out.device.type=='cuda'),'pre-gate Y boundary')
    need(self.book.bytes+m*d*2<=self.book.limit,'Y capacity preclone');saved=self.charged({'cache_y_clone':1,'cache_y_clone_bytes':m*d*2},lambda:out.detach().clone());self.book.put(block,saved);self.cache_puts+=1;return out
   if active and step==1 and mode=='high_y_cache_dense' and suffix=='ffn.2':
    out=super().mm_call(mm,value,meta,original);self.charged({'cache_release_sync':1},lambda:None);self.book.release(block);self.cache_releases+=1;return out
   return super().mm_call(mm,value,meta,original)
  def install(self):
   super().install()
   for expert,model in enumerate(self.runner.model.model):
    infer=model.transformer_infer;old=infer.infer_ffn
    def wrapped(phase,x,attn,shift,scale,gate=None,*,_e=expert,_i=infer,_old=old):
     step=self.step(_e);block=int(_i.block_idx)
     if _e==0 and step==1 and mode!='none':validate_active_compute_buffer(self.guard.torch,_i,phase)
     need(threading.get_ident()==self.thread_id and self.pending is None and self.block==(_e,block),'FFN transaction scope')
     p={'phase':phase,'expert':_e,'step':step,'block':block};self.pending=p;F=self.guard.torch.nn.functional;old_gelu=F.gelu;locked=False
     def gelu(x,*args,**kwargs):
      if x is p.get('sentinel') and p.get('sentinel') is not None:
       need(mode=='high_y_reuse' and _e==0 and step==1 and kwargs.get('approximate')=='tanh','sentinel GELU');return x
      return old_gelu(x,*args,**kwargs)
     try:
      if mode=='high_y_reuse' and _e==0 and step==1:need(_LOCK.acquire(blocking=False),'concurrent GELU');locked=True;F.gelu=gelu
      out=_old(phase,x,attn,shift,scale,gate)
      if mode=='high_y_reuse' and _e==0 and step==1:need(out is p.get('out'),'pre-gate output identity')
      if mode!='none' and _e==0 and step==1 and block==39:
       self.book.empty();before=memory_snapshot(self.guard.torch);emit({'event':'expert_boundary_pre_cleanup','mode':mode,'step':step,'memory':before})
       def cleanup():
        t=time.perf_counter();self.guard.torch.cuda.synchronize();self.guard.torch.cuda.empty_cache();self.guard.torch.cuda.synchronize();return time.perf_counter()-t
       elapsed=self.guard.ledger.call(self.rid,{'expert_boundary_cleanup':1},cleanup);emit({'event':'expert_boundary_cleanup','mode':mode,'step':step,'before':before,'after':memory_snapshot(self.guard.torch),'wall_seconds_including_synchronization':elapsed})
      return out
     finally:
      if locked:F.gelu=old_gelu;_LOCK.release()
      self.pending=None
    self.stack.enter_context(native.patch_method(infer,'infer_ffn',wrapped))
  def finish(self):
   super().finish();self.book.empty();need((self.cache_puts,self.cache_releases,self.reused)==((0,0,0) if mode=='none' else (40,40,40 if mode=='high_y_reuse' else 0)),'Y cache coverage')
 return Tap
