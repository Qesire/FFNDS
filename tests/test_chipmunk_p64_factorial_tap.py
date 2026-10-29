import sys,types,unittest,contextlib
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from chipmunk_p64_factorial_tap import CacheBook,make_tap,validate_live_w2,validate_active_compute_buffer
class Ledger:
 def __init__(self):self.c=[]
 def call(self,rid,cost,fn,*a,**kw):self.c.append(cost);return fn(*a,**kw)
class Base:
 def __init__(self):self.guard=types.SimpleNamespace(torch=torch,ledger=Ledger());self.rid='test';self.block=(0,0);self.curstep=0;self.mapping={}
 def step(self,e=None):return self.curstep
 def mm_call(self,mm,value,meta,original):return original(mm,value)
class M:
 def __init__(self,w,b):self.weight=w;self.bias=b
class Test(unittest.TestCase):
 def test_cache_book_identity_capacity(self):
  b=CacheBook(100);h=torch.zeros(3,dtype=torch.bfloat16);y=torch.zeros(4,dtype=torch.bfloat16);b.put(0,0,h,y)
  self.assertEqual(b.bytes,14)
  with self.assertRaises(ValueError):b.get(1,0)
  with self.assertRaises(ValueError):b.put(0,0,h,y)
  b.release(0,0);b.empty()
 def test_dense_cache_then_candidate_dispatch_and_release(self):
  m,d,h,k=64,2,3,1;rows=torch.arange(m);w1=M(torch.ones((d,h),dtype=torch.bfloat16),torch.zeros(h,dtype=torch.bfloat16));w2=M(torch.ones((h,d),dtype=torch.bfloat16),torch.zeros(d,dtype=torch.bfloat16));phase=types.SimpleNamespace(ffn_0=w1,ffn_2=w2)
  events=[];cls=make_tap(types.SimpleNamespace(LightTap=Base),None,mode='high_only',rows=rows,kernels=None,emit=events.append,geometry=(m,d,h,k));tap=cls();tap.mapping={id(w1):('block',0,'ffn.0'),id(w2):('block',0,'ffn.2')};tap.pending={'phase':phase,'expert':0,'step':0,'block':0}
  oldh=torch.ones((m,h),dtype=torch.bfloat16);oldy=torch.full((m,d),2,dtype=torch.bfloat16)
  fake=types.ModuleType('chipmunk_wan_packed_pregate');fake.packed_pregate=lambda x,w1,b1,w2,h,y,indices:y.clone()+1
  observer=types.ModuleType('chipmunk_packed_native_math');observer.packed_action=lambda x,w1,b1,w2,h,y,gate,indices:y.clone()+1
  with patch.object(torch.cuda,'synchronize',lambda:None),patch('chipmunk_p64_factorial_tap.validate_live_w2',lambda *args:None),patch.dict(sys.modules,{'chipmunk_wan_packed_pregate':fake,'chipmunk_packed_native_math':observer}):
   tap.mm_call(w2,oldh,{},lambda mm,x:oldy);self.assertEqual(tap.cache_puts,1)
   tap.curstep=1;tap.pending={'phase':phase,'expert':0,'step':1,'block':0};sent=tap.mm_call(w1,torch.ones((m,d),dtype=torch.bfloat16),{},lambda mm,x:None)
   self.assertEqual(tuple(sent.shape),(m,h));out=tap.mm_call(w2,sent,{},lambda mm,x:None);self.assertTrue(torch.equal(out,oldy+1));tap.book.empty();self.assertEqual(tap.sparse_sites,1)
   self.assertEqual(len([e for e in events if e['event']=='packed_pregate_control']),1)
   tap.book.put(0,0,oldh,oldy);tap.pending={'phase':phase,'expert':0,'step':1,'block':0};observer.packed_action=lambda x,w1,b1,w2,h,y,gate,indices:y.clone()+2
   with self.assertRaisesRegex(ValueError,'packed pregate control mismatch'):tap.mm_call(w1,torch.ones((m,d),dtype=torch.bfloat16),{},lambda mm,x:None)
   self.assertGreater([e for e in events if e['event']=='packed_pregate_control'][-1]['relative_l2'],1e-5)
 def test_shim_restored_after_original_ffn_throws(self):
  def broken(*a):raise RuntimeError('mid-FFN')
  inf=types.SimpleNamespace(block_idx=0,infer_ffn=broken)
  class InstallBase(Base):
   def __init__(self):
    super().__init__();self.curstep=1;self.stack=contextlib.ExitStack();self.runner=types.SimpleNamespace(model=types.SimpleNamespace(model=[types.SimpleNamespace(transformer_infer=inf)]))
   def install(self):pass
  @contextlib.contextmanager
  def method(obj,name,fn):
   old=getattr(obj,name);setattr(obj,name,fn)
   try:yield
   finally:setattr(obj,name,old)
  cls=make_tap(types.SimpleNamespace(LightTap=InstallBase),types.SimpleNamespace(patch_method=method),mode='high_only',rows=torch.arange(64),kernels=None,emit=lambda e:None,geometry=(64,2,3,1));tap=cls();old=torch.nn.functional.gelu
  tap.install()
  with patch('chipmunk_p64_factorial_tap.validate_active_compute_buffer',lambda *args:None):
   with self.assertRaises(RuntimeError):inf.infer_ffn(None,None,None,None,None,None)
  self.assertIs(torch.nn.functional.gelu,old);self.assertIsNone(tap.pending);tap.stack.close()
 def test_invalid_w2_rejected_before_dispatch(self):
  class FakeTensor:
   def __init__(self,shape,stride):self.shape=shape;self._stride=stride;self.dtype='bf16';self.device=types.SimpleNamespace(type='cuda',index=0)
   def stride(self):return self._stride
   def data_ptr(self):return id(self)
  MMWeight=type('MMWeight',(),{'__module__':'lightx2v.common.ops.mm.mm_weight'})
  mm=MMWeight();mm.weight=FakeTensor((3,2),(1,3));mm.bias=FakeTensor((2,),(1,));mm.weight_cuda_buffer=mm.weight;mm.bias_cuda_buffer=mm.bias;mm.has_lora_branch=False;mm.has_diff=False
  fake=types.SimpleNamespace(is_tensor=lambda x:isinstance(x,FakeTensor),bfloat16='bf16');mapping={id(mm):('block',0,'ffn.2')}
  validate_live_w2(fake,mm,mapping,0,3,2)
  mm.weight._stride=(2,1)
  with self.assertRaisesRegex(ValueError,'W2 layout'):validate_live_w2(fake,mm,mapping,0,3,2)
  mm.weight._stride=(1,3);mm.weight.dtype='fp32'
  with self.assertRaisesRegex(ValueError,'W2 layout'):validate_live_w2(fake,mm,mapping,0,3,2)
  with self.assertRaisesRegex(ValueError,'W2 buffer mapping'):validate_live_w2(fake,mm,{},0,3,2)
 def test_active_compute_buffer_and_stream(self):
  phase=object();buf=types.SimpleNamespace(compute_phases=[None,None,phase]);stream=object();manager=types.SimpleNamespace(lazy_load=False,cuda_buffers=[buf,object()],compute_stream=stream);infer=types.SimpleNamespace(lazy_load=False,offload_manager=manager);fake=types.SimpleNamespace(cuda=types.SimpleNamespace(current_stream=lambda:stream))
  validate_active_compute_buffer(fake,infer,phase)
  with self.assertRaisesRegex(ValueError,'buffer'):validate_active_compute_buffer(fake,infer,object())
  fake.cuda.current_stream=lambda:object()
  with self.assertRaisesRegex(ValueError,'stream'):validate_active_compute_buffer(fake,infer,phase)
 def test_wrong_w2_is_rejected_before_packed_action(self):
  m,d,h,k=64,2,3,1;w1=M(torch.ones((d,h),dtype=torch.bfloat16),torch.zeros(h,dtype=torch.bfloat16));w2=M(torch.ones((h,d),dtype=torch.bfloat16),torch.zeros(d,dtype=torch.bfloat16));phase=types.SimpleNamespace(ffn_0=w1,ffn_2=w2);events=[]
  cls=make_tap(types.SimpleNamespace(LightTap=Base),None,mode='high_only',rows=torch.arange(m),kernels=None,emit=events.append,geometry=(m,d,h,k));tap=cls();tap.curstep=1;tap.mapping={id(w1):('block',0,'ffn.0')};tap.pending={'phase':phase,'expert':0,'step':1,'block':0};tap.book.put(0,0,torch.ones((m,h),dtype=torch.bfloat16),torch.ones((m,d),dtype=torch.bfloat16))
  with self.assertRaisesRegex(ValueError,'W2 buffer mapping'):tap.mm_call(w1,torch.ones((m,d),dtype=torch.bfloat16),{},lambda mm,x:None)
  self.assertEqual(tap.guard.ledger.c,[]);self.assertEqual(events,[])
 def test_inactive_expert_never_caches_or_dispatches(self):
  m,d,h,k=64,2,3,1;w=M(torch.ones((h,d),dtype=torch.bfloat16),torch.zeros(d,dtype=torch.bfloat16));value=torch.ones((m,h),dtype=torch.bfloat16);expected=torch.zeros((m,d),dtype=torch.bfloat16)
  cls=make_tap(types.SimpleNamespace(LightTap=Base),None,mode='high_only',rows=torch.arange(m),kernels=None,emit=lambda x:None,geometry=(m,d,h,k));tap=cls();tap.block=(1,0);tap.curstep=2;tap.pending={'phase':types.SimpleNamespace(ffn_2=w),'expert':1,'step':2,'block':0};tap.mapping={id(w):('block',1,'ffn.2')};self.assertIs(tap.mm_call(w,value,{},lambda mm,x:expected),expected);tap.book.empty();self.assertEqual(tap.cache_puts,0)
 def test_over_capacity_does_not_mutate(self):
  b=CacheBook(2);x=torch.ones(3,dtype=torch.bfloat16)
  with self.assertRaises(ValueError):b.put(0,0,x,x)
  b.empty()
if __name__=='__main__':unittest.main()
