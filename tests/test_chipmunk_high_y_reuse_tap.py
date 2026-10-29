import sys,types,unittest,contextlib
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from chipmunk_high_y_reuse_tap import YCache,make_tap
class Ledger:
 def __init__(self):self.calls=[]
 def call(self,rid,cost,fn,*a):self.calls.append(cost);return fn(*a)
class Base:
 def __init__(self):self.guard=types.SimpleNamespace(torch=torch,ledger=Ledger());self.rid='test';self.block=(0,0);self.curstep=0;self.mapping={}
 def step(self,e=None):return self.curstep
 def mm_call(self,mm,value,meta,fn):return fn(mm,value)
class Test(unittest.TestCase):
 def test_y_cache_identity_and_limit(self):
  c=YCache(8);y=torch.ones(4,dtype=torch.bfloat16);c.put(0,y);self.assertEqual(c.bytes,8)
  with self.assertRaises(ValueError):c.put(1,y)
  with self.assertRaises(ValueError):c.get(1)
  c.release(0);c.empty()
 def test_cached_pre_gate_y_and_dense_control(self):
  for mode in ('high_y_cache_dense','high_y_reuse'):
   w1=object();w2=object();phase=types.SimpleNamespace(ffn_0=w1,ffn_2=w2);tap=make_tap(types.SimpleNamespace(LightTap=Base),None,mode=mode,emit=lambda e:None,geometry=(2,2,3))();tap.mapping={id(w1):('block',0,'ffn.0'),id(w2):('block',0,'ffn.2')};tap.pending={'phase':phase,'expert':0,'step':0,'block':0};oldy=torch.full((2,2),2,dtype=torch.bfloat16);h=torch.ones((2,3),dtype=torch.bfloat16)
   with patch.object(torch.cuda,'synchronize',lambda:None):
    tap.mm_call(w2,h,{},lambda a,b:oldy);self.assertEqual(tap.cache_puts,1);tap.curstep=1;tap.pending={'phase':phase,'expert':0,'step':1,'block':0}
    if mode=='high_y_reuse':
     sentinel=tap.mm_call(w1,torch.ones((2,2),dtype=torch.bfloat16),{},lambda *a:self.fail('W1 not skipped'));out=tap.mm_call(w2,sentinel,{},lambda *a:self.fail('W2 not skipped'));self.assertTrue(torch.equal(out,oldy));self.assertIsNot(out,oldy);self.assertEqual(tap.reused,1)
    else:
     fresh=torch.full((2,2),3,dtype=torch.bfloat16);out=tap.mm_call(w2,h,{},lambda *a:fresh);self.assertIs(out,fresh);self.assertEqual(tap.reused,0)
    self.assertEqual(tap.cache_releases,1);tap.book.empty()
 def test_real_ffn_shape_shim_and_current_gate(self):
  w1=object();w2=object();phase=types.SimpleNamespace(ffn_0=w1,ffn_2=w2);holder={}
  def ffn(phase,x,attn,shift,scale,gate):
   y=holder['tap'].mm_call(w1,x,{},lambda *a:self.fail('W1 not skipped'));y=torch.nn.functional.gelu(y,approximate='tanh');return holder['tap'].mm_call(w2,y,{},lambda *a:self.fail('W2 not skipped'))
  infer=types.SimpleNamespace(block_idx=0,infer_ffn=ffn)
  class InstallBase(Base):
   def __init__(self):
    super().__init__();self.curstep=1;self.stack=contextlib.ExitStack();self.runner=types.SimpleNamespace(model=types.SimpleNamespace(model=[types.SimpleNamespace(transformer_infer=infer)]))
   def install(self):pass
  @contextlib.contextmanager
  def method(obj,name,fn):
   old=getattr(obj,name);setattr(obj,name,fn)
   try:yield
   finally:setattr(obj,name,old)
  tap=make_tap(types.SimpleNamespace(LightTap=InstallBase),types.SimpleNamespace(patch_method=method),mode='high_y_reuse',emit=lambda e:None,geometry=(2,2,3))();holder['tap']=tap;tap.mapping={id(w1):('block',0,'ffn.0'),id(w2):('block',0,'ffn.2')};tap.book.put(0,torch.full((2,2),2,dtype=torch.bfloat16));oldgelu=torch.nn.functional.gelu;tap.install()
  with patch.object(torch.cuda,'synchronize',lambda:None),patch('chipmunk_high_y_reuse_tap.validate_active_compute_buffer',lambda *a:None):
   pregate=infer.infer_ffn(phase,torch.ones((2,2),dtype=torch.bfloat16),None,None,None,None);gate=torch.tensor([1,3],dtype=torch.bfloat16);self.assertTrue(torch.equal(pregate*gate,torch.tensor([[2,6],[2,6]],dtype=torch.bfloat16)))
  self.assertIs(torch.nn.functional.gelu,oldgelu);tap.book.empty();tap.stack.close()
 def test_shim_restores_after_failure(self):
  def broken(*args):raise RuntimeError('interrupted')
  infer=types.SimpleNamespace(block_idx=0,infer_ffn=broken)
  class InstallBase(Base):
   def __init__(self):
    super().__init__();self.curstep=1;self.stack=contextlib.ExitStack();self.runner=types.SimpleNamespace(model=types.SimpleNamespace(model=[types.SimpleNamespace(transformer_infer=infer)]))
   def install(self):pass
  @contextlib.contextmanager
  def method(obj,name,fn):
   old=getattr(obj,name);setattr(obj,name,fn)
   try:yield
   finally:setattr(obj,name,old)
  tap=make_tap(types.SimpleNamespace(LightTap=InstallBase),types.SimpleNamespace(patch_method=method),mode='high_y_reuse',emit=lambda e:None,geometry=(2,2,3))();original=torch.nn.functional.gelu;tap.install()
  with patch('chipmunk_high_y_reuse_tap.validate_active_compute_buffer',lambda *a:None):
   with self.assertRaises(RuntimeError):infer.infer_ffn(None,None,None,None,None,None)
  self.assertIs(torch.nn.functional.gelu,original);self.assertIsNone(tap.pending);tap.stack.close()
 def test_wrong_expert_remains_dense(self):
  w=object();tap=make_tap(types.SimpleNamespace(LightTap=Base),None,mode='high_y_reuse',emit=lambda e:None,geometry=(2,2,3))();tap.block=(1,0);tap.curstep=3;tap.mapping={id(w):('block',1,'ffn.2')};tap.pending={'phase':types.SimpleNamespace(ffn_2=w),'expert':1,'step':3,'block':0};x=object();self.assertIs(tap.mm_call(w,x,{},lambda *a:x),x);self.assertEqual(tap.cache_puts,0)
if __name__=='__main__':unittest.main()
