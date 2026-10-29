import sys,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
import chipmunk_input_video_tap as t
class Ledger:
 def __init__(self):self.calls=[]
 def call(self,rid,cost,fn):self.calls.append(cost);return fn()
class Base:
 def __init__(self):self.guard=types.SimpleNamespace(torch=torch,ledger=Ledger());self.rid='tiny';self.block=(0,0);self.curstep=0;self.mapping={}
 def step(self,e=None):return self.curstep
 def mm_call(self,mm,value,meta,original):return original(mm,value)
class M:
 def __init__(self,w,b):self.weight=w;self.bias=b
class Test(unittest.TestCase):
 def test_tuple_cache_lifecycle(self):
  b=t.CacheBook(100);x=torch.zeros(3,dtype=torch.bfloat16);s=torch.zeros(2,dtype=torch.bfloat16);y=torch.zeros(4,dtype=torch.bfloat16);b.put(0,0,(x,s),y);self.assertEqual(b.bytes,18);b.release(0,0);b.empty()
 def test_x_cache_dense_clones_and_releases(self):
  m,d,h,k=64,2,3,1;x=torch.ones((m,d),dtype=torch.bfloat16);hidden=torch.ones((m,h),dtype=torch.bfloat16);y=torch.ones((m,d),dtype=torch.bfloat16);w1=M(torch.ones((d,h),dtype=torch.bfloat16),torch.zeros(h,dtype=torch.bfloat16));w2=M(torch.ones((h,d),dtype=torch.bfloat16),torch.zeros(d,dtype=torch.bfloat16));phase=types.SimpleNamespace(ffn_0=w1,ffn_2=w2);events=[];cls=t.make_tap(types.SimpleNamespace(LightTap=Base),None,mode='x_cache_dense',rows=torch.arange(m),kernels=None,emit=events.append,geometry=(m,d,h,k));tap=cls();tap.mapping={id(w1):('block',0,'ffn.0'),id(w2):('block',0,'ffn.2')};tap.pending={'phase':phase,'expert':0,'step':0,'block':0}
  with patch.object(torch.cuda,'synchronize',lambda:None):
   tap.mm_call(w1,x,{},lambda mm,v:hidden);tap.mm_call(w2,hidden,{},lambda mm,v:y);tap.curstep=1;tap.pending={'phase':phase,'expert':0,'step':1,'block':0};result=tap.mm_call(w2,hidden,{},lambda mm,v:y);self.assertIs(result,y);tap.book.empty();self.assertEqual((tap.cache_puts,tap.cache_releases,tap.sparse_sites),(1,1,0));self.assertEqual(events[0]['event'],'cache_boundary_memory');self.assertEqual(events[0]['cache_bytes'],m*d*2+64*h*2+m*d*2);self.assertTrue(any(c.get('cache_boundary_memory_observer')==1 for c in tap.guard.ledger.calls))
 def test_x_candidate_uses_same_block_input_and_shadow(self):
  m,d,h,k=64,2,3,1;oldx=torch.ones((m,d),dtype=torch.bfloat16);cur=oldx*2;w1=M(torch.ones((d,h),dtype=torch.bfloat16),torch.zeros(h,dtype=torch.bfloat16));w2=M(torch.ones((h,d),dtype=torch.bfloat16),torch.zeros(d,dtype=torch.bfloat16));phase=types.SimpleNamespace(ffn_0=w1,ffn_2=w2);hidden=torch.nn.functional.gelu(torch.nn.functional.linear(oldx,w1.weight.T,w1.bias),approximate='tanh');y=torch.ones((m,d),dtype=torch.bfloat16);cls=t.make_tap(types.SimpleNamespace(LightTap=Base),None,mode='high_x_recompute_p64',rows=torch.arange(m),kernels=None,emit=lambda e:None,geometry=(m,d,h,k));tap=cls();tap.mapping={id(w1):('block',0,'ffn.0'),id(w2):('block',0,'ffn.2')};tap.pending={'phase':phase,'expert':0,'step':0,'block':0}
  with patch.object(torch.cuda,'synchronize',lambda:None),patch.object(t,'validate_live_w2',lambda *args:None),patch('chipmunk_input_state_math.readonly_action',lambda x,w1,b,w2,old_y,mask,old,mm_fp32:old_y.clone()):
   tap.mm_call(w1,oldx,{},lambda mm,v:hidden);tap.mm_call(w2,hidden,{},lambda mm,v:y);tap.curstep=1;tap.pending={'phase':phase,'expert':0,'step':1,'block':0};sent=tap.mm_call(w1,cur,{},lambda mm,v:None);out=tap.mm_call(w2,sent,{},lambda mm,v:None);self.assertTrue(torch.equal(out,y));tap.book.empty();self.assertEqual(tap.sparse_sites,1);self.assertTrue(any(c.get('extra_old_W1')==1 for c in tap.guard.ledger.calls))
if __name__=='__main__':unittest.main()
