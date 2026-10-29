import unittest
import torch
from tools.chipmunk_history_encoding_math import encode, selected, readonly_action
class TestHistoryEncoding(unittest.TestCase):
 def test_shadow_and_shapes(self):
  h=torch.tensor([[0.,1.,-2.],[2.,-1.,4.]],dtype=torch.bfloat16);rows=torch.tensor([1]);mask=torch.tensor([0,2])
  for kind in ('fp8','int8'):
   q,shadow,scale=encode(h,rows,kind)
   self.assertTrue(torch.equal(shadow,h[rows]));self.assertEqual(selected(q,mask,scale).shape,(2,2))
 def test_zero_and_tiny_scale_finite(self):
  h=torch.tensor([[0.,1e-38], [0.,-1e-38]],dtype=torch.bfloat16);q,_,s=encode(h,torch.tensor([0]),'int8');self.assertTrue(torch.isfinite(s).all());self.assertEqual(float(s[0]),1.0);self.assertTrue(torch.isfinite(selected(q,torch.tensor([0,1]),s)).all())
 def test_readonly_same_arithmetic(self):
  torch.manual_seed(1);x=torch.randn(3,4,dtype=torch.bfloat16);w1=torch.randn(5,4,dtype=torch.bfloat16);b=torch.randn(5,dtype=torch.bfloat16);w2=torch.randn(4,5,dtype=torch.bfloat16);h=torch.randn(3,5,dtype=torch.bfloat16);y=torch.randn(3,4,dtype=torch.bfloat16);m=torch.tensor([1,3]);mm=lambda a,b:a.float()@b.float()
  from tools.chipmunk_wan_packed_pregate import packed_pregate
  a=packed_pregate(x,w1,b,w2,h,y,m,mm_fp32=mm);z=readonly_action(x,w1,b,w2,y,m,h.index_select(1,m),mm_fp32=mm)
  self.assertTrue(torch.equal(a,z))
if __name__=='__main__':unittest.main()

class TestFiniteCore(unittest.TestCase):
 def test_five_calls_per_arm_and_same_mask(self):
  import sys
  from pathlib import Path
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
  from chipmunk_history_encoding_core import exercise
  torch.manual_seed(2);x=torch.randn(64,4,dtype=torch.bfloat16);h=torch.randn(64,5,dtype=torch.bfloat16);w1=torch.randn(5,4,dtype=torch.bfloat16);w2=torch.randn(4,5,dtype=torch.bfloat16);b=torch.randn(5,dtype=torch.bfloat16);y=torch.randn(64,4,dtype=torch.bfloat16);counts={}
  def measured(name,c,fn):
   for k,v in c.items():counts[k]=counts.get(k,0)+v
   return fn()
  masks,_,meta=exercise(x,w1,b,w2,h,y,torch.ones(4,dtype=torch.bfloat16),torch.arange(64),2,measured=measured,mm_fp32=lambda a,b:a.float()@b.float(),check_mask=lambda arm,mask:None)
  self.assertEqual(set(meta),{'fp8','int8'});self.assertEqual([counts['action_'+k] for k in 'ABCD'],[5]*4);self.assertTrue(all(torch.equal(masks['A'],masks[k]) for k in 'BCD'))
 def test_bad_baseline_stops_before_encode(self):
  import sys
  from pathlib import Path
  from unittest.mock import patch
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
  import chipmunk_history_encoding_core as core
  torch.manual_seed(3);x=torch.randn(64,4,dtype=torch.bfloat16);h=torch.randn(64,5,dtype=torch.bfloat16);w1=torch.randn(5,4,dtype=torch.bfloat16);w2=torch.randn(4,5,dtype=torch.bfloat16);b=torch.randn(5,dtype=torch.bfloat16);y=torch.randn(64,4,dtype=torch.bfloat16);counts={}
  def measured(name,c,fn):
   for k,v in c.items():counts[k]=counts.get(k,0)+v
   return fn()
  with patch.object(core,'readonly_action',return_value=torch.zeros_like(y)):
   with self.assertRaisesRegex(ValueError,'bitwise gate'):
    core.exercise(x,w1,b,w2,h,y,torch.ones(4,dtype=torch.bfloat16),torch.arange(64),2,measured=measured,mm_fp32=lambda a,b:a.float()@b.float(),check_mask=lambda arm,mask:None)
  self.assertNotIn('encode_fp8',counts);self.assertNotIn('action_C',counts)
