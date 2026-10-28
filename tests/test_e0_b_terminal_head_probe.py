import unittest
import numpy as np
from unittest.mock import patch
import torch
from tools import e0_b_terminal_head_probe as m

class Tests(unittest.TestCase):
 def test_full_shape_and_unchanged_rows(self):
  torch.manual_seed(7);u=torch.randn(12,8).bfloat16();w=torch.randn(8,3).bfloat16();b=torch.randn(3).bfloat16();scale=torch.randn(8).bfloat16();shift=torch.randn(8).bfloat16();ids=torch.tensor([1,7])
  out,inp=m.native_head(torch,u,w,b,scale,shift);changed=u.clone();changed[ids]+=torch.tensor(.25,dtype=torch.bfloat16);other,_=m.native_head(torch,changed,w,b,scale,shift);mask=torch.ones(12,dtype=torch.bool);mask[ids]=False
  self.assertTrue(m.same_bits(torch,out[mask],other[mask]));self.assertEqual(inp.shape,(12,8));self.assertFalse(m.same_bits(torch,out[ids],other[ids]))
 def test_strided_and_rows(self):
  t=torch.arange(12).reshape(3,4).bfloat16();x=m.restore_layout(torch,t,(1,3),'cpu');self.assertEqual(x.stride(),(1,3));self.assertTrue(m.same_bits(torch,t,x));m.strict_rows(torch,torch.tensor([0,2]),3,expected=2)
  for bad in (torch.tensor([1,1]),torch.tensor([0,3])):
   with self.assertRaises(ValueError):m.strict_rows(torch,bad,3,expected=2)
 def test_ledger_guard_and_scope(self):
  r={'predebited':dict.fromkeys(m.LIMITS,0)};persist=[]
  for _ in range(72):m.reserve(r,{'head_calls':1,'layer_norm_calls':1},lambda:persist.append(1))
  self.assertEqual(len(persist),72)
  with self.assertRaises(ValueError):m.reserve(r,{'head_calls':1},lambda:None)
  self.assertEqual(len(m.variants()),16);self.assertEqual(sum(k=='pregate' for k,_,_ in m.variants()),2)
 def test_head_observer_failure_counts(self):
  calls=[];u=torch.ones((2,4),dtype=torch.bfloat16);w=torch.ones((4,2),dtype=torch.bfloat16);b=torch.zeros(2,dtype=torch.bfloat16);z=torch.zeros(4,dtype=torch.bfloat16)
  with patch.object(torch,'addmm',side_effect=RuntimeError('broken MM')):
   with self.assertRaises(RuntimeError):m.native_head(torch,u,w,b,z,z,observe=lambda *x:calls.append(x))
  self.assertEqual(calls,[('layer_norm_calls','invoked'),('layer_norm_calls','completed'),('head_calls','invoked')])
 def test_fp64_delta_and_readout(self):
  u=np.array([[1.,2.,3.,4.]],dtype=np.float64);d=np.array([[.25,-.25,0.,0.]],dtype=np.float64);T=np.array([[1.,0.,0.,0.]]);native=np.array([[.5]]);r=m.response_readout(np,u,d,T,native,native,native*2)
  self.assertEqual(r['native_vs_finite']['error_SSE'],0.);self.assertAlmostEqual(r['native_vs_linear']['error_SSE'],.25);self.assertGreater(r['delta_relative_l2']['mean'],0)
 def test_pregate_writeback_bf16(self):
  r=torch.tensor([[1.,-2.]],dtype=torch.bfloat16);y=torch.tensor([[.5,.75]],dtype=torch.bfloat16);g=torch.tensor([.5,-.25],dtype=torch.bfloat16);d=torch.tensor([[.25,-.25]])
  actual=r.clone();actual.add_((y.float()+d).bfloat16()*g)
  expected=(r+(y.float()+d).bfloat16()*g).bfloat16();self.assertTrue(m.same_bits(torch,actual,expected))
if __name__=='__main__':unittest.main()
