import importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_sharq_wan_ffn_probe.py'
s=importlib.util.spec_from_file_location('sharq_probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_unbound_cannot_freeze(self):
  p={'id':m.ID,'resources':m.RES,'design':m.DESIGN,'output':m.OUTPUT,'status':'FROZEN_PENDING_REGISTRATION','review':None,'assets':None,'software':None,'worker_config_sha256':None}
  m.validate(p)
  with self.assertRaises(Exception):m.validate(p,True)
 def test_predebit_precedes_actual_calls_and_detects_overrun(self):
  r={'predebited':{},'invoked':{},'returned':{}};events=[];l=m.Ledger(r,lambda:events.append(dict(r['predebited'])))
  l.reserve('original_bf16');self.assertEqual(events,[{'gelu_tanh':1,'gate_multiply':1,'bf16_linear':2}]);l.call('bf16_linear',lambda:1);l.call('bf16_linear',lambda:2)
  with self.assertRaises(Exception):l.call('bf16_linear',lambda:3)
  l.call('gelu_tanh',lambda:4);l.call('gate_multiply',lambda:5);l.finish();self.assertEqual(r['predebited'],r['returned'])
 def test_failure_records_invoked_not_returned(self):
  r={'predebited':{},'invoked':{},'returned':{}};l=m.Ledger(r,lambda:None);l.reserve('original_bf16')
  def fail():raise RuntimeError('synthetic')
  with self.assertRaises(RuntimeError):l.call('bf16_linear',fail)
  self.assertEqual(r['invoked']['bf16_linear'],1);self.assertNotIn('bf16_linear',r['returned'])
 def test_extra_predebit_and_failure_remains_inflight(self):
  r={'predebited':{},'invoked':{},'returned':{}};states=[];l=m.Ledger(r,lambda:states.append(dict(r)))
  def fail():raise ValueError('interrupted')
  with self.assertRaises(ValueError):l.extra('bundle_hashes',fail)
  self.assertEqual(r['predebited']['bundle_hashes'],1);self.assertEqual(r['invoked']['bundle_hashes'],1);self.assertNotIn('bundle_hashes',r['returned']);self.assertEqual(r['inflight_extra']['bundle_hashes'],'UNKNOWN_UNTIL_RETURN')
  with self.assertRaises(Exception):l.extra('bundle_hashes',lambda:None)
 def test_missing_actual_parent_fails_draft(self):
  with self.assertRaises(Exception):m.draft(Path('/nonexistent/protocol'),Path('/nonexistent/profile'))
 def test_both_timing_groups_and_reference_are_budgeted(self):
  groups=2*2*(m.DESIGN['correctness']+m.DESIGN['warmup']+m.DESIGN['measured']+m.DESIGN['uninstrumented_warmup']+m.DESIGN['uninstrumented_measured'])
  self.assertEqual(groups,60);self.assertEqual(m.LIMITS['sparse_gemm'],2*groups);self.assertEqual(m.LIMITS['dense_residual_gemm'],2*groups);self.assertEqual(m.LIMITS['reference_linear'],12);self.assertEqual(m.LIMITS['reference_decode'],28);self.assertEqual(m.LIMITS['reference_fp32_gemm'],16)
 def test_tiny_group16_same_encoding_and_bad_output(self):
  try:import torch
  except ImportError:self.skipTest('torch unavailable in this test interpreter')
  from types import SimpleNamespace
  q=torch.full((2,8),0x22,dtype=torch.uint8);w=torch.full((1,8),0x22,dtype=torch.uint8);sf=torch.ones((128,4),dtype=torch.float8_e4m3fn)
  layer=SimpleNamespace(dense16_alpha=torch.ones(1),input_global_scale=torch.ones(1),dense16_weight=w,dense16_scale=sf,dense16_bias=torch.zeros(1))
  api=SimpleNamespace(dense16_prepare=lambda x,s:(q,sf));ledger=SimpleNamespace(extra=lambda key,fn:fn());x=torch.zeros((2,16));actual=torch.full((2,1),16.)
  self.assertEqual(m.linear_reference(torch,api,'dense_group16',x,layer,actual,ledger,(q,sf))['relative_l2'],0)
  with self.assertRaises(Exception):m.linear_reference(torch,api,'dense_group16',x,layer,actual*2,ledger,(q,sf))
 def test_tiny_sparse_residual_reference_checks_debug_consistency(self):
  try:import torch
  except ImportError:self.skipTest('torch unavailable in this test interpreter')
  from types import SimpleNamespace
  q=torch.full((2,8),0x22,dtype=torch.uint8);sparse=q.clone();sparse[:,4:]=0;w=torch.full((1,8),0x22,dtype=torch.uint8);sf=torch.ones((128,4),dtype=torch.float8_e4m3fn);raw=sf.view(torch.uint8)
  layer=SimpleNamespace(dense16_alpha=torch.ones(1),input_global_scale=torch.ones(1),weight_q=w,scale_w_dense=raw,weight_scale=torch.ones(1),bias=torch.zeros(1),out_features=1,in_features=16)
  ext=SimpleNamespace(fused_sparse_residual_quantize_x_debug=lambda x,n:(sparse,raw,q,raw),duplicate_sfa32_to_sfa16=lambda *a:raw)
  api=SimpleNamespace(sharq_prepare=lambda x,n:(None,None,raw,q,raw,torch.ones(1)),extension=ext);ledger=SimpleNamespace(extra=lambda key,fn:fn());x=torch.zeros((2,16));actual=torch.full((2,1),24.)
  captured=(None,None,raw,q,raw,torch.ones(1));self.assertEqual(m.linear_reference(torch,api,'sharq',x,layer,actual,ledger,captured)['relative_l2'],0)
  ext.fused_sparse_residual_quantize_x_debug=lambda x,n:(sparse,raw,q+1,raw)
  with self.assertRaises(Exception):m.linear_reference(torch,api,'sharq',x,layer,actual,ledger,captured)
if __name__=='__main__':unittest.main()
