import importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_sharq_wan_ffn_probe_v2.py'
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
  padded=sf.clone();padded.view(torch.uint8).reshape(-1)[32]=64
  self.assertEqual(m.linear_reference(torch,api,'dense_group16',x,layer,actual,ledger,(q,padded))['relative_l2'],0)
  bad=sf.clone();bad.view(torch.uint8).reshape(-1)[0]=64
  with self.assertRaises(Exception):m.linear_reference(torch,api,'dense_group16',x,layer,actual,ledger,(q,bad))
  with self.assertRaises(Exception):m.linear_reference(torch,api,'dense_group16',x,layer,actual*2,ledger,(q,sf))
 def test_tiny_sparse_residual_reference_checks_debug_consistency(self):
  try:import torch
  except ImportError:self.skipTest('torch unavailable in this test interpreter')
  from types import SimpleNamespace
  q=torch.full((2,32),0x22,dtype=torch.uint8);sparse=q.clone();sparse[:,16:]=0;w=torch.full((1,32),0x22,dtype=torch.uint8);sf=torch.ones((128,4),dtype=torch.float8_e4m3fn);raw=sf.view(torch.uint8)
  layer=SimpleNamespace(dense16_alpha=torch.ones(1),input_global_scale=torch.ones(1),weight_q=w,scale_w_dense=raw,weight_scale=torch.ones(1),bias=torch.zeros(1),out_features=1,in_features=64)
  ext=SimpleNamespace(fused_sparse_residual_quantize_x_debug=lambda x,n:(sparse,raw,q,raw),duplicate_sfa32_to_sfa16=lambda *a:raw)
  api=SimpleNamespace(sharq_prepare=lambda x,n:(None,None,raw,q,raw,torch.ones(1)),extension=ext);ledger=SimpleNamespace(extra=lambda key,fn:fn());x=torch.zeros((2,64));actual=torch.full((2,1),96.)
  captured=(None,None,raw,q,raw,torch.ones(1));self.assertEqual(m.linear_reference(torch,api,'sharq',x,layer,actual,ledger,captured)['relative_l2'],0)
  ext.fused_sparse_residual_quantize_x_debug=lambda x,n:(sparse,raw,q+1,raw)
  with self.assertRaises(Exception):m.linear_reference(torch,api,'sharq',x,layer,actual,ledger,captured)
 def test_author_aligned_extra_scale_tile_and_lightx_strict_size(self):
  try:import torch
  except ImportError:self.skipTest('torch unavailable in this test interpreter')
  q=torch.full((128,32),0x22,dtype=torch.uint8);one=torch.ones((128,4),dtype=torch.float8_e4m3fn);tail=torch.full((128,4),2.,dtype=torch.float8_e4m3fn);author=torch.cat((one.view(torch.uint8),tail.view(torch.uint8)))
  got=m.decode_actual(torch,q,author,'author');self.assertTrue(torch.equal(got,torch.ones_like(got)))
  self.assertTrue(torch.equal(m.decode_actual(torch,q,one,'lightx'),got))
  with self.assertRaises(Exception):m.decode_actual(torch,q,one,'author')
  with self.assertRaises(Exception):m.decode_actual(torch,q,author,'lightx')
  q2=torch.full((129,32),0x22,dtype=torch.uint8);got2=m.decode_actual(torch,q2,author,'author');self.assertEqual(float(got2[-1,0]),2.)
 def test_dense32_reference_reads_author_padded_weight(self):
  try:import torch
  except ImportError:self.skipTest('torch unavailable in this test interpreter')
  from types import SimpleNamespace
  q=torch.full((2,32),0x22,dtype=torch.uint8);w=torch.full((128,32),0x22,dtype=torch.uint8);a_sf=torch.ones((128,4),dtype=torch.float8_e4m3fn).view(torch.uint8);w_sf=torch.ones((256,4),dtype=torch.float8_e4m3fn).view(torch.uint8);scale=torch.ones(1)
  layer=SimpleNamespace(dense16_alpha=scale,input_global_scale=scale,weight_q=w,scale_w_dense=w_sf,weight_scale=scale,bias=torch.zeros(128));api=SimpleNamespace(dense_prepare=lambda x:(q,a_sf,scale));ledger=SimpleNamespace(extra=lambda key,fn:fn())
  result=m.linear_reference(torch,api,'dense_shared32',torch.zeros((2,64)),layer,torch.full((2,128),64.),ledger,(q,a_sf,scale));self.assertEqual(result['relative_l2'],0.)
 def test_nonuniform_author_scale_coordinates_across_tiles(self):
  try:import torch
  except ImportError:self.skipTest('torch unavailable in this test interpreter')
  rows,k=129,128;s=torch.zeros(256*(k//16),dtype=torch.uint8);expected={}
  for i,(row,group) in enumerate((r,g) for r in (0,31,32,127,128) for g in (0,3,4,7)):
   offset=(row//128)*((k+63)//64)*512+(group//4)*512+(row%32)*16+((row%128)//32)*4+group%4
   expected[row,group]=i+1;s[offset]=i+1
  logical=m.logical_scale_bytes(torch,s,rows,k,'author')
  for (row,group),value in expected.items():self.assertEqual(int(logical[row,group]),value)
  q=torch.full((rows,k//2),0x22,dtype=torch.uint8);decoded=m.decode_actual(torch,q,s,'author')
  for (row,group),value in expected.items():
   wanted=torch.tensor([value],dtype=torch.uint8).view(torch.float8_e4m3fn).float()[0];self.assertEqual(float(decoded[row,group*16]),float(wanted))
if __name__=='__main__':unittest.main()
