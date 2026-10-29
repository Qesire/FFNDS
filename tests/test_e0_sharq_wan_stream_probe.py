import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_stream_probe as m
class Tests(unittest.TestCase):
 def test_separate_scopes_and_actual_work_bounds(self):
  self.assertNotEqual(m.identity('safe'),m.identity('unsafe'));self.assertEqual(m.COUNTS['safe']['weight_quant'],8);self.assertEqual(m.COUNTS['unsafe']['weight_quant'],4);self.assertEqual(m.COUNTS['unsafe']['dense32_linear'],0);self.assertEqual(m.COUNTS['safe']['dense32_gemm'],8);self.assertEqual(m.COUNTS['safe']['sparse_gemm'],8);self.assertEqual(m.COUNTS['safe']['dense_residual_gemm'],8)
 def test_failed_bf16_gate_stops_before_candidates(self):
  trace=[];item={'weight_bitwise_unchanged':True,'activation_bitwise_unchanged':True,'BF16_pre_post_bitwise_equal':False,'BF16_finite':True}
  def next_stage():m.require_BF16_gate(item,lambda:trace.append('persist'));trace.append('candidate')
  with self.assertRaisesRegex(ValueError,'BF16 input'):next_stage()
  self.assertEqual(trace,[])
  item['BF16_pre_post_bitwise_equal']=True;next_stage();self.assertEqual(trace,['persist','candidate']);self.assertEqual(item['BF16_gate'],'PASS_BEFORE_ANY_CANDIDATE')
 def test_nested_predebit_preserves_inflight_on_failure(self):
  r={'predebited':{},'invoked':{},'returned':{},'inflight':{}};events=[];ledger=m.Ledger(r,lambda:events.append(dict(r['inflight'])),{'outer':1,'inner':1})
  def child():raise RuntimeError('failure')
  with self.assertRaises(RuntimeError):ledger.call('outer',lambda:ledger.call('inner',child))
  self.assertEqual(r['predebited'],{'outer':1,'inner':1});self.assertEqual(r['invoked'],{'outer':1,'inner':1});self.assertEqual(set(r['inflight']),{'outer','inner'});self.assertEqual(r['returned'],{});self.assertEqual(events[0],{'outer':'UNKNOWN'})
 def test_bitwise_rejects_bf16_negative_zero(self):
  try:import torch
  except ImportError:self.skipTest('CPU torch unavailable')
  a=torch.tensor([0.],dtype=torch.bfloat16);b=torch.tensor([-0.],dtype=torch.bfloat16)
  self.assertTrue(torch.equal(a,b));self.assertFalse(m.bitwise_equal(torch,a,b));self.assertTrue(m.bitwise_equal(torch,a,a.clone()))
 def test_sparse_logical_scales_ignore_padding_across_tiles(self):
  try:import torch
  except ImportError:self.skipTest('CPU torch unavailable')
  rows,k=129,256;groups=k//32;raw=torch.zeros((rows//128+1)*128*(k//16),dtype=torch.uint8);expected={}
  for i,(row,group) in enumerate((r,g) for r in (0,31,32,127,128) for g in (0,3,4,7)):
   offset=(row//128)*(groups//4)*512+(group//4)*512+(row%32)*16+((row%128)//32)*4+group%4;raw[offset]=i+1;expected[row,group]=i+1
  logical=m.sparse_logical_scale(torch,raw,rows,k)
  for (row,group),value in expected.items():self.assertEqual(int(logical[row,group]),value)
  altered=raw.clone();altered[-1]=99;self.assertTrue(torch.equal(logical,m.sparse_logical_scale(torch,altered,rows,k)))
  altered=raw.clone();altered[0]=99;self.assertFalse(torch.equal(logical,m.sparse_logical_scale(torch,altered,rows,k)))
 def test_draft_requires_actual_profile(self):
  with self.assertRaises(Exception):m.draft('bad',Path('/tmp/missing'))
if __name__=='__main__':unittest.main()
