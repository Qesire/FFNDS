import importlib.util
import unittest
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'tools/e0_teal_wan_sparse_reachability_cpu.py';spec=importlib.util.spec_from_file_location('reach',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestReachability(unittest.TestCase):
 def test_real_mask_cold_upper_and_exact_four_counts(self):
  old=(m.ROWS,m.D);m.ROWS,m.D=2,8
  try:
   h=torch.tensor([[0,1,0,2,0,0,0,0],[0,0,0,3,1,2,3,4]],dtype=torch.bfloat16)
   x=m.compute(h,0,torch)
   self.assertEqual(x['deleted'],9);self.assertEqual(x['cold_columns_sample_upper_bound'],2)
   self.assertEqual(x['K4_nonzero_histogram'],[1,1,1,0,1]);self.assertEqual(x['K4_groups_eligible_at_most2'],3);self.assertEqual(x['rows_entirely_eligible'],1)
  finally:m.ROWS,m.D=old
 def test_original_native_bf16_scalar_comparison_boundary(self):
  root=Path(__file__).resolve().parents[1];q=root/'tools/e0_teal_wan_all160_video_v3.py';spec=importlib.util.spec_from_file_location('teal_video',q);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
  old=(m.ROWS,m.D);m.ROWS,m.D=1,4
  try:
   h=torch.tensor([[1.0,-1.0,0.5,2.0]],dtype=torch.bfloat16);threshold=0.9999
   masked,record=v.mask_values(torch,h,threshold);result=m.compute(h,threshold,torch)
   self.assertEqual(result['deleted'],record['selected']);self.assertEqual(result['threshold_ties'],record['threshold_ties']);self.assertEqual((masked!=0).tolist(),(h.abs()>threshold).tolist())
   self.assertNotEqual((h.abs()>threshold).tolist(),(h.float().abs()>threshold).tolist())
  finally:m.ROWS,m.D=old
 def test_evaluation_interruption_and_limit_are_predebited(self):
  state={'predebited':{'site_rate_evaluations':0},'completed':{'site_rate_evaluations':0},'inflight':{}};calls=[]
  m.predebit_evaluation(state,lambda:calls.append((state['predebited']['site_rate_evaluations'],state['completed']['site_rate_evaluations'])),'0.25')
  self.assertEqual(calls,[(1,0)]);self.assertEqual(state['inflight']['rate_progress'],'UNKNOWN_UNTIL_COMPLETE')
  m.complete_evaluation(state,lambda:calls.append((state['predebited']['site_rate_evaluations'],state['completed']['site_rate_evaluations'])))
  self.assertEqual(calls[-1],(1,1));state['predebited']['site_rate_evaluations']=640
  with self.assertRaises(Exception):m.predebit_evaluation(state,lambda:None,'0.5')
 def test_ties_and_finite(self):
  old=(m.ROWS,m.D);m.ROWS,m.D=1,4
  try:
   h=torch.tensor([[0.5,-0.5,1,0]],dtype=torch.bfloat16);x=m.compute(h,0.5,torch);self.assertEqual(x['deleted'],3);self.assertEqual(x['threshold_ties'],2)
   h[0,0]=float('nan')
   with self.assertRaises(Exception):m.compute(h,0.5,torch)
  finally:m.ROWS,m.D=old
if __name__=='__main__':unittest.main()
