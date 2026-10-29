import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_rtlynx_wan_ffn_ops as m
class Tests(unittest.TestCase):
 def test_costs_match_path_and_zero_others(self):
  self.assertEqual(m.expected_calls('dense')['W2_bf16_addmm'],1);self.assertEqual(m.expected_calls('top2')['cutlass_sparse_W2'],1);self.assertEqual(m.expected_calls('top2_row_norm')['row_norm_and_BF16_scale'],1)
  with self.assertRaises(ValueError):m.expected_calls('lora')
 def test_norm_and_tie_are_explicit_source_limits(self):
  self.assertEqual(m.EPSILON,1e-8);self.assertEqual(m.ARMS,('dense','top2','top2_row_norm'))
 def test_row_norm_zero_and_nonzero_synthetic(self):
  import torch
  h=torch.tensor([[0,0,0,0],[2,2,1,1]],dtype=torch.bfloat16);masked=torch.tensor([[0,0,0,0],[2,2,0,0]],dtype=torch.bfloat16)
  y,scale=m.row_compensated(torch,h,masked);self.assertEqual(float(scale[0]),0.);self.assertAlmostEqual(float(scale[1]),(10/8)**.5,places=6);self.assertEqual(y.dtype,torch.bfloat16);self.assertEqual(y[1,2:].tolist(),[0,0])
if __name__=='__main__':unittest.main()
