import importlib.util,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('components',Path(__file__).resolve().parents[1]/'tools/e0_activation24_online_components_probe.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_rank_matches_stable_selection_with_ties(self):
  for v in ([1,-1,1,-1],[0,0,0,0],[4,-9,9,3],[-3,5,-5,3],[2,-2,1,-2]):
   ranks=m.tie_rank(v);chosen=[i for i in range(4) if ranks[i]<2]
   self.assertEqual(chosen,sorted(sorted(range(4),key=lambda i:(-abs(v[i]),i))[:2]))
 def test_rank_selection_matches_cpu_stable_top2(self):
  import torch
  samples=torch.tensor([[1,-1,1,-1],[0,0,0,0],[4,-9,9,3],[-3,5,-5,3]],dtype=torch.bfloat16)
  order=torch.argsort(samples.abs(),dim=-1,descending=True,stable=True)[:,:2]
  expected=torch.zeros_like(samples,dtype=torch.bool).scatter_(1,order,True)
  actual=torch.tensor([[r<2 for r in m.tie_rank(row.tolist())] for row in samples],dtype=torch.bool)
  self.assertTrue(torch.equal(expected,actual))
 def test_precharge_and_fail_preserved(self):
  count=dict.fromkeys(m.LIMITS,0);m.charge(count,{'stable_select':1,'compress':1,'sparse_gemm':1})
  self.assertEqual((count['stable_select'],count['compress'],count['sparse_gemm']),(1,1,1))
  before=count.copy()
  with self.assertRaises(ValueError):m.charge(count,{'stable_select':1,'compress':1000})
  self.assertEqual(count,before)
 def test_exact_budget(self):
  per_case={'random_tensor_init':2,'stable_select':1+7+7,'triton_select':1+7+7,'compress':1+7+7+7,'dense_gemm':1+7,'sparse_gemm':1+7+7+7,'search_api':0,'model_calls':0,'fits':0,'video_generations':0}
  self.assertEqual({k:2*v for k,v in per_case.items()},m.LIMITS)
if __name__=='__main__':unittest.main()
