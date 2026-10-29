import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import chipmunk_input_state_core as core
class InputStateCoreTests(unittest.TestCase):
 def test_four_arms_and_recompute(self):
  torch.manual_seed(2);x=torch.randn(64,4,dtype=torch.bfloat16);oldx=torch.randn(64,4,dtype=torch.bfloat16);w1=torch.randn(5,4,dtype=torch.bfloat16);b=torch.randn(5,dtype=torch.bfloat16);w2=torch.randn(4,5,dtype=torch.bfloat16);h=torch.nn.functional.gelu(torch.nn.functional.linear(oldx,w1,b),approximate='tanh');y=torch.randn(64,4,dtype=torch.bfloat16);counts={};checks={}
  def measured(name,c,fn):
   for k,v in c.items():counts[k]=counts.get(k,0)+v
   return fn()
  masks,outputs,meta,old=core.exercise(x,oldx,w1,b,w2,h,y,torch.ones(4,dtype=torch.bfloat16),torch.arange(64),2,measured=measured,mm_fp32=lambda a,b:a.float()@b.float(),check_mask=lambda a,m:checks.setdefault(a,m.clone()))
  self.assertEqual([counts['action_'+a] for a in 'ABCD'],[5]*4);self.assertTrue(all(torch.equal(masks['A'],masks[a]) for a in 'BCD'));self.assertEqual(set(old),{'C','D'});self.assertEqual(set(meta),set('ABCD'));self.assertTrue(torch.equal(old['C'],old['D']))
 def test_prior_mask_failure_stops_before_B_cache(self):
  x=torch.ones(64,4,dtype=torch.bfloat16);h=torch.ones(64,5,dtype=torch.bfloat16);w1=torch.ones(5,4,dtype=torch.bfloat16);w2=torch.ones(4,5,dtype=torch.bfloat16);b=torch.ones(5,dtype=torch.bfloat16);y=torch.ones(64,4,dtype=torch.bfloat16);events=[]
  def measured(name,c,fn):events.append(name);return fn()
  def reject(arm,mask):raise ValueError('prior mask differs')
  with self.assertRaisesRegex(ValueError,'prior mask differs'):
   core.exercise(x,x,w1,b,w2,h,y,torch.ones(4,dtype=torch.bfloat16),torch.arange(64),2,measured=measured,mm_fp32=lambda a,b:a.float()@b.float(),check_mask=reject)
  self.assertNotIn('cache_B',events)
