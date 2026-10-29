import unittest
import torch
from tools.chipmunk_p4_proxy_math import select,sample_rms_score,mean_score

class TestP4(unittest.TestCase):
    def test_sampling_reveals_cancellation(self):
        a=torch.tensor([[1.,0.],[-1.,0.],[1.,0.],[-1.,0.]],dtype=torch.bfloat16);old=torch.zeros_like(a)
        self.assertGreater(float(sample_rms_score(a,old)[0]),0)
        self.assertEqual(float(mean_score(a.mean(0),torch.zeros(2,dtype=torch.bfloat16))[0]),0)
        self.assertEqual(select(sample_rms_score(a,old),1).tolist(),[0])
    def test_stable_ties_and_invalid(self):
        self.assertEqual(select(torch.tensor([2.,2.,1.]),1).tolist(),[0])
        with self.assertRaises(ValueError):sample_rms_score(torch.zeros(3,2,dtype=torch.bfloat16),torch.zeros(3,2,dtype=torch.bfloat16))
        with self.assertRaises(ValueError):select(torch.tensor([float('nan')]),1)
if __name__=='__main__':unittest.main()

class OrderTests(unittest.TestCase):
    def test_reference_occurs_only_after_all_selection_actions(self):
        from tools.chipmunk_p4_proxy_math import pair_core
        calls=[]
        def charge(name,fn):
            calls.append(name)
            return fn()
        def action(mask,old_h,old_y):
            self.assertNotIn('reference_w1_after_selections',calls)
            return old_y.clone()
        x=torch.zeros(4,2,dtype=torch.bfloat16)
        r=pair_core(x,x,torch.ones(3,2,dtype=torch.bfloat16),torch.zeros(3,dtype=torch.bfloat16),torch.ones(2,3,dtype=torch.bfloat16),torch.zeros(2,dtype=torch.bfloat16),[0,1,2,3],[2],1,charged=charge,action=action)
        self.assertEqual(calls.count('sample_w1'),1)
        self.assertEqual(len(r['masks']),3)
        self.assertLess(calls.index('sparse_action_random'),calls.index('reference_w1_after_selections'))
