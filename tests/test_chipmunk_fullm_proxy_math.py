import unittest
import torch
from tools.chipmunk_fullm_proxy_math import fixed_prefix_permutation,rms_score,selection,mean_score
class Tests(unittest.TestCase):
    def test_prefix_deterministic_unique(self):
        a=fixed_prefix_permutation(80,42);b=fixed_prefix_permutation(80,42)
        self.assertTrue(torch.equal(a,b));self.assertEqual(len(set(a.tolist())),64);self.assertEqual(a[:4].tolist(),b[:4].tolist())
        with self.assertRaises(ValueError):fixed_prefix_permutation(63,42)
    def test_variance_vs_mean_and_ties(self):
        a=torch.tensor([[1.,0.],[-1.,0.],[1.,0.],[-1.,0.]],dtype=torch.bfloat16);old=torch.zeros_like(a)
        self.assertGreater(float(rms_score(a,old)[0]),0);self.assertEqual(float(mean_score(a.mean(0),old.mean(0))[0]),0)
        self.assertEqual(selection(torch.tensor([1.,1.,0.]),1).tolist(),[0])
        with self.assertRaises(ValueError):rms_score(a[:3],old[:3])
if __name__=='__main__':unittest.main()

class CausalTests(unittest.TestCase):
    def test_online_selector_uses_prefix_and_no_full_current_h(self):
        from tools.chipmunk_fullm_proxy_math import online_mask
        x=torch.zeros(64,2,dtype=torch.bfloat16);x[:4,0]=2
        w=torch.eye(2,dtype=torch.bfloat16);b=torch.zeros(2,dtype=torch.bfloat16);old=torch.zeros(64,2,dtype=torch.bfloat16)
        rows=torch.arange(64);idx=torch.tensor([1])
        self.assertEqual(online_mask('p4',x,w,b,old,torch.zeros(2,dtype=torch.bfloat16),rows,1,idx).tolist(),[0])
        self.assertEqual(online_mask('p64',x,w,b,old,torch.zeros(2,dtype=torch.bfloat16),rows,1,idx).tolist(),[0])
        self.assertEqual(online_mask('random',x,w,b,old,torch.zeros(2,dtype=torch.bfloat16),rows,1,idx).tolist(),[1])

class OrderTests(unittest.TestCase):
    def test_all_online_precede_reference_oracle(self):
        from tools.chipmunk_fullm_proxy_math import execute_pair
        calls=[]
        def measured(name,debits,fn):calls.append(name);return fn()
        def action(mask,h,y):return y.clone()
        x=torch.zeros(64,2,dtype=torch.bfloat16);w=torch.eye(2,dtype=torch.bfloat16);b=torch.zeros(2,dtype=torch.bfloat16);rows=torch.arange(64);idx=torch.tensor([0])
        z=execute_pair(x,x,w,b,w,b,torch.ones(2,dtype=torch.bfloat16),rows,1,idx,measured=measured,action=action)
        self.assertEqual(calls,['old_cache','cold_kernel_setup','online_mean','online_p4','online_p64','online_random','dense_reference','oracle_selection','oracle_action'])
        self.assertEqual(set(z['masks']),{'mean','p4','p64','random','oracle'})
