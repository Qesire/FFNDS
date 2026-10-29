import unittest
import torch
from tools.chipmunk_p4_sparse_action import selected_action

class Kernel:
    def __init__(self,name,events):self.name=name;self.events=events
    def __getitem__(self,grid):
        def invoke(*args):self.events.append((self.name,grid,args[7]))
        return invoke
class Kernels:
    def __init__(self,events):
        self.selected_w1_gelu_delta=Kernel('W1',events);self.selected_w2_add_cache=Kernel('W2',events)
class TestAction(unittest.TestCase):
    def test_pays_both_calls_and_preserves_old(self):
        e=[];counts=[]
        def charge(name,fn):counts.append(name);return fn()
        x=torch.zeros(4,2,dtype=torch.bfloat16);w1=torch.ones(3,2,dtype=torch.bfloat16);w2=torch.ones(2,3,dtype=torch.bfloat16);b=torch.zeros(3,dtype=torch.bfloat16);h=torch.zeros(4,3,dtype=torch.bfloat16);y=torch.ones(4,2,dtype=torch.bfloat16)
        ret=selected_action(x,w1,b,w2,h,y,torch.tensor([0,2]),charged=charge,kernels=Kernels(e))
        self.assertEqual([v[0] for v in e],['W1','W2']);self.assertIn('action_cache_clone',counts);self.assertIn('action_delta_allocate',counts);self.assertTrue(torch.equal(ret,y));self.assertNotEqual(ret.data_ptr(),y.data_ptr())
if __name__=='__main__':unittest.main()
