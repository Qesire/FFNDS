import importlib.util
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
p=Path(__file__).resolve().parents[1]/'tools/e0_q2_direct_collateral_full_shape.py';spec=importlib.util.spec_from_file_location('full_shape',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FullShapeTests(unittest.TestCase):
    def test_padding_and_selection(self):
        x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16); seen=[]
        def op(full):
            seen.append(full.clone());return full*2
        actual=m.padded_apply(x,[3,0],5,op)
        self.assertTrue(torch.equal(actual,x*2));self.assertTrue(torch.equal(seen[0][1:3],torch.zeros((2,2),dtype=x.dtype)))
        for ids in ([0,0],[0,5],[-1,0],[0]):
            with self.assertRaises(ValueError):m.padded_apply(x,ids,5,op)
    def test_qdq_and_linear_sampled(self):
        x=torch.tensor([[2.,4.],[6.,8.]],dtype=torch.bfloat16); smooth=torch.tensor([[2.,4.]])
        def fake_qdq(full,force_fp32):
            self.assertTrue(force_fp32);return full
        q=m.qdq_sampled(x,smooth,[3,1],fake_qdq,total_rows=5)
        self.assertTrue(torch.equal(q,(x.float()/smooth).to(torch.bfloat16)))
        w=torch.eye(2,dtype=torch.bfloat16);b=torch.ones(2,dtype=torch.bfloat16)
        self.assertTrue(torch.equal(m.linear_sampled(q,w,b,[3,1],F.linear,total_rows=5),q+1))
    def test_strict_bitwise_baseline(self):
        a=torch.tensor([0.,1.],dtype=torch.bfloat16);m.require_base_match(a,a.clone())
        b=torch.tensor([-0.,1.],dtype=torch.bfloat16)
        with self.assertRaises(ValueError):m.require_base_match(a,b)
    def test_authorization(self):
        plan={'id':m.ID,'limits':dict(m.LIMITS),'cases':[{'source':s,'step':t,'block':b} for s in (0,12) for t in (0,2) for b in (0,39)],'shards':[{'expert':e,'block':b} for e in ('high_noise_model','low_noise_model') for b in (0,39)]}
        a={'experiment_authority':{'current_user_authorized_scopes':[{'id':m.ID,'protocol_sha256':'ok','execution_status':'AUTHORIZED_NOT_STARTED'}]}}
        m.check_authorization(plan,'ok',a)
        with self.assertRaises(ValueError):m.check_authorization(plan,'bad',a)
        plan['limits']['qdq']=33
        with self.assertRaises(ValueError):m.check_authorization(plan,'ok',a)

if __name__=='__main__':unittest.main()
