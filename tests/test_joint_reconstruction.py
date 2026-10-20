import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from joint_reconstruction import joint_update,output
from reduced_rank_residual import predict_y


class JointReconstructionTest(unittest.TestCase):
    def test_complementary_features_are_recovered_with_original_bias_and_negative_gate(self):
        x=torch.tensor([[a,b] for a in [-1.,1.] for b in [-1.,1.] for _ in range(2)],dtype=torch.bfloat16)
        feature=torch.tensor([[-1.],[1.]]*4,dtype=torch.bfloat16)
        bias=torch.tensor([.5,-.25],dtype=torch.bfloat16);gate=torch.tensor([[1.,-2.]],dtype=torch.bfloat16)
        weight=torch.zeros(2,1,dtype=torch.bfloat16);base=bias[None].expand(8,-1).clone()
        rr={'first':torch.tensor([[1.,0.]],dtype=torch.bfloat16),'second':torch.tensor([[1.],[0.]],dtype=torch.bfloat16),
            'bias_first':torch.zeros(1,dtype=torch.bfloat16),'bias_second':torch.zeros(2,dtype=torch.bfloat16)}
        dense=base+predict_y(x,rr)+feature*torch.tensor([[0.,2.]],dtype=torch.bfloat16)
        before=float(((base+predict_y(x,rr)-dense).float()*gate).square().sum())
        model=joint_update(feature,x,dense,base,gate,weight,bias,rr,lambda w,**_:w,
            regularization_W=1e-6,regularization_R=1e-6,rank=1,basis_rank=1)
        actual=output(feature,x,model)
        self.assertGreater(before,0)
        torch.testing.assert_close(actual,dense,rtol=0,atol=0)
        torch.testing.assert_close(model['bias'],bias,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
