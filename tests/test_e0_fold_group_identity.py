import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from e0_intrinsic_math import group_values


class FoldGroupIdentity(unittest.TestCase):
    def test_signed_group_remainder_completes_fixed_linear_update(self):
        gen=torch.Generator().manual_seed(7)
        dx=torch.randn(5,3,generator=gen,dtype=torch.float64)
        w1=torch.randn(128,3,generator=gen,dtype=torch.float64)
        w2=torch.randn(3,128,generator=gen,dtype=torch.float64)
        gate=torch.tensor([[1.,-.7,.2]],dtype=torch.float64)
        dq=torch.randn(5,128,generator=gen,dtype=torch.float64)
        kappa=torch.randn(1,128,generator=gen,dtype=torch.float64)
        scale=torch.rand(1,128,generator=gen,dtype=torch.float64)+.5
        linear_coeff=dx@w1.T*kappa/scale
        base=linear_coeff@w2.T*gate
        remainder=group_values(dq-linear_coeff,w2,gate)
        torch.testing.assert_close(base+remainder.sum(1),dq@w2.T*gate)
        torch.testing.assert_close(base+remainder[:,0],
                                  base+(dq-linear_coeff)[:,:64]@w2[:,:64].T*gate)

    def test_whole_group_cost_control(self):
        base_plus_27=5120/(2*13824)+1.5*27/216
        plain_54=1.5*54/216
        self.assertLess(abs(base_plus_27-plain_54),.0024)


if __name__=='__main__':unittest.main()
