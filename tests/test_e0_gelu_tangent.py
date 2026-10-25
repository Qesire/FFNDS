import sys
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from e0_gelu_tangent import derivative,folding_mac_ratio


class TangentTests(unittest.TestCase):
    def test_derivative_matches_original_tanh_gelu_autograd(self):
        a=torch.linspace(-8,8,401,dtype=torch.float64,requires_grad=True)
        exact,=torch.autograd.grad(F.gelu(a,approximate='tanh').sum(),a)
        torch.testing.assert_close(derivative(a),exact,atol=1e-14,rtol=1e-12)

    def test_shared_linear_folding_and_composition_cost(self):
        torch.manual_seed(3)
        x=torch.randn(11,3,dtype=torch.float64);w1=torch.randn(5,3,dtype=torch.float64)
        w2=torch.randn(3,5,dtype=torch.float64);slope=torch.randn(5,dtype=torch.float64)
        torch.testing.assert_close((x@w1.T*slope)@w2.T,x@((w1.T*slope)@w2.T))
        self.assertLess(folding_mac_ratio(1),.27)
        self.assertLess(folding_mac_ratio(10),1.)
        self.assertGreater(folding_mac_ratio(11),1.)
        self.assertGreater(folding_mac_ratio(248),20.)


if __name__=='__main__':unittest.main()
