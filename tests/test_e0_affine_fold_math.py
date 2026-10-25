import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from e0_affine_fold_math import fit_coefficients,absolute_fold_mac_ratio


class AbsoluteFoldTests(unittest.TestCase):
    def test_calibration_known_affine_and_constant_column(self):
        a=torch.tensor([[-2.,3.],[0.,3.],[1.,3.],[4.,3.]],dtype=torch.float64)
        labels=a*torch.tensor([[.7,2.]])+torch.tensor([[-1.,4.]])
        slope,bias=fit_coefficients(a,labels)
        torch.testing.assert_close(a*slope+bias,labels)
        self.assertEqual(float(slope[0,1]),0.)

    def test_fold_includes_both_original_biases_and_exact_group_budget(self):
        torch.manual_seed(11)
        x=torch.randn(7,3,dtype=torch.float64);w1=torch.randn(5,3,dtype=torch.float64)
        w2=torch.randn(3,5,dtype=torch.float64);b1=torch.randn(5,dtype=torch.float64)
        b2=torch.randn(3,dtype=torch.float64);s=torch.randn(5,dtype=torch.float64);b=torch.randn(5,dtype=torch.float64)
        matrix=(w2*s)@w1;offset=w2@(s*b1+b)+b2
        torch.testing.assert_close(((x@w1.T+b1)*s+b)@w2.T+b2,x@matrix.T+offset)
        self.assertEqual(absolute_fold_mac_ratio(27),67/216)
        self.assertEqual(absolute_fold_mac_ratio(54),94/216)


if __name__=='__main__':unittest.main()
