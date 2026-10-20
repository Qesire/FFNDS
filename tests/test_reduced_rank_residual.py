import unittest

import torch

from tools.reduced_rank_residual import fit_rrr,native_parameters,predict_y


class ReducedRankResidualTest(unittest.TestCase):
    def test_full_rank_matches_independent_primal_ridge(self):
        generator=torch.Generator().manual_seed(13)
        x=torch.randn(30,5,generator=generator,dtype=torch.float64)
        y=torch.randn(30,4,generator=generator,dtype=torch.float64)+2.
        model=fit_rrr(x,y,.1,4)
        xc=(x-x.mean(0))/model['scale_x']
        expected=torch.linalg.solve(xc.T@xc/len(x)+.1*torch.eye(5,dtype=torch.float64),xc.T@(y-y.mean(0))/len(x))
        actual=model['left']@model['right'].T*model['scale_x']
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)

    def test_zero_gate_is_safe_and_native_predictor_finite(self):
        x=torch.arange(60,dtype=torch.float64).reshape(20,3)/10
        y=torch.stack([x[:,0],torch.zeros(20),x[:,2]],dim=1)
        model=fit_rrr(x,y,.1,3)
        parameters=native_parameters(model,torch.tensor([1.,0.,-2.]),3)
        prediction=predict_y(x,parameters)
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        self.assertTrue(bool((prediction[:,1]==0).all()))


if __name__=='__main__':
    unittest.main()
