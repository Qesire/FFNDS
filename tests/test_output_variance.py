import unittest

import torch

from tools.output_variance import affine_reference,fit_output_moments


class OutputVarianceTest(unittest.TestCase):
    def test_known_bias_free_affine_and_negative_gate_are_recovered(self):
        source=torch.tensor([[1.,2.],[2.,0.],[0.,-1.],[-1.,1.]],dtype=torch.bfloat16)
        bias=torch.tensor([.5,-.25],dtype=torch.bfloat16)
        shift=torch.tensor([1.,-2.],dtype=torch.float32)
        target=affine_reference(source,1.5,shift,bias)
        fit=fit_output_moments(target,source,torch.tensor([1.,-2.],dtype=torch.bfloat16),bias)
        self.assertEqual(fit['beta'],1.5)
        torch.testing.assert_close(fit['shift'],shift,rtol=0,atol=0)
        self.assertTrue(torch.equal(affine_reference(source,fit['beta'],fit['shift'],bias),target))
        # At zero signal the original bias is reinserted once, plus the learned
        # signal shift. Scaling the original bias twice would fail this case.
        self.assertTrue(torch.equal(affine_reference(bias[None],1.5,shift,bias),(shift+bias.float())[None].to(torch.bfloat16)))

    def test_constant_candidate_is_rejected_without_silent_gain_explosion(self):
        with self.assertRaises(ValueError):
            fit_output_moments(torch.tensor([[1.,2.],[2.,3.]],dtype=torch.bfloat16),
                torch.ones(2,2,dtype=torch.bfloat16),torch.ones(2,dtype=torch.bfloat16),torch.zeros(2,dtype=torch.bfloat16))


if __name__=='__main__':unittest.main()
