import unittest

import torch

from tools.anchored_reconstruction import fit_reconstruction,bf16_bits,from_bf16_bits


class AnchoredReconstructionTest(unittest.TestCase):
    def test_dual_fold_matches_independent_primal_ridge_and_gate_inverse(self):
        rng=torch.Generator().manual_seed(314)
        x=torch.randn(5,8,generator=rng,dtype=torch.float64)
        dense=torch.randn(5,3,generator=rng,dtype=torch.float64).to(torch.bfloat16)
        base=torch.randn(5,3,generator=rng,dtype=torch.float64).to(torch.bfloat16)
        gate=torch.tensor([1.,0.,-2.],dtype=torch.bfloat16)
        prior=torch.randn(3,8,generator=rng,dtype=torch.float64)
        bias=torch.randn(3,generator=rng,dtype=torch.float64)
        w,b=fit_reconstruction(x,dense,base,gate,prior,bias,.3)
        xc=x-x.mean(0);scale=xc.square().mean().sqrt();scaled=xc/scale
        r=((dense.float()-base.float())*gate.float()).double()
        coefficients=torch.linalg.solve(scaled.T@scaled/5+.3*torch.eye(8,dtype=torch.float64),scaled.T@(r-r.mean(0))/5)/scale
        inverse=torch.tensor([1.,0.,-.5],dtype=torch.float64)
        expected_w=(prior+coefficients.T*inverse[:,None]).to(torch.bfloat16)
        expected_b=(bias+(r.mean(0)-x.mean(0)@coefficients)*inverse).to(torch.bfloat16)
        self.assertTrue(torch.equal(w,expected_w));self.assertTrue(torch.equal(b,expected_b))

    def test_bf16_encoding_preserves_signed_zero_and_every_storage_bit(self):
        bits=torch.arange(65536,dtype=torch.int32).to(torch.uint16)
        source=bits.view(torch.bfloat16).reshape(256,256)
        recovered=from_bf16_bits(bf16_bits(source))
        self.assertTrue(torch.equal(recovered.view(torch.uint16),bits.reshape(256,256)))


if __name__=='__main__':unittest.main()
