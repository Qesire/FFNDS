"""M1 tiny algebra/software fixtures; no model or native performance claims."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import torch
import torch.nn.functional as F
import a5_hwg_delta_math_20261030 as m


class Algebra(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.g = torch.Generator().manual_seed(730)

    def rand(self, *shape):
        return torch.randn(*shape, generator=self.g, dtype=torch.float64)

    def test_linear_difference_and_nonlinear_history(self):
        x0,x1,w,b = self.rand(3,8),self.rand(3,8),self.rand(6,8),self.rand(6)
        a0=F.linear(x0,w,b);a1=F.linear(x1,w,b)
        da=F.linear(x1-x0,w,None)
        torch.testing.assert_close(a0+da,a1)
        torch.testing.assert_close(m.delta_hidden(a0,da),F.gelu(a1,approximate="tanh")-F.gelu(a0,approximate="tanh"))
        self.assertGreater(float((m.delta_hidden(a0,da)-F.gelu(da,approximate="tanh")).abs().max()),.1)

    def test_delta_bias_and_gate_once(self):
        h0,h1,w,b=self.rand(5,8),self.rand(5,8),self.rand(4,8),self.rand(4)
        y0,y1=F.linear(h0,w,b),F.linear(h1,w,b)
        # Helper intentionally computes FP32 for deployment comparison.
        got=m.retained_delta(y0,h0,h1,w,torch.ones_like(h0,dtype=torch.bool))
        torch.testing.assert_close(got,y1.float(),rtol=1e-5,atol=2e-6)
        g0,g1=self.rand(4),self.rand(4)
        torch.testing.assert_close(got*g1.float(),y1.float()*g1.float(),rtol=1e-5,atol=2e-6)
        self.assertFalse(torch.allclose(got*g1.float(),y0.float()*g0.float()))

    def test_zero_delta_keeps_old_output(self):
        h,w,y=self.rand(5,8),self.rand(4,8),self.rand(5,4)
        torch.testing.assert_close(m.retained_delta(y,h,h,w,torch.ones_like(h,dtype=torch.bool)),y.float())

    def test_quantization_does_not_commute_with_difference(self):
        # Small nonlinear rounding counterexample, explicitly NOT native HiF4.
        q=lambda x:torch.round(x)
        h0=torch.tensor([.49]);h1=torch.tensor([.51])
        self.assertFalse(torch.equal(q(h1-h0),q(h1)-q(h0)))

    def test_smooth_coordinates_and_current_gate(self):
        w=torch.tensor([[2.,4.],[3.,1.]])
        s=torch.tensor([2.,.5]);g=torch.tensor([.1,2.])
        # Without Q2, raw-h sensitivity divides the effective W2 by smooth2.
        torch.testing.assert_close(m.hwg(w/s,g),m.hwg(w,g)/s)
        self.assertFalse(torch.allclose(m.hwg(w,g),m.hwg(w,torch.ones(2))))

    def test_full_original_groups_and_tie(self):
        score=torch.ones(2,128);groups=m.stable_keep(m.group_scores(score),1)
        mask=m.expand_groups(groups)
        self.assertTrue(bool(mask[:,:64].all()));self.assertFalse(bool(mask[:,64:].any()))
        with self.assertRaises(ValueError):m.group_scores(torch.ones(2,65))

    def test_signed_cancellation_defeats_energy_ranking(self):
        w=torch.tensor([[1.,-1.,.1]])
        dh=torch.tensor([[10.,10.,1.]])
        self.assertLess(abs(float(F.linear(dh,w))),abs(float(F.linear(dh*torch.tensor([[1.,0.,1.]]),w))))

    def test_changing_support_exact_decomposition(self):
        a,b=self.rand(4,8),self.rand(4,8)
        s0,s1=m.stable_keep(a.abs(),3),m.stable_keep(b.abs(),3)
        value,flip=m.signed_support_temporal(a,b,s0,s1)
        torch.testing.assert_close(value+flip,b*s1-a*s0)

    def test_adapter_recipe_zero_rows_and_identity(self):
        w=self.rand(6,128).float();w[0]=0
        p,g,b=m.proxy_weight(w)
        self.assertEqual(m.ADAPTER,"A5_INSPIRED_MEANABS64_ROW_LS_V1")
        torch.testing.assert_close(p,b[:,None]*g.repeat_interleave(64,1)*w.sign())
        self.assertTrue(bool(torch.isfinite(p).all()));self.assertEqual(float(b[0]),0.)


if __name__ == "__main__":
    unittest.main()
