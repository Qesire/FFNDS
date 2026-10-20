import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from folded_adapter import FoldedAdapters, packed_forward, straight_through


class FoldedAdapterControl(unittest.TestCase):
    def test_ste_has_exact_forward_bits_and_identity_gradient(self):
        def quantizer(x, *, force_fp32):
            self.assertTrue(force_fp32)
            return torch.copysign(torch.floor(x.abs() * 4) / 4, x)
        x = torch.tensor([-0., -.12, .45, -.7], requires_grad=True)
        actual = straight_through(x, quantizer); expected = quantizer(x, force_fp32=True)
        self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))
        (actual * torch.tensor([1., 2., 3., 4.])).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([1., 2., 3., 4.]))

    def test_zero_adapter_keeps_original_and_separately_requantizes(self):
        w0 = torch.full((8, 5), .3, dtype=torch.bfloat16); w2 = torch.full((5, 8), -.3, dtype=torch.bfloat16)
        model = FoldedAdapters(w0, w2, rank=2)
        def quantizer(x, *, force_fp32): return x.round()
        original = model.weights(quantizer, original=True); projected = model.weights(quantizer)
        self.assertEqual(original[0].data_ptr(), w0.data_ptr())
        self.assertFalse(torch.equal(original[0], projected[0]))
        self.assertTrue(torch.equal(projected[0], quantizer(w0.float(), force_fp32=True).bfloat16()))

    def test_bf16_nonlinear_path_keeps_forward_and_backpropagates_to_both_updates(self):
        generator = torch.Generator().manual_seed(92)
        w0 = torch.randn((8, 5), generator=generator).bfloat16()
        w2 = torch.randn((5, 8), generator=generator).bfloat16()
        model = FoldedAdapters(w0, w2, rank=2)
        def quantizer(x, *, force_fp32): return (x.detach().float() * 16).round().div(16).to(x.dtype)
        weights = model.weights(quantizer)
        x = torch.randn((9, 5), generator=generator).bfloat16()
        b0, b2 = torch.full((8,), .1).bfloat16(), torch.full((5,), -.1).bfloat16()
        args = (x, weights[0], b0, weights[1], b2, torch.ones(8), 1.1, quantizer)
        training = packed_forward(*args, training=True); native = packed_forward(*args)
        self.assertTrue(torch.equal(training.view(torch.int16), native.view(torch.int16)))
        (training.float() - 1).square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            if name.startswith('b'): self.assertGreater(float(parameter.grad.abs().sum()), 0.)
            else: self.assertEqual(float(parameter.grad.abs().sum()), 0.)

    def test_zero_addition_reference_must_use_identical_effective_zero_bits(self):
        prior = torch.tensor([[-0., .5], [-0., -.5]], dtype=torch.bfloat16)
        model = FoldedAdapters(prior, prior, rank=1)
        # A positive factor makes the zero product +0 on CPU as well as CUDA;
        # a negative single factor can instead leave the product at -0.
        with torch.no_grad():
            model.a0.fill_(1.); model.a2.fill_(1.)
        def quantizer(x, *, force_fp32): return x.clone()
        actual = model.weights(quantizer)[0]
        effective = prior.float() + model.b0 @ model.a0
        expected = quantizer(effective, force_fp32=True).bfloat16()
        self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))
        wrong_reference = quantizer(prior.float(), force_fp32=True).bfloat16()
        self.assertFalse(torch.equal(actual.view(torch.int16), wrong_reference.view(torch.int16)))
        self.assertTrue(torch.equal(actual, wrong_reference))


if __name__ == '__main__': unittest.main()
