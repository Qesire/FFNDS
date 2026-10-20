import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from folded_adapter import packed_forward
from full_weight_adapter import FullMainWeights


class FullMainWeightControl(unittest.TestCase):
    def test_forward_is_native_and_both_full_matrices_receive_gradients(self):
        generator = torch.Generator().manual_seed(72)
        w0 = torch.randn((8, 5), generator=generator).bfloat16()
        w2 = torch.randn((5, 8), generator=generator).bfloat16()
        model = FullMainWeights(w0, w2)
        def quantizer(x, *, force_fp32): return (x.detach().float() * 16).round().div(16).to(x.dtype)
        actual = model.weights(quantizer)
        for value, prior in zip(actual, [w0, w2]):
            expected = quantizer(prior.float(), force_fp32=True).bfloat16()
            self.assertTrue(torch.equal(value.view(torch.int16), expected.view(torch.int16)))
        self.assertEqual(float(model.relative_deviation().detach()), 0.)
        x = torch.randn((9, 5), generator=generator).bfloat16()
        output = packed_forward(x, actual[0], torch.zeros(8).bfloat16(), actual[1], torch.zeros(5).bfloat16(),
            torch.ones(8), 1.1, quantizer, training=True)
        (output.float() - 1).square().mean().backward()
        for value in model.parameters():
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(float(value.grad.abs().sum()), 0.)
        with torch.no_grad(): model.weight0.add_(.01)
        self.assertGreater(float(model.relative_deviation().detach()), 0.)


if __name__ == '__main__': unittest.main()
