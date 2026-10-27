import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ub1_fixed_rr_consumer_math import baseline_error, fit_delta_consumer


class FixedResidualConsumerTests(unittest.TestCase):
    def test_zero_measured_error_preserves_old_weight_and_bias_bits(self):
        f = torch.tensor([[1., 0.], [-1., 2.], [0., 1.]], dtype=torch.bfloat16)
        w = torch.tensor([[.5, -2.]], dtype=torch.bfloat16)
        b = torch.tensor([.25], dtype=torch.bfloat16)
        counts = []
        fitted = fit_delta_consumer(f, torch.zeros((3, 1), dtype=torch.float32), w, b, counts.append)
        self.assertTrue(torch.equal(fitted['weight'].view(torch.int16), w.view(torch.int16)))
        self.assertTrue(torch.equal(fitted['bias'].view(torch.int16), b.view(torch.int16)))
        self.assertEqual(counts, ['consumer_delta_fit'])

    def test_old_BF16_sum_is_the_target_and_RR_is_used_once(self):
        main = torch.tensor([[1.]], dtype=torch.bfloat16)
        residual = torch.tensor([[1 / 256]], dtype=torch.bfloat16)
        y = main + residual
        measured = baseline_error(y, main, residual)
        real_branch_error = y.float() - (main.float() + residual.float())
        self.assertEqual(float(measured), 0.)
        self.assertEqual(float(real_branch_error), -1 / 256)

    def test_correction_preserves_unidentified_prior_and_learns_only_increment(self):
        f = torch.tensor([[-1., 0.], [0., 0.], [1., 0.]], dtype=torch.bfloat16)
        w = torch.tensor([[1., 9.]], dtype=torch.bfloat16)
        b = torch.tensor([.5], dtype=torch.bfloat16)
        e = torch.tensor([[-1.], [0.], [1.]], dtype=torch.float32)
        fit = fit_delta_consumer(f, e, w, b, lambda key: None)
        self.assertGreater(float(fit['weight'][0, 0]), 1.9)
        self.assertEqual(float(fit['weight'][0, 1]), 9.)
        self.assertEqual(float(fit['bias'][0]), .5)

    def test_wrong_dtype_geometry_or_nonfinite_is_rejected_before_fit(self):
        f = torch.ones((3, 2), dtype=torch.bfloat16)
        e = torch.ones((3, 1), dtype=torch.float32)
        w = torch.ones((1, 2), dtype=torch.bfloat16)
        b = torch.ones((1,), dtype=torch.bfloat16)
        counts = []
        for features, errors in [(f.float(), e), (f, e[:1]), (f, e * float('nan'))]:
            with self.assertRaises(ValueError):
                fit_delta_consumer(features, errors, w, b, counts.append)
        self.assertEqual(counts, [])


if __name__ == '__main__':
    unittest.main()
