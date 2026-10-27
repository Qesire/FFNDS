import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ub1_recovery_math import gated_metrics, group_weight_gram, ridge_consumer, scalar_parameters, select_supports


class RecoveryMathTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.charge = self.calls.append

    def test_group_quadratic_matches_signed_direct_output(self):
        a = torch.tensor([[1., -1., 2., 1.], [2., -2., 1., 2.]], dtype=torch.bfloat16)
        w = torch.tensor([[1., 1., 1., -1.], [2., 2., -2., 1.]], dtype=torch.bfloat16)
        gate = torch.tensor([[2., -1.]], dtype=torch.bfloat16)
        gram = group_weight_gram(w, gate, self.charge, group_size=2)
        supports = select_supports(a, gram, 1, self.charge)
        brute = [float(((a[:, i:i + 2].double() @ w[:, i:i + 2].double().T) * gate.double()).square().sum() / len(a)) for i in [0, 2]]
        self.assertEqual(supports['scores']['cmean'], brute)
        self.assertEqual(supports['cmean'].tolist(), [1])
        self.assertEqual(self.calls, ['group_weight_gram', 'group_activation_gram'])

    def test_fluctuation_includes_between_source_means_and_stable_ties(self):
        a = torch.tensor([[0., 0.], [0., 0.], [2., 2.], [2., 2.]], dtype=torch.bfloat16)
        gram = torch.ones((2, 1, 1), dtype=torch.float64)
        result = select_supports(a, gram, 1, self.charge)
        self.assertEqual(result['scores']['flap'], [1., 1.])
        self.assertEqual(result['flap'].tolist(), [0])

    def test_consumer_retains_prior_for_unobserved_feature_directions(self):
        x = torch.tensor([[-1., 0.], [0., 0.], [1., 0.]], dtype=torch.bfloat16)
        w = torch.tensor([[1., 7.]], dtype=torch.bfloat16)
        b = torch.tensor([.5], dtype=torch.bfloat16)
        y = torch.tensor([[-1.5], [.5], [2.5]], dtype=torch.bfloat16)
        fit = ridge_consumer(x, y, w, b, self.charge, fraction=.01)
        self.assertEqual(float(fit['weight'][0, 1]), 7.)
        self.assertGreater(float(fit['weight'][0, 0]), 1.9)
        self.assertEqual(float(fit['bias'][0]), .5)
        self.assertEqual(self.calls, ['consumer_ridge_fit'])

    def test_consumer_neutral_prior_and_finite_regularized_constant_features(self):
        x = torch.tensor([[1., 2.], [-1., -2.], [0., 1.]], dtype=torch.bfloat16)
        w = torch.tensor([[2., -1.]], dtype=torch.bfloat16)
        b = torch.tensor([1.], dtype=torch.bfloat16)
        y = torch.nn.functional.linear(x, w, b)
        fit = ridge_consumer(x, y, w, b, self.charge)
        self.assertTrue(torch.equal(fit['weight'].view(torch.int16), w.view(torch.int16)))
        self.assertTrue(torch.equal(fit['bias'].view(torch.int16), b.view(torch.int16)))
        const = ridge_consumer(torch.zeros_like(x), torch.ones_like(y), w, b, self.charge)
        self.assertTrue(bool(torch.isfinite(const['weight']).all()))
        self.assertEqual(float(const['bias'][0]), 1.)

    def test_scalar_recipe_and_BF16_gate_metric(self):
        a = torch.tensor([[-1.], [0.], [1.], [2.]], dtype=torch.bfloat16)
        w = torch.tensor([[1.]], dtype=torch.bfloat16)
        dense = 2 * a.float()
        b = torch.tensor([.5], dtype=torch.bfloat16)
        gate = torch.tensor([[2.]], dtype=torch.bfloat16)
        scalar = scalar_parameters(a, w, b, gate, dense, self.charge, rows_per_source=2)
        self.assertEqual(scalar['alpha'], 2.)
        self.assertEqual(float(scalar['bias'][0]), .5)
        self.assertEqual(self.calls, ['scalar_signal_FP32_linear'] * 2)
        metrics = gated_metrics(a + 1, a, gate)
        self.assertEqual(metrics['SSE'], 16.)
        self.assertEqual(metrics['target_energy'], 24.)
        with self.assertRaises(ValueError):
            gated_metrics(a, a[:1], gate)


if __name__ == '__main__':
    unittest.main()
