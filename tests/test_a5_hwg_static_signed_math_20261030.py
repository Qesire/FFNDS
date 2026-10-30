"""Tiny signed calibration counterexamples; no corpus/model execution."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('a5_static', Path(__file__).resolve().parents[1] / 'tools/a5_hwg_static_signed_math_20261030.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SignedStaticFixtures(unittest.TestCase):
    def test_float32_signed_Gram_can_lose_small_cancellation_energy(self):
        vectors = torch.tensor([[10000., 1.], [-10000., 1.]], dtype=torch.float64)
        real = vectors @ vectors.T
        rounded = vectors.float() @ vectors.float().T
        direct = vectors.sum(0).square().sum()
        self.assertEqual(float(real.sum()), float(direct))
        self.assertEqual(float(direct), 4.)
        self.assertEqual(float(rounded.sum()), 0.)

    def test_greedy_prefers_small_correct_direction_to_large_cancelling_terms(self):
        a = torch.tensor([[3.], [-3.], [1.]], dtype=torch.float64)
        support, order, _ = m.greedy_support(a @ a.T, a[:, 0], 1)
        self.assertEqual(order, [2])
        self.assertTrue(torch.equal(support, torch.tensor([False, False, True])))

    def test_group_statistics_preserve_signed_cross_and_source_normalization(self):
        delta = torch.tensor([[3., -3., 1.], [6., -6., 2.]])
        w = torch.ones(1, 3)
        target = torch.tensor([[1.], [2.]])
        gram, corr, info = m.group_quadratic(delta, w, target, torch.ones(1), target, group_size=1)
        self.assertLess(float(gram[0, 1]), 0.)
        self.assertEqual(info['batched_W2_shape'], [3, 2, 1, 1])
        self.assertEqual(m.greedy_support(gram, corr, 1)[1], [2])
        doubled = m.group_quadratic(delta * 2, w, target * 2, torch.ones(1), target * 2, group_size=1)
        self.assertTrue(torch.allclose(gram, doubled[0]))
        self.assertTrue(torch.allclose(corr, doubled[1]))

    def test_negative_gain_is_kept_when_exact_K_requires_it_and_ties_are_stable(self):
        support, order, gains = m.greedy_support(torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 2)
        self.assertEqual(order, [0, 1])
        self.assertEqual(gains, [-1., -1.])
        self.assertEqual(int(support.sum()), 2)


if __name__ == '__main__': unittest.main()
