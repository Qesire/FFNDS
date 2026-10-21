"""Learned decomposition and signed group selection, independent known vectors."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from affine_remainder import fit_channel_affine, fold_learned, greedy_fixed_support


class AffineRemainderTests(unittest.TestCase):
    def test_fit_recovers_channel_slope_intercept_and_constant_channel(self):
        x = torch.tensor([[0., 2.], [1., 2.], [2., 2.]], dtype=torch.float64)
        y = torch.tensor([[3., 7.], [5., 7.], [7., 7.]], dtype=torch.float64)
        slope, intercept = fit_channel_affine(x, y)
        self.assertTrue(torch.equal(slope, torch.tensor([2., 0.])))
        self.assertTrue(torch.equal(intercept, torch.tensor([3., 7.])))

    def test_folded_bias_contains_original_bias_once(self):
        full = {'w0': torch.tensor([[1., 2.], [2., -1.]]), 'w2': torch.tensor([[1., -2.], [3., 1.]]),
                'b0': torch.tensor([1., -1.]), 'smooth2': torch.tensor([2., .5])}
        a, bias = fold_learned(full, torch.tensor([2., 3.]), torch.tensor([.5, 1.]), torch.tensor([1., -2.]))
        self.assertTrue(torch.equal(a.float(), torch.tensor([[-7.75, 4.5], [4.75, -.5]])))
        self.assertTrue(torch.equal(bias.float(), torch.tensor([11.25, 2.75])))

    def test_signed_selection_uses_target_and_cross_terms(self):
        c = torch.tensor([[2., 0.], [-2., 0.], [0., 1.]], dtype=torch.float64)
        target = torch.tensor([0., 1.], dtype=torch.float64)
        result = greedy_fixed_support(c @ c.T, c @ target, 1, float(target.square().sum()))
        self.assertEqual(result['groups'], [2])
        self.assertEqual(result['ideal_training_risk'], 0.)
