"""Decisive algebraic controls for the new saved-data representations."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from e0_function_basis_math import gaussian_gain, ridge_factors


class FunctionBasisMathTests(unittest.TestCase):
    def test_symmetric_gaussian_half_gain_and_odd_even_secant(self):
        nodes, weights = [torch.as_tensor(a, dtype=torch.float64) for a in np.polynomial.hermite.hermgauss(32)]
        mean = torch.zeros(3, dtype=torch.float64)
        gain = gaussian_gain(mean, torch.tensor([.01, 1, 100.], dtype=torch.float64), nodes, weights)
        self.assertTrue(torch.allclose(gain, torch.full_like(gain, .5), atol=1e-14, rtol=0))
        a = torch.tensor([.2, 2., 10.], dtype=torch.float64)
        phi = lambda t: torch.nn.functional.gelu(t, approximate='tanh')
        self.assertTrue(torch.allclose(phi(a) - phi(-a), a, atol=1e-14, rtol=0))

    def test_predictive_RRR_minimizes_ridge_objective_better_than_output_PCA(self):
        generator = torch.Generator().manual_seed(100)
        x = torch.randn(20, 7, generator=generator, dtype=torch.float64)
        y = torch.randn(20, 9, generator=generator, dtype=torch.float64)
        fit = ridge_factors(x, y, .1, 3, lambda a: torch.linalg.svd(a, full_matrices=False))
        xc, yc = x - fit['mean_x'], y - fit['mean_y']
        objective = lambda b: (yc - xc @ b).square().sum() + fit['lambda'] * b.square().sum()
        pca = fit['pca_left'] @ fit['pca_right']
        predictive = fit['predictive_left'] @ fit['predictive_right']
        self.assertLessEqual(float(objective(predictive)), float(objective(pca)) + 1e-10)
        self.assertLessEqual(int(torch.linalg.matrix_rank(predictive)), 3)

    def test_rank_one_output_need_not_be_predictable_from_input_sketch(self):
        # The cheap observer sees identical values, while the omitted input
        # coordinate determines the rank-one output's sign.
        observed = torch.ones(2, 1, dtype=torch.float64)
        output = torch.tensor([[1., 2., 3.], [-1., -2., -3.]], dtype=torch.float64)
        self.assertEqual(int(torch.linalg.matrix_rank(output)), 1)
        fit = ridge_factors(observed, output, .1, 1, lambda a: torch.linalg.svd(a, full_matrices=False))
        guess = fit['mean_y'] + (observed - fit['mean_x']) @ fit['predictive_left'] @ fit['predictive_right']
        self.assertTrue(torch.equal(guess[0], guess[1]))
        self.assertGreater(float((guess - output).square().sum()), 0)


if __name__ == '__main__':
    unittest.main()
