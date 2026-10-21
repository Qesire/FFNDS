"""Ideal full-support identity with nonunit smoothing and both original biases."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from native_affine_fold import fold_parameters


class AffineFoldIdentityTests(unittest.TestCase):
    def test_bias_and_smoothing_are_included_exactly_once(self):
        x = torch.tensor([[1., -2.], [.5, 3.]], dtype=torch.float64)
        w0 = torch.tensor([[1., 2.], [-3., .5], [2., -1.]], dtype=torch.float64)
        w2 = torch.tensor([[1., -2., 3.], [.5, 2., -1.]], dtype=torch.float64)
        b0 = torch.tensor([.25, -.5, .75], dtype=torch.float64)
        b2 = torch.tensor([2., -3.], dtype=torch.float64)
        smooth = torch.tensor([.5, 2., 4.], dtype=torch.float64)
        z = x @ w0.T + b0
        # Arbitrary post-Q2 values: no assumption about GELU or quantizer linearity.
        v2 = torch.tensor([[2., .25, 1.], [-1., 2., 3.]], dtype=torch.float64)
        phi = v2 - .5 * z / smooth
        a, bias = fold_parameters(w0, w2, b0, smooth)
        reconstructed = x @ a.T + bias + phi @ w2.T + b2
        self.assertTrue(torch.allclose(reconstructed, v2 @ w2.T + b2, atol=1e-12, rtol=0))
