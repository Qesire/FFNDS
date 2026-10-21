"""Conditional minimizer from the original quadratic, independent of GPTQ code."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from group_hessian_quant import conditional_correction, encode_groups


class GroupHessianQuantTests(unittest.TestCase):
    def test_joint_block_correction_equals_direct_conditional_minimizer(self):
        h = torch.tensor([[4., 1., 1.], [1., 3., 2.], [1., 2., 5.]], dtype=torch.float64)
        upper = torch.linalg.cholesky(torch.linalg.inv(h), upper=True)
        error = torch.tensor([[.2, -.3]], dtype=torch.float64)
        actual = conditional_correction(error, upper[:2, :2], upper[:2, 2:])
        expected = error @ h[:2, 2:] @ torch.linalg.inv(h[2:, 2:])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=0))
        self.assertAlmostEqual(float(actual), -.08)

    def test_identity_quantizer_has_no_compensation_or_weight_change(self):
        w = torch.tensor([[.25, -.5, .75, 1.], [1., 2., 3., 4.]], dtype=torch.float64)
        h = torch.eye(4, dtype=torch.float64) + torch.ones((4, 4), dtype=torch.float64)
        q, meta = encode_groups(w, h, lambda value: value.clone(), group_size=2)
        self.assertTrue(torch.equal(q, w))
        self.assertEqual(meta['groups'], 2)
