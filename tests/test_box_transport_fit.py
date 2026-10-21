"""A two-parameter constrained quadratic must not become a clipping heuristic."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from residual_gain_transport import box_quadratic


class BoxTransportFitTests(unittest.TestCase):
    def test_correlated_active_constraint_reoptimizes_free_coordinate(self):
        fit = box_quadratic([[2., 1.], [1., 2.]], [-2., 0.])
        self.assertTrue(np.allclose(fit['delta'], [.25, -.125]))

    def test_known_unconstrained_solution(self):
        fit = box_quadratic([[2., 0.], [0., 4.]], [-.2, .4])
        self.assertTrue(np.allclose(fit['delta'], [.1, -.1]))

    def test_flat_objective_prefers_identity(self):
        self.assertEqual(box_quadratic(np.zeros((2, 2)), [0., 0.])['gains'], [1., 1.])

    def test_nonconvex_objective_is_rejected(self):
        with self.assertRaises(ValueError):
            box_quadratic([[-1., 0.], [0., 1.]], [0., 0.])
