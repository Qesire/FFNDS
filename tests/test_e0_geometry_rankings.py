import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from analyze_e0_geometry_rankings import compare, scores


class GeometryIdentities(unittest.TestCase):
    def test_finite_normalization_has_centered_angular_identity(self):
        # Independent closed-form cosine identity, including nonzero channel means.
        x = np.array([[3., 1., 2., 4.], [2., 5., 0., -1.]])
        delta = np.array([[[.1, -.2, .3, .1], [.2, .2, .2, .2]],
                          [[-.2, .3, .1, .2], [.1, .1, .1, .1]]])
        measured = scores(x, delta, eps=0)
        for row in range(2):
            for group in range(2):
                a = x[row]-x[row].mean()
                b = x[row]-delta[row, group]
                b -= b.mean()
                cosine = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
                expected = 2*len(a)*(1-cosine)
                self.assertAlmostEqual(measured["finite_ideal_LN"][row, group], expected, places=13)

    def test_radial_alignment_can_reverse_energy_order(self):
        x = np.array([[1., -1., 0., 0.]])
        delta = np.array([[[.1, -.1, .01, -.01], [0., 0., .05, -.05]]])
        s = scores(x, delta, eps=0)
        self.assertEqual(s["gated_energy"].argmin(axis=1).tolist(), [1])
        self.assertEqual(s["finite_ideal_LN"].argmin(axis=1).tolist(), [0])
        self.assertEqual(s["uncentered_angle_times_2d"].argmin(axis=1).tolist(), [0])
        c = compare(s["gated_energy"], s["finite_ideal_LN"])
        self.assertEqual(c["pair_order_reversals"], 1)
        self.assertGreater(c["relative_excess_target_loss"], 0)

    def test_degenerate_angular_state_is_rejected(self):
        with self.assertRaises(ValueError):
            scores(np.ones((2, 4)), np.ones((2, 3, 4)))


if __name__ == "__main__":
    unittest.main()
