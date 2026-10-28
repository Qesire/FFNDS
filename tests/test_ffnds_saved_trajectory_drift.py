import unittest
import numpy as np
from tools.ffnds_saved_trajectory_drift import diagnostic


class TrajectoryMathTest(unittest.TestCase):
    def test_signed_increment_identity(self):
        dense = np.array([[[[2.0]], [[3.0]]]])
        previous = np.array([[[[1.0]], [[-2.0]]]])
        candidate = dense + np.array([[[[-1.0]], [[2.0]]]])
        r, error = diagnostic(dense, candidate, previous)
        self.assertAlmostEqual(r['error_norm_squared'], 5)
        self.assertAlmostEqual(r['previous_error_norm_squared'], 5)
        self.assertAlmostEqual(r['increment_norm_squared'], 20)
        self.assertAlmostEqual(r['signed_twice_inner_product'], -20)
        self.assertAlmostEqual(r['identity_residual'], 0)
        self.assertEqual(len(r['time_slots']), 2)
        np.testing.assert_array_equal(error, candidate-dense)

    def test_zero_denominators(self):
        z = np.zeros((1, 2, 1, 1))
        r, _ = diagnostic(z, z, z)
        self.assertIsNone(r['relative_sse'])
        self.assertIsNone(r['cosine'])
        self.assertIsNone(r['increment_cosine_previous_error'])
        self.assertIsNone(r['error_norm_ratio_to_previous'])
        self.assertIsNone(r['time_slot_equal_mean_relative_sse'])

if __name__ == '__main__': unittest.main()
