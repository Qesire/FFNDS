import unittest

import numpy as np

from tools.proxy_energy import corrected_scores, fit_energy_residual


class EnergyCorrectionTest(unittest.TestCase):
    def test_weighted_donor_fit_and_new_input_prediction(self):
        energy = np.array([1., 2., 4.])
        weights = np.array([1., 3., 2.])
        mean = weights @ energy / weights.sum()
        raw = np.full((3, 2), 7.)
        beta = np.array([2., -1.])
        target = raw + (energy / mean)[:, None] * beta
        model = fit_energy_residual(raw, target, energy, weights)
        np.testing.assert_allclose(model["beta"], beta, atol=1e-14)
        new_energy = np.array([3., 5.])
        new_raw = np.full((2, 2), 9.)
        expected = new_raw + (new_energy / mean)[:, None] * beta
        np.testing.assert_allclose(corrected_scores(new_raw, new_energy, model), expected, atol=1e-14)
        self.assertEqual(model["mean_energy"], mean)

    def test_clipping_and_training_geometry(self):
        model = {"beta": np.array([-20., 2.]), "mean_energy": 1.}
        np.testing.assert_array_equal(corrected_scores(np.ones((1, 2)), np.ones(1), model), [[0., 3.]])
        with self.assertRaises(ValueError):
            fit_energy_residual(np.ones((3, 2)), np.ones((3, 2)), np.ones(2), np.ones(3))


if __name__ == "__main__":
    unittest.main()
