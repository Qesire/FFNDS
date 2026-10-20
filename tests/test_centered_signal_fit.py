import unittest

import numpy as np

from tools.centered_signal_fit import fit_signal_moments


class CenteredSignalFitTest(unittest.TestCase):
    def test_gain_and_offset_match_independent_weighted_intercept_fit(self):
        signal = np.array([[1., -2.], [3., 2.], [-2., 1.], [5., 4.]])
        offset = np.array([10., -3.])
        dense = signal * 1.7 + offset
        weights, gate = np.array([1., 2., 4., 3.]), np.array([2., -.5])
        m = fit_signal_moments(weights.sum(), weights @ signal, weights @ dense,
            np.sum(weights[:, None] * (signal * gate) ** 2),
            np.sum(weights[:, None] * signal * dense * gate ** 2), gate)
        self.assertAlmostEqual(m['centered_alpha'], 1.7, places=13)
        learned_offset = m['mean_dense'] - m['centered_alpha'] * m['mean_signal']
        np.testing.assert_allclose(learned_offset, offset, atol=1e-13)
        # Original bias is added once after fitting the bias-free signal.
        original_bias = np.array([7., 9.])
        np.testing.assert_allclose(signal * m['centered_alpha'] + learned_offset + original_bias,
                                   dense + original_bias, atol=1e-13)

    def test_negative_centered_covariance_is_clamped(self):
        signal = np.array([[-1., 1.], [1., -1.]])
        dense = -signal + 3.
        m = fit_signal_moments(2., signal.sum(0), dense.sum(0), (signal ** 2).sum(), (signal * dense).sum(), [1., 1.])
        self.assertEqual(m['centered_alpha'], 0.)


if __name__ == '__main__':
    unittest.main()
