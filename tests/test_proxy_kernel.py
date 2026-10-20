from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from proxy_kernel import fit_prepared, predict, prepare_fit


class ProxyKernelNumerics(unittest.TestCase):
    def test_dual_linear_ridge_matches_independent_primal_solution(self):
        xx = np.array([[-2., 0.], [-1., 1.], [0., -1.], [1., .5], [2., -.5], [3., 1.5]])
        yy = np.column_stack([10 + 2 * xx[:, 0] - xx[:, 1], 20 - xx[:, 0] + .5 * xx[:, 1]])
        query = np.array([[.2, .3], [-2., 2.]])
        lam = .03
        prepared = prepare_fit(xx, yy, "linear")
        model = fit_prepared(prepared, lam)
        train = (xx - xx.mean(0)) / prepared["scale_x"]
        penalty = len(xx) * lam * xx.shape[1] * prepared["kernel_scale"]
        primal = np.linalg.solve(train.T @ train + penalty * np.eye(2), train.T @ (yy - yy.mean(0)))
        expected = ((query - xx.mean(0)) / prepared["scale_x"]) @ primal + yy.mean(0)
        np.testing.assert_allclose(predict(model, query), expected, rtol=1e-12, atol=1e-12)

    def test_poly2_represents_positive_quadratic_labels(self):
        xx = np.array([(i / 2, j / 2) for i in range(-2, 3) for j in range(-2, 3)])
        yy = (5 + xx[:, 0] ** 2 + .2 * xx[:, 1] ** 2)[:, None]
        model = fit_prepared(prepare_fit(xx, yy, "poly2"), 1e-7)
        self.assertLess(float(np.max(np.abs(predict(model, xx) - yy))), 1e-4)

    def test_zero_variance_input_and_static_model_stay_finite(self):
        xx = np.ones((8, 3))
        yy = np.arange(16).reshape(8, 2) + 1.
        for kind in ["static", "linear", "poly2"]:
            model = fit_prepared(prepare_fit(xx, yy, kind), .01)
            np.testing.assert_allclose(predict(model, xx), np.broadcast_to(yy.mean(0), yy.shape))


if __name__ == "__main__":
    unittest.main()
