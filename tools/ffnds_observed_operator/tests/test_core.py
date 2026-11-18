"""Algebraic checks only: no model tensors, CAL fitting, replay, or GPU jobs."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from tools.ffnds_observed_operator import (
    compile_native_modes, dense_ffn, fit_mode_coefficients, fit_observed_readout,
    make_mode_anchor, make_observed_anchor, mode_hit, native_mode_templates,
    observed_hit, paired_features, prepare_support, select_probes,
)


class ObservedOperatorAlgebra(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(72)

    def test_exact_recovery_zero_row_and_changed_output_direction(self):
        x = self.rng.normal(size=(97, 8))
        x[0] = 0
        wanted = self.rng.normal(size=(8, 19))
        prior = self.rng.normal(size=wanted.shape)
        rows = np.arange(1, 33)
        target = x @ wanted
        fit = fit_observed_readout(x, target[rows], rows, ridge=0, cal_floor=0, prior=prior)
        assert_allclose(x @ fit.readout, target, rtol=1e-12, atol=1e-12)
        assert_allclose(x[0] @ fit.readout, np.zeros(19), atol=0)
        self.assertEqual(fit.feature_rank, 8)
        self.assertEqual(fit.probe_rank, 8)

        scalar = np.arange(1, 18, dtype=float)[:, None]
        rows = np.arange(8)
        for direction in (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])):
            fit = fit_observed_readout(scalar, (scalar @ direction)[rows], rows,
                                      ridge=0, cal_floor=0)
            assert_allclose(fit.readout, direction, atol=1e-13)

        # Identifiable after current whitening: raw lstsq(rcond) would drop the
        # second column again despite it surviving the declared support check.
        varied_scale = np.ones((64, 2))
        varied_scale[:32, 1] = np.tile([1e-11, -1e-11], 16)
        varied_scale[32:, 1] = np.tile([1e-9, -1e-9], 16)
        response = varied_scale @ np.array([[0.0], [1e9]])
        fit = fit_observed_readout(varied_scale, response[:32], np.arange(32),
                                  ridge=0, cal_floor=0)
        self.assertEqual(fit.feature_rank, 2)
        self.assertEqual(fit.probe_rank, 2)
        assert_allclose(varied_scale @ fit.readout, response, rtol=1e-11, atol=1e-11)

    def test_covariant_metric_and_small_displacement_noise_floor(self):
        x = self.rng.normal(size=(97, 8))
        rows = np.arange(32)
        target = self.rng.normal(size=(32, 11))
        prior = self.rng.normal(size=(8, 11))
        basis = np.eye(8) + 0.12 * self.rng.normal(size=(8, 8))
        first = fit_observed_readout(x, target, rows, ridge=0.2, prior=prior,
                                    cal_floor=0.01, cal_gram=np.eye(8))
        second = fit_observed_readout(x @ basis, target, rows, ridge=0.2,
                                     prior=np.linalg.solve(basis, prior),
                                     cal_floor=0.01, cal_gram=basis.T @ basis)
        assert_allclose(x @ first.readout, x @ basis @ second.readout, rtol=1e-11, atol=1e-11)

        # Fixed observer noise must not produce an O(1) update as X tends to 0.
        noise = self.rng.normal(size=(32, 11)) * 0.03
        amplitudes = []
        pure_relative = []
        for epsilon in (1.0, 1e-3, 1e-6):
            small_x = epsilon * x
            noisy_observation = small_x[rows] @ prior + noise
            fit = fit_observed_readout(small_x, noisy_observation, rows,
                                      ridge=0.1, prior=prior, cal_floor=0.01)
            amplitudes.append(fit.prediction_update_rms)
            old_limit = fit_observed_readout(small_x, noisy_observation, rows,
                                            ridge=0.1, prior=prior, cal_floor=0)
            pure_relative.append(old_limit.prediction_update_rms)
        self.assertGreater(amplitudes[0], amplitudes[1])
        self.assertLess(amplitudes[2], amplitudes[1] * 2e-6)
        assert_allclose(pure_relative, np.repeat(pure_relative[0], 3), rtol=1e-10)

    def test_zero_and_unobserved_feature_directions(self):
        rows = np.arange(16)
        x = np.zeros((64, 2))
        x[:32, 0] = 1
        x[32:, 1] = 1
        target = np.ones((16, 3))
        with self.assertRaisesRegex(ValueError, "do not identify"):
            fit_observed_readout(x, target, rows, ridge=0)
        fit = fit_observed_readout(x, target, rows, ridge=0.1)
        self.assertEqual(fit.probe_rank, 1)
        assert_allclose(fit.readout[1], np.zeros(3), atol=1e-14)

        prior = self.rng.normal(size=(2, 3))
        zero = fit_observed_readout(np.zeros_like(x), target, rows, ridge=0.1, prior=prior)
        self.assertEqual(zero.feature_rank, 0)
        assert_allclose(zero.readout, prior, atol=0)
        self.assertEqual(zero.prediction_update_rms, 0)

    def test_anchor_observer_uses_current_rows_and_keeps_endpoint_identity(self):
        n, d, m = 97, 4, 3
        w1, w2 = self.rng.normal(size=(m, d)), self.rng.normal(size=(d, m))
        b1, b2 = self.rng.normal(size=m), self.rng.normal(size=d)
        matrix = (w1.T * self.rng.normal(size=m)) @ w2.T
        u0 = self.rng.normal(size=(n, d))
        u1 = u0 + self.rng.normal(size=(n, d)) * 0.3
        u1[0] = u0[0]
        beta = self.rng.normal(size=m)
        scales = np.linspace(0.7, 1.3, 2 * m)
        phi = lambda u: paired_features(u, w1, b1, np.arange(m), beta, scales)
        native = lambda u: dense_ffn(u, w1, b1, w2, b2)
        anchor = make_observed_anchor(u0, native(u0), matrix, phi(u0))
        calls = []

        def observer(rows):
            calls.append(rows.copy())
            return native(rows)

        prediction, receipt = observed_hit(u1, phi(u1), anchor, observer, p=64,
                                           ridge=0, cal_floor=0, policy="geometry")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].shape, (64, d))
        assert_allclose(calls[0], u1[receipt["probe_rows"]], atol=0)
        assert_allclose(prediction, native(u1), rtol=1e-10, atol=1e-10)
        assert_allclose(prediction[0], native(u0)[0], rtol=1e-13, atol=1e-13)
        prior = self.rng.normal(size=(2 * m, d))
        static, static_receipt = observed_hit(u1, phi(u1), anchor, observer, p=64,
                                              ridge=0.1, prior=prior, adaptation=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(static_receipt["probe_rows"]), 0)
        assert_allclose(static, u1 @ matrix + anchor.residual + (phi(u1) - phi(u0)) @ prior,
                        rtol=0, atol=0)

    def test_probe_selection_tracks_global_ids_and_retained_geometry(self):
        x = self.rng.normal(size=(97, 8))
        ids = self.rng.permutation(np.arange(1000, 1097))
        order = self.rng.permutation(97)
        for policy in ("fixed", "geometry"):
            first = select_probes(x, 64, policy=policy, token_ids=ids)
            second = select_probes(x[order], 64, policy=policy, token_ids=ids[order])
            assert_allclose(ids[first], ids[order][second], atol=0)
            self.assertEqual(len(np.unique(first)), 64)
        exact_ids = ids[[9, 3, 40, 7]]
        selected = select_probes(x, 4, token_ids=ids, fixed_ids=exact_ids)
        assert_allclose(selected, np.array([9, 3, 40, 7]), atol=0)
        zero = select_probes(np.zeros((97, 2)), 64, policy="geometry", token_ids=ids)
        assert_allclose(ids[zero], np.sort(ids)[:64], atol=0)
        support = prepare_support(np.column_stack((x[:, 0], x[:, 0], np.zeros(97))))
        self.assertEqual(support.rank, 1)

    def test_native_mode_contractions_and_full_rank_family_hit(self):
        n, d, m, q = 53, 5, 9, 4
        w1, w2 = self.rng.normal(size=(m, d)), self.rng.normal(size=(d, m))
        modes = self.rng.normal(size=(q, m))
        theta = self.rng.normal(size=q)
        full = np.stack([(w1.T * mode) @ w2.T for mode in modes])
        du = self.rng.normal(size=(n, d))
        templates = native_mode_templates(du[:16], w1, w2, modes)
        assert_allclose(templates, np.stack([du[:16] @ b for b in full]), rtol=1e-12, atol=1e-12)
        compiled = compile_native_modes(theta, w1, w2, modes)
        assert_allclose(compiled, np.einsum("j,jic->ic", theta, full), rtol=1e-12, atol=1e-12)
        self.assertEqual(np.linalg.matrix_rank(compiled), d)

        base = self.rng.normal(size=(d, d))
        u0 = self.rng.normal(size=(n, d))
        u1 = u0 + du
        true_matrix = base + compiled
        bias = self.rng.normal(size=d)
        native = lambda u: u @ true_matrix + bias
        anchor = make_mode_anchor(u0, native(u0), base)
        gate = np.linspace(0.5, 2, d)
        prediction, receipt = mode_hit(u1, anchor, native, w1, w2, modes,
                                       gate, np.arange(16), ridge=0)
        assert_allclose(receipt["fit"].theta, theta, rtol=1e-11, atol=1e-11)
        assert_allclose(prediction, native(u1), rtol=1e-11, atol=1e-11)
        static, receipt = mode_hit(u1, anchor, None, w1, w2, modes, gate, None,
                                   ridge=0.1, adaptation=False)
        assert_allclose(static, native(u0) + du @ base, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(receipt["probe_rows"]), 0)
        fixed_theta = theta * 0.7
        static, receipt = mode_hit(u1, anchor, None, w1, w2, modes, gate, None,
                                   ridge=0.1, prior=fixed_theta, adaptation=False)
        fixed_matrix = base + compile_native_modes(fixed_theta, w1, w2, modes)
        assert_allclose(static, native(u0) + du @ fixed_matrix, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(receipt["probe_rows"]), 0)

    def test_gate_changes_shared_mode_optimum(self):
        templates = np.ones((1, 3, 2))
        target = np.tile(np.array([1.0, 3.0]), (3, 1))
        left = fit_mode_coefficients(templates, target, np.array([10.0, 1.0]), ridge=0)
        right = fit_mode_coefficients(templates, target, np.array([1.0, 10.0]), ridge=0)
        assert_allclose(left.theta, np.array([103.0 / 101.0]), atol=1e-13)
        assert_allclose(right.theta, np.array([301.0 / 101.0]), atol=1e-13)
        self.assertGreater(right.theta[0] - left.theta[0], 1.9)


if __name__ == "__main__":
    unittest.main()
