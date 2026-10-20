"""Scientific invariants for the repaired AR2 primitives; synthetic data only."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ar2_core import (NativeProjection, ScopeKey, design_weights, donor_key,
                      fit_nonnegative_gain, score_families, static_groups,
                      support_mask, top_groups, weighted_quantile)


class AR2Invariants(unittest.TestCase):
    def test_crossfit_donors_are_identical_for_fit_and_validation(self):
        for seed, opposite in [(20261001, 20261002), (20261002, 20261001)]:
            for step, expert in [(0, "high_noise_model"), (2, "low_noise_model")]:
                expected = ScopeKey(0, opposite, step, expert)
                self.assertEqual(donor_key(ScopeKey(0, seed, step, expert)), expected)
                self.assertEqual(donor_key(ScopeKey(6, seed, step, expert)), expected)

    def test_invalid_scope_binding_is_rejected(self):
        for key in [ScopeKey(6, 42, 0, "high_noise_model"),
                    ScopeKey(6, 20261001, 2, "high_noise_model"),
                    ScopeKey(10, 20261001, 0, "high_noise_model")]:
            with self.assertRaises(ValueError):
                donor_key(key)

    def test_static_support_uses_training_design_weights_and_stable_ties(self):
        train = np.array([[10, 0, 1, 1], [0, 2, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(static_groups(train, [1, 10], 1), [1])
        np.testing.assert_array_equal(top_groups(np.array([4, 4, 4, 3]), 2), [0, 1])

    def test_actual_511_and_512_rows_are_preserved(self):
        for rows in [511, 512]:
            weights = design_weights(np.arange(rows) + 1, rows)
            mask = support_mask(np.array([1, 3]), rows=rows, groups=4, group_size=2)
            self.assertEqual(len(weights), rows)
            self.assertEqual(mask.shape, (rows, 8))
            np.testing.assert_array_equal(mask.sum(1), np.full(rows, 4))
        with self.assertRaises(ValueError):
            design_weights(np.ones(511), 512)

    def test_c_detects_constructive_and_cancelling_cross_terms(self):
        a = torch.tensor([[1., 1., 1., 1.]])
        w = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [-1., 0.]])
        scores = score_families(a, w, torch.ones(1, 2), group_size=2)
        np.testing.assert_array_equal(scores["A"], [[2, 2]])
        np.testing.assert_array_equal(scores["E"], [[2, 2]])
        np.testing.assert_array_equal(scores["C"], [[4, 0]])

    def test_e_and_c_are_invariant_to_positive_channel_rescaling(self):
        a = torch.tensor([[1., 2., -1., 3.], [2., 1., 4., -2.]])
        w = torch.tensor([[1., 2.], [-2., 3.], [4., 1.], [3., -1.]])
        g = torch.tensor([[1., 0.5]])
        scale = torch.tensor([2., 0.5, 4., 2.])
        before = score_families(a, w, g, group_size=2)
        after = score_families(a / scale, w * scale[:, None], g, group_size=2)
        for name in "EC":
            np.testing.assert_array_equal(before[name], after[name])
        self.assertFalse(np.array_equal(before["A"], after["A"]))

    def test_native_gain_does_not_scale_bias_and_preserves_cast_order(self):
        w = torch.tensor([[1., 2.], [3., 4.], [2., -1.], [-1., 2.]], dtype=torch.bfloat16)
        bias = torch.tensor([8., -4.], dtype=torch.bfloat16)
        a = torch.tensor([[1., 2., 3., 4.]], dtype=torch.bfloat16)
        g = torch.tensor([[0.5, 2.]], dtype=torch.bfloat16)
        projection = NativeProjection(w, bias)
        mask = np.array([[1, 1, 0, 0]], dtype=np.float32)
        y = projection.native(a, g, mask, 2.)
        # Retained a=[1,2,0,0], gained a=[2,4,0,0], linear=[14,20].
        self.assertTrue(torch.equal(y, torch.tensor([[11., 32.]], dtype=torch.bfloat16)))
        zero = projection.native(a, g, mask, 0.)
        self.assertTrue(torch.equal(zero, bias * g))

    def test_gain_uses_explicit_weights_and_handles_zero_denominator(self):
        signal = np.array([[1.], [2.]])
        target = np.array([[4.], [2.]])
        fit = fit_nonnegative_gain(signal, target, [1., 3.])
        self.assertEqual(fit.numerator, 16.)
        self.assertEqual(fit.denominator, 13.)
        self.assertAlmostEqual(fit.alpha, 16 / 13)
        self.assertIsNone(fit_nonnegative_gain(signal * 0, target, [1, 3]).alpha)
        self.assertEqual(fit_nonnegative_gain(-signal, target, [1, 3]).alpha, 0.)

    def test_bias_free_gain_anchor_reconstructs_a_biased_dense_output(self):
        weight = torch.tensor([[1.], [1.]], dtype=torch.bfloat16)
        bias = torch.tensor([10.], dtype=torch.bfloat16)
        a = torch.ones(1, 2, dtype=torch.bfloat16)
        gate = torch.ones(1, 1, dtype=torch.bfloat16)
        projection = NativeProjection(weight, bias)
        kept, full = np.array([[1., 0.]]), np.ones((1, 2))
        signal = projection.fp32_signal(a, gate, kept).numpy()
        dense_signal = projection.fp32_signal(a, gate, full).numpy()
        fit = fit_nonnegative_gain(signal, dense_signal, [1])
        dense = projection.native(a, gate, full, 1.)
        result = projection.native(a, gate, kept, fit.alpha)
        self.assertTrue(torch.equal(result, dense))
        self.assertEqual(fit.alpha, 2.)
        # This is the old biased-anchor mistake: bias is fitted as signal but
        # still added independently by F.linear, yielding 22 instead of 12.
        biased_fit = fit_nonnegative_gain(signal, dense.float().numpy(), [1])
        wrong = projection.native(a, gate, kept, biased_fit.alpha)
        self.assertEqual(float(wrong[0, 0]), 22.)
        self.assertEqual(float(dense[0, 0]), 12.)

    def test_weighted_inverse_cdf_is_not_unweighted_median(self):
        np.testing.assert_array_equal(weighted_quantile([1, 2, 100], [1, 1, 10], [0, .5, 1]), [1, 100, 100])

    def test_duplicate_support_and_nonfinite_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            support_mask(np.array([1, 1]), rows=2, groups=4, group_size=2)
        with self.assertRaises(ValueError):
            top_groups(np.array([1., np.nan]), 1)


if __name__ == "__main__":
    unittest.main()
