"""Analytic, non-model checks for metric-feedback semantics."""

import unittest

import numpy as np

from tools.video_metric_feedback.optimizer import Evaluation, OptimizerConfig, optimize


def evaluation(loss, margins=None, fingerprint=None):
    return Evaluation(float(loss), {"source-a": float(loss), "source-b": float(loss) + 0.1}, margins or {}, fingerprint)


def iteration(result):
    return [event for event in result.history if event["type"] == "iteration"][-1]


class OptimizerTests(unittest.TestCase):
    def test_quadratic_step_budget_and_trust_region(self):
        seen = []

        def evaluate(theta):
            seen.append(theta.copy())
            return evaluation(0.5 * np.sum((theta - np.array([0.2, 0.8])) ** 2))

        result = optimize([1.0, 1.0], evaluate, OptimizerConfig([0.0, 0.0], [2.0, 2.0], trust_radius=0.3, damping=0.1))
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.evaluations_used, 6)
        self.assertEqual(len(seen), 6)
        self.assertLess(result.evaluation.loss, 0.34)
        np.testing.assert_allclose(iteration(result)["gradient"], [0.8, 0.2], atol=1e-12)
        np.testing.assert_allclose(iteration(result)["curvature"], [1.0, 1.0], atol=1e-12)
        self.assertTrue(np.all(np.abs(np.asarray(result.theta) - 1.0) <= 0.3))

    def test_unequal_three_point_stencil_at_near_boundary(self):
        result = optimize([0.95], lambda x: evaluation(x[0] ** 2), OptimizerConfig([0.0], [1.0], finite_difference_step=0.2, trust_radius=0.3, max_evaluations=4))
        record = iteration(result)
        self.assertAlmostEqual(record["gradient"][0], 1.9, places=12)
        self.assertAlmostEqual(record["curvature"][0], 2.0, places=12)
        self.assertAlmostEqual(record["probe_diagnostics"][0]["plus_step"], 0.05)
        self.assertAlmostEqual(record["probe_diagnostics"][0]["minus_step"], 0.2)

    def test_one_sided_probe_reuses_boundary_center(self):
        calls = []

        def evaluate(theta):
            calls.append(float(theta[0]))
            return evaluation((theta[0] - 0.5) ** 2)

        result = optimize([0.0], evaluate, OptimizerConfig([0.0], [1.0], max_evaluations=3))
        self.assertEqual(iteration(result)["probe_diagnostics"][0]["stencil"], "one_sided")
        self.assertEqual(calls.count(0.0), 1)
        self.assertLessEqual(result.evaluations_used, 3)
        self.assertEqual(result.status, "accepted")

    def test_linear_guard_changes_the_update_direction(self):
        def evaluate(theta):
            return evaluation(0.5 * np.sum(theta * theta), {"preserve_motion": float(np.sum(theta) - 1.8)})

        result = optimize([1.0, 1.0], evaluate, OptimizerConfig([0.0, 0.0], [2.0, 2.0], trust_radius=0.4))
        self.assertEqual(result.status, "accepted")
        self.assertGreaterEqual(result.evaluation.guard_margins["preserve_motion"], 0.0)
        np.testing.assert_allclose(result.theta, [0.9, 0.9], atol=1e-7)

    def test_actual_nonlinear_guard_can_reject_predicted_feasible_candidate(self):
        def evaluate(theta):
            return evaluation(theta[0] ** 2, {"motion": 0.01 - (theta[0] - 1.0) ** 2})

        result = optimize([1.0], evaluate, OptimizerConfig([0.0], [2.0], max_evaluations=4))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.theta, [1.0])
        self.assertFalse(iteration(result)["candidate_feasible"])

    def test_positive_actual_gain_below_model_ratio_is_rejected(self):
        result = optimize([0.0], lambda x: evaluation(-x[0] + 0.9 * x[0] ** 4), OptimizerConfig([-2.0], [2.0], trust_radius=1.0, damping=0.982, max_evaluations=4, acceptance_ratio=0.5))
        record = iteration(result)
        self.assertGreater(record["metric_reduction"], 0.0)
        self.assertLess(record["agreement_ratio"], 0.5)
        self.assertEqual(result.status, "rejected")

    def test_minimum_improvement_is_checked_on_measured_loss(self):
        result = optimize([1.0], lambda x: evaluation(x[0] ** 2), OptimizerConfig([0.0], [2.0], max_evaluations=4, min_improvement=1.0))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.theta, [1.0])

    def test_budget_reserves_complete_stencil_and_candidate(self):
        result = optimize([1.0, 1.0], lambda x: evaluation(np.sum(x * x)), OptimizerConfig([0.0, 0.0], [2.0, 2.0], max_evaluations=5))
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.evaluations_used, 1)

    def test_flat_metrics_and_equal_execution_fingerprints_are_separate(self):
        config = OptimizerConfig([0.0], [2.0], max_evaluations=4)
        result = optimize([1.0], lambda x: evaluation(2.0, fingerprint="same-bf16-matrix"), config)
        diagnostic = iteration(result)["probe_diagnostics"][0]
        self.assertTrue(diagnostic["flat_metric"])
        self.assertTrue(diagnostic["same_execution_fingerprint"])
        self.assertEqual(result.status, "no_proposal")
        result = optimize([1.0], lambda x: evaluation(2.0), config)
        diagnostic = iteration(result)["probe_diagnostics"][0]
        self.assertTrue(diagnostic["flat_metric"])
        self.assertFalse(diagnostic["same_execution_fingerprint"])

    def test_cohort_and_guard_identity_cannot_change(self):
        calls = 0

        def evaluate(theta):
            nonlocal calls
            calls += 1
            return Evaluation(1.0, {"source-a" if calls == 1 else "source-c": 1.0}, {})

        with self.assertRaisesRegex(ValueError, "remain identical"):
            optimize([1.0], evaluate, OptimizerConfig([0.0], [2.0], max_evaluations=4))

    def test_metric_noise_on_identical_executed_parameters_cannot_be_accepted(self):
        calls = 0

        def evaluate(theta):
            nonlocal calls
            calls += 1
            return evaluation(1.0 - 0.1 * calls, fingerprint="identical-execution")

        result = optimize([1.0], evaluate, OptimizerConfig([0.0], [2.0], max_evaluations=4))
        self.assertEqual(result.status, "rejected")
        self.assertTrue(iteration(result)["same_execution_as_center"])
        self.assertTrue(iteration(result)["probe_diagnostics"][0]["repeated_execution_score_disagreement"])
        self.assertEqual(result.theta, [1.0])

    def test_invalid_score_fails_before_a_parameter_update(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            optimize([1.0], lambda x: evaluation(float("nan")), OptimizerConfig([0.0], [2.0]))

    def test_initial_guard_violation_does_not_start_probes(self):
        result = optimize([1.0], lambda x: evaluation(1.0, {"motion": -0.1}), OptimizerConfig([0.0], [2.0]))
        self.assertEqual(result.status, "initial_guard_violation")
        self.assertEqual(result.evaluations_used, 1)

    def test_uncertified_projection_does_not_render_a_candidate(self):
        result = optimize([1.0, 1.0], lambda x: evaluation(np.sum(x * x)), OptimizerConfig([0.0, 0.0], [2.0, 2.0], projection_max_iterations=1))
        self.assertEqual(result.status, "no_proposal")
        self.assertEqual(iteration(result)["reason"], "projection_not_certified")
        self.assertEqual(result.evaluations_used, 5)

    def test_prior_reference_is_initial_and_limits_step(self):
        plain = optimize([1.0], lambda x: evaluation(x[0] ** 2), OptimizerConfig([0.0], [2.0], max_evaluations=4, trust_radius=1.0))
        anchored = optimize([1.0], lambda x: evaluation(x[0] ** 2), OptimizerConfig([0.0], [2.0], max_evaluations=4, trust_radius=1.0, prior_strength=10.0))
        self.assertGreater(anchored.theta[0], plain.theta[0])
        self.assertEqual(anchored.status, "accepted")

    def test_probe_distance_respects_a_smaller_trust_region(self):
        result = optimize([1.0], lambda x: evaluation(x[0] ** 2), OptimizerConfig([0.0], [2.0], finite_difference_step=0.4, trust_radius=0.05, max_evaluations=4))
        diagnostic = iteration(result)["probe_diagnostics"][0]
        self.assertAlmostEqual(diagnostic["minus_step"], 0.05)
        self.assertAlmostEqual(diagnostic["plus_step"], 0.05)
        self.assertLessEqual(abs(result.theta[0] - 1.0), 0.05)

    def test_rejection_shrinks_region_and_reuses_identical_probes(self):
        def evaluate(theta):
            return evaluation(theta[0] ** 2, {"motion": 0.04 - (theta[0] - 1.0) ** 2})

        result = optimize([1.0], evaluate, OptimizerConfig([0.0], [2.0], max_iterations=2, max_evaluations=5))
        records = [event for event in result.history if event["type"] == "iteration"]
        self.assertEqual([record["status"] for record in records], ["rejected", "accepted"])
        self.assertEqual(result.evaluations_used, 5)
        self.assertEqual(result.accepted_steps, 1)
        self.assertAlmostEqual(result.theta[0], 0.875, places=8)


if __name__ == "__main__":
    unittest.main()
