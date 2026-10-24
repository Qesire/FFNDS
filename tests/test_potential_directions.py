"""Decisive mathematical cases for the new research construction."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_potential_directions import (
    bf16, build, equivalent, folded_head_fixture, log_target_counterexample,
    projection_cost, read, OUT, signed_advantage_fixture, staged_fixture, toy_q,
    no_completion_fixture,
)


class PotentialDirections(unittest.TestCase):
    def test_no_completion_or_correction_is_mathematically_necessary(self):
        r = no_completion_fixture()
        self.assertEqual(r['dynamic_oracle_errors'], [0., 0.])
        self.assertGreater(r['best_fixed_mean_error'], 0)
        self.assertTrue(all(v > 0 for v in r['independent_magnitude_errors']))
        self.assertEqual(r['dense_completion_rows'], 0)
        self.assertEqual(r['additive_correction_branches'], 0)

    def test_single_exchange_failure_does_not_prove_joint_impossibility(self):
        r = no_completion_fixture()
        self.assertTrue(all(v > r['base_error'] for v in r['all_single_exchange_errors']))
        self.assertLess(r['joint_exchange_error'], r['base_error'])

    def test_expected_loss_and_log_loss_can_reverse_decisions(self):
        r = log_target_counterexample()
        self.assertGreater(r['arithmetic_means_A_B'][0], r['arithmetic_means_A_B'][1])
        self.assertLess(r['geometric_means_A_B'][0], r['geometric_means_A_B'][1])
        self.assertAlmostEqual(r['expected_capture_regret'], 3.05)

    def test_large_response_can_harm_and_joint_value_is_not_additive(self):
        r = signed_advantage_fixture()
        self.assertEqual(r['signed_advantage'], -3)
        self.assertEqual(r['signed_advantage'], r['identity_value'])
        self.assertNotEqual(r['interacting_actions_sum_individual_advantages'],
                            r['interacting_actions_joint_advantage'])

    def test_staged_computation_keeps_both_endpoints_and_mixed_rows(self):
        for promoted in ((), (1, 4), range(6)):
            r = staged_fixture(promoted)
            self.assertEqual(r['max_output_error_in_toy'], 0)
            self.assertAlmostEqual(r['projection_fraction'], r['expected_projection_fraction'])
            self.assertEqual(r['complement_hidden_computations'], 8*r['promoted_rows'])

    def test_partitioned_quantization_is_not_assumed_to_commute(self):
        self.assertNotEqual(toy_q([.3, 4.])[0], toy_q([.3])[0])

    def test_shared_prefix_saves_duplicate_base_projection(self):
        for b in (.25, .5, .75):
            q = .25
            shared = b/2 + q*(1-b)/2 + ((1-q)*b+q)/2
            self.assertAlmostEqual(shared, projection_cost(b, q))
            late = b+q*(1-b/2)
            self.assertAlmostEqual(late-shared, q*b/2)

    def test_rich_head_redundant_norm_can_fold_into_diagonal(self):
        self.assertLess(folded_head_fixture()['absolute_error'], 1e-10)

    def test_bf16_midpoint_rounding(self):
        self.assertEqual(bf16(1 + 1/256), 1.)
        self.assertEqual(bf16(1 + 3/256), 1 + 2/128)

    def test_receipts_reproduce_and_sources_remain_the_units(self):
        r = build()
        self.assertTrue(equivalent(r, read(OUT)))
        saved = r['saved_receipt_comparison']
        self.assertEqual(saved['sources'], 6)
        self.assertEqual(saved['positive_source_count'], 6)
        self.assertTrue(math.isclose(saved['rich_current_minus_C2_pp'], 5.1945250743974))


if __name__ == '__main__':
    unittest.main()
