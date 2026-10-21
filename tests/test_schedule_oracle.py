from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from schedule_oracle import enumerate_assignments


class ScheduleOracleTests(unittest.TestCase):
    def test_one_arm_per_whole_source_prevents_metricwise_fake_pass(self):
        dense = np.ones((1, 3))
        first = np.array([[1., .94, 1.]])
        second = np.array([[.94, 1., 1.]])
        result = enumerate_assignments(dense, first, second)
        self.assertFalse(result['oracle_joint_gate'])
        self.assertEqual(result['joint_feasible_assignments'], 0)
        self.assertAlmostEqual(result['minimax_relative_drop'], .06)

    def test_blocked_enumeration_matches_direct_bruteforce(self):
        rng = np.random.default_rng(17)
        dense = rng.uniform(.4, 1., size=(6, 3))
        first, second = [dense * rng.uniform(.90, 1.10, size=dense.shape) for _ in range(2)]
        risks = []
        for mask in range(1 << len(dense)):
            chosen = np.array([(mask >> i) & 1 for i in range(len(dense))], bool)
            risks.append(np.max(1 - np.where(chosen[:, None], second, first).mean(0) / dense.mean(0)))
        result = enumerate_assignments(dense, first, second, block_size=7)
        self.assertEqual(result['best_mask'], int(np.argmin(risks)))
        self.assertAlmostEqual(result['minimax_relative_drop'], min(risks))
        self.assertEqual(result['joint_feasible_assignments'], sum(v <= .02 for v in risks))

    def test_wrong_shapes_or_nonfinite_scores_are_rejected(self):
        dense = np.ones((2, 3))
        for other in [np.ones((2, 2)), np.full((2, 3), np.nan)]:
            with self.assertRaises(ValueError):
                enumerate_assignments(dense, other, dense)

    def test_exact_two_percent_boundary_uses_original_score_threshold(self):
        dense = np.ones((1, 3))
        result = enumerate_assignments(dense, np.full((1, 3), .98), np.full((1, 3), .97))
        self.assertTrue(result['oracle_joint_gate'])
        self.assertEqual(result['joint_feasible_assignments'], 1)


if __name__ == '__main__':
    unittest.main()
