import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from analyze_e0_joint_and_temporal import joint_description, temporal_description


class SavedGeometryControls(unittest.TestCase):
    def test_perfect_cosine_does_not_bound_reuse_error(self):
        current = np.array([[1., 2., -1., -2.], [-1., 2., 1., -2.]])
        r = temporal_description(2*current, current, np.ones((1,4)))
        self.assertAlmostEqual(r["current_gate_weighted_cosine"], 1.)
        self.assertAlmostEqual(r["relative_SSE_reuse"], 1.)
        self.assertAlmostEqual(r["oracle_one_scalar"], .5)
        self.assertAlmostEqual(r["relative_SSE_oracle_scalar"], 0.)

    def test_exact_joint_cancellation_beats_small_individual_directions(self):
        # Group0 and group1 cancel exactly; the two smallest independent norms do not.
        x = np.array([[1., -1., 2., -2.]])
        d = np.array([[[.2, .1, -.1, -.2], [-.2, -.1, .1, .2],
                       [.01, -.01, .01, -.01], [.02, -.02, -.02, .02],
                       [.03, .03, -.03, -.03]]])
        # Avoid undefined relative-to-zero diagnostic ratios while retaining strong cancellation.
        d[:,1,0] += 1e-4
        r = joint_description(x,d)
        pair = r["subsets"]["2"]
        self.assertLess(pair["gram_identity_max_relative_error"], 1e-8)
        self.assertEqual(pair["choices"]["individual_gated_energy_sum"]["chosen_group_changed_rows"], 1)
        self.assertGreater(pair["choices"]["individual_gated_energy_sum"]["relative_excess_target_loss"], 100)


if __name__ == "__main__":
    unittest.main()
