import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_e0_shared_reference import moments, subspaces


class SharedReferenceIdentities(unittest.TestCase):
    def test_uncorrelated_unequal_arm_errors_are_exact_star_distances(self):
        values = np.diag(np.sqrt([1.0, 2.0, 3.0, 4.0]))
        row = moments(list(values))
        self.assertLess(row["star_null_fit"]["relative_L2_discrepancy_of_six_squared_distances"], 1e-14)
        self.assertEqual(row["all_three_reference_cross_moments"], [1.0, 1.0, 1.0])

    def test_shared_reference_does_not_make_additive_actions_nonadditive(self):
        values = [np.array(v, dtype=float) for v in [(3, 4), (4, 4), (3, 6), (4, 6)]]
        row = moments(values)
        self.assertEqual(row["factorial_interaction_energy"], 0.0)
        self.assertGreater(row["star_null_fit"]["relative_L2_discrepancy_of_six_squared_distances"], .1)

    def test_global_temporal_spatial_subspaces_are_orthogonal(self):
        value = np.arange(120, dtype=float).reshape(1, 2, 3, 4, 5)
        parts = subspaces(value)
        names = [n for n in parts if n != "full"]
        np.testing.assert_allclose(sum(parts[n] for n in names), value)
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                self.assertAlmostEqual(float(np.sum(parts[name] * parts[other])), 0.0)
        native = subspaces(value[0])
        for name in names:
            np.testing.assert_array_equal(native[name], parts[name])


if __name__ == "__main__":
    unittest.main()
