"""Equal-size bank assignment must keep its quota even under extreme imbalance."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from balanced_router_banks import balanced_assignments, fit_balanced_banks


class BalancedRouterBankTests(unittest.TestCase):
    def test_equal_quota_and_stable_ties(self):
        distances = np.zeros((16, 4))
        labels = balanced_assignments(distances)
        np.testing.assert_array_equal(labels, np.repeat(np.arange(4), 4))
        distances[:, 1:] = 100.
        np.testing.assert_array_equal(np.bincount(balanced_assignments(distances), minlength=4), [4] * 4)

    def test_imbalanced_profiles_reproduce_without_reseeding(self):
        profiles = np.zeros((16, 216))
        profiles[:13, 0] = 1.
        profiles[13:, 1:4] = np.eye(3)
        first, second = [fit_balanced_banks(profiles, 108, 20261021, min_bank=1) for _ in range(2)]
        np.testing.assert_array_equal(first['counts'], [4] * 4)
        np.testing.assert_array_equal(first['masks'], second['masks'])
        self.assertEqual(first['masks'].shape, (4, 108))


if __name__ == '__main__':
    unittest.main()
