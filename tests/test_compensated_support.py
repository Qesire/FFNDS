from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from compensated_support import FixedDesign, discrepancy_certificate, finite_swaps, linear_upper_bound, proposal_pool


class CompensatedSupportTests(unittest.TestCase):
    def test_design_span_change_has_zero_matched_support_value(self):
        h = torch.arange(8, dtype=torch.float64).reshape(-1, 1)
        design = FixedDesign(h)
        change = 3 * h + 2
        self.assertLess(float(design.residual(change).abs().max()), 1e-12)

    def test_validation_delta_uses_training_fit_instead_of_validation_projection(self):
        design = FixedDesign(torch.tensor([[1.], [2.]]))
        coefficient = design.fit(torch.tensor([[2.], [4.]]))
        delta = 7 - FixedDesign.features(torch.tensor([[3.]])) @ coefficient
        self.assertAlmostEqual(float(delta), 1.)

    def test_rank_deficient_design_has_finite_minimum_norm_fit(self):
        h = torch.ones((12, 3), dtype=torch.float64)
        design = FixedDesign(h)
        coefficient = design.fit(torch.full((12, 2), 4.))
        self.assertEqual(design.rank, 1)
        self.assertTrue(torch.isfinite(coefficient).all())
        self.assertLess(float((design.phi @ coefficient - 4).abs().max()), 1e-12)

    def test_joint_swap_can_win_when_each_single_swap_loses(self):
        contributions = np.array([[0., 0.], [0., 0.], [-.75, 1.], [-.75, -1.]])
        linear = -2 * contributions @ np.array([1., 0.])
        result = finite_swaps([0, 1], [2, 3], linear, contributions @ contributions.T)
        self.assertEqual(result['best']['swap_count'], 2)
        self.assertAlmostEqual(result['best']['proxy_advantage'], .75)
        self.assertTrue(all(v['proxy_advantage'] < 0 for v in result['all_actions'] if v['swap_count'] == 1))

    def test_linear_relaxation_bounds_all_finite_joint_swaps(self):
        rng = np.random.default_rng(11)
        contributions = rng.normal(size=(8, 7))
        linear = -2 * contributions @ rng.normal(size=7)
        removed, added = proposal_pool([0, 1, 2, 3], linear, width=4)
        result = finite_swaps(removed, added, linear, contributions @ contributions.T)
        self.assertLessEqual(result['best']['proxy_advantage'], linear_upper_bound([0, 1, 2, 3], linear) + 1e-10)

    def test_numeric_discrepancy_certificate_covers_changed_native_output(self):
        e, proxy = torch.tensor([[1., -.5]]), torch.tensor([[.2, -.7]])
        result = discrepancy_certificate(e, proxy, proxy + torch.tensor([[.01, -.03]]))
        self.assertLessEqual(result['absolute_advantage_discrepancy'], result['bound'] + 1e-12)


if __name__ == '__main__':
    unittest.main()
