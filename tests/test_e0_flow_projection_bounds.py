"""Projection-family counterexamples; no native models or GPU execution."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_e0_whole_flow_results import projection_bounds


def shaped(values):
    return torch.tensor(values, dtype=torch.float64).reshape(1, 1, 1, -1)


class ProjectionBoundsTest(unittest.TestCase):
    def test_target_orthogonal_to_all_fixed_online_directions_remains(self):
        z = shaped([1, -1, 1, -1])
        flow = shaped([1, 1, -1, -1])
        target = shaped([1, -1, -1, 1])
        values = {'z_dense': z * 0, 'z_student': z,
                  'v_student': flow, 'v_dense': target - z + flow}
        out = projection_bounds(values, 1)
        self.assertEqual(out['centered_basis_numerical_rank'], 2)
        for bound in out['bounds'].values():
            self.assertAlmostEqual(bound['ideal_next_SSE_oracle_residual_estimate'], 4)
            self.assertAlmostEqual(bound['fraction_of_uncorrected_ideal_next_SSE'], 1)

    def test_dependent_flow_and_latent_basis_uses_one_direction(self):
        z = shaped([1, -1, 1, -1])
        values = {'z_dense': z * 0, 'z_student': z, 'v_student': z * 2, 'v_dense': z * 2}
        out = projection_bounds(values, .5)
        self.assertEqual(out['centered_basis_numerical_rank'], 1)
        self.assertAlmostEqual(out['bounds']['channel_bias_and_flow_and_latent_scale']['ideal_next_SSE_oracle_residual_estimate'], 0)

    def test_channel_bias_handles_constant_modes_with_zero_centered_rank(self):
        z = torch.tensor([1., 2.], dtype=torch.float64).reshape(2, 1, 1, 1).expand(2, 1, 1, 4)
        values = {'z_dense': z * 0, 'z_student': z, 'v_student': z * 0, 'v_dense': z * 0}
        out = projection_bounds(values, 1)
        self.assertEqual(out['centered_basis_numerical_rank'], 0)
        self.assertEqual(out['channel_bias_rank'], 2)
        self.assertAlmostEqual(out['bounds']['channel_bias_only']['ideal_next_SSE_oracle_residual_estimate'], 0)
        self.assertEqual(out['bounds']['flow_scale_only']['fraction_of_uncorrected_ideal_next_SSE'], 1)

    def test_tiny_decisive_direction_is_not_dropped_by_raw_gram_scale(self):
        u = shaped([1, -1, 0, 0])
        target = shaped([1, 1, -1, -1])
        b = target * 1e-7
        values = {'z_dense': b - u - target, 'z_student': b,
                  'v_student': u, 'v_dense': u * 0}
        out = projection_bounds(values, 1)
        self.assertAlmostEqual(out['bounds']['channel_bias_and_flow_and_latent_scale']['ideal_next_SSE_oracle_residual_estimate'], 0)
        self.assertFalse(out['strict_family_exclusion_allowed'])


if __name__ == '__main__':
    unittest.main()
