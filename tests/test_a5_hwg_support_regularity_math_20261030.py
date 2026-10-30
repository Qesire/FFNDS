"""Small mathematical counterexamples, not scientific tensor experiments."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('a5_regularity', Path(__file__).resolve().parents[1] / 'tools/a5_hwg_support_regularity_math_20261030.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RegularityFixtures(unittest.TestCase):
    def test_logical_sparsity_does_not_imply_physical_tile_sparsity(self):
        mask = torch.zeros(16, 128, dtype=torch.bool)
        mask[:8, 0] = True
        mask[8:, 64] = True
        result = m.tile_occupancy(mask)
        self.assertEqual(result['nonempty_tiles'], 2)
        self.assertEqual(result['occupied_fraction'], 1.)
        self.assertEqual(result['real_selected_elements'], 16)

    def test_whole64_shared_mask_really_removes_one_tile(self):
        mask, _ = m.shared_support(torch.ones(16, 128), 64)
        self.assertEqual(m.tile_occupancy(mask)['all_zero_tiles'], 1)
        self.assertTrue(mask[:, :64].all())
        self.assertFalse(mask[:, 64:].any())

    def test_group_selection_sums_energy_before_topk(self):
        score = torch.tensor([[2., 0., 1.5, 1.5], [2., 0., 1.5, 1.5]])
        mask, _ = m.shared_support(score, 2, hidden_group=2)
        self.assertTrue(mask[:, 2:].all())

    def test_shuffle_changes_membership_not_original_coordinates(self):
        score = torch.tensor([[9., 0.], [8., 0.], [0., 7.], [0., 6.]])
        mask, _ = m.shared_support(score, 1, hidden_group=1, row_group=2)
        mixed, _ = m.shared_support(score, 1, hidden_group=1, row_group=2,
                                    permutation=torch.tensor([0, 2, 1, 3]))
        self.assertTrue(torch.equal(mask, score > 0))
        self.assertTrue(mixed[:, 0].all())

    def test_duplicate_shuffle_and_nonwhole_K_fail(self):
        with self.assertRaises(ValueError):
            m.shared_support(torch.ones(2, 128), 63)
        with self.assertRaises(ValueError):
            m.shared_support(torch.ones(2, 128), 64, permutation=torch.tensor([0, 0]))

    def test_tail_padding_is_charged_in_physical_MAC(self):
        result = m.tile_occupancy(torch.ones(17, 64, dtype=torch.bool), m_tile=16)
        self.assertEqual(result['padded_M'], 32)
        self.assertAlmostEqual(result['physical_tile_MAC_fraction_with_M_padding'], 32 / 17)

    def test_signed_cancellation_defeats_diagonal_energy_ranking(self):
        d = torch.tensor([[3., -3., 1.]])
        w = torch.ones(1, 3)
        old = torch.zeros(1, 1)
        current = torch.ones(1, 1)
        diagonal = m.signed_output_metrics(old, current, d, w, torch.ones(1), torch.tensor([[True, False, False]]))
        weak = m.signed_output_metrics(old, current, d, w, torch.ones(1), torch.tensor([[False, False, True]]))
        self.assertGreater(diagonal['captured_postQ2_delta_energy_fraction'], weak['captured_postQ2_delta_energy_fraction'])
        self.assertGreater(diagonal['gated_current_relative_L2'], weak['gated_current_relative_L2'])

    def test_full_delta_rounding_difference_can_cancel_omission(self):
        d = torch.tensor([[1., .25]])
        w = torch.ones(1, 2)
        result = m.signed_output_metrics(torch.zeros(1, 1), torch.ones(1, 1), d, w, torch.ones(1), torch.tensor([[True, False]]))
        self.assertEqual(result['gated_current_relative_L2'], 0.)
        self.assertGreater(result['full_delta_rounding_difference_relative_to_current'], 0.)
        self.assertLess(result['rounding_omission_signed_cross'], 0.)
        self.assertAlmostEqual(result['signed_error_decomposition_residual'], 0.)

    def test_four_step_first_encounter_and_full_proxy_cost(self):
        core = m.four_step_cost_identity(5120)
        dense_proxy = m.four_step_cost_identity(5120, proxy_full_w1_calls=4)
        self.assertAlmostEqual(core['four_step_selected_main_MAC_fraction'], .6851851851851851)
        self.assertGreater(dense_proxy['core_plus_proxy_MAC_fraction_excluding_other_costs'], 1.)

    def test_zero_directions_and_denominators_are_explicitly_undefined(self):
        result = m.signed_output_metrics(torch.zeros(1, 1), torch.zeros(1, 1),
                 torch.zeros(1, 2), torch.ones(1, 2), torch.ones(1),
                 torch.zeros(1, 2, dtype=torch.bool))
        for key in ['gated_current_relative_L2', 'gated_change_relative_L2',
                    'gated_change_direction_cosine', 'captured_postQ2_delta_energy_fraction']:
            self.assertIsNone(result[key])
        self.assertTrue(result['zero_actual_gated_change'])
        self.assertEqual(result['squared_error'], 0.)

    def test_real_neighbor_counts_exclude_flattened_boundaries(self):
        result = m.support_summary(torch.ones(12, 64, dtype=torch.bool), grid=[2, 2, 3])
        self.assertEqual(result['horizontal_real_neighbor_pairs'], 8)
        self.assertEqual(result['vertical_real_neighbor_pairs'], 6)
        self.assertEqual(result['latent_tau_real_neighbor_pairs'], 6)


if __name__ == '__main__': unittest.main()
