"""Proxy labels are source-fitted, and zero-energy/degenerate banks are rejected."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from router_bank_feasibility import (DegenerateBanks, bank_energy, describe_proxy, fit_banks,
                                     folded_parameters, native_predict_rows, normalized_profiles)


class RouterBankTests(unittest.TestCase):
    def profiles(self):
        values = np.zeros((16, 216))
        values[np.arange(16), np.arange(16) % 4] = np.arange(16) + 1.
        return normalized_profiles(values)

    def test_row_normalization_and_seeded_cluster_identity(self):
        x = self.profiles()
        np.testing.assert_array_equal(x, normalized_profiles(x * np.arange(1, 17)[:, None]))
        a, b = [fit_banks(x, 1, 20261021, min_bank=1) for _ in range(2)]
        np.testing.assert_array_equal(a['masks'], b['masks'])
        self.assertEqual(a['counts'].tolist(), [4] * 4)

    def test_oracle_gain_is_not_assigned_to_a_failed_router(self):
        x = self.profiles()
        banks = fit_banks(x, 1, 20261021, min_bank=1)
        oracle = bank_energy(x, banks['masks']).argmax(1)
        good = describe_proxy(x, banks, oracle)
        bad = describe_proxy(x, banks, np.zeros(len(x), dtype=np.int64))
        self.assertEqual(good['static_retained_C_fraction'], .25)
        self.assertEqual(good['oracle_retained_C_fraction'], 1.)
        self.assertEqual(good['native_fraction_of_oracle_gain'], 1.)
        self.assertTrue(good['gate_pass'])
        self.assertFalse(bad['gate_pass'])

    def test_zero_energy_and_collapsed_banks_fail_without_reseeding(self):
        with self.assertRaises(ValueError):
            normalized_profiles(np.zeros((16, 216)))
        with self.assertRaises(DegenerateBanks):
            fit_banks(np.full((16, 216), 1 / 216), 108, 1, min_bank=1)

    def test_native_row_mapping_and_standardization_folding(self):
        model = torch.nn.Sequential(torch.nn.Linear(5120, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
        with torch.no_grad():
            for value in model.parameters(): value.zero_()
            model[0].weight[0, 0] = 2.
            model[0].bias[0] = 3.
            model[2].weight[0, 0] = 4.
            model[2].bias[0] = -1.
        mean, scale = torch.zeros(5120), torch.ones(5120)
        mean[0], scale[0] = 6., 2.
        params = folded_parameters(model, mean, scale)
        self.assertEqual(float(params['w1'][0, 0]), 1.)
        self.assertEqual(float(params['b1'][0]), -3.)
        x = torch.zeros((2, 5120), dtype=torch.bfloat16)
        x[:, 0] = torch.tensor([8., -10.], dtype=torch.bfloat16)
        case = {'V1': x, 'scope': {'native_V1_shape': [1, 8, 5120],
                                  'native_V1_stride': [40960, 5120, 1], 'full_rows': 8, 'selected_rows': [1, 6]}}
        np.testing.assert_array_equal(native_predict_rows(case, params), [0, 1])


if __name__ == '__main__':
    unittest.main()
