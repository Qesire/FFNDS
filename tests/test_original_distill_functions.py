from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from original_distill_functions import paired_error,error_alignment,normalized_concentration


class OriginalDistillFunctionTests(unittest.TestCase):
    def test_error_reference_units_and_signed_alignment(self):
        original = torch.tensor([[1.,3.]])
        distilled = torch.tensor([[2.,3.]])
        quantized = torch.tensor([[3.,3.]])
        record = paired_error(distilled,original)
        self.assertEqual(record['squared_error'],1.)
        self.assertEqual(record['reference_squared_norm'],10.)
        self.assertAlmostEqual(record['relative_RMSE'],.1**.5)
        self.assertEqual(error_alignment(distilled,original,quantized),1.)
        self.assertEqual(error_alignment(distilled,original,original),-1.)
        self.assertIsNone(error_alignment(original,original,quantized))
        with self.assertRaises(ValueError):
            paired_error(quantized,torch.zeros_like(quantized))

    def test_uniform_and_concentrated_group_limits_without_epsilon_bias(self):
        uniform = normalized_concentration(torch.ones(2,216))
        self.assertAlmostEqual(uniform['mean_row_oracle_top108_retained_C'],.5)
        self.assertAlmostEqual(uniform['mean_row_normalized_C_entropy'],1.)
        scores = torch.zeros(2,216)
        scores[:,17] = 1.
        concentrated = normalized_concentration(scores)
        self.assertEqual(concentrated['mean_row_oracle_top108_retained_C'],1.)
        self.assertEqual(concentrated['mean_row_normalized_C_entropy'],0.)
        with self.assertRaises(ValueError):
            normalized_concentration(torch.zeros(2,216))


if __name__ == '__main__':
    unittest.main()
