import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from reused_hidden_readout import readout_features, training_moments


class ReusedHiddenReadoutTests(unittest.TestCase):
    def test_constant_training_channels_remain_finite_on_distinct_validation_rows(self):
        train = torch.full((8, 256), 2., dtype=torch.bfloat16)
        moments = training_moments(train)
        self.assertTrue(torch.equal(readout_features(train, moments, nonlinear=False), torch.zeros_like(train)))
        validation = torch.full((3, 256), 2.5, dtype=torch.bfloat16)
        for nonlinear in [False, True]:
            result = readout_features(validation, moments, nonlinear=nonlinear)
            self.assertEqual(result.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(torch.all(result > 0))

    def test_validation_batch_does_not_change_fitted_moments_or_other_rows(self):
        train = (torch.arange(16 * 256).reshape(16, 256).float() / 128).bfloat16()
        moments = training_moments(train)
        original = {k: v.clone() for k, v in moments.items()}
        row = train[:1] * 2
        mixed = torch.cat([row, torch.full_like(row, -200)])
        for nonlinear in [False, True]:
            alone = readout_features(row, moments, nonlinear=nonlinear)
            together = readout_features(mixed, moments, nonlinear=nonlinear)
            self.assertTrue(torch.equal(alone, together[:1]))
        self.assertTrue(all(torch.equal(moments[k], original[k]) for k in moments))

    def test_invalid_feature_or_moment_identity_is_rejected(self):
        train = torch.zeros((4, 256), dtype=torch.bfloat16)
        moments = training_moments(train)
        for hidden, parameters in [(train.float(), moments),
                (train[:, :128], moments), (train, {**moments, 'scale': torch.zeros(256)}),
                (train, {**moments, 'mean': torch.full((256,), float('nan'))})]:
            with self.assertRaises(ValueError):
                readout_features(hidden, parameters, nonlinear=True)


if __name__ == '__main__':
    unittest.main()
