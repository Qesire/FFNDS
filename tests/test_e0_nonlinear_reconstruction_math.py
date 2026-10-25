import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from e0_nonlinear_reconstruction_math import dual_correction


class NonlinearReconstructionMath(unittest.TestCase):
    def test_selected_nonlinear_residual_folds_into_two_paths(self):
        generator = torch.Generator().manual_seed(31)
        def sample(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float64)
        x, w1, w2 = sample(7, 5), sample(11, 5), sample(4, 11)
        b1, b2, slope, offset = sample(11), sample(4), sample(11), sample(11)
        ids = torch.tensor([1, 3, 5]); v = sample(4, 3)
        fmean, error_mean = sample(3), sample(4)
        a = x @ w1.T + b1
        q = torch.nn.functional.gelu(a, approximate='tanh')
        r = q[:, ids] - a[:, ids] * slope[ids] - offset[ids]
        target = (a * slope + offset) @ w2.T + b2 + (r - fmean) @ v.T + error_mean
        matrix = (w2 * slope) @ w1 - (v * slope[ids]) @ w1[ids]
        bias = w2 @ (slope * b1 + offset) + b2 - v @ (slope[ids] * b1[ids] + offset[ids] + fmean) + error_mean
        torch.testing.assert_close(target, x @ matrix.T + q[:, ids] @ v.T + bias)

    def test_dual_fit_equals_absorbed_ridge_and_bias(self):
        generator = torch.Generator().manual_seed(29)
        features = torch.randn(9, 13, generator=generator, dtype=torch.float64)
        targets = torch.randn(9, 4, generator=generator, dtype=torch.float64)
        held = torch.randn(5, 13, generator=generator, dtype=torch.float64)
        fm, ym = features.mean(0), targets.mean(0)
        f, y = features - fm, targets - ym
        dual, _, penalty = dual_correction(f, y, .01, torch)
        primal = torch.linalg.solve(f.T @ f + penalty * torch.eye(13), f.T @ y)
        torch.testing.assert_close((held - fm) @ f.T @ dual + ym, held @ primal + ym - fm @ primal)
        self.assertLess(float(((f @ f.T @ dual) - y).square().sum()), float(y.square().sum()))


if __name__ == '__main__':
    unittest.main()
