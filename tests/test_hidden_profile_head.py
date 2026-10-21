import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from hidden_profile_head import fit_head, predict_head, retained_fraction, support


class HiddenProfileHeadTests(unittest.TestCase):
    def test_diagonal_square_predicts_symmetric_signal_linear_cannot(self):
        hidden = torch.tensor([[-2.], [-1.], [1.], [2.]]).repeat(8, 1)
        target = hidden.square()
        linear = predict_head(hidden, fit_head(hidden, target, False), False)
        square = predict_head(hidden, fit_head(hidden, target, True), True)
        self.assertLess(float((square-target).square().sum()), float((linear-target).square().sum()) * .3)
        self.assertTrue(torch.equal(linear[0], linear[-1]))

    def test_budget_ties_and_row_oracle(self):
        profiles = torch.tensor([[.1, .2, .4, .3], [.5, .1, .1, .3]], dtype=torch.float64)
        oracle = support(profiles, 2)
        self.assertEqual(oracle.tolist(), [[2, 3], [0, 3]])
        self.assertAlmostEqual(retained_fraction(profiles, oracle), .75)
        self.assertEqual(support(torch.ones(2, 4), 2).tolist(), [[0, 1], [0, 1]])


if __name__ == '__main__':
    unittest.main()
