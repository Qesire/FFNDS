import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from analyze_counterfactual_actions import analyze


class CounterfactualTest(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((ROOT/"research/20261022/examples/counterfactual_demo.json").read_text())

    def test_oracle_is_not_static(self):
        r = analyze(self.p)
        self.assertEqual(r["best_fixed_action"], "A_216_108")
        self.assertAlmostEqual(r["best_fixed_mean_loss"], 290/3)
        self.assertAlmostEqual(r["oracle_mean_loss"], 275/3)
        self.assertAlmostEqual(r["oracle_gain_over_best_fixed"], 5)
        self.assertAlmostEqual(r["policy_regret_over_oracle"], 0)
        self.assertEqual(r["evidence_level"], "SYNTHETIC_TEST_ONLY")

    def test_single_winner_zero_headroom(self):
        p = copy.deepcopy(self.p)
        for row in p["rows"]:
            row["losses"] = {"UB1_162_162": 100, "A_216_108": 90, "B_108_216": 99}
        self.assertAlmostEqual(analyze(p)["oracle_gain_over_best_fixed"], 0)

    def test_invalid_budget(self):
        p = copy.deepcopy(self.p)
        p["action_budget_groups"]["B_108_216"] = 325
        with self.assertRaises(ValueError):
            analyze(p)

    def test_missing_action(self):
        p = copy.deepcopy(self.p)
        del p["rows"][0]["losses"]["A_216_108"]
        with self.assertRaises(ValueError):
            analyze(p)

    def test_partial_predictions(self):
        p = copy.deepcopy(self.p)
        del p["rows"][1]["predicted_action"]
        with self.assertRaises(ValueError):
            analyze(p)


if __name__ == "__main__":
    unittest.main()
