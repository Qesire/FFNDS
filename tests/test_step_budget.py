import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from step_budget_runtime import validate_budget


class StepBudgetControl(unittest.TestCase):
    def test_frozen_budget_preserves_total_compute_and_distinguishes_identity(self):
        for identity, counts in [(True, [108] * 4), (False, [54, 108, 108, 162])]:
            self.assertEqual(sum(validate_budget({'identity_control': identity, 'groups_by_step': counts})), 432)
        for identity, counts in [(False, [108] * 4), (True, [54, 108, 108, 162]), (False, [54, 108, 108, 161])]:
            with self.assertRaises(RuntimeError): validate_budget({'identity_control': identity, 'groups_by_step': counts})


if __name__ == '__main__': unittest.main()
