"""A valid local native measurement cannot silently become a stronger result."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from check_r14_native_value import validate


class NativeStripeRecordTests(unittest.TestCase):
    def setUp(self):
        self.records = [json.loads((ROOT / 'research/20261022' / name).read_text()) for name in
                        ('R14N0_native_results.json', 'R14N0_capture_receipt.json',
                         'R14N0_budget_debit.json', 'R14N0_completed_resources.json')]

    def test_executed_native_values_reproduce(self):
        self.assertEqual(validate(*self.records), [])

    def test_local_oracle_cannot_be_promoted_to_video_quality(self):
        self.records[0]['native_video_or_speed_acceptance'] = True
        self.assertIn('Local writeback cannot claim free-video/model-speed acceptance', validate(*self.records))

    def test_bad_source_cannot_be_dropped(self):
        self.records[0]['cases'].pop()
        self.assertIn('Exactly four original source action records required', validate(*self.records))

    def test_oracle_numeric_summary_cannot_replace_actual_native_masks(self):
        self.records[0]['actual_execution'][0]['actual_composition_controls']['oracle']['executed_post_bitwise_matches_local_values'] = False
        self.assertIn('Output splice relabelled executed action', validate(*self.records))

    def test_extra_model_or_linear_work_cannot_disappear(self):
        self.records[3]['extra_linear_calls_including_capture_teacher'] = 103
        self.assertIn('Completed model/FFN/linear resource accounting changed', validate(*self.records))


if __name__ == '__main__':
    unittest.main()
