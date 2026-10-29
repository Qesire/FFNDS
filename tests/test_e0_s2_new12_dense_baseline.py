import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_s2_new12_dense_baseline as m

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.bank = json.loads((ROOT/m.BANK).read_text())
        self.neutral = json.loads((ROOT/'config/e0_s2_spatial_student_collect_part0_20261029_v2.json').read_text())['cases'][0]

    def test_exact_partition_and_control_first(self):
        cases = m.cases(self.bank, 0, self.neutral)
        self.assertEqual([x['native_source_ordinal'] for x in cases], [898, 911, 978, 1039, 922, 616, 456])
        self.assertEqual(cases[0]['case_role'], 'HISTORICAL_DENSE_REPEAT_CONTROL_NOT_NEW_SOURCE')

    def test_original_prompt_drift_is_rejected(self):
        altered = copy.deepcopy(self.bank)
        altered['sources'][0]['prompt'] += ' changed'
        with self.assertRaises(ValueError): m.validate_bank(altered)

    def test_budget_setup_vs_seven_videos(self):
        l = m.limits(0)
        self.assertEqual(l['original_expert_loads'], 2)
        self.assertEqual(l['T5_buffer_Default_MM_loads'], 7)
        self.assertEqual(l['video_generations'], 7)
        self.assertEqual(l['native_FFN_calls'], 1120)
        self.assertEqual(l['state_output_writes'], 70)
        self.assertEqual(l['historical_state_byte_checks'], 10)
        self.assertEqual(l['core_CPU_reports'], 0)
        self.assertEqual(l['prefetch_transfer_observations'], 1134)

    def test_single_arm_registration_keeps_control_identity(self):
        cases = m.cases(self.bank, 0, self.neutral)
        manifest = m.registration(cases, 0, Path('/tmp'), 'a'*64, 'synthetic')
        self.assertEqual([x['variant'] for x in manifest['samples']], ['dense']*7)
        self.assertTrue(manifest['sources'][0]['exposure_history']['prior_evaluation'])
        self.assertFalse(manifest['sources'][1]['exposure_history']['prior_evaluation'])
        self.assertIn('CONTROL', manifest['sources'][0]['acquisition_role'])

    def test_neutral_gate_rejects_incomplete_or_failed(self):
        with self.assertRaises(ValueError): m.require_neutral({'matched_olddev_all_ten_states': True, 'states': dict.fromkeys(range(9))})
        with self.assertRaises(ValueError): m.require_neutral({'matched_olddev_all_ten_states': False, 'states': dict.fromkeys(range(10))})
        m.require_neutral({'matched_olddev_all_ten_states': True, 'states': dict.fromkeys(range(10))})

    def test_dense_only_rejects_factor(self):
        with self.assertRaises(ValueError): m.run_id(self.neutral, 'factor')

    def test_duplicate_source_is_rejected(self):
        altered = copy.deepcopy(self.bank)
        altered['sources'][1]['image']['sha256'] = altered['sources'][0]['image']['sha256']
        with self.assertRaises(ValueError): m.validate_bank(altered)

if __name__ == '__main__': unittest.main()
