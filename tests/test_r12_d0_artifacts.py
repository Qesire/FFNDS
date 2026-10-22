"""Completed native evidence must not silently become a different claim."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from check_r12_d0_artifacts import validate
from analyze_r12_budget_debit import analyze


def read(relative):
    return json.loads((ROOT / relative).read_text())


class D0EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.records = [read(p) for p in ('research/20261022/R12_D0_native_results.json',
            'research/CURRENT_DECISION.json', 'research/20261021/method_triage.json',
            'research/20261021/R7D1_results.json')]

    def errors(self):
        return validate(*self.records)

    def test_completed_native_records_reproduce(self):
        self.assertEqual(self.errors(), [])

    def test_oracle_must_beat_best_fixed_with_frozen_gates(self):
        d = self.records[0]['decision']
        d['diagnostic_gate_pass'] = not d['diagnostic_gate_pass']
        self.assertTrue(any('pooled gates' in e for e in self.errors()))

    def test_missing_bad_action_cannot_improve_the_reported_result(self):
        self.records[0]['rows'] = [r for r in self.records[0]['rows'] if r['arm'] != 'NA']
        self.assertTrue(any('unregistered case' in e for e in self.errors()))

    def test_main_half_budget_and_removed_correction_are_audited(self):
        r = next(r for r in self.records[0]['rows'] if r['arm'] == 'NA')
        r['execution']['actual_residual_positions'] = 160
        self.assertTrue(any('execution budget' in e for e in self.errors()))

    def test_step_restoration_cannot_be_claimed_as_free_quality(self):
        self.records[0]['free_student_quality_or_FFN_speed_acceptance'] = True
        self.assertTrue(any('free quality' in e for e in self.errors()))

    def test_deterministic_repeat_requires_flow_bytes_not_only_SSE(self):
        r = next(r for r in self.records[0]['rows'] if r['arm'] == 'N0_repeat')
        r['steps'][3]['flow_sha256'] = 'different-native-bytes'
        self.assertTrue(any('Repeated N0 native bytes' in e for e in self.errors()))

    def test_completed_diagnostic_does_not_auto_authorize_more_models(self):
        self.records[1]['eligible_model_method_ids'] = ['R12E1_D0_EQUAL_BUDGET_NATIVE_VALUE']
        self.assertTrue(any('auto-authorize' in e for e in self.errors()))

    def test_cross_round_debit_requires_identical_intermediate_state(self):
        actual = analyze(self.records[0], self.records[3])
        self.assertEqual(actual['new_model_generations'], 0)
        altered = copy.deepcopy(self.records[0])
        row = next(r for r in altered['rows'] if r['arm'] == 'NA')
        row['scopes'][1]['norm_input_sha256'] = 'different-state'
        with self.assertRaisesRegex(ValueError, 'identical incoming state'):
            analyze(altered, self.records[3])


if __name__ == '__main__':
    unittest.main()
