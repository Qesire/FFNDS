import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_full160_row_dispatch import analyze
from check_full160_row_dispatch import BASE, read, validate


class Full160EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.records = [read(BASE / name) for name in (
            'R16V0_native_results.json', 'R16V0_budget_debit.json',
            'R16V0_video_review.json', 'R16V0_completed_resources.json')]

    def test_actual_records_pass_and_scalar_analysis_reproduces(self):
        self.assertEqual(validate(*self.records), [])
        saved = read(BASE / 'R16V0_analysis.json')
        for key, value in analyze(self.records[0], self.records[2]).items():
            self.assertEqual(value, saved[key], key)

    def test_missing_whole_model_dispatch_is_not_full160_evidence(self):
        self.records[0]['requests'][-1]['dispatch_events'].pop()
        self.assertTrue(validate(*self.records))

    def test_extra_model_work_cannot_disappear(self):
        self.records[1]['used']['full_generations'] = 6
        self.assertTrue(validate(*self.records))

    def test_single_generation_wall_cannot_be_a_performance_benchmark(self):
        self.records[0]['requests'][0]['whole_generation_performance_benchmark'] = True
        self.assertTrue(validate(*self.records))

    def test_wrong_own_noise_or_frame_aggregation_fails(self):
        self.records[0]['requests'][-1]['initial_noise_sha256'] = 'different'
        self.records[2]['RGB_fidelity_to_dense']['dynamic_residual']['pooled_RGB_MSE_0_255'] *= .5
        self.assertTrue(validate(*self.records))


if __name__ == '__main__':
    unittest.main()
