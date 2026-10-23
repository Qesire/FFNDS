from collections import Counter, defaultdict
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from fresh_quality_contract import ARMS, Budget, CAPS, generation_jobs, validate_registration


class FreshQualityContractTests(unittest.TestCase):
    def test_no_fresh_model_authority_from_completed_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(validate_registration({}, Path(folder)))

    def test_exact145_actual_jobs_and_distinct_quality_pairs(self):
        jobs = generation_jobs()
        self.assertEqual(len(jobs), 145)
        self.assertEqual(len({j['name'] for j in jobs}), 145)
        self.assertEqual(Counter(j['phase'] for j in jobs), {'warmup': 5, 'timing': 25, 'quality': 115})
        pairs = {(j['source'], j['arm']) for j in jobs if j['phase'] == 'quality'}
        self.assertEqual(pairs, {(i, a) for i in range(23) for a in ARMS})

    def test_each_arm_appears_once_in_each_timing_order_position(self):
        timings = [j for j in generation_jobs() if j['phase'] == 'timing']
        positions = defaultdict(list)
        for repeat in range(5):
            cycle = [j for j in timings if j['repeat'] == repeat]
            self.assertEqual(len(cycle), 5)
            for position, job in enumerate(cycle):
                positions[job['arm']].append(position)
        for arm in ARMS:
            self.assertEqual(sorted(positions[arm]), list(range(5)))

    def test_metric_phase_keeps_original_debit_and_failure_is_not_retryable(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = Budget(folder)
            with budget.call('first_generation', 'full_generations'):
                pass
            budget.close()
            metrics = Budget(folder, existing=True)
            with self.assertRaises(ValueError):
                with metrics.call('dimension_arm', 'metric_evaluator_calls'):
                    raise ValueError('failure')
            self.assertEqual(metrics.state['used']['full_generations'], 1)
            self.assertEqual(metrics.state['used']['metric_evaluator_calls'], 1)
            with self.assertRaises(RuntimeError):
                with metrics.call('dimension_arm', 'metric_evaluator_calls'):
                    pass
            with self.assertRaises(RuntimeError):
                with metrics.call('train', 'training_steps'):
                    pass
            metrics.close()


if __name__ == '__main__':
    unittest.main()
