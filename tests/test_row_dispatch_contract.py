import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from row_dispatch_contract import Budget, CAPS, validate_registration


class RowDispatchContractTests(unittest.TestCase):
    def test_missing_authority_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(validate_registration({}, Path(folder)))

    def test_failure_remains_debited_and_cannot_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = Budget(folder)
            with self.assertRaises(ValueError):
                with budget.call('candidate', 'full_generations'):
                    raise ValueError('simulated failure')
            record = json.loads((Path(folder) / 'BUDGET.json').read_text())
            self.assertEqual(record['used']['full_generations'], 1)
            self.assertEqual(record['debits'][0]['status'], 'FAILED_DEBIT_PRESERVED')
            with self.assertRaises(RuntimeError):
                with budget.call('candidate', 'full_generations'):
                    pass
            budget.close()

    def test_full_generation_cap_and_zero_training(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = Budget(folder)
            for i in range(CAPS['full_generations']):
                with budget.call(str(i), 'full_generations'):
                    pass
            with self.assertRaises(RuntimeError):
                with budget.call('sixth', 'full_generations'):
                    pass
            with self.assertRaises(RuntimeError):
                with budget.call('fit', 'training_steps'):
                    pass
            budget.close()


if __name__ == '__main__':
    unittest.main()
