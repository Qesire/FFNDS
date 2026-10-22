"""R14N0 cannot silently retry, overspend or import a model before authority."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from r14_stripe_contract import Budget, CALLS, CAPS, ORDER


class StripeBudgetTests(unittest.TestCase):
    def test_exact_registered_calls_reach_but_do_not_exceed_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = Budget(tmp, create=True)
            try:
                for index in ORDER:
                    with budget.call(index, 'capture'):
                        pass
                    for label in CALLS:
                        with budget.call(index, label):
                            pass
                self.assertEqual(budget.state['used'], CAPS)
                with self.assertRaises(RuntimeError):
                    budget.debit(0, 'dynamic')
            finally:
                budget.close()

    def test_failed_call_keeps_debit_and_prevents_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = Budget(tmp, create=True)
            try:
                with self.assertRaises(ValueError):
                    with budget.call(1, 'complement_A'):
                        raise ValueError('Synthetic failure before completion')
            finally:
                budget.close()
            resumed = Budget(tmp)
            try:
                self.assertEqual(resumed.state['used']['extra_linear_calls'], 3)
                self.assertEqual(resumed.state['debits'][0]['status'], 'FAILED_DEBIT_PRESERVED')
                with self.assertRaises(RuntimeError):
                    resumed.debit(1, 'complement_A')
            finally:
                resumed.close()

    def test_new_source_or_action_not_in_fixed_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = Budget(tmp, create=True)
            try:
                with self.assertRaises(ValueError):
                    budget.debit(3, 'capture')
                with self.assertRaises(ValueError):
                    budget.debit(1, 'new_seed')
                self.assertEqual(budget.state['used']['model_generations'], 0)
            finally:
                budget.close()

    def test_edited_budget_summary_cannot_erase_prior_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = Budget(tmp, create=True)
            budget.debit(1, 'capture')
            budget.close()
            path = Path(tmp) / 'BUDGET.json'
            value = json.loads(path.read_text())
            value['used']['model_generations'] = 0
            path.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError):
                Budget(tmp)

    def test_closed_route_refusal_precedes_tensor_import_in_isolated_fixture(self):
        # Never run against live registration: the test owns a rejecting guard
        # and a fake torch module that would fail if imported, on CPU only.
        with tempfile.TemporaryDirectory() as tmp:
            tools = Path(tmp) / 'tools'
            tools.mkdir()
            for name in ('r14_stripe_contract.py', 'run_complete_stripe_value.py'):
                shutil.copyfile(ROOT / 'tools' / name, tools / name)
            (tools / 'check_research_routes.py').write_text('raise SystemExit(2)\n')
            (tools / 'torch.py').write_text('raise AssertionError("Torch imported before authority")\n')
            run = subprocess.run([sys.executable, str(tools / 'run_complete_stripe_value.py'), '--stage', 'capture'],
                                 cwd=tmp, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('no eligible registered model route', run.stderr)
            self.assertNotIn('Torch imported before authority', run.stderr)


if __name__ == '__main__':
    unittest.main()
