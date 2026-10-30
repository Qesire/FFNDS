"""A foreign process must stop reads before the next large asset allocation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('a5_relocation', Path(__file__).resolve().parents[1] / 'tools/a5_hwg_matched_capture_relocation_v3_20261030.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class EarlyResourceFixtures(unittest.TestCase):
    def test_asset_predebit_and_chunk_both_call_existing_monitor(self):
        events = []
        budget = m.repaired.owner.ReadBudget(20, check=lambda: events.append('wall'))
        monitor = SimpleNamespace(check=lambda: events.append('assigned_GPU'))
        m.attach_monitor_to_reads(budget, monitor)
        budget.before(10, 'fixture')
        budget.after(4)
        self.assertEqual(events, ['wall', 'assigned_GPU', 'wall', 'assigned_GPU'])

    def test_foreign_compute_preserves_partial_read_and_stops_next_charge(self):
        foreign = [False]
        def check():
            if foreign[0]: raise RuntimeError('foreign compute fixture')
        budget = m.repaired.owner.ReadBudget(20)
        m.attach_monitor_to_reads(budget, SimpleNamespace(check=check))
        budget.before(10, 'fixture')
        budget.after(4)
        foreign[0] = True
        with self.assertRaisesRegex(RuntimeError, 'foreign compute'):
            budget.after(3)
        self.assertEqual(budget.actual, 7)
        self.assertIsNotNone(budget.inflight)


if __name__ == '__main__': unittest.main()
