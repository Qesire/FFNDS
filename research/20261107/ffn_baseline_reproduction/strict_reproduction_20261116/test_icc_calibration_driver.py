"""CPU orchestration checks using fake scalars; no model/data experiment."""
import json
import tempfile
import unittest
from pathlib import Path
from icc_calibration_driver import NativeAdapter, NFEJournal, execute, validate_and_plan

class FakeAdapter(NativeAdapter):
    def __init__(self): self.encodes = self.forwards = self.steps = 0
    def encode_source(self, record): self.encodes += 1; return record['source_cluster_id']
    def prepare_transaction(self, encoded, record, txn):
        branch = 'main' if txn.may_reuse_main else txn.branch
        return (encoded, txn.main_id, branch)
    def same_native_input_and_conditions(self, left, right): return left == right
    def begin_transaction(self, prepared, txn): pass
    def forward(self, prepared, txn, step, phase):
        self.forwards += 1; return step, {'phase': phase}
    def native_step_post(self, prepared, txn, first): self.steps += 1
    def finish_transaction(self, prepared, txn, responses): return {'steps':len(responses)}
    def compare_diagnostic(self, main, alternate, txn): return {'paired': True}
    def abort_transaction(self, prepared, txn): pass

class DriverTest(unittest.TestCase):
    def test_frozen_bound_counts_and_sequence(self):
        manifest = json.loads(Path(__file__).with_name('ICC_CLEAN_CAL11_BOUND.json').read_text(encoding='utf8'))
        jobs, budget = validate_and_plan(manifest)
        self.assertEqual((len(jobs), budget['main_nfe'], budget['posterior_extra_nfe_max'],budget['text_extra_nfe_max']), (68,66,9,18))
        with tempfile.TemporaryDirectory() as d:
            adapter = FakeAdapter(); journal = NFEJournal(Path(d)/'events.jsonl')
            result = execute(manifest, adapter, journal, authorize=lambda *x: None)
            self.assertEqual((result['attempted_nfe'], result['main_nfe'], result['diagnostic_nfe']), (93,66,27))
            self.assertEqual((adapter.encodes, adapter.forwards), (11,93))
            self.assertEqual(sum(v.get('status') == 'REUSED_IDENTICAL_INPUT' for v in result['transactions'].values()), 6)
            with self.assertRaises(FileExistsError): NFEJournal(Path(d)/'events.jsonl')

    def test_retry_executes_only_one_registered_cd_pair(self):
        manifest = json.loads(Path(__file__).with_name('ICC_CLEAN_CAL11_BOUND.json').read_text(encoding='utf8'))
        source = 'pe-video:693694740'
        pair = f'{source}|CD|low:main'
        with tempfile.TemporaryDirectory() as directory:
            journal = NFEJournal(Path(directory)/'events.jsonl')
            adapter = FakeAdapter()
            result = execute(manifest, adapter, journal, authorize=lambda *args: None,
                             source_ids=[source], retry_pair=pair)
            self.assertEqual((adapter.forwards, adapter.steps, result['diagnostic_nfe']), (2, 1, 0))
            self.assertEqual(list(result['transactions']), [pair])
            attempts = [json.loads(line) for line in journal.path.read_text().splitlines()][::2]
            self.assertEqual([row['step'] for row in attempts], [2, 3])
            with self.assertRaises(ValueError):
                execute(manifest, adapter, journal, authorize=lambda *args: None,
                        source_ids=[source], retry_pair=f'{source}|CA|low:main')

if __name__ == '__main__': unittest.main()
