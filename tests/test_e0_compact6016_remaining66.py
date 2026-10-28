import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import e0_compact6016_remaining66 as m


class RemainingDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT / 'config/e0_compact6016_remaining66_20261028_draft.json').read_text())
        cls.old = json.loads(m.contract.FROZEN_PATH.read_text())

    def test_draft_is_not_authorized_and_frozen_design_matches(self):
        self.assertFalse(self.plan['assets_finalized'])
        self.assertTrue(all(v == 'NOT_BOUND' for v in self.plan['completion_partial_audit'].values()))
        m.validate_frozen_design(self.plan, self.old)
        with self.assertRaises(ValueError):
            m.authorize(self.plan, 'x', {}, 0)
        for field in ('sources', 'inputs', 'random_supports', 'source_split'):
            p = copy.deepcopy(self.plan)
            p[field] = None
            with self.assertRaises(ValueError):
                m.validate_frozen_design(p, self.old)

    def test_budget_guard_and_each_part_actual_cell_counts(self):
        self.assertEqual([len(m.run_ids(p)) for p in range(4)], [18, 17, 15, 16])
        for part in range(4):
            ledger = m.Ledger({}, part)
            amount = m.part_limits(part)['Adam_steps']
            ledger.reserve({'Adam_steps': amount}, m.run_ids(part)[0])
            with self.assertRaises(ValueError):
                ledger.reserve({'Adam_steps': 1}, m.run_ids(part)[0])
            ledger.observe('Adam_steps', amount)
            with self.assertRaises(ValueError):
                ledger.observe('Adam_steps')

    def test_training_math_body_is_identical(self):
        old = (ROOT / 'tools/e0_compact6016_full80_completion.py').read_text()
        new = (ROOT / 'tools/e0_compact6016_remaining66.py').read_text()
        start = '                selected = groups if init == "energy94" else random_map[expert, block]'
        end = '            del raw, up, down, local_cases, local_refs, b2'
        self.assertEqual(old[old.index(start):old.index(end)], new[new.index(start):new.index(end)])
        self.assertIn('for init in initializations_for(expert, block):', new)
        self.assertIn('support_run_id = function_id(expert, block) + "_" + initializations_for(expert, block)[0]', new)
        start_random, end_random = "def random_supports():", "def select_support("
        self.assertEqual(old[old.index(start_random):old.index(end_random)],
                         new[new.index(start_random):new.index(end_random)])

    def test_assembler_rejects_bad_part_exit_before_payload(self):
        p = copy.deepcopy(self.plan)
        p['assets_finalized'] = True
        cells = [{'expert': e, 'block': b, 'initialization': i} for e,b,i in m.contract.REUSED_CELLS]
        parts = [{'part': i, 'RUN': {'sha256':'r'}, 'launch_exit': {'sha256':'x'}} for i in range(4)]
        def fake_read(root, entry):
            if entry is protocol:
                return p
            return {'id': m.ID, 'part': 0, 'protocol_sha256': 't', 'launcher_returncode': 1}
        protocol = {'sha256': 't'}
        with patch.object(m, '_read_binding', side_effect=fake_read), patch.object(m, 'validate_partial_reuse', return_value=cells):
            with self.assertRaises(ValueError):
                m.assemble_manifest(protocol, parts, ROOT)

    def test_entry_preflight_is_required_and_bound_to_actual_part_version(self):
        scope = {'id': m.ID}
        active = {'experiment_authority': {'current_user_authorized_scopes': [scope]}}
        with self.assertRaisesRegex(ValueError, 'binding missing'):
            m.validate_cpu_preflight(self.plan, 'frozen', active, 0, ROOT)
        with tempfile.TemporaryDirectory() as folder:
            hf = Path(folder)
            path = hf / 'PREFLIGHT.json'
            report = {'id': m.ID, 'part': 0, 'protocol_sha256': 'frozen',
                      'status': 'PASS_REMAINING66_CPU_ENTRY_NO_TORCH_NO_GPU',
                      'source_bindings_verified': len(self.plan['repo_sources']),
                      'reused_cells': 94, 'original_weight_shards_size_verified': 9,
                      'operand_files_size_verified': 432, 'Torch_or_CUDA_imported': False,
                      'all_CUDA_devices_hidden': True, 'native_model_calls': 0}
            def bind(value):
                path.write_text(json.dumps(value))
                scope['CPU_preflights'] = [{'part': 0, 'binding': {
                    'path': 'PREFLIGHT.json', 'sha256': m.sha(path), 'bytes': path.stat().st_size}}]
            bind(report)
            m.validate_cpu_preflight(self.plan, 'frozen', active, 0, hf)
            for key, value in [('protocol_sha256', 'stale'), ('reused_cells', 93),
                               ('Torch_or_CUDA_imported', True), ('native_model_calls', 1)]:
                bind({**report, key: value})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'version or result'):
                    m.validate_cpu_preflight(self.plan, 'frozen', active, 0, hf)
            bind(report)
            path.write_text('changed')
            with self.assertRaises(ValueError):
                m.validate_cpu_preflight(self.plan, 'frozen', active, 0, hf)


if __name__ == '__main__':
    unittest.main()
