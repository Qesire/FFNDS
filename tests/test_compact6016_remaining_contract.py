"""Portable metadata tests; no native model or external filesystem required."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import compact6016_remaining_contract as m


def fixture():
    plan = json.loads(m.FROZEN_PATH.read_text())
    runs = []
    for part in range(4):
        cells = []
        for e, b in m.completion.functions_for(part):
            for init in m.INITIALIZATIONS:
                if (e, b, init) in m.PARENT_CELLS:
                    cells.append({'expert': e, 'block': b, 'initialization': init,
                                  'run_id': m.completion.function_id(e, b) + '_' + init,
                                  'status': m.completion.CELL_STATUS})
        runs.append({'part': part, 'id': m.completion.ID, 'protocol_sha256': m.PROTOCOL_SHA256,
                     'status': 'FAILED_NO_RETRY', 'error_type': 'BrokenPipeError',
                     'source_split': copy.deepcopy(plan['source_split']), 'cells': cells})
    return plan, runs


class ContractTests(unittest.TestCase):
    def test_partition_and_special_half_pairs(self):
        p, r = fixture()
        self.assertEqual(m.verify_partition(p, r), {'reused_cells': 94, 'missing_cells': 66,
                                                   'missing_functions': 34, 'overlap': 0})
        self.assertEqual([len(m.functions_for(i)) for i in range(4)], [9, 9, 8, 8])
        self.assertEqual([len(m.run_ids(i)) for i in range(4)], [18, 17, 15, 16])
        for block in (16, 31):
            self.assertEqual(m.initializations_for(m.EXPERTS[1], block), ('random94',))
        self.assertEqual(len(set(m.run_ids())), 66)
        self.assertTrue(all(x.startswith('compact6016remain_') for x in m.run_ids()))
        self.assertFalse(m.INITIALIZATION_CONTRACT['load_partial_parameters'])
        self.assertFalse(m.INITIALIZATION_CONTRACT['load_partial_optimizer'])

    def test_missing_duplicate_half_and_foreign_scope(self):
        for mutate in (
            lambda r: r[0]['cells'].__setitem__(0, copy.deepcopy(r[0]['cells'][1])),
            lambda r: r[2]['cells'].pop(),
            lambda r: r[2]['cells'][-1].update(initialization='random94'),
            lambda r: r[0].update(id='foreign'),
            lambda r: r[0].update(protocol_sha256='0' * 64),
            lambda r: r[0].update(error_type='OtherError'),
            lambda r: r[0]['source_split']['training'].pop(),
        ):
            p, r = fixture()
            mutate(r)
            with self.assertRaises(ValueError):
                m.verify_partition(p, r)

    def test_frozen_design_rejects_source_seed_split_operand_and_random_changes(self):
        for mutate in (
            lambda p: p['sources'][0].update(seed=999),
            lambda p: p['source_split']['training'].pop(),
            lambda p: p['inputs'][0].update(sha256='0' * 64),
            lambda p: p['random_supports'][0]['groups'].__setitem__(0, 0),
            lambda p: p['random_supports'].pop(),
            lambda p: p['training'].update(seed=43),
        ):
            p, r = fixture()
            mutate(p)
            with self.assertRaises(ValueError):
                m.verify_partition(p, r)

    def test_budget_derived_and_enforced(self):
        self.assertEqual(m.TOTAL_LIMITS['Adam_steps'], 16896)
        self.assertEqual(m.TOTAL_LIMITS['support_FP32_batched_bmm'], 29376)
        self.assertEqual(m.RESOURCE_LIMITS['new_HF_bytes_per_part'], 3 << 30)
        self.assertEqual(m.RESOURCE_LIMITS['new_HF_bytes_total'], 12 << 30)
        for part in range(4):
            limit = m.part_limits(part)
            m.validate_budget(part, limit, limit)
            invalid = dict(limit, Adam_steps=limit['Adam_steps'] + 1)
            with self.assertRaises(ValueError):
                m.validate_budget(part, invalid, limit)
            invalid = dict(limit, Adam_steps=-1)
            with self.assertRaises(ValueError):
                m.validate_budget(part, limit, invalid)
        with self.assertRaises(ValueError):
            m.functions_for(True)


if __name__ == '__main__':
    unittest.main()
