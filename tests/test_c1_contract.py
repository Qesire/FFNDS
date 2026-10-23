"""C1 finite accounting and compensation-support invariants, without models."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from c1_contract import ARMS, BLOCKS, METHOD, SOURCES, Budget, CAPS, atomic, sha, select_swaps, validate_registration


class C1Contract(unittest.TestCase):
    def test_whole_group_swaps_keep_exact_budget(self):
        for count in (54, 108, 162):
            for groups in (list(range(count)), list(range(216 - count, 216))):
                swaps = select_swaps(groups)
                self.assertEqual(len(set(swaps)), 2)
                for outgoing, incoming in swaps:
                    self.assertIn(outgoing, groups)
                    self.assertNotIn(incoming, groups)
                    self.assertEqual(len(set(groups) - {outgoing} | {incoming}), count)

    def test_failed_call_stays_debited_and_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = Budget(folder)
            with self.assertRaisesRegex(ValueError, 'fixture'):
                with budget.call('first', 'full_generations'):
                    raise ValueError('fixture')
            self.assertEqual(budget.state['used']['full_generations'], 1)
            self.assertEqual(budget.state['debits'][0]['status'], 'FAILED_DEBIT_PRESERVED')
            with self.assertRaises(RuntimeError):
                with budget.call('first', 'full_generations'):
                    self.fail('Duplicate call entered')
            with self.assertRaises(RuntimeError):
                with budget.call('no_training', 'training_steps'):
                    self.fail('Zero-cap call entered')
            budget.close()
            with self.assertRaises(RuntimeError):
                Budget(folder)

    def test_cap_is_checked_before_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = Budget(folder)
            for i in range(CAPS['full_generations']):
                with budget.call(str(i), 'full_generations'):
                    pass
            with self.assertRaises(RuntimeError):
                with budget.call('over_limit', 'full_generations'):
                    self.fail('Over-limit call entered')
            budget.close()

    def test_bound_registration_requires_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic(root / 'authority.json', {'id': METHOD, 'caps': CAPS,
                   'status': 'EXPLICIT_USER_EXECUTION_AUTHORITY', 'user_reply': '那你跑啊！！！干嘛呢'})
            (root / 'source.py').write_text('# synthetic source identity fixture\n')
            p = {'id': METHOD, 'caps': CAPS, 'sources': SOURCES, 'blocks': BLOCKS,
                 'steps': [0, 1, 2, 3], 'arms': ARMS, 'automatic_retry': False,
                 'authorization': {'path': 'authority.json', 'sha256': sha(root / 'authority.json')},
                 'source_sha256': {'source.py': sha(root / 'source.py')}, 'input_sha256': {},
                 'max_concurrent_GPUs': 1, 'new_output_bytes_max': 1 << 30}
            atomic(root / 'protocol.json', p)
            registration = {'protocol': 'protocol.json', 'protocol_sha256': sha(root / 'protocol.json')}
            self.assertEqual(validate_registration(registration, root), [])
            changed = copy.deepcopy(registration)
            changed['protocol_sha256'] = '0' * 64
            self.assertIn('C1 exact finite scope changed', validate_registration(changed, root))
            (root / 'source.py').write_text('# changed fixture\n')
            self.assertIn('C1 immutable source/input changed: source.py', validate_registration(registration, root))


if __name__ == '__main__':
    unittest.main()
