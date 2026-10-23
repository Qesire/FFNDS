"""Pending C2 authority, immutable scope and predebit checks; no models."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from c2_contract import ARMS, CAPS, METHOD, ROOT, Budget, atomic, sha, require_registration, validate_registration


class C2Contract(unittest.TestCase):
    def test_pending_entry_denies_before_model_import(self):
        with patch('c2_contract.subprocess.run', return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(RuntimeError, 'Torch/model/CUDA not imported'):
                require_registration()

    def test_exact_scope_and_source_authority(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            atomic(root / 'authority.json', {'id': METHOD, 'caps': CAPS,
                'status': 'EXPLICIT_USER_EXECUTION_AUTHORITY', 'user_reply': 'Synthetic test fixture only'})
            (root / 'source.py').write_text('# synthetic identity\n')
            p = {'id': METHOD, 'caps': CAPS, 'arms': ARMS, 'source_index': 6,
                 'status': 'REGISTERED_FINITE_C2_NATIVE_SMOKE', 'automatic_retry': False,
                 'reset_between_steps': False, 'max_concurrent_GPUs': 1, 'new_output_bytes_max': 1 << 29,
                 'authorization': {'path': 'authority.json', 'sha256': sha(root / 'authority.json')},
                 'source_sha256': {'source.py': sha(root / 'source.py')}, 'input_sha256': {}}
            atomic(root / 'protocol.json', p)
            registration = {'protocol': 'protocol.json', 'protocol_sha256': sha(root / 'protocol.json')}
            self.assertEqual(validate_registration(registration, root), [])
            (root / 'source.py').write_text('# changed fixture\n')
            self.assertIn('C2 immutable source/input changed: source.py', validate_registration(registration, root))

    def test_failed_debit_no_retry_and_zero_ffn_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            b = Budget(folder)
            with self.assertRaises(ValueError):
                with b.call('synthetic', 'full_generations'): raise ValueError('fixture')
            self.assertEqual(b.state['used']['full_generations'], 1)
            self.assertEqual(b.state['debits'][0]['status'], 'FAILED_DEBIT_PRESERVED')
            with self.assertRaises(RuntimeError):
                with b.call('synthetic', 'full_generations'): self.fail('Duplicate entered')
            with self.assertRaises(RuntimeError):
                with b.call('not_permitted', 'standalone_FFN_forwards'): self.fail('Zero budget entered')
            b.close()
            with self.assertRaises(RuntimeError): Budget(folder)


if __name__ == '__main__': unittest.main()
