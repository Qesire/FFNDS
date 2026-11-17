"""CPU-only constructor safety checks; no Torch, GPU or research execution."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class TailGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_torch = types.ModuleType('torch'); fake_torch.__path__ = []
        fake_nn = types.ModuleType('torch.nn'); fake_nn.__path__ = []
        fake_functional = types.ModuleType('torch.nn.functional')
        fake_port = types.ModuleType('wan_ffn_port'); fake_port.dense = lambda *a: None
        modules = {'torch':fake_torch, 'torch.nn':fake_nn, 'torch.nn.functional':fake_functional,
                   'wan_ffn_port':fake_port}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location('chip_port_safety_test', Path(__file__).with_name('wan_chip_author_port.py'))
            cls.mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.mod)

    def test_default_dense_and_reject_unknown(self):
        port = self.mod.NativeAuthorChipPort(force_dense=True)
        self.assertEqual(port.chip_tail_mode, 'dense')
        with self.assertRaises(ValueError): self.mod.NativeAuthorChipPort(force_dense=True, chip_tail_mode='other')

    def test_valid52_requires_verified_callback_before_any_scratch_write(self):
        with tempfile.TemporaryDirectory() as d:
            scratch = Path(d)/'must_not_exist'
            with self.assertRaises(RuntimeError):
                self.mod.NativeAuthorChipPort(force_dense=False, chip_tail_mode='valid52', scratch=scratch)
            with self.assertRaises(RuntimeError):
                self.mod.NativeAuthorChipPort(force_dense=True, chip_tail_mode='valid52', tail_evidence_check=lambda: True, scratch=scratch)
            with self.assertRaises(RuntimeError):
                self.mod.NativeAuthorChipPort(force_dense=False, chip_tail_mode='valid52', tail_evidence_check=lambda: False, scratch=scratch)
            self.assertFalse(scratch.exists())

if __name__ == '__main__': unittest.main()
