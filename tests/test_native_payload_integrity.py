import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ar2_checkpoint import digest
from native_payload_integrity import verify_payload


class NativePayloadIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'step0_block00.safetensors'
        self.values = {name: torch.zeros(256, 5120, dtype=torch.bfloat16)
                       for name in ['V1', 'BaseY', 'TeacherY', 'StudentY']}
        self.values.update(b2=torch.zeros(5120, dtype=torch.bfloat16),
                           gate=torch.ones(1, 5120, dtype=torch.bfloat16))

    def save(self):
        save_file(self.values, str(self.path))
        return {'path': str(self.path), 'bytes': self.path.stat().st_size, 'sha256': digest(self.path)}

    def test_valid_native_payload(self):
        scope = self.save()
        self.assertEqual(verify_payload(scope, self.root), scope['bytes'])

    def test_changed_payload_byte_is_rejected(self):
        scope = self.save()
        with self.path.open('r+b') as handle:
            handle.seek(-1, 2)
            handle.write(b'\x01')
        with self.assertRaises(RuntimeError):
            verify_payload(scope, self.root)

    def test_wrong_sample_shape_is_rejected_even_with_valid_sha(self):
        self.values['V1'] = self.values['V1'][:255].contiguous()
        with self.assertRaises(RuntimeError):
            verify_payload(self.save(), self.root)

    def test_wrong_dtype_is_rejected_even_with_valid_sha(self):
        self.values['gate'] = self.values['gate'].float()
        with self.assertRaises(RuntimeError):
            verify_payload(self.save(), self.root)

    def test_foreign_input_directory_is_rejected(self):
        with self.assertRaises(RuntimeError):
            verify_payload(self.save(), self.root / 'other_input')
