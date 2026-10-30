"""Evidence hash/partial-read fixtures without any CUDA call."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_20261030 as owner
spec = importlib.util.spec_from_file_location('a5_io', ROOT / 'tools/a5_hwg_evidence_io_20261030.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class EvidenceReadFixtures(unittest.TestCase):
    def test_bound_bytes_single_count_and_corruption_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').write_bytes(b'abcdef')
            ref = {'path': 'data', 'bytes': 6, 'sha256': hashlib.sha256(b'abcdef').hexdigest()}
            budget = owner.ReadBudget(12)
            self.assertEqual(m.bound_bytes(root, ref, budget), b'abcdef')
            self.assertEqual(budget.actual, 6)
            (root / 'data').write_bytes(b'abcdeg')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                m.bound_bytes(root, ref, budget)
            self.assertEqual(budget.actual, 12)
            self.assertIsNotNone(budget.inflight)

    def test_escape_and_oversize_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').write_bytes(b'ab')
            budget = owner.ReadBudget(20)
            for ref in [{'path': '../outside', 'bytes': 2}, {'path': 'data', 'bytes': 1}]:
                with self.assertRaises(ValueError): m.bound_bytes(root, ref, budget)
            self.assertEqual(budget.predebited, 0)

    def test_original_bias_slices_are_charged_and_recovered_without_mmap(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fake_model.safetensors'
            values = {f'blocks.{block}.ffn.{linear}.bias':
                      torch.full((width,), block + linear, dtype=torch.bfloat16)
                      for block in range(40) for linear, width in [(0, 13824), (2, 5120)]}
            save_file(values, str(path))
            ref = {'bytes': path.stat().st_size, 'sha256': owner.sha(path)}
            budget = owner.ReadBudget(8 << 20)
            def charged_sha(filename):
                data = filename.read_bytes()
                budget.before(len(data), 'explicit_full_checkpoint_SHA')
                budget.after(len(data))
                budget.finish()
                return hashlib.sha256(data).hexdigest()
            result = m.checkpoint_biases(path, ref, budget, charged_sha)
            self.assertEqual(set(result), set(values))
            self.assertTrue(all(torch.equal(result[k], v) for k, v in values.items()))
            self.assertEqual(budget.actual, ref['bytes'] * 2)
            self.assertIsNone(budget.inflight)

    def test_wrong_checkpoint_hash_stops_before_bias_slices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'not_a_checkpoint'
            path.write_bytes(b'fake')
            budget = owner.ReadBudget(100)
            with self.assertRaisesRegex(ValueError, 'checkpoint SHA'):
                m.checkpoint_biases(path, {'bytes': 4, 'sha256': 'wrong'}, budget,
                                    lambda _: 'actual')
            self.assertEqual(budget.actual, 0)


if __name__ == '__main__': unittest.main()
