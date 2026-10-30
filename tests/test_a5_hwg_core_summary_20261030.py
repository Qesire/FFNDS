"""Synthetic source clustering and typed-byte integrity, no CUDA/model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_20261030 as guard
original_resources = guard.RES
import a5_hwg_registered_core_summary_20261030 as runner
from a5_hwg_core_summary_math_20261030 import summarize_sources
guard.RES = original_resources


class CoreSummaryFixtures(unittest.TestCase):
    def test_sources_get_equal_weight_despite_different_nested_counts(self):
        result = summarize_sources([[1.] * 80, [3.], [5.] * 40])
        self.assertEqual(result['source_equal_mean'], 3.)
        self.assertEqual(result['independent_source_count'], 3)
        self.assertEqual(result['defined_sites_per_source'], [80, 1, 40])
        self.assertFalse(result['new_source_or_video_quality_confirmation'])

    def test_bootstrap_reproducibility_and_missing_source_rejected(self):
        self.assertEqual(summarize_sources([[1., None], [2.]]), summarize_sources([[1., None], [2.]]))
        with self.assertRaises(ValueError): summarize_sources([[1.], [None]])

    def test_raw_mask_corruption_rejected_even_with_updated_file_SHA(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = b'\x01\x02\x80'
            header = json.dumps({'mask': {'dtype': 'U8', 'shape': [1, 3], 'data_offsets': [0, 3]}}).encode()
            data = struct.pack('<Q', len(header)) + header + raw
            path = root / 'mask.safetensors'
            path.write_bytes(data)
            ref = {'path': path.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            schema = {'mask': {'shape': [1, 3], 'dtype': 'uint8', 'raw_tensor_sha256': hashlib.sha256(raw).hexdigest()}}
            budget = guard.ReadBudget(1 << 20)
            runner.verify_masks(root, ref, schema, budget)
            self.assertEqual(budget.actual, len(data))
            path.write_bytes(data[:-1] + b'\x81')
            ref['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, 'raw tensor SHA'):
                runner.verify_masks(root, ref, schema, guard.ReadBudget(1 << 20))


if __name__ == '__main__': unittest.main()
