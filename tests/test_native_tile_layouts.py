"""Row parity, actual layout and evidence boundaries for future spatial actions."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from analyze_native_tile_layouts import analyze


class TileLayoutTests(unittest.TestCase):
    def test_equal_spatial_regions_have_distinct_physical_layout(self):
        result = analyze()
        self.assertEqual(result['full_native_rows'], 31668)
        self.assertEqual([r['contiguous_segments'] for r in result['layouts']], [[609, 609], [21, 21], [1, 1]])
        for row in result['layouts']:
            self.assertEqual(row['rows'], [15834, 15834])
            self.assertEqual(row['pair_row_weighted_groups'], 162.)

    def test_odd_frame_count_breaks_naive_equal_budget(self):
        result = analyze()['whole_frame_10_vs11_counterexample']
        self.assertLess(result[0]['global_main_MAC_fraction'], .5)
        self.assertGreater(result[1]['global_main_MAC_fraction'], .5)
        self.assertFalse(result[1]['within_50pct'])

    def test_recorded_reference_proof_is_not_native_model_evidence(self):
        record = json.loads((ROOT / 'research/20261022/R12_HiF4_group_separability.json').read_text())
        self.assertFalse(record['CUDA_initialized'])
        self.assertEqual(record['new_model_generations'], 0)
        self.assertEqual(len(record['saved_post_Q1_reference_cases']), 8)
        self.assertTrue(all(all(r['checks'].values()) for r in record['saved_post_Q1_reference_cases']))
        self.assertEqual(record['within64_hierarchy_counterexample']['BF16_element_mismatches'], 401)
        self.assertEqual(record['audit_source_sha256'], hashlib.sha256((ROOT / 'tools/audit_hif4_group_separability.py').read_bytes()).hexdigest())
        self.assertIn('Does not prove CUDA/Python equality', record['scope'])


if __name__ == '__main__':
    unittest.main()
