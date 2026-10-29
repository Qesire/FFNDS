"""Independent toy identities and real work-count boundaries; no saved operands."""
import json
import subprocess
import sys
import unittest

import numpy as np
from tools import e0_raw_bf16_full21tau_factor_replay as m


class FactorTemporalControls(unittest.TestCase):
    def test_AC_direction_energy_and_DC_error_are_separate(self):
        t = np.broadcast_to((np.arange(21)-10)[:, None, None], (21, 8, 2)).astype(float)
        reference = 100+t
        v = m.error_components(100-t, reference, np)
        self.assertAlmostEqual(v[-3], 0)
        self.assertAlmostEqual(v[-2], -1)
        self.assertAlmostEqual(v[-1], -1)
        self.assertAlmostEqual(v[7], 1)
        erased = m.error_components(np.full_like(reference, 100), reference, np)
        self.assertAlmostEqual(erased[-2], 0)
        self.assertIsNone(erased[-1])

    def test_zero_DC_or_time_variance_stays_undefined(self):
        zero = np.zeros((21, 8, 2))
        v = m.error_components(zero, zero, np)
        self.assertEqual(v[-3:], [None, None, None])
        t = np.broadcast_to((np.arange(21)-10)[:, None, None], (21, 8, 2)).astype(float)
        self.assertIsNone(m.error_components(t, t, np)[-3])

    def test_full160_source_work_and_original_rounding_control_are_paid(self):
        packet = {'payload': {'bytes': 100}}
        index = {'cases': [{'packets': [packet]*160} for _ in range(10)]}
        costs = m.limits(index)
        self.assertEqual(costs['BF16_factor_GEMM'], 6400)
        self.assertEqual(costs['BF16_original_W2_GEMM'], 1600)
        self.assertEqual(costs['raw_W2_tensors'], 80)
        self.assertEqual(costs['raw_H2D_tensors'], 160)
        self.assertEqual(costs['original_u_byte_neutrals'], 1600)
        self.assertEqual(costs['packet_full_SHA_bytes'], 160000)
        with self.assertRaises(ValueError):
            m.limits({'cases': index['cases'][:-1]})

    def test_source_equal_uses_nested_position_then_image_and_tracks_missing(self):
        values = np.full((10, 4, 40, len(m.labels())), np.nan)
        values[:5] = 1
        values[5:] = 3
        values[0, 0, 0, 0] = np.nan
        result = m.source_equal(values, np)
        self.assertEqual(result['source_equal_all160'][0], 2)
        self.assertEqual(result['source_equal_all160_source_N'][0], 10)
        self.assertEqual(result['per_source_valid_positions'][0][0], 159)
        self.assertEqual(result['joint_source_N'][0][0][0], 9)

    def test_standalone_import_has_no_numerical_import_or_GPU_side_effect(self):
        script = "import sys;sys.path.insert(0,'tools');import e0_raw_bf16_full21tau_factor_replay;assert not {'numpy','torch'}&set(sys.modules)"
        p = subprocess.run([sys.executable, '-c', script], cwd=m.ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == '__main__':
    unittest.main()
