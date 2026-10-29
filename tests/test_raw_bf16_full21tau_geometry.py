"""Mathematical and finite BF16 controls; no saved model data or GPU."""
import unittest
import numpy as np

from tools import raw_bf16_full21tau_geometry as m


class CompleteTrajectoryGeometry(unittest.TestCase):
    def test_balanced_decomposition_and_explicit_all_temporal_pairs(self):
        rng = np.random.default_rng(731)
        x = rng.normal(size=(21, 8, 3))+2
        values = m.two_way(x, np)
        self.assertAlmostEqual(sum(values[1:5]), 1, places=13)
        self.assertAlmostEqual(values[5], values[2]+values[4], places=13)
        numerator = sum(float(((x[i]-x[j])**2).sum()) for i in range(21) for j in range(i+1, 21))
        denominator = sum(float((x[i]**2).sum()+(x[j]**2).sum()) for i in range(21) for j in range(i+1, 21))
        self.assertAlmostEqual(values[-1], numerator/denominator, places=13)
        self.assertLess(values[7], 2e-15)

    def test_distinct_common_time_static_space_and_interaction(self):
        t = np.arange(21, dtype=np.float64)[:, None, None]-10
        a = np.arange(8, dtype=np.float64)[None, :, None]-3.5
        for x, component in ((np.broadcast_to(t, (21, 8, 1)), 2),
                             (np.broadcast_to(a, (21, 8, 1)), 3), (t*a, 4)):
            values = m.two_way(x, np)
            self.assertAlmostEqual(values[component], 1, places=13)
            self.assertAlmostEqual(sum(values[1:5])-values[component], 0, places=13)
        # Mixing tau and spatial axes would swap these mechanisms.
        self.assertNotEqual(m.two_way(t*np.ones((1, 8, 1)), np)[2],
                            m.two_way(a*np.ones((21, 1, 1)), np)[2])

    def test_equal_energy_opposite_motion_and_large_DC_hiding_error(self):
        t = np.broadcast_to((np.arange(21)-10)[:, None, None], (21, 8, 2)).astype(float)
        flipped = m.error_geometry(-t, t, np)
        self.assertAlmostEqual(flipped[7], 1)
        self.assertAlmostEqual(flipped[6], 4)
        self.assertAlmostEqual(flipped[8], -2)
        self.assertAlmostEqual(flipped[10], -1)
        reference = 1000+t
        erased = m.error_geometry(np.full_like(reference, 1000), reference, np)
        self.assertLess(erased[2], .0001)
        self.assertAlmostEqual(erased[6], 1)
        self.assertAlmostEqual(erased[7], 0)

    def test_zero_norm_and_single_time_are_explicitly_missing(self):
        x = np.zeros((21, 8, 2))
        e = m.error_geometry(x, x, np)
        self.assertTrue(all(e[i] is None for i in (6, 7, 8, 10)))
        self.assertIsNone(m.two_way(x[:1], np)[-1])
        p = m.spatial_pairs(x, np)
        self.assertEqual(p[-3:], [0, 0, 0])
        self.assertIsNone(p[3])

    def test_native_row_maps_protect_conditioning_time_and_fixed_work(self):
        self.assertEqual(m.rows(), sorted(set(m.rows())))
        m.validate_rows(m.rows())
        with self.assertRaises(ValueError):
            m.validate_rows(list(reversed(m.rows())))
        for action, count in zip(m.ACTIONS, (16, 88, 88), strict=True):
            mapping = m.representative_map(action)
            self.assertEqual(mapping[:8], list(range(8)))
            self.assertEqual(len(set(mapping)), count)
            self.assertTrue(all(0 <= i < 168 for i in mapping))
            if action != 'T10_KEEP_T0':
                self.assertTrue(all(i//8 == j//8 for j, i in enumerate(mapping)))

    def test_bf16_ties_round_even_and_split_product_add_is_observable(self):
        ties = np.array([1+1/256, 1+3/256], dtype=np.float32)
        np.testing.assert_array_equal(m.round_bf16(ties, np), np.array([1, 1+4/256], np.float32))
        # All inputs are exactly BF16. Product rounding erases .00390625;
        # retaining the product in FP32 until addition gives a different result.
        y = np.array([[1+1/128]], np.float32)
        gate = np.array([1.5], np.float32)
        r = np.array([[-1.5]], np.float32)
        update, u = m.postwrite(r, y, gate, np)
        fused = m.round_bf16(r+y*gate, np)
        self.assertNotEqual(m.bf16_bytes(u, np), m.bf16_bytes(fused, np))
        packet = {'r': r, 'Y': y, 'gate': gate, 'u': u}
        m.neutral_postwrite(packet, np)
        packet['u'] = fused
        with self.assertRaises(ValueError):
            m.neutral_postwrite(packet, np)

    def test_source_equal_summary_counts_sources_and_rejects_partial_or_duplicate(self):
        summary = m.SourceEqualSummary(['a', 'b'], np)
        for source, value in [('a', 1.), ('b', 3.)]:
            for step in range(4):
                for block in range(40):
                    summary.add(source, step, block, [value]*len(m.metric_labels()))
        with self.assertRaises(ValueError):
            summary.add('a', 0, 0, [1.]*len(m.metric_labels()))
        result = summary.result()
        self.assertEqual(result['source_N'], 2)
        self.assertEqual(result['source_equal_all_positions'][0], 2)
        self.assertEqual(result['source_equal_all_positions_N'][0], 2)
        partial = m.SourceEqualSummary(['a'], np)
        with self.assertRaises(ValueError):
            partial.result()


if __name__ == '__main__':
    unittest.main()
