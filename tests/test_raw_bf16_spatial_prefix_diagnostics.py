import unittest
import numpy as np
from tools import raw_bf16_spatial_prefix_diagnostics as m


class PrefixGeometryControls(unittest.TestCase):
    def test_signed_cancellation_separates_state_from_local_error(self):
        t = np.ones((20, 3, 2))
        r = m.component_geometry(t, .7*t, 1.1*t, 1.2*t, np)
        for key, expected in (('teacher_local_error_fraction', .04), ('student_local_error_fraction', .16),
                              ('prefix_error_fraction', .09), ('twice_local_prefix_cross_fraction', -.24), ('total_error_fraction', .01)):
            self.assertAlmostEqual(r[key], expected)
        self.assertAlmostEqual(r['energy_identity_signed_remainder_fraction'], 0)
        self.assertAlmostEqual(r['interaction_error_fraction'], .04)
        self.assertAlmostEqual(r['L_P_I_energy_signed_remainder_fraction'], 0)

    def test_different_state_gate_and_rounded_residual_still_telescope(self):
        t = np.array([2., -4., 8.]); s = np.array([3., -2., 7.])
        ct = np.array([2.5, -3., 8.5]); cs = np.array([2.75, -2.5, 7.5])
        r = m.component_geometry(t, s, cs, ct, np)
        self.assertAlmostEqual(r['L_P_I_energy_signed_remainder_fraction'], 0)
        self.assertAlmostEqual(r['energy_identity_signed_remainder_fraction'], 0)
        self.assertGreater(r['interaction_error_fraction'], 0)

    def test_frequency_energy_preservation_cannot_hide_phase_flip(self):
        t = np.arange(21, dtype=float)
        x = np.broadcast_to(np.sin(2*np.pi*4*t/21)[:, None, None], (21, 3, 2))
        r = m.temporal_views(x, x, -x, -x, {'roi': [0, 1, 2]}, np)['all21']['roi']['frequency']['mid_3_6']
        self.assertAlmostEqual(r['candidate_energy_ratio'], 1)
        self.assertAlmostEqual(r['total_error_fraction'], 4)
        self.assertAlmostEqual(r['candidate_complex_coherence_real'], -1)
        self.assertAlmostEqual(abs(r['candidate_phase_radians']), np.pi)

    def test_opposite_frequency_phases_are_visible_after_band_cross_cancels(self):
        t = np.arange(21.)
        x = (np.sin(2*np.pi*t/21)+np.sin(4*np.pi*t/21))[:, None, None]
        y = (np.cos(2*np.pi*t/21)-np.cos(4*np.pi*t/21))[:, None, None]
        r = m.temporal_views(x, x, y, y, {'roi': [0]}, np)['all21']['roi']
        self.assertIsNone(r['frequency']['low_1_2']['candidate_phase_radians'])
        self.assertAlmostEqual(abs(r['frequency_bins']['1']['candidate_phase_radians']), np.pi/2)
        self.assertAlmostEqual(abs(r['frequency_bins']['2']['candidate_phase_radians']), np.pi/2)
        self.assertLess(r['frequency_bins']['1']['candidate_phase_radians']*r['frequency_bins']['2']['candidate_phase_radians'], 0)

    def test_tau0_is_not_mixed_into_later_innovation(self):
        x = np.ones((21, 3, 2)); candidate = x.copy(); candidate[0] = 2
        r = m.temporal_views(x, x, candidate, candidate, {'roi': [0, 1, 2]}, np)
        self.assertEqual(r['tau1_to20']['roi']['total']['total_error_fraction'], 0)
        self.assertEqual(r['tau0']['roi']['total']['total_error_fraction'], 1)
        self.assertIsNone(r['tau1_to20']['roi']['AC']['total_error_fraction'])

    def test_spatial_replication_does_not_change_relative_decomposition(self):
        x = np.arange(21, dtype=float)[:, None, None]+1
        one = m.temporal_views(x, .9*x, 1.1*x, 1.05*x, {'roi': [0]}, np, spectrum=False)
        many = [np.broadcast_to(a, (21, 8, 1)) for a in (x, .9*x, 1.1*x, 1.05*x)]
        eight = m.temporal_views(*many, {'roi': list(range(8))}, np, spectrum=False)
        for view in one:
            for component in ('total', 'DC', 'AC'):
                self.assertAlmostEqual(one[view]['roi'][component]['total_error_fraction'], eight[view]['roi'][component]['total_error_fraction'])

    def test_mismatched_operands_and_duplicate_spatial_points_fail(self):
        x = np.ones((21, 3, 2))
        with self.assertRaises(ValueError): m.temporal_views(x, x[:20], x, x, {'roi': [0]}, np)
        with self.assertRaises(ValueError): m.temporal_views(x, x, x, x, {'roi': [0, 0]}, np)

    def test_saved_native_anchor_includes_signed_small_m_rounding_bridge(self):
        rt = np.broadcast_to(np.arange(21.)[:, None, None]+1, (21, 52, 2))
        rc = .9*rt; nt = 1.01*rt; nc = 1.02*rc
        result = m.native_rounding_bridge(nt, nc, rt, rc, {'roi': list(range(52))}, np)
        for view in result.values():
            for part in view['roi'].values():
                if part['energy_signed_remainder_fraction'] is not None:
                    self.assertAlmostEqual(part['energy_signed_remainder_fraction'], 0)
        self.assertGreater(result['all21']['roi']['total']['combined_shape_bridge_fraction'], 0)


if __name__ == '__main__': unittest.main()
