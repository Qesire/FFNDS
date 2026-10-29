import unittest
from tools import recompute_s2_spatial_prefix_replay as m


class SourceEqualControls(unittest.TestCase):
    def test_eligible_positions_do_not_weight_source_cluster_means(self):
        a = m.mean_vectors([{'x': 0.}]*40, ['x'])
        b = m.mean_vectors([{'x': 10.}], ['x'])
        equal = m.equal_image_vectors([a, b])
        self.assertEqual(equal['mean'], [5.])
        self.assertEqual(equal['valid_independent_image_counts'], [2])

    def test_missing_energy_keeps_explicit_source_and_position_counts(self):
        a = m.mean_vectors([{'x': None}, {'x': None}], ['x'])
        b = m.mean_vectors([{'x': 2.}, {'x': None}], ['x'])
        self.assertEqual(a['mean'], [None]); self.assertEqual(b['valid_position_counts'], [1])
        self.assertEqual(m.equal_image_vectors([a, b])['valid_independent_image_counts'], [1])
        self.assertEqual(m.equal_image_vectors([a, b])['mean'], [2.])

    def test_phase_angles_are_not_averaged_as_linear_scalars(self):
        r = {key: {} for key in ('gY', 'u', 'rowLN_u', 'native_fullM_anchor_bridge', 'observed_prefix_input_relative_SSE', 'M1092_vs_fullM_Y_relative_SSE')}
        r['gY'] = {'candidate_phase_radians': 3.14, 'frequency_bins': {'1': {'candidate_phase_radians': -3.14}},
            'candidate_complex_coherence_real': -.9}
        self.assertEqual(m.scalar_metrics(r), {'gY/candidate_complex_coherence_real': -.9})


if __name__ == '__main__': unittest.main()
