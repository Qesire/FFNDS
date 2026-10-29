"""ALG controls: centering/differencing alone can create negative correlations."""
import unittest
import numpy as np
from tools import raw_bf16_temporal_statistics as s


class OrthogonalReferenceTests(unittest.TestCase):
    def test_centered_orthogonal_time_is_not_a_motion_reversal(self):
        t = 20
        x = np.broadcast_to(np.eye(t)[:, None, :], (t, 8, t))
        result = s.summarize(x, np)
        for row in result['lag']:
            self.assertAlmostEqual(row['centered_correlation'], -1/(t-1), places=12)
        self.assertAlmostEqual(result['events']['successive_difference_direction_cosine'], -.5, places=12)
        self.assertAlmostEqual(result['temporal_gram']['effective_rank_participation'], 19, places=10)
        self.assertEqual(result['temporal_gram']['rank95'], 19)

    def test_odd_and_even_fft_band_reference_accounts_for_nyquist(self):
        for t in (20, 21):
            x = np.broadcast_to(np.eye(t)[:, None, :], (t, 8, t))
            bands = s.spectrum(x, np)['band_fractions']
            self.assertAlmostEqual(bands['low_1_2'], 4/(t-1), places=12)
            self.assertAlmostEqual(bands['mid_3_6'], 8/(t-1), places=12)
            self.assertAlmostEqual(bands['high_7_plus'], (t-13)/(t-1), places=12)

    def test_large_static_component_does_not_make_innovations_predictable(self):
        x = np.broadcast_to(np.eye(20)[:, None, :], (20, 8, 20))
        original, shifted = s.summarize(x, np), s.summarize(x+100, np)
        self.assertLess(shifted['AC_fraction_of_total_signal_energy'], .00001)
        self.assertAlmostEqual(original['lag'][0]['centered_correlation'], shifted['lag'][0]['centered_correlation'], places=10)
        self.assertAlmostEqual(original['temporal_gram']['effective_rank_participation'], shifted['temporal_gram']['effective_rank_participation'], places=10)

    def test_historical_half_window_field_has_an_explicit_anchor_factor(self):
        # Preserve the executed kernel; audit its field rather than silently
        # rewriting old results. Divide by P for a replication-invariant view.
        x = np.zeros((20, 1, 1)); x[10:] = 1
        one = s.temporal_events(x, np)['half_window_mean_shift_over_total_AC']
        eight = s.temporal_events(np.broadcast_to(x, (20, 8, 1)), np)['half_window_mean_shift_over_total_AC']
        self.assertAlmostEqual(eight/one, 8)
        self.assertAlmostEqual(eight/8, one)


if __name__ == '__main__': unittest.main()
