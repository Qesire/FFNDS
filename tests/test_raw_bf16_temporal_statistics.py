"""Synthetic CPU-only S2 time-statistics tests. No saved data/model/GPU."""
import unittest
import numpy as np

from tools import raw_bf16_temporal_statistics as s


class TemporalStatisticsTests(unittest.TestCase):
    def test_constant_and_zero_AC_are_not_false_correlation(self):
        x = np.ones((21, 8, 3), dtype=float)*5
        r = s.summarize(x,np)
        self.assertEqual(r["AC_fraction_of_total_signal_energy"], 0)
        self.assertIsNone(r["lag"][0]["centered_correlation"])
        self.assertEqual(r["events"]["nonzero_differences"],0)
        self.assertIsNone(r["temporal_gram"]["rank95"])
        self.assertIsNone(r["spectrum"]["band_fractions"]["high_7_plus"])

    def test_low_and_high_frequency_are_separated_and_parseval_holds(self):
        t=np.arange(21,dtype=float)
        lo=np.broadcast_to(np.sin(2*np.pi*2*t/21)[:,None,None],(21,8,3))
        hi=np.broadcast_to(np.sin(2*np.pi*9*t/21)[:,None,None],(21,8,3))
        l=s.summarize(lo,np)
        h=s.summarize(hi,np)
        self.assertAlmostEqual(l["spectrum"]["band_fractions"]["low_1_2"],1,places=11)
        self.assertAlmostEqual(h["spectrum"]["band_fractions"]["high_7_plus"],1,places=11)
        self.assertAlmostEqual(sum(v for v in l["spectrum"]["bin_fraction"] if v is not None),1,places=10)
        self.assertGreater(l["lag"][0]["centered_correlation"],0)
        self.assertLess(h["lag"][0]["centered_correlation"],0)
        self.assertGreater(l["spectrum"]["AC_energy"],0)

    def test_effective_temporal_rank_one_and_two(self):
        t=np.linspace(-1,1,21)
        x=np.stack((t,2*t,-t),axis=-1)[:,None,:]
        x=np.broadcast_to(x,(21,8,3))
        out=s.temporal_gram(x,np)
        self.assertEqual(out["rank95"],1)
        self.assertAlmostEqual(out["effective_rank_participation"],1,places=9)
        self.assertAlmostEqual(out["effective_rank_entropy"],1,places=9)

    def test_flip_preserves_frequency_energy_but_reverses_direction(self):
        t=np.arange(21,dtype=float)
        x=np.broadcast_to(np.sin(2*np.pi*4*t/21)[:,None,None],(21,8,2))
        result=s.compare(x,-x,np)["bands"]["mid_3_6"]
        self.assertAlmostEqual(result["energy_ratio"],1,places=12)
        self.assertAlmostEqual(result["relative_error"],4,places=12)
        self.assertAlmostEqual(result["signed_gain"],-1,places=12)
        self.assertAlmostEqual(result["complex_coherence_real"],-1,places=12)
        identical=s.compare(x,x,np)["bands"]["mid_3_6"]
        self.assertAlmostEqual(identical["relative_error"],0)
        self.assertAlmostEqual(identical["signed_gain"],1)

    def test_even_T_nyquist_and_tau0_exclusion(self):
        t=np.arange(20)
        x=np.broadcast_to((-1.)**t[:,None,None],(20,8,1))
        out=s.summarize(x,np)
        self.assertEqual(out["geometry"]["T"],20)
        self.assertAlmostEqual(out["spectrum"]["band_fractions"]["high_7_plus"],1,places=12)
        self.assertEqual(len(out["lag"]),10)
        self.assertEqual(out["lag"][9]["eligible_temporal_pairs_per_anchor"],10)

    def test_nonstationary_event_and_validation(self):
        x=np.zeros((21,8,1),float)
        x[11:]=1.
        r=s.summarize(x,np)
        self.assertEqual(r["events"]["nonzero_differences"],1)
        self.assertGreater(r["events"]["first_difference_peak_to_mean"],10)
        self.assertGreater(r["events"]["half_window_mean_shift_over_total_AC"],0)
        with self.assertRaises(ValueError):
            s.summarize(np.full((21,8,3),np.nan),np)
        with self.assertRaises(ValueError):
            s.compare(x,np.zeros((20,8,1)),np)


if __name__ == "__main__":
    unittest.main()
