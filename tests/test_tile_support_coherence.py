"""Verify pair design weights against enumeration, including empty sampled tiles."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from tile_support_coherence import pair_populations,transfer_statistics


class TileSupportTests(unittest.TestCase):
    def test_population_counts_match_explicit_pairs_across_frames_and_tail(self):
        expected={k:0 for k in ['within_tile','other_tile_same_frame','all_other_tiles']}
        for i in range(12):
            for j in range(12):
                if i==j:continue
                if i//5==j//5:expected['within_tile']+=1
                else:
                    expected['all_other_tiles']+=1
                    if i//4==j//4:expected['other_tile_same_frame']+=1
        self.assertEqual(pair_populations(12,5,4),expected)

    def test_exact_average_over_all_samples_is_unbiased_for_census(self):
        rng=np.random.default_rng(17);p=rng.random((8,3));p/=p.sum(1,keepdims=True)
        full=transfer_statistics(p,np.arange(8),8,3,4,np.array([0]),1)
        estimates={k:[] for k in full['subsets']}
        for sample in itertools.combinations(range(8),3):
            ids=np.array(sample);value=transfer_statistics(p[ids],ids,8,3,4,np.array([0]),1)
            for name in estimates:estimates[name].append(value['subsets'][name]['HT_gain_mean'])
        for name,values in estimates.items():
            self.assertAlmostEqual(np.mean(values),full['subsets'][name]['HT_gain_mean'],places=12)
        self.assertTrue(any(transfer_statistics(p[np.array(s)],np.array(s),8,3,4,np.array([0]),1)['subsets']['within_tile']['sample_ordered_pairs']==0 for s in itertools.combinations(range(8),3)))

    def test_duplicate_rows_and_non_normalized_profiles_are_rejected(self):
        p=np.ones((3,4))/4
        with self.assertRaises(ValueError):transfer_statistics(p,np.array([0,0,2]),8,3,4,np.array([0]),1)
        with self.assertRaises(ValueError):transfer_statistics(p*2,np.arange(3),8,3,4,np.array([0]),1)


if __name__=='__main__':unittest.main()
