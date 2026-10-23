"""Scientific counterexamples, source leakage and sampling coverage controls."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import analyze_prd_successor as a


class PRDSuccessorTests(unittest.TestCase):
    def test_native_receipts_reproduce(self):
        observed=a.analyze();saved=a.read(a.OUT)
        self.assertTrue(a.equivalent(observed,saved))
        changed=copy.deepcopy(saved)
        changed['saved_native_reanalysis']['C2_pooled']['scaled_capture']+=.001
        self.assertFalse(a.equivalent(observed,changed))

    def test_nextnorm_ranking_can_reverse_after_residual_bypass(self):
        r = a.mathematical_checks()['residual_bypass_rank_reversal']
        self.assertLess(r['nextnorm_radial_loss'], r['nextnorm_tangent_loss'])
        self.assertGreater(r['residual_block_radial_loss'],r['residual_block_tangent_loss'])

    def test_mean_shift_and_epsilon_finite_identity(self):
        x=[0.,1.,-3.,2.]; y=[v+7 for v in x]
        self.assertLess(a.sse(a.norm(x),a.norm(y)),1e-20)
        for eps in (1e-6,.1,1.):
            y=[2*v for v in x]
            self.assertAlmostEqual(a.finite_norm_distance(x,y,eps),a.sse(a.norm(x,eps),a.norm(y,eps)),places=12)

    def test_parent_scene_and_seed_do_not_make_new_independent_sources(self):
        m=[{'image_sha256':'A','scene_family':'scene1','split':'fit'},
           {'image_sha256':'B','scene_family':'scene2','split':'test'}]
        self.assertTrue(a.split_is_disjoint(m))
        leaked=copy.deepcopy(m); leaked[1]['scene_family']='scene1'
        self.assertFalse(a.split_is_disjoint(leaked))
        leaked=copy.deepcopy(m); leaked[1]['image_sha256']='A'
        self.assertFalse(a.split_is_disjoint(leaked))

    def test_rotated_coverage_and_balanced_prefix(self):
        s=a.coverage_schedule()
        self.assertEqual(len(s),96)
        self.assertEqual(set(Counter((r['step'],r['depth_band']) for r in s).values()),{6})
        self.assertEqual(len({(r['source_slot'],r['seed_slot'],r['step']) for r in s}),96)
        self.assertEqual({r['block']//10 for r in s},{0,1,2,3})
        for i in range(0,len(s),3): self.assertEqual({r['stratum'] for r in s[i:i+3]},{0,1,2})

    def test_camera_image_command_crossing(self):
        b=a.camera_blocks(); counts=Counter(c for r in b for c in r['commands'])
        pairs=Counter(tuple(sorted((x,y))) for r in b for x in r['commands'] for y in r['commands'] if x<y)
        self.assertEqual(set(counts.values()),{6});self.assertEqual(len(pairs),21)
        self.assertEqual(set(pairs.values()),{2})

    def test_sample_size_tracks_effect_variance_and_multiplicity(self):
        n=a.approximate_sources(1,.5,.3)
        self.assertGreater(a.approximate_sources(2,.5,.3),n)
        self.assertGreater(a.approximate_sources(1,.25,.3),n)
        self.assertGreater(a.approximate_sources(1,.5,.3,primary_count=3),n)
        with self.assertRaises(ValueError): a.approximate_sources(0,.5,.3)

    def test_retired_routes_stay_ineligible_and_history_keeps_status(self):
        import check_research_routes as guard
        d,t,_,_=guard.load_records(ROOT)
        for mid in d['retired_research_method_ids']:
            m=next(x for x in t['methods'] if x['id']==mid)
            self.assertEqual(m['research_disposition'],'RETIRED_AS_ACTIVE_RESEARCH')
            self.assertFalse(guard.method_eligibility(d,t,mid)[0])
            for alias in m.get('aliases',[]): self.assertFalse(guard.method_eligibility(d,t,alias)[0])
        self.assertFalse(set(d['eligible_model_method_ids']) & set(d['retired_research_method_ids']))
        self.assertEqual(d['latest_fresh_validation']['actual_debits']['metric_evaluator_calls'],0)
        self.assertEqual(d['latest_c1_native_diagnostic']['completed_finite_scopes'],72)


if __name__=='__main__': unittest.main()
