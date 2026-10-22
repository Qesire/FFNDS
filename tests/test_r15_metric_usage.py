"""Frozen scalar metric usage and synthetic mathematical counterexamples only."""
import math
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from verify_r15_metric_usage import (run,paired_mse,paired_psnr,spatial_mean_contrast,
                                     static_roi_temporal_mae,source_aggregation)

class MetricUsageTests(unittest.TestCase):
    def test_actual_saved_scalar_mappings_and_scopes(self):
        out=run()
        self.assertEqual(out['status'],'OFFLINE_METRIC_USAGE_PASS')
        self.assertEqual(out['observed_levels'],['L1','L5'])
        self.assertGreaterEqual(out['frozen_native_local']['numeric_observations_checked'],10)
        self.assertAlmostEqual(out['frozen_native_local']['dynamic_to_UB1_SSE_ratio'],1.31164172660792255)
        self.assertIsNone(out['synthetic_backend']['actual_wan_model_speedup'])
        self.assertIsNone(out['independent_video_quality'])
    def test_per_frame_psnr_versus_pooled_mse_psnr(self):
        mse=[paired_mse([0,0],[0,1]),paired_mse([0,0],[1,1])]
        self.assertEqual(mse,[.5,1.])
        self.assertGreater(sum(paired_psnr(x,1) for x in mse)/2,paired_psnr(sum(mse)/2,1))
        self.assertIsNone(paired_psnr(0,1))
    def test_input_domains(self):
        for a,b in (([],[]),([0],[0,1]),([float('nan')],[0])):
            with self.assertRaises(AssertionError): paired_mse(a,b)
        with self.assertRaises(AssertionError): paired_psnr(-1,1)
        with self.assertRaises(AssertionError): paired_psnr(0,0)
    def test_same_sse_different_spatial_structure(self):
        m=spatial_mean_contrast([1,1,1,1]);c=spatial_mean_contrast([1,-1,1,-1])
        self.assertEqual(m['SSE'],c['SSE'])
        self.assertEqual(m['mean_response_part'],4)
        self.assertEqual(c['within_stripe_part'],4)
    def test_static_lighting_roi_temporal_flicker(self):
        self.assertEqual(static_roi_temporal_mae([[.4,.4],[.4,.4]],[True,False]),0)
        self.assertAlmostEqual(static_roi_temporal_mae([[.4,.4],[.6,.4],[.4,.4]],[True,False]),.2)
        for frames,mask in (([[.4]],[True]),([[0],[0]],[False])):
            with self.assertRaises(AssertionError): static_roi_temporal_mae(frames,mask)
    def test_cluster_balanced_not_pooled(self):
        a=source_aggregation([1,100],[2,90])
        self.assertAlmostEqual(a['equal_source_mean_ratio'],1.45)
        self.assertNotAlmostEqual(a['equal_source_mean_ratio'],a['ratio_of_sums'])
        with self.assertRaises(AssertionError): source_aggregation([0,1],[.3,.2])
    def test_paired_pixel_metrics_do_not_require_neural_backbone(self):
        import json
        acad=json.loads((ROOT/'research/metric_system/academic_metric_families.json').read_text(encoding='utf8'))
        m={x['id']:x for x in acad['metrics']}
        for key in ('paired_mse','paired_psnr','paired_ssim','paired_ms_ssim'):
            self.assertNotIn('backbone',m[key]['required_protocol_fields'])
        self.assertIn('backbone',m['paired_lpips']['required_protocol_fields'])

    def test_synthetic_formula_not_official_i2v(self):
        a=run()
        self.assertTrue(a['synthetic_math_fixtures']['not_official_vbench'])
        self.assertEqual(a['capabilities']['official_nine_dimensional_i2v'],
                         'REQUIRES_GENERATED_VIDEOS_OFFICIAL_EVALUATOR')
        self.assertEqual(a['new_model_generations'],0)
if __name__=='__main__':unittest.main()
