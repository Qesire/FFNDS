"""Guard fusion arithmetic and the separation of latency from model quality."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_r14_backend import validate


class BackendRecordTests(unittest.TestCase):
    def setUp(self):
        self.result=json.loads((ROOT/'research/20261022/R14_three_direction_backend_results.json').read_text())
        self.protocol=json.loads((ROOT/'research/20261022/R14_three_direction_backend_protocol.json').read_text())

    def test_actual_samples_and_controls_reproduce(self):
        self.assertEqual(validate(self.result,self.protocol),[])

    def test_full_output_mismatch_prevents_claiming_fusion_correctness(self):
        result=copy.deepcopy(self.result)
        c=next(c for c in result['fusion_controls'] if c['name']=='full_pipeline/dynamic')
        c.update(numeric_exact=False,max_absolute_difference=.01)
        self.assertIn('Fusion changed declared arithmetic',validate(result,self.protocol))

    def test_selector_reduction_cannot_use_output_equality_tolerance(self):
        result=copy.deepcopy(self.result)
        c=next(c for c in result['fusion_controls'] if c['name']=='dynamic/score')
        c['max_relative_difference_clamped_at1']=1e-4
        self.assertIn('Norm reduction error exceeds declared tolerance',validate(result,self.protocol))

    def test_old_projection_cost_cannot_substitute_for_rewrite(self):
        result=copy.deepcopy(self.result);result['three_direction_projection_MAC_fraction']=.497677290780739
        self.assertIn('Reduced affine projection or extra W1 point omitted',validate(result,self.protocol))

    def test_synthetic_speed_is_not_model_quality(self):
        result=copy.deepcopy(self.result);result['native_model_or_quality_acceptance']=True
        self.assertIn('Synthetic timing is not video/model acceptance',validate(result,self.protocol))


if __name__=='__main__':
    unittest.main()
