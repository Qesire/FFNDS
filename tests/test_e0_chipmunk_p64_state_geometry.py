import sys,pathlib,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import e0_chipmunk_p64_state_geometry as m
class Tests(unittest.TestCase):
 def test_orthogonal_dc_time_space(self):
  a=np.zeros((1,2,1,2));b=np.array([[[[1.,3.]],[[3.,5.]]]])
  r=m.components(a,b);self.assertEqual((r['dc_energy'],r['latent_time_ac_energy'],r['spatial_ac_energy'],r['error_energy']),(36.,4.,4.,44.));self.assertEqual(r['closure_error'],0);self.assertIsNone(r['relative_l2']);self.assertTrue(r['zero_reference'])
 def test_signed_cross_report_and_axis_validation(self):
  rng=np.random.default_rng(4);a=rng.normal(size=(2,3,4,5));b=rng.normal(size=(2,3,4,5));r=m.components(a,b);self.assertLessEqual(abs(r['closure_error']),r['closure_tolerance']);self.assertAlmostEqual(r['relative_l2']**2,r['relative_sse'])
  with self.assertRaises(ValueError):m.components(a,b[:1])
 def test_partial_case_remains_visible_on_failure(self):
  report={'cases':[]};row=m.attach_case(report,{'source':898});row['steps'].append({'step':0,'flow':{'relative_sse':1.}})
  self.assertEqual(report['cases'][0]['steps'][0]['step'],0)
 def test_unlimited_memory_rejected(self):
  with self.assertRaises(ValueError):m.check_rlimit(-1,-1)
  m.check_rlimit(2<<30,2<<30)
 def test_no_registration_for_draft(self):
  with self.assertRaises(ValueError):m.authorize({'id':m.ID,'assets_finalized':False},'x')
if __name__=='__main__':unittest.main()
