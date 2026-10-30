import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_student_observation_results as r
class Test(unittest.TestCase):
 def test_ratio_energy_and_signed_cross(self):
  p={k:[0.]*84 for k in r.KEYS};p['count']=[1]*84;p['pre_squared'][0]=4;p['dense_pre_squared'][0]=16;p['gated_squared'][0]=9;p['dense_gated_squared'][0]=36;p['write_squared'][0]=1;p['dense_write_squared'][0]=100;p['ideal_total'][0]=3;p['ideal_diagonal'][0]=5;p['ideal_cross'][0]=-2
  s=r.group_stats(p);a=s['groups'][0];self.assertEqual(a['pre_relative_L2'],.5);self.assertEqual(a['gated_relative_L2'],.5);self.assertEqual(a['write_relative_L2'],.1);self.assertEqual(a['ideal_cross'],-2);self.assertIsNone(s['groups'][1]['pre_relative_L2']);self.assertEqual(s['total']['pre_relative_L2'],.5)
 def test_nonfinite_and_missing_rejected(self):
  p={k:[0.]*84 for k in r.KEYS};p['ideal_cross'][2]=float('nan')
  with self.assertRaises(ValueError):r.group_stats(p)
  with self.assertRaises(ValueError):r.group_stats({})
 def test_v2_missing_proof_fails_before_output(self):
  import tempfile,json
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'config').mkdir();(root/'config/e0_chipmunk_student_observation_20261030_part0_pending_review_v2.json').write_text(json.dumps({'id':'test'}))
   with patch.object(r,'ROOT',root):
    with self.assertRaises(FileNotFoundError):r.main()
   self.assertFalse((root/'research/20261030/CHIPMUNK_STUDENT_OBSERVATION_RESULTS_20261030.json').exists())
if __name__=='__main__':unittest.main()
