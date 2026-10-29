import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_all160_video_plan as p
class Tests(unittest.TestCase):
 def test_source_and_complete_budget_are_unregistered(self):
  a,b=p.plan(0),p.plan(1)
  self.assertEqual([a['case']['native_source_ordinal'],b['case']['native_source_ordinal']],[1040,979]);self.assertFalse(a['model_run_authorized']);self.assertEqual(a['scope']['original_block_prefetch_observations'],648);self.assertEqual(a['scope']['sparse_gemms'],320);self.assertEqual(a['scope']['dense_residual_fullK_gemms'],320);self.assertEqual(len(a['models']),2)
 def test_wrong_part_rejected(self):
  with self.assertRaises(ValueError):p.plan(2)
if __name__=='__main__':unittest.main()
