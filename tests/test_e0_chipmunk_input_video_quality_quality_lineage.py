import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_input_video_quality_quality_lineage as line
class Test(unittest.TestCase):
 def test_scope_and_parent_required(self):
  self.assertEqual(line.ORDINALS,(898,96,1040,979));self.assertEqual(line.MODES,('none','high_h_readonly_p64','high_x_recompute_p64'))
  with self.assertRaises(ValueError):line.parent_path(4)
 def test_reject_wrong_hf_before_any_parent(self):
  with self.assertRaisesRegex(ValueError,'HF differs'):line.parent_data('/tmp',0)
