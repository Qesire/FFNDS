import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import chipmunk_input_video_driver as d
class Test(unittest.TestCase):
 def test_modes(self):self.assertEqual(d.MODES,('none','h_cache_dense','x_cache_dense','high_h_readonly_p64','high_x_recompute_p64'))
 def test_wrong_count_rejected_before_model(self):
  with self.assertRaisesRegex(ValueError,'five distinct'):
   d.execute(torch=None,pipe=None,guard=None,vm=None,native=None,run_ids=[],image_path=None,prompt='',output_paths=[],capture_factory=None,inspect_video=None,check=None,authorize=None,gate=None,progress=None,kernels=None,emit=None)
