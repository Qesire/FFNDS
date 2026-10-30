import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_input_video_quality_execution_visual_v2 as ex
class Test(unittest.TestCase):
 def test_visual_affinity(self):
  self.assertEqual(ex.affinity('visual',2),[12,13]);self.assertEqual(ex.affinity('visual',3),[14,15])
