import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_wan_fullm_raster_driver as m
class Tests(unittest.TestCase):
 def test_state_contract(self):self.assertEqual(len(m.STATE_NAMES),10);self.assertIn('flow_step3',m.STATE_NAMES)
 def test_reject_duplicate_before_model(self):
  with self.assertRaisesRegex(ValueError,'two fresh'):m.execute(torch=None,pipe=None,guard=None,vm=None,native=None,run_ids=['x','x'],image_path=None,prompt=None,output_paths=['a','b'],capture_factory=None,transaction=None,inspect_video=None,check=None,authorize=None,gate=None,progress=None)
if __name__=='__main__':unittest.main()
