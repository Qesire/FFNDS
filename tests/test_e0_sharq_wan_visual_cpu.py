import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_visual_cpu as v
class Tests(unittest.TestCase):
 def test_real_per_source_budgets(self):
  self.assertEqual(v.LIMITS['videos'],3);self.assertEqual(v.LIMITS['successful_decoded_frames'],243);self.assertEqual(v.LIMITS['JPEG_pages'],6);self.assertEqual(v.LIMITS['model_calls'],0)
 def test_unclosed_parent_refused(self):
  with patch.object(v.line,'parent_data',side_effect=ValueError('not closed')):
   with self.assertRaisesRegex(ValueError,'not closed'):v.draft(Path('/tmp'),0)
if __name__=='__main__':unittest.main()
