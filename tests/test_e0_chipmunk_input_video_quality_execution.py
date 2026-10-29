import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_input_video_quality_execution as ex
class Test(unittest.TestCase):
 def test_four_devices_and_disjoint_cpu_sets(self):
  self.assertEqual([ex.physical_gpu('quality',p) for p in range(4)],[0,1,2,3]);self.assertEqual([ex.physical_gpu('raft',p) for p in range(4)],[0,1,2,3]);groups=[set(ex.affinity(k,p)) for k in ('quality','raft','visual') for p in range(4)];self.assertTrue(all(not a&b for i,a in enumerate(groups) for b in groups[i+1:]))
 def test_invalid_part(self):
  with self.assertRaises(ValueError):ex.physical_gpu('quality',4)
