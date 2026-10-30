import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import chipmunk_p64_sync_abba_contract as c
class Test(unittest.TestCase):
 def test_balanced_mirror_and_artifacts(self):
  a=c.schedule(0);b=c.schedule(1);self.assertTrue(all(x[0]!=y[0] for x,y in zip(a[3:],b[3:])));self.assertEqual(c.expected_counts(0)['verification_artifacts'],79);self.assertEqual(c.expected_counts(1)['dense_FFN_GEMMs'],1760);self.assertEqual(c.expected_counts(1)['deferred_completion_barriers'],120)
 def test_part_rejected(self):
  with self.assertRaises(ValueError):c.schedule(2)
if __name__=='__main__':unittest.main()
