import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_metric_adapter as adapter
import e0_sharq_wan_visual_cpu as visual
import e0_sharq_wan_metric_verify as verify
class Tests(unittest.TestCase):
 def test_direct_old_closed_call_is_denied(self):
  for e in (adapter.install('quality'),adapter.install('raft'),visual):
   with self.assertRaisesRegex(ValueError,'independent registered verifier'):e.verify_closed(SimpleNamespace())
 def test_verifier_resources_are_separate_finite_cpu(self):
  self.assertEqual(verify.RES['GPU'],0);self.assertEqual(verify.RES['CPU_threads'],1);self.assertEqual(verify.RES['whole_file_sha_bytes_max'],16<<30);self.assertFalse(verify.RES['automatic_retry'])
if __name__=='__main__':unittest.main()
