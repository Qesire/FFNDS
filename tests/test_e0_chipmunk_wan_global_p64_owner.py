import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_wan_global_p64_owner as owner
class Test(unittest.TestCase):
 def test_frozen_scope_resources_and_fixed_cost(self):
  d=owner.limits();self.assertEqual(owner.RES['GPU'],1);self.assertEqual(owner.RES['new_HF_bytes'],512<<20);self.assertEqual(owner.DESIGN['source_ordinals'],[898,96]);self.assertEqual(len(owner.run_ids(0)),4)
  self.assertEqual(d['native_FFN_calls'],640);self.assertEqual(d['native_FFN_linears'],1280);self.assertEqual(d['p64_score'],80);self.assertEqual(d['p64_W1_gemm'],80);self.assertEqual(d['observer_gate_product'],2);self.assertEqual(d['packed_W1_gemm'],80);self.assertEqual(d['packed_W2_gemm_fp32'],80);self.assertEqual(d['cache_h_clone'],160);self.assertEqual(d['cache_release_sync'],160)
 def test_scope_identity_and_limits(self):
  with self.assertRaises(ValueError):owner.sid(2)
  self.assertEqual(owner.DESIGN['cache_per_expert_logical_bytes'],40*31668*(13824+5120)*2)
if __name__=='__main__':unittest.main()
