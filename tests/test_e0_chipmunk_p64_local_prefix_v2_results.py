import hashlib,json,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_p64_local_prefix_v2_results as r
class Test(unittest.TestCase):
 def test_registered_large_run_metadata_and_sha_gate(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);path=root/'RUN.json';obj={'padding':'x'*(4<<20)};data=json.dumps(obj).encode();path.write_bytes(data);binding={'path':'RUN.json','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
   with patch.object(r,'HF',root):
    self.assertEqual(r.run_metadata(binding),obj)
    binding['sha256']='0'*64
    with self.assertRaises(ValueError):r.run_metadata(binding)
 def test_pooled_energy_signed_cross_and_numeric_guard(self):
  g={k:0. for k in r.ENERGIES};g.update(count=2,weight_sum=4,vector_tolerance_excess_max=0,identity_within_tolerance=True,gate_identity_within_tolerance=True,L2=4,P2=9,T2=9,cross2=-4,dense2=16,gate_L2=4,gate_P2=9,gate_T2=9,gate_cross2=-4,gate_dense2=16)
  x=r.aggregate([g]);self.assertEqual(x['relative_L2']['L2'],.5);self.assertEqual(x['gate_cross_over_dense2'],-.25);self.assertEqual(x['weight_sum'],4)
  g['gate_identity_within_tolerance']=False
  with self.assertRaises(ValueError):r.aggregate([g])
 def test_local_shadow_and_free_dense_denominators_differ(self):
  g={k:0. for k in r.ENERGIES};g.update(count=1,weight_sum=1,vector_tolerance_excess_max=0,identity_within_tolerance=True,gate_identity_within_tolerance=True,L2=4,shadow2=16,dense2=4,gate_L2=4,gate_shadow2=16,gate_dense2=4)
  value=r.aggregate([g]);self.assertEqual(value['gate_relative_L2']['L2'],1);self.assertEqual(value['gate_local_relative_L2_over_student_shadow'],.5);self.assertEqual(value['gate_shadow_energy_over_dense_energy'],4)
 def test_reject_missing_site_coverage(self):
  with self.assertRaises(ValueError):r.collection([],'eval')
if __name__=='__main__':unittest.main()
