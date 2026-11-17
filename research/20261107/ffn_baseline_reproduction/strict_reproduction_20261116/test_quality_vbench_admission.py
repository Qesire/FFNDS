"""Batch admission tests; fail before any metric model import."""
import hashlib,tempfile,unittest
from pathlib import Path
from quality_vbench_batch import run
class AdmissionTest(unittest.TestCase):
 def test_camera_not_applicable_and_alias_rejected(self):
  scope={'comparison':'main','metric_seed':20261103,'dimension':'camera_motion','records':[{'source':'vbi_1013','arm':'Dense','seed':42,'video_sha256':'unused','video_path':'unused'}]}
  with self.assertRaises(ValueError):run(scope,scope_sha256='x',authorize=lambda s:None,verify_generation=lambda r,s:None,check_resources=lambda:None)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);v=p/'v';i=p/'i';v.write_bytes(b'v');i.write_bytes(b'i')
   sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();proof='a'*64
   r={'source':'vbi_1013','arm':'Dense','seed':42,'video_path':str(v),'video_sha256':sha(v),'image_path':str(i),'image_sha256':sha(i), 'prompt':'p','prompt_sha256':hashlib.sha256(b'p').hexdigest(),'input_identity':'id','source_cluster_id':'group','generation_proof_sha256':proof}
   q=dict(r,arm='ICC');scope={'comparison':'main','metric_seed':20261103,'dimension':'imaging_quality','records':[r,q]}
   with self.assertRaises(ValueError):run(scope,scope_sha256='x',authorize=lambda s:None,verify_generation=lambda r,s:{'status':'PASS','proof_sha256':proof},check_resources=lambda:None)
if __name__=='__main__':unittest.main()


