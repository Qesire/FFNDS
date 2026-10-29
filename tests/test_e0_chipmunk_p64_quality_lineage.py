import pathlib,sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_p64_quality_lineage as line
class Tests(unittest.TestCase):
 def test_reject_unclosed_producer(self):
  with patch.object(line,'parent_entry',return_value={'protocol':{'path':'unused'}}),patch.object(line,'parent_owner',return_value=line.owner),patch.object(line.native,'read',return_value={}),patch.object(line.owner,'validate',side_effect=ValueError('producer not frozen')):
   with self.assertRaisesRegex(ValueError,'not frozen'):line.parent_data(pathlib.Path('/tmp'),0)
 def test_reject_changed_verifier_binding_before_read(self):
  with patch.object(line,'parent_data',return_value={'RUN':{'sha256':'actual'}}),patch.object(line.native,'read') as read:
   with self.assertRaises(Exception):line.closed_rows({'RUN':{'sha256':'wrong'}},pathlib.Path('/tmp'),0)
   read.assert_not_called()
 def test_verifier_accepts_only_v2_success_and_matching_id(self):
  line.check_verification({'status':'PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_P64_VIDEO_V2','id':'v2'},'v2')
  for status,identifier in [('PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_P64_VIDEO','v2'),('FAILED','v2'),('PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_P64_VIDEO_V2','wrong')]:
   with self.assertRaisesRegex(Exception,'actual V2'):line.check_verification({'status':status,'id':identifier},'v2')
 def test_pair_rejects_wrong_noise_identity(self):
  sample={'sha256':'a','bytes':5,'raw_tensor_sha256':'b','shape':[1],'dtype':'BF16'}
  states={'raw_noise_FP32':dict(sample),'initial_noise':dict(sample)}
  phases=[{'states':states},{},{},{'states':{'raw_noise_FP32':dict(sample),'initial_noise':dict(sample,sha256='other')}}]
  with self.assertRaises(Exception):line.paired_initial_noise(phases)
 def test_explicit_mapping_rejects_wrong_bound_sha(self):
  fake={'protocol':{'path':'a','bytes':1,'sha256':'0'*64},'owner_source':{'path':'b','bytes':1,'sha256':'1'*64}}
  with patch.object(line,'PARENT_PATHS',(fake,fake)),patch.object(line.native,'bound',side_effect=ValueError('sha mismatch')):
   with self.assertRaisesRegex(ValueError,'sha mismatch'):line.parent_entry(0)
 def test_reject_unbound_mapping(self):
  with patch.object(line,'PARENT_PATHS',None):
   with self.assertRaisesRegex(Exception,'NOT_BOUND'):line.parent_entry(0)
 def test_reject_wrong_part(self):
  with self.assertRaises(Exception):line.parent_data(pathlib.Path('/tmp'),2)
if __name__=='__main__':unittest.main()
