"""Read-only verification of actual admitted Dense proof and tamper detection."""
import json,unittest
from pathlib import Path
from quality_dense_proof import verify,sha
class ProofTest(unittest.TestCase):
 def test_admission_and_condition_mismatch(self):
  base=Path(__file__).resolve().parent;repo=base.parents[3]
  admission=repo/'research/experiment_reviews/FFNDS_STAGE4_DENSE_ADMISSION_V1/DENSE_ADMISSION.json';receipt=admission.with_name('AUDIT_RECEIPT.json');contract=base/'QUALITY_PAIRED_CONTRACT.json'
  data=json.loads(admission.read_text(encoding='utf8'));row=next(x for x in data['slots'] if x['source']=='vbi_1013')
  identity={**row['input_identity'],'initial_noise':row['initial_noise'],'scheduler':row['scheduler']}
  record={'arm':'Dense','source':row['source'],'seed':42,'video_sha256':row['video']['sha256'],'video_path':row['video']['path'],'image_sha256':identity['image_sha256'],'prompt_sha256':identity['prompt_sha256'],'source_cluster_id':identity['source_cluster_id'],'input_identity':identity}
  scope={'paired_contract_path':str(contract),'paired_contract_sha256':sha(contract),'dense_admission_path':str(admission),'dense_admission_sha256':sha(admission),'dense_admission_receipt_path':str(receipt),'dense_admission_receipt_sha256':sha(receipt)}
  self.assertEqual(verify(record,scope)['status'],'PASS')
  record['input_identity']=dict(identity,seed=43)
  with self.assertRaises(ValueError):verify(record,scope)
  record['arm']='ICC'
  with self.assertRaises(NotImplementedError):verify(record,scope)
if __name__=='__main__':unittest.main()
