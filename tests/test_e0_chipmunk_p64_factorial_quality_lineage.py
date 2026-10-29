import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_p64_factorial_quality_lineage as x
class LineageTest(unittest.TestCase):
 def sample(self):
  state={'sha256':'a'*64,'bytes':1,'raw_tensor_sha256':'b'*64,'shape':[1],'dtype':'BF16'};states={'raw_noise_FP32':state,'initial_noise':state}
  phases=[{'mode':m,'video':{'path':'/home/xiang/.cache/huggingface/factorial_'+m+'.mp4','sha256':'c'*64 if m=='none' else m,'bytes':3},'states':states,'run_id':m} for m in x.MODES[:3]]
  old=[{'ordinal':898,'mode':m,'video':{'path':'old_'+m,'sha256':'c'*64 if m=='original_dense' else 'd'*64,'bytes':3},'states':states} for m in x.prior.MODES]
  return phases,old
 def test_valid_and_wrong_none(self):
  p,o=self.sample();a=x.assemble(0,p,o,{},{});self.assertEqual([r['mode'] for r in a],list(x.MODES));self.assertTrue(a[3]['reused_previous_video']);p[0]['video']['sha256']='bad'
  with self.assertRaises(ValueError):x.assemble(0,p,o,{},{})
 def test_no_duplicate_modes(self):
  p,o=self.sample();p[2]['mode']='high_only'
  with self.assertRaises(ValueError):x.assemble(0,p,o,{},{})
if __name__=='__main__':unittest.main()
