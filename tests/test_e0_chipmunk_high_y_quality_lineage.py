import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_high_y_quality_lineage as m
class LineageTest(unittest.TestCase):
 def test_actual_three_arms_and_identity(self):
  with tempfile.TemporaryDirectory() as td:
   hf=Path(td);files=[]
   for i in range(3):
    p=hf/str(i);p.write_bytes(bytes([i]));files.append(p)
   state={'sha256':'s','bytes':1,'raw_tensor_sha256':'r','shape':[1],'dtype':'bf16'};states={'raw_noise_FP32':state,'initial_noise':state};case={'source_id':'x','prompt':'p','image':{'sha256':'i'}}
   phases=[{'mode':mode,'video':{'path':str(files[i]),'bytes':1,'sha256':str(i)},'states':states,'run_id':str(i)} for i,mode in enumerate(('none','high_y_cache_dense','high_y_reuse'))]
   old=[{'mode':mode,'ordinal':898,'case':case,'video':{'path':str(files[0].relative_to(hf)),'bytes':1,'sha256':'0'},'states':states,'run_id':'old'} for mode in m.factorial.MODES];old[1]['video']={'path':'oldhigh','bytes':1,'sha256':'h'}
   rows=m.assemble(0,phases,old,case,{'path':'r'},hf);self.assertEqual([x['mode'] for x in rows],list(m.MODES));self.assertTrue(rows[1]['reused_previous_video'])
   phases[0]['video']['sha256']='wrong'
   with self.assertRaises(Exception):m.assemble(0,phases,old,case,{'path':'r'},hf)
