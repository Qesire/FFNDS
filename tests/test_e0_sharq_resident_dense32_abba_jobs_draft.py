import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_resident_dense32_abba_jobs_draft as x
class JobsTest(unittest.TestCase):
 def test_three_specs_and_cpu_verify_isolation(self):
  p={'id':'test','status':'FROZEN_PENDING_REGISTRATION','assets_finalized':True,'execution_private_worker_config_sha256':'sha','cpu_affinity':[0,1,2,3],'output_HF_relative':'test'};paths={k:Path('/tmp')/(k+'.json') for k in ('cpu','gpu','verify')}
  with patch.object(x.owner,'validate'),patch.object(x,'sha',return_value='sha'):
   j=x.jobs(p,Path('/tmp/protocol'),Path('/tmp/profile'),Path('/tmp/local'),0,'GPU-fake',paths)
  self.assertEqual(set(j),{'cpu','gpu','verify'});self.assertIn('CUDA_VISIBLE_DEVICES=',j['verify']['argv']);self.assertEqual(j['verify']['id'],'test_VERIFY');self.assertIn('-I',j['cpu']['argv'])
if __name__=='__main__':unittest.main()
