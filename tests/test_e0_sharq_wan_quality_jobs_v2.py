import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_quality_jobs_v2 as m
class Tests(unittest.TestCase):
 def test_exact_gpu_mapping_and_closed_required(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);profile=root/'profile';profile.write_bytes(b'profile');protocol=root/'protocol';protocol.write_bytes(b'protocol');paths={k:root/(k+'.json') for k in ('cpu','gpu','verify')};p={'id':'scope','shard':0,'status':'FROZEN_PENDING_REGISTRATION','assets_finalized':True,'execution_private_worker_config_sha256':m.execution.PROFILE_SHA,'execution_device':{'physical_gpu_index':1},'output_HF_relative':'scope/worker/results','import_environment':{'python_HF_relative':'python'},'CPU_preflight_resources':{'outer_seconds':1020},'resources':{'outer_seconds':3900}}
   with patch.object(m.execution,'PROFILE_SHA',m.sha(profile)):
    p['execution_private_worker_config_sha256']=m.execution.PROFILE_SHA;j=m.jobs('quality',p,protocol,profile,'GPU-example',paths);self.assertEqual(j['gpu']['argv'][8:10],['1','GPU-example']);self.assertEqual(j['verify']['id'],'scope_VERIFY');self.assertEqual(j['cpu']['external_timeout_seconds'],1020)
    p['execution_device']['physical_gpu_index']=0
    with self.assertRaises(ValueError):m.jobs('quality',p,protocol,profile,'GPU-example',paths)
    p['status']='DRAFT_NOT_AUTHORIZED'
    with self.assertRaises(ValueError):m.jobs('quality',p,protocol,profile,'GPU-example',paths)
if __name__=='__main__':unittest.main()
