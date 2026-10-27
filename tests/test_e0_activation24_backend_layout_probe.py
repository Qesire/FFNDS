import importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/e0_activation24_backend_layout_probe.py';s=importlib.util.spec_from_file_location('probe_c2',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class BackendProbeTests(unittest.TestCase):
 def test_search_tuple(self):
  self.assertEqual(m.search_parameters((3,2,1,7)),{'alg_id':3,'split_k':2,'split_k_mode':1,'max_alg_id':7})
  for v in [(1,0,0,2),(4,1,0,3),(1,2,3),(1,2,3,4.0)]:
   with self.assertRaises(ValueError):m.search_parameters(v)
 def test_output_layout_and_crop(self):
  import torch
  x=torch.arange(24).reshape(6,4)
  for transpose in (False,True):
   a=m.normalize_output(x.t() if transpose else x,transpose,5)
   self.assertTrue(torch.equal(a,x[:5]));self.assertTrue(a.is_contiguous())
 def test_charge_before_failed_api(self):
  counts={}
  def bad():m.charge(counts,'search_api');raise ValueError('failed')
  with self.assertRaises(ValueError):bad()
  self.assertEqual(counts['search_api'],1)
  counts['search_api']=8
  with self.assertRaises(RuntimeError):m.charge(counts,'search_api')
 def test_exact_authorization(self):
  p={'id':m.ID,'limits':dict(m.LIMITS),'shapes':[list(x) for x in m.SHAPES],'warmups':2,'repetitions':5,'wall_seconds':300,'assets_finalized':True}
  a={'experiment_authority':{'current_user_authorized_scopes':[{'id':m.ID,'protocol_sha256':'yes','execution_status':'AUTHORIZED_NOT_STARTED'}]}}
  m.authorize(p,'yes',a)
  with self.assertRaises(ValueError):m.authorize(p,'no',a)
