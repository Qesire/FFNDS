import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
s=importlib.util.spec_from_file_location('inc',Path(__file__).resolve().parents[1]/'tools/ub1_disjoint20_latent_increments.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_bf16_decode_and_corrupt(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a';data=np.array([0x3f80,0xc000,0,0x3f00],dtype='<u2').tobytes();header=json.dumps({'latents':{'dtype':'BF16','shape':[1,2,1,2],'data_offsets':[0,len(data)]}}).encode();p.write_bytes(len(header).to_bytes(8,'little')+header+data)
   out=m.decode_bf16(p,(1,2,1,2),hashlib.sha256(data).hexdigest());np.testing.assert_array_equal(out,np.array([1,-2,0,.5],dtype=np.float32).reshape(1,2,1,2))
   with self.assertRaises(ValueError):m.decode_bf16(p,(1,2,1,2),'0'*64)
 def test_post_roundtrip(self):
  good=np.array([1,-2,0,.5],dtype=np.float32).reshape(1,2,1,2);self.assertIs(m.verify_bf16_post(good,good.shape),good)
  bad=good.copy();bad.flat[0]=np.float32(1.0001)
  with self.assertRaises(ValueError):m.verify_bf16_post(bad,bad.shape)
 def test_signed_increment_and_time_projection(self):
  dp=np.zeros((1,2,1,1));cp=np.array([[[[1]],[[2]]]],dtype=float);dn=np.zeros_like(dp);cn=np.array([[[[0]],[[4]]]],dtype=float);r=m.increment_stats(dp,cp,dn,cn,.5);a=r['identity'];self.assertEqual(a['previous_error_energy'],5);self.assertEqual(a['increment_difference_energy'],5);self.assertEqual(a['twice_previous_increment_dot'],6);self.assertEqual(a['current_error_energy'],16);self.assertEqual(a['effective_velocity_from_state_difference_norm'],np.sqrt(5)/.5);self.assertEqual(r['latent_time_first_difference']['previous_error_energy'],1);self.assertEqual(r['latent_time_first_difference']['increment_difference_energy'],9)
  with self.assertRaises(ValueError):m.increment_stats(dp,cp,dn,cn,0)
 def test_source_weighting_not_step_count(self):
  def row(i,v):
   x={k:v for k in ('previous_error_energy','increment_difference_energy','twice_previous_increment_dot','current_error_energy','increment_difference_norm','effective_velocity_from_state_difference_norm')};return {'index':i,'steps':[{'statistics':{'identity':x,'latent_time_first_difference':x}} for _ in range(4)]}
  r=m.summarize([row(0,2.),row(10,8.)]);self.assertEqual(r['all']['source_count'],2);self.assertEqual(r['all']['steps'][0]['identity']['current_error_energy'],5.)
 def test_external_address_limit_contract(self):
  m.validate_address_limit(1<<30,1<<30)
  for soft,hard in [(-1,-1),(1<<30,2<<30),(2<<30,2<<30)]:
   with self.assertRaises(ValueError):m.validate_address_limit(soft,hard)
 def test_preflight_failure_persists_and_cannot_reuse_attempt(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);plan={'output_HF_relative':'fresh/attempt1'}
   def failing_preflight(result,out,check,save,started):raise ValueError('missing bound input')
   with self.assertRaises(ValueError):m.guarded_attempt(root,plan,'digest',failing_preflight)
   out=root/'fresh/attempt1';self.assertTrue((out/'ATTEMPT.json').is_file());run=json.loads((out/'RUN.json').read_text());self.assertEqual(run['status'],'FAILED_NO_RETRY');self.assertEqual(run['error_type'],'ValueError')
   with self.assertRaises(FileExistsError):m.guarded_attempt(root,plan,'digest',failing_preflight)
 def test_preflight_rejects_unbound_sources(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):m.preflight({'repo_sources':[],'inputs':[],'cases':[],'producers':[]},Path(d))
if __name__=='__main__':unittest.main()
