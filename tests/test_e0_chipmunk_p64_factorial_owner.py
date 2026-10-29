import pathlib,sys,tempfile,unittest,json,struct,hashlib
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_p64_factorial_owner as m
class Tests(unittest.TestCase):
 def test_fixed_budget(self):
  x=m.limits();self.assertEqual((x['video_generations'],x['native_FFN_calls'],x['packed_action'],x['cache_h_clone'],x['cache_release_sync'],x['expert_boundary_cleanup']),(3,480,80,80,80,2));self.assertEqual((x['reference_state_checks'],x['reference_video_checks']),(26,1));self.assertEqual(x['observer_packed_action'],2)
 def test_byte_gate_detects_mismatch_and_wrong_sha(self):
  with tempfile.TemporaryDirectory() as d:
   h=pathlib.Path(d);a=h/'old';b=h/'new';a.write_bytes(b'one');b.write_bytes(b'one')
   with patch.object(m,'HF',h):
    self.assertTrue(m.compare_reference(a,b,3,'same','same'));b.write_bytes(b'two');self.assertFalse(m.compare_reference(a,b,3,'same','same'))
    with self.assertRaises(ValueError):m.compare_reference(a,b,3,'same','wrong')
 def test_verifier_rejects_wrong_reference_and_duplicate_control(self):
  names=['raw_noise_FP32','initial_noise']+[f'{kind}_step{i}' for kind in ('flow','latent') for i in range(4)]
  def phase(mode):return {'mode':mode,'states':{n:dict(sha256='a',bytes=1,raw_tensor_sha256='r',shape=[1],dtype='float32') for n in names},'video':dict(sha256='v',bytes=1)}
  phases=[phase(x) for x in m.driver.MODES];prior=[phase('dense'),phase('x'),phase('x'),phase('both')]
  prefix=names[:2]+[f'{kind}_step{i}' for kind in ('flow','latent') for i in range(3)]
  controls=[dict(mode=mode,state=n,equal=True) for mode in m.driver.MODES for n in (names if mode=='none' else prefix)]
  m.verify_reference_metadata(phases,prior,controls)
  phases[1]['states']['flow_step1']['sha256']='wrong'
  with self.assertRaises(ValueError):m.verify_reference_metadata(phases,prior,controls)
  phases[1]['states']['flow_step1']['sha256']='a';controls[-1]=controls[-2]
  with self.assertRaises(ValueError):m.verify_reference_metadata(phases,prior,controls)
 def test_payload_checks_key_dtype_and_raw_sha(self):
  with tempfile.TemporaryDirectory() as d:
   path=pathlib.Path(d)/'x';payload=b'\x00\x00\x00\x00';header=json.dumps({'latents':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}}).encode();data=struct.pack('<Q',len(header))+header+payload;path.write_bytes(data);file_sha=hashlib.sha256(data).hexdigest();raw_sha=hashlib.sha256(payload).hexdigest()
   m.inspect_payload(path,'latents',[2],'BF16',file_sha,raw_sha)
   for args in [('flow','BF16',raw_sha),('latents','F32',raw_sha),('latents','BF16','wrong')]:
    with self.assertRaises(ValueError):m.inspect_payload(path,args[0],[2],args[1],file_sha,args[2])
 def test_three_modes_and_two_sources(self):
  self.assertEqual(m.driver.MODES,('none','high_only','low_only'));self.assertEqual(m.DESIGN['source_ordinals'],[898,96]);self.assertEqual(len(m.run_ids(0)),3)
if __name__=='__main__':unittest.main()
