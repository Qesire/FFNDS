import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_high_y_reuse_owner as o
class ContractTest(unittest.TestCase):
 def test_scope_and_cost(self):
  x=o.limits();self.assertEqual(o.driver.MODES,('none','high_y_cache_dense','high_y_reuse'));self.assertEqual(x['cache_y_clone'],80);self.assertEqual(x['reuse_y_clone'],40);self.assertEqual(x['reference_state_checks'],24);self.assertEqual(x['reference_video_checks'],2);self.assertNotIn('p64_score',x)
 def test_reference_coverage(self):
  states={name:{'sha256':name,'bytes':1,'raw_tensor_sha256':name,'shape':[1],'dtype':'BF16'} for name in ('raw_noise_FP32','initial_noise',*(f'flow_step{i}' for i in range(4)),*(f'latent_step{i}' for i in range(4)))};video={'sha256':'v','bytes':1};phases=[{'mode':m,'states':states,'video':video} for m in o.driver.MODES];prior=[{'states':states,'video':video}];prefix={'raw_noise_FP32','initial_noise','flow_step0','latent_step0'};controls=[{'mode':m,'state':n,'equal':True} for m in o.driver.MODES for n in (prefix if m=='high_y_reuse' else states)];o.verify_reference_metadata(phases,prior,controls);controls[-1]=controls[0]
  with self.assertRaises(ValueError):o.verify_reference_metadata(phases,prior,controls)
if __name__=='__main__':unittest.main()

class IsolatedEntrypointTest(unittest.TestCase):
 def test_preflight_and_verify_reject_without_active(self):
  import json,os,subprocess,tempfile
  root=Path(__file__).resolve().parents[1];draft=root/'.runtime/chipmunk_high_y_reuse_20261030/draft_part0.json'
  if not draft.is_file():self.skipTest('metadata draft unavailable')
  p=json.loads(draft.read_text());p.update(status='FROZEN_PENDING_REGISTRATION',assets_finalized=True)
  with tempfile.TemporaryDirectory() as t:
   protocol=Path(t)/'p.json';protocol.write_text(json.dumps(p));env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
   for action,expected in [('preflight','ACTIVE required'),('verify-closed','independent verifier ACTIVE required')]:
    cmd=[str(root/'.venv/bin/python'),'-I','-B',str(root/'tools/e0_chipmunk_high_y_reuse_owner.py'),action,'--protocol',str(protocol),'--hf-root','/home/xiang/.cache/huggingface','--local-root',t,'--job-spec',str(Path(t)/'none'),'--out',str(Path(t)/'out'),'--run',str(Path(t)/'run'),'--exit',str(Path(t)/'exit'),'--verify-out',str(Path(t)/'verify')];r=subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True,timeout=30);self.assertNotEqual(r.returncode,0);self.assertIn(expected,r.stderr);self.assertFalse((Path(t)/'out').exists());self.assertFalse((Path(t)/'verify').exists())

class ReferenceIdentityTest(unittest.TestCase):
 def test_same_path_and_hardlink_rejected(self):
  import tempfile, pathlib, hashlib, os
  from unittest.mock import patch
  import e0_chipmunk_high_y_reuse_owner as o
  with tempfile.TemporaryDirectory() as td:
   d=pathlib.Path(td); a=d/'old'; b=d/'new'; a.write_bytes(b'abc'); b.write_bytes(b'abc'); sha=hashlib.sha256(b'abc').hexdigest()
   with patch.object(o,'HF',d):
    self.assertTrue(o.compare_reference(a,b,3,sha,sha))
    with self.assertRaises(Exception): o.compare_reference(a,a,3,sha,sha)
    b.unlink(); os.link(a,b)
    with self.assertRaises(Exception): o.compare_reference(a,b,3,sha,sha)
