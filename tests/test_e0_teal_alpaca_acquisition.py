import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('alpaca_scope',ROOT/'tools/e0_teal_alpaca_acquisition.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class AcquisitionChecks(unittest.TestCase):
 def protocol(self):
  return dict(id=m.ID,repo=m.REPO,revision=m.REV,filename=m.NAME,sha256=m.DIGEST,size=m.SIZE,cache=m.CACHE,output=m.OUT,resources=m.RESOURCES,limits=m.LIMITS,source_bindings=[{'path':p} for p in m.SOURCES])
 def test_identity_is_fixed(self):
  p=self.protocol();m.validate(p)
  p['revision']='other'
  with self.assertRaises(Exception):m.validate(p)
 def test_unreviewed_draft_cannot_run(self):
  p=self.protocol();p.update(status='DRAFT_REVIEW_REQUESTED',review=None,worker_config_sha256=None)
  with self.assertRaises(Exception):m.validate(p,True)
 def test_hash_rejects_altered_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'blob';p.write_bytes(b'abc')
   item={'bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest(),'git_blob':None}
   self.assertEqual(m.common.verify_file(p,item)['bytes'],3)
   p.write_bytes(b'abd')
   with self.assertRaises(Exception):m.common.verify_file(p,item)

 def test_scan_missing_root_fails(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(FileNotFoundError):m.worker_usage(Path(d)/'missing')
 def test_closed_requires_actual_success_and_two_hashes(self):
  p=self.protocol();p.update(status='FROZEN_PENDING_REGISTRATION',review=None,worker_config_sha256='a'*64)
  r={'id':m.ID,'status':'COMPLETE_PUBLIC_BYTES_NOT_MODEL_EXECUTED','protocol_sha256':'p','launch_spec_sha256':'s','download_predebit_bytes':m.SIZE,'hash_predebit_bytes':m.SIZE*2,'first_hash_completed_bytes':m.SIZE,'second_hash_completed_bytes':m.SIZE,'result':{'bytes':m.SIZE,'sha256':m.DIGEST},'worker_usage':{'apparent_bytes':1},'wall_seconds':1,'cpu_seconds':1,'hub_api_calls':1,'model_calls':0,'gpu_calls':0,'automatic_retry':False}
  e={'id':m.ID,'process_exited':True,'child_returncode':0,'timeout':False,'automatic_retry':False,'run_sha256':'r','protocol_sha256':'p','spec_sha256':'s'}
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   with patch.object(m,'HF',Path(d)):
    asset=Path(d)/m.CACHE/'datasets--tatsu-lab--alpaca'/'blobs'/m.DIGEST;asset.parent.mkdir(parents=True);asset.touch()
    with asset.open('r+b') as f:f.truncate(m.SIZE)
    r['cache_path']=str(asset.relative_to(d))
    self.assertTrue(m.validate_closed(p,'p',r,e,'r','s')['status'].startswith('PASS'))
    asset.unlink()
    with self.assertRaises(Exception):m.validate_closed(p,'p',r,e,'r','s')
    with self.assertRaises(Exception):m.validate_closed(p,'p',r,e,'r','s',Path(d)/'wrong')
  e['child_returncode']=1
  with self.assertRaises(Exception):m.validate_closed(p,'p',r,e,'r','s')
  e['child_returncode']=0;r['second_hash_completed_bytes']=0
  with self.assertRaises(Exception):m.validate_closed(p,'p',r,e,'r','s')
 def test_scan_vanished_descendant(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'vanish';path.write_text('x')
   original=Path.lstat
   def flaky(p,*a,**kw):
    if p==path: raise FileNotFoundError(str(p))
    return original(p,*a,**kw)
   with patch.object(Path,'lstat',flaky): self.assertEqual(m.worker_usage(d)['missing_entries'],1)
 def test_scan_non_enoent_fails(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'denied';path.write_text('x');original=Path.lstat
   def flaky(p,*a,**kw):
    if p==path: raise PermissionError(str(p))
    return original(p,*a,**kw)
   with patch.object(Path,'lstat',flaky):
    with self.assertRaises(PermissionError):m.worker_usage(d)

if __name__=='__main__':unittest.main()
