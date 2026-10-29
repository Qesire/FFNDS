import hashlib, importlib.util, pathlib, tempfile, unittest, resource
P=pathlib.Path(__file__).resolve().parents[1]/'tools/e0_teal_llama3_public_weight_acquisition.py'
s=importlib.util.spec_from_file_location('llama_assets',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_official_weights_and_fixed_manifest(self):
  files=m.manifest();self.assertEqual(len(files),13);w=[x for x in files if x['filename'].endswith('.safetensors')];self.assertEqual(len(w),4);self.assertEqual(sum(x['bytes'] for x in w),16060556376);self.assertTrue(all(x['official_match']=='PUBLIC_METADATA_MATCH' for x in w))
 def test_output_file_cap_budget(self):
  files=m.manifest(); self.assertTrue(all(m.file_size_limit(x)>=x['bytes'] for x in files));self.assertGreaterEqual(min(m.file_size_limit(x) for x in files),16<<20);self.assertLessEqual(sum(m.file_size_limit(x) for x in files)+(64<<20),m.RESOURCES['new_HF_bytes'])
 def test_git_and_sha_and_modified_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'f';b=b'abc';p.write_bytes(b);git=hashlib.sha1(b'blob 3\0'+b).hexdigest();sha=hashlib.sha256(b).hexdigest()
   self.assertEqual(m.verify_file(p,{'filename':'f','bytes':3,'sha256':None,'git_blob':git})['sha256'],sha)
   self.assertEqual(m.verify_file(p,{'filename':'f','bytes':3,'sha256':sha,'git_blob':None})['git_blob'],git)
   p.write_bytes(b'bad')
   with self.assertRaises(ValueError):m.verify_file(p,{'filename':'f','bytes':3,'sha256':sha,'git_blob':git})
 def test_private_usage_counts_partial_and_ignores_symlinks(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);(root/'partial').write_bytes(b'12345');(root/'snapshot').symlink_to(root/'partial');self.assertEqual(m.usage(root),{'apparent_bytes':5,'regular_files':1});(root/'partial').write_bytes(b'1234567');self.assertEqual(m.usage(root)['apparent_bytes'],7)
 def test_backend_failure_restores_file_limit_and_partial_is_accounted(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d); before=resource.getrlimit(resource.RLIMIT_FSIZE)
   def backend(**kwargs):
    self.assertEqual(resource.getrlimit(resource.RLIMIT_FSIZE)[0],16<<20);(root/'x.incomplete').write_bytes(b'partial');raise OSError('injected')
   with self.assertRaises(OSError):m.call_backend({'bytes':3},[{'bytes':3}],backend)
   self.assertEqual(resource.getrlimit(resource.RLIMIT_FSIZE),before);self.assertEqual(m.usage(root)['apparent_bytes'],7)
 def test_clone_collision_and_original_unchanged(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);src=root/'src';dst=root/'dst';src.write_bytes(b'old');dst.write_bytes(b'new')
   with self.assertRaises(FileExistsError):m.clone_fresh(src,dst)
   self.assertEqual(src.read_bytes(),b'old');self.assertEqual(dst.read_bytes(),b'new')
 def test_two_scope_authorization_is_independent(self):
  d='a'*64;base={'id':m.ID,'protocol_sha256':d,'execution_status':'COMPLETE','resources':m.RESOURCES,'limits':m.LIMITS,'max_gpu_cards':0};verify={'id':m.ID+'_VERIFY','protocol_sha256':d,'execution_status':'AUTHORIZED_NOT_STARTED','resources':m.VERIFY_RESOURCES,'limits':m.VERIFY_LIMITS,'max_gpu_cards':0};active={'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},'experiment_authority':{'current_user_authorized_scopes':[base,verify]}}
  self.assertEqual(m.select_scope(active,{},d,True)[0],m.ID+'_VERIFY')
  with self.assertRaises(ValueError):m.select_scope(active,{},d,False)
  active['experiment_authority']['current_user_authorized_scopes']=[base]
  with self.assertRaises(ValueError):m.select_scope(active,{},d,True)
  verify['resources']={**m.VERIFY_RESOURCES,'GPU':1};active['experiment_authority']['current_user_authorized_scopes']=[verify]
  with self.assertRaises(ValueError):m.select_scope(active,{},d,True)
 def test_actual_parent_exit_required(self):
  d='a'*64;r={'status':'COMPLETE_PUBLIC_BYTES_NOT_MODEL_EXECUTED','launch_spec_sha256':'s'};e={'id':m.ID,'child_returncode':0,'process_exited':True,'timeout':False,'automatic_retry':False,'run_sha256':'r','protocol_sha256':d,'spec_sha256':'s'};m.validate_parent_receipt(r,e,d,'r')
  for key,val in [('child_returncode',1),('run_sha256','wrong'),('process_exited',False),('timeout',True)]:
   broken={**e,key:val}
   with self.assertRaises(ValueError):m.validate_parent_receipt(r,broken,d,'r')
 def test_wrong_size_and_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'f';p.write_bytes(b'abc');q=pathlib.Path(d)/'link';q.symlink_to(p)
   for f,n in [(p,2),(q,3)]:
    with self.assertRaises(ValueError):m.verify_file(f,{'filename':'f','bytes':n,'sha256':'0'*64,'git_blob':None})
 def test_draft_not_frozen(self):
  p=m.draft();m.validate(p)
  with self.assertRaises(ValueError):m.validate(p,True)
if __name__=='__main__':unittest.main()
