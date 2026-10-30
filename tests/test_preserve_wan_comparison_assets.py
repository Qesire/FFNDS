import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import preserve_wan_comparison_assets as keep

class PreservationTests(unittest.TestCase):
 def fixture(self,root,sha=None):
  hf=root/'hf';hf.mkdir();source=hf/'example';source.write_bytes(b'evidence')
  row={'relative_path':'HF:example','bytes':8,'sha256':sha or hashlib.sha256(b'evidence').hexdigest()}
  manifest=root/'manifest.json';manifest.write_text(json.dumps({'assets':[row]*868}));return hf,source,manifest
 def test_independent_copy_and_exact_duplicate_prefix_resolution(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);hf,source,manifest=self.fixture(root)
   result=keep.preserve(manifest,hf,root/'vault',root/'receipts'/'RUN.json')
   target=next((root/'vault'/'objects').glob('*/*'))
   self.assertEqual(result['source_paths_verified'],1);self.assertEqual(result['independent_objects_verified'],1)
   self.assertEqual(result['read_completed_bytes'],16);self.assertEqual(target.read_bytes(),source.read_bytes());self.assertNotEqual(target.stat().st_ino,source.stat().st_ino)
   with self.assertRaises(ValueError):keep.preserve(manifest,hf,root/'vault',root/'second.json')
 def test_corrupt_expected_digest_fails_without_deleting_source(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);hf,source,manifest=self.fixture(root,'0'*64);run=root/'receipts'/'RUN.json'
   with self.assertRaisesRegex(ValueError,'source SHA'):keep.preserve(manifest,hf,root/'vault',run)
   result=json.loads(run.read_text());self.assertEqual(result['status'],'FAILED_MAINTENANCE_NO_RETRY');self.assertEqual(result['source_paths_verified'],0);self.assertEqual(result['new_copy_predebited_bytes'],8);self.assertEqual(source.read_bytes(),b'evidence')
 def test_path_escape_rejected(self):
  for value in ('HF:../secret','/tmp/secret',''):
   with self.assertRaises(ValueError):keep.normalized(value)
