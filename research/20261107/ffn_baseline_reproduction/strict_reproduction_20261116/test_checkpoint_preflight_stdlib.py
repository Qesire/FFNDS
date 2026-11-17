"""Stateless checkpoint-preflight test with stubbed torch; no model calls."""
import importlib.util,sys,tempfile,types,unittest
from pathlib import Path
class TestCheckpointIdentity(unittest.TestCase):
 def test_hash_once_and_detect_mutation(self):
  path=Path(__file__).with_name('icc_stage3_assets.py'); old=sys.modules.get('torch');sys.modules['torch']=types.ModuleType('torch')
  try:
   spec=importlib.util.spec_from_file_location('stage3_assets_test_copy',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   with tempfile.TemporaryDirectory() as d:
    f=Path(d)/'checkpoint';f.write_bytes(b'weights'); expected=module.sha(f); calls=[]; original=module.sha
    def counted(p):calls.append(str(p));return original(p)
    module.sha=counted; bound={'path':str(f),'sha256':expected}
    module.preflight_checkpoint(bound);module.preflight_checkpoint(bound)
    self.assertEqual(len(calls),1)
    f.write_bytes(b'changed!')
    with self.assertRaises(ValueError):module.preflight_checkpoint(bound)
  finally:
   if old is None:sys.modules.pop('torch',None)
   else:sys.modules['torch']=old
if __name__=='__main__':unittest.main()
