import json, pathlib, tempfile, unittest
from unittest.mock import patch
from tools import e0_chipmunk_saved_h_discrimination as driver

class BoundaryTests(unittest.TestCase):
    def test_asset_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d);(p/'x').write_bytes(b'abc')
            with self.assertRaises(ValueError): driver.checked(p,{'path':'x','bytes':3,'sha256':'0'*64})
            with self.assertRaises(ValueError): driver.checked(p,{'path':'../escape','bytes':3,'sha256':'0'*64})
    def test_unfinalized_rejected_before_active_lookup(self):
        with self.assertRaises(ValueError): driver.authorize({'id':driver.ID,'assets_finalized':False,'output':driver.OUTPUT},'0'*64)
    def test_atomic_json_limit(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'RUN.json';driver.atomic(p,{'status':'FAILED_NO_RETRY'})
            self.assertEqual(json.loads(p.read_text())['status'],'FAILED_NO_RETRY')
            with self.assertRaises(ValueError):driver.atomic(p,{'x':'x'*(5<<20)})
            self.assertEqual(json.loads(p.read_text())['status'],'FAILED_NO_RETRY')
if __name__=='__main__':unittest.main()
