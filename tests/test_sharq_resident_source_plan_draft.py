import pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from sharq_resident_source_plan_draft import bindings,source_paths
class T(unittest.TestCase):
 def test_binding_and_bad_parent(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=pathlib.Path(tmp);(p/'a').write_bytes(b'a');b=bindings(['a'],p);self.assertEqual(b[0]['bytes'],1)
   with self.assertRaises(ValueError):bindings(['../a'],p)
   with self.assertRaises(ValueError):source_paths({'repo_sources':[{'path':'x'},{'path':'x'}]})
if __name__=='__main__':unittest.main()
