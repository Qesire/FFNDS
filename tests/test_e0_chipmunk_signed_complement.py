import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_signed_complement as d
class Entry(unittest.TestCase):
 def test_budget_denial_is_atomic(self):
  counts={'first':0,'second':1}
  with self.assertRaisesRegex(ValueError,'operation budget'):d.reserve_counts(counts,{'first':1,'second':1},{'first':1,'second':1})
  self.assertEqual(counts,{'first':0,'second':1})
 def test_unfinalized_is_inert(self):
  with self.assertRaisesRegex(ValueError,'unfinalized'):d.authorize({'id':'E0_CHIPMUNK_SIGNED_COMPLEMENT_20261030_PART0','assets_finalized':False},'x')
if __name__=='__main__':unittest.main()
