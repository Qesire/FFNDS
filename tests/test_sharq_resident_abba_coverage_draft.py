import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from sharq_resident_abba_coverage_draft import Coverage
class T(unittest.TestCase):
 def test_exact_resident_and_bad_coverage(self):
  c=Coverage(lambda x:None);c.begin('resident_B1')
  for i in range(162):c.event({'event':'block_load_returned'})
  for i in range(324):c.event({'weight_copy':'SKIPPED','bias_copy':'ISSUED','skipped_weight_logical_bytes':8,'requested_bias_logical_bytes':2})
  c.finish();self.assertEqual(c.rows[0]['weight_skipped'],324)
  c.begin('legacy_A1')
  with self.assertRaises(ValueError):c.finish()
if __name__=='__main__':unittest.main()
