import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_quality_aggregate as m
class Tests(unittest.TestCase):
 def test_equal_source_not_exposure_weight(self):
  rows=[{'ordinal':1040,'scores':{'dense':{'x':2},'dense32':{'x':4},'sharq':{'x':8}}},{'ordinal':979,'scores':{'dense':{'x':10},'dense32':{'x':8},'sharq':{'x':6}}}];r=m.reduce_vectors(rows,['x']);self.assertEqual(r['source_equal_means']['dense']['x'],6);self.assertEqual(r['source_equal_candidate_minus_dense']['dense32']['x'],0);self.assertEqual(r['source_equal_candidate_minus_dense']['sharq']['x'],1);self.assertFalse(r['heldout'])
 def test_duplicate_source_and_nonfinite_rejected(self):
  row={'ordinal':1040,'scores':{x:{'a':1} for x in m.MODES}}
  with self.assertRaises(ValueError):m.reduce_vectors([row,row],['a'])
  row2={'ordinal':979,'scores':{x:{'a':float('nan')} for x in m.MODES}}
  with self.assertRaises(ValueError):m.reduce_vectors([row,row2],['a'])
if __name__=='__main__':unittest.main()
