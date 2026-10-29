import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from e0_chipmunk_p64_factorial_quality_aggregate import aggregate,MODES,repeated_video_scores
class AggregateTest(unittest.TestCase):
 def test_paired_source_equal_interaction(self):
  rows=[{'ordinal':898,'scores':{m:{'q':x} for m,x in zip(MODES,[1,3,4,9])}},{'ordinal':96,'scores':{m:{'q':x} for m,x in zip(MODES,[2,3,6,11])}}]
  a=aggregate(rows,('q',))['dimensions']['q'];self.assertEqual([p['interaction'] for p in a['per_source']],[3,4]);self.assertEqual(a['source_equal_interaction'],3.5);self.assertEqual(a['source_equal_high_minus_none'],1.5)
 def test_missing_or_nonfinite(self):
  rows=[{'ordinal':s,'scores':{m:{'q':0.} for m in MODES}} for s in (898,96)];del rows[0]['scores']['both']
  with self.assertRaises(ValueError):aggregate(rows,('q',))
  rows[0]['scores']['both']={'q':float('nan')}
  with self.assertRaises(ValueError):aggregate(rows,('q',))
 def test_repeated(self):
  x=repeated_video_scores({'a':2.,'b':4.},{'a':2.,'b':3.});self.assertFalse(x['all_exact_equal']);self.assertEqual(x['per_dimension']['b']['difference'],1.)
  with self.assertRaises(ValueError):repeated_video_scores({'a':1},{'b':1})
if __name__=='__main__':unittest.main()
