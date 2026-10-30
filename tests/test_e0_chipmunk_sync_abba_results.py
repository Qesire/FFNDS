import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_sync_abba_results as r
import chipmunk_p64_sync_abba_contract as c
class Test(unittest.TestCase):
 def test_warmups_excluded_and_infer(self):
  rows=[]
  for i,label in enumerate(c.schedule(0)):
   rows.append({'sync_arm':label,'FFN_boundary_CUDA_seconds':100 if 'warm' in label else i+1,'wall_seconds_including_observers':i+1,'stages':[{'name':'infer','step':j,'CUDA_seconds':1,'wall_seconds':2} for j in range(4)]})
  x=r.summary(rows,0);v=x['measured_excluding_warmups']['FFN_boundary_CUDA_seconds'];self.assertEqual(v['A_measured'],[4,7]);self.assertEqual(v['B_measured'],[5,6]);self.assertEqual(v['A_mean'],5.5);self.assertEqual(x['raw_phase_timings']['none']['infer_four_CUDA_seconds'],4)
 def test_reject_incomplete(self):
  with self.assertRaises(ValueError):r.summary([],0)
 def test_source_equal_ratios_not_ratio_of_means(self):
  def row(src,a,b):return {'source_ordinal':src,'measured_excluding_warmups':{'metric':{'A_mean':a,'B_mean':b,'B_minus_A':b-a,'B_over_A':b/a}}}
  x=r.source_equal([row(1,1,2),row(2,9,9)])['metric'];self.assertEqual(x['A_source_equal_mean'],5);self.assertEqual(x['B_source_equal_mean'],5.5);self.assertEqual(x['source_equal_mean_B_minus_A'],.5);self.assertEqual(x['mean_of_within_source_B_over_A_ratios'],1.5);self.assertNotEqual(x['mean_of_within_source_B_over_A_ratios'],5.5/5)
 def test_source_equal_reject_duplicate_source(self):
  with self.assertRaises(ValueError):r.source_equal([{'source_ordinal':1},{'source_ordinal':1}])
if __name__=='__main__':unittest.main()
