import sys,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_p64_factorial_metric_adapter as adapter
class FactorialLimitsTest(unittest.TestCase):
 def test_quality_limits(self):
  e=adapter.install('quality');self.assertEqual(e.MODES,('none','high_only','low_only','both'));self.assertEqual(e.LIMITS['video_dimension_results'],32);self.assertEqual(e.LIMITS['RAFT_forwards'],160);self.assertEqual(e.LIMITS['RAFT_iterations'],3200);self.assertEqual(e.LIMITS['decoded_frames'],324)
 def test_raft_limits(self):
  e=adapter.install('raft');self.assertEqual(e.LIMITS['video_results'],4);self.assertEqual(e.LIMITS['RAFT_forwards'],160);self.assertEqual(e.LIMITS['RAFT_iterations'],3200)
 def test_raft_four_arms_and_interaction(self):
  e=adapter.install('raft'); rows=[]
  for mode,value in zip(e.MODES,(2.,3.,4.,9.)):
   rows.append({'ordinal':898,'mode':mode,'summary':{'mean_top5percent_flow_px_per_2frame_pair':value,'five_fixed_8pair_windows':[{'mean_top5percent_flow_px':value} for _ in range(5)]}})
  r=e.summarize({'shard':0,'prior_score_report':e.base.binding(Path('research/20261030/CHIPMUNK_P64_V2_QUALITY_RESULTS_20261030.json').resolve(),Path('.').resolve())},rows);self.assertEqual(r['per_source'][0]['factorial_interaction'],4.);self.assertEqual(r['per_source'][0]['high_minus_none'],1.)
 def test_parent_not_closed(self):
  e=adapter.install('quality')
  with patch.object(adapter.line,'parent_data',side_effect=ValueError('parent not closed')):
   with self.assertRaises(ValueError):e.draft(Path('/home/xiang/.cache/huggingface'),0)
 def test_four_arm_summary(self):
  e=adapter.install('quality'); modes=e.MODES;videos=[{'ordinal':898,'mode':m,'video':{'path':m}} for m in modes];plan={'shard':0,'videos':videos,'prior_score_report':e.base.binding(Path('research/20261030/CHIPMUNK_P64_V2_QUALITY_RESULTS_20261030.json').resolve(),Path('.').resolve())};dims=[{'dimension':d,'result':None} for d in e.DIMENSIONS]
  def rows(result,paths,dimension):return [{'video_path':p,'video_results':i,'image_path':'vbench2_beta_i2v/data/crop/16-9/898.png'} for i,p in enumerate(paths)]
  with patch.object(e.base,'check_result',side_effect=rows):result=e.summarize(plan,Path('/tmp'),dims)
  self.assertEqual(list(result['per_source'][0]['scores']),list(modes));self.assertEqual(result['per_source'][0]['candidate_minus_same_fresh_Dense']['both']['dynamic_degree'],3)
if __name__=='__main__':unittest.main()
