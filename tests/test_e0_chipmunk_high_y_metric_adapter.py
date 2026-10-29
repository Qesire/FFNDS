import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_high_y_metric_adapter as a
import e0_chipmunk_high_y_visual_cpu as visual
class Counts(unittest.TestCase):
 def test_three_actual_arm_costs(self):
  q=a.install('quality');r=a.install('raft')
  self.assertEqual(q.MODES,('none','high_p64','high_y_reuse'))
  self.assertEqual(q.LIMITS['video_dimension_results'],24)
  self.assertEqual((q.LIMITS['RAFT_forwards'],q.LIMITS['RAFT_iterations']),(120,2400))
  self.assertEqual((r.LIMITS['video_results'],r.LIMITS['RAFT_forwards'],r.LIMITS['RAFT_iterations']),(3,120,2400))
  self.assertEqual((visual.LIMITS['videos'],visual.LIMITS['successful_decoded_frames'],visual.LIMITS['JPEG_pages']),(1,81,2))

 def test_three_arm_raft_summary(self):
  r=a.install('raft'); rows=[{'ordinal':898,'mode':m,'summary':{'mean_top5percent_flow_px_per_2frame_pair':v}} for m,v in zip(r.MODES,(1.,2.,3.))]
  x=r.summarize({'shard':0},rows);self.assertEqual(x['per_source'][0]['candidate_minus_dense'],{'high_p64':1.,'high_y_reuse':2.})
  with self.assertRaises(Exception):r.summarize({'shard':0},rows[:-1])
 def test_quality_three_arm_summary(self):
  from unittest.mock import patch
  import tempfile
  q=a.install('quality')
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);videos=[{'ordinal':898,'mode':m,'video':{'path':m}} for m in q.MODES];dim=[{'dimension':d,'result':None} for d in q.DIMENSIONS];old={'official_eight_axes':{'per_source':[{'ordinal':898,'scores':{'original_dense':{d:0. for d in q.DIMENSIONS}}}]}}
   def result(_,paths,d):
    return [{'video_path':p,'image_path':'vbench2_beta_i2v/data/crop/16-9/898.png','video_results':i} for i,p in enumerate(paths)]
   with patch.object(q.base,'check_result',side_effect=result),patch.object(q.base,'read_bound',return_value=old):
    got=q.summarize({'videos':videos,'shard':0,'prior_score_report':{}},root,dim)
   self.assertEqual(got['per_source'][0]['candidate_minus_same_fresh_Dense']['high_y_reuse'][q.DIMENSIONS[0]],2.)
