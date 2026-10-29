import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('test_teal_quality',ROOT/'tools/e0_teal_wan_full_quality_v2.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_eight_dimensions_three_videos_exact_budget(self):
  self.assertEqual(len(m.DIMENSIONS),8);self.assertEqual(m.LIMITS['video_dimension_results'],24)
  self.assertEqual(m.LIMITS['RAFT_forwards'],120);self.assertEqual(m.LIMITS['RAFT_iterations'],2400)
  self.assertEqual(m.COSTS['imaging_quality']['MUSIQ'],243);self.assertEqual(m.COSTS['i2v_subject']['DINO'],246)
 def test_missing_parent_blocks_draft(self):
  with patch.object(m.motion,'parent_data',side_effect=FileNotFoundError):
   with self.assertRaises(FileNotFoundError):m.draft(Path('/tmp'),0)
 def test_source_equal_no_false_source_replication(self):
  vids=[{'ordinal':1040,'mode':k,'video':{'path':f'{k}.mp4'}} for k in m.MODES]
  dims=[{'dimension':d,'result':d} for d in m.DIMENSIONS]
  def check(_result,paths,dimension):
   return [{'video_path':p,'video_results':1.,'image_path':'vbench2_beta_i2v/data/crop/16-9/1040.png'} for p in paths]
  with patch.object(m.base,'check_result',side_effect=check):r=m.summarize({'shard':0,'videos':vids},Path('/tmp'),dims)
  self.assertEqual(r['independent_image_sources'],1)
  self.assertTrue(all(x['source_count']==1 for x in r['per_axis'].values()))
  self.assertFalse(r['official_five_independent_samples'])
if __name__=='__main__':unittest.main()
