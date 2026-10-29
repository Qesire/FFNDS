import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_p64_metric_adapter as adapter
class Tests(unittest.TestCase):
 def test_two_arm_budgets_and_identity(self):
  q=adapter.install('quality');r=adapter.install('raft')
  self.assertEqual(q.MODES,('original_dense','global_p64'));self.assertEqual(q.LIMITS['video_dimension_results'],16);self.assertEqual(q.LIMITS['RAFT_forwards'],80);self.assertEqual(r.LIMITS['video_results'],2);self.assertEqual(r.LIMITS['RAFT_iterations'],1600)
 def test_draft_rejects_missing_actual_parent(self):
  from unittest.mock import patch
  q=adapter.install('quality')
  with patch.object(adapter.line,'parent_data',side_effect=ValueError('no actual verifier')):
   with self.assertRaisesRegex(ValueError,'actual verifier'):q.draft(pathlib.Path('/tmp'),0)
if __name__=='__main__':unittest.main()
class ResultTests(unittest.TestCase):
 def test_quality_two_clips_pairing_and_source_mean(self):
  from unittest.mock import patch
  q=adapter.install('quality');hf=pathlib.Path('/tmp');videos=[{'ordinal':898,'mode':'original_dense','video':{'path':'dense.mp4'}},{'ordinal':898,'mode':'global_p64','video':{'path':'candidate.mp4'}}];plan={'shard':0,'videos':videos}
  dims=[{'dimension':d,'result':d} for d in q.DIMENSIONS]
  def rows(result,paths,dimension):
   return [{'video_path':p,'video_results':float(i+1),**({'image_path':'vbench2_beta_i2v/data/crop/16-9/898.png'} if dimension.startswith('i2v_') else {})} for i,p in enumerate(paths)]
  with patch.object(q.base,'check_result',side_effect=rows):out=q.summarize(plan,hf,dims)
  self.assertEqual(out['independent_image_sources'],1);self.assertEqual(out['per_source'][0]['candidate_minus_same_fresh_Dense']['global_p64']['imaging_quality'],1.)
class VisualTests(unittest.TestCase):
 def test_two_video_81_frames_and_unclosed_refusal(self):
  import e0_chipmunk_p64_visual_cpu as v
  from unittest.mock import patch
  self.assertEqual(v.LIMITS['videos'],2);self.assertEqual(v.LIMITS['successful_decoded_frames'],162);self.assertEqual(v.LIMITS['JPEG_pages'],4)
  with patch.object(v.line,'parent_data',side_effect=ValueError('no verified parent')):
   with self.assertRaisesRegex(ValueError,'verified parent'):v.draft(pathlib.Path('/tmp'),0)
class AggregateTests(unittest.TestCase):
 def test_source_level_equal_weights_and_duplicate_rejection(self):
  from e0_chipmunk_p64_quality_aggregate import aggregate,DIMENSIONS
  def row(source,baseline,candidate):return {'ordinal':source,'scores':{'original_dense':dict.fromkeys(DIMENSIONS,baseline),'global_p64':dict.fromkeys(DIMENSIONS,candidate)}}
  a,b=row(898,1.,3.),row(96,10.,6.);r=aggregate([a,b]);self.assertEqual(r['independent_image_sources'],2);self.assertEqual(r['per_axis']['imaging_quality']['source_equal_paired_difference'],-1.)
  with self.assertRaises(ValueError):aggregate([a,a])
class VisualValidationTests(unittest.TestCase):
 def test_visual_validate_does_not_hash_before_guard(self):
  import json
  from unittest.mock import patch
  import e0_chipmunk_p64_visual_cpu as v
  p=json.loads((pathlib.Path(__file__).resolve().parents[1]/'config/e0_chipmunk_p64_visual_20261030_part0_v1.json').read_text())
  with patch.object(v.base,'sha',side_effect=AssertionError('unexpected unmetered SHA')),patch.object(v,'draft',side_effect=AssertionError('unexpected rebinding')):
   v.validate(p,pathlib.Path('/tmp'),True)
