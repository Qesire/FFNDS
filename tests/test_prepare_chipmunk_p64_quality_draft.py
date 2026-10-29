import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from prepare_chipmunk_p64_quality_draft import build
class Tests(unittest.TestCase):
 def test_six_scopes_are_unbound_and_exact_pairs(self):
  p=build();self.assertEqual(len(p['scopes']),6)
  for s in p['scopes']:
   self.assertEqual(s['status'],'NOT_BOUND_NOT_EXECUTABLE');self.assertFalse(s['assets_finalized']);self.assertEqual([x['mode'] for x in s['clips']],['original_dense','global_p64']);self.assertTrue(all(x['video']=='PENDING_ACTUAL_VIDEO_SHA_BYTES' for x in s['clips']))
 def test_budget_tracks_two_videos(self):
  s=build()['scopes'];self.assertEqual(next(x for x in s if x['kind']=='quality')['limits']['video_dimension_results'],16);self.assertEqual(next(x for x in s if x['kind']=='raft')['limits']['RAFT_forwards'],80);self.assertEqual(next(x for x in s if x['kind']=='visual')['limits']['successful_decoded_frames'],162)
if __name__=='__main__':unittest.main()
