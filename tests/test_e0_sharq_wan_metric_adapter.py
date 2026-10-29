import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_metric_adapter as m
class Tests(unittest.TestCase):
 def test_engines_preserve_original_physical_budgets(self):
  q=m.install('quality');r=m.install('raft')
  self.assertEqual(q.LIMITS['official_dimension_calls'],8);self.assertEqual(q.LIMITS['video_dimension_results'],24);self.assertEqual(q.LIMITS['RAFT_forwards'],120);self.assertEqual(r.LIMITS['RAFT_forwards'],120);self.assertEqual(r.LIMITS['RAFT_iterations'],2400)
  self.assertEqual(q.MODES,('dense','dense32','sharq'));self.assertEqual(r.SHARDS,((1040,),(979,)))
 def test_real_engines_summarize_each_source_and_arm(self):
  q=m.install('quality');hf=Path('/tmp');p={'shard':0,'videos':[{'ordinal':1040,'mode':mode,'video':{'path':mode+'.mp4'}} for mode in q.MODES]};dims=[]
  for d in q.DIMENSIONS:
   rows=[{'video_path':str((hf/v['video']['path']).resolve()),'video_results':i+.25,'image_path':'vbench2_beta_i2v/data/crop/16-9/1040.png'} for i,v in enumerate(p['videos'])];dims.append({'dimension':d,'result':rows})
  with patch.object(q.base,'check_result',side_effect=lambda result,paths,dimension:result):result=q.summarize(p,hf,dims)
  self.assertEqual(result['independent_image_sources'],1);self.assertEqual(result['per_source'][0]['candidate_minus_same_fresh_Dense']['sharq'][q.DIMENSIONS[0]],2.)
 def test_synthetic_closed_parent_draft_and_validation(self):
  case=json.loads((m.ROOT/'config/e0_sharq_wan_all160_video_v2_part0_20261030_frozen_pending_v3.json').read_text())['case'];proof={'protocol':{'path':m.line.PARENT.format(0),'bytes':1,'sha256':'a'*64},**{k:{'path':k,'bytes':1,'sha256':'b'*64} for k in ('RUN','exit','verification_RUN','verification_exit')}}
  rows=[{'ordinal':1040,'mode':mode,'case':case,'generation':proof['RUN'],'video':{'path':mode,'bytes':2,'sha256':'c'*64},'states':{}} for mode in m.line.MODES]
  for kind in ('quality','raft'):
   e=m.install(kind)
   def bind(path,root):return {'path':str(Path(path).relative_to(root)),'bytes':1,'sha256':'f'*64}
   with patch.object(m.line,'parent_data',return_value=proof),patch.object(m.line,'closed_rows',return_value=rows),patch.object(m.line,'contract',return_value={'contract':'synthetic'}),patch.object(e.base,'binding',side_effect=bind),patch.object(e.base,'read_bound',return_value={'repo_sources':[]}),patch.object(e.execution,'monitor_binding',return_value={'synthetic':True}):
    p=e.draft(Path('/tmp'),0);e.validate(p);self.assertEqual(p['known_hash_work']['preflight_extra_video_hash_calls'],3);self.assertEqual(p['known_hash_work']['GPU_extra_video_bytes'],12)
 def test_draft_rejects_unclosed_parent(self):
  for kind in ('quality','raft'):
   e=m.install(kind)
   with patch.object(m.line,'parent_data',side_effect=ValueError('not closed')):
    with self.assertRaisesRegex(ValueError,'not closed'):e.draft(Path('/tmp'),0)
 def test_invalid_kind_and_part(self):
  with self.assertRaises(ValueError):m.install('other')
  for kind in ('quality','raft'):
   e=m.install(kind)
   for p in (-1,2,True):
    with self.assertRaises(ValueError):e.identity(p)
if __name__=='__main__':unittest.main()
