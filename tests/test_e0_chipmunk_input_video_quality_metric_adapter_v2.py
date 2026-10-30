import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_input_video_quality_visual_cpu as visual
class Test(unittest.TestCase):
 def test_three_video_budget(self):
  self.assertEqual((visual.LIMITS['videos'],visual.LIMITS['successful_decoded_frames'],visual.LIMITS['JPEG_pages']),(3,243,6))
  self.assertIn('PART3',visual.identity(3))
  with self.assertRaises(ValueError):visual.identity(4)
class MetricSummaries(unittest.TestCase):
 def test_counts_and_summaries(self):
  import e0_chipmunk_input_video_quality_metric_adapter_v2 as a
  from unittest.mock import patch
  import tempfile
  q=a.install('quality');r=a.install('raft');self.assertEqual(q.LIMITS['video_dimension_results'],24);self.assertEqual((q.LIMITS['RAFT_forwards'],q.LIMITS['RAFT_iterations']),(120,2400));self.assertEqual((r.LIMITS['video_results'],r.LIMITS['decoded_frames']),(3,243))
  rows=[{'ordinal':1040,'mode':m,'summary':{'mean_top5percent_flow_px_per_2frame_pair':v}} for m,v in zip(r.MODES,(1.,2.,3.))];x=r.summarize({'shard':2},rows);self.assertEqual(x['per_source'][0]['candidate_minus_dense'],{'high_h_readonly_p64':1.,'high_x_recompute_p64':2.});self.assertEqual(x['per_source'][0]['candidate_over_dense'],{'high_h_readonly_p64':2.,'high_x_recompute_p64':3.})
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);videos=[{'ordinal':1040,'mode':m,'video':{'path':m}} for m in q.MODES];dims=[{'dimension':d,'result':None} for d in q.DIMENSIONS]
   def result(_,paths,d):return [{'video_path':p,'image_path':'vbench2_beta_i2v/data/crop/16-9/1040.png','video_results':i} for i,p in enumerate(paths)]
   with patch.object(q.base,'check_result',side_effect=result):got=q.summarize({'videos':videos,'shard':2},root,dims)
   self.assertEqual(got['per_source'][0]['candidate_minus_same_fresh_Dense']['high_x_recompute_p64'][q.DIMENSIONS[0]],2.)
class VisualClosedSixPages(unittest.TestCase):
 def test_real_closed_accepts_six_and_rejects_hardlink_alias(self):
  import tempfile,types,os
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as td:
   hf=Path(td);part=0;out=(hf/visual.output(part)).resolve();pages=out/'pages';pages.mkdir(parents=True);run=out/'RUN.json';ex=out.parent/'EXIT_RECEIPT.json';protocol=hf/'p.json';videos=[];actual=[]
   for mode in visual.line.MODES:
    expected={'ordinal':898,'mode':mode};videos.append(expected);entry=dict(expected,decoded_frames=81,status='COMPLETE',decoded_geometry={'width':832,'height':464,'fps':16.0,'reported_frames':81},pages=[])
    for i in range(2):
     positions=visual.visual.old.page_positions(i);labels=[None if n is None else {'source_ordinal':898,'mode':mode,'frame':n,'timestamp_seconds':n/16} for n in positions];p=pages/f'source898_{mode}_page{i}.jpg';p.write_bytes(b'jpeg');entry['pages'].append({'page':i,'frames':positions,'labels':labels,'width':1920,'height':1414,'quality':90,'jpeg':{'path':str(p.relative_to(hf)),'bytes':4}})
    actual.append(entry)
   p={'id':'CASE','part':0,'parent':{},'videos':videos};r={'id':'CASE','protocol_sha256':'digest','status':'COMPLETE_CHIPMUNK_INPUT_VIDEO_ALL81_CONTACT_SHEETS_NOT_HUMAN_REVIEW','predebited':visual.LIMITS,'invoked':visual.LIMITS,'completed':visual.LIMITS,'wall_seconds':1,'max_RSS_bytes':1,'excess_frames_detected':0,'input_proof':{'tag':'proof'},'journal':{'path':'journal'},'manifest':{'tag':'manifest'},'videos':actual,'job_spec_sha256':'spec'};manifest={'schema':'ffnds.chipmunk.input_video.all81.contact_sheets.v1','videos':actual,'parent':{},'interpretation':visual.INTERPRETATION};receipt={'spec_sha256':'spec'};args=types.SimpleNamespace(hf_root=hf,protocol=protocol,run=run,exit=ex)
   def read(path):return p if path==protocol else r if path==run else receipt
   def read_bound(root,ref):return {'records':[]} if ref.get('tag')=='proof' else manifest
   with patch.object(visual,'validate'),patch.object(visual.base,'read',side_effect=read),patch.object(visual.base,'sha',return_value='digest'),patch.object(visual.base,'binding',return_value={}),patch.object(visual.base,'exited'),patch.object(visual.base,'read_bound',side_effect=read_bound),patch.object(visual.base,'bound',side_effect=lambda root,b:(hf/b['path']).resolve()),patch.object(visual,'inputs',return_value=[]),patch.object(visual.visual,'check_stats'),patch.object(visual.line,'visual_rows',return_value=videos),patch.object(visual.visual,'replay_journal',return_value={'predebited':visual.LIMITS,'invoked':visual.LIMITS,'completed':visual.LIMITS}):
    self.assertEqual(visual._verify_closed(args)['status'],'PASS_CLOSED_CHIPMUNK_INPUT_VIDEO_VISUAL');a=pages/'source898_none_page0.jpg';b=pages/'source898_high_h_readonly_p64_page0.jpg';b.unlink();os.link(a,b)
    with self.assertRaisesRegex(ValueError,'Page alias'):visual._verify_closed(args)
class QualityCLI(unittest.TestCase):
 def test_real_parser_accepts_shard_2_and_3(self):
  import e0_chipmunk_input_video_quality_metric_adapter_v2 as adapter
  from unittest.mock import patch
  import tempfile
  q=adapter.install('quality')
  for shard in (2,3):
   with self.subTest(shard=shard),tempfile.TemporaryDirectory() as td:
    out=Path(td)/'draft.json'
    with patch.object(sys,'argv',['quality','draft','--shard',str(shard),'--out',str(out)]),patch.object(q,'draft',return_value={'part':shard}) as draft,patch.object(q.base,'write') as write:
     q.main();draft.assert_called_once();self.assertEqual(draft.call_args.args[1],shard);write.assert_called_once()
