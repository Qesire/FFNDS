"""Synthetic quality-report arithmetic and evidence checks, no video metrics."""
import unittest,json,tempfile,hashlib
from pathlib import Path
from quality_pair_report import DEV,ARMS,aggregate,load_verified_runs
class QualityReportTest(unittest.TestCase):
 def test_cluster_weight_motion_floor_and_missing(self):
  groups={s:('shared' if i<2 else f'g{i}') for i,s in enumerate(DEV)}
  spec={'sources':DEV,'arms':ARMS,'source_clusters':groups,'motion_dense_floor':0.1}
  def cell(i,arm,value,motion):return {'source':DEV[i],'arm':arm,'dimension':'dynamic_degree','status':'COMPLETE','value':value,'continuous_motion':motion,'motion_windows':[motion]*5,'scope_sha256':'a'*64,'video_sha256':('b' if arm=='Dense' else 'c')*64,'receipt_sha256':'d'*64,'metric_protocol_sha256':'e'*64,'metric_batch_identity':'f'*64,'input_identity':f'input{i}','source_cluster_id':groups[DEV[i]]}
  cells=[cell(0,'Dense',0.,.05),cell(0,'ICC',2.,.1),cell(1,'Dense',0.,1.),cell(1,'ICC',4.,2.),cell(2,'Dense',0.,1.),cell(2,'ICC',9.,2.)]
  result=aggregate(spec,cells)
  summary=next(x for x in result['summaries'] if x['dimension']=='dynamic_degree' and x['arm']=='ICC')
  self.assertEqual((summary['valid_source_count'],summary['valid_cluster_count']),(3,2))
  self.assertEqual(summary['cluster_equal_mean_delta'],6.)
  self.assertEqual(summary['source_delta_range'],[2.,9.])
  pair=next(x for x in result['source_pairs'] if x['source']==DEV[0] and x['arm']=='ICC' and x['dimension']=='dynamic_degree')
  self.assertIsNone(pair['motion_ratio'])
  self.assertEqual(summary['missing_source_count'],8)
  cells[-1]['input_identity']='wrong'
  with self.assertRaises(ValueError):aggregate(spec,cells)
 def test_receipt_binding(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); scores=root/'scores.json'; ex=root/'EXIT.json'; digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
   scores.write_text(json.dumps([{'scope_sha256':'x'*64,'dimension':'imaging_quality'}]))
   receipt={'exit_code':0,'scope_sha256':'x'*64,'result':{'status':'COMPLETE','dimension':'imaging_quality','video_count':1,'per_video_sha256':digest(scores)}}
   ex.write_text(json.dumps(receipt)); item={'per_video_path':str(scores),'per_video_sha256':digest(scores),'exit_path':str(ex),'exit_sha256':digest(ex),'scope_sha256':'x'*64}
   self.assertEqual(len(load_verified_runs([item])),1)
   receipt['exit_code']=1;ex.write_text(json.dumps(receipt));item['exit_sha256']=digest(ex)
   with self.assertRaises(ValueError):load_verified_runs([item])
 def test_duplicate_and_invalid_floor(self):
  groups={s:s for s in DEV};spec={'sources':DEV,'arms':ARMS,'source_clusters':groups,'motion_dense_floor':-1}
  with self.assertRaises(ValueError):aggregate(spec,[])
  spec['motion_dense_floor']=0
  x={'source':DEV[0],'arm':'Dense','dimension':'imaging_quality','status':'MISSING','value':None}
  with self.assertRaises(ValueError):aggregate(spec,[x,x])
if __name__=='__main__':unittest.main()



