import unittest
from quality_ablation_report import aggregate_ablation, DEV3

class AblationTest(unittest.TestCase):
 def test_pair_batch_identity_and_motion_floor(self):
  spec={'source_clusters':{s:s for s in DEV3},'motion_dense_floor':.001,'explicit_camera_instruction_by_source':{}}
  def cell(arm,motion):
   return {'source':DEV3[0],'arm':arm,'dimension':'dynamic_degree','status':'COMPLETE','value':motion,'continuous_motion':motion,'source_cluster_id':DEV3[0],'input_identity':{'seed':42},'metric_batch_identity':'a'*64,'metric_protocol_sha256':'b'*64,'comparison':'E4'}
  cells=[cell('ICC',.0005),cell('ICC 2R E4',.002)]
  out=aggregate_ablation(spec,cells,'E4')
  row=next(x for x in out['pairs'] if x['source']==DEV3[0] and x['dimension']=='dynamic_degree')
  self.assertEqual(row['status'],'COMPLETE');self.assertIsNone(row['motion_ratio'])
  self.assertEqual(next(x for x in out['pairs'] if x['dimension']=='camera_motion')['status'],'NOT_APPLICABLE')
  cells[1]['metric_batch_identity']='c'*64
  with self.assertRaises(ValueError):aggregate_ablation(spec,cells,'E4')
 def test_e6_requires_frozen_variable(self):
  spec={'source_clusters':{s:s for s in DEV3},'motion_dense_floor':.001}
  with self.assertRaises(ValueError):aggregate_ablation(spec,[],'E6')
  self.assertEqual(aggregate_ablation(spec,[],'E6',e6_variable='tail')['e6_variable'],'tail')
if __name__=='__main__':unittest.main()
