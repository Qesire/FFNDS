import unittest,tempfile,json,hashlib
from pathlib import Path
from icc_stage3_selected_shards import build,validate_shard,DEV,SIX,REMAINING,FAMILIES
from icc_stage3_freeze import assemble

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8');return hashlib.sha256(path.read_bytes()).hexdigest()
def job(stage,jobid):
 sh=write(stage/'scope.json',{'job':jobid});write(stage/'receipt/RUN.json',{'scope_sha256':sh,'slurm_job_id':jobid});write(stage/'receipt/EXIT.json',{'exit_code':0,'slurm_job_id':jobid});return {'stage':str(stage),'scope_sha256':sh,'job_id':jobid}
class ShardsTest(unittest.TestCase):
 def fixture(self,root):
  g=root/'grid';gs=job(g,'1');p=root/'six';ps=job(p,'2');rows=[];factors=[]
  for e,l in SIX:
   file=p/'output'/f'SVD__{e}__{l:02d}__r1408.pt';file.parent.mkdir(parents=True,exist_ok=True);file.write_text(e+str(l));h=hashlib.sha256(file.read_bytes()).hexdigest();ident={'family':'SVD','expert':e,'layer':l,'rank':1408,'checkpoint_sha256':e,'calibration_manifest_sha256':None,'svd_driver':'gesvd','factor_split':'U*S, Vh','decomposition_dtype':'float32','execution_factor_dtype':'bfloat16','svd_full_matrices':False,'prefix_rank512_is_same_decomposition':True};receipt={'identity':ident,'path':str(file),'sha256':h,'bytes':file.stat().st_size};write(Path(str(file)+'.json'),receipt);factors.append(receipt)
   for source in DEV:
    for family in FAMILIES:
     for rank in (512,1408):
      for mode in ('4S','2R'):
       rows.append({'source':source,'expert':e,'layer':l,'family':family,'rank':rank,'mode':mode,'risk':float(FAMILIES.index(family)+1),'scope_sha256':gs['scope_sha256'],'factor_sha256':h if family=='SVD' else ('a' if family=='CA' else 'b')*64})
  ps['factor_manifest_sha256']=write(p/'output/FACTOR_MANIFEST.json',{'status':'FACTORS_WRITTEN_PENDING_ORACLE','matrix_decompositions':12,'factor_pairs':factors})
  scores=write(g/'output/scores.json',{'scope_sha256':gs['scope_sha256'],'cell_count':792,'rows':rows});selection={'family':'SVD','risk_by_family':{'SVD':1.,'CA':2.,'CD':3.},'rule':'source/expert/layer equal-weight, r1408 4S; exact ties SVD then CA then CD'}
  sel=write(g/'output/selection.json',{'scope_sha256':gs['scope_sha256'],**selection})
  gs['summary_sha256']=write(g/'output/SUMMARY.json',{'scope_sha256':gs['scope_sha256'],'status':'GRID_COMPLETE_PENDING_Q1_GP64_AND_ERROR_DECOMPOSITION','count':792,'scores_sha256':scores,'selection_sha256':sel,'selection':selection})
  sites=sorted(REMAINING);return {'grid':gs,'factor_producers':[ps],'shards':[{'id':'part1','sites':[list(x) for x in sites[:37]]},{'id':'part2','sites':[list(x) for x in sites[37:]]}]}
 def test_complete_two_shards_and_no_original_recompute(self):
  with tempfile.TemporaryDirectory() as d:
   spec=self.fixture(Path(d));plan=build(spec);self.assertEqual(plan['family'],'SVD');self.assertEqual(build(plan),plan)
   scope={'shard_id':'part1','sites':plan['shards'][0]['sites'],'max_matrix_decompositions':74,'selected_family':'SVD','checkpoints':{'high':{'sha256':'high'},'low':{'sha256':'low'}},'svd_driver':'gesvd'}
   self.assertEqual(len(validate_shard(plan,scope)),37)
   plan['shards'][0]['sites'][0]=list(SIX[0])
   with self.assertRaises(ValueError):validate_shard(plan,scope)
 def test_reject_overlap_missing_and_factor_mutation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);spec=self.fixture(root);spec['shards'][1]['sites'][0]=spec['shards'][0]['sites'][0]
   with self.assertRaises(ValueError):build(spec)
   spec=self.fixture(root/'second');(root/'second/six/output/SVD__high__04__r1408.pt').write_text('changed')
   with self.assertRaises(ValueError):build(spec)
 def test_shard_manifests_freeze_to_all80(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);spec=self.fixture(root);plan=build(spec);producers=list(spec['factor_producers'])
   for shard in plan['shards']:
    stage=root/shard['id'];producer=job(stage,'3' if shard['id']=='part1' else '4');pairs=[]
    for e,l in shard['sites']:
     path=stage/'output'/f'{e}_{l}.pt';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(e+str(l));ident={'family':'SVD','expert':e,'layer':l,'rank':1408,'checkpoint_sha256':e,'calibration_manifest_sha256':None,'svd_driver':'gesvd','factor_split':'U*S, Vh','execution_factor_dtype':'bfloat16','prefix_rank512_is_same_decomposition':True};pairs.append({'identity':ident,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
    producer['factor_manifest_sha256']=write(stage/'output/FACTOR_MANIFEST.json',{'factor_pairs':pairs,'matrix_decompositions':2*len(pairs),'status':'FACTORS_WRITTEN_PENDING_ORACLE'});producers.append(producer)
   frozen=assemble({'grid':spec['grid'],'selection_sha256':plan['selection_sha256'],'scores_sha256':plan['scores_sha256'],'factor_producers':producers,'checkpoint_sha256':{'high':'high','low':'low'},'clean_cal11_manifest_sha256':'unused'})
   self.assertEqual(frozen['status'],'ALL_80_FROZEN');self.assertEqual(len(frozen['factor_pairs']),80)
 def test_recovery_manifest_original_six(self):
  with tempfile.TemporaryDirectory() as d:
   import sys
   sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'work')) if False else None
   root=Path(d);spec=self.fixture(root);stage=root/'six';producer=spec['factor_producers'][0]
   producer['scope_sha256']=write(stage/'scope.json',{'mode':'six_site_svd_recover_high04','max_matrix_decompositions':10,'recovered_matrix_decompositions':2})
   write(stage/'receipt/RUN.json',{'scope_sha256':producer['scope_sha256'],'slurm_job_id':'2'})
   path=stage/'output/FACTOR_MANIFEST.json';m=json.loads(path.read_text());m.update(mode='six_site_svd',new_matrix_decompositions=10,recovered_matrix_decompositions=2,recovery_provenance={'source_job_id':'168881','source_scope_sha256':'9d472dea3acddedebc5bc706b60f3d496c65adce43612a0e51dd53387bb56071','source_factor_sha256':'827a3e57a36e2c719cf4950ea2b58c30ad7477e153d36342391e8f8840ff4011','source_factor_bytes':106695389,'reused_matrix_decompositions':2,'tensor_sha256':{'first_left':'beb4b534a83d336198f42df3a7f2325b5de930f442004418a73296667549aa21','first_right':'5de9f19b34d18de4d8b5ccf5fec53f5e0bfa6c0302ced35a4178915804ffd1f7','second_left':'7b5ac07dffefd50fb2729ee4bf17fc04f15ece7f85cd6fa0198836588adb2324','second_right':'706427929d069e7a0525bfcdddfda698bf1f17bf4f50e90ed240d497479183ba'}})
   producer['factor_manifest_sha256']=write(path,m);self.assertEqual(build(spec)['family'],'SVD')
   m['new_matrix_decompositions']=12;producer['factor_manifest_sha256']=write(path,m)
   with self.assertRaises(ValueError):build(spec)
if __name__=='__main__':unittest.main()
