import unittest,tempfile,json,hashlib
from unittest.mock import patch
from pathlib import Path
from quality_wan_proof import verify

def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x),encoding='utf8');return hashlib.sha256(p.read_bytes()).hexdigest()
class WanProofTest(unittest.TestCase):
 def test_real_schema_forced_control_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);stage=root/'stage';stage.mkdir();code={}
   for name in ('wan_comparison_job.py','icc_native_job.py','wan_comparison_job.sh'):
    p=stage/name;p.write_text('fixture');code[name]=hashlib.sha256(p.read_bytes()).hexdigest()
   scope=put(stage/'scope.json',{'force_dense':True,'bundle_sha256':code,'arm':'ICC'})
   bundle=root/'bundle.json';digest=put(bundle,{'stage':str(stage),'files':{'scope.json':scope}})
   rec={'arm':'ICC','generation_proof_key':'x','generation_proof_sha256':digest}
   with self.assertRaisesRegex(ValueError,'forced Dense'):verify(rec,{'generation_proof_bundles':{'x':{'path':str(bundle),'sha256':digest}}})
 def test_mutated_bundle_rejected_before_execution(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'p';p.write_text('{}')
   with self.assertRaisesRegex(ValueError,'proof bundle identity'):verify({'generation_proof_key':'x','generation_proof_sha256':'0'*64},{'generation_proof_bundles':{'x':{'path':str(p),'sha256':'0'*64}}})
 def test_complete_icc_trajectory_and_changed_event(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);stage=root/'stage';stage.mkdir();ref=root/'ref';ref.mkdir();out=stage/'output';out.mkdir();src='vbi_1013';noise={'sha256':'1'*64,'shape':[16,21,58,104],'dtype':'torch.bfloat16'};schedule=[{'timestep':i,'sigma':i} for i in range(4)]
   code={}
   for name in ('wan_comparison_job.py','icc_native_job.py','wan_comparison_job.sh'):
    f=stage/name;f.write_text('fixture');code[name]=hashlib.sha256(f.read_bytes()).hexdigest()
   noise_path=root/'noise.json';noise_hash=put(noise_path,{src:noise})
   checkpoints={'high':{'sha256':'a'*64},'low':{'sha256':'b'*64}};recipe='ee780352c8cf1c1e4efc8aea7c354b3be225c525141942cbd9c15d24cabae5b6'
   rs=put(ref/'scope.json',{'checkpoints':checkpoints,'recipe_binding':{'recipe_sha256':recipe}})
   condition={'record_id':src,'image_sha256':'c'*64,'prompt_sha256':'d'*64,'source_cluster_id':'group','seed':42};rc=put(ref/'v2_conditions.json',[condition]);refs={'scope.json':rs,'v2_conditions.json':rc}
   ps={'scope_id':'fixture','force_dense':False,'arm':'ICC','cache_mode':'4S','bundle_sha256':code,'sources':[src],'max_nfe':4,'schedule':schedule,'v2_reference_dir':str(ref),'reference_sha256':refs,'dense_initial_noise':{'path':str(noise_path),'sha256':noise_hash}}
   files={'scope.json':put(stage/'scope.json',ps)};ss=files['scope.json']
   files['REVIEW.json']=put(stage/'REVIEW.json',{'reviewer_agent_id':'reviewer','author_agent_id':'author','scope_sha256':ss,'recheck_disposition':'APPROVE'})
   files['ACTIVE_RESEARCH.json']=put(stage/'ACTIVE_RESEARCH.json',{'fixture':{'scope_sha256':ss,'review_sha256':files['REVIEW.json'],'runnable':True}})
   files['receipt/RUN.json']=put(stage/'receipt/RUN.json',{'scope_sha256':ss,'allocation':{'job_id':'123'}})
   files['receipt/EXIT.json']=put(stage/'receipt/EXIT.json',{'exit_code':0,'slurm_job_id':'123'})
   video=out/(src+'.mp4');video.write_bytes(b'video');vd=hashlib.sha256(video.read_bytes()).hexdigest();latent=out/(src+'_latent.pt');latent.write_bytes(b'latent');ld=hashlib.sha256(latent.read_bytes()).hexdigest()
   sites=[{'step':i//40,'layer':i%40,'expert':'high' if i//40<2 else 'low','kind':'anchor' if i//40%2==0 else 'hit'} for i in range(160)]
   row={'source':src,'arm':'ICC','seed':42,'schedule':schedule,'initial_noise':noise,'dispatch':[{'step':i,'expert':i//2} for i in range(4)],'sites':sites,'video':{'mp4_sha256':vd,'decoded_rgb24_sha256':'rgb','frames':81,'width':832,'height':464,'fps':16},'latent_file':{'path':str(latent),'sha256':ld},'final_latent':{'sha256':'tensor'}}
   rel='output/'+src+'.json';files[rel]=put(stage/rel,row);files['output/SUMMARY.json']=put(out/'SUMMARY.json',{'status':'COMPLETE','arm':'ICC','nfe':4,'records':[{'source':src,'receipt_sha256':files[rel]}]})
   events=[]
   for i in range(4):events.extend([{'event':'NFE_ATTEMPT','source':src,'step':i,'attempt':i+1},{'event':'NFE_COMPLETED','source':src,'step':i,'expert':i//2}])
   log=out/'NFE.jsonl';log.write_text(''.join(json.dumps(e)+'\n' for e in events));files['output/NFE.jsonl']=hashlib.sha256(log.read_bytes()).hexdigest()
   ident={'source':src,'seed':42,'image_sha256':'c'*64,'prompt_sha256':'d'*64,'source_cluster_id':'group','checkpoint_sha256':{'high':'a'*64,'low':'b'*64}}
   admission=root/'admission.json';ad=put(admission,{'slots':[{'source':src,'input_identity':ident,'initial_noise':noise,'scheduler':schedule}]})
   contract=root/'contract.json';ch=put(contract,{'dev_cohort_admission_sha256':ad});audit=root/'audit.json';ah=put(audit,{'status':'READ_ONLY_AUDIT_COMPLETE','exit_code':0,'output_sha256':ad})
   exe=root/'ffmpeg';exe.write_bytes(b'exe');eh=hashlib.sha256(exe.read_bytes()).hexdigest()
   bundle=root/'bundle.json';bh=put(bundle,{'stage':str(stage),'files':files,'reference_sha256':refs,'recipe_sha256':recipe})
   rec={'arm':'ICC','source':src,'generation_proof_key':'x','generation_proof_sha256':bh,'input_identity':{**ident,'initial_noise':noise,'scheduler':schedule},'image_sha256':'c'*64,'prompt_sha256':'d'*64,'source_cluster_id':'group','video_path':str(video),'video_sha256':vd}
   scope={'comparison':'main','generation_proof_bundles':{'x':{'path':str(bundle),'sha256':bh}},'dense_admission_path':str(admission),'dense_admission_sha256':ad,'paired_contract_path':str(contract),'paired_contract_sha256':ch,'dense_admission_receipt_path':str(audit),'dense_admission_receipt_sha256':ah,'ffmpeg_path':str(exe),'ffmpeg_sha256':eh}
   with patch('quality_wan_proof.decoded',return_value='rgb'),patch('quality_wan_proof.tensor_storage_sha',return_value='tensor'):
    self.assertEqual(verify(rec,scope)['status'],'PASS')
    events[1]['step']=3;log.write_text(''.join(json.dumps(e)+'\n' for e in events));files['output/NFE.jsonl']=hashlib.sha256(log.read_bytes()).hexdigest();bh=put(bundle,{'stage':str(stage),'files':files,'reference_sha256':refs,'recipe_sha256':recipe});rec['generation_proof_sha256']=bh;scope['generation_proof_bundles']['x']['sha256']=bh
    with self.assertRaisesRegex(ValueError,'NFE attempt'):verify(rec,scope)
if __name__=='__main__':unittest.main()
