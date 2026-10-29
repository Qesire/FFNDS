import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_all160_video_evidence as e
class Tests(unittest.TestCase):
 def example(self):
  plan={'id':'scope','case':{'native_source_ordinal':1040}};state={'sha256':'a','bytes':1,'raw_tensor_sha256':'b','shape':[1],'dtype':'bf16'};video={'frames':81,'fps':16,'size':[832,464],'bytes':1,'sha256':'f'*64};phases=[]
  for i,mode in enumerate(('bf16','bf16_cache','dense32','sharq')):phases.append({'run_id':str(i),'mode':mode,'actual_full_FFN_sites':160,'actual_FFN_linear_dispatches':320,'states':{k:dict(state) for k in e.STATE_NAMES},'original_prefetch':{'original_FFN_prefetch_observations':162,'original_FFN_prefetch_skips':0},'video':dict(video),'weight_cache':{'cache_entries':0 if i<1 else 160}})
  counts={'video_generations':4,'original_FFN_prefetch_observations':648,'sharq_static_W_materializations':160,'sharq_static_W_quantizations':160,'dense32_prepare':320,'dense32_gemm':320,'sharq_prepare':320,'sparse_gemm':320,'dense_residual_gemm':320,'output_scale':640,'bias_add':640}
  run={'id':'scope','protocol_sha256':'p','job_spec_sha256':'j','status':'COMPLETE_SHARQ_WAN_ALL160_EXIT_PENDING','source_ordinal':1040,'seed':42,'quality_acceptance':False,'phases':phases,'operation_ledger':{'predebited':counts,'observed':dict(counts)},'GPU_monitor':{'sampled_isolation_pass':True,'violations':[]}}
  receipt={'id':'scope','protocol_sha256':'p','spec_sha256':'j','run_sha256':'r','process_exited':True,'child_returncode':0,'timeout':False,'automatic_retry':False};return plan,run,receipt
 def test_actual_identity_and_all_phases_required(self):
  p,r,x=self.example();self.assertEqual(e.verify_record(p,r,x,'r','p')['status'],'RECORD_PREDICATES_PASS_REQUIRES_ARTIFACT_BINDINGS')
  x['child_returncode']=1
  with self.assertRaises(ValueError):e.verify_record(p,r,x,'r','p')
 def test_wrong_coverage_residual_or_noise_rejected(self):
  p,r,x=self.example()
  for mutation in ('coverage','residual','noise'):
   z=copy.deepcopy(r)
   if mutation=='coverage':z['phases'][2]['actual_FFN_linear_dispatches']=319
   elif mutation=='residual':z['operation_ledger']['observed']['dense_residual_gemm']=319
   else:z['phases'][3]['states']['raw_noise_FP32']['raw_tensor_sha256']='different'
   with self.assertRaises(ValueError):e.verify_record(p,z,x,'r','p')
if __name__=='__main__':unittest.main()
