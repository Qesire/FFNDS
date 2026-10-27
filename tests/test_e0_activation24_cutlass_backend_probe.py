import importlib.util,unittest,json
from pathlib import Path
s=importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'tools/e0_activation24_cutlass_backend_probe.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_budget_correctness_and_failure_atomicity(self):
  counts=dict.fromkeys(m.LIMITS,0);m.charge(counts,{'cutlass_compress':1,'cutlass_reconstruct':1});old=counts.copy()
  with self.assertRaises(ValueError):m.charge(counts,{'cutlass_compress':30,'cusparselt_compress':1})
  self.assertEqual(counts,old)
  per={'random_tensor_init':2,'stable_select':1,'triton_select':1+7+7,'cutlass_compress':1+7+7,'cusparselt_compress':1+7+7,'cutlass_reconstruct':1,'dense_gemm':1+7,'cutlass_sparse_gemm':1+7+7,'cusparselt_sparse_gemm':1+7+7,'search_api':0,'model_calls':0,'fits':0,'video_generations':0}
  self.assertEqual({k:v*2 for k,v in per.items()},m.LIMITS)
 def test_original_cpu_cutlass_pack_roundtrip(self):
  import torch
  from torch.sparse._semi_structured_conversions import sparse_semi_structured_from_dense_cutlass,sparse_semi_structured_to_dense_cutlass
  q=torch.tensor([1,0,-2,0,0,3,0,4,0,0,0,0,0,0,0,-5],dtype=torch.bfloat16);dense=q.repeat(32,4)
  packed,meta=sparse_semi_structured_from_dense_cutlass(dense);actual=sparse_semi_structured_to_dense_cutlass(packed,meta)
  self.assertEqual(tuple(packed.shape),(32,32));self.assertEqual(meta.dtype,torch.int16);self.assertTrue(torch.equal(actual,dense))
 def test_user_authorization_and_scope(self):
  plan=json.loads((Path(__file__).resolve().parents[1]/'config/e0_activation24_cutlass_backend_probe_20261027_v1.json').read_text());plan['assets_finalized']=True
  active={'user_continuation_authorization_20261027':{'id':'USER_RESUME_OFFLINE_IFC_20261027','status':'ACTIVE_CURRENT_USER_DIRECTED_SESSION','new_scope_registration_allowed':True,'max_simultaneously_assigned_idle_GPUs':4},'experiment_authority':{'current_user_authorized_scopes':[{'id':m.ID,'protocol_sha256':'digest','execution_status':'AUTHORIZED_NOT_STARTED'}]}}
  m.authorize(plan,'digest',active);active['user_continuation_authorization_20261027']['status']='REVOKED'
  with self.assertRaises(ValueError):m.authorize(plan,'digest',active)
 def test_monitor_violation_and_original_exception_priority(self):
  class Monitor:
   def close(self):pass
   def report(self):return {'violations':['foreign process']}
  report={'status':'COMPLETE'};error=m.finish_monitor(report,Monitor());self.assertEqual(report['status'],'FAILED_GPU_MONITOR')
  with self.assertRaises(RuntimeError):m.raise_monitor_if_unmasked(error,False)
  m.raise_monitor_if_unmasked(error,True)
 def test_padding_constraints_and_tie_order(self):
  self.assertEqual(((31668+31)//32)*32,31680)
  for _,k,_ in m.SHAPES:self.assertEqual(k%64,0)
  self.assertEqual([i for i,r in enumerate(m.tie_rank([1,-1,1,-1])) if r<2],[0,1])
if __name__=='__main__':unittest.main()
