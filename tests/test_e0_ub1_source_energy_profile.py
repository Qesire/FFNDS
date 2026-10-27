import importlib.util,sys,unittest,json,tempfile
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'tools'));spec=importlib.util.spec_from_file_location('source_energy',root/'tools/e0_ub1_source_energy_profile.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_predebit_atomic_and_ledger(self):
  counts=dict.fromkeys(m.LIMITS,0)
  for _ in range(96):m.reserve(counts)
  self.assertEqual(counts,m.LIMITS)
  with self.assertRaises(ValueError):m.reserve(counts)
 def test_bmm_delegate_and_failure_count(self):
  seen=[];result=object()
  def original(*a,**kw):seen.append((a,kw));return result
  observed={'FP32_batched_bmm':0};self.assertIs(m.instrument_bmm(original,observed)(1,2,foo=3),result);self.assertEqual(seen,[((1,2),{'foo':3})]);self.assertEqual(observed['FP32_batched_bmm'],1)
  observed['FP32_batched_bmm']=2592
  with self.assertRaises(RuntimeError):m.instrument_bmm(original,observed)(1,2)
 def test_reference_tolerance_and_invalid(self):
  v=np.arange(216,dtype=np.float64);self.assertTrue(m.reference_compare(v,v)['bitwise_equal']);w=v.copy();w[40]+=1;self.assertFalse(m.reference_compare(w,v)['within_declared_tolerance'])
  with self.assertRaises(ValueError):m.reference_compare(np.array([1]),v)
 def test_reference_failure_persists_then_stops(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp); report={'positions':[]}; profile={'records':[{'index':i} for i in range(24)]}; position={'step':0,'block':39,'reference_comparison':{'within_declared_tolerance':False,'support_equal_to_original_C_mean':False}}
   with self.assertRaisesRegex(ValueError,'reference reproduction failed'):
    m.save_position_and_require_reference(out,report,profile,position)
   self.assertEqual(len(json.loads((out/'PROFILE.json').read_text())['records']),24)
   self.assertFalse(json.loads((out/'RUN.json').read_text())['positions'][0]['reference_comparison']['within_declared_tolerance'])
   position['reference_comparison']['within_declared_tolerance']=True
   m.save_position_and_require_reference(out,report,profile,position)
 def test_cluster_profile_source_unit(self):
  from ffnds_calibration_design import diagnose
  records=[]
  for i in range(24):records.append({'source_id':str(i),'image_id':str(i),'step':0,'block':39,'source_state':'teacher_input','statistic_semantics':'native_gate_times_W2_group_contribution_l2_squared','group_energy':list(map(float,range(216)))})
  out=diagnose({'schema':'ffnds.calibration.source_statistics.v2','evidence_kind':'ACTUAL_NATIVE_OBSERVATION','records':records},keep_groups_by_step=[54,108,108,162],bootstrap=2,seed=42);value=out['positions']['step0_block39_teacher_input'];self.assertEqual(value['independent_image_clusters'],24);self.assertEqual(value['leave_one_cluster_out_jaccard']['min'],1.)
 def test_exact_authorization_and_revocation(self):
  plan=json.loads((root/'config/e0_ub1_source_energy_profile_20261027_v1.json').read_text());plan['assets_finalized']=True
  active={'user_continuation_authorization_20261027':{'id':'USER_RESUME_OFFLINE_IFC_20261027','status':'ACTIVE_CURRENT_USER_DIRECTED_SESSION','max_simultaneously_assigned_idle_GPUs':4,'new_scope_registration_allowed':True},'experiment_authority':{'current_user_authorized_scopes':[{'id':m.ID,'protocol_sha256':'digest','execution_status':'AUTHORIZED_NOT_STARTED'}]}}
  m.authorize(plan,'digest',active);active['user_continuation_authorization_20261027']['status']='REVOKED'
  with self.assertRaises(ValueError):m.authorize(plan,'digest',active)
 def test_wrong_scope_rejected(self):
  with self.assertRaises(ValueError):m.validate_plan({'cases':[],'sources':[]})
if __name__=='__main__':unittest.main()
