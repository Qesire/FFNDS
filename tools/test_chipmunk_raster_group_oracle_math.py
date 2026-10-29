import unittest
import subprocess,sys,tempfile,pathlib,textwrap
import torch
from chipmunk_raster_group_oracle_math import groups,selected_delta,project,selection_debits
from e0_chipmunk_raster_group_oracle import reserve_counts,perform_charged

class RasterOracleTest(unittest.TestCase):
 def test_partition_and_tail(self):
  for name in ('raster','random'):
   g=groups(260,name,128,101); self.assertEqual([len(x) for x in g],[128,128,4]); self.assertEqual(sorted(torch.cat(g).tolist()),list(range(260)))
  self.assertEqual(len(groups(260,'global')),1)
 def test_equal_budget_and_local_advantage(self):
  old=torch.zeros((4,2),dtype=torch.bfloat16); new=torch.tensor([[3,0],[3,0],[0,3],[0,3]],dtype=torch.bfloat16)
  glob, a=selected_delta(old,new,1,'global',2); local,b=selected_delta(old,new,1,'raster',2)
  self.assertEqual(a['selected_elements'],b['selected_elements']); self.assertTrue(torch.equal(local,new)); self.assertFalse(torch.equal(glob,new)); self.assertGreater(b['selected_energy'],a['selected_energy'])
 def test_tie_determinism(self):
  h=torch.zeros((4,3),dtype=torch.bfloat16); _,a=selected_delta(h,h,1,'random',2,101); _,b=selected_delta(h,h,1,'random',2,101); self.assertEqual(a,b)
 def test_selection_budget_and_interrupt_semantics(self):
  a=selection_debits(260,2,"raster",128); self.assertEqual(a["score_sort_groups"],3); self.assertEqual(a["indices_D2H_logical_bytes"],48); self.assertEqual(a["group_rows_H2D_logical_bytes"],2080)
  budget=dict(a); reserved={}; completed={}
  def fail(): raise RuntimeError("interrupted mid-group")
  with self.assertRaises(RuntimeError):
   perform_charged(a,lambda c:reserve_counts(reserved,budget,c),fail,lambda:None,lambda c:completed.update(c))
  # Failure in actual charged wrapper preserves reservation and no completion.
  self.assertEqual(reserved,a); self.assertEqual(completed,{})
  with self.assertRaises(ValueError):
   reserve_counts(reserved,budget,{"selection":0,"score_sort_groups":1})
  self.assertEqual(reserved,a)
 def test_actual_run_rejects_before_torch_or_assets(self):
  tool_dir=str(pathlib.Path(__file__).resolve().parent)
  script=textwrap.dedent("""
   import json,pathlib,sys,types
   sys.path.insert(0,sys.argv[1])
   import e0_chipmunk_raster_group_oracle as m
   base=pathlib.Path(sys.argv[2]); m.ROOT=base
   (base/'research').mkdir()
   (base/'research/ACTIVE_RESEARCH.json').write_text(json.dumps({'experiment_authority':{'current_user_authorized_scopes':[]},'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True}}))
   proto=base/'protocol.json';proto.write_text(json.dumps({'assets_finalized':True,'id':'E0_CHIPMUNK_RASTER_GROUP_ORACLE_20261030_PART0'}))
   job=base/'job.json';job.write_text('{}')
   a=types.SimpleNamespace(protocol=proto,job_spec=job,hf_root=base/'unavailable-assets')
   try: m.run(a)
   except ValueError as e: assert str(e)=='authority'
   else: raise AssertionError('missing active accepted')
   assert 'torch' not in sys.modules and not a.hf_root.exists()
   m.authorize=lambda p,d:{'launch_spec_sha256':'bad'}
   try: m.run(a)
   except ValueError as e: assert str(e)=='job/isolated'
   else: raise AssertionError('wrong job accepted')
   assert 'torch' not in sys.modules and not a.hf_root.exists()
  """)
  with tempfile.TemporaryDirectory() as d:
   r=subprocess.run([sys.executable,'-I','-B','-c',script,tool_dir,d],capture_output=True,text=True)
   self.assertEqual(r.returncode,0,r.stderr)
 def test_project(self):
  d=torch.tensor([[2,0]],dtype=torch.bfloat16); w=torch.eye(2,dtype=torch.bfloat16); y=torch.tensor([[1,2]],dtype=torch.bfloat16); gate=torch.ones(2,dtype=torch.bfloat16)
  self.assertTrue(torch.equal(project(y,d,w,gate),torch.tensor([[3,2]],dtype=torch.bfloat16)))
if __name__=='__main__': unittest.main()
