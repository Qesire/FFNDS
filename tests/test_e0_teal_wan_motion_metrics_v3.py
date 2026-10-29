import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('test_teal_motion',ROOT/'tools/e0_teal_wan_motion_metrics_v3.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_work_budget_and_three_variants(self):
  self.assertEqual(m.LIMITS['video_results'],3);self.assertEqual(m.LIMITS['decoded_frames'],243)
  self.assertEqual(m.LIMITS['RAFT_forwards'],3*40);self.assertEqual(m.LIMITS['RAFT_iterations'],3*40*20)
  self.assertEqual(m.MODES,('dense','threshold25','threshold50'))
 def test_missing_parent_blocks_draft(self):
  with patch.object(m,'parent_data',side_effect=FileNotFoundError):
   with self.assertRaises(FileNotFoundError):m.draft(Path('/tmp'),0)
 def test_same_source_ratio_and_zero_baseline(self):
  def row(mode,x):return {'ordinal':1040,'mode':mode,'summary':{'mean_top5percent_flow_px_per_2frame_pair':x,'five_fixed_8pair_windows':[{'mean_top5percent_flow_px':x}]*5}}
  r=m.summarize({'shard':0},[row('dense',2),row('threshold25',1),row('threshold50',0)])
  self.assertEqual(r['per_source'][0]['candidate_to_same_fresh_Dense_ratios']['threshold25']['value'],0.5)
  r=m.summarize({'shard':0},[row('dense',0),row('threshold25',1),row('threshold50',0)])
  self.assertIsNone(r['mean_of_source_ratios']['threshold25'])
if __name__=='__main__':unittest.main()

class SnapshotTests(unittest.TestCase):
 def test_scoped_snapshot_counts_partial_and_detects_mutation(self):
  import tempfile
  from e0_teal_output_budget_v3 import SnapshotBudget
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);out=root/'out';snap=out/'EXECUTED_SOURCES';snap.mkdir(parents=True);proto=root/'p';proto.write_bytes(b'proto');(snap/'protocol.json').write_bytes(b'proto')
   (snap/'source').write_bytes(b'source');(out/'atomic.tmp').write_bytes(b'partial')
   plan={'repo_sources':[{'path':'source','bytes':6,'sha256':m.base.sha(snap/'source')}]};budget=SnapshotBudget(out,plan,proto,1000)
   m._scope_budget=budget
   try:
    before=Path.rglob;self.assertEqual(m.output_bytes(out),18);self.assertIs(Path.rglob,before)
    (snap/'source').write_bytes(b'change')
    with self.assertRaises(ValueError):budget.final()
   finally:m._scope_budget=None
 def test_snapshot_rejects_symlink(self):
  import tempfile
  from e0_teal_output_budget_v3 import SnapshotBudget
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);out=root/'out';snap=out/'EXECUTED_SOURCES';snap.mkdir(parents=True);proto=root/'p';proto.write_bytes(b'x')
   (snap/'protocol.json').write_bytes(b'x');(snap/'link').symlink_to(proto)
   budget=SnapshotBudget(out,{'repo_sources':[]},proto,100)
   with self.assertRaises(ValueError):budget.initialize()

class PairTests(unittest.TestCase):
 def test_raft_exact_40_stride2_windows_and_reject_39(self):
  row={'scores':[float(i) for i in range(40)],'params':{'thres':10.875,'count_num':10},'official_binary':True}
  summary=m.base.validate_raft(row)
  self.assertEqual(len(summary['five_fixed_8pair_windows']),5)
  with self.assertRaises(ValueError):m.base.validate_raft(dict(row,scores=row['scores'][:-1]))

class FailureBudgetTests(unittest.TestCase):
 def test_failure_restores_local_budget_and_retains_error(self):
  import tempfile,json
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as t:
   hf=Path(t);proto=hf/'protocol.json';proto.write_text('{}');out=hf/'out';out.mkdir();(out/'RUN.json').write_text('{"status":"STARTED"}')
   plan={'output_HF_relative':'out','repo_sources':[]}
   def read(path):return plan if Path(path)==proto else json.loads(Path(path).read_text())
   orig=Path.rglob
   with patch.object(m.base,'read',side_effect=read),patch.object(m,'_run_original',side_effect=RuntimeError('synthetic')):
    with self.assertRaises(RuntimeError):m.run(SimpleNamespace(protocol=proto,hf_root=hf))
   self.assertIsNone(m._scope_budget);self.assertIs(Path.rglob,orig)
   self.assertEqual(json.loads((out/'RUN.json').read_text())['output_audit_error'],'RuntimeError')
