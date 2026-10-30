import json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_input_video_quality_visual_cpu_visual_v2 as v
import e0_chipmunk_input_video_quality_metric_verify_visual_v2 as verifier
class Tests(unittest.TestCase):
 def test_only_parts_2_and_3(self):
  self.assertIn('VISUAL_V2',v.identity(2));self.assertIn('VISUAL_V2',v.identity(3))
  with self.assertRaises(ValueError):v.identity(0)
 def test_run_no_active(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'p.json';p.write_text(json.dumps({'id':'S'}));a=SimpleNamespace(protocol=p,hf_root=Path(d))
   with patch.object(v,'validate'),patch.object(v.base,'read',side_effect=lambda q: {'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},'experiment_authority':{'current_user_authorized_scopes':[]}} if str(q).endswith('ACTIVE_RESEARCH.json') else json.loads(Path(q).read_text())):
    with self.assertRaisesRegex(ValueError,'ACTIVE'):v._run(a)
   self.assertFalse((Path(d)/'RUN.json').exists())
 def test_verify_no_active(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'p.json';p.write_text(json.dumps({'id':'S'}));job=root/'job';job.write_text('{}');a=SimpleNamespace(kind='visual',protocol=p,job_spec=job,hf_root=root)
   with patch.object(verifier,'HF',root),patch.object(verifier,'ROOT',root),patch.object(verifier,'load',return_value=v),patch.object(v.base,'read',side_effect=lambda q: {'experiment_authority':{'current_user_authorized_scopes':[]}} if str(q).endswith('ACTIVE_RESEARCH.json') else json.loads(Path(q).read_text())):
    with self.assertRaisesRegex(ValueError,'ACTIVE'):verifier.verify(a)
   self.assertFalse((root/'RUN.json').exists())
 def test_verify_closed_rejects_bad_parent_exit(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run=root/'results/RUN.json';run.parent.mkdir();run.write_text('{}');ex=root/'EXIT_RECEIPT.json';ex.write_text(json.dumps({'child_returncode':1}));protocol=root/'p.json';protocol.write_text(json.dumps({'part':2,'id':'S'}));a=SimpleNamespace(hf_root=root,protocol=protocol,run=run,exit=ex)
   with patch.object(v,'validate'),patch.object(v,'output',return_value='results'),patch.object(v.base,'binding',return_value={'path':'results/RUN.json','sha256':'x','bytes':2}):
    with self.assertRaises((ValueError,KeyError,TypeError)):v._verify_closed(a)
   self.assertFalse((root/'verification/RUN.json').exists())
 def test_foreign_affinity_rejected(self):
  ex=v.execution
  with patch.object(ex.os,'sched_getaffinity',return_value={16,17}),patch.object(ex,'affinity',return_value=[12,13]),patch.object(ex,'ROOT',Path('/tmp')):
   with self.assertRaisesRegex(ValueError,'affinity'):
    ex.verify_job(SimpleNamespace(job_spec=Path('/tmp/j'),protocol=Path('/tmp/p'),hf_root=Path('/tmp')),{'part':2},{'launch_spec_sha256':'a'},SimpleNamespace(sha=lambda _: 'a',read=lambda _:{}),'visual','run',v.SELF,{})
if __name__=='__main__':unittest.main()
