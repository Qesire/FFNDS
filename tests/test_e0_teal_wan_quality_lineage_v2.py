import importlib.util,json,os,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('test_lineage_v2',ROOT/'tools/e0_teal_wan_quality_lineage_v2.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_child_ledger_persists_before_launch_and_failure(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  proof={'protocol':{'path':m.PARENT.format(0)}}
  cases=[(subprocess.TimeoutExpired('child',300),'TIMED_OUT_STARTED'),(OSError('launch'),'LAUNCH_OR_IO_FAILED_UNKNOWN'),(SimpleNamespace(returncode=1,stderr='bad',stdout=''),'FAILED_OR_INTERRUPTED_LAUNCH_UNKNOWN'),(SimpleNamespace(returncode=0,stderr='',stdout='{}'),'FAILED_OR_INTERRUPTED_LAUNCH_UNKNOWN')]
  for result,state in cases:
   record={};snapshots=[];ledger=m.ChildLedger(record,lambda:snapshots.append(dict(record)),'SYNTHETIC')
   def fake(*a,**kw):
    self.assertEqual(snapshots[-1]['predebited'],1);self.assertEqual(snapshots[-1]['launch_state'],'IN_FLIGHT_OR_LAUNCH_UNKNOWN')
    if isinstance(result,BaseException):raise result
    return result
   with patch.object(m.subprocess,'run',side_effect=fake):
    with self.assertRaises(Exception):m.closed_rows(proof,Path('/tmp'),0,ledger)
   self.assertEqual(record['predebited'],1);self.assertEqual(record['completed'],0);self.assertEqual(record['launch_state'],state);self.assertIsNotNone(record['error'])
   with self.assertRaises(ValueError):ledger.call(lambda:None)
 def test_contract_binds_generation_interpreter_and_cpu(self):
  c=m.contract();self.assertEqual(c['argv'][1:3],['-I','-B']);self.assertEqual(c['max_child_processes_per_parent_check'],1);self.assertEqual(c['CUDA_VISIBLE_DEVICES'],'')
 def test_isolated_stdlib_actual_closed_parent(self):
  hf=Path('/home/xiang/.cache/huggingface');p=ROOT/m.PARENT.format(0)
  if not p.exists():self.skipTest('Actual closed parent not present')
  code='''import importlib.util,importlib.metadata,json,sys\nfrom pathlib import Path\ntry: importlib.metadata.version("torch")\nexcept importlib.metadata.PackageNotFoundError: pass\nelse: raise AssertionError("torch metadata unexpectedly visible in isolated interpreter")\ns=importlib.util.spec_from_file_location("v2",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nhf=Path(sys.argv[2]);p=m.parent_data(hf,0);rows=m.closed_rows(p,hf,0)\nassert [x["mode"] for x in rows]==["dense","threshold25","threshold50"]\nassert not {"torch","numpy"}&set(sys.modules)\nprint(json.dumps({"rows":len(rows),"torch":False,"numpy":False}))'''
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='');r=subprocess.run([str(ROOT/'.venv/bin/python'),'-I','-S','-B','-c',code,str(ROOT/m.SELF),str(hf)],capture_output=True,text=True,env=env,timeout=300)
  self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),{'rows':3,'torch':False,'numpy':False})
 def test_isolated_real_metric_lineage_entry(self):
  hf=Path('/home/xiang/.cache/huggingface')
  code='''import importlib.util,json,sys\nfrom pathlib import Path\ns=importlib.util.spec_from_file_location("metric",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\np=m.base.read(Path(sys.argv[2]));h=Path(sys.argv[3]);bridge=m.video_parent if hasattr(m,"video_parent") else m.motion.video_parent\nr={};led=bridge.ChildLedger(r,lambda:None,"TEST");m.verify_lineage(p,h,led)\nassert r["predebited"]==r["completed"]==1 and not {"torch","numpy"}&set(sys.modules)\nprint("PASS")'''
  for kind,stem in [('raft','motion_metrics_v2'),('quality','full_quality_v2')]:
   plan=ROOT/f'config/e0_teal_wan_{kind}_v2_part0_20261029_pending_review_v1.json'
   if not plan.exists():self.skipTest('Actual closed parent protocol not present')
   r=subprocess.run([str(ROOT/'.venv/bin/python'),'-I','-S','-B','-c',code,str(ROOT/f'tools/e0_teal_wan_{stem}.py'),str(plan),str(hf)],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=300)
   self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(r.stdout.strip(),'PASS')
if __name__=='__main__':unittest.main()
