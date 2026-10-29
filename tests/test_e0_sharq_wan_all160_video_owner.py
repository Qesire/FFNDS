import sys,unittest,tempfile,hashlib
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_all160_video_owner as m
class Tests(unittest.TestCase):
 def test_budget_preserves_all_real_physical_calls(self):
  b=m.limits();self.assertEqual(b['video_generations'],4);self.assertEqual(b['native_FFN_calls'],640);self.assertEqual(b['native_FFN_linears'],1280);self.assertEqual(b['sharq_static_W_quantizations'],160);self.assertEqual(b['original_FFN_prefetch_observations'],648);self.assertEqual(b['dense32_gemm'],320);self.assertEqual(b['sparse_gemm'],320);self.assertEqual(b['dense_residual_gemm'],320)
 def test_asset_hash_budget_predebits_and_retains_failure(self):
  report={};snap=[]
  with tempfile.TemporaryDirectory() as tmp:
   f=Path(tmp)/'file';f.write_bytes(b'abcdef')
   with self.assertRaisesRegex(ValueError,'asset SHA read budget'):
    with patch.object(m,'ASSET_READ_MAX',6):
     with m.AssetHashBudget(report,lambda:snap.append(dict(report['asset_hash_reads']))) as budget:
      self.assertEqual(budget.hash(f),hashlib.sha256(b'abcdef').hexdigest());budget.hash(f)
  self.assertEqual(snap[0]['reserved_bytes'],6);self.assertEqual(snap[-1]['completed_bytes'],6);self.assertEqual(snap[-1]['inflight'],'UNKNOWN_PARTIAL_FILE')
 def test_interrupted_hash_restores_hooks_and_marks_unknown(self):
  import local_python_package_mirror as mirror
  old_sha,old_digest=m.native.sha,mirror.digest;report={};states=[]
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'part';path.write_bytes(b'hello');orig=Path.open
   class Interrupted:
    def __enter__(self):self.f=orig(path,'rb');self.calls=0;return self
    def __exit__(self,*a):self.f.close()
    def read(self,n):
     self.calls+=1
     if self.calls==2:raise OSError('synthetic interrupted read')
     return self.f.read(n)
   with patch.object(Path,'open',lambda p,*a,**k:Interrupted() if p==path else orig(p,*a,**k)):
    with self.assertRaisesRegex(OSError,'interrupted read'):
     with m.AssetHashBudget(report,lambda:states.append(dict(report['asset_hash_reads']))) as budget:budget.hash(path)
  self.assertIs(m.native.sha,old_sha);self.assertIs(mirror.digest,old_digest);self.assertEqual(states[-1]['inflight'],'UNKNOWN_PARTIAL_FILE')
 def test_guard_enter_failure_preserves_run_and_closes_journals(self):
  class Journal:
   def __init__(self):self.closed=False
   def write(self,*a):pass
   def close(self):self.closed=True
  class Counter:
   def __init__(self,out,*a):self.out=out;self.files={};self.accounting_errors=[];self.journals=[]
   def path(self,p):return Path(p)
   def hold(self,*a):pass
   def open_journal(self,p):x=Journal();self.journals.append(x);return x
   def write_json(self,path,obj):Path(path).write_text(__import__('json').dumps(obj))
   def summary(self):return {'unconsumed_reserved_bytes':0,'accounting_errors':[]}
   def binding(self,*a):return {'path':'journal','bytes':0,'sha256':'0'*64}
  class Monitor:
   def __init__(self,*a):self.stopped=False
   def start(self):pass
   def stop(self):self.stopped=True;return {'fault':None,'thread_stopped':True}
  class BadGuard:
   def __enter__(self):raise RuntimeError('guard failed')
  with tempfile.TemporaryDirectory() as td:
   hf=Path(td);scope={'execution_status':'RUNNING_FIXED_SCOPE','native_group_active_slots':1,'gpu_launch_spec_sha256':'s'}
   plan={'id':'scope','part':0,'output_HF_relative':'out','case':{'native_source_ordinal':1040},'kernel_parent':{},'models':[],'runtime_versions':{},'run_ids':['a','b','c','d']}
   args=SimpleNamespace(protocol=hf/'p',local_root=hf,job_spec=hf/'job');holder={}
   def makecounter(*a):holder['counter']=Counter(*a);return holder['counter']
   def makemonitor(*a):holder['monitor']=Monitor(*a);return holder['monitor']
   module=SimpleNamespace(CheckedPackageFinder=lambda *a:None)
   with ExitStack() as stack:
    for target,name,value in ((m,'HF',hf),(m.native,'read',lambda p:plan),(m.native,'sha',lambda p:'digest'),(m,'authorize',lambda *a:scope),(m,'verify_job',lambda *a:None),(m,'require_cpu',lambda *a:{}),(m,'charged_assets',lambda *a:(hf,[])),(m.output_support,'reserve_emergency',lambda *a:None),(m.output_support,'UsageMonitor',makemonitor),(m.io,'WrittenBytes',makecounter),(m.io,'single_file_bound',BadGuard),(m.native,'NoHiF4',lambda:object()),(m.vm,'Ledger',lambda *a:SimpleNamespace()),(m,'limits',lambda:{})):
     stack.enter_context(patch.object(target,name,value))
    stack.enter_context(patch.dict(sys.modules,{'e0_b_final_block_capture_local_package':module}))
    with self.assertRaisesRegex(RuntimeError,'guard failed'):m.run(args)
   report=__import__('json').loads((hf/'out/RUN.json').read_text());self.assertEqual(report['status'],'FAILED_SHARQ_WAN_ALL160_PRESERVED_NO_RETRY');self.assertTrue(holder['monitor'].stopped);self.assertTrue(all(j.closed for j in holder['counter'].journals))
 def test_verifier_without_active_is_rejected(self):
  from types import SimpleNamespace
  with patch.object(m.native,'read',return_value={'experiment_authority':{'current_user_authorized_scopes':[]}}):
   with self.assertRaisesRegex(ValueError,'verifier ACTIVE'):m.authorize_verification({'id':'scope'},'hash',SimpleNamespace())
 def test_no_unfrozen_scope_authorization(self):
  with self.assertRaises(Exception):m.validate({'part':0,'status':'DRAFT_REVIEW_REQUESTED'},True)
 def test_group_scope_and_source_ids(self):
  self.assertNotEqual(m.scope_id(0),m.scope_id(1));self.assertEqual(m.DESIGN['sources'],[1040,979]);self.assertEqual(len(set(m.run_ids(0))),4)
if __name__=='__main__':unittest.main()
