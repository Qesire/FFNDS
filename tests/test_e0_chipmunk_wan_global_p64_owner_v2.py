import sys,unittest,subprocess,tempfile,textwrap,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_wan_global_p64_owner_v2 as owner
class Test(unittest.TestCase):
 def test_frozen_scope_resources_and_fixed_cost(self):
  d=owner.limits();self.assertEqual(owner.RES['GPU'],1);self.assertEqual(owner.RES['new_HF_bytes'],512<<20);self.assertEqual(owner.DESIGN['source_ordinals'],[898,96]);self.assertEqual(len(owner.run_ids(0)),4)
  self.assertEqual(d['native_FFN_calls'],640);self.assertEqual(d['native_FFN_linears'],1280);self.assertEqual(d['p64_score'],80);self.assertEqual(d['p64_W1_gemm'],80);self.assertEqual(d['observer_gate_product'],2);self.assertEqual(d['packed_W1_gemm'],80);self.assertEqual(d['packed_W2_gemm_fp32'],80);self.assertEqual(d['cache_h_clone'],160);self.assertEqual(d['cache_release_sync'],160)
 def test_actual_preflight_rejects_missing_active_before_torch(self):
  script=textwrap.dedent("""
   import json,pathlib,sys,types
   sys.path.insert(0,sys.argv[1]);import e0_chipmunk_wan_global_p64_owner_v2 as o
   base=pathlib.Path(sys.argv[2]);o.ROOT=base;(base/'research').mkdir();(base/'research/ACTIVE_RESEARCH.json').write_text(json.dumps({'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},'experiment_authority':{'current_user_authorized_scopes':[]}}))
   p=base/'p.json';p.write_text(json.dumps({'id':'missing'}));o.validate=lambda *a,**k:None
   try:o.preflight(types.SimpleNamespace(protocol=p))
   except ValueError as e:assert str(e)=='ACTIVE required'
   else:raise AssertionError('missing ACTIVE accepted')
   assert 'torch' not in sys.modules
  """)
  with tempfile.TemporaryDirectory() as d:
   r=subprocess.run([sys.executable,'-I','-B','-c',script,str(Path(__file__).resolve().parents[1]/'tools'),d],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_uninitialized_cuda_not_observed_or_initialized(self):
  fake=types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda:False))
  self.assertEqual(owner.final_memory(fake),{'status':'UNKNOWN_CUDA_NOT_INITIALIZED'})
 def test_run_failure_after_torch_import_never_initializes_cuda(self):
  script=textwrap.dedent("""
   import json,pathlib,sys,types
   sys.path.insert(0,sys.argv[1]);import e0_chipmunk_wan_global_p64_owner_v2 as o
   base=pathlib.Path(sys.argv[2]);o.HF=base
   p={'id':'fake','runtime_versions':{},'output_HF_relative':'scope/results','run_ids':['a','b','c','d'],'case':{'native_source_ordinal':898}}
   proto=base/'p.json';proto.write_text(json.dumps(p))
   o.authorize=lambda *a:{'execution_status':'RUNNING_FIXED_SCOPE','native_group_active_slots':1,'gpu_launch_spec_sha256':'synthetic'};o.verify_job=lambda *a:None;o.require_cpu=lambda *a:{}
   class J:
    def write(self,*a):pass
    def close(self):pass
   class C:
    def __init__(self,*a):self.files={};self.accounting_errors=[]
    def hold(self,*a):pass
    def path(self,p):return p
    def open_journal(self,p):p.write_text('');return J()
    def write_json(self,p,x):p.write_text(json.dumps(x))
    def summary(self):return {'unconsumed_reserved_bytes':0}
    def binding(self,p,root):return {'path':'dummy','bytes':0,'sha256':'x'}
    def check(self):pass
   class Monitor:
    def __init__(self,*a):pass
    def start(self):pass
    def stop(self):return {'fault':False,'thread_stopped':True}
   o.io.WrittenBytes=C;o.support.UsageMonitor=Monitor;o.support.reserve_emergency=lambda *a:None
   def fail(*args):
    class Cuda:
     def is_initialized(self):return False
     def __getattr__(self,key):raise AssertionError('CUDA accessed while uninitialized: '+key)
    sys.modules['torch']=types.SimpleNamespace(cuda=Cuda());raise RuntimeError('injected after torch import')
   o.charged_assets=fail
   try:o.run(types.SimpleNamespace(protocol=proto,job_spec=None,local_root=base))
   except RuntimeError as e:assert str(e)=='injected after torch import'
   else:raise AssertionError('failure missing')
   r=json.loads((base/'scope/results/RUN.json').read_text());assert r['status']=='FAILED_CHIPMUNK_P64_VIDEO_V2_PRESERVED_NO_RETRY' and r['final_memory']=={'status':'UNKNOWN_CUDA_NOT_INITIALIZED'}
  """)
  with tempfile.TemporaryDirectory() as d:
   r=subprocess.run([sys.executable,'-I','-B','-c',script,str(Path(__file__).resolve().parents[1]/'tools'),d],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_scope_identity_and_limits(self):
  with self.assertRaises(ValueError):owner.sid(2)
  self.assertEqual(owner.DESIGN['cache_per_expert_logical_bytes'],40*31668*(13824+5120)*2)
if __name__=='__main__':unittest.main()
