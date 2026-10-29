"""Small metadata fixtures exercising real producer and verifier entry guards."""
import hashlib,json,os,pathlib,sys,tempfile,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_input_video_quality_metric_adapter as adapter
import e0_chipmunk_input_video_quality_metric_verify as verify

def put(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return p
class ProducerEntry(unittest.TestCase):
 def test_quality_and_raft_all_four_parts_no_active(self):
  for kind in ('quality','raft'):
   for part in range(4):
    with self.subTest(kind=kind,part=part),tempfile.TemporaryDirectory() as t:
     e=adapter.install(kind);root=pathlib.Path(t);p=put(root/'p.json',{'id':f'CASE_{part}'});out=root/'out';args=types.SimpleNamespace(protocol=p,out=out,hf_root=root)
     real_read=e.base.read
     def read(path):
      if str(path).endswith('ACTIVE_RESEARCH.json'):return {'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},'experiment_authority':{'current_user_authorized_scopes':[]}}
      return real_read(path)
     with patch.object(e,'validate'),patch.object(e.base,'read',side_effect=read),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(os,'sched_getaffinity',return_value=({0,1} if kind=='quality' else {0})):
      with self.assertRaisesRegex(ValueError,'ACTIVE exact'):e.preflight(args)
     self.assertFalse(out.exists())
class VerifyEntry(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name);self.hf=self.root/'hf';self.hf.mkdir();self.profile=self.root/'profile';self.profile.write_text('profile');self.q=put(self.root/'protocol.json',{'id':'CASE','execution_private_worker_config_sha256':hashlib.sha256(self.profile.read_bytes()).hexdigest(),'output_HF_relative':'case/worker/results','import_environment':{'python_HF_relative':'python'}});self.d=hashlib.sha256(self.q.read_bytes()).hexdigest();self.job=self.root/'job.json';self.cpu=next(iter(os.sched_getaffinity(0)));self.run=self.hf/'case/worker/results/RUN.json';self.ex=self.run.parent.parent/'EXIT_RECEIPT.json';self.dest=self.hf/'case/verification/worker/results/RUN.json';self.args=types.SimpleNamespace(kind='quality',protocol=self.q,job_spec=self.job,hf_root=self.hf);self.e=adapter.install('quality');self.base=self.e.base;self.argv=['env','FFNDS_WORKER_CONFIG='+str(self.profile),'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PATH=/usr/bin:/bin','CUDA_VISIBLE_DEVICES=','taskset','-c',str(self.cpu),str(self.hf/'python'),'-I','-S','-B',str(self.root/'tools/e0_chipmunk_input_video_quality_metric_verify.py'),'verify','--kind','quality','--protocol',str(self.q),'--hf-root',str(self.hf),'--job-spec',str(self.job)];put(self.job,{'argv':self.argv,'id':'CASE_VERIFY','protocol_sha256':self.d,'protocol_path':str(self.q),'cwd':str(self.root),'hf_root':str(self.hf),'output_dir':str(self.dest.parent.parent),'run_json':str(self.dest),'external_timeout_seconds':verify.RES['outer_seconds']});self.jsha=hashlib.sha256(self.job.read_bytes()).hexdigest();self.review=put(self.root/'review.json',{'review_schema':'FFNDS_PREEXEC_REVIEW_V1','scope_id':'CASE_VERIFY','state':'REVIEWER_RECHECK_APPROVED','recheck_disposition':'APPROVE','author_agent_id':'a','reviewer_agent_id':'b','approval_scope_fingerprint':'synthetic','verification_spec_sha256':self.jsha,'author_review':{},'independent_review':{},'reviewer_recheck':{}});self.active=self.root/'research/ACTIVE_RESEARCH.json';self.scope={'id':'CASE_VERIFY','execution_status':'RUNNING_FIXED_SCOPE','protocol_sha256':self.d,'parent_scope_id':'CASE','resources':verify.RES,'max_gpu_cards':0,'review':{},'verification_spec_sha256':self.jsha};self.active_set([self.scope]);worker=types.SimpleNamespace(require_worker=lambda:None);self.patches=[patch.object(verify,'ROOT',self.root),patch.object(verify,'HF',self.hf),patch.object(verify,'load',return_value=self.e),patch.object(verify,'fingerprint',return_value='synthetic'),patch.object(self.base,'read_bound',return_value=json.loads(self.review.read_text())),patch.object(self.base,'bound',return_value=self.review),patch.object(verify.sys,'flags',types.SimpleNamespace(isolated=1)),patch.object(os,'sched_getaffinity',return_value={self.cpu}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.dict(sys.modules,{'worker_identity':worker})]
  for x in self.patches:x.start();self.addCleanup(x.stop)
 def active_set(self,scopes):put(self.active,{'experiment_authority':{'current_user_authorized_scopes':scopes}})
 def make_parent(self,**kw):
  run=put(self.run,{'id':'CASE','job_spec_sha256':'gpu'});b=hashlib.sha256(run.read_bytes()).hexdigest();e={'id':'CASE','protocol_sha256':self.d,'run_sha256':b,'process_exited':True,'child_returncode':0,'timeout':False,'automatic_retry':False,'spec_sha256':'gpu'};e.update(kw);put(self.ex,e)
 def test_no_active_rejected_without_receipt(self):
  self.active_set([])
  with self.assertRaisesRegex(ValueError,'verifier ACTIVE required'):verify.verify(self.args)
  self.assertFalse(self.dest.exists())
 def test_parent_exit_nonzero(self):
  self.make_parent(child_returncode=1)
  with self.assertRaises(Exception):verify.verify(self.args)
  self.assertFalse(self.dest.exists())
 def test_parent_run_sha_mismatch(self):
  self.make_parent(run_sha256='wrong')
  with self.assertRaises(Exception):verify.verify(self.args)
  self.assertFalse(self.dest.exists())
 def test_parent_spec_mismatch(self):
  self.make_parent(spec_sha256='wrong')
  with self.assertRaisesRegex(ValueError,'parent actual EXIT differs'):verify.verify(self.args)
  self.assertFalse(self.dest.exists())
class VisualEntry(unittest.TestCase):
 def test_all_parts_no_active_before_output(self):
  import e0_chipmunk_input_video_quality_visual_cpu as v
  for part in range(4):
   with self.subTest(part=part),tempfile.TemporaryDirectory() as t:
    root=pathlib.Path(t);q=put(root/'protocol.json',{'id':f'VISUAL_{part}'});out=root/'out';args=types.SimpleNamespace(protocol=q,hf_root=root)
    real_read=v.base.read
    def read(path):
     if str(path).endswith('ACTIVE_RESEARCH.json'):return {'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},'experiment_authority':{'current_user_authorized_scopes':[]}}
     return real_read(path)
    with patch.object(v,'validate'),patch.object(v.base,'read',side_effect=read):
     with self.assertRaisesRegex(ValueError,'Exact ACTIVE scope required'):v._run(args)
    self.assertFalse(out.exists())
class NativeParentEntry(unittest.TestCase):
 def test_actual_native_parent_exit_failures_before_rows(self):
  import e0_chipmunk_input_video_quality_quality_lineage as line
  import e0_chipmunk_input_video_owner as owner
  for variant in ('exit','sha'):
   with self.subTest(variant=variant),tempfile.TemporaryDirectory() as t:
    root=pathlib.Path(t);hf=root/'hf';hf.mkdir();q=put(root/line.CONFIGS[0],{'id':'NATIVE','output_HF_relative':'native/worker/results'});run=put(hf/'native/worker/results/RUN.json',{'id':'NATIVE'});exitp=run.parent.parent/'EXIT_RECEIPT.json';put(exitp,{'id':'NATIVE','protocol_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'run_sha256':('bad' if variant=='sha' else hashlib.sha256(run.read_bytes()).hexdigest()),'process_exited':True,'child_returncode':(1 if variant=='exit' else 0),'timeout':False,'automatic_retry':False});fake=types.SimpleNamespace(validate=lambda *a:None,inspect_verification=lambda *a:{'status':'PASS_ACTUAL_EXIT_CLOSED_CHIPMUNK_INPUT_VIDEO','verification_RUN':{},'verification_exit':{}},vm=owner.vm);spec=types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda m:None));origpath=pathlib.Path
    def path(value):return hf if str(value)=='/home/xiang/.cache/huggingface' else origpath(value)
    with patch.object(line,'ROOT',root),patch.object(line,'Path',side_effect=path),patch.object(line.importlib.util,'spec_from_file_location',return_value=spec),patch.object(line.importlib.util,'module_from_spec',return_value=fake):
     with self.assertRaisesRegex(ValueError,'exit0'):line.parent_data(hf,0)
