"""Exercise real verify(args) against temporary metadata only, without CUDA."""
import hashlib,json,os,pathlib,tempfile,types,unittest
from unittest.mock import patch
from tools import e0_chipmunk_input_state_verify as v

def put(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return p
def binding(p,base):
 b=p.read_bytes();return {'path':str(p.relative_to(base)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
class VerifyEntryTest(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=pathlib.Path(self.temp.name);self.hf=self.root/'hf';self.hf.mkdir();self.profile=self.root/'profile';self.profile.write_text('private test');self.protocol=self.root/'protocol.json';self.job=self.root/'job.json';self.cpu=next(iter(os.sched_getaffinity(0)));self.res={'wall_seconds':300,'outer_seconds':360};self.limit={'typed_results':1};self.p={'id':'TEST','assets_finalized':True,'worker_config_sha256':v.sha(self.profile),'verification_resources':self.res,'verification_limits':self.limit,'output_HF_relative':'data/worker/results','repo_sources':[]};put(self.protocol,self.p);self.pdigest=v.sha(self.protocol);self.argv=[str(self.root/'verify.py'),'--protocol',str(self.protocol),'--job-spec',str(self.job),'--hf-root',str(self.hf)];self.dest=self.hf/'verify/worker/results/RUN.json';self.j={'id':'TEST_VERIFY','protocol_sha256':self.pdigest,'driver_argv':self.argv,'cwd':str(self.root),'cpu_affinity':[self.cpu],'protocol_path':str(self.protocol),'hf_root':str(self.hf),'external_timeout_seconds':360,'run_json':str(self.dest),'output_dir':str(self.dest.parent.parent)};self.j['argv']=['env','FFNDS_WORKER_CONFIG='+str(self.profile),'CUDA_VISIBLE_DEVICES=','taskset','-c',str(self.cpu),str(self.root/'.venv/bin/python'),'-I','-B']+self.argv;put(self.job,self.j);self.jdigest=v.sha(self.job);self.args=types.SimpleNamespace(protocol=self.protocol,job_spec=self.job,hf_root=self.hf)
  self.scope={'id':'TEST_VERIFY','execution_status':'RUNNING_FIXED_SCOPE','protocol_sha256':self.pdigest,'launch_spec_sha256':self.jdigest,'resources':self.res,'limits':self.limit};review=put(self.root/'review.json',{'state':'REVIEWER_RECHECK_APPROVED','scope_id':'TEST','protocol_sha256':self.pdigest,'verification_job_spec_sha256':self.jdigest,'author_agent_id':'author','reviewer_agent_id':'reviewer','critical_source_hashes':{},'cpu_job_spec_sha256':'cpu_spec','job_spec_sha256':'gpu_spec'});self.scope['review']=binding(review,self.root)
  self.active=self.root/'research/ACTIVE_RESEARCH.json';self.set_active([self.scope]);self.run=self.hf/'data/worker/results/RUN.json';self.exit=self.run.parent.parent/'EXIT_RECEIPT.json';self.worker=types.SimpleNamespace(require_worker=lambda:None,private_config_path=lambda:self.profile)
  self.patches=[patch.object(v,'ROOT',self.root),patch.object(v.sys,'flags',types.SimpleNamespace(isolated=1)),patch.object(v.sys,'argv',self.argv),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(v.os,'sched_getaffinity',return_value={self.cpu}),patch.dict(v.sys.modules,{'worker_identity':self.worker})]
  for x in self.patches:x.start();self.addCleanup(x.stop)
 def set_active(self,scopes):put(self.active,{'experiment_authority':{'current_user_authorized_scopes':scopes}})
 def make_producer(self,exit_override=None,cpu_bad=False):
  cr=put(self.hf/'cpu/RUN.json',{'status':'PASS_ISOLATED_ENTRY_CPU_PREFLIGHT','protocol_sha256':self.pdigest,'job_spec_sha256':'cpu_spec'});ce=put(self.hf/'cpu/EXIT.json',{'child_returncode':1 if cpu_bad else 0,'process_exited':True,'run_sha256':v.sha(cr),'spec_sha256':'cpu_spec'});run=put(self.run,{'id':'TEST','protocol_sha256':self.pdigest,'status':'COMPLETE_DIAGNOSTIC','CPU_preflight':{'RUN':binding(cr,self.hf),'EXIT':binding(ce,self.hf)}});ex={'id':'TEST','child_returncode':0,'process_exited':True,'timeout':False,'run_sha256':v.sha(run),'protocol_sha256':self.pdigest};ex.update(exit_override or {});put(self.exit,ex)
 def test_missing_active_rejected_before_run(self):
  self.set_active([])
  with self.assertRaisesRegex(ValueError,'scope not active'):v.verify(self.args)
  self.assertFalse(self.dest.exists())
 def test_wrong_verifier_job_spec_rejected_before_run(self):
  self.scope['launch_spec_sha256']='wrong';self.set_active([self.scope])
  with self.assertRaisesRegex(ValueError,'verify identity'):v.verify(self.args)
  self.assertFalse(self.dest.exists())
 def test_failed_producer_exit_is_preserved(self):
  self.make_producer({'child_returncode':1})
  with self.assertRaisesRegex(ValueError,'actual GPU exit'):v.verify(self.args)
  self.assertEqual(json.loads(self.dest.read_text())['status'],'FAILED_NO_RETRY')
 def test_wrong_cpu_exit_is_preserved(self):
  self.make_producer(cpu_bad=True)
  with self.assertRaisesRegex(ValueError,'CPU actual EXIT proof'):v.verify(self.args)
  self.assertEqual(json.loads(self.dest.read_text())['status'],'FAILED_NO_RETRY')
 def test_wrong_producer_run_sha_rejected(self):
  self.make_producer({'run_sha256':'bad'})
  with self.assertRaisesRegex(ValueError,'actual GPU exit'):v.verify(self.args)
  self.assertEqual(json.loads(self.dest.read_text())['status'],'FAILED_NO_RETRY')
if __name__=='__main__':unittest.main()
