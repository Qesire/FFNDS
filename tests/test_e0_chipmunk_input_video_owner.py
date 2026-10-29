import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_input_video_owner as o
class Test(unittest.TestCase):
 def test_reference_contract_and_missing_control(self):
  names=['raw_noise_FP32','initial_noise']+[f'{kind}_step{i}' for kind in ('flow','latent') for i in range(4)];b={'sha256':'a','bytes':1,'raw_tensor_sha256':'b','shape':[1],'dtype':'float32'};v={'sha256':'c','bytes':1};phases=[{'mode':mode,'states':{n:copy.deepcopy(b) for n in names},'video':copy.deepcopy(v)} for mode in o.driver.MODES];prefix={'raw_noise_FP32','initial_noise','flow_step0','latent_step0'};controls=[{'mode':mode,'state':n,'equal':True} for mode in o.driver.MODES[1:] for n in (prefix if mode.startswith('high_') else names)];o.verify_reference_metadata(phases,controls)
  with self.assertRaises(ValueError):o.verify_reference_metadata(phases,controls[:-1])
  old=copy.deepcopy(phases[3]);controls.extend({'mode':'old_high','state':n,'equal':True} for n in names);o.verify_reference_metadata(phases,controls,old);old['video']['sha256']='wrong'
  with self.assertRaises(ValueError):o.verify_reference_metadata(phases,controls,old)
 def test_limits_old_vs_new(self):
  a=o.limits(0);b=o.limits(2);self.assertEqual((a['reference_state_checks'],b['reference_state_checks']),(38,28));self.assertEqual((a['reference_video_checks'],b['reference_video_checks']),(3,2));self.assertEqual(a['extra_old_W1'],40)
if __name__=='__main__':unittest.main()
class Entry(unittest.TestCase):
 def test_preflight_rejects_no_active_before_output(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  p={'id':'SYNTHETIC'};args=SimpleNamespace(protocol=Path('/unused'))
  def read(path):
   if str(path).endswith('ACTIVE_RESEARCH.json'):return {'experiment_authority':{'current_user_authorized_scopes':[]},'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True}}
   return p
  with patch.object(o,'validate'),patch.object(o.native,'read',side_effect=read),patch.object(o.native,'sha',return_value='synthetic'):
   with self.assertRaisesRegex(ValueError,'ACTIVE required'):o.preflight(args)
class VerifyClosedEntry(unittest.TestCase):
 def setUp(self):
  import tempfile,json,os,types
  from unittest.mock import patch
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.hf=self.root/'hf';self.hf.mkdir();self.profile=self.root/'profile';self.profile.write_text('fake');self.cpu=next(iter(os.sched_getaffinity(0)));self.protocol=self.root/'p.json';self.spec=self.root/'job.json';self.out=self.hf/'output/worker/results';self.run=self.out/'RUN.json';self.exit=self.out.parent/'EXIT_RECEIPT.json';self.dest=self.hf/'output/verification/worker/results/RUN.json';self.p={'id':'SYNTH','part':0,'output_HF_relative':'output/worker/results','cpu_affinity':[self.cpu],'execution_private_worker_config_sha256':o.native.sha(self.profile)};self.protocol.write_text(json.dumps(self.p));self.digest=o.native.sha(self.protocol);self.py=self.root/'python';self.argv=['env','FFNDS_WORKER_CONFIG='+str(self.profile),'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PATH=/usr/bin:/bin','CUDA_VISIBLE_DEVICES=','taskset','-c',str(self.cpu),str(self.py),'-I','-B',str(self.root/o.SELF),'verify-closed','--protocol',str(self.protocol),'--hf-root',str(self.hf),'--job-spec',str(self.spec),'--run',str(self.run),'--exit',str(self.exit),'--verify-out',str(self.dest)];self.job={'argv':self.argv,'id':'SYNTH_VERIFY','protocol_sha256':self.digest,'protocol_path':str(self.protocol),'cwd':str(self.root),'hf_root':str(self.hf),'output_dir':str(self.dest.parent.parent),'run_json':str(self.dest),'external_timeout_seconds':o.VERIFY['outer_seconds']};self.spec.write_text(json.dumps(self.job));self.review=self.root/'review.json';self.review.write_text(json.dumps({'scope_id':'SYNTH_VERIFY','review_schema':'FFNDS_PREEXEC_REVIEW_V1','state':'REVIEWER_RECHECK_APPROVED','recheck_disposition':'APPROVE','author_agent_id':'author','reviewer_agent_id':'reviewer','approval_scope_fingerprint':o.fingerprint(self.p),'verification_spec_sha256':o.native.sha(self.spec),'author_review':{},'independent_review':{},'reviewer_recheck':{}}));self.scope={'id':'SYNTH_VERIFY','execution_status':'RUNNING_FIXED_SCOPE','protocol_sha256':self.digest,'resources':o.VERIFY,'max_gpu_cards':0,'parent_scope_id':'SYNTH','verification_spec_sha256':o.native.sha(self.spec),'review':{}};self.active=self.root/'research/ACTIVE_RESEARCH.json';self.active.parent.mkdir();self.set_active([self.scope]);self.args=types.SimpleNamespace(protocol=self.protocol,job_spec=self.spec,run=self.run,exit=self.exit,verify_out=self.dest);worker=types.SimpleNamespace(require_worker=lambda:None);real_bound=o.native.bound
  def bound(root,ref):return self.review if root==self.root else real_bound(root,ref)
  self.patches=[patch.object(o,'ROOT',self.root),patch.object(o,'HF',self.hf),patch.object(o,'PYTHON',self.py),patch.object(o,'validate'),patch.object(o.native,'bound',side_effect=bound),patch.object(o.sys,'flags',types.SimpleNamespace(isolated=1)),patch.object(o.os,'sched_getaffinity',return_value={self.cpu}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.dict(sys.modules,{'worker_identity':worker})]
  for x in self.patches:x.start();self.addCleanup(x.stop)
 def set_active(self,scopes):
  import json
  self.active.write_text(json.dumps({'experiment_authority':{'current_user_authorized_scopes':scopes}}))
 def producer(self,override=None):
  import json
  self.out.mkdir(parents=True,exist_ok=True);self.run.write_text(json.dumps({'id':'SYNTH','protocol_sha256':self.digest}));e={'id':'SYNTH','protocol_sha256':self.digest,'run_sha256':o.native.sha(self.run),'process_exited':True,'child_returncode':0,'timeout':False,'automatic_retry':False};e.update(override or {});self.exit.write_text(json.dumps(e))
 def test_no_verify_scope_rejected_before_receipt(self):
  self.set_active([])
  with self.assertRaisesRegex(ValueError,'independent verifier ACTIVE required'):o.verify_closed(self.args)
  self.assertFalse(self.dest.exists())
 def test_nonzero_gpu_exit_rejected_before_receipt(self):
  self.producer({'child_returncode':1})
  with self.assertRaisesRegex(ValueError,'exit0'):o.verify_closed(self.args)
  self.assertFalse(self.dest.exists())
 def test_wrong_gpu_run_sha_rejected_before_receipt(self):
  self.producer({'run_sha256':'wrong'})
  with self.assertRaisesRegex(ValueError,'exit0'):o.verify_closed(self.args)
  self.assertFalse(self.dest.exists())
 def test_wrong_gpu_spec_rejected_before_receipt(self):
  import json
  self.producer();r=json.loads(self.run.read_text());r['job_spec_sha256']='expected';self.run.write_text(json.dumps(r));e=json.loads(self.exit.read_text());e.update(run_sha256=o.native.sha(self.run),spec_sha256='wrong');self.exit.write_text(json.dumps(e))
  with self.assertRaisesRegex(ValueError,'parent completed identity'):o.verify_closed(self.args)
  self.assertFalse(self.dest.exists())
