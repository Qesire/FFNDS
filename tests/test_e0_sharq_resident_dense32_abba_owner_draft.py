import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import sharq_resident_owner_accounting_draft as d
class OwnerDraftTest(unittest.TestCase):
 def test_debit_failure_preserves_inflight(self):
  seen=[];x=d.DurableDebit(10,seen.append,'compare')
  def bad():raise RuntimeError('failure')
  with self.assertRaises(RuntimeError):x.call(5,bad,label='one',completion='SYNCHRONIZED')
  self.assertEqual((x.reserved,x.completed),(5,0));self.assertEqual(seen[-1]['inflight']['completion'],'UNKNOWN')
  with self.assertRaises(ValueError):x.call(1,lambda:None,label='two',completion='SYNCHRONIZED')
 def test_budget_and_complete(self):
  seen=[];x=d.DurableDebit(5,seen.append,'x');self.assertEqual(x.call(5,lambda:7,label='a',completion='CPU_RETURN_GPU_UNKNOWN'),7);self.assertEqual(x.completed,5)
  with self.assertRaises(ValueError):x.call(1,lambda:None,label='b',completion='SYNCHRONIZED')
 def test_adapters_charge_and_preserve_failure(self):
  seen=[];a=d.OwnerAccounting(seen.append);self.assertEqual(a.charge_block(0,2,lambda:4),4);self.assertEqual(a.block_loads.completed,1)
  with self.assertRaises(ZeroDivisionError):a.synchronize_checkpoint(lambda:1/0)
  self.assertEqual(a.checkpoints.completed,0);self.assertEqual(a.checkpoints.inflight['completion'],'UNKNOWN')
 def test_phase_and_fail_closed(self):
  r={'phases':[]};events=[];d.record_progress(r,'phase_started',{'index':0,'arm':'warmup','mode':'bf16_cache'},lambda:events.append(True));d.record_progress(r,'phase_evidence_complete',{'run_id':'a','arm':'warmup','mode':'bf16_cache'},lambda:events.append(True));self.assertEqual(len(r['phases']),1)
  with self.assertRaises(RuntimeError):d.fail_closed()
if __name__=='__main__':unittest.main()

class OwnerContractTest(unittest.TestCase):
 def test_real_cpu_entry_rejects_unregistered_before_model_import(self):
  import json,os,subprocess,tempfile
  root=Path(__file__).resolve().parents[1];draft=root/'.runtime/sharq_resident_dense32_abba_20261030/draft_part0.json'
  if not draft.is_file():self.skipTest('actual metadata draft unavailable')
  data=json.loads(draft.read_text());data.update(status='FROZEN_PENDING_REGISTRATION',assets_finalized=True)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'protocol.json';p.write_text(json.dumps(data));out=Path(t)/'out';env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
   r=subprocess.run([str(root/'.venv/bin/python'),'-I','-B',str(root/'tools/e0_sharq_resident_dense32_abba_owner_draft.py'),'preflight','--protocol',str(p),'--hf-root','/home/xiang/.cache/huggingface','--local-root',t,'--job-spec',str(Path(t)/'missing.json'),'--out',str(out)],cwd=root,env=env,text=True,capture_output=True,timeout=30)
   self.assertNotEqual(r.returncode,0);self.assertIn('ACTIVE required',r.stderr);self.assertFalse(out.exists())

class VerifyContractTest(unittest.TestCase):
 def test_verifier_entry_rejects_without_separate_active(self):
  import json,os,subprocess,tempfile
  root=Path(__file__).resolve().parents[1];draft=root/'.runtime/sharq_resident_dense32_abba_20261030/draft_part0.json'
  if not draft.is_file():self.skipTest('actual metadata draft unavailable')
  p=json.loads(draft.read_text());p.update(status='FROZEN_PENDING_REGISTRATION',assets_finalized=True)
  with tempfile.TemporaryDirectory() as t:
   protocol=Path(t)/'p.json';protocol.write_text(json.dumps(p));env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
   r=subprocess.run([str(root/'.venv/bin/python'),'-I','-B',str(root/'tools/e0_sharq_resident_dense32_abba_owner_draft.py'),'verify-closed','--protocol',str(protocol),'--hf-root','/home/xiang/.cache/huggingface','--job-spec',str(Path(t)/'missing.json'),'--run',str(Path(t)/'missing-run'),'--exit',str(Path(t)/'missing-exit'),'--verify-out',str(Path(t)/'verify')],cwd=root,env=env,text=True,capture_output=True,timeout=30)
   self.assertNotEqual(r.returncode,0);self.assertIn('independent verifier ACTIVE required',r.stderr);self.assertFalse((Path(t)/'verify').exists())

class ReferenceGateTest(unittest.TestCase):
 def data(self):
  prior={k:{name:{'bytes':1,'sha256':k} for name in (*d.STATES,'video')} for k in ('bf16','dense32')};phases=[];checks=[]
  for arm in d.ARMS:
   key='bf16' if arm=='warmup' else 'dense32';states={k:dict(prior[key][k]) for k in d.STATES};phases.append({'arm':arm,'states':states,'video':dict(prior[key]['video'])})
   checks.extend({'arm':arm,'kind':name,'byte_equal':True,'bytes_each':1} for name in (*d.STATES,'video'))
  return phases,{'checks':checks},prior
 def test_unique_and_phase_reference(self):
  phases,gate,prior=self.data();self.assertTrue(d.validate_reference_gate(phases,gate,prior));gate['checks'][-1]=gate['checks'][0]
  with self.assertRaises(ValueError):d.validate_reference_gate(phases,gate,prior)
  phases,gate,prior=self.data();phases[1]['video']['sha256']='wrong'
  with self.assertRaises(ValueError):d.validate_reference_gate(phases,gate,prior)
 def test_file_hash_failure_unknown(self):
  import tempfile
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x';p.write_bytes(b'a');a=d.OwnerAccounting(lambda _:None)
   with self.assertRaises(RuntimeError):a.hash_state_file(p,lambda:(_ for _ in ()).throw(RuntimeError('bad')))
   self.assertEqual(a.state_file_sha.completed,0);self.assertEqual(a.state_file_sha.inflight['completion'],'UNKNOWN')

class PayloadBoundTest(unittest.TestCase):
 def test_fp32_state_above_average_is_allowed(self):
  from types import SimpleNamespace
  class P:
   def stat(self):return SimpleNamespace(st_size=2026752*4+88)
  a=d.OwnerAccounting(lambda _:None);self.assertEqual(a.hash_state_file(P(),lambda:1),1)
  self.assertEqual(a.state_file_sha.completed,2026752*4+88)
  class TooLarge:
   def stat(self):return SimpleNamespace(st_size=2026752*4+(1<<20)+9)
  with self.assertRaises(ValueError):a.hash_state_file(TooLarge(),lambda:None)
