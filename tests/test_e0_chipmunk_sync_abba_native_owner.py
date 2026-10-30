"""Metadata-only entry tests; no Torch, tensor reads or model execution."""
import copy,hashlib,json,struct,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_chipmunk_sync_abba_native_owner as o
import chipmunk_p64_sync_abba_contract as c

class OwnerTest(unittest.TestCase):
 def test_limits_and_seven_phase_reference(self):
  for part in (0,1):
   lim=o.limits(part);counts=c.expected_counts(part)
   for k,v in [('video_generations',7),('state_output_writes',70),('prefetch_transfer_observations',1134),('deferred_completion_barriers',120),('packed_action',240),('extra_old_W1',240)]:self.assertEqual(lim[k],v)
   self.assertEqual(counts['verification_artifacts'],79)
   ids=o.run_ids(part);names=o.driver.STATE_NAMES;state={k:{'sha256':'1'*64,'bytes':1,'raw_tensor_sha256':'2'*64,'shape':[1],'dtype':'float32'} for k in names};video={'sha256':'3'*64,'bytes':2}
   p={'part':part,'run_ids':ids,'source_parent':{'dense':{'states':state,'video':video},'X':{'states':state,'video':video}}}
   phases=[{'run_id':rid,'mode':'none' if i==0 else 'high_x_recompute_p64','sync_arm':label,'states':copy.deepcopy(state),'video':dict(video)} for i,(rid,label) in enumerate(zip(ids,c.schedule(part)))]
   controls=[{'mode':label,'state':name,'equal':True} for label in c.schedule(part) for name in names]
   o.verify_reference_metadata(p,phases,controls)
   phases[-1]['states']['initial_noise']['sha256']='bad'
   with self.assertRaises(ValueError):o.verify_reference_metadata(p,phases,controls)
 def test_real_authorize_rejects_missing_active(self):
  with patch.object(o,'validate'),patch.object(o.native,'read',return_value={'user_continuation_authorization_20261027':{'new_scope_registration_allowed':True},'experiment_authority':{'current_user_authorized_scopes':[]}}):
   with self.assertRaisesRegex(ValueError,'ACTIVE'):o.authorize({'id':'case'},'digest')
 def test_real_verify_rejects_exit_runsha_and_spec(self):
  for kind in ('bad_exit','bad_sha','bad_spec'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
    hf=Path(d);out=hf/'case/worker/results';out.mkdir(parents=True);run=out/'RUN.json';ex=out.parent/'EXIT_RECEIPT.json';protocol=hf/'p.json';dest=hf/'verify/RUN.json';p={'id':'E0_TEST','part':0,'output_HF_relative':'case/worker/results'};protocol.write_text(json.dumps(p));digest=hashlib.sha256(protocol.read_bytes()).hexdigest();r={'id':p['id'],'protocol_sha256':digest,'job_spec_sha256':'good'};run.write_text(json.dumps(r));sha=hashlib.sha256(run.read_bytes()).hexdigest();e={'id':p['id'],'protocol_sha256':digest,'process_exited':True,'child_returncode':0,'timeout':False,'automatic_retry':False,'run_sha256':sha,'spec_sha256':'good'}
    if kind=='bad_exit':e['child_returncode']=1
    if kind=='bad_sha':e['run_sha256']='0'*64
    if kind=='bad_spec':e['spec_sha256']='bad'
    ex.write_text(json.dumps(e));a=SimpleNamespace(protocol=protocol,run=run,exit=ex,verify_out=dest)
    with patch.object(o,'HF',hf),patch.object(o,'validate'),patch.object(o,'authorize_verification',return_value={'verification_spec_sha256':'x'}):
     with self.assertRaises((ValueError,KeyError)):o.verify_closed(a)
    self.assertFalse(dest.exists())
 def test_tiny_typed_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'tiny.safetensors';raw=b'\0\0';header=json.dumps({'latents':{'dtype':'BF16','shape':[1],'data_offsets':[0,2]}},separators=(',',':')).encode();data=struct.pack('<Q',len(header))+header+raw;p.write_bytes(data);o.inspect_payload(p,'latents',[1],'BF16',hashlib.sha256(data).hexdigest(),hashlib.sha256(raw).hexdigest())
   with self.assertRaises(ValueError):o.inspect_payload(p,'wrong',[1],'BF16',hashlib.sha256(data).hexdigest(),hashlib.sha256(raw).hexdigest())
if __name__=='__main__':unittest.main()
