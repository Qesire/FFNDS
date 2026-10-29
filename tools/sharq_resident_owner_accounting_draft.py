"""Resident ABBA owner components under construction; NOT an experiment entry.

This module intentionally does not expose run, draft or ACTIVE registration.
Use its bounded durable accounting and reference adapters only within a future
independently reviewed owner. No GPU, model, or checkpoint I/O on import.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from sharq_resident_dense32_driver_draft import ARMS
from sharq_resident_reference_gate_draft import STATES
from sharq_resident_copy_ledger_draft import CopyLedger
from sharq_resident_source_plan_draft import source_paths

CPU_ASSET_HASH_MAX=512<<20
GPU_ASSET_HASH_MAX=512<<20
OUTPUT_MAX=512<<20
COMPARE_MAX=1<<30
VERIFY_PAYLOAD_MAX=512<<20
BLOCK_LOADS=5*162
LOGICAL_COPY_CALLS=5*324
BOUNDARY_CONTROLS=4*2
CONTROL_ELEMENTS=31668*13824
CONTROL_COMPONENTS={'dense32_prepare':1,'dense32_gemm':1,'output_scale':1,'bias_add':1,'author_bridge':1,'event_records':2,'event_waits':2,'finite_reductions_max':2,'bitwise_comparisons':1,'sync':1,'logical_elements_per_output':CONTROL_ELEMENTS,'scalar_host_reads_max':3}
PHASES=5
STATE_WRITES=50
VIDEO_WRITES=5
REFERENCE_COMPARISONS=55

def need(value,message):
 if not value:raise ValueError(message)

class DurableDebit:
 """Persist predebit and UNKNOWN before work; complete only after callback return.

Asynchronous CUDA operations must synchronize inside callback before completion;
CPU enqueues must explicitly declare that GPU completion remains UNKNOWN.
"""
 def __init__(self,limit,persist,label):
  need(type(limit) is int and limit>=0 and callable(persist),'budget')
  self.limit=limit;self.persist=persist;self.label=label;self.reserved=0;self.completed=0;self.inflight=None;self.returned_GPU_completion_unknown=0
 def snapshot(self):return {'label':self.label,'limit':self.limit,'predebited':self.reserved,'completed':self.completed,'inflight':self.inflight,'returned_GPU_completion_unknown':self.returned_GPU_completion_unknown}
 def call(self,amount,callback,*,label,completion):
  need(type(amount) is int and amount>0 and self.inflight is None and self.reserved+amount<=self.limit,'budget exceeded or operation already inflight')
  need(completion in ('SYNCHRONIZED','CPU_RETURN_GPU_UNKNOWN'),'unknown completion contract')
  self.reserved+=amount;self.inflight={'label':label,'amount':amount,'completion':'UNKNOWN','declared_completion':completion};self.persist(self.snapshot())
  result=callback()
  self.completed+=amount
  if completion=='CPU_RETURN_GPU_UNKNOWN':self.returned_GPU_completion_unknown+=amount
  self.inflight=None;self.persist(self.snapshot());return result

def source_closure(parent,extra=()):
 return source_paths(parent,extra=set(extra)|{'tools/e0_sharq_resident_dense32_abba_owner_draft.py','tools/sharq_resident_owner_accounting_draft.py','tests/test_e0_sharq_resident_dense32_abba_owner_draft.py'})

def previous_reference(prior_run):
 """Use old already verified phase metadata; byte gates still perform full reads."""
 phases=prior_run['phases'];need([x['mode'] for x in phases]==['bf16','bf16_cache','dense32','sharq'],'parent phase order')
 output={}
 for key,index in (('bf16',0),('dense32',2)):
  phase=phases[index];need(set(phase['states'])==STATES,'state coverage');output[key]={k:{z:phase['states'][k][z] for z in ('path','bytes','sha256')} for k in STATES}
  video=phase['video'];p=Path(video['path']).resolve();hf=Path('/home/xiang/.cache/huggingface').resolve();need(p.is_relative_to(hf),'old video outside HF');output[key]['video']={'path':str(p.relative_to(hf)),'bytes':video['bytes'],'sha256':video['sha256']}
 return output

def record_progress(report,event,data,persist):
 need(event in ('phase_started','phase_evidence_complete'),'phase event')
 if event=='phase_started':
  i=data['index'];need(i==len(report['phases']) and i<len(ARMS) and data['arm']==ARMS[i],'phase order');report['current_phase']={**data,'status':'STARTED'};report['current_state']=None
 else:
  i=len(report['phases']);need(i<len(ARMS) and data['arm']==ARMS[i] and report['current_phase']['status']=='STARTED' and data['run_id'],'phase completion differs');report['phases'].append(data);report['current_phase']={'run_id':data['run_id'],'arm':data['arm'],'mode':data['mode'],'status':'EVIDENCE_COMPLETE'};report['current_state']=None
 persist()

def fail_closed():
 raise RuntimeError('NOT_EXECUTABLE: exact ACTIVE/job/worker binding, numerical runtime, full verifier and independent review remain required')

class OwnerAccounting:
 """Bounded adapters for core callbacks; persists every transition.

The caller still owns measured device sync and serial process lifecycle. The
control wrapper performs a synchronous reference callback; the load wrapper
records only CPU return and never infers CUDA completion.
"""
 def __init__(self,persist):
  self.persist=persist;self.state_file_sha=DurableDebit(304012800+50*((1<<20)+8),lambda _:self.publish(),'state_file_sha_bytes');self.video_file_sha=DurableDebit(5*(32<<20),lambda _:self.publish(),'video_file_sha_bytes');self.state_hash_calls=0;self.video_hash_calls=0;self.compares=DurableDebit(COMPARE_MAX,lambda _:self.publish(),'reference_compare_bytes')
  self.block_loads=DurableDebit(BLOCK_LOADS,lambda _:self.publish(),'block_load_calls')
  self.phase_sync=DurableDebit(5,lambda _:self.publish(),'phase_device_synchronizations')
  self.checkpoints=DurableDebit(9,lambda _:self.publish(),'cache_checkpoint_synchronizations')
  self.controls=DurableDebit(BOUNDARY_CONTROLS,lambda _:self.publish(),'same_input_controls')
  self.selector=CopyLedger(LOGICAL_COPY_CALLS,lambda _:self.publish())
 def snapshot(self):return {'state_file_sha':self.state_file_sha.snapshot(),'state_hash_calls':self.state_hash_calls,'video_file_sha':self.video_file_sha.snapshot(),'video_hash_calls':self.video_hash_calls,'reference_compare':self.compares.snapshot(),'block_loads':self.block_loads.snapshot(),'phase_synchronizations':self.phase_sync.snapshot(),'cache_checkpoints':self.checkpoints.snapshot(),'boundary_controls':self.controls.snapshot(),'selector_copies':self.selector.snapshot(),'control_components_per_call':CONTROL_COMPONENTS,'control_component_completion':'UNKNOWN_UNTIL_CONTROL_SYNCHRONIZED_AND_RETURNED'}
 def publish(self):self.persist(self.snapshot())
 def hash_state_file(self,path,callback):
  n=path.stat().st_size;need(0<n<=2026752*4+(1<<20)+8,'state file hash size')
  result=self.state_file_sha.call(n,callback,label='state_file_sha',completion='SYNCHRONIZED');self.state_hash_calls+=1;self.publish();return result
 def hash_video_file(self,path,callback):
  n=path.stat().st_size;need(0<n<=32<<20,'video file hash size')
  result=self.video_file_sha.call(n,callback,label='video_file_sha',completion='SYNCHRONIZED');self.video_hash_calls+=1;self.publish();return result
 def charge_compare(self,n,callback):return self.compares.call(n,callback,label='reference_byte_compare',completion='SYNCHRONIZED')
 def charge_block(self,expert,block,callback):return self.block_loads.call(1,callback,label=f'block_load_expert{expert}_block{block}',completion='CPU_RETURN_GPU_UNKNOWN')
 def synchronize_phase(self,fn):return self.phase_sync.call(1,fn,label='phase_device_synchronize',completion='SYNCHRONIZED')
 def synchronize_checkpoint(self,fn):return self.checkpoints.call(1,fn,label='cache_checkpoint_synchronize',completion='SYNCHRONIZED')
 def charge_control(self,callback):return self.controls.call(1,callback,label='same_input_dense32_control_and_synchronize',completion='SYNCHRONIZED')
 def verify_complete(self):
  for item in (self.block_loads,self.phase_sync,self.checkpoints,self.controls):need(item.inflight is None and item.reserved==item.completed==item.limit,'incomplete ABBA work')
  need(self.state_file_sha.inflight is None and self.video_file_sha.inflight is None and self.state_hash_calls==50 and self.video_hash_calls==5,'state/video hash accounting incomplete')
  need(self.compares.inflight is None and self.compares.reserved==self.compares.completed and self.selector.inflight is None and self.selector.reserved==self.selector.returned==LOGICAL_COPY_CALLS,'incomplete comparison/copy')

def validate_reference_gate(phases,gate,prior):
 """Independent metadata check of all 55 unique gated phase/reference pairs."""
 need(len(phases)==5 and [p['arm'] for p in phases]==list(ARMS),'phase order differs')
 checks=gate['checks'];expected={(arm,k) for arm in ARMS for k in (*STATES,'video')}
 need(len(checks)==55 and {(x['arm'],x['kind']) for x in checks}==expected and all(x.get('byte_equal') is True for x in checks),'reference gate coverage differs')
 for phase in phases:
  arm=phase['arm'];refs=prior['bf16' if arm=='warmup' else 'dense32'];need(set(phase['states'])==STATES,'phase states differ')
  for kind in (*STATES,'video'):
   cur=phase['video'] if kind=='video' else phase['states'][kind];old=refs[kind];check=next(x for x in checks if x['arm']==arm and x['kind']==kind)
   need(cur['bytes']==old['bytes']==check['bytes_each'] and cur['sha256']==old['sha256'],'phase/reference identity differs')
 return True
