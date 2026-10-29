import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import e0_sharq_wan_all160_video_driver_v2 as d
class Context:
 def __enter__(self):return self
 def __exit__(self,*a):pass
class Event:
 def elapsed_time(self,other):return 1.
class Tap(Context):
 def __init__(self,*a,**kw):self.events=[(Event(),Event())]*160;self.transfers=[{}]*162;self.finished=False;self.proxy=kw['runtime']
 def finish(self):self.finished=True
class Audit(Context):
 def __init__(self,*a,**kw):pass
 def finish(self):return {'original_FFN_prefetch_observations':162,'original_FFN_prefetch_skips':0}
class Ledger:
 def call(self,rid,cost,fn):return fn()
class Runtime(Context):
 def __init__(self):self.phases=[];self.rid=None;self.seen=set();self.cache={};self.cache_bytes=0
 def begin(self,rid,mode):self.phases.append((rid,mode));self.rid=rid
 def finish(self):return {'cache_entries':160}
class Tests(unittest.TestCase):
 def test_registered_phase_invokes_generation_and_preserves_prefetch(self):
  called=[];torch=SimpleNamespace(inference_mode=Context,cuda=SimpleNamespace(synchronize=lambda:None));pipe=SimpleNamespace(generate=lambda **kw:called.append(kw),runner=SimpleNamespace(scheduler=SimpleNamespace(generator='old')));runtime=Runtime();vm=SimpleNamespace(LightTap=Tap);guard=SimpleNamespace(ledger=Ledger());states=[]
  with patch.object(d,'OriginalFFNPrefetchAudit',Audit):
   result=d.run_clip(torch=torch,pipe=pipe,guard=guard,runtime=runtime,vm=vm,rid='r',mode='sharq',image_path='i',prompt='p',output_path='v',save_state=lambda *a:None,check=lambda:None,authorize=lambda:states.append('authorized'),emit=lambda e:None)
  self.assertEqual(states,['authorized']);self.assertEqual(called[0]['seed'],42);self.assertEqual(runtime.phases,[('r','sharq')]);self.assertEqual(result['actual_FFN_linear_dispatches'],320);self.assertEqual(result['original_prefetch']['original_FFN_prefetch_skips'],0)
 def test_bad_mode_cannot_call_authorizer(self):
  with self.assertRaises(ValueError):d.run_clip(torch=None,pipe=None,guard=None,runtime=None,vm=None,rid='r',mode='bad',image_path='',prompt='',output_path='',save_state=None,check=None,authorize=lambda:None,emit=None)
 def test_neutral_rejection_stops_before_quantized_generation(self):
  calls=[];state={'sha256':'a','bytes':1,'raw_tensor_sha256':'b','shape':[1],'dtype':'bf16'}
  def factory(rid):
   values={name:dict(state) for name in d.STATE_NAMES}
   if rid=='new':values['flow_step0']['sha256']='changed'
   return lambda *a:None,values
  def fake(**kw):calls.append(kw['mode']);return {'run_id':kw['rid'],'mode':kw['mode']}
  video=lambda rid,path:{'frames':81,'fps':16,'size':[832,464],'sha256':'0'*64,'bytes':1}
  with patch.object(d,'run_clip',fake):
   with self.assertRaises(ValueError):d.execute_part(torch=None,pipe=None,guard=None,runtime=Runtime(),vm=None,run_ids=['old','new','q','s'],image_path='i',prompt='p',output_paths=['a','b','c','d'],capture_factory=factory,inspect_video=video,check=lambda:None,authorize=lambda:None,emit=lambda e:None)
  self.assertEqual(calls,['bf16','bf16_cache'])
 def test_complete_four_phase_order_with_same_noise(self):
  calls=[];state={'sha256':'a','bytes':1,'raw_tensor_sha256':'b','shape':[1],'dtype':'bf16'}
  def factory(rid):return lambda *a:None,{name:dict(state) for name in d.STATE_NAMES}
  def fake(**kw):calls.append(kw['mode']);return {'run_id':kw['rid'],'mode':kw['mode']}
  video=lambda rid,path:{'frames':81,'fps':16,'size':[832,464],'sha256':'0'*64,'bytes':1}
  with patch.object(d,'run_clip',fake):result=d.execute_part(torch=None,pipe=None,guard=None,runtime=Runtime(),vm=None,run_ids=['old','new','q','s'],image_path='i',prompt='p',output_paths=['a','b','c','d'],capture_factory=factory,inspect_video=video,check=lambda:None,authorize=lambda:None,emit=lambda e:None)
  self.assertEqual(calls,['bf16','bf16_cache','dense32','sharq']);self.assertTrue(result['same_device_BF16_neutral']);self.assertTrue(result['paired_noise']);self.assertFalse(result['quality_acceptance'])
 def test_first_phase_runs_before_patch_and_cleanup_occurs(self):
  trace=[]
  class Traced(Context):
   def __enter__(self):trace.append('install');return self
   def __exit__(self,*a):trace.append('restore')
  state={'sha256':'a','bytes':1,'raw_tensor_sha256':'b','shape':[1],'dtype':'bf16'}
  def fake(**kw):trace.append(kw['mode']);return {'mode':kw['mode']}
  def factory(rid):return lambda *a:None,{name:dict(state) for name in d.STATE_NAMES}
  video=lambda rid,path:{'frames':81,'fps':16,'size':[832,464],'sha256':'0'*64,'bytes':1}
  with patch.object(d,'run_clip',fake):d.execute_part(torch=None,pipe=None,guard=None,runtime=Traced(),vm=None,run_ids=['a','b','c','d'],image_path='i',prompt='p',output_paths=['a','b','c','d'],capture_factory=factory,inspect_video=video,check=lambda:None,authorize=lambda:None,emit=lambda e:None)
  self.assertEqual(trace,['bf16','install','bf16_cache','dense32','sharq','restore'])
 def test_progress_preserves_first_phase_before_second_failure(self):
  state={'sha256':'a','bytes':1,'raw_tensor_sha256':'b','shape':[1],'dtype':'bf16'};events=[]
  def factory(rid):return lambda *a:None,{name:dict(state) for name in d.STATE_NAMES}
  def fake(**kw):
   if kw['mode']=='bf16_cache':raise RuntimeError('step0 failure')
   return {'run_id':kw['rid'],'mode':kw['mode']}
  video=lambda rid,path:{'frames':81,'fps':16,'size':[832,464],'sha256':'0'*64,'bytes':1}
  with patch.object(d,'run_clip',fake):
   with self.assertRaisesRegex(RuntimeError,'step0 failure'):d.execute_part(torch=None,pipe=None,guard=None,runtime=Runtime(),vm=None,run_ids=['a','b','c','d'],image_path='i',prompt='p',output_paths=['a','b','c','d'],capture_factory=factory,inspect_video=video,check=lambda:None,authorize=lambda:None,emit=lambda e:None,progress=lambda event,data:events.append((event,data['run_id'])))
  self.assertEqual(events,[('phase_started','a'),('phase_evidence_complete','a'),('phase_started','b')])
 def test_failure_reports_live_partial_cache_without_retry(self):
  def fail(**kwargs):raise RuntimeError('synthetic')
  torch=SimpleNamespace(inference_mode=Context,cuda=SimpleNamespace(synchronize=lambda:None));pipe=SimpleNamespace(generate=fail,runner=SimpleNamespace(scheduler=SimpleNamespace(generator=None)));events=[];rt=Runtime()
  with patch.object(d,'OriginalFFNPrefetchAudit',Audit):
   with self.assertRaises(RuntimeError):d.run_clip(torch=torch,pipe=pipe,guard=SimpleNamespace(ledger=Ledger()),runtime=rt,vm=SimpleNamespace(LightTap=Tap),rid='r',mode='dense32',image_path='',prompt='',output_path='',save_state=lambda *a:None,check=lambda:None,authorize=lambda:None,emit=events.append)
  self.assertTrue(events[-1]['runtime_phase_still_active']);self.assertEqual(events[-1]['completed_static_weight_packs'],0)
if __name__=='__main__':unittest.main()
