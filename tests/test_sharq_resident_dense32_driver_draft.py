import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import sharq_resident_dense32_driver_draft as d
class Runtime:
 def __init__(self):self.modes=[]
 def set_next_resident(self,v):self.modes.append(v)
class T(unittest.TestCase):
 def test_order_and_gates(self):
  r=Runtime();checks=[];phases=[];p={'bf16':'oldbf','dense32':'oldfp'}
  def phase(i,m,a):phases.append(a);return {'mode':m,'arm':a,'ffn_sites':160,'linear_sites':320}
  z=d.run(runtime=r,phase=phase,validate=lambda *x:checks.append(x),checkpoint=lambda:{'entries':160,'sha':'x'},prior=p,progress=lambda *x:None)
  self.assertEqual(phases,list(d.ARMS));self.assertEqual(r.modes,[False,True,True,False]);self.assertEqual([x[2] for x in checks],['oldbf','oldfp','oldfp','oldfp','oldfp']);self.assertEqual(len(z['phases']),5)
 def test_failed_gate_prevents_later_phases(self):
  phases=[]
  def phase(i,m,a):phases.append(a);return {'mode':m,'arm':a,'ffn_sites':160,'linear_sites':320}
  def fail(a,*_):
   if a=='resident_B1':raise ValueError('different')
  with self.assertRaises(ValueError):d.run(runtime=Runtime(),phase=phase,validate=fail,checkpoint=lambda:{'entries':160},prior={'bf16':1,'dense32':2},progress=lambda *x:None)
  self.assertEqual(phases,['warmup','legacy_A1','resident_B1'])
 def test_native_phase_invokes_registered_chain(self):
  from types import SimpleNamespace as NS
  from sharq_resident_prefetch_audit_draft import PrefetchAudit
  class Buffer:
   def load_state_dict(self,d,b,a=None):return None
  buffers=[Buffer() for _ in range(4)];models=[NS(transformer_infer=NS(offload_manager=NS(cuda_buffers=buffers[:2]))),NS(transformer_infer=NS(offload_manager=NS(cuda_buffers=buffers[2:])))]
  class Context:
   def __enter__(self):return self
   def __exit__(self,*a):pass
  class Event:
   def elapsed_time(self,b):return 1.0
  class Tap(Context):
   def __init__(self,*a,**kw):self.events=[(Event(),Event())]*160;self.transfers=[1]*162
   def finish(self):pass
  class Pipe:
   runner=NS(model=NS(model=models),scheduler=NS(generator=None))
   def generate(self,**kwargs):
    for i in range(162):buffers[i%4].load_state_dict({},i%40)
  class RT:
   def begin(self,rid,mode):self.started=(rid,mode)
   def finish(self):return {'entries':160}
  class Ledger:
   def call(self,rid,cost,fn):return fn()
  torch=NS(inference_mode=lambda:Context(),cuda=NS(synchronize=lambda:None));events=[];sync=[]
  x=d.native_phase(torch=torch,pipe=Pipe(),guard=NS(ledger=Ledger()),runtime=RT(),vm=NS(LightTap=Tap),rid='x',mode='dense32',arm='resident_B1',image_path='i',prompt='p',output_path='o',save_state=lambda *a:None,check=lambda:None,authorize=lambda:None,emit=events.append,charge_block=lambda e,b,fn:fn(),PrefetchAudit=PrefetchAudit,phase_synchronize=lambda:sync.append('paid'))
  self.assertEqual(x['block_prefetch']['actual_block_load_calls_returned'],162);self.assertEqual(sync,['paid']);self.assertEqual(len(events),162);self.assertEqual(x['physical_bus_bytes'],'UNKNOWN')
if __name__=='__main__':unittest.main()
