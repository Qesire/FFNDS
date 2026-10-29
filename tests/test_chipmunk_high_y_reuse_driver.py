import sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import chipmunk_high_y_reuse_driver as d
class Context:
 def __enter__(self):return self
 def __exit__(self,*a):pass
class Event:
 def elapsed_time(self,other):return 1
class Tap(Context):
 def __init__(self,*a,**kw):
  self.events=[(Event(),Event())]*160;self.transfers=[{'requested_H2D_tensor_bytes':1,'requested_D2D_tensor_bytes':0,'original_W2_H2D_bytes':1,'enqueue_wall_seconds':0,'metadata_observer_wall_seconds':0}]*162;self.cache_puts=40;self.cache_releases=40;self.reused=0;self.book=NS(peak=5);self.stages=[]
 def finish(self):pass
class Ledger:
 def call(self,rid,cost,fn):return fn()
class Test(unittest.TestCase):
 def test_modes_and_costs(self):
  class Pipe:
   runner=NS(scheduler=NS(generator=None))
   def generate(self,**kw):pass
  torch=NS(inference_mode=lambda:Context(),cuda=NS(synchronize=lambda:None,max_memory_allocated=lambda:1,max_memory_reserved=lambda:1));phases=[]
  def factory(rid):return lambda *a:None,{k:1 for k in d.STATE_NAMES}
  def mk(vm,native,mode,emit):
   class TT(Tap):
    def __init__(self,*a,**kw):super().__init__();self.reused=40 if mode=='high_y_reuse' else 0
   return TT
  with patch.object(d,'make_tap',mk),patch.object(d,'memory_snapshot',lambda t:{}):
   r=d.execute(torch=torch,pipe=Pipe(),guard=NS(ledger=Ledger()),vm=None,native=None,run_ids=['a','b','c'],image_path='i',prompt='p',output_paths=['a','b','c'],capture_factory=factory,inspect_video=lambda *a:{'frames':81,'fps':16,'size':[832,464],'sha256':'0'*64},check=lambda:None,authorize=lambda x:None,gate=None,progress=lambda e,x:phases.append((e,x)),kernels=None,emit=lambda x:None)
  self.assertEqual([x['actual_dense_FFN_GEMMs'] for x in r['phases']],[320,320,240]);self.assertEqual([x['actual_reused_FFN_sites'] for x in r['phases']],[0,0,40]);self.assertEqual(sum(x['actual_dense_FFN_GEMMs'] for x in r['phases']),880)
if __name__=='__main__':unittest.main()
