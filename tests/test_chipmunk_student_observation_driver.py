"""Real driver orchestration with synthetic fake runtime; no model/CUDA."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import chipmunk_student_observation_driver as d
class T:
 def __getitem__(self,x):return self
 def to(self,x):return self
 def tolist(self):return list(range(64))
class Ledger:
 def call(self,rid,cost,fn):return fn()
class FakeEvent:
 def elapsed_time(self,other):return 1
class FakeTap:
 def __init__(self,*a,**k):self.events=[(FakeEvent(),FakeEvent())]*160;self.transfers=[{'requested_H2D_tensor_bytes':0,'requested_D2D_tensor_bytes':0,'original_W2_H2D_bytes':0,'enqueue_wall_seconds':0,'metadata_observer_wall_seconds':0}]*162;self.stages=[];self.book=SimpleNamespace(peak=0);self.cache_puts=self.cache_releases=40;self.sparse_sites=40
 def __enter__(self):return self
 def __exit__(self,*x):pass
 def finish(self):pass
class Context:
 def __enter__(self):pass
 def __exit__(self,*x):pass
class Test(unittest.TestCase):
 def test_two_phase_real_driver_uses_distinct_observers(self):
  cuda=SimpleNamespace(synchronize=lambda:None,max_memory_allocated=lambda:0,max_memory_reserved=lambda:0,get_allocator_backend=lambda:'native',memory_allocated=lambda:0,memory_reserved=lambda:0)
  t=SimpleNamespace(randperm=lambda *a,**k:T(),Generator=lambda **k:SimpleNamespace(manual_seed=lambda _:None),cuda=cuda,inference_mode=Context)
  pipe=SimpleNamespace(runner=SimpleNamespace(scheduler=SimpleNamespace(generator=None)),generate=lambda **k:None);guard=SimpleNamespace(ledger=Ledger());names=[]
  def factory(mode,rows,selector_list):names.append(mode);return SimpleNamespace(sink=SimpleNamespace(finalize=lambda:{'sites':40}))
  def capture(rid):return None,{x:{} for x in d.STATE_NAMES}
  with patch.object(d,'make_tap',return_value=FakeTap):
   result=d.execute(observer_factory=factory,torch=t,pipe=pipe,guard=guard,vm=None,native=None,run_ids=['a','b'],image_path='i',prompt='p',output_paths=['x','y'],capture_factory=capture,inspect_video=lambda *_:{'frames':81,'fps':16,'size':[832,464],'sha256':'a'*64},check=lambda:None,authorize=lambda _:None,gate=None,progress=lambda *a:None,kernels=None,emit=lambda _:None)
  self.assertEqual(names,list(d.MODES));self.assertEqual([x['mode'] for x in result['phases']],list(d.MODES));self.assertTrue(all(x['observer_manifest']['sites']==40 for x in result['phases']))
if __name__=='__main__':unittest.main()
