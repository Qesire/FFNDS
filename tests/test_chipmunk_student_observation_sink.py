import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from chipmunk_student_observation_sink import Sink
class Counter:
 def __init__(self):self.n=0;self.hold=0
 def summary(self):return {'peak_tracked_bytes_plus_reserved_and_atomic_temp':self.n+self.hold}
 def reserve_payload(self,n):self.hold=n
 def observe_external(self,p):self.n+=Path(p).stat().st_size;self.hold=0
 def binding(self,p,hf):return {'path':str(Path(p).relative_to(hf)),'bytes':Path(p).stat().st_size,'sha256':'synthetic'}
 def write_json(self,p,x):Path(p).write_text(json.dumps(x));self.n+=Path(p).stat().st_size
class Test(unittest.TestCase):
 def test_sample_snapshot_and_duplicate(self):
  with tempfile.TemporaryDirectory() as d:
   h=Path(d);counter=Counter();saved=[]
   def save(tensors,path):
    from safetensors.torch import save_file
    saved.append(set(tensors));save_file(tensors,path)
   s=Sink(torch=torch,counter=counter,out=h/'obs',hf_root=h,save_file=save,max_bytes=100000,host_guard=lambda _:None,max_host_copy_bytes=100000)
   sel=torch.tensor([0]);v=torch.tensor([[1.,2.],[3.,4.]])
   gate=torch.ones((1,6,5120),dtype=torch.bfloat16)[:,:1,:];s.metadata(0,sel,torch.tensor([.5]),sel,gate,[{'row':1}]);self.assertEqual(len(json.loads((h/'obs/block00_metadata.json').read_text())['gate']),5120);self.assertEqual(list(gate.stride()),[30720,5120,1]);s.sample(0,sel,[{'row':1}],v,v,v,v,v,v);s.snapshot(0,v,v,v,v,v,v,torch.tensor([1.,1.]),sel,sel,sel,[])
   self.assertEqual(len(saved),7);self.assertEqual(len(s.records[0]['snapshot']),6)
   with self.assertRaises(ValueError):s.metadata(0,sel,sel,sel,gate,[])
 def test_fails_incomplete_and_perrow_gate(self):
  with tempfile.TemporaryDirectory() as d:
   s=Sink(torch=torch,counter=Counter(),out=Path(d)/'obs',hf_root=Path(d),save_file=lambda *x:None,max_bytes=100,host_guard=lambda _:None,max_host_copy_bytes=100)
   with self.assertRaises(ValueError):s.gate_slice(torch.ones(2,2),0,2)
   with self.assertRaises(ValueError):s.finalize()
if __name__=='__main__':unittest.main()

class FailureTest(unittest.TestCase):
 def test_partial_temp_retained_and_counted(self):
  with tempfile.TemporaryDirectory() as d:
   h=Path(d);counter=Counter()
   def fail(tensors,path):Path(path).write_bytes(b'partial');raise OSError('synthetic failure')
   s=Sink(torch=torch,counter=counter,out=h/'obs',hf_root=h,save_file=fail,max_bytes=1000,host_guard=lambda _:None,max_host_copy_bytes=1000)
   with self.assertRaises(OSError):s._tensor_file(h/'obs'/'x.safetensors',{'x':torch.ones(2)})
   self.assertEqual((h/'obs'/'x.safetensors.tmp').read_bytes(),b'partial');self.assertFalse((h/'obs'/'x.safetensors').exists());self.assertEqual(counter.n,7);self.assertEqual(counter.hold,0)
 def test_host_guard_before_copy_and_reserve(self):
  with tempfile.TemporaryDirectory() as d:
   h=Path(d);counter=Counter()
   def reject(n):raise MemoryError('budget')
   s=Sink(torch=torch,counter=counter,out=h/'obs',hf_root=h,save_file=lambda *args:None,max_bytes=1000,host_guard=reject,max_host_copy_bytes=1000)
   with self.assertRaises(MemoryError):s._tensor_file(h/'obs'/'x',{'x':torch.ones(2)})
   self.assertEqual(counter.n,0);self.assertEqual(counter.hold,0);self.assertFalse((h/'obs'/'x').exists())

class ActualMetadataVerifierTest(unittest.TestCase):
 def test_student_and_dense_real_gate_shape_roundtrip(self):
  import hashlib,struct
  import e0_chipmunk_student_observation_owner as owner
  from chipmunk_student_observation_coordinates import coordinates
  with tempfile.TemporaryDirectory() as d:
   h=Path(d);gate=torch.arange(6*5120,dtype=torch.bfloat16).reshape(1,6,5120)[:,:1,:];self.assertEqual(gate.shape,(1,1,5120));self.assertEqual(gate.stride(),(30720,5120,1))
   selector=list(range(64));ev=coordinates(selector);rows=torch.tensor(selector);mask=torch.arange(5120);score=torch.ones(13824);masksha=hashlib.sha256(struct.pack('<'+'q'*5120,*range(5120))).hexdigest()
   for mode in ('h_cache_dense','high_h_readonly_p64'):
    s=Sink(torch=torch,counter=Counter(),out=h/mode,hf_root=h,save_file=lambda *a:None,max_bytes=1<<20,host_guard=lambda _:None,max_host_copy_bytes=1<<20)
    if mode=='h_cache_dense':s.dense_metadata(0,ev,selector,gate)
    else:s.metadata(0,mask,score,rows,gate,ev)
    meta=json.loads((h/mode/'block00_metadata.json').read_text());self.assertTrue(owner.validate_site_metadata(meta,mode,0,selector,masksha if mode!='h_cache_dense' else None))
    meta['gate_stride']=[5120,5120,1]
    with self.assertRaises(ValueError):owner.validate_site_metadata(meta,mode,0,selector,masksha if mode!='h_cache_dense' else None)
 def test_sink_rejects_wrong_gate_stride(self):
  from chipmunk_student_observation_sink import gate_metadata
  with self.assertRaises(ValueError):gate_metadata(torch,torch.ones((1,1,5120),dtype=torch.bfloat16))
