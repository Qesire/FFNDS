import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from sharq_resident_same_input_control_draft import control
class T(unittest.TestCase):
 def test_strict_and_failure_preserves_result(self):
  events=[]
  def paid(f):events.append('predebit');v=f();events.append('complete');return v
  def persist(x):events.append(x)
  out=control(b'good',lambda:b'good',lambda a,b:{'byte_equal':a==b},lambda:events.append('sync'),paid,persist)
  self.assertTrue(out['byte_equal']);self.assertEqual(events[0],'predebit');self.assertEqual(events[-1],{'byte_equal':True})
  with self.assertRaises(ValueError):control(b'a',lambda:b'b',lambda a,b:{'byte_equal':a==b},lambda:None,paid,persist)
  self.assertEqual(events[-1],{'byte_equal':False})
 def test_factory_uses_shared_input_cache_and_bias(self):
  from types import SimpleNamespace as NS
  from sharq_resident_same_input_control_draft import make_dense32_reference
  class Tensor:
   shape=(1,2);dtype='bf16';device='cuda'
   def contiguous(self):return self
   def view(self,dtype):return self
   def numel(self):return 2
  class Finite:
   def all(self):return True
  x=Tensor();out=Tensor();bias=Tensor();q,ss,sd,sw=[object() for _ in range(4)]
  bias.shape=(2,);calls=[]
  def linear(v,layer,api,meter):
   self.assertIs(v,x);self.assertIs(layer.bias,bias);self.assertIs(layer.weight_q,q)
   for k in ('dense32_prepare','output_scale','dense32_gemm','bias_add'):meter.call(k,lambda:None)
   return out
  bridge=NS(author_call=lambda torch,fn,*args:fn(*args));torch=NS(uint8='u8',isfinite=lambda x:Finite(),equal=lambda a,b:True)
  f=make_dense32_reference(torch=torch,ops=NS(dense32_linear=linear),bridge=bridge,api=None,charge=lambda fn:fn(),synchronize=lambda:calls.append('sync'),persist=lambda x:calls.append(x))
  r=NS(cache={(0,0,'ffn.0'):(q,ss,sd,sw)});result=f(r,NS(bias=bias),x,out,(0,0,'ffn.0'))
  self.assertTrue(result['byte_equal']);self.assertTrue(result['finite']);self.assertEqual(calls[0],'sync')
if __name__=='__main__':unittest.main()
