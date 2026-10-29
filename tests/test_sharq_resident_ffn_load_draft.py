import sys, pathlib, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
from sharq_resident_ffn_load_draft import SelectiveFFNLoad

class Buffer:
 def __init__(self): self.calls=[]
 def copy_(self, value, non_blocking=False): self.calls.append((value,non_blocking));return self
class MM:
 def __init__(self):
  self.has_lora_branch=False;self.has_diff=False;self.is_post_adapter=False;self.lora_attrs={}
  self.base_attrs=[('w','weight',False),('b','bias',False)];self.weight_cuda_buffer=Buffer();self.bias_cuda_buffer=Buffer();self.original=0
 def load_state_dict(self,d,b,a=None):
  self.original+=1;self.weight=self.weight_cuda_buffer.copy_(d['w'],non_blocking=True);self.bias=self.bias_cuda_buffer.copy_(d['b'],non_blocking=True)
class T(unittest.TestCase):
 def test_copy_skip_and_restore(self):
  x=MM();events=[];calls=[];cache={(0,3,'ffn.0'):(1,2,3,4)}
  with SelectiveFFNLoad([(x,0,'ffn.0')],cache,lambda n,*_:n,lambda k,w,b,skip,f:(calls.append(k),f()),events.append,lambda *args:None,lambda x:4,lambda *args:None) as s:
   x.load_state_dict({'w':1,'b':2},3);self.assertEqual(x.original,1)
   s.enabled=True;x.load_state_dict({'w':1,'b':2},3)
   self.assertEqual(x.weight_cuda_buffer.calls,[(1,True)]);self.assertEqual(x.bias_cuda_buffer.calls,[(2,True),(2,True)]);self.assertEqual(calls,[(0,3,'ffn.0'),(0,3,'ffn.0')])
   with self.assertRaises(ValueError):x.load_state_dict({'w':1,'b':2},2)
   with self.assertRaises(ValueError):x.load_state_dict({'w':1},3)
  self.assertNotIn('load_state_dict',x.__dict__)
 def test_two_buffers_and_mode_transitions(self):
  first,second=MM(),MM();cache={(0,b,'ffn.0'):(1,2,3,4) for b in range(40)};events=[]
  with SelectiveFFNLoad([(first,0,'ffn.0'),(second,0,'ffn.0')],cache,lambda n,*_:n,lambda k,w,b,skip,f:f(),events.append,lambda *args:None,lambda x:4,lambda *args:None) as scope:
   scope.enabled=True
   for block in range(40):
    (first if block%2==0 else second).load_state_dict({'w':block,'b':block},block)
   self.assertEqual(len(events),40)
   self.assertEqual(len(first.bias_cuda_buffer.calls)+len(second.bias_cuda_buffer.calls),40)
   self.assertEqual(len(first.weight_cuda_buffer.calls)+len(second.weight_cuda_buffer.calls),0)
   scope.enabled=False;first.load_state_dict({'w':1,'b':2},0);self.assertEqual(first.original,1)
 def test_adapter_source_rejected(self):
  x=MM();x.lora_attrs={'lora_down':'lora_down_name'};x.lora_down_name='adapter';cache={(0,3,'ffn.0'):(1,2,3,4)}
  with SelectiveFFNLoad([(x,0,'ffn.0')],cache,lambda n,*_:n,lambda k,w,b,skip,f:f(),lambda x:None,lambda *args:None,lambda x:4,lambda *args:None) as scope:
   scope.enabled=True
   with self.assertRaises(ValueError):x.load_state_dict({'w':1,'b':2,'adapter':3},3)
 def test_partial_install_rolls_back(self):
  a=MM();b=MM();b.has_diff=True
  with self.assertRaises(ValueError):
   with SelectiveFFNLoad([(a,0,'ffn.0'),(b,0,'ffn.2')],{},lambda *x:x[0],lambda k,w,b,skip,f:f(),lambda e:None,lambda *args:None,lambda x:4,lambda *args:None):pass
  self.assertNotIn('load_state_dict',a.__dict__)
if __name__=='__main__':unittest.main()
