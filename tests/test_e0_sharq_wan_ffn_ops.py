import collections,importlib.util,unittest
from pathlib import Path
from types import SimpleNamespace
p=Path(__file__).resolve().parents[1]/'tools/e0_sharq_wan_ffn_ops.py'
s=importlib.util.spec_from_file_location('sharq_ops_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Val:
 def __init__(self,name):self.name=name;self.shape=(256,5120)
class Ledger:
 def __init__(self):self.count=collections.Counter()
 def call(self,name,fn,*args):self.count[name]+=1;return fn(*args)
class Fake:
 def __init__(self):self.prepares=[]
 def sharq_prepare(self,x,n):self.prepares.append(x);return ('sp','meta','sps','res','res_scale',3)
 def dense_prepare(self,x):self.prepares.append(x);return ('dense','scale',3)
 def output_scale(self,a,w):return a*w
 def sparse_gemm(self,*a):return Val('sparse')
 def dense_accum(self,*a):return Val('combined')
 def dense_gemm(self,*a):return Val('dense')
 def bias_add(self,a,b):return Val('biased')
 def bf16_linear(self,*a):return Val('bf16')
 def dense16_prepare(self,x,s):self.prepares.append(x);return ('packed','scale')
 def dense16_gemm(self,*a):return Val('dense16')
 def gelu_tanh(self,x):return Val('h_from_'+x.name)
 def gate_multiply(self,x,g):return Val('gate_'+x.name)
def layer():return SimpleNamespace(out_features=13824,in_features=5120,weight_scale=2,weight_q='payload',scale_w_sparse='sparse',scale_w_dense='dense',bias='bias',input_global_scale=1)
class Tests(unittest.TestCase):
 def test_all_actual_arm_call_ledgers_and_current_q2(self):
  for arm in m.ARMS:
   api=Fake();ledger=Ledger();x=Val('x');out=m.forward(arm,x,layer(),layer(),'gate',api,ledger)
   self.assertEqual(dict(ledger.count),m.expected_calls(arm))
   if arm in ('sharq','dense_shared32','dense_group16'):
    self.assertIs(api.prepares[0],x);self.assertIs(api.prepares[1],out['h'])
 def test_reject_unknown_arm(self):
  with self.assertRaises(ValueError):m.forward('invalid',Val('x'),layer(),layer(),None,Fake(),Ledger())
if __name__=='__main__':unittest.main()
