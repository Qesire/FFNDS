"""CPU synthetic test for real installed postprocess wrapper."""
import contextlib,sys,threading,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
import chipmunk_student_observation_tap as t
class Base:
 def __init__(self,runner):self.runner=runner;self.guard=types.SimpleNamespace(torch=torch,ledger=object());self.rid='r';self.block=(0,0);self.stack=contextlib.ExitStack()
 def install(self):pass
 def step(self,e=None):return 1
class Test(unittest.TestCase):
 def test_actual_postprocess_called_once_then_byte_check(self):
  calls=[];infer=types.SimpleNamespace(block_idx=0,infer_ffn=lambda *a:None,post_process=lambda x,y,g,p=None:(calls.append('original') or x.add_(y*g.squeeze())))
  runner=types.SimpleNamespace(model=types.SimpleNamespace(model=[types.SimpleNamespace(transformer_infer=infer)]))
  observer=types.SimpleNamespace(check_postprocess=lambda **kw:calls.append(('check',torch.equal(kw['expected'].view(torch.uint8),kw['actual'].view(torch.uint8)))))
  cls=t.make_tap(types.SimpleNamespace(LightTap=Base),types.SimpleNamespace(patch_method=lambda obj,name,fn:patch.object(obj,name,fn)),mode='h_cache_dense',rows=torch.arange(1),kernels=None,emit=lambda _:None,observer=observer,geometry=(1,2,3,1));tap=cls(runner);tap.install()
  x=torch.tensor([[1.,2.]],dtype=torch.bfloat16);y=torch.tensor([[.5,1.]],dtype=torch.bfloat16);gate=torch.ones(1,1,2,dtype=torch.bfloat16);expected=x.clone().add_(y*gate.squeeze());tap.pending_post=(0,expected)
  result=infer.post_process(x,y,gate);self.assertIs(result,x);self.assertEqual(calls,['original',('check',True)]);self.assertIsNone(tap.pending_post);tap.stack.close()
if __name__=='__main__':unittest.main()
