import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import torch
from chipmunk_wan_packed_pregate import packed_pregate
from chipmunk_packed_native_math import packed_action
class Test(unittest.TestCase):
 def test_same_mask_reference_and_one_gate(self):
  g=torch.Generator().manual_seed(15);x=torch.randn(5,4,generator=g).bfloat16();w1=torch.randn(6,4,generator=g).bfloat16();b1=torch.randn(6,generator=g).bfloat16();w2=torch.randn(4,6,generator=g).bfloat16();h=torch.randn(5,6,generator=g).bfloat16();y=torch.randn(5,4,generator=g).bfloat16();gate=torch.randn(4,generator=g).bfloat16();mask=torch.tensor([0,2,5])
  mm=lambda a,b:a.float()@b.float();h0=h.clone();y0=y.clone()
  new=packed_pregate(x,w1,b1,w2,h,y,mask,mm_fp32=mm);ref=packed_action(x,w1,b1,w2,h,y,gate,mask,mm_fp32=mm)
  self.assertTrue(torch.equal((new*gate).bfloat16(),ref));self.assertTrue(torch.equal(h,h0));self.assertTrue(torch.equal(y,y0))
 def test_no_second_bias_or_gate(self):
  x=torch.zeros((2,2),dtype=torch.bfloat16);w1=torch.zeros((3,2),dtype=torch.bfloat16);b1=torch.zeros(3,dtype=torch.bfloat16);w2=torch.zeros((2,3),dtype=torch.bfloat16);h=torch.zeros((2,3),dtype=torch.bfloat16);y=torch.full((2,2),2,dtype=torch.bfloat16)
  out=packed_pregate(x,w1,b1,w2,h,y,torch.tensor([0]),mm_fp32=lambda a,b:a.float()@b.float());self.assertTrue(torch.equal(out,y))
if __name__=='__main__':unittest.main()
