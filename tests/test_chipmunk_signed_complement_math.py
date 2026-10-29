import pathlib,sys,unittest,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
import torch
from chipmunk_signed_complement_math import packed,split_dc_ac,checked_mask,p64_observer
class Signed(unittest.TestCase):
 def test_complement_excludes_selected(self):
  x=torch.zeros((2,2),dtype=torch.bfloat16);w1=torch.zeros((2,2),dtype=torch.bfloat16);w2=torch.eye(2,dtype=torch.bfloat16);b=torch.zeros(2,dtype=torch.bfloat16);h=torch.zeros((2,2),dtype=torch.bfloat16);y=torch.zeros((2,2),dtype=torch.bfloat16);g=torch.ones(2,dtype=torch.bfloat16);idx=torch.tensor([0]);mm=lambda a,b:a.float()@b.float()
  _,out=packed(x,w1,b,w2,h,y,g,idx,torch.tensor([9.,2.]),mm_fp32=mm);self.assertTrue(torch.equal(out,torch.tensor([[0.,2.],[0.,2.]],dtype=torch.bfloat16)))
 def test_baseline_matches_frozen_packed_action(self):
  from chipmunk_packed_native_math import packed_action
  from unittest.mock import patch
  x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16);w=torch.eye(2,dtype=torch.bfloat16);b=torch.zeros(2,dtype=torch.bfloat16);h=torch.zeros((2,2),dtype=torch.bfloat16);y=torch.ones((2,2),dtype=torch.bfloat16);g=torch.ones(2,dtype=torch.bfloat16);idx=torch.tensor([1]);mm=lambda a,b:a.float()@b.float()
  _,ours=packed(x,w,b,w,h,y,g,idx,None,mm_fp32=mm)
  with patch.object(torch,'mm',side_effect=lambda a,b,**kw:mm(a,b)):
   prior=packed_action(x,w,b,w,h,y,g,idx)
  self.assertTrue(torch.equal(ours,prior))
 def test_dc_ac_known(self):
  r=split_dc_ac(torch.tensor([[2.,0.],[0.,0.]]));self.assertEqual((r['dc_energy'],r['ac_energy'],r['total_energy']),(2.,2.,4.))
 def test_sha_gate(self):
  with self.assertRaises(ValueError):checked_mask(torch.tensor([1]),'0'*64)
 def test_probe_is_one_w1_and_shared(self):
  from unittest.mock import patch
  import chipmunk_signed_complement_math as m
  x=torch.zeros((64,2),dtype=torch.bfloat16);w=torch.eye(2,dtype=torch.bfloat16);b=torch.zeros(2,dtype=torch.bfloat16);h=torch.ones((64,2),dtype=torch.bfloat16)
  original=m.F.linear
  with patch.object(m.F,'linear',wraps=original) as linear:
   idx,mean=p64_observer(x,w,b,h,torch.arange(64),1)
  self.assertEqual(linear.call_count,1);self.assertEqual(idx.tolist(),[0]);self.assertTrue(torch.equal(mean,torch.tensor([-1.,-1.])))
if __name__=='__main__':unittest.main()
