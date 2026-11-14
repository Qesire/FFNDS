"""Same packed BF16 numerical path as the measured local helper, before gate.

Caller predebits whole action and synchronizes before marking completion.
"""
import torch
import torch.nn.functional as F

def packed_pregate(x,w1,b1,w2,old_h,old_y,mask,*,mm_fp32=None):
 if mm_fp32 is None:mm_fp32=lambda a,b:torch.mm(a,b,out_dtype=torch.float32)
 hc=old_h.clone();yc=old_y.clone()
 packed_w1=w1.index_select(0,mask);packed_b1=b1.index_select(0,mask);packed_w2=w2.index_select(1,mask).contiguous()
 old_selected=hc.index_select(1,mask)
 pre=F.linear(x,packed_w1,packed_b1);new=F.gelu(pre,approximate='tanh');delta=(new.float()-old_selected.float()).to(torch.bfloat16)
 hc.index_copy_(1,mask,new)
 fp32=mm_fp32(delta,packed_w2.transpose(0,1));return (yc.float()+fp32).to(torch.bfloat16)
