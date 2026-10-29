"""Causal P64 omitted-channel signed mean, evaluated on real full-M inputs."""
import hashlib,time
import torch
import torch.nn.functional as F
from chipmunk_fullm_proxy_math import selection

def p64_observer(x,w1,b1,old_h,permutation,k):
    rows=permutation[:64].to(device=x.device)
    now=F.gelu(F.linear(x.index_select(0,rows),w1,b1),approximate='tanh')
    delta=now.float()-old_h.index_select(0,rows).float()
    rms=delta.square().mean(0).sqrt()
    return selection(rms,k),delta.mean(0)

def checked_mask(mask,expected):
    t=time.perf_counter();actual=hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()
    if actual!=expected:raise ValueError('P64 mask identity')
    return time.perf_counter()-t

def packed(x,w1,b1,w2,old_h,old_y,gate,mask,mean_delta=None,*,mm_fp32=None):
    if mm_fp32 is None:mm_fp32=lambda a,b:torch.mm(a,b,out_dtype=torch.float32)
    hc=old_h.clone();yc=old_y.clone();w1s=w1.index_select(0,mask);b1s=b1.index_select(0,mask);w2s=w2.index_select(1,mask).contiguous();hprev=hc.index_select(1,mask)
    new=F.gelu(F.linear(x,w1s,b1s),approximate='tanh');delta=(new.float()-hprev.float()).to(torch.bfloat16);hc.index_copy_(1,mask,new)
    fp32=yc.float()+mm_fp32(delta,w2s.T)
    if mean_delta is not None:
        complement=mean_delta.clone();complement.index_fill_(0,mask,0);correction=mm_fp32(complement.to(torch.bfloat16).reshape(1,-1),w2.T);fp32=fp32+correction
    y=fp32.to(torch.bfloat16);return y,(y*gate).to(torch.bfloat16)

def split_dc_ac(error):
    e=error.double();mu=e.mean(0);dc=e.shape[0]*mu.square().sum();ac=(e-mu).square().sum();total=e.square().sum();return {'dc_energy':float(dc),'ac_energy':float(ac),'total_energy':float(total),'closure_difference':float(dc+ac-total)}
