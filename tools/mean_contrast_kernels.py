"""Fixed auxiliary kernels for an exact-mean contrast representation.

Synthetic backend only. Original Q1/Q2 and all GEMMs remain external.
Explicit BF16 boundaries define the implemented operator; they do not promise
exact real mean preservation after rounded reconstruction.
"""
import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _prepare(X, H3, MU, D: tl.constexpr, B: tl.constexpr):
    group, chunk = tl.program_id(0), tl.program_id(1)
    d = chunk*B+tl.arange(0,B)
    a = tl.load(X+(4*group+0)*D+d,d<D,0).to(tl.float32)
    b = tl.load(X+(4*group+1)*D+d,d<D,0).to(tl.float32)
    c = tl.load(X+(4*group+2)*D+d,d<D,0).to(tl.float32)
    e = tl.load(X+(4*group+3)*D+d,d<D,0).to(tl.float32)
    tl.store(MU+group*D+d,(((a+b)+(c+e))*.25).to(tl.bfloat16),d<D)
    tl.store(H3+(3*group+0)*D+d,((a+b-c-e)*.5).to(tl.bfloat16),d<D)
    tl.store(H3+(3*group+1)*D+d,((a-b+c-e)*.5).to(tl.bfloat16),d<D)
    tl.store(H3+(3*group+2)*D+d,((a-b-c+e)*.5).to(tl.bfloat16),d<D)


@triton.jit
def _reconstruct(MEAN, CH3, Z, H: tl.constexpr, B: tl.constexpr):
    group,chunk=tl.program_id(0),tl.program_id(1)
    h=chunk*B+tl.arange(0,B)
    m=tl.load(MEAN+group*H+h,h<H,0).to(tl.float32)
    a=tl.load(CH3+(3*group+0)*H+h,h<H,0).to(tl.float32)
    b=tl.load(CH3+(3*group+1)*H+h,h<H,0).to(tl.float32)
    c=tl.load(CH3+(3*group+2)*H+h,h<H,0).to(tl.float32)
    tl.store(Z+(4*group+0)*H+h,(m+(a+b+c)*.5).to(tl.bfloat16),h<H)
    tl.store(Z+(4*group+1)*H+h,(m+(a-b-c)*.5).to(tl.bfloat16),h<H)
    tl.store(Z+(4*group+2)*H+h,(m+(-a+b-c)*.5).to(tl.bfloat16),h<H)
    tl.store(Z+(4*group+3)*H+h,(m+(-a-b+c)*.5).to(tl.bfloat16),h<H)


@triton.jit
def _pool(V2, OUT, H: tl.constexpr, B: tl.constexpr):
    group,chunk=tl.program_id(0),tl.program_id(1)
    h=chunk*B+tl.arange(0,B)
    a=tl.load(V2+(4*group+0)*H+h,h<H,0).to(tl.float32)
    b=tl.load(V2+(4*group+1)*H+h,h<H,0).to(tl.float32)
    c=tl.load(V2+(4*group+2)*H+h,h<H,0).to(tl.float32)
    e=tl.load(V2+(4*group+3)*H+h,h<H,0).to(tl.float32)
    tl.store(OUT+group*H+h,(((a+b)+(c+e))*.25).to(tl.bfloat16),h<H)


def prepare(v1):
    m,d=v1.shape
    hh=torch.empty((m//4*3,d),dtype=v1.dtype,device=v1.device)
    mu=torch.empty((m//4,d),dtype=v1.dtype,device=v1.device)
    _prepare[(m//4,triton.cdiv(d,BLOCK))](v1,hh,mu,d,BLOCK,num_warps=4,enable_fp_fusion=False)
    return hh,mu


def reconstruct(mean,contrasts):
    groups,h=mean.shape
    z=torch.empty((groups*4,h),dtype=mean.dtype,device=mean.device)
    _reconstruct[(groups,triton.cdiv(h,BLOCK))](mean,contrasts,z,h,BLOCK,num_warps=4,enable_fp_fusion=False)
    return z


def pool(v2):
    m,h=v2.shape
    out=torch.empty((m//4,h),dtype=v2.dtype,device=v2.device)
    _pool[(m//4,triton.cdiv(h,BLOCK))](v2,out,h,BLOCK,num_warps=4,enable_fp_fusion=False)
    return out
