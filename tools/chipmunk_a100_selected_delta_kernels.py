"""A100 BF16 selected-delta kernels; independent controlled port, not Chipmunk Hopper code.

One group only. Inputs and weights are BF16 contiguous. Selection is unique and
in-range, checked by the caller. Delta is rounded to BF16 before the W2 dot.
Each output element has one writer. Empty selection causes no dispatch.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def selected_w1_gelu_delta(X, W1, B1, HCache, Indices, Delta,
                           M:tl.constexpr, D:tl.constexpr, H:tl.constexpr, S:tl.constexpr,
                           BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    ind=tl.load(Indices+cols,mask=cols<S,other=0)
    kk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(D,BK)):
        k=start*BK+kk
        x=tl.load(X+rows[:,None]*D+k[None,:],mask=(rows[:,None]<M)&(k[None,:]<D),other=0)
        w=tl.load(W1+ind[None,:]*D+k[:,None],mask=(cols[None,:]<S)&(k[:,None]<D),other=0)
        acc=tl.dot(x,w,acc)
    bias=tl.load(B1+ind,mask=cols<S,other=0).to(tl.float32)
    # Native torch Linear writes BF16 before the separate GELU operator.
    pre=(acc+bias[None,:]).to(tl.bfloat16).to(tl.float32)
    a=0.7978845608028654*(pre+0.044715*pre*pre*pre)
    new=(0.5*pre*(1.0+libdevice.tanh(a))).to(tl.bfloat16)
    old=tl.load(HCache+rows[:,None]*H+ind[None,:],mask=(rows[:,None]<M)&(cols[None,:]<S),other=0)
    delta=(new.to(tl.float32)-old.to(tl.float32)).to(tl.bfloat16)
    tl.store(Delta+rows[:,None]*S+cols[None,:],delta,mask=(rows[:,None]<M)&(cols[None,:]<S))
    tl.store(HCache+rows[:,None]*H+ind[None,:],new,mask=(rows[:,None]<M)&(cols[None,:]<S))

@triton.jit
def selected_w2_add_cache(Delta, W2, Indices, YCache,
                          M:tl.constexpr, D:tl.constexpr, H:tl.constexpr, S:tl.constexpr,
                          BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    out=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(S,BK)):
        k=start*BK+kk
        ind=tl.load(Indices+k,mask=k<S,other=0)
        delta=tl.load(Delta+rows[:,None]*S+k[None,:],mask=(rows[:,None]<M)&(k[None,:]<S),other=0)
        w=tl.load(W2+out[None,:]*H+ind[:,None],mask=(out[None,:]<D)&(k[:,None]<S),other=0)
        acc=tl.dot(delta,w,acc)
    old=tl.load(YCache+rows[:,None]*D+out[None,:],mask=(rows[:,None]<M)&(out[None,:]<D),other=0).to(tl.float32)
    tl.store(YCache+rows[:,None]*D+out[None,:],(old+acc).to(tl.bfloat16),mask=(rows[:,None]<M)&(out[None,:]<D))
