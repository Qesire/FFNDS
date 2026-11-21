"""Prepared BF16 joint FFN hit, preserving the native observer and the FP64 GLS."""
import math

import torch
import torch.nn.functional as F

from native_joint import prep_hit
from fused_ops import compile_for, apply


def prepare_static(phase,pack,device):
    device=torch.device(device)
    tensors=prep_hit(phase,pack,device,pack["A0"].shape[0],64)
    tensors["vd"]=tensors["v"].double()
    tensors["vv"]=tensors["vd"]@tensors["vd"].T
    tensors["kernels"]=compile_for(tensors["A0"].device.index)
    return tensors


def hit_prepared(u,phase,cached,prepared,probes,inv_metric,tile_rows=2048,*,audit=False):
    t=prepared
    k,rank=t["zmap"].shape
    n,d=u.shape
    assert tile_rows>0 and u.is_contiguous() and all(x.is_contiguous() for x in cached.values())
    kernels=t["kernels"]

    def factors(now,u0,z0):
        du=torch.empty_like(now,dtype=torch.float32)
        dub=torch.empty_like(now)
        apply(kernels["difference"],du.numel(),now,u0,dub,du,du.numel())
        linear=dub@t["A0"]
        q=du@t["jmap"]
        del du,dub
        z1=F.linear(now,t["w1k"],t["b1k"])
        h0=F.gelu(z0,approximate="tanh")
        h1=F.gelu(z1,approximate="tanh")
        dh=torch.empty_like(z1)
        dh32=torch.empty_like(z1,dtype=torch.float32)
        x=torch.empty_like(dh32)
        apply(kernels["secant"],z1.numel(),z1,z0,h1,h0,t["kernel_beta"],t["kernel_scale"],
              dh,dh32,x,z1.numel(),k)
        q=(dh32@t["zmap"]).sub_(q)
        selected=torch.mm(dh,t["w2k"],out_dtype=torch.float32)
        return linear,selected,q,x

    paid=u.index_select(0,probes)
    linear,sel,qp,xp=factors(paid,cached["u0"][probes],cached["z0k"][probes])
    observed=phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(paid),approximate="tanh"))
    residual=(observed.float()-((cached["y0"][probes].float()+linear.float())+sel)).double()
    v,qd=t["vd"],qp.double()
    matrix=((xp@xp.T)/k).double()+inv_metric
    chol,status=torch.linalg.cholesky_ex(matrix)
    # Broadcast errors to the current stream. These assertions also fire if bad
    # numerical input reaches an otherwise well-shaped unobserved direction.
    torch._assert_async((status==0).all(),"Current probe metric is not SPD")
    siq=torch.cholesky_solve(qd,chol)
    gram=(qd.T@siq)*t["vv"]
    gains,info=torch.linalg.solve_ex(gram,(siq*(residual@v.T)).sum(0))
    torch._assert_async((info==0).all(),"Current probe response is singular")
    coeff=torch.cholesky_solve(residual-(qd*gains)@v,chol)
    torch._assert_async(torch.isfinite(coeff).all()&torch.isfinite(gains).all(),"Non-finite GLS")
    readout=torch.cat((gains.float()[:,None]*t["v"],0.5*coeff.float()))
    out=torch.empty_like(u)
    for start in range(0,n,tile_rows):
        sl=slice(start,start+tile_rows)
        linear,selected,q,x=factors(u[sl],cached["u0"][sl],cached["z0k"][sl])
        cross=(x@xp.T).div_(k)
        correction=torch.cat((q,cross),dim=1)@readout
        dst=out[sl]
        apply(kernels["writeback"],dst.numel(),cached["y0"][sl],linear,selected,correction,dst,dst.numel())
        del linear,selected,correction,q,x,cross
    out.index_copy_(0,probes,observed)
    stats=dict(probe_calls=1,p=len(probes),k=k,rank=rank,selected_gemm="bf16_to_fp32",
               tile_rows=tile_rows,full_readout_writes=1,tiles=math.ceil(n/tile_rows))
    if audit:
        stats.update(probe_exact=torch.equal(out[probes],observed),gain_min=float(gains.min()),gain_max=float(gains.max()))
    return out,stats
