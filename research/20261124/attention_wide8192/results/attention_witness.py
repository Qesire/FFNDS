"""Full-key FP64 self finite secants and fixed-context cross directional replay."""
import math
from pathlib import Path
import sys

import torch

from attention_panel_fit import Assets, digest, save


HEADS = (0,13,26,39)
SCALE = 128**-0.5


def score(q,k):
    return (q @ k.transpose(-2,-1))*SCALE


def secant(q0,q1,k0,k1,v0,v1):
    s0,s1=score(q0,k0),score(q1,k1)
    log0,log1=(x.log_softmax(-1) for x in (s0,s1))
    p0,p1=log0.exp(),log1.exp()
    diff=(log1-log0).abs()
    ratio=torch.where(diff==0,1,-(-diff).expm1()/diff)
    ell=log0.maximum(log1).exp()*ratio
    vavg=(v0+v1)*0.5
    q_delta,q_mid=q1-q0,(q0+q1)*0.5
    k_delta,k_mid=k1-k0,(k0+k1)*0.5
    value=(p0+p1)*0.5 @ (v1-v0)
    def action(weight,dscore,v):
        r=(weight*dscore).sum(-1,keepdim=True)/weight.sum(-1,keepdim=True)
        return (weight*(dscore-r)) @ v
    query=action(ell,score(q_delta,k_mid),vavg)
    key=action(ell,score(q_mid,k_delta),vavg)
    direct=p1@v1-p0@v0
    first=p0@(v1-v0)+action(p0,score(q_delta,k0)+score(q0,k_delta),v0)
    return direct,first,torch.stack((value,query,key)),ell.sum(-1)


def self_witness(site,assets):
    row=site.row
    info=site.index_row["witness"]
    path=site.root/info["path"]
    assert digest(path)==info["sha256"]
    d=assets.load(str(path),device=assets.device)
    light=assets.load(str(site.root/row["tensor_file"]),device=assets.device)
    q=light["q_effective"][:,assets.h][:,:,HEADS].permute(0,2,1,3).double()
    k,v=(d[name].permute(0,2,1,3).double() for name in ("K","V"))
    gram=torch.zeros((len(HEADS),3,3),device=assets.device,dtype=torch.float64)
    max_err,eta_min,eta_max = 0,1,0
    accumulators = []
    for start in range(0,q.shape[2],16):
        block=q[:,:,start:start+16]
        exact,tangent,channels,eta=secant(*block,*k,*v)
        channel_error=exact-channels.sum(0)
        max_err=max(max_err,float(channel_error.abs().max()))
        eta_min=min(eta_min,float(eta.min()))
        eta_max=max(eta_max,float(eta.max()))
        gram += torch.einsum("chqd,ehqd->hce",channels,channels)
        accumulators.append((exact,tangent,channel_error))
    exact,first,err=[torch.cat(xs,dim=1) for xs in zip(*accumulators)]
    assert max_err<5e-10 and 0<eta_min<=eta_max<=1+1e-10
    # Keep the visual subspace of the four observed heads, before other heads contribute.
    channels = torch.cat([torch.arange(h*128,(h+1)*128,device=assets.device) for h in HEADS])
    w=assets.weight(site)[channels].double()
    gate=light["self_gate_s"].double()
    def E(x):
        return float(x.square().sum())
    def projected(x):
        return (x.transpose(0,1).reshape(len(assets.h),len(HEADS)*128)@w)*gate
    return dict(kind="self",heads=HEADS,raw_witness_sha256=info["sha256"],query_role="H448",
        channels=["value","query","key"],signed_gram_by_head=gram.tolist(),eta_range=[eta_min,eta_max],
        identity_max_abs=max_err,identity_relative=math.sqrt(E(err)/E(exact)),
        tangent_relative=math.sqrt(E(exact-first)/E(exact)),
        tangent_relative_after_WO_gate_four_head_contribution=math.sqrt(E(projected(exact-first))/E(projected(exact))),
        finite_energy=E(exact),first_order_energy=E(first),remainder_energy=E(exact-first))


def fixed_kv_direction(q,k,v,dq):
    s=score(q,k)
    ds=score(dq,k)
    p=s.softmax(-1)
    o=p@v
    centered=ds-(p*ds).sum(-1,keepdim=True)
    first=(p*centered)@v
    term=p*centered.square()
    second=term@v-term.sum(-1,keepdim=True)*o
    result={}
    for scale in (0.25,0.5,1.0):
        exact=(s+scale*ds).softmax(-1)@v-o
        result[scale]=(exact.square().sum(),
            (exact-scale*first).square().sum(),
            (exact-scale*first-0.5*scale**2*second).square().sum())
    # Explicit covariance/JVP equality with an alternative weighted reduction.
    direct=(p*ds)@v-(p*ds).sum(-1,keepdim=True)*o
    assert bool(torch.allclose(first,direct,atol=2e-10,rtol=1e-8))
    return result


def cross_witness(site,assets):
    row=site.row
    kv0,kv1=site.index_row["cross_KV"]
    assert kv0["sha256"]==kv1["sha256"] and kv0["path"]==kv1["path"]
    path=site.root/kv0["path"]
    assert digest(path)==kv0["sha256"]
    d=assets.load(str(path),device=assets.device)
    light=assets.load(str(site.root/row["tensor_file"]),device=assets.device)
    q0,q1=light["q_effective"][:,assets.h].permute(0,2,1,3).double()
    k,v=(d[n].transpose(0,1).double() for n in ("K","V"))
    traces={}
    for name,q,delta in (("anchor_toward_hit",q0,q1-q0),("hit_toward_anchor",q1,q0-q1)):
        accum={scale:torch.zeros(3,device=assets.device,dtype=torch.float64) for scale in (0.25,0.5,1.0)}
        for q_rows,d_rows in zip(q.split(32,dim=1),delta.split(32,dim=1)):
            for scale,stats in fixed_kv_direction(q_rows,k,v,d_rows).items():
                accum[scale]+=torch.stack(stats)
        traces[name]={str(scale):dict(delta_energy=a,tangent_relative=math.sqrt(b/a),
            second_order_relative=math.sqrt(c/a)) for scale,(a,b,c) in ((k,v.tolist()) for k,v in accum.items())}
    return dict(kind="text_cross",heads=list(range(40)),query_role="H448",context_sha256=kv0["sha256"],directions=traces)


@torch.inference_mode()
def main(panel_root,frozen,out):
    a=Assets(panel_root,frozen,"cuda:0")
    result=[]
    for (module,expert),ss in sorted(a.groups.items()):
        for site in ss:
            if module=="self" and "witness" not in site.index_row:
                continue
            value=self_witness(site,a) if module=="self" else cross_witness(site,a)
            idx={k:site.row[k] for k in ("source_id","source_cluster_id","module","expert","background","layer")}
            result.append(dict(**idx,**value))
        print("WITNESS",module,expert,len(result),flush=True)
    assert sum(r["kind"]=="self" for r in result)==8 and len(result)==104
    save(out,dict(status="complete",source_results_sha256=digest(panel_root/"RESULTS.sha256"),
        script_sha256=digest(Path(__file__)),records=result))
    print("WITNESS_COMPLETE",flush=True)


if __name__ == "__main__":
    main(*(Path(x).resolve() for x in sys.argv[1:]))
