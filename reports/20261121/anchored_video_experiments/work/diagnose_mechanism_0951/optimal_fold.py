"""Fit a missing static secant mean on three disjoint CAL sources and fold it into existing weights."""
from optimal_panel import ROOT, OUT, WORK, budget, features, fixed, readout, transform_edges
import json
import numpy as np
import torch


def fit(moments):
    gram=sum(g for g,_ in moments)
    gram.diagonal().add_(1.)
    return torch.cholesky_solve(sum(y for _,y in moments),torch.linalg.cholesky(gram))


def run(expert):
    rows,model,_=budget.setup(ROOT,expert)
    selected=model["sel"]
    pairs=[features(ROOT,rows[s],model) for s in fixed.CAL]
    beta=(sum((a.double()*b).mean(0) for a,b in pairs)/sum(a.double().square().mean(0) for a,b in pairs).clamp_min(1e-30)).float()
    scale=torch.stack([(b-a*beta).square().mean(0) for a,b in pairs]).mean(0).sqrt().clamp_min(1e-8)
    moments=[]
    permuted_moments=[]
    for source in fixed.CAL:
        ids,coords,b,q,y0,delta,_=budget.panel(ROOT,rows[source],model)
        x=features(ROOT,rows[source],model,beta,scale)/len(scale)**.5
        graph=budget.edge_indices(torch.arange(len(ids)).numpy(),coords,ids)
        X=transform_edges(x,graph)*len(ids)**.5
        residual=delta+y0-b-q@model["v"]
        R=transform_edges(residual,graph)*len(ids)**.5
        moments.append((X.T@X,X.T@R))
        # Negative control preserves the source and latent-frame labels of all targets.
        rng=np.random.default_rng(7215)
        order=np.arange(len(ids))
        for t in np.unique(coords[:,0]):
            ind=np.flatnonzero(coords[:,0]==t)
            order[ind]=rng.permutation(ind)
        permuted=transform_edges(residual[order],graph)*len(ids)**.5
        permuted_moments.append((moments[-1][0],X.T@permuted))
    cross=norm=0.
    for i in range(len(moments)):
        w=fit(moments[:i]+moments[i+1:])
        gram,xy=moments[i]
        cross+=(w.double()*xy).sum().item()
        norm+=(w.double()*(gram@w)).sum().item()
    alpha=max(0.,min(1.,cross/norm))
    w=alpha*fit(moments)
    delta_w=w/len(scale)**.5/scale[:,None]
    folded=dict(model)
    folded["A0"]=(model["A0"].float()-(model["w1"][selected].float().T*beta)@delta_w).bfloat16()
    folded["b"]=(model["b"]+delta_w).bfloat16().float()
    gram=sum(g for g,_ in moments)
    wp=fit(permuted_moments)
    amp=torch.sqrt((w.double()*(gram@w)).sum()/(wp.double()*(gram@wp)).sum()).item()
    delta_p=amp*wp/len(scale)**.5/scale[:,None]
    sham=dict(model,A0=(model["A0"].float()-(model["w1"][selected].float().T*beta)@delta_p).bfloat16(),
              b=(model["b"]+delta_p).bfloat16().float())
    baseline=json.loads((WORK/"conditional1120"/(expert+".json")).read_bytes())
    output=[]
    for src in sorted({r["source"] for r in baseline}):
        ids,coords,b,q,y0,delta,observe=budget.panel(ROOT,rows[src],model)
        _,_,b_fold,*_=budget.panel(ROOT,rows[src],folded)
        _,_,b_sham,*_=budget.panel(ROOT,rows[src],sham)
        x=features(ROOT,rows[src],model,beta,scale)
        b_ideal=b+(x/len(scale)**.5)@w
        k=x@x.T/x.shape[1]
        gate=fixed.panel(ROOT,rows[src]["hit"])["gate"].reshape(-1).cuda().float()
        for fold,(pool,h) in enumerate(budget.split_panel(ids)):
            p=budget.layout(pool,ids,coords,1120,"uniform")[:64]
            ep=budget.edge_indices(p,coords,ids)
            eh=budget.edge_indices(h,coords,ids)
            eye=torch.eye(len(p),dtype=torch.float64,device=q.device)
            d=eye[ep[:,1]]-eye[ep[:,0]]
            inv=torch.cholesky_inverse(torch.linalg.cholesky(eye+len(p)/len(ep)*(d.T@d)))
            state=budget.eval_cache(delta[h],eh,gate)
            obs=observe(p)
            target=delta[h]+y0[h]
            r=dict(expert=expert,source=src,fold=fold)
            for name,base in (("global",b),("ideal",b_ideal),("folded",b_fold),("sham",b_sham)):
                mean,c,_=readout(k,p,h,inv,obs,base,q,model["v"])
                pred=(mean+.5*c).bfloat16().float()
                met=budget.metrics(pred-y0[h],state)
                if name=="global":
                    old=next(v["gls_half"] for v in baseline if v["source"]==src and v["fold"]==fold)
                    assert abs(met["J"]/old["J"]-1)<5e-5
                r[name]=dict(metrics=met,gated_J=transform_edges((pred-target)*gate,eh).square().sum().item())
            # Nested lower-cost arm on the very same 256 query rows; reproduce native
            # 32-row GEMM labels, since its BF16 output can depend on the batch shape.
            p32=p[:32]
            ep32=budget.edge_indices(p32,coords,ids)
            eye32=eye[:32,:32].contiguous()
            d32=eye32[ep32[:,1]]-eye32[ep32[:,0]]
            inv32=torch.cholesky_inverse(torch.linalg.cholesky(eye32+len(p32)/len(ep32)*(d32.T@d32)))
            obs32=observe(p32)
            for name,base in (("global32",b),("folded32",b_fold)):
                mean,c,_=readout(k,p32,h,inv32,obs32,base,q,model["v"])
                pred=(mean+.5*c).bfloat16().float()
                r[name]=dict(metrics=budget.metrics(pred-y0[h],state),
                             gated_J=transform_edges((pred-target)*gate,eh).square().sum().item())
            r["folded_base_error_rel_mean_update"]=((b_fold[h]-b_ideal[h]).norm() /
                                                      (b_ideal[h]-b[h]).norm().clamp_min(1e-10)).item()
            output.append(r)
        print(expert,src,flush=True)
    file=OUT/("静态剩余折叠_"+expert+".json")
    file.write_text(json.dumps(dict(alpha=alpha,alpha_unclipped=cross/norm,perm_seed=7215,wrong_pair_scale=amp,cal=fixed.CAL,
                      shrink_scope="leave one CAL source out for the extra map, base and standardization pretrained on all three CAL",
                      rows=output),indent=1,allow_nan=False),encoding="utf8")


if __name__=="__main__":
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(3)
    with torch.inference_mode():
        for e in ("high","low"):run(e)
