"""Paired, nested observer-budget study on the same held-out pre-gate targets."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / "joint-readout-1120"))
import online
import run as fixed

BUDGETS = (0, 8, 16, 32, 64, 96, 128, 256)
SEEDS = (1120, 3041, 4919, 5671)
LAYOUTS = ("uniform", "near_pair")


def split_panel(ids):
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f"ffnds-budget-1120-v1|{ids[i]}".encode()).digest())
    a, b = np.asarray(order[:256]), np.asarray(order[256:])
    assert len(b)==len(a)==256 and not(set(a)&set(b))
    return ((a,b),(b,a))


def layout(pool, ids, coords, seed, kind):
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(pool)
    if kind == "uniform":
        return ordered
    remaining = set(pool.tolist())
    selected = []
    for row in ordered:
        if row not in remaining:
            continue
        remaining.remove(row)
        selected.append(row)
        same = np.array([x for x in remaining if coords[row,0]==coords[x,0]],dtype=int)
        if len(same):
            d=((coords[same,1:]-coords[row,1:])**2).sum(1)
            other=same[np.lexsort((ids[same],d))[0]]
            selected.append(other)
            remaining.remove(other)
    assert set(selected)==set(pool) and len(selected)==len(pool)
    return np.asarray(selected)


def edge_indices(rows,coords,ids):
    pairs=online.probe_edges(rows,coords,ids)
    if not len(pairs):
        return np.empty((0,2),dtype=int)
    i,j=pairs.T.copy()
    dy,dx=(coords[j,1:]-coords[i,1:]).T
    flip=np.where(abs(dx)>=abs(dy),dx,dy)<0
    i[flip],j[flip]=j[flip],i[flip].copy()
    lookup={j:k for k,j in enumerate(rows)}
    return np.array([(lookup[l],lookup[r]) for l,r in zip(i,j)])


def eval_cache(t, edges, gate=None):
    i,j=edges.T
    tc,td=(t[i]+t[j])/2**.5,(t[j]-t[i])/2**.5
    tv=torch.stack([tc.flatten(),td.flatten()])
    tn=tv.norm(dim=1)
    norm=torch.outer(tn,tn)
    gram=tv@tv.T/norm
    cache = dict(t=t,tv=tv,tc=tc,td=td,norm=norm,gram=gram,iv=gram.inverse(),
                 edges=(i,j),tn2=td.square().sum(1))
    if gate is not None:
        cache.update(gate=gate, gated_td=td*gate)
    return cache


def metrics(pred,cache):
    e=pred-cache["t"]
    i,j=cache["edges"]
    ec,ed=(e[i]+e[j])/2**.5,(e[j]-e[i])/2**.5
    ev=torch.stack((ec.flatten(),ed.flatten()))
    te=cache["tv"]@ev.T/cache["norm"]
    ee=ev@ev.T/cache["norm"]
    coeff=cache["iv"]@te
    perp=ee-te.T@coeff
    p=e.square().sum(1).mean()
    edge=2*ed.square().sum(1).mean()
    t=cache["td"]; yh=t+ed
    cs=F.cosine_similarity(t,yh,dim=1)
    scal=(yh*t).sum(1)/cache["tn2"]
    out=torch.stack((p+edge,p,edge,torch.quantile(cs.clamp(-1,1).acos().rad2deg(),.5),
                     torch.quantile(scal,.5),perp[1,1],coeff[0,0]+1,coeff[1,1]+1,ee[0,1]))
    names=("J","point","edge","angle","signed_edge_scale","orthogonal_difference",
           "common_gain","difference_gain","common_difference_error")
    result=dict(zip(names,out.tolist()))
    identity=torch.eye(2,device=coeff.device)
    result.update(joint_scale=(identity+coeff).tolist(),signed_target_error=te.tolist(),
                  error_gram=ee.tolist(),perpendicular_joint_energy=perp.tolist())
    if "gate" in cache:
        gt,gy=cache["gated_td"],yh*cache["gate"]
        dot=(gt*gy).sum(1)
        nt,ny=gt.norm(dim=1),gy.norm(dim=1)
        assert nt.min()>1e-8 and ny.min()>1e-8
        gc=(dot/(nt*ny)).clamp(-1,1)
        gs=dot/nt.square()
        result.update(gated_angle=torch.quantile(gc.acos().rad2deg(),.5).item(),
                      gated_signed_scale=torch.quantile(gs,.5).item(),
                      gated_amplitude=torch.quantile(ny/nt,.5).item(),
                      gated_negative=int((gs<0).sum()),
                      gated_relative=(gy-gt).norm().div(gt.norm()).item())
    return result


def setup(root, expert):
    mf=root/"BINDINGS.json"
    assert fixed.digest(mf)=="8f7320c61dbafd44df0b1662235405de8961dd5c5d52d846c979fdb4179b1ab6"
    manifest=json.loads(mf.read_bytes())
    rows={r["source"]:r for r in manifest["pairs"] if r["expert"]==expert and r["layer"]==19}
    ws=fixed.load_weights(root,expert,manifest["sites"][expert+"19"])
    w1,b1,w2,b2=ws
    num=torch.zeros(len(w1),device="cuda",dtype=torch.float64);den=torch.zeros_like(num)
    for name in fixed.CAL:
        dz,dh,_,_=fixed.operands(root,rows[name],ws)
        num+=(dh.double()*dz).mean(0);den+=dz.double().square().mean(0)
    beta=(num/den.clamp_min(1e-30)).float()
    receipt=json.loads((WORK/"compiled"/(expert+".json")).read_bytes())
    sel=torch.tensor(receipt["selected_ids"],device="cuda")
    assert fixed.thash(sel)==receipt["selected_sha256"] and fixed.thash(beta)==receipt["beta_sha256"]
    cal=[fixed.data(root,rows[n],ws,beta,sel) for n in fixed.CAL]
    scale=torch.stack([p["z"][p["q"]].square().mean(0) for p in cal]).mean(0).sqrt().clamp_min(1e-8)
    left,right,fit=fixed.fit(cal,scale,gamma=1)
    del cal
    sigma=right.norm(dim=1)
    v=right/sigma[:,None]
    zmap=left*sigma
    om=beta.clone(); om[sel]=0
    mat=dict(v=v,zmap=zmap,jmap=(w1[sel].T*beta[sel])@zmap,A0=((w1.T*om)@w2.T).bfloat16(),
             b=w2[:,sel].T.contiguous(),sel=sel,w1=w1.bfloat16(),b1=b1.bfloat16(),
             w2=w2.bfloat16(),b2=b2.bfloat16())
    return rows,mat,dict(fit=fit,selection_sha256=receipt["selected_sha256"])


def panel(root,row,m):
    a,h=[fixed.panel(root,row[k]) for k in ("anchor","hit")]
    assert all(torch.equal(a[k],h[k]) for k in ("token_indices","token_thw"))
    u0,u1=a["U"].cuda(),h["U"].cuda()
    y0=a["Y_pre_gate"].cuda().float()
    def hidden(x,w,b):
        return F.gelu(F.linear(x,w,b),approximate="tanh")
    s=m["sel"]
    h0=hidden(u0,m["w1"],m["b1"])[:,s]
    hk=hidden(u1,m["w1"][s],m["b1"][s])
    dh=(hk.float()-h0.float()).bfloat16().float()
    du=u1.float()-u0.float()
    base=y0+(du.bfloat16()@m["A0"]).float()+dh@m["b"]
    Q=dh@m["zmap"]-du@m["jmap"]
    def observe(indices):
        return F.linear(hidden(u1[indices],m["w1"],m["b1"]),m["w2"],m["b2"]).float()
    truth=h["Y_pre_gate"].cuda().float()-y0
    return h["token_indices"].numpy(),h["token_thw"].numpy(),base,Q,y0,truth,observe


def run(expert):
    root=WORK/"native-inputs"
    rows,m,config=setup(root,expert)
    out=dict(config=config,expert=expert,cal=fixed.CAL,dev=fixed.DEV,budgets=BUDGETS,seeds=SEEDS,
             layouts=LAYOUTS,rank=16,results=[],folds=[])
    for name in fixed.DEV:
        ids,coords,base,Q,y0,truth,observe=panel(root,rows[name],m)
        v=m["v"]
        for fold,(pool,test) in enumerate(split_panel(ids)):
            # Score always uses the same heldout rows and their own graph.
            assert not set(pool).intersection(test)
            et=edge_indices(test,coords,ids)
            gate=fixed.panel(root,rows[name]["hit"])["gate"].view(-1).float().cuda()
            state=eval_cache(truth[test],et,gate)
            def score(g):
                y=(base[test]+(Q[test]*g.float())@v).bfloat16().float()-y0[test]
                return metrics(y,state)
            ones=torch.ones(16,device="cuda",dtype=torch.float64)
            prior=score(ones)
            raw_prior=(base[test]+Q[test]@v).bfloat16().float()-y0[test]
            i,j=et.T
            check=fixed.metrics(raw_prior,truth[test],np.arange(len(test)),i,j)
            assert abs(prior["J"]/check["objective"]-1)<1e-6
            p=pool
            full_proj=(observe(p)-base[p]).double()@v.double().T
            p_edges=edge_indices(p,coords,ids)
            gold,_=online.coefficients(Q[p].double(),full_proj,p_edges,True)
            # Upper-bound diagnostic for this frozen basis: fit the diagonal readout on query labels.
            target_projection=(truth[test]+y0[test]-base[test]).double()@v.double().T
            half_oracle,_=online.coefficients(Q[test].double(),target_projection,et,True)
            oracle=score(2*half_oracle-1)
            out["folds"].append(dict(source=name,fold=fold,pool=ids[pool].tolist(),test=ids[test].tolist(),
                                     test_edges=len(et),prior=prior,query_fitted=oracle))
            for seed in SEEDS:
                for kind in LAYOUTS:
                    order=layout(pool,ids,coords,seed,kind)
                    for n in BUDGETS[1:]:
                        p=order[:n]
                        if n==256:
                            # At full capacity all orderings select the same rows. Canonicalize the GEMM.
                            p=pool
                        ed=edge_indices(p,coords,ids)
                        proj=(observe(p)-base[p]).double()@v.double().T if n!=256 else full_proj
                        q=Q[p].double()
                        half,h=online.coefficients(q,proj,ed,True)
                        assert h.min()>0
                        a=q.square()/n
                        if len(ed):
                            ii,jj=ed.T
                            a=torch.cat((a,(q[jj]-q[ii]).square()/len(ed)))
                        ess=h.square()/a.square().sum(0)
                        rec=dict(source=name,fold=fold,layout=kind,seed=seed,p=n,edges=len(ed),
                                 frames=int(len(set(coords[p,0]))),support_min=h.min().item(),
                                 support_max=h.max().item(),support_ess_min=ess.min().item(),
                                 support_ess_median=ess.median().item(),
                                 half_gain_rms_to_pool256=(half-gold).square().mean().sqrt().item(),
                                 half_gain=half.tolist(),obs_rows_sha256=hashlib.sha256(ids[p].tobytes()).hexdigest(),
                                 half=score(half))
                        shrunk=ones+(2*half-2)*n/(n+64)
                        rec["support_scaled"]=rec["half"] if n==64 else score(shrunk)
                        rec["full_trust"]=score(2*half-1)
                        out["results"].append(rec)
            print(expert,name,fold,len(et),prior["J"],flush=True)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("expert",choices=("high","low"))
    args=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4)
    with torch.inference_mode():
        output=run(args.expert)
    (HERE/(args.expert+".json")).write_text(json.dumps(output,indent=1,allow_nan=False))
