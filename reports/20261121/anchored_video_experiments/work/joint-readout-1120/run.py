"""Reduced-rank regression for the original pre-gate point and edge target."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import scipy.linalg as sla
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK))
from native_replay import digest, load_weights, make_queries, panel
from native_slopes import operands

CAL = ("vbi_0032", "vbi_0304", "vbi_0456")
DEV = ("vbi_0392", "vbi_0800", "vbi_1013")
RANK = 16


def thash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def data(root, row, ws, beta, sel):
    dz, dh, _, h = operands(root, row, ws)
    e = dh-dz*beta
    target = F.linear(dh, ws[2])
    base = F.linear(dz*beta,ws[2]) + F.linear(e[:,sel],ws[2][:,sel])
    ids = h["token_indices"].numpy()
    coords = h["token_thw"].numpy()
    _, _, q, edges = make_queries(ids, coords)
    unique = np.unique(np.sort(edges,axis=1),axis=0)
    i,j = unique.T.copy()
    dy,dx = (coords[j,1:] - coords[i,1:]).T
    flip = np.where(abs(dx)>=abs(dy),dx,dy)<0
    i[flip],j[flip]=j[flip],i[flip].copy()
    assert set(i).union(j) <= set(q)
    return dict(z=e[:,sel],target=target,base=base,q=q,i=i,j=j)


def fit(packs, scale, gamma, match_trace=False):
    Xs,Ys=[],[]
    for item in packs:
        z,y = item["z"]/scale,item["target"]-item["base"]
        q,i,j=[item[k] for k in ("q","i","j")]
        Xs.append(z[q] / (len(CAL)*len(q))**.5)
        Ys.append(y[q] / (len(CAL)*len(q))**.5)
        if gamma:
            Xs.append((z[j]-z[i])*(gamma/(len(CAL)*len(i)))**.5)
            Ys.append((y[j]-y[i])*(gamma/(len(CAL)*len(i)))**.5)
    x,y = torch.cat(Xs),torch.cat(Ys)
    gram, cross = x.T@x, x.T@y
    data_trace_per_feature = gram.diagonal().mean().item()
    ridge = .1 * data_trace_per_feature if match_trace else .1
    gram.diagonal().add_(ridge)
    L = torch.linalg.cholesky(gram)
    whitened = torch.linalg.solve_triangular(L,cross,upper=False)
    sym = (whitened @ whitened.T).cpu().double().numpy()
    val, vec = sla.eigh(sym,subset_by_index=[len(sym)-RANK-1,len(sym)-1],check_finite=True,driver="evr")
    u = torch.from_numpy(vec[:,-RANK:].copy()).cuda().float()
    eig_residual = ((whitened@(whitened.T@u))-u*torch.tensor(val[-RANK:],device="cuda")).norm()/(whitened@(whitened.T@u)).norm()
    assert eig_residual < 2e-5, eig_residual
    left = torch.linalg.solve_triangular(L.T,u,upper=True) / scale[:,None]
    right = u.T @ whitened
    gain = float(val[-RANK:].sum())
    actual = (y.square().sum() - (y-(x@(scale[:,None]*left))@right).square().sum()
              - ridge*((scale[:,None]*left)@right).square().sum()).item()
    assert abs(gain-actual) < 3e-5*gain, (gain,actual)
    return left.contiguous(), right.contiguous(), dict(
        trace_svd_rank16=gain, direct_cal_regularized_improvement=actual,
        data_trace_per_feature=data_trace_per_feature, ridge=ridge,
        sv_16=float(np.sqrt(val[-RANK])),sv_17=float(np.sqrt(max(0,val[-RANK-1]))),
        relative_eigensystem_residual=float(eig_residual))


def metrics(pred,t,q,i,j):
    e=(pred-t).double();t=t.double()
    ed,td=e[j]-e[i],t[j]-t[i]
    ec,tc=(e[j]+e[i])/2**.5,(t[j]+t[i])/2**.5
    estimate=td+ed
    assert td.norm( dim=1).min()>0 and estimate.norm(dim=1).min()>0
    def scalar(x):
        return x.item()
    def pair(a,b):
        return torch.stack([torch.stack([(x*y).sum() for y in b]) for x in a])
    ep,tp=[ec,ed/2**.5],[tc,td/2**.5]
    t2,te,e2=pair(tp,tp),pair(tp,ep),pair(ep,ep)
    norm=t2.diag().sqrt()
    n=torch.outer(norm,norm)
    gt,h,c=t2/n,te/n,e2/n
    slopes=torch.linalg.solve(gt,h)
    perp=c-h.T@slopes
    p_loss=e[q].square().sum()/len(q)
    r_loss=ed.square().sum()/len(i)
    return dict(point_relative=scalar(e[q].norm()/t[q].norm()),edge_relative=scalar(ed.norm()/td.norm()),
                point_loss=scalar(p_loss), edge_loss=scalar(r_loss), objective=scalar(p_loss+r_loss),
                directional_angle_median_deg=scalar(torch.quantile(torch.rad2deg(torch.acos(
                    F.cosine_similarity(estimate,td,dim=1).clamp(-1,1))),.5)),
                median_signed_edge_scale=scalar(torch.quantile((estimate*td).sum(1)/td.square().sum(1),.5)),
                common_relative=scalar(ec.norm()/tc.norm()),joint_scale=(torch.eye(2,device=t.device)+slopes).tolist(),
                perpendicular_joint_energy=perp.tolist(),signed_target_error=h.tolist(),error_gram=c.tolist())


def run(root,expert,trace_control=False):
    path=root/"BINDINGS.json"
    assert digest(path)=="8f7320c61dbafd44df0b1662235405de8961dd5c5d52d846c979fdb4179b1ab6"
    binding=json.loads(path.read_text())
    rows={r["source"]:r for r in binding["pairs"] if r["expert"]==expert and r["layer"]==19}
    ws=load_weights(root,expert,binding["sites"][expert+"19"])
    w1,b1,w2,_=ws
    num=torch.zeros(len(w1),device="cuda",dtype=torch.float64); den=torch.zeros_like(num)
    for src in CAL:
        dz,dh,_,_=operands(root,rows[src],ws)
        num+=(dh.double()*dz).mean(0); den+=dz.double().square().mean(0)
    beta=(num/den.clamp_min(1e-30)).float()
    baseline=json.loads((WORK/"compiled"/(expert+".json")).read_text())
    assert thash(beta)==baseline["beta_sha256"]
    sel=torch.tensor(baseline["selected_ids"],device="cuda")
    assert thash(sel)==baseline["selected_sha256"]
    packs=[data(root,rows[s],ws,beta,sel) for s in CAL]
    scale=torch.stack([p["z"][p["q"]].square().mean(0) for p in packs]).mean(0).sqrt().clamp_min(1e-8)
    result=dict(method="fixed R16, common RMS from CAL query rows, pre-gate, one point mean and one edge mean per source",
                CAL=list(CAL),DEV=list(DEV),gamma=1,rank=RANK,ridge=.1,beta_sha256=thash(beta),
                selection_sha256=thash(sel),scale_min=scale.min().item(),fit={},cases={},numerics={})
    models={}
    plan=(("joint16_trace_matched",1),) if trace_control else (("point16",0),("joint16",1))
    for name,gamma in plan:
        l,r,stats=fit(packs,scale,gamma,match_trace=trace_control)
        models[name]=(l,r)
        result["fit"][name]=stats
    del packs
    omitted=beta.clone();omitted[sel]=0
    a0=(w1.T*omitted)@w2.T
    b0=w2[:,sel].T.contiguous()
    coeffs={k:l@r for k,(l,r) in models.items()}
    mats={"fixed":(a0,b0)}
    small=w1[sel].T*beta[sel]
    for k,c in coeffs.items():
        # Left-associated small@l@r makes offline folding  O(d*k*r + d*d*r).
        l,r=models[k]
        mats[k]=(a0-(small@l)@r,b0+c)
    hfun=lambda u,w,b:F.gelu(F.linear(u,w,b),approximate="tanh")
    for src in DEV:
        pack=data(root,rows[src],ws,beta,sel)
        q,i,j=[pack[k] for k in ("q","i","j")]
        evals={}; eq={}
        a,h=[panel(root,rows[src][k]) for k in ("anchor","hit")]
        assert torch.equal(a["token_indices"],h["token_indices"]) and torch.equal(a["token_thw"],h["token_thw"])
        u0,u1=[p["U"].cuda() for p in (a,h)]
        du=u1.float()-u0.float()
        h32=hfun(u1.float(),w1[sel],b1[sel])-hfun(u0.float(),w1[sel],b1[sel])
        h0=hfun(u0,w1.bfloat16(),b1.bfloat16())[:,sel]
        hhit=hfun(u1,w1[sel].bfloat16(),b1[sel].bfloat16())
        y0=a["Y_pre_gate"].cuda()
        native_delta=h["Y_pre_gate"].cuda().float()-y0.float()
        for name,(A,B) in mats.items():
            raw=pack["base"]
            if name in models:
                l,r=models[name]
                raw=raw+(pack["z"]@l)@r
            folded=du@A+h32@B
            gap=(folded-raw).norm()/raw.norm()
            assert gap<2e-5, (src,name,gap)
            # Folded BF16 weights, historical narrow W2 FP32 accumulation and BF16 final addition.
            mixed=(y0.float()+(du.bfloat16()@A.bfloat16()).float()
                   +(hhit.float()-h0.float()).bfloat16().float()@B.bfloat16().float()).bfloat16().float()-y0.float()
            eq[name]=dict(fp32_fold_relative=float(gap),bf16_vs_structural_relative=float((mixed-raw).norm()/raw.norm()))
            evals[name]=dict(fp32=metrics(raw,pack["target"],q,i,j),mixed=metrics(mixed,native_delta,q,i,j))
        result["cases"][src]=dict(query=len(q),edges=len(i),arms=evals)
        result["numerics"][src]=eq
        print(expert,src,{k:round(v["mixed"]["objective"],6) for k,v in evals.items()},flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("expert",choices=["high","low"])
    p.add_argument("--trace-control",action="store_true")
    args=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4)
    with torch.inference_mode():
        result=run(WORK/"native-inputs",args.expert,args.trace_control)
    stem=args.expert+("_trace_control" if args.trace_control else "")
    (HERE/(stem+".json")).write_text(json.dumps(result,indent=2,allow_nan=False))
