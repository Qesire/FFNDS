"""Prespecified FFN audit on the frozen layer-19 teacher panels. No video inference."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / "kernel1120"))
from kernel_study import budget, features, fixed

ROOT = WORK / "native-inputs"
OUT = WORK.parent / "outputs/FFNDS_0951_mechanism"
WIDTH = 3.0              # latent-frame units, fixed before replay
MIX = 0.5                # convex sum; same within-frame covariance
RANDOM_SEED = 5951       # shuffled chronological labels, same permutation for all


def transform_edges(a, edges):
    i, j = edges.T
    return torch.cat((a / len(a)**0.5, (a[j] - a[i]) / len(edges)**0.5))


def readout(kernel, p, h, inv_metric, obs, base, q, v, score_gate=None):
    s = kernel[p][:, p].double() + inv_metric
    chol = torch.linalg.cholesky(s)
    y = (obs - base[p]).double()
    qp, vh = q[p].double(), v.double()
    weighted = torch.cholesky_solve(qp, chol)
    right = vh if score_gate is None else vh * score_gate.double()
    target = y if score_gate is None else y * score_gate.double()
    gram = (qp.T @ weighted) * (right @ right.T)
    gains = torch.linalg.solve(gram, (weighted * (target @ right.T)).sum(0))
    resid = y - (qp * gains) @ vh
    cross = kernel[h][:, p].double()
    mean = base[h] + (q[h] * gains.float()) @ v
    correction = (cross @ torch.cholesky_solve(resid, chol)).float()
    return mean, correction, cross


def oracle(energy_target, query_features, cross, v_gate, error, remaining, kernel_term):
    # X has shape (transformed evaluation positions, 64). Remove its column space
    # first. The 16 remaining fitted gains are tied across all output channels.
    u, sig, _ = torch.linalg.svd(cross, full_matrices=False)
    basis = u[:, sig > sig[0]*1e-10]
    remove = lambda x: x - basis @ (basis.T @ x)
    z, y = remove(query_features), remove(energy_target)
    gram = (z.T @ z) * (v_gate @ v_gate.T)
    rhs = ((y @ v_gate.T) * z).sum(0)
    gains = torch.linalg.pinv(gram, hermitian=True) @ rhs
    remainder = y - (z * gains) @ v_gate
    target2 = energy_target.square().sum().item()
    observed2 = error.square().sum().item()
    oracle2 = remainder.square().sum().item()
    a = ((remaining * kernel_term).sum()/kernel_term.square().sum()).item()
    amplitude_floor = (remaining - a*kernel_term).square().sum().item()
    norm_r = remaining - .5*kernel_term
    deployed_continuous = norm_r.square().sum().item()
    assert oracle2 < deployed_continuous*(1+1e-5) and amplitude_floor < deployed_continuous*(1+1e-5)
    return dict(target_energy=target2,deployed_energy=observed2,
                deployed_continuous_energy=deployed_continuous, oracle_energy=oracle2,
                irreducible_fraction=oracle2/deployed_continuous,
                best_scalar=a,scalar_improvement=1-amplitude_floor/deployed_continuous,
                cross_rank=basis.shape[1],cross_sigma_ratio=(sig[-1]/sig[0]).item())


def run(expert):
    rows,model,meta = budget.setup(ROOT,expert)
    pairs = [features(ROOT,rows[s],model) for s in fixed.CAL]
    beta = (sum((a.double()*b).mean(0) for a,b in pairs) /
            sum(a.double().square().mean(0) for a,b in pairs).clamp_min(1e-30)).float()
    scale = torch.stack([(b-a*beta).square().mean(0) for a,b in pairs]).mean(0).sqrt().clamp_min(1e-8)
    baseline=json.loads((WORK/"conditional1120"/(expert+".json")).read_bytes())
    output=[]
    for src in sorted({r["source"] for r in baseline}):
        ids,coords,b,q,y0,delta,observe = budget.panel(ROOT,rows[src],model)
        gate = fixed.panel(ROOT,rows[src]["hit"])["gate"].reshape(-1).cuda().float()
        x = features(ROOT,rows[src],model,beta,scale)
        k = x @ x.T / x.shape[1]
        labels = coords[:,0]
        scrambled = np.random.default_rng(RANDOM_SEED).permutation(21)[labels]
        kernels={"global":k,"gate_weighted":k}
        for name,seq in (("temporal",labels),("permuted",scrambled)):
            t=torch.as_tensor(seq,device=k.device,dtype=k.dtype)
            kernels[name]=k*((1-MIX)+MIX*torch.exp(-.5*((t[:,None]-t[None,:])/WIDTH).square()))
        for fold,(pool,h) in enumerate(budget.split_panel(ids)):
            p=budget.layout(pool,ids,coords,1120,"uniform")[:64]
            edges_p=budget.edge_indices(p,coords,ids)
            edges_h=budget.edge_indices(h,coords,ids)
            state=budget.eval_cache(delta[h],edges_h,gate)
            eye=torch.eye(len(p),device=k.device,dtype=torch.float64)
            d=eye[edges_p[:,1]]-eye[edges_p[:,0]]
            inv=torch.cholesky_inverse(torch.linalg.cholesky(eye + len(p)/len(edges_p)*(d.T@d)))
            obs=observe(p)
            truth=y0[h]+delta[h]
            transform=lambda a:transform_edges(a.double()*gate.double(),edges_h)
            report=dict(expert=expert,source=src,fold=fold)
            for name,gram in kernels.items():
                m,c,z=readout(gram,p,h,inv,obs,b,q,model["v"],gate if name=="gate_weighted" else None)
                out=(m+.5*c).bfloat16().float()
                met=budget.metrics(out-y0[h],state)
                if name=="global":
                    old=next(r["gls_half"] for r in baseline if r["source"]==src and r["fold"]==fold)
                    assert abs(met["J"]/old["J"]-1)<5e-5,(src,fold,met["J"],old["J"])
                stats=oracle(transform(truth-b[h]),transform_edges(q[h].double(),edges_h),
                             transform_edges(z,edges_h),model["v"].double()*gate.double(),
                             transform(truth-out),transform(truth-m),transform(c))
                report[name]=dict(metrics=met,oracle=stats)
            output.append(report)
        print(expert,src,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/("历史教师回放_"+expert+".json")).write_text(json.dumps(dict(width=WIDTH,mix=MIX,permutation_seed=RANDOM_SEED,
                      cal=fixed.CAL,fit=meta,rows=output),indent=1,ensure_ascii=False,allow_nan=False),encoding="utf8")


if __name__=="__main__":
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(3)
    with torch.inference_mode():
        for e in ["high","low"]:
            run(e)
