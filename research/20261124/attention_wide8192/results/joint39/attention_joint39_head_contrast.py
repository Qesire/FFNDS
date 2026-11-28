"""Nested P64 comparison with the preregistered centered head kernel alone."""
import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open
import torch

from attention_panel_fit import Assets, design, digest, save


RIDGE=0.1


def normalize(x,probe):
    rms=x[probe].square().mean(0).sqrt()
    return x * torch.where(rms>0,rms.reciprocal(),0)


def energies(e,held,edges):
    score=dict(point=float(e[held].square().sum()))
    for name,edge in edges.items():
        i,j=edge.T
        score[name]=float((e[i]-e[j]).square().sum())
    return score


def solve(x,b,truth,probe,held,edges):
    rhs=truth-b
    xp=x[probe]
    kernel=xp@xp.T
    reg=len(probe)*RIDGE*torch.eye(len(probe),dtype=x.dtype)
    dual=torch.linalg.solve(kernel+reg,rhs[probe])
    theta=xp.T@dual
    pred=(b+x@theta)
    pred[probe]=truth[probe]
    pred=pred.float().double()
    xh=x[held]
    floor=torch.linalg.lstsq(xh,rhs[held],driver="gelsd").solution
    hreg=torch.linalg.solve(xh.T@xh+(len(held)*RIDGE)*torch.eye(x.shape[1],dtype=x.dtype),xh.T@rhs[held])
    return dict(P64_fit=energies(pred-truth,held,edges),
        H448_OLS_floor=energies(x@floor-rhs,held,edges),
        H448_same_ridge=energies(x@hreg-rhs,held,edges),
        P64_rank=int(torch.linalg.matrix_rank(xp)))


@torch.inference_mode()
def main(reduced,fit,index,out):
    idx=Assets(reduced,fit/"plan_assets","cpu")
    index_record=json.loads(index.read_bytes())
    maps={}
    for expert in ("high","low"):
        f=fit/"fit_full"/f"self_{expert}_transform.npz"
        recipe=json.loads((f.parent/(f"self_{expert}_recipe.json")).read_bytes())
        assert digest(f)==recipe["transform_sha256"]
        with np.load(f,allow_pickle=False) as z:
            maps[expert]={k:{"basis":torch.from_numpy(z[k+"_basis"])} for k in ("finite","out_current","out_delta")}
    by_key={(s.source,s.row["background"],s.row["expert"]):s for (module,_),sites in idx.groups.items()
            if module=="self" for s in sites if s.row["layer"]==39}
    records=[]
    edges=dict(zip(("spatial","temporal"),idx.edges))
    has_full_ffn_kernel_fields=False
    for item in index_record["assets"]:
        key=(item["source"],item["background"],item["expert"])
        site=by_key[key]
        path=Path(item["path"])
        assert digest(path)==item["sha256"]
        with safe_open(str(path),framework="pt",device="cpu") as f:
            cols=set(f.keys())
            has_full_ffn_kernel_fields |= any(k in cols for k in (
                "u0","y0","z0k","x","kernel_x","ffn_kernel_x","ffn_kernel_features"))
            tensors={k:f.get_tensor(k) for k in ("probe_rows","held_rows","joint_phi","flow_B00","flow_cheap","flow_B11_fixed")}
        p,h=idx.p,idx.h
        assert torch.equal(p,tensors["probe_rows"]) and torch.equal(h,tensors["held_rows"])
        case=idx.prepare(site)
        phi=tensors["joint_phi"].double()
        arm="out_current" if key[2]=="high" else "finite"
        head=design(case,maps[key[2]])[arm][...,1:].double()
        assert head.shape==(40,512,31)
        mean=head.mean(dim=0)
        mean_discrepancy=float((mean-phi[:,1:32]).norm()/phi[:,1:32].norm())
        assert mean_discrepancy<0.004,(key,mean_discrepancy)
        centered=head-mean[None]
        Z=centered.permute(1,0,2).reshape(512,-1)
        KC=(Z@Z[p].T)/40
        assert KC.shape==(512,64) and torch.allclose(KC[p],KC[p].T)
        D0=normalize(phi,p)
        D1=torch.cat((D0,normalize(KC,p)),dim=1)
        assert D1.shape[1]==112 and torch.equal(D1[:,:48],D0)
        truth=tensors["flow_B00"].double()
        scores={}
        for baseline,tensor in (("cheap","flow_cheap"),("B11_fixed","flow_B11_fixed")):
            y=tensors[tensor].double()
            x0=solve(D0,y,truth,p,h,edges)
            x1=solve(D1,y,truth,p,h,edges)
            assert x1["H448_OLS_floor"]["point"]<=x0["H448_OLS_floor"]["point"]+1e-7
            scores[baseline]={"base":energies(y-truth,h,edges),"phi48":x0,"phi48_head64":x1}
        records.append(dict(source=key[0],background=key[1],expert=key[2],site_asset_sha256=item["sha256"],
            head_mean_relative_reconstruction_difference=mean_discrepancy,scores=scores))
    assert len(records)==8
    save(out,dict(status="complete",records=records,new_model_executions=0,gpu_used=False,
        ridge=RIDGE,kernel_centers="fixed P64",affine_nested_columns=[48,112],
        head_kernel="sum of centered-head feature products divided by 40; extra columns normalized by P64 RMS",
        sealed_vectors_contain_full_ffn_kernel_or_anchor=has_full_ffn_kernel_fields,
        scope="Compare head-only addition. FFN kernel and full joint D176 were not evaluated.",
        capture_sha256=digest(reduced/"RESULTS.sha256"),producer_sha256=digest(Path(__file__))))


if __name__=="__main__":
    main(*(Path(s).resolve() for s in sys.argv[1:]))
