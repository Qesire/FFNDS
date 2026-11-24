"""Isolate head contrast, FFN kernel, and their combination on the same probes."""
import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open
import torch

from attention_joint39_head_contrast import Assets, design, digest, energies, normalize, save, solve


ORDER=("phi48","head112","ffn112","D176")
METRICS=("point","spatial","temporal")


def checked_tensors(record,keys):
    path=Path(record["path"])
    assert digest(path)==record["sha256"] and path.stat().st_size==record["bytes"],path
    with safe_open(str(path),framework="pt",device="cpu") as f:
        return {key:f.get_tensor(key) for key in keys}


def read(path):
    return json.loads(path.read_bytes())


def dicts(phi,ab,ff,p):
    original=normalize(phi,p)
    a,b=(normalize(v,p) for v in (ab,ff))
    designs={
        "phi48":original,
        "head112":torch.cat((original,a),dim=1),
        "ffn112":torch.cat((original,b),dim=1),
        "D176":torch.cat((original,a,b),dim=1),
    }
    assert torch.equal(designs["D176"][:,:112],designs["head112"])
    assert torch.equal(designs["D176"][:,112:],designs["ffn112"][:,48:])
    return designs


def four_arm_metrics(d,base,truth,p,h,e):
    out={key:solve(d[key],base,truth,p,h,e) for key in ORDER}
    for higher,lower in (
            ("head112","phi48"),("ffn112","phi48"),("D176","head112"),("D176","ffn112")):
        assert out[higher]["H448_OLS_floor"]["point"]<=out[lower]["H448_OLS_floor"]["point"]+1e-7
    out["base"]=energies(base-truth,h,e)
    # The additive-scale interaction is an observed contrast of these fits,
    # evaluated on the same H448 and expressed in absolute energy units.
    out["P64_addition_interaction"]={m:
        out["D176"]["P64_fit"][m]-out["head112"]["P64_fit"][m]
        -out["ffn112"]["P64_fit"][m]+out["phi48"]["P64_fit"][m] for m in METRICS}
    return out


@torch.inference_mode()
def main(reduced,fit,export,index,out):
    idx=Assets(reduced,fit/"plan_assets","cpu")
    raw=read(export)
    assert raw["status"]=="complete" and len(raw["records"])==8 and len(raw["trace_controls"])==4
    background_trace={(x["source"],x["background"]):x["exact_trace"] for x in raw["trace_controls"]}
    assert len(background_trace)==4 and all(background_trace.values())
    manifest=read(index)
    assert manifest["export_sha256"]==digest(export)
    assert raw["panel_results_sha256"]==digest(reduced/"RESULTS.sha256")
    by_key={(r["source"],r["background"],r["expert"]):r for r in raw["records"]}
    old={(r["source"],r["background"],r["expert"]):r for r in manifest["previous"]}
    assert set(by_key)==set(old) and len(old)==8
    panel={(s.source,s.row["background"],s.row["expert"]):s for (module,_),sites in idx.groups.items()
           if module=="self" for s in sites if s.row["layer"]==39}
    maps={}
    for expert in ("high","low"):
        f=fit/"fit_full"/f"self_{expert}_transform.npz"
        recipe=read(f.parent/f"self_{expert}_recipe.json")
        assert digest(f)==recipe["transform_sha256"]
        with np.load(f,allow_pickle=False) as z:
            maps[expert]={name:{"basis":torch.from_numpy(z[name+"_basis"])} for name in ("finite","out_current","out_delta")}
    results=[]
    edges=dict(zip(("spatial","temporal"),idx.edges))
    p,h=idx.p,idx.h
    for key,r in by_key.items():
        previous=old[key]
        assert r["source_asset_sha256"]==previous["sha256"]
        assert r["cheap_q16_bitwise"] and all(r[k] for k in ("live_X","live_A","live_Y"))
        assert r["normalization_divisor"]==2560
        prior=checked_tensors(previous,("probe_rows","held_rows","joint_phi","flow_B00","flow_cheap","flow_B11_fixed"))
        ffn=checked_tensors(r["asset"],("probe_rows","held_rows","ffn_kernel_x"))
        for v in (prior,ffn):
            assert torch.equal(p,v["probe_rows"]) and torch.equal(h,v["held_rows"])
        x=ffn["ffn_kernel_x"].double()
        assert x.shape==(512,2560)
        k_f=x@x[p].T/r["normalization_divisor"]
        phi=prior["joint_phi"].double()
        arm="out_current" if key[2]=="high" else "finite"
        heads=design(idx.prepare(panel[key]),maps[key[2]])[arm][...,1:].double()
        assert heads.shape==(40,512,31)
        mean_h=heads.mean(0)
        assert float((mean_h-phi[:,1:32]).norm()/phi[:,1:32].norm())<0.004
        c=(heads-mean_h[None]).transpose(0,1).reshape(512,-1)
        k_a=c@c[p].T/40
        for k in (k_a,k_f):
            assert k.shape==(512,64) and torch.allclose(k[p],k[p].T)
        d=dicts(phi,k_a,k_f,p)
        truth=prior["flow_B00"].double()
        scores={base:four_arm_metrics(d,prior[name].double(),truth,p,h,edges)
                for base,name in (("cheap","flow_cheap"),("B11_fixed","flow_B11_fixed"))}
        results.append(dict(source=key[0],background=key[1],expert=key[2],
            source_asset_sha256=previous["sha256"],ffn_kernel_asset_sha256=r["asset"]["sha256"],scores=scores))
    assert len(results)==8
    save(out,dict(status="complete",sites=results,kernel_export_job=raw["job"],
        plan_commit=raw["plan_commit"],kernel_export_sha256=manifest["export_sha256"],
        panel_results_sha256=raw["panel_results_sha256"],producer_sha256=digest(Path(__file__)),
        widths=dict(zip(ORDER,[48,112,112,176])),lambda_per_query=0.1,
        kernel_centers="P64 in original production order",scales="P64 RMS per column",
        subdomain="layer39, previous two DEV sources, same teacher-forced block input on each background",
        new_model_executions_during_replay=0,solver_precision="CPU FP64; P64 predictions FP32, diagnostic floor FP64"))


if __name__=="__main__":
    main(*(Path(a).resolve() for a in sys.argv[1:]))
