"""Refit the unmodified 16-gain + half-kernel readout on frozen observed rows."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

def hashfile(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def metric(ids,time_edges):
    coords=np.stack(np.unravel_index(ids,(21,29,52)),axis=1)
    pairs=set()
    for i,c in enumerate(coords):
        same=np.flatnonzero((coords[:,0]==c[0])&(np.arange(len(ids))!=i))
        if len(same):
            dist=((coords[same,1:]-c[1:])**2).sum(axis=1)
            j=int(same[np.lexsort((ids[same],dist))[0]])
            pairs.add(tuple(sorted((i,j))))
    if time_edges:
        for i,c in enumerate(coords):
            j=np.flatnonzero((coords[:,0]==c[0]+1)&(coords[:,1]==c[1])&(coords[:,2]==c[2]))
            pairs.update((i,int(k)) for k in j)
    eye=np.eye(len(ids))
    i,j=np.array(sorted(pairs)).T
    diff=eye[i]-eye[j]
    return np.linalg.inv(eye+(len(ids)/len(i))*(diff.T@diff)),len(i)

def solve(ids,sort_index,raw,v,add_time=False,original_kernel=False):
    pos=np.array([sort_index[i] for i in ids])
    x,q,e=raw["X"],raw["D"][:,:16],raw["E"]
    k=x.shape[1]
    cross=raw["D"][:,16:] if original_kernel else (x@x[pos].T)/k
    if original_kernel:
        # Probe eval uses an independent 64-row FFN/feature GEMM. Its output C
        # was hard-copied at P, while E used a full row-shape dense invocation.
        ep=raw["C"][pos].astype(np.float64)
    else: ep=e[pos].astype(np.float64)
    vp=v.astype(np.float64)
    qp=q[pos].astype(np.float64)
    inv,nedge=metric(np.asarray(ids),add_time)
    system=cross[pos].astype(np.float64)+inv
    iq=np.linalg.solve(system,qp)
    normal=(qp.T@iq)*(vp@vp.T)
    gain=np.linalg.solve(normal,(iq*(ep@vp.T)).sum(axis=0))
    beta=np.linalg.solve(system,ep-(qp*gain)@vp)
    pred=(q*gain.astype(np.float32))@v+np.float32(.5)*(cross@beta.astype(np.float32))
    pred[pos]=ep
    return pred,dict(edges=nedge,condition_kernel=float(np.linalg.cond(system)),
                     condition_normal=float(np.linalg.cond(normal)),gain_range=[float(gain.min()),float(gain.max())])

def stats(r,f):
    rr=np.einsum("ij,ij->",r,r,dtype=np.float64)
    ff=np.einsum("ij,ij->",f,f,dtype=np.float64)
    rf=np.einsum("ij,ij->",r,f,dtype=np.float64)
    return dict(n=len(r),err_sq=float(rr),delta_sq=float(ff),error_dot_delta=float(rf),
                rms=float(np.sqrt(rr/r.size)),
                projection_gain=float(1+rf/ff),orth_relative=float(np.sqrt(max(0,rr/ff-(rf/ff)**2))))

def analyze(data,v,layout,panel,source):
    ids=np.asarray(panel["ids"])
    ix={int(i):a for a,i in enumerate(ids)}
    rows=np.array([ix[i] for i in layout["holdout"]])
    triples=np.asarray(layout["holdout_triples"])
    nd=np.array([[ix[i] for i in a[2:]] for a in triples])
    labels=np.asarray(panel["labels"])
    s=0 if source=="vbi_0951" else 1
    def summarize(pred):
        r=(pred-data["E"])*data["gate"]
        f=data["F"]*data["gate"]
        output={"all_point":stats(r[rows],f[rows])}
        for k,name in enumerate(("other","core","boundary","fence")):
            pick=rows[labels[s,rows]==k]
            if len(pick): output[name+"_point"]=stats(r[pick],f[pick])
        for group,t in [("all",nd),*[(name,nd[(triples[:,0]==s)&(triples[:,1]==k)]) for k,name in ((1,"core"),(2,"boundary"),(3,"fence"))]]:
            for end,name in ((1,"space"),(2,"time")):
                i,j=t[:,0],t[:,end]
                for sign,tag in ((1,"common"),(-1,"diff")):
                    output[f"{group}_{name}_{tag}"]=stats((r[j]+sign*r[i])*np.float32(2**-.5),(f[j]+sign*f[i])*np.float32(2**-.5))
        return output
    result={"deployed":{"stats":summarize(data["C"])}}
    for tag,layout_key,temporal,recon in (("actual64","actual64",False,False),
            ("reference_check","actual64",False,True),("uniform64","uniform64",False,False),
            ("paired64","paired64",False,False),("paired_time64","paired64",True,False)):
        pred,diag=solve(layout["panels"][layout_key],ix,data,v,temporal,recon)
        err=(pred[rows]-data["C"][rows])*data["gate"]
        reference=(data["C"][rows]-data["E"][rows])*data["gate"]
        diag["deployed_disagreement_over_deployed_error"]=float(np.linalg.norm(err)/np.linalg.norm(reference))
        result[tag]={"details":diag,"stats":summarize(pred)}
    return result

def main():
    import torch
    stage,dst=map(Path,sys.argv[1:3])
    layout=json.loads((dst/"relocation_layout.json").read_bytes())
    assert hashfile(stage/"panel.json")==layout["pool_sha256"]
    panel=json.loads((stage/"panel.json").read_bytes())
    config=json.loads((stage/"base.json").read_bytes())
    ref=config["cal_manifest"]
    assert hashfile(ref["path"])==ref["sha256"]
    cal=json.loads(Path(ref["path"]).read_bytes())["sites"]
    audit=json.loads((stage/"output/audit.json").read_bytes())
    assert all(all(o.values()) for o in audit["checks"].values()) and len(audit["files"])==16
    result={"job":os.environ.get("SLURM_JOB_ID"),"cpus":os.environ.get("SLURM_CPUS_PER_TASK"),
            "layout_sha256":hashfile(dst/"relocation_layout.json"),"sites":{}}
    for site in audit["files"]:
        n=site["name"]
        key=("high" if site["step"]==1 else "low")+f"{site['layer']:02}"
        packed=cal[key]
        weights=torch.load(packed["path"],weights_only=True,map_location="cpu",mmap=True)
        v=weights["v"].float().numpy().copy()
        del weights
        with np.load(stage/"output/raw"/n) as f:
            data={k:f[k] for k in ("E","D","F","C","gate","X")}
        assert v.shape==(16,5120) and data["E"].shape==(len(panel["ids"]),5120)
        assert np.isfinite(v).all() and all(np.isfinite(x).all() for x in data.values())
        result["sites"][n]=dict(source=site["source"],step=site["step"],layer=site["layer"],
            raw_sha256=hashfile(stage/"output/raw"/n),v_sha256=hashlib.sha256(v.tobytes()).hexdigest(),
            fits=analyze(data,v,layout,panel,site["source"]))
        print(n,flush=True)
    temp=dst/"relocation_results.tmp"
    temp.write_text(json.dumps(result,indent=1,allow_nan=False))
    temp.replace(dst/"relocation_results.json")

if __name__=="__main__":
    main()
