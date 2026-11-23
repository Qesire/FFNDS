"""Same-site, same-image cross-prompt residual-mean diagnostic. CPU-only replay."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

BASE="vbi_0951"
PAN=BASE+"_pan_left"
# Frozen from independent layer-19 CAL, before any new trajectory target is read.
ALPHA={1:.16065411025566734,3:.1858082977625006}


def read(p):return json.loads(p.read_bytes())


def inverse_metric(ids,p):
    co=np.array(np.unravel_index(ids,(21,29,52))).T
    eye=np.eye(len(p))
    pairs=set()
    for n,i in enumerate(p):
        choices=np.flatnonzero((co[p,0]==co[i,0])&(p!=i))
        if len(choices):
            dist=((co[p[choices],1:]-co[i,1:])**2).sum(1)
            m=choices[np.lexsort((ids[p[choices]],dist))[0]]
            pairs.add(tuple(sorted((int(n),int(m)))))
    left,right=np.array(sorted(pairs)).T
    d=eye[left]-eye[right]
    return np.linalg.inv(eye+len(p)/len(d)*(d.T@d))


def fit(base,triples,ids,v):
    pos=np.searchsorted(ids,triples)
    idx=np.unique(pos)
    i,j=pos[:,0],pos[:,1]
    scale=np.sqrt(len(idx)/len(i))
    def design(a):return np.concatenate([a[idx],scale*(a[j]-a[i])])
    x=design(base["X"].astype(np.float32))/np.sqrt(np.float32(base["X"].shape[1]))
    y=design(base["E"]-base["D"][:,:16]@v)
    gram=(x@x.T).astype(np.float64)
    gram.flat[::len(gram)+1]+=1.
    # W*=X.T (X X.T+I)^-1 Y; retain the dual so the offline 2560x5120 map needn't be stored.
    return x,np.linalg.solve(gram,y).astype(np.float32),idx


def predict(data,p,inv,v,mean):
    d=data["D"].astype(np.float64)
    q,k=d[:,:16],d[:,16:]
    cov=(k[p]+k[p].T)/2+inv
    iq=np.linalg.solve(cov,q[p])
    y=data["C"][p].astype(np.float64)-mean[p]
    vd=v.astype(np.float64)
    g=np.linalg.solve((q[p].T@iq)*(vd@vd.T),(iq*(y@vd.T)).sum(0))
    trend=(q*g)@vd
    pred=mean+trend+(.5*k)@np.linalg.solve(cov,y-trend[p])
    pred[p]=data["C"][p]
    return pred


def metrics(data,pred,triples,ids):
    i,j,t=np.searchsorted(ids,triples).T
    gate=data["gate"]
    er=(pred-data["E"])*gate
    true=data["F"]*gate
    def n2(x):return (x*x).sum(1)
    point=n2(er[i]).mean()
    out={"point":float(point)}
    for name,nb in (("spatial",j),("time",t)):
        e,truth=er[nb]-er[i],true[nb]-true[i]
        predicted=truth+e
        gt=n2(truth)
        dot=(truth*predicted).sum(1)
        rho=dot/np.maximum(np.sqrt(gt*n2(predicted)),1e-25)
        out[name]=float(n2(e).mean())
        out[name+"_angle"]=float(np.median(np.degrees(np.arccos(rho.clip(-1,1)))))
        out[name+"_signed"]=float(np.median(dot/np.maximum(gt,1e-25)))
    out["joint"]=float(point+out["spatial"]+out["time"])
    return out


def run(stage,dest):
    config=read(stage/"base.json")
    manifest=read(Path(config["cal_manifest"]["path"]))
    panel=read(stage/"panel.json")
    ids=np.asarray(panel["ids"]); p=np.searchsorted(ids,panel["probes"])
    inv=inverse_metric(ids,p)
    edges=np.asarray(panel["edges"])
    train=edges[edges[:,0]==0,2:]
    out=[]
    for s in (1,3):
        for l in (0,13,26,39):
            ref=manifest["sites"][("high" if s==1 else "low")+f"{l:02d}"]
            path=Path(ref["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==ref["sha256"]
            v=torch.load(path,map_location="cpu",weights_only=True,mmap=True)["v"].numpy()
            def load(name):
                with np.load(stage/"output/raw"/f"{name}_s{s}_l{l:02d}.npz") as data:
                    return dict(data)
            base=load(BASE)
            x,z,idx=fit(base,train,ids,v)
            del base
            # The left-pan labels are used below only for the 64 registered probes and scoring.
            pan=load(PAN)
            mean=((pan["X"]/np.sqrt(np.float32(x.shape[1])))@x.T)@z*ALPHA[s]
            uncorr=predict(pan,p,inv,v,np.zeros_like(mean))
            improved=predict(pan,p,inv,v,mean)
            row=dict(step=s,layer=l,alpha=ALPHA[s],train_positions=len(idx),results={})
            for n,name in enumerate(("all","warm_core","color_boundary","textured_fence")):
                sub=edges[(edges[:,0]==1)&((edges[:,1]==n) if n else True),2:]
                assert not np.isin(sub,ids[p]).any()
                row["results"][name]=dict(n=len(sub),shared_train_coordinate_count=int(np.isin(np.unique(sub),ids[idx]).sum()),
                    deployed=metrics(pan,pan["C"],sub,ids),continuous_baseline=metrics(pan,uncorr,sub,ids),
                    cross_prompt_mean=metrics(pan,improved,sub,ids))
            out.append(row)
            print("XW",s,l,flush=True)
    (dest/"optimal_xw.json").write_text(json.dumps(dict(rows=out,
        protocol="Same source image, native candidate base-prompt site state and labels fit static residual mean; different prompt held out. Ideal continuous XW, original 64 paid observations refit; no final BF16 or video."),indent=1,allow_nan=False))


if __name__=="__main__":
    torch.set_num_threads(2)
    run(Path(sys.argv[1]),Path(sys.argv[2]))
