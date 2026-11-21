"""Exact held-out design-family floors from the 0951 same-input gated FFN Gram witnesses."""
import hashlib
import json
from pathlib import Path
import os
os.environ["OPENBLAS_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
import numpy as np
import scipy.sparse as sp

OUT=Path(__file__).resolve().parents[2]/"outputs/FFNDS_0951_mechanism"
RUN=OUT/"runtime"


def incidence(triples,ids,kind):
    look={int(s):i for i,s in enumerate(ids)}
    ii,jj,kk=np.asarray([[look[x] for x in triple] for triple in triples]).T
    selector=sp.eye(len(ids),format="csr")
    parts={"point":selector[ii],"spatial":selector[jj]-selector[ii],"time":selector[kk]-selector[ii]}
    chosen=list(parts.values()) if kind=="joint" else [parts[kind]]
    return sp.vstack(chosen,format="csr")/np.sqrt(len(ii))


def both(A,G):return (A@(A@G).T).T


def result(a,pack,obs,D,p,inv):
    ae=a@pack["EV"]
    de=a@D
    z=de[:,16:]
    gram_e=both(a,pack["EE"])
    gram_r=both(a,pack["R2"])
    actual=float(gram_r.trace())
    ug,sv,_=np.linalg.svd(z,full_matrices=False)
    rank=np.sum(sv>sv[0]*1e-10)
    ug=ug[:,:rank]
    remove=lambda mat:mat-ug@(ug.T@mat)
    left=remove(de[:,:16])
    vv=pack["VV"]
    normal=(left.T@left)*vv
    rhs=(left*remove(ae)).sum(0)
    outside_kernel=float(gram_e.trace()-np.sum(ug*(gram_e@ug)))
    squared=outside_kernel-rhs@np.linalg.pinv(normal,hermitian=True)@rhs
    all_u,s,_=np.linalg.svd(de/np.maximum(np.linalg.norm(de,axis=0),1e-30),full_matrices=False)
    all_u=all_u[:,s>s[0]*1e-10]
    relaxed=gram_e.trace()-np.sum(all_u*(gram_e@all_u))
    # Best rank-80 matrix on the scored rows, allowing position directions unrelated to deployed features.
    # Rank of a left-transformed target depends on the finite panel and its scoring map A.
    eigenvalues=np.linalg.eigvalsh(gram_e)[::-1].clip(0.)
    rank80=float(eigenvalues[80:].sum())
    # Independent cheap-observation comparator. Only the registered 64 paid full-FFN outputs
    # (stored in C after probe overwrite), plus the visible deployed design, enter the fit.
    scale=np.sqrt(np.mean(D[p]**2,axis=0)).clip(1e-9)*np.sqrt(D.shape[1])
    dp = D[p]/scale
    weights = (de/scale) @ dp.T @ np.linalg.inv(dp@dp.T+inv)
    predicted=(gram_e.trace()-2*np.sum(weights*(a@pack["EC"][:,p]))
               +np.sum(weights*(weights@obs)))
    cross=float(both(a,pack["RK"]).trace())
    norm=float(both(a,pack["KK"]).trace())
    assert -.00005*actual<=rank80<=relaxed+actual*.00005 and relaxed<=squared+actual*.00005
    assert squared<=actual*1.001 and actual>0 and norm>0
    return dict(rank_kernel=int(rank),rank_design=int(all_u.shape[1]),energy=actual,
                exact_family_floor=float(squared),relaxed_floor=float(relaxed),rank80_floor=rank80,
                rank80_fraction=rank80/actual,
                irreducible_fraction=float(squared/actual),relaxed_fraction=float(relaxed/actual),
                exact_family_gain_cap=1.-float(squared/actual),same64_ridge_ratio=float(predicted/actual),
                scalar_cross=cross,scalar_norm=norm,scalar_oracle_eta=0.5*(1+cross/norm),
                scalar_gain_cap=cross**2/(norm*actual))


def probe_invm(ids,P):
    coord=np.column_stack((ids//1508,ids%1508//52,ids%52))
    edges=set()
    for k,idx in enumerate(P):
        eligible=np.flatnonzero((coord[P,0]==coord[idx,0]) & (P!=idx))
        if len(eligible):
            dist=np.square(coord[P[eligible],1:]-coord[idx,1:]).sum(1)
            other=eligible[np.lexsort((ids[P[eligible]],dist))[0]]
            edges.add(tuple(sorted((k,int(other)))))
    eye=np.eye(len(P))
    ed=np.array(sorted(edges))
    inc=eye[ed[:,1]]-eye[ed[:,0]]
    return np.linalg.inv(eye+len(P)/len(ed)*(inc.T@inc))


def main():
    cfg=json.loads((RUN/"audit.json").read_bytes())
    meta=cfg["rows"]
    ids=np.asarray(meta["ids"]); probes=np.asarray(meta["probes"])
    p=np.searchsorted(ids,probes); assert np.array_equal(ids[p],probes)
    inv=probe_invm(ids,p)
    source_names=("vbi_0951","vbi_0951_pan_left")
    out=[]
    for site in cfg["files"]:
        loc=RUN/"audit"/site["name"]
        assert hashlib.sha256(loc.read_bytes()).hexdigest()==site["sha256"]
        with np.load(loc) as z:pack={key:z[key] for key in z.files}
        assert np.array_equal(pack["ids"],ids)
        D=pack["D"].astype(np.float64)
        pack["R2"]=pack["EE"]+pack["CC"]-pack["EC"]-pack["EC"].T
        assert np.allclose(pack["R2"].diagonal(),pack["direct_sq"],rtol=5e-5,atol=1e-5)
        obs=pack["CC"][np.ix_(p,p)]
        source=source_names.index(site["source"])
        edges=np.asarray(meta["edges"])
        groups={"common":np.unique(edges[:,2:],axis=0)}
        for label,name in enumerate(meta["label_names"][1:],1):
            groups[name]=edges[(edges[:,0]==source)&(edges[:,1]==label),2:]
        assert not np.isin(groups["common"],probes).any()
        for group,triples in groups.items():
            for kind in (("joint","point","spatial","time") if group=="common" else ("joint",)):
                a=incidence(triples,ids,kind)
                stats=result(a,pack,obs,D,p,inv)
                out.append(dict(source=site["source"],step=site["step"],layer=site["layer"],
                                group=group,kind=kind,triples=len(triples),**stats))
    before={(r["step"],r["layer"]):r for r in out if r["source"]==source_names[0] and r["group"]=="common" and r["kind"]=="joint"}
    for r in out:
        a=before[(r["step"],r["layer"])]["scalar_oracle_eta"]
        beta=2*min(1,max(0,a))-1
        r["clipped_base_eta"]=.5*(beta+1)
        r["base_scalar_transfer_ratio"]=1+(beta**2*r["scalar_norm"]-2*beta*r["scalar_cross"])/r["energy"]
    with (OUT/"0951表示空间与估计上限.json").open("w",encoding="utf8") as f:
        json.dump(dict(source_index_sha=hashlib.sha256((RUN/"audit.json").read_bytes()).hexdigest(),
                       metric="real gated point+same-frame neighbor+fixed-coordinate next-latent-frame; excludes all probes",
                       continuous_scalar="new estimate anchored at the actual BF16 prediction, adding a multiple of measured pre-rounding half-kernel",
                       rows=out),f,allow_nan=False,indent=1)


if __name__=="__main__":
    main()
