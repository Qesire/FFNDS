"""Frozen v2 CAL leave-one-source-out and six-source panel replay."""
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean
import sys

import torch


CAL = ("vbi_0032", "vbi_0304", "vbi_0456")
DEV = ("vbi_0392", "vbi_0800", "vbi_1013")
GRID = (0.001, 0.01, 0.1)
TRANSFORM_KEY = {
    "OUT-D":"out_delta", "OUT-ABS":"out_current",
    **{arm:"finite" for arm in ("FIN-D","FIN-C0","FIN-C1","FIN-CP","FIN-ABS")},
}
TUNED = ("OUT-D","OUT-ABS","FIN-D","FIN-C0","FIN-ABS")
MU = {"FIN-C0":0, "FIN-C1":1, "FIN-CP":1, "FIN-ABS":1}
ENERGIES = ("point","spatial","temporal")


def save(path, row):
    temp = path.with_suffix(".pending")
    temp.write_text(json.dumps(row,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def digest(path):
    checksum=sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def batched_outer(x):
    return x.transpose(-2,-1) @ x


def apply_transform(a, transform):
    """Use the same BF16 endpoint rounding at both ends of each difference."""
    mapped = (a @ transform).bfloat16().float()
    return torch.cat((torch.ones_like(mapped[...,:1]),mapped),dim=-1)


def schur_fit(x, t, c, m, ridge, mu=0, floor=False, zero_rhs=False):
    """Shapes [head,row,rank], [head,row,value], and optional rowwise mass/c."""
    xt = x.transpose(-2,-1)
    rank = x.shape[-1]
    if floor:
        u, s, vh = torch.linalg.svd(x,full_matrices=False)
        tol = torch.finfo(x.dtype).eps*max(x.shape[-2:])*s[:,:1]
        inv = torch.where(s>tol,s.reciprocal(),0)
        pinv = (vh.transpose(-2,-1)*inv[:,None,:]) @ u.transpose(-2,-1)
    else:
        eye = torch.eye(rank,device=x.device,dtype=x.dtype)
        L = torch.linalg.cholesky(xt @ x + ridge*eye)
        pinv = torch.cholesky_solve(xt,L)
    if c is None:
        return pinv @ t, None, None
    p = x.shape[1]
    H = torch.eye(p,device=x.device,dtype=x.dtype) - x @ pinv
    H = (H+H.transpose(-2,-1))*0.5
    a_matrix = xt @ ((H*(c @ c.transpose(-2,-1))) @ x)
    z = ((H @ t)*c).sum(-1,keepdim=True)
    rhs = -(xt @ z)
    if mu:
        a_matrix = a_matrix + mu*(xt @ x)
        if not zero_rhs:
            rhs = rhs + mu*(xt @ m)
    a_matrix = (a_matrix+a_matrix.transpose(-2,-1))*0.5
    spectrum = None
    if floor:
        # H and the Schur Gram use the FP64 subspace projection. Singleton
        # eigenvalues at roundoff have no coefficient solution to report.
        ev,V = torch.linalg.eigh(a_matrix)
        tol = ev[:,-1:].clamp_min(0)*torch.finfo(ev.dtype).eps*max(p,rank)*8
        inv = torch.where(ev>tol,ev.reciprocal(),0)
        a = V @ (inv[...,None]*(V.transpose(-2,-1) @ rhs))
        spectrum = {"min_schur_rank":int((ev>tol).sum(-1).min()),
                    "min_eigenvalue":float(ev.min())}
    else:
        a = torch.linalg.solve(a_matrix+ridge*eye,rhs)
    return pinv @ (t+(x @ a)*c), a, spectrum


def optimality(x,t,c,m,B,a,mu,ridge,zero=False):
    r = x@B-t
    if a is not None:
        r = r-(x@a)*c
    xt = x.transpose(-2,-1)
    gB = xt@r+ridge*B
    gb = float(gB.norm()/(xt@t).norm().clamp_min(torch.finfo(x.dtype).tiny))
    if a is None:
        return gb
    mass = x@a
    if mu and not zero:
        mass = mass-m
    gr = xt@(-(r*c).sum(-1,keepdim=True)+mu*mass)+ridge*a
    scale = (xt@(t*c).sum(-1,keepdim=True)).norm() + mu*(xt@m).norm()
    return max(gb,float(gr.norm()/scale.clamp_min(torch.finfo(x.dtype).tiny)))


@dataclass(frozen=True)
class Site:
    sidecar: Path
    root: Path
    row: dict
    index_row: dict

    @property
    def source(self):
        return self.row["source_id"]


class Assets:
    def __init__(self, path, frozen, device):
        from safetensors.torch import load_file
        import numpy as np
        self.device = device
        self.load = load_file
        self.path = path
        self.pv = torch.tensor(np.load(frozen/"bases/P_v.npy",allow_pickle=False),device=device,dtype=torch.float32)[:,:64]
        panel_path = frozen/"attention_panel_index.json"
        panel = read(panel_path)
        permutation = read(frozen/"c_direction_permutation.json")
        assert permutation["panel_sha256"]==digest(panel_path)
        self.p,self.h = [torch.tensor(panel[k],device=device) for k in ("probe_rows_in_production_order","held_rows")]
        assert self.p.numel()==64 and self.h.numel()==448
        assert set(self.p.tolist()).isdisjoint(self.h.tolist())
        self.perm = torch.tensor(permutation["source_row_for_destination"],device=device)
        assert sorted(self.perm.tolist())==list(range(512))
        self.edges = [torch.tensor(panel[k],device=device) for k in ("spatial_edges_rows","temporal_edges_rows")]
        assert all(edge.shape==(224,2) and torch.isin(edge,self.h).all() for edge in self.edges)
        self.groups = defaultdict(list)
        self.weights = {}
        for line in (path/"RESULTS.sha256").read_text().splitlines():
            expected,rel = line.split(maxsplit=1)
            index_path = path/rel
            assert digest(index_path)==expected
            root = index_path.parent
            index = read(index_path)
            for r in index["assets"]:
                sidecar = root/r["sidecar"]
                row = read(sidecar)
                assert row["query_manifest_sha256"]==digest(panel_path)
                self.groups[row["module"],row["expert"]].append(Site(sidecar,root,row,r))
        expected = set(product(CAL+DEV,("Dense","wide8192"),(0,13,26,39)))
        assert len(self.groups)==4
        for ss in self.groups.values():
            assert len(ss)==48 and {(s.source,s.row["background"],s.row["layer"]) for s in ss}==expected

    def prepare(self,site):
        r=site.row
        d=self.load(str(site.root/r["tensor_file"]),device=self.device)
        # Numerical feature data use [head,query,channel].
        a,y = (x.float().transpose(0,1) for x in d["o_native"])
        osa,oss = (x.float().transpose(0,1) for x in d["o_selected"])
        rho_a,rho_s = ((-(small-full).expm1()).transpose(0,1).unsqueeze(-1)
                      for small,full in zip(d["lse_selected"],d["lse_native"]))
        # Match the capture's independent FP32 logsumexp tolerance. Preserve
        # the signed value inside that rounding envelope as the plan requires.
        assert bool((rho_a>=-5.1e-5).all()) and bool((rho_s>=-5.1e-5).all())
        vstar = d["mean_V"][0,:,None,:]
        u0 = (a-(1-rho_a)*osa).bfloat16().float()
        u1 = y-(1-rho_s)*oss
        anchor = (1-rho_a)*oss+u0
        o_a,o_s = osa @ self.pv,oss @ self.pv
        fin_a,fin_s = (x.transpose(0,1) for x in d["feature_raw64"])
        prepared = dict(y=y,selected=oss,anchor=anchor, innovation=y-anchor, mass=rho_s-rho_a,
                    rho=rho_s,w0=u0-rho_a*vstar,w1=u1-rho_s*vstar,c=oss-vstar,
                    gate=d.get("self_gate_s"),
                    rho_roundoff=dict(anchor_negative_rows=int((rho_a<0).sum()),hit_negative_rows=int((rho_s<0).sum()),
                                      smallest_retained=float(torch.minimum(rho_a.min(),rho_s.min()))),
                    endpoints={"out_delta":(o_a,o_s),"out_current":(None,o_s),"finite":(fin_a,fin_s)})
        return prepared

    def weight(self,site):
        ref=site.row["projection_binding"]
        k=ref["weight_sha256"]
        if k not in self.weights:
            path=Path(ref["path"])
            assert digest(path)==k,path
            self.weights[k]=self.load(str(path),device=self.device)["weight_in_out"].float()
        return self.weights[k]

    def native_noise(self,site,gate):
        raw_lookup = read(self.raw[site.source])
        raw_root = self.raw[site.source].parent/site.row["background"]
        module = "self" if site.row["module"]=="self" else "cross"
        key = f"endpoints/{module}/s{site.row['hit_step']:02d}_b{site.row['layer']:02d}"
        file = raw_root/(key+".safetensors")
        assert file.stat().st_size==raw_lookup["backgrounds"][site.row["background"]]["assets"][key]["bytes"]
        z=self.load(str(file),device=self.device)
        k=z["k_effective"].permute(1,2,0).float()
        v=z["v_effective"].transpose(0,1).float()
        q=z["q_effective"].index_select(0,self.h)
        oracle=[]
        for qs in q.split(16):
            s=(qs.transpose(0,1).float() @ k)*(128**-0.5)
            oracle.append((s.softmax(-1)@v).transpose(0,1))
        delta=(torch.cat(oracle)-z["o_native"].index_select(0,self.h).float()).reshape(448,5120)
        downstream=delta@self.weight(site)
        if gate is not None:
            downstream=downstream*gate
        errors=[downstream]
        lookup=torch.empty(512,device=self.device,dtype=torch.long)
        lookup[self.h]=torch.arange(len(self.h),device=self.device)
        for i,j in (lookup[e].T for e in self.edges):
            errors.append(downstream[i]-downstream[j])
        return {key:float(err.square().sum(dtype=torch.float64)) for key,err in zip(ENERGIES,errors)}


def sufficient_stats(d):
    # Shared CAL scales use whole CAL training panels, never held-out sources.
    features={}
    for name,(a,s) in d["endpoints"].items():
        u=(s-a if a is not None else s).reshape(-1,64).double()
        features[name]=(u.T@u)/u.shape[0]
    return (features,float(d["innovation"].square().mean(dtype=torch.float64)),
                      float(d["mass"].square().mean(dtype=torch.float64)))


def transform_cal(stats, sources):
    matrices={}
    for name in ("finite","out_delta","out_current"):
        gram = torch.stack([torch.stack([x[0][name] for x in stats[s]]).mean(0) for s in sources]).mean(0)
        rms=gram.diag().clamp_min(0).sqrt()
        irms=torch.where(rms>0,rms.reciprocal(),0)
        eig,vec=torch.linalg.eigh(irms[:,None]*gram*irms[None,:])
        v=vec[:,-31:].flip(1)
        # The largest entry fixes each SVD column sign for replay portability.
        sign=v[v.abs().argmax(dim=0),torch.arange(31,device=v.device)].sign()
        basis=(irms[:,None]*v*sign[None,:]).float()
        matrices[name]=dict(basis=basis,rms=rms,spectrum=eig.flip(0).clamp_min(0).sqrt())
    output_rms=math.sqrt(mean(mean(x[1] for x in stats[s]) for s in sources))
    mass_rms=math.sqrt(mean(mean(x[2] for x in stats[s]) for s in sources))
    assert output_rms>1e-7 and mass_rms>1e-7,(output_rms,mass_rms)
    return matrices,output_rms,mass_rms


def design(d,matrices):
    x={}
    for k,(a,s) in d["endpoints"].items():
        basis=matrices[k]["basis"]
        if a is None:
            x[k]=apply_transform(s,basis)
        else:
            aa,ss=(u.bfloat16().float() for u in (a@basis,s@basis))
            diff=ss-aa
            x[k]=torch.cat((torch.ones_like(diff[...,:1]),diff),dim=-1)
    return x


def readout(d,x,arm,rows,lam,scale_r,scale_m,perm,floor=False,zero=False,fp64=False):
    base=d["selected"] if arm.endswith("ABS") or arm=="SP" else d["anchor"]
    if arm in ("SP","ANCH"):
        return base, {}
    dtype=torch.float64 if (floor or fp64) else torch.float32
    mu=0 if floor else MU.get(arm,0)
    X=x[TRANSFORM_KEY[arm]].to(dtype)
    c=d["c"][:,perm] if arm=="FIN-CP" else d["c"]
    c=(c*(scale_m/scale_r)).to(dtype) if arm in MU else None
    target=((d["y"]-base)/scale_r).to(dtype)
    mass=((d["rho"] if arm=="FIN-ABS" else d["mass"])/scale_m).to(dtype)
    ridge=len(rows)*lam
    Xp,tp,cp,mp=(u.index_select(1,rows) if u is not None else None for u in (X,target,c,mass))
    B,a,spec=schur_fit(Xp,tp,cp,mp,ridge,mu,floor=floor,zero_rhs=zero)
    prediction=X@B
    rec={"optimality_relative":optimality(Xp,tp,cp,mp,B,a,mu,ridge,zero),
         "floor_coupled_rank":spec}
    if a is not None:
        m_hat=X@a
        prediction=prediction-m_hat*c
        rec["m_hat"]=(m_hat*scale_m).float()
        rec["w_hat"]=(X@B*scale_r).float()
    return base+(prediction*scale_r).float(),rec


def metrics(pred,d,w,idx,base,components=None,arm=None):
    delta=(pred-d["y"]).transpose(0,1).reshape(512,5120)
    out=(delta@w)
    wanted=((d["y"]-base).transpose(0,1).reshape(512,5120) @ w)
    if d["gate"] is not None:
        out=out*d["gate"]
        wanted=wanted*d["gate"]
    e=out[idx.h]
    target=wanted[idx.h]
    def energy(t):
        return float(t.square().sum(dtype=torch.float64))
    vals=dict(point=energy(e),head=energy(delta[idx.h]),
              **{k:energy(out[p]-out[q]) for k,(p,q) in zip(("spatial","temporal"),(a.T for a in idx.edges))})
    target_energy=energy(target)
    projection=float(((e+target)*target).sum(dtype=torch.float64))
    predicted_energy=energy(e+target)
    vals.update(target_energy=target_energy,signed_projection=projection/target_energy if target_energy else None,
        orthogonal_energy=max(0,predicted_energy-projection**2/target_energy) if target_energy else None,
        prediction_target_cosine=projection/math.sqrt(target_energy*predicted_energy) if target_energy and predicted_energy else None)
    if components and "m_hat" in components:
        mass=components.pop("m_hat")[:,idx.h]
        moment=components.pop("w_hat")[:,idx.h]
        rho=mass if arm=="FIN-ABS" else mass+d["rho"][:,idx.h]-d["mass"][:,idx.h]
        err_m=rho-d["rho"][:,idx.h]
        err_w=moment-(d["w1"] if arm=="FIN-ABS" else d["w1"]-d["w0"])[:,idx.h]
        direction=d["c"][:,idx.h]
        if arm=="FIN-CP":
            # Preserve the effect of evaluating the intentionally wrong direction.
            eval_dir=d["c"][:,idx.perm[idx.h]]
            mass_direction=mass*eval_dir-(mass-err_m)*direction
        else:
            mass_direction=err_m*direction
        vals.update(rho_mse=float(err_m.square().mean(dtype=torch.float64)),
            w_mse=float(err_w.square().mean(dtype=torch.float64)),w_error_energy=energy(err_w),
            directed_mass_error_energy=energy(mass_direction),
            signed_mass_moment_cross=float(-2*(err_w*mass_direction).sum(dtype=torch.float64)),
            rho_out_of_range=float(((rho<0)|(rho>1)).float().mean()))
    return vals


def site_base(d,arm):
    return d["selected"] if arm.endswith("ABS") or arm=="SP" else d["anchor"]


def downstream_point(pred,d,w,idx):
    residual=(pred[:,idx.h]-d["y"][:,idx.h]).transpose(0,1).reshape(len(idx.h),5120) @ w
    if d["gate"] is not None:
        residual=residual*d["gate"]
    return float(residual.square().sum(dtype=torch.float64))


def select_lambda(idx,sites,means):
    jobs = defaultdict(list)
    payload = {}
    for held in CAL:
        sources=tuple(s for s in CAL if s!=held)
        maps,sr,sm=transform_cal(means,sources)
        payload[held] = {"cal_training_sources":sources,"output_rms":sr,"mass_rms":sm}
        for site in sites:
            if site.source!=held:
                continue
            d=idx.prepare(site)
            x=design(d,maps)
            w=idx.weight(site)
            noise=idx.noise[site.sidecar]
            selected=downstream_point(d["selected"],d,w,idx)
            if selected<=noise["point"]:
                continue
            for arm,lam in product(TUNED,GRID):
                pred,rec=readout(d,x,arm,idx.p,lam,sr*math.sqrt(128),sm,idx.perm)
                assert rec["optimality_relative"]<0.02
                e=downstream_point(pred,d,w,idx)
                assert math.isfinite(e)
                jobs[arm,lam,held].append(e/selected)
    scores={arm:[
        {"lambda":lam,"mean_SP_normalized_point":mean(mean(jobs[arm,lam,s]) for s in CAL),
         "source_scores":{s:mean(jobs[arm,lam,s]) for s in CAL},
         "site_counts":{s:len(jobs[arm,lam,s]) for s in CAL}}
        for lam in GRID] for arm in TUNED}
    chosen={arm:min(scores[arm],key=lambda r:(r["mean_SP_normalized_point"],r["lambda"]))["lambda"] for arm in TUNED}
    chosen.update({"FIN-C1":chosen["FIN-C0"],"FIN-CP":chosen["FIN-C0"]})
    return chosen, scores, payload


def observation_diagnostics(X,idx,C):
    p,h=X[:,idx.p].double(),X[:,idx.h].double()
    u,s,vh=torch.linalg.svd(p,full_matrices=False)
    cut=torch.finfo(s.dtype).eps*max(p.shape[-2:])*s[:,:1]
    inverse=torch.where(s>cut,s.reciprocal(),0)
    inverse_p=(vh.transpose(-1,-2)*inverse[:,None,:]) @ u.transpose(-1,-2)
    L=inverse_p.transpose(-1,-2) @ batched_outer(h) @ inverse_p
    gain=torch.linalg.eigvalsh(L)[:,-1].clamp_min(0).sqrt()
    H=torch.eye(len(idx.p),dtype=p.dtype,device=p.device)-p@inverse_p
    c=C[:,idx.p].double()
    p_t=p.transpose(-2,-1)
    schur=p_t@((H*(c@c.transpose(-2,-1)))@p)
    eig0=torch.linalg.eigvalsh((schur+schur.transpose(-2,-1))*0.5)
    eig1=torch.linalg.eigvalsh((schur+schur.transpose(-2,-1))*0.5+batched_outer(p))
    return dict(per_head_XP_singular_values=s.tolist(),rank_fp64=(s>cut).sum(-1).tolist(),
        rank_relative_1e_3=(s>1e-3*s[:,:1]).sum(-1).tolist(),gain_XH_XP_pinv=gain.tolist(),
        C0_Schur_eigenvalues=eig0.tolist(), C1_Schur_eigenvalues=eig1.tolist())


def evaluate_site(idx,site,matrices,sr,sm,chosen,transform_sha,recipe_sha):
    d=idx.prepare(site)
    x=design(d,matrices)
    w=idx.weight(site)
    rr=sr*math.sqrt(128)
    noise=idx.noise[site.sidecar]
    obs={key:observation_diagnostics(X,idx,d["c"]*(sm/rr)) for key,X in x.items()}
    sp=metrics(d["selected"],d,w,idx,d["selected"])
    results={}
    for arm in ("SP","ANCH",*TRANSFORM_KEY):
        base=site_base(d,arm)
        if arm in ("SP","ANCH"):
            results[arm]={"zero_fit_reference":sp if arm=="SP" else metrics(base,d,w,idx,base)}
            continue
        result={}
        for name,fit_rows in (("P64_estimate",idx.p),("all_H_estimator_reference",idx.h),("family_floor_head",idx.h)):
            floor=name=="family_floor_head"
            if floor and arm=="FIN-C1":
                result[name]=dict(results["FIN-C0"][name])
                continue
            pred,diag=readout(d,x,arm,fit_rows,0 if floor else chosen[arm],rr,sm,idx.perm,floor=floor)
            loss=metrics(pred,d,w,idx,base,diag,arm=arm)
            assert diag["optimality_relative"]<0.02,(site.row,arm,name,diag)
            result[name]=dict(loss,**diag)
            if name=="P64_estimate" and arm in ("OUT-D","FIN-C1") and site.row["layer"]==13:
                hi,_=readout(d,x,arm,fit_rows,chosen[arm],rr,sm,idx.perm,fp64=True)
                result[name]["fp32_vs_fp64_prediction_relative"]=float((hi-pred).norm()/hi.norm().clamp_min(1e-18))
        assert result["family_floor_head"]["head"]<=result["all_H_estimator_reference"]["head"]*(1+0.0002)+1e-7
        if arm=="FIN-C1":
            p,diag=readout(d,x,arm,idx.p,chosen[arm],rr,sm,idx.perm,zero=True)
            result["C1_zero_mass_rhs"]=dict(metrics(p,d,w,idx,base,diag,arm),**diag)
        results[arm]=result
    for levels in results.values():
        for m in levels.values():
            m.update({k+"_relative_SP":m[k]/sp[k] if sp[k]>noise[k] else None for k in ENERGIES})
    key={k:site.row[k] for k in ("source_id","source_cluster_id","module","expert","background","layer")}
    return dict(**key,capture_record_sha256=digest(site.sidecar),native_vs_recomputed_noise=noise,
        rho_roundoff=d["rho_roundoff"],
        transform_sha256=transform_sha,recipe_sha256=recipe_sha,observation=obs,arms=results)


def paired_summary(rows,a,b,metric="point",level="P64_estimate",background=None,module="self"):
    active=[r for r in rows if r["source_id"] in DEV and r["module"]==module
            and (background is None or r["background"]==background)]
    source_values={}
    counts={}
    for source in DEV:
        vals=[]
        for r in active:
            if r["source_id"]!=source:
                continue
            A=r["arms"][a]
            B=r["arms"][b]
            if metric=="head":
                ea,eb=(M.get(level,M.get("zero_fit_reference"))["head"] for M in (A,B))
            else:
                ea,eb=(M.get(level,M.get("zero_fit_reference"))[metric] for M in (A,B))
                if min(ea,eb)<=r["native_vs_recomputed_noise"][metric]:
                    continue
            if min(ea,eb)>0:
                vals.append(math.log(ea/eb))
        counts[source]=len(vals)
        source_values[source]=math.exp(mean(vals)) if vals else None
    values=[v for v in source_values.values() if v is not None]
    return dict(ratio=math.exp(mean(map(math.log,values))) if values else None,
                source_ratios=source_values,covered_sites=counts,
                improved_sources=sum(v<1 for v in values))


def summarize(rows):
    contrasts={}
    pairs=(("FIN-D","OUT-D"),("FIN-C1","FIN-C0"),("FIN-C1","FIN-CP"),("FIN-C1","FIN-ABS"),
           ("FIN-C1","ANCH"),("ANCH","SP"),("OUT-D","OUT-ABS"))
    for module,background in product(("self","text_cross"),(None,"Dense","wide8192")):
        key=module+"/"+(background or "both")
        contrasts[key]={}
        for a,b in pairs:
            levels={m:paired_summary(rows,a,b,m,background=background,module=module) for m in ENERGIES}
            levels["head_floor"]=paired_summary(rows,a,b,"head","family_floor_head",background,module)
            contrasts[key][a+"/"+b]=levels
    out=contrasts["self/both"]["OUT-D/OUT-ABS"]["point"]["ratio"]
    assert out is not None
    reference="OUT-D" if out<=1 else "OUT-ABS"
    comparison={bg:{metric:paired_summary(rows,"FIN-C1",reference,metric,background=bg)
                    for metric in ENERGIES} for bg in ("Dense","wide8192")}
    passes = all(
        metric["point"]["ratio"] is not None and metric["point"]["ratio"]<=0.90
        and metric["point"]["improved_sources"]>=2
        and all(metric[m]["ratio"] is not None
                and not (metric[m]["ratio"]>1 and metric[m]["improved_sources"]<=1)
                for m in ("spatial","temporal"))
        for metric in comparison.values())
    full_cross={}
    for bg in ("Dense","wide8192"):
        pairs_site=[r for r in rows if r["source_id"] in DEV and r["module"]=="self" and r["background"]==bg]
        full_cross[bg]={
            "FIN-C1/zero_mass_rhs_point":math.exp(mean(
                math.log(r["arms"]["FIN-C1"]["P64_estimate"]["point"]/r["arms"]["FIN-C1"]["C1_zero_mass_rhs"]["point"])
                for r in pairs_site)),
            "FIN-C1_mean_rho_out_of_range":mean(r["arms"]["FIN-C1"]["P64_estimate"]["rho_out_of_range"] for r in pairs_site)}
    return dict(reference=reference,candidate="FIN-C1",candidate_vs_reference=comparison,
                engineering_gate_pass=passes,contrasts=contrasts,mechanistic_diagnostics=full_cross,
                status="complete",sites=len(rows))


@torch.inference_mode()
def main(panel_root,frozen,output,raw_index,mode):
    import numpy as np
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    raw=read(raw_index)
    idx=Assets(panel_root,frozen,"cuda:0")
    idx.raw={k:Path(v) for k,v in raw.items()}
    assert set(idx.raw)==set(CAL+DEV)
    output.mkdir(exist_ok=False,parents=True)
    idx.noise={}
    records=[]
    recipes={}
    site_rows=output/"site_metrics.jsonl"
    with site_rows.open("w",encoding="utf8") as stream:
        for group,sites in sorted(idx.groups.items()):
            if mode=="sentinel":
                sites=sites[:3]
                for site in sites:
                    d=idx.prepare(site)
                    idx.noise[site.sidecar]=idx.native_noise(site,d["gate"])
                    key=site.source
                    stats={key:[sufficient_stats(d)]}
                    mats,sr,sm=transform_cal(stats,(key,))
                    all_lam={a:0.01 for a in TRANSFORM_KEY}
                    row=evaluate_site(idx,site,mats,sr,sm,all_lam,"SENTINEL","SENTINEL")
                    stream.write(json.dumps(row,allow_nan=False)+"\n")
                    stream.flush()
                break
            tag="_".join(group)
            print("START_GROUP",tag,flush=True)
            stats=defaultdict(list)
            for site in sites:
                d=idx.prepare(site)
                idx.noise[site.sidecar]=idx.native_noise(site,d["gate"])
                if site.source in CAL:
                    stats[site.source].append(sufficient_stats(d))
            print("NATIVE_NOISE",tag,flush=True)
            lam,cv,folds=select_lambda(idx,sites,stats)
            matrices,sr,sm=transform_cal(stats,CAL)
            transform=output/(tag+"_transform.npz")
            np.savez(transform,cal_sources=np.asarray(CAL),output_rms=sr,mass_rms=sm,
                     **{f"{fam}_{k}":v.cpu().numpy() for fam,parts in matrices.items() for k,v in parts.items()})
            transform_sha=digest(transform)
            recipe=output/(tag+"_recipe.json")
            recipes[tag]=dict(transform_sha256=transform_sha,CAL=CAL,held_sources=DEV,
                CAL_scales=dict(s_R=sr,s_m=sm),CAL_LOSO=folds,lambda_values=lam,lambda_scores=cv,
                fitting_precision="float32; layer13 paired FP64 audit for OUT-D/FIN-C1; floors FP64",
                score_selection="mean site SP-normalized H448 downstream point error; source equal",
                T_rounding="separate BF16 endpoints then FP32 difference; OUT-ABS current")
            save(recipe,recipes[tag])
            print("CAL_FROZEN",tag,json.dumps(lam),flush=True)
            for n,site in enumerate(sites,1):
                row=evaluate_site(idx,site,matrices,sr,sm,lam,transform_sha,digest(recipe))
                records.append(row)
                stream.write(json.dumps(row,allow_nan=False)+"\n")
                stream.flush()
                if n%8==0:
                    print("FITTED",tag,n,flush=True)
            idx.weights.clear()
    result=summarize(records) if mode=="full" else dict(status="sentinel_only",sites=sum(1 for _ in site_rows.open()))
    result.update(mode=mode,panel_results_sha256=digest(panel_root/"RESULTS.sha256"),producer_sha256=digest(Path(__file__)))
    if mode=="full":
        assert result["sites"]==192
    save(output/"FIT_RESULT.json",result)
    print("FIT_COMPLETE",result["status"],result["sites"],flush=True)


if __name__=="__main__":
    main(*(Path(s).resolve() for s in sys.argv[1:5]),sys.argv[5])
