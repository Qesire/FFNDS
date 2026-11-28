"""Layer-39 teacher-forced joint attention/FFN fork on archived DEV trajectories."""
from contextlib import contextmanager
from functools import wraps
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from attention_panel_fit import Assets, design, readout
from generate import Experiment, digest, equal_trace, load, save


PLAN = "29639e148244e19752c0acbe40ed0928291a8263"
SOURCES = ("vbi_0392","vbi_1013")
ARMS = {"high":"OUT-ABS","low":"FIN-C0"}
FFN_NAMES = ("u0","y0","z0k")
TARGETS = ("B00","B10","B01","B11_fixed","B11_refit","cheap")


def rms_rel(a,b):
    return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-16))


def tensor_file(path, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".pending")
    save_file({name:v.detach().to("cpu",copy=True).contiguous() for name,v in values.items()},str(tmp))
    tmp.replace(path)
    return dict(path=path.as_posix(),sha256=digest(path),bytes=path.stat().st_size)


def native_ffn(u, phase):
    return phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(u),approximate="tanh"))


def ffn_operands(u, cached, t):
    # Match the production wide8192 BF16/FP32 graph and sum order.
    from exact_port import apply, lookup_secant

    u = u.contiguous()
    du = torch.empty_like(u,dtype=torch.float32)
    dub = torch.empty_like(u)
    apply(t["difference"],du.numel(),u,cached["u0"],dub,du,du.numel())
    linear = dub @ t["A0"]
    ju = du @ t["jmap"]
    z = F.linear(u,t["w1k"],t["b1k"])
    dh,dh32,x = lookup_secant(z,cached["z0k"],t["kernel_beta"],t["kernel_scale"],*t["lut"])
    q = (dh32 @ t["zmap"]).sub_(ju)
    base = torch.mm(dh,t["w2k"],out_dtype=torch.float32)
    apply(t["base"],base.numel(),cached["y0"],linear,base,base.numel())
    return base,q,x


def ffn_fit(operands, label, t, inv_metric):
    base, q, x = operands
    k = t["zmap"].shape[0]
    yd, qd, vd = (label.float()-base).double(),q.double(),t["vd"]
    system = ((x @ x.T)/k).double()+inv_metric
    chol = torch.linalg.cholesky(system)
    iq = torch.cholesky_solve(qd,chol)
    gain = torch.linalg.solve((qd.T @ iq)*t["vv"],(iq*(yd @ vd.T)).sum(0))
    coeff = torch.cholesky_solve(yd-(qd*gain) @ vd,chol)
    weights = torch.cat((gain.float()[:,None]*t["v"],0.5*coeff.float()))
    assert bool(torch.isfinite(weights).all())
    return x,weights


def ffn_predict(operands, state, k):
    base,q,x = operands
    xp,weights = state
    return torch.addmm(base,torch.cat((q,(x @ xp.T)/k),dim=1),weights).bfloat16()


def energy(e):
    return float(e.double().square().sum())


def deltas(arms):
    # Convert before differencing to prevent BF16 rounding in the analysis.
    a,b,c,d = (arms[k].double() for k in TARGETS[:4])
    return (b-a,c-a,d-b-c+a,d-a)


def joint_summary(arms, indices, edges):
    E,F_,interaction,total = deltas(arms)
    assert rms_rel(E+F_+interaction,total)<2e-7
    names=("E_A","E_F","I_AF")

    def summary(rows):
        v = [x[rows] for x in (E,F_,interaction,total)]
        return {"terms":names,"gram":[[float((a*b).sum()) for b in v[:3]] for a in v[:3]],
                "E_A_energy":energy(v[0]),"E_F_energy":energy(v[1]),"I_AF_energy":energy(v[2]),
                "B11_error":energy(v[3])}

    raw=summary(indices)
    raw["edge_energies"]={}
    for name,edge in zip(("spatial","temporal"),edges):
        i,j=edge.T
        raw["edge_energies"][name]={k:energy(x[i]-x[j]) for k,x in zip((*names,"total"),(E,F_,interaction,total))}
    return raw


def ridge_flow(y0, truth, phi, p, ridge_per_query=0.1):
    # Scale only with current cheap P64 features. No H448 labels enter the fit.
    z=phi.double()
    zp=z[p]
    scale=zp.square().mean(0).sqrt()
    iz=torch.where(scale>0,scale.reciprocal(),0)
    z=z*iz
    zp=z[p]
    teacher=truth[p].double()-y0[p].double()
    normal=zp.T@zp + (len(p)*ridge_per_query)*torch.eye(z.shape[1],dtype=z.dtype,device=z.device)
    weights=torch.linalg.solve(normal,zp.T@teacher)
    y=y0.double()+z@weights
    y[p]=truth[p].double()
    return y.float(),{"columns":z.shape[1],"rank":int(torch.linalg.matrix_rank(zp)),
        "ridge_per_query":ridge_per_query,"probe_rows":len(p),"weight_norm":float(weights.norm())}


def score_flow(pred, teacher, assets):
    e=(pred.double()-teacher.double())
    out={"point":energy(e[assets.h])}
    for key,edge in zip(("spatial","temporal"),assets.edges):
        i,j=edge.T
        out[key]=energy(e[i]-e[j])
    return out


class Layer39:
    def __init__(self,exp,assets,panel_ids,fit,source,background,port,dest):
        self.exp,self.assets,self.dest=exp,assets,dest
        self.p,self.h=assets.p,assets.h
        self.ids=panel_ids
        self.source,self.background,self.port=source,background,port
        self.raw_manifest=load(exp.base["cal_manifest"]["path"])
        assert digest(exp.base["cal_manifest"]["path"])==exp.base["cal_manifest"]["sha256"]
        assert self.ids[self.p].tolist()==exp.base["observer_ids"]
        self.idx={s.row["expert"]:s for key,rows in assets.groups.items() if key[0]=="self" for s in rows
                  if s.source==source and s.row["background"]==background and s.row["layer"]==39}
        assert set(self.idx)=={"high","low"}
        self.fits={}
        for e in ("high","low"):
            tag=f"self_{e}"
            mapfile=fit/(tag+"_transform.npz")
            recipe=load(fit/(tag+"_recipe.json"))
            assert digest(mapfile)==recipe["transform_sha256"]
            with np.load(mapfile,allow_pickle=False) as item:
                maps={k:{"basis":torch.tensor(item[k+"_basis"],device=self.ids.device)}
                      for k in ("finite","out_current","out_delta")}
                self.fits[e]=(maps,recipe,float(item["output_rms"]),float(item["mass_rms"]))
        from native_joint import probe_metric_inverse
        probes=np.asarray(exp.base["observer_ids"],dtype=np.int64)
        self.inv_metric,_=probe_metric_inverse(probes,np.column_stack(np.unravel_index(probes,(21,29,52))),self.ids.device)
        self.active=None
        self.waiting=None
        self.cached={}
        self.rows={}

    @contextmanager
    def tap_dense_anchor(self,phase,selector):
        old_w1,old_w2=phase.ffn_0.apply,phase.ffn_2.apply
        saved={}

        def w1(x):
            raw=old_w1(x)
            saved["u0"]=x[self.ids].contiguous()
            saved["z0k"]=raw[self.ids].index_select(1,selector).contiguous()
            return raw

        def w2(x):
            out=old_w2(x)
            saved["y0"]=out[self.ids].contiguous()
            return out

        phase.ffn_0.apply,phase.ffn_2.apply=w1,w2
        try:
            yield saved
        finally:
            phase.ffn_0.apply,phase.ffn_2.apply=old_w1,old_w2

    @contextmanager
    def bind(self):
        owner=self.exp.owner
        orig={k:getattr(owner,k) for k in ("infer_block","infer_self_attn","infer_non_blocks",
            "infer_cross_attn","infer_ffn","post_process")}
        assert all(digest(inspect.getfile(f))==self.exp.base["native_transformer_sha256"] for f in orig.values())
        self.orig=orig

        @wraps(orig["infer_self_attn"])
        def self_attn(model,*args,**kwargs):
            result=orig["infer_self_attn"](model,*args,**kwargs)
            if self.active is not None:
                assert "true_A" not in self.active
                self.active["true_A"]=result[self.ids].contiguous()
            return result

        @wraps(orig["infer_block"])
        def block(model,current,x,pre):
            if int(model.block_idx)!=39:
                return orig["infer_block"](model,current,x,pre)
            return self.on_block(model,current,x,pre)

        @wraps(orig["infer_non_blocks"])
        def head(model,weights,x,e):
            real=orig["infer_non_blocks"](model,weights,x,e)
            assert real.shape==(31668,64)
            info=self.waiting
            assert info is not None and int(model.scheduler.step_index)==info["step"]
            self.waiting=None
            if "Y" in info:
                flow={k:orig["infer_non_blocks"](model,weights,y.clone(),e) for k,y in info["Y"].items()}
                self.finish(info,flow,real[self.ids].contiguous())
            return real

        owner.infer_self_attn,owner.infer_block,owner.infer_non_blocks=self_attn,block,head
        try:
            yield
        finally:
            owner.infer_self_attn,owner.infer_block,owner.infer_non_blocks=(orig[k] for k in ("infer_self_attn","infer_block","infer_non_blocks"))

    def on_block(self,model,block,x,pre):
        from exact_port import prepare
        assert self.active is None and self.waiting is None
        step=int(model.scheduler.step_index)
        expert="high" if step<2 else "low"
        m,c,f=block.compute_phases[:3]
        assert model.config["use_image_encoder"] is False and model.config["seq_parallel"] is False
        assert not model.has_post_adapter and not hasattr(f,"after_proj")
        assert not hasattr(m,"before_proj") or m.before_proj.weight is None
        assert not hasattr(f,"smooth_norm2_weight") and not model._use_mxfp8_quant_fuse()
        assert x.shape==(31668,5120) and x.dtype==torch.bfloat16
        info={"step":step,"expert":expert,"X":x[self.ids].contiguous()}
        self.active=info
        try:
            if step%2==0 and self.port is None:
                key=expert+"39"
                raw=self.raw_manifest["sites"][key]
                assert digest(raw["path"])==raw["sha256"]
                pack=torch.load(raw["path"],weights_only=True,map_location="cpu",mmap=True)
                prepared=prepare(f,pack,self.ids.device)
                with self.tap_dense_anchor(f,pack["sel"].to(self.ids.device)) as cached:
                    result=self.orig["infer_block"](model,block,x,pre)
                assert set(cached)==set(FFN_NAMES)
                self.cached[expert]=(cached,prepared)
            else:
                result=self.orig["infer_block"](model,block,x,pre)
                if step%2==0:
                    inner=self.port.inner
                    assert inner.slots.owner[39]==expert
                    cached={k:getattr(inner.slots,k)[39][self.ids].contiguous() for k in FFN_NAMES}
                    self.cached[expert]=(cached,inner.weights[expert,39])
        finally:
            self.active=None
        assert "true_A" in info
        if step%2:
            info["live_Y"]=result[self.ids].contiguous()
            self.replay(info,model,m,c,f,pre)
        self.waiting=info
        return result

    def replay(self,info,model,m,c,f,pre):
        expert=info["expert"]
        site=self.idx[expert]
        attn=self.assets.prepare(site)
        maps,recipe,sr,sm=self.fits[expert]
        XX=design(attn,maps)
        arm=ARMS[expert]
        pred,_=readout(attn,XX,arm,self.p,recipe["lambda_values"][arm],sr*math.sqrt(128),sm,self.assets.perm)
        candidate=pred.transpose(0,1).reshape(512,5120).to(info["true_A"].dtype).contiguous()
        truth=attn["y"].transpose(0,1).reshape(512,5120).to(candidate.dtype).contiguous()
        a0=(attn["selected"] if arm=="OUT-ABS" else attn["anchor"]).transpose(0,1).reshape(512,5120).to(candidate.dtype).contiguous()
        native=info["true_A"]
        projected=m.self_attn_o.apply(truth)
        info["attention_projection_noise_relative"]=rms_rel(projected,native)
        assert info["attention_projection_noise_relative"]<0.01
        independent=m.self_attn_o.apply(candidate)
        cheap=m.self_attn_o.apply(a0)
        shift,scale,ga,sf,sc,gf=model.pre_process(m.modulation,pre.embed0)
        assert ga.numel()==gf.numel()==5120
        assert model.sensitive_layer_dtype==model.infer_dtype==torch.bfloat16

        tail={}
        for key,A in (("native",native),("independent",independent),("cheap",cheap)):
            X=info["X"].clone()
            R,cross=self.orig["infer_cross_attn"](model,c,X,pre.context,A,ga)
            T=R.clone().add_(cross)
            u=model.modulate_func(f.norm2.apply(T),scale=sc,shift=sf).squeeze().contiguous()
            assert u.dtype==torch.bfloat16 and u.shape==(512,5120)
            tail[key]=dict(R=R,T=T,U=u)
        n,a,b=(tail[k] for k in ("native","independent","cheap"))
        ynative,ycandidate=(native_ffn(t["U"],f) for t in (n,a))
        cached,prepared=self.cached.pop(expert)
        subset={k:v[self.p].contiguous() for k,v in cached.items()}
        def obs(u):
            return ffn_operands(u[self.p],subset,prepared)
        def pred_operands(u):
            return ffn_operands(u,cached,prepared)
        weights=ffn_fit(obs(n["U"]),ynative[self.p],prepared,self.inv_metric)
        alt_weights=ffn_fit(obs(a["U"]),ycandidate[self.p],prepared,self.inv_metric)
        n_op,a_op,b_op=(pred_operands(t["U"]) for t in (n,a,b))
        k=prepared["zmap"].shape[0]
        orig_post=self.orig["post_process"]
        cases={"B00":(n,ynative),"B10":(a,ycandidate),
            "B01":(n,ffn_predict(n_op,weights,k)),
            "B11_fixed":(a,ffn_predict(a_op,weights,k)),
            "B11_refit":(a,ffn_predict(a_op,alt_weights,k)),
            "cheap":(b,b_op[0].bfloat16())}
        y={label:orig_post(model,t["T"].clone(),ffn,gf,pre) for label,(t,ffn) in cases.items()}
        if self.background=="wide8192":
            label="B01"
        else:
            label="B00"
        e=y[label].double()-info["live_Y"].double()
        info["real_block_error_relative"]=rms_rel(y[label][self.h],info["live_Y"][self.h])
        info["real_block_error_energy"]=energy(e[self.h])
        assert info["real_block_error_relative"]<0.025,(self.background,expert,info["real_block_error_relative"])
        info["phi"] = torch.cat((XX["out_current" if expert=="high" else "finite"][...,1:].mean(0),b_op[1]),dim=-1)
        assert info["phi"].shape==(512,47)
        info["phi"] = torch.cat((torch.ones_like(info["phi"][:,:1]),info["phi"]),dim=-1)
        info["Y"]=y
        info["tail"]=tail
        info["attention_arm"]=arm
        info["gates"]=(ga.squeeze(),gf.squeeze())
        info["projection_inputs"]=(truth,candidate,a0)
        info["projected_A"]=(native,independent,cheap)
        info["FFN"]= {k:v[1] for k,v in cases.items()}

    def finish(self,info,flow,real):
        p,h=self.p,self.h
        teacher=flow["B00"]
        y=info["Y"]
        projected_error=score_flow(flow["B00"] if self.background=="Dense" else flow["B01"],real,self.assets)
        reference_rel=rms_rel((flow["B00"] if self.background=="Dense" else flow["B01"])[h],real[h])
        assert reference_rel<0.04,(self.source,self.background,info["step"],reference_rel)
        joint,joint_config=ridge_flow(flow["cheap"],teacher,info["phi"],p)
        independent=flow["B11_fixed"].clone()
        independent[p]=teacher[p]
        assert torch.equal(independent[p],teacher[p]) and torch.equal(joint[p],teacher[p].float())
        comparisons={k:score_flow(v,teacher,self.assets) for k,v in flow.items()}
        comparisons["joint_flow"]=score_flow(joint,teacher,self.assets)
        comparisons["independent_flow"]=score_flow(independent,teacher,self.assets)
        residuals={}
        for k,v in info["tail"].items():
            residuals[k+"_R"],residuals[k+"_T"],residuals[k+"_U"]=(v[n] for n in ("R","T","U"))
        layer=info["expert"]
        file=self.dest/(layer+".safetensors")
        vecs={"X_P64_H448":info["X"],"probe_rows":p,"held_rows":h,"joint_phi":info["phi"],
              "attn_gate":info["gates"][0],"ffn_gate":info["gates"][1],"native_live_Y":info["live_Y"],
              "native_live_flow":real,"flow_joint":joint,"flow_independent":independent,**residuals}
        for key,x in (("O",info["projection_inputs"]),("A",info["projected_A"])):
            vecs.update({f"{key}_{k}":v for k,v in zip(("native","candidate","cheap"),x)})
        for stage,tensors in (("block",y),("flow",flow),("ffn",info["FFN"])):
            vecs.update({f"{stage}_{key}":v for key,v in tensors.items()})
        artifact=tensor_file(file,vecs)
        assert layer not in self.rows
        self.rows[layer]=dict(step=info["step"],expert=layer,attention_arm=info["attention_arm"],
            attention_projection_noise_relative=info["attention_projection_noise_relative"],
            real_block_error_relative=info["real_block_error_relative"],real_block_error_energy=info["real_block_error_energy"],
            live_flow_panel_replay_relative=reference_rel,live_flow_panel_error=projected_error,
            flow_scores=comparisons,joint_config=joint_config,
            branch_gram={name:joint_summary(arms,h,self.assets.edges) for name,arms in (("block_exit",y),("flow",flow))},
            prefix_response={k:energy((info["tail"]["independent"][k].double()-info["tail"]["native"][k].double())[h]) for k in ("R","T","U")},
            feedback_refit_delta_energy=energy((flow["B11_refit"].double()-flow["B11_fixed"].double())[h]),
            asset=artifact,input_x_fingerprint=self.exp.v2.finite_tensor_sha(info["X"],torch)["sha256"])
        print("FORK",self.source,self.background,layer,json.dumps({k:round(v["point"],6) for k,v in comparisons.items()}),flush=True)


@torch.inference_mode()
def main(panel_root,frozen,fit,raw_path,mode):
    assets=Assets(panel_root,frozen,"cuda:0")
    panel=load(frozen/"attention_panel_index.json")
    ids=torch.tensor(panel["token_ids"],device="cuda:0")
    raw=load(raw_path)
    assert set(raw)==set(SOURCES)
    chosen=SOURCES[:1] if mode=="sentinel" else SOURCES[1:]
    # The sealed school release selects the variant at build(), not at construction.
    exp=Experiment()
    index=exp.output/"JOINT39_RESULT.json"
    receipt=dict(status="running",plan_commit=PLAN,gpu=exp.state["gpu"],job=exp.job,mode=mode,
        calibration=exp.base["cal_manifest"],panel_results_sha256=digest(panel_root/"RESULTS.sha256"),
        prior_fit_sha256=digest(fit/"FIT_RESULT.json"),sources=list(chosen),results={},
        frozen_ffn_fit="same P64 labels from native attention; fixed across four arms",
        refit="separate diagnostic fits FFN on unoverwritten approximate attention probes",
        standalone_joint="SP/ANCH attention base + folded FFN base -> head -> 48 column P64 readout")
    save(index,receipt)
    for background in ("Dense","wide8192"):
        method=exp.build("wide8192") if background=="wide8192" else None
        for source in chosen:
            base=load(Path(raw[source]))["backgrounds"][background]["record"]
            tap=Layer39(exp,assets,ids,fit,source,background,method,exp.output/"forks"/source/background)
            with tap.bind():
                actual=exp.render(source,background,method=method)
            assert set(tap.rows)=={"high","low"} and tap.waiting is None
            assert equal_trace(actual,base),(source,background)
            if method is not None:
                c=actual["counters"]
                assert (c["variant"],c["tile_rows"],c["anchors"],c["hits"],c["anchor_fallback"])==("wide8192",8192,80,80,0)
            receipt["results"][background]=dict(source=source,trace_matches_archive=True,profiled_elapsed_s=actual["wall"],experts=tap.rows)
            save(index,receipt)
    receipt["status"]="complete"
    save(index,receipt)


if __name__=="__main__":
    main(*(Path(x).resolve() for x in sys.argv[1:5]),sys.argv[5])
