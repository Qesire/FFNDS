"""Four low-attention arms replayed on a fixed native-attention wide8192 Wan trajectory."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import inspect
from pathlib import Path
import sys

import numpy as np
import torch

from attention_joint_action import JointTail, gram
from attention_mass_centered_low import moments, encode
from attention_online_structured import AUDIT_LAYERS, BACKEND, HEADS, HEAD_DIM, N, SOURCES_SHA, OnlineAttention, native_flash, schur_fit
from attention_paired_value_ops import paired_map, pfrr
from generate import Experiment, digest, equal_trace, load, save


PLAN = "9dec5f1baf85c089b65dcbc142d0115fdf2308ae"
DEV = ("vbi_0392","vbi_0800","vbi_1013")
ARMS = ("A_PFRR","B_diagonal","C_paired","D_shuffled")


class PairReplay(OnlineAttention):
    def __init__(self, experiment, fit, config):
        super().__init__(experiment,fit)
        self.ridge=float(config["value_ridge"])
        assert self.ridge==0.001 and (config["fit_rows"],config["held_value_rows"])==(256,64)
        rng=np.random.default_rng(config["value_seed"])
        positions=rng.choice(N,size=320,replace=False)
        permutation=rng.permutation(256)
        while np.any(permutation==np.arange(256)):
            permutation=rng.permutation(256)
        self.fit_rows,self.held_rows,self.shuffle=[torch.as_tensor(x,device="cuda",dtype=torch.long) for x in (positions[:256],positions[256:],permutation)]
        self.panels=self.panel_ids
        self.p=self.panel_rows
        rest=torch.ones(len(self.panels),dtype=torch.bool,device=self.p.device)
        rest[self.p]=False
        self.h=rest.nonzero().flatten()

    def anchor(self,q,k,v):
        native,lse=native_flash(q,k,v)
        support,indices=self.route(q,k)
        sparse,small_lse,_=self.selected(q,k,v,support,False)
        full_rho,_=self.anchor_mass(q,k,support,small_lse,lse[0])
        rho=full_rho.index_select(1,self.panels)[...,None]
        mu,_,inv=moments(v)
        a=native[0].index_select(1,self.panels)
        selected=sparse.index_select(1,self.panels)
        z=encode(a,selected,rho,mu,inv)
        cal=self.fits["low"][0]["out_delta"]["basis"]
        projected=((selected.float() @ self.pv) @ cal).bfloat16().float()
        def samples(rows):
            v_at=v.index_select(0,rows).transpose(0,1).float()
            return ((v_at-mu)*inv).bfloat16().contiguous()
        return native[0].transpose(0,1).reshape(N,-1).contiguous(), (support,rho,z,projected,samples(self.fit_rows),samples(self.held_rows))

    def hit(self,q,k,v,native,state):
        support,rho,z,anchor_design,u,u_h=state
        mat,lam,scale_r,_=self.fits["low"]
        selected,_,_=self.selected(q,k,v,support,False)
        s=selected.index_select(1,self.panels).float()
        t=native.reshape_as(q).index_select(0,self.panels).transpose(0,1).contiguous()
        t_p=t.index_select(1,self.p)
        mu,scale,_=moments(v)
        y=v.index_select(0,self.fit_rows).transpose(0,1).float()-mu
        y_h=v.index_select(0,self.held_rows).transpose(0,1).float()-mu
        u,u_h=u.float(),u_h.float()
        map_c,system=paired_map(u,y,scale,self.ridge)
        map_d,_=paired_map(u,y.index_select(1,self.shuffle),scale,self.ridge)
        value_errors=(u_h*scale-y_h, u_h@map_c-y_h, u_h@map_d-y_h)
        condition=torch.linalg.eigvalsh(system.double())
        raw_features=((s @ self.pv) @ mat["out_delta"]["basis"]).bfloat16().float()-anchor_design
        x=torch.cat((torch.ones_like(raw_features[...,:1]),raw_features),-1)
        xp=x.index_select(1,self.p)
        baseline=(1-rho)*s+rho*mu
        outputs={ARMS[0]:pfrr(s.double(),t_p.double(),self.p).to(t.dtype)}
        transport_bases=(baseline+z.float()*scale,baseline+z.float()@map_c,baseline+z.float()@map_d)
        for name,base in zip(ARMS[1:],transport_bases,strict=True):
            label=(t_p.float()-base.index_select(1,self.p))/scale_r
            coef=schur_fit(xp,label,None,None,len(self.p)*lam["OUT-D"],mu=0)[0]
            out=(base+(x@coef)*scale_r).to(t.dtype)
            out.index_copy_(1,self.p,t_p)
            outputs[name]=out
        assert all(torch.equal(out.index_select(1,self.p),t_p) for out in outputs.values())
        held=lambda x:x.index_select(1,self.h).float()
        y_h_norm=max(float(y_h.double().square().sum()),1e-30)
        return dict(native=t,**outputs), dict(value_held_gram=gram(*value_errors), held_value_target_energy=y_h_norm,
           held_attention_gram=gram(*(held(out)-held(t) for out in outputs.values())),
           held_base_gram=gram(*(held(out)-held(t) for out in (s,*transport_bases))),
           ridge_system_max_condition=float((condition[:,-1]/condition[:,0]).amax()),
           fixed_attention_probes_exact=True)


class PairTail(JointTail):
    arms = ARMS

    def __init__(self, experiment, wide, online):
        super().__init__(experiment,wide,online)
        self.edges={k:torch.tensor(online.panel[f"{k}_edges_rows"],device="cuda").T for k in ("spatial","temporal")}
        assert all(edge.shape==(2,224) and torch.isin(edge,self.h).all() for edge in self.edges.values())


@contextmanager
def instrument(experiment,method,tail):
    from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight

    owner=experiment.owner
    original_block,original_self,original_apply=owner.infer_block,owner.infer_self_attn,TorchSDPAWeight.apply
    assert digest(Path(inspect.getfile(original_self)))==experiment.base["native_transformer_sha256"]
    assert digest(Path(inspect.getfile(original_apply)))==BACKEND
    sig=inspect.signature(original_apply)
    active=None
    slots,metrics={},{}

    @wraps(original_block)
    def block(model,weights,x,pre):
        step,layer=int(model.scheduler.step_index),int(model.block_idx)
        audit=step in (2,3) and layer in AUDIT_LAYERS
        if not audit:
            return original_block(model,weights,x,pre)
        panel=x.index_select(0,tail.ids) if step==3 else None
        with tail.capture_native_probe(weights) if step==3 else nullcontext():
            value=original_block(model,weights,x,pre)
        if step==2:
            tail.anchor(model)
        else:
            metrics[layer]["tail"]=tail.replay(model,weights,pre,panel,value)
            tail_metrics=metrics[layer]["tail"]
            assert tail_metrics["shared_probe_vs_live_relative_l2"]<1e-4
        return value

    @wraps(original_self)
    def self_attention(model,phase,*args,**kwargs):
        nonlocal active
        step,layer=int(model.scheduler.step_index),int(model.block_idx)
        if step not in (2,3) or layer not in AUDIT_LAYERS:
            return original_self(model,phase,*args,**kwargs)
        assert active is None and not model.config["seq_parallel"] and not model.config["use_image_encoder"]
        active=(step,layer,phase.self_attn_1)
        try:
            return original_self(model,phase,*args,**kwargs)
        finally:
            active=None

    @wraps(original_apply)
    def apply(op,*args,**kwargs):
        if active is None:
            return original_apply(op,*args,**kwargs)
        step,layer,expected=active
        assert op is expected
        call=sig.bind(op,*args,**kwargs)
        call.apply_defaults()
        p=call.arguments
        q,k,v=(p[key] for key in ("q","k","v"))
        assert all(t.dtype==torch.bfloat16 and tuple(t.shape)==(N,HEADS,HEAD_DIM) for t in (q,k,v))
        assert p["attn_mask"] is None and p["max_seqlen_q"]==p["max_seqlen_kv"]==N and not p["causal"] and not p["drop_rate"]
        if step==2:
            value,slots[layer]=method.anchor(q,k,v)
            return value
        native=original_apply(op,*args,**kwargs)
        tail.branches,metrics[layer]=method.hit(q,k,v,native,slots.pop(layer))
        return native

    owner.infer_block,owner.infer_self_attn,TorchSDPAWeight.apply=block,self_attention,apply
    try:
        yield metrics
        assert not slots and not tail.cached and len(metrics)==4
    finally:
        owner.infer_block,owner.infer_self_attn,TorchSDPAWeight.apply=original_block,original_self,original_apply


@torch.inference_mode()
def main(fit,config_path):
    conf=load(config_path)
    assert conf["plan_commit"]==PLAN and tuple(conf["arms"])==ARMS
    assert digest("sources_locked.json")==SOURCES_SHA
    known={row["source_id"]:row for row in load("sources_locked.json")["sources"]}
    torch.backends.cuda.matmul.allow_tf32=False
    exp=Experiment()
    online=PairReplay(exp,fit,conf)
    wide=exp.build("wide8192")
    path=exp.output/"PAIRED_VALUE_RESULT.json"
    result=dict(status="running",job=exp.job,gpu=exp.state["gpu"],plan_commit=PLAN,source_sha256=digest(__file__),
                arms=ARMS,coverage="equal supplied probes",fold="DEV native attention plus BF16 wide8192",
                additional_native_attention_queries=0,records={},value_fit_rows=online.fit_rows.tolist(),
                value_held_rows=online.held_rows.tolist(),shuffled_current_order=online.shuffle.tolist(),
                ridge=online.ridge)
    for source in DEV:
        archive_path=Path(conf["controls"][source]); archive=load(archive_path)
        assert archive["status"]=="complete" and archive["source"]["record_id"]==source
        assert archive["native_sha256"]==exp.base["native_transformer_sha256"] and archive["backend_sha256"]==BACKEND
        assert all(archive["source"][k]==known[source][k]==exp.rows[source][k] for k in ("seed","prompt_sha256","image_sha256","source_cluster_id"))
        old=archive["backgrounds"]["wide8192"]["record"]
        tail=PairTail(exp,wide,online)
        with instrument(exp,online,tail) as sites:
            track=exp.render(source,"paired_fixed",wide)
        assert equal_trace(track,old),(source,"native wide8192 replay")
        result["records"][source]=dict(control_index_sha256=digest(archive_path),trace_exact=True,track=track,sites=sites)
        save(path,result)
        print("PAIRED_VALUE_FIXED_VERIFIED",source,flush=True)
    result["status"]="complete"
    save(path,result)


if __name__=="__main__":
    main(*(Path(p).resolve() for p in sys.argv[1:]))
