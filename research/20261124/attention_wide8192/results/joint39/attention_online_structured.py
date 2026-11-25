"""All-layer HHLL attention candidates on the unmodified wide8192 FFN path."""
from collections import Counter
from contextlib import contextmanager, nullcontext
from functools import wraps
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention

from attention_panel_fit import Assets, apply_transform, design, readout, schur_fit
from generate import Experiment, digest, equal_trace, load, save


PLAN = "d9cb9952d0daa939727c6308c742e4dcb6089918"
BACKEND = "ae87ba65009c454f62e0e558cf821ee2bada1420928f408e5a7e26760984cf7d"
SOURCES_SHA = "61c142597d1b3d4455feaa7fc4079bbb7cc7621d3e015685fd4a690c5d7f53fa"
SOURCES = ("vbi_0392", "vbi_1013")
N, HEADS, HEAD_DIM, BLOCK, TOP = 31668, 40, 128, 128, 31
TILES = (N + BLOCK - 1) // BLOCK
SCALE = HEAD_DIM ** -0.5
AUDIT_LAYERS = (0, 13, 26, 39)


def pooled(tensor):
    head = tensor[:(TILES - 1) * BLOCK].reshape(TILES - 1, BLOCK, HEADS, HEAD_DIM)
    remainder = tensor[(TILES - 1) * BLOCK:]
    return torch.cat((head.mean(1, dtype=torch.float32), remainder.mean(0, dtype=torch.float32)[None]))


def relative(actual, expected):
    a, b = actual.float(), expected.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def head_first(t):
    return t.transpose(0, 1).unsqueeze(0)


def native_flash(q, k, v):
    return torch.ops.aten._scaled_dot_product_flash_attention(*map(head_first, (q,k,v)), 0.0, False, False, scale=SCALE)[:2]


class OnlineAttention:
    def __init__(self, experiment, fit):
        self.exp = experiment
        self.panel = load(fit / "plan_assets/attention_panel_index.json")
        assert self.panel["probe_ids_in_production_order"] == self.exp.base["observer_ids"]
        self.probes = torch.tensor(self.exp.base["observer_ids"],device="cuda",dtype=torch.long)
        self.panel_ids = torch.tensor(self.panel["token_ids"],device="cuda",dtype=torch.long)
        self.panel_rows = torch.tensor(self.panel["probe_rows_in_production_order"],device="cuda",dtype=torch.long)
        assert torch.equal(self.panel_ids[self.panel_rows],self.probes)
        bases = fit / "plan_assets"
        binding = load(bases / "bases/manifest.json")["bases"]
        projections=[]
        for key in ("P_v", "P_k"):
            row = binding[key]
            path = bases / row["path"] if key=="P_v" else Path("attention_P_k.npy")
            assert digest(path) == row["sha256"]
            projections.append(torch.tensor(np.load(path,allow_pickle=False),device="cuda",dtype=torch.float32))
        assert all(proj.shape == (128,128) for proj in projections)
        self.pv,self.pk=(proj[:,:64].contiguous() for proj in projections)
        self.fits = {}
        self.calibration={}
        for expert in ("high","low"):
            root = fit / "fit_full"
            recipe_path=root/f"self_{expert}_recipe.json"
            transform=root/f"self_{expert}_transform.npz"
            recipe=load(recipe_path)
            assert digest(transform)==recipe["transform_sha256"]
            with np.load(transform,allow_pickle=False) as asset:
                matrices={key:{"basis":torch.tensor(asset[f"{key}_basis"],device="cuda",dtype=torch.float32)}
                          for key in ("out_current","out_delta","finite")}
                self.fits[expert]=(matrices,recipe["lambda_values"],float(asset["output_rms"])*math.sqrt(128),float(asset["mass_rms"]))
            self.calibration[expert]={"recipe_sha256":digest(recipe_path),"transform_sha256":digest(transform),"lambda":recipe["lambda_values"]}
        self.flex=torch.compile(flex_attention,fullgraph=True,dynamic=False)

    def route(self,q,k):
        scores=(pooled(q).transpose(0,1) @ pooled(k).permute(1,2,0))*SCALE
        selected=torch.argsort(scores,dim=-1,descending=True,stable=True)[...,:TOP].to(torch.int32)
        # BlockMask carries logical full grid extent while the count limits the
        # kernel to the first TOP entries in each route row.
        padded=F.pad(selected,(0,TILES-TOP)).unsqueeze(0)
        count=torch.full((1,HEADS,TILES),TOP,device=q.device,dtype=torch.int32)
        return BlockMask.from_kv_blocks(count,padded,BLOCK_SIZE=BLOCK,seq_lengths=(N,N)),selected

    def selected(self,q,k,v,support,finite):
        value=head_first(v).contiguous()
        if not finite:
            selected,aux=self.flex(head_first(q),head_first(k),value,block_mask=support,scale=SCALE,return_aux=AuxRequest(lse=True))
            return selected[0],aux.lse[0],None

        k32=k.transpose(0,1).float()
        v32=value[0].float()
        kp=k32 @ self.pk[:,:8]
        vp=v32 @ self.pv[:,:8]
        augmented=torch.cat((value, kp.bfloat16()[None], (kp*vp).bfloat16()[None]),dim=-1)
        out,aux=self.flex(head_first(q),head_first(k),augmented,block_mask=support,scale=SCALE,return_aux=AuxRequest(lse=True))
        lse=aux.lse
        out=out[0]
        o, ek, ekv=out.float().split((128,8,8),dim=-1)
        mean_k=k32.mean(1)
        mean_v=v32.mean(1)
        centered_key=k32-mean_k[:,None,:]
        centered_vp=vp-vp.mean(1,keepdim=True)
        cross=(centered_key.transpose(1,2)@centered_vp)/N
        q32=q.transpose(0,1).float()
        global_vk=(q32@cross)*SCALE
        query7=(q32@self.pk[:,:7])*SCALE
        counts=(support.kv_indices[0,:,:,:TOP]==TILES-1).any(-1).to(torch.float32)
        counts=(TOP*BLOCK-counts*(TILES*BLOCK-N)).log()
        centered_lme=lse[0]-(q32*mean_k[:,None,:]).sum(-1)*SCALE-counts.repeat_interleave(BLOCK,dim=1)[:,:N]
        raw=torch.cat((o@self.pv[:,:32],ek,ekv-ek*(o@self.pv[:,:8]),global_vk,query7,centered_lme[...,None]),dim=-1)
        return out[...,:HEAD_DIM],lse[0],(raw,mean_v)

    def anchor_mass(self,q,k,support,selected_lse,full_lse):
        difference=selected_lse-full_lse[..., :N]
        # In nearly saturated supports, the two fused backends can differ
        # enough in FP32 reduction order to violate subset monotonicity. Use
        # the positive omitted mass directly for those query/head pairs.
        bad=torch.nonzero((difference>5e-5) | ~torch.isfinite(difference))
        rho=-difference.expm1()
        if not len(bad):
            return rho,0
        before=float(difference.amax())
        offsets=torch.arange(BLOCK,device=q.device)
        for head in bad[:,0].unique().tolist():
            rows=bad[bad[:,0]==head,1]
            keys=k[:,head].float().T.contiguous()
            for ids in rows.split(128):
                queries=q[:,head].index_select(0,ids).float()
                logits=(queries@keys)*SCALE
                blocks=support.kv_indices[0,head,ids//BLOCK,:TOP].long()
                selected=(blocks[...,None]*BLOCK+offsets).flatten(1)
                weights=(logits-logits.amax(dim=-1,keepdim=True)).exp()
                denominator=weights.sum(dim=-1)
                # Padding only occurs when the final block is selected, so
                # clamping those indices repeats an already selected key.
                weights.scatter_(1,selected.clamp_max(N-1),0.0)
                rho[head,ids]=(weights.sum(dim=-1)/denominator).clamp_(0,1)
        assert bool(torch.isfinite(rho).all())
        print("LSE_RECONCILE",len(bad),before,flush=True)
        return rho,len(bad)

    def low_cache(self,q,k,v,route,native,lse,arm):
        use_finite=arm=="FIN-C0"
        sel,lse_sel,stats=self.selected(q,k,v,route,use_finite)
        matrices,_,_,_=self.fits["low"]
        if use_finite:
            raw,vstar=stats
            projected=(raw@matrices["finite"]["basis"]).bfloat16()
        else:
            projected=((sel.float()@self.pv)@matrices["out_delta"]["basis"]).bfloat16()
            vstar=None
        rho,recomputed=self.anchor_mass(q,k,route,lse_sel,lse)
        rho=rho.unsqueeze(-1)
        u0=(native.float()-(1-rho)*sel.float()).bfloat16()
        return (route,rho,u0,projected,None if vstar is None else vstar[:,None]),recomputed

    def approximate(self,q,k,v,cache,arm):
        if arm=="OUT-ABS":
            route=cache
        else:
            route,rho,u0,anchor_design,vstar=cache
        expert="high" if arm=="OUT-ABS" else "low"
        matrices,lambdas,scale_r,scale_m=self.fits[expert]
        sel,_,stats=self.selected(q,k,v,route,arm=="FIN-C0")
        selected=sel.float()
        if arm=="OUT-ABS":
            x=apply_transform(selected@self.pv,matrices["out_current"]["basis"])
            base=selected
            c=None
        else:
            key="finite" if arm=="FIN-C0" else "out_delta"
            raw=stats[0] if stats is not None else selected@self.pv
            diff=(raw@matrices[key]["basis"]).bfloat16().float()-anchor_design.float()
            x=torch.cat((torch.ones_like(diff[...,:1]),diff),dim=-1)
            base=(1-rho)*selected+u0.float()
            c=(selected-vstar)*(scale_m/scale_r) if stats is not None else None
        query=q.index_select(0,self.probes)
        truth=F.scaled_dot_product_attention(*map(head_first,(query,k,v)),dropout_p=0.0,is_causal=False,scale=SCALE)[0]
        xp=x.index_select(1,self.probes)
        cp=None if c is None else c.index_select(1,self.probes)
        target=(truth.float()-base.index_select(1,self.probes))/scale_r
        coef,mass,_=schur_fit(xp,target,cp,None,len(self.probes)*lambdas[arm],mu=0)
        residual=x@coef
        if mass is not None:
            residual.sub_((x@mass)*c)
        predicted=(base+residual*scale_r).transpose(0,1).to(q.dtype).contiguous()
        predicted.index_copy_(0,self.probes,truth.transpose(0,1))
        return predicted,truth,selected

    @contextmanager
    def bind(self,arm, audit=None, profile=False, archive_routes=False, high_step0=False):
        from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight

        assert not high_step0 or (arm=="OUT-D" and audit is None)
        owner=self.exp.owner
        orig=owner.infer_self_attn
        original_apply=TorchSDPAWeight.apply
        assert digest(Path(inspect.getfile(orig)))==self.exp.base["native_transformer_sha256"]
        assert digest(Path(inspect.getfile(original_apply)))==BACKEND
        sig=inspect.signature(original_apply)
        slots={}
        counter=Counter()
        events=[]
        anchors=[]
        active=None

        @wraps(orig)
        def wrapper(model,phase,*args,**kwargs):
            nonlocal active
            step,layer=int(model.scheduler.step_index),int(model.block_idx)
            assert active is None and step in (0,1,2,3) and 0<=layer<40
            if audit is not None and layer not in AUDIT_LAYERS:
                return orig(model,phase,*args,**kwargs)
            assert not model.config["seq_parallel"] and not model.config["use_image_encoder"]
            active=(step,layer,phase)
            try:
                return orig(model,phase,*args,**kwargs)
            finally:
                active=None

        @wraps(original_apply)
        def apply(op,*args,**kwargs):
            if active is None:
                return original_apply(op,*args,**kwargs)
            step,layer,phase=active
            assert op is phase.self_attn_1
            call=sig.bind(op,*args,**kwargs)
            call.apply_defaults()
            p=call.arguments
            q,k,v=(p[key] for key in ("q","k","v"))
            assert q.shape==k.shape==v.shape==(N,HEADS,HEAD_DIM) and q.dtype==k.dtype==v.dtype==torch.bfloat16
            assert p["attn_mask"] is None and not p["causal"] and not p["drop_rate"]
            assert p["max_seqlen_q"]==p["max_seqlen_kv"]==N
            if profile:
                start,stop=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
            if step%2==0:
                route,selected=self.route(q,k)
                if step==0:
                    slots[layer]=route
                    if high_step0:
                        predicted,_,_=self.approximate(q,k,v,route,"OUT-ABS")
                        returned=predicted.reshape(N,-1)
                        counter["hits"]+=1
                        counter["native_probe_queries"]+=len(self.probes)
                    else:
                        returned=original_apply(op,*args,**kwargs)
                else:
                    dense,lse=native_flash(q,k,v)
                    slots[layer],recomputed=self.low_cache(q,k,v,route,dense[0],lse[0],arm)
                    counter["low_anchor_query_heads_with_direct_omitted_mass"]+=recomputed
                    returned=dense[0].transpose(0,1).reshape(N,-1).contiguous()
                    if audit is not None:
                        native=original_apply(op,*args,**kwargs)
                        assert torch.equal(native,returned),relative(native,returned)
                if archive_routes:
                    support=sha256(selected.cpu().contiguous().numpy().tobytes()).hexdigest()
                    tensors=() if step==0 else tuple(z for z in slots[layer][1:] if z is not None)
                    anchors.append(dict(step=step,layer=layer,support_sha256=support,
                         anchor_fields=[dict(shape=list(z.shape),dtype=str(z.dtype)) for z in tensors]))
                if not high_step0 or step==2:
                    counter["anchors"]+=1
                if high_step0:
                    counter["route_builds"]+=1
                if audit is not None:
                    audit.anchor(step,layer,selected)
            else:
                tag="OUT-ABS" if step==1 else arm
                predicted,teacher,selected=self.approximate(q,k,v,slots.pop(layer),tag)
                returned=predicted.reshape(N,-1)
                counter["hits"]+=1
                counter["native_probe_queries"]+=len(self.probes)
                if audit is not None:
                    returned=original_apply(op,*args,**kwargs)
                    audit.hit(step,layer,tag,predicted,teacher,selected,returned)
            counter["attention_sites"]+=1
            if profile:
                stop.record()
                events.append((step,start,stop))
            return returned

        owner.infer_self_attn=wrapper
        TorchSDPAWeight.apply=apply
        try:
            yield counter,events,anchors
            assert not slots
            expected=len(AUDIT_LAYERS) if audit is not None else 40
            recoveries=3 if high_step0 else 2
            assert (counter["attention_sites"],counter["anchors"],counter["hits"],counter["native_probe_queries"])==(
                4*expected,(4-recoveries)*expected,recoveries*expected,64*recoveries*expected)
            if high_step0:
                assert counter["route_builds"]==80
        finally:
            slots.clear()
            owner.infer_self_attn=orig
            TorchSDPAWeight.apply=original_apply


class PanelAudit:
    def __init__(self, online, assets, source):
        self.online,self.assets=online,assets
        self.sites={(expert,r.row["layer"]):r for (module,expert),rows in assets.groups.items() if module=="self"
                    for r in rows if r.source==source and r.row["background"]=="wide8192"}
        assert len(self.sites)==8
        self.rows=[]

    def anchor(self,step,layer,route):
        expert="high" if step==0 else "low"
        site=self.sites[expert,layer]
        record=site.index_row
        path=site.root/record["support_file"]
        assert digest(path)==site.row["support_sha256"]
        expected=self.assets.load(str(path),device="cuda")["selected_blocks"]
        assert torch.equal(expected,route),(step,layer,"anchor support and stable tie order")

    def hit(self,step,layer,arm,predicted,teacher,selected,native):
        expert="high" if step==1 else "low"
        site=self.sites[expert,layer]
        d=self.assets.prepare(site)
        r=self.online.panel_ids
        p,h=self.assets.p,self.assets.h
        matrices,lambdas,scale_r,scale_m=self.online.fits[expert]
        expected,_=readout(d,design(d,matrices),arm,p,lambdas[arm],scale_r,scale_m,self.assets.perm)
        candidate=predicted[r].transpose(0,1)
        native_panel=native[r].reshape(512,HEADS,HEAD_DIM).transpose(0,1)
        assert torch.equal(native_panel,d["y"].bfloat16()),(step,layer,"native archive projection input")
        relative_probe=relative(teacher,native_panel[:,p])
        relative_selected=relative(selected[:,r],d["selected"])
        relative_replay=relative(candidate[:,h],expected[:,h].bfloat16())
        assert torch.equal(candidate[:,p],teacher),(step,layer,"P64 overwrite")
        row=dict(step=step,layer=layer,arm=arm,probe_relative_l2=relative_probe,
                 selected_relative_l2=relative_selected,held_readout_relative_l2=relative_replay)
        self.rows.append(row)
        print("AUDIT",json.dumps(row),flush=True)
        assert relative_probe<0.006 and relative_selected<0.006 and relative_replay<0.006,row


@contextmanager
def save_trajectory(exp,root):
    from safetensors.torch import save_file

    root.mkdir(parents=True)
    scheduler=exp.pipe.runner.scheduler
    owner=exp.struct
    original_dispatch=owner.get_current_model_index
    original_post=scheduler.step_post
    records={}

    def archive(name,**values):
        path=root/(name+".safetensors")
        save_file({k:v.detach().to("cpu",copy=True).contiguous() for k,v in values.items()},str(path))
        records[name]=dict(sha256=digest(path),bytes=path.stat().st_size)

    def dispatch(model):
        if int(scheduler.step_index)==0:
            archive("initial",noise=scheduler.latents)
        return original_dispatch(model)

    def post():
        step=int(scheduler.step_index)
        flow=scheduler.noise_pred.detach().clone()
        result=original_post()
        archive(f"step{step}",flow=flow,latent=scheduler.latents)
        return result

    owner.get_current_model_index=dispatch
    scheduler.step_post=post
    try:
        yield records
        assert set(records)=={"initial",*(f"step{i}" for i in range(4))}
    finally:
        owner.get_current_model_index=original_dispatch
        scheduler.step_post=original_post


def checked_controls(base,cohort,config):
    source_path=Path("sources_locked.json")
    assert digest(source_path)==SOURCES_SHA
    locked={x["source_id"]:x for x in load(source_path)["sources"]}
    rows={x["record_id"]:x for x in cohort}
    out={}
    for source in SOURCES:
        path=Path(config["controls"][source])
        info=load(path)
        assert info["status"]=="complete" and info["source"]["record_id"]==source
        assert all(info["source"][k]==rows[source][k]==locked[source][k]
                   for k in ("seed","prompt_sha256","image_sha256","source_cluster_id"))
        assert info["native_sha256"]==base["native_transformer_sha256"] and info["backend_sha256"]==BACKEND
        entry=info["backgrounds"]["wide8192"]["record"]
        assert entry["counters"]["variant"]=="wide8192" and entry["counters"]["tile_rows"]==8192
        for background in ("wide8192","Dense"):
            video=path.parents[1]/background/(source+".mp4")
            assert digest(video)==info["backgrounds"][background]["record"]["video"]["mp4_sha256"]
        out[source]=dict(index_sha256=digest(path),wide8192=entry,dense=info["backgrounds"]["Dense"]["record"])
    return out


@torch.inference_mode()
def main(fit,panels,config_path,phase):
    config=load(config_path)
    assert config["plan_commit"]==PLAN and digest(panels/"RESULTS.sha256")==config["panel_results_sha256"]
    control=checked_controls(load(Path("base.json")),load(Path("cohort.json")),config)
    exp=Experiment()
    wide=exp.build("wide8192")
    active=OnlineAttention(exp,fit)
    arms={"A":None,"B":"FIN-C0","C":"OUT-D"}
    path=exp.output/"ONLINE_RESULT.json"
    receipt=dict(status="running",plan_commit=PLAN,phase=phase,job=exp.job,gpu=exp.state["gpu"],
                 implementation_sha256=digest(Path(__file__)),
                 calibration=active.calibration,arms={"A":"native","B":"OUT-ABS/FIN-C0","C":"OUT-ABS/OUT-D"},
                 registration=dict(exploratory=True,seed=42,sources=list(SOURCES),steps=4,layers=40,
                    selected_keys_per_query_block=31,query_block=128,ffn="BF16 wide8192",attention_probes=64,
                    anchor_lse="natural-log ATen Flash SDPA",selected_augmented_value_precision="BF16"),
                 control=control,audit={},renders=[],init_seconds=exp.state["init_seconds"],wide_setup=exp.state["setup"])
    save(path,receipt)

    def render(source,arm,group,save_states=False,profile=False):
        kernel=arms[arm]
        binder=active.bind(kernel,profile=profile,archive_routes=save_states) if kernel else nullcontext(({},[],[]))
        trace=save_trajectory(exp,exp.output/group/(source+"_trajectory")) if save_states else nullcontext({})
        with binder as (counts,events,anchors),trace as trajectories:
            result=exp.render(source,group,wide)
        ref=control[source]["wide8192"]
        assert result["initial"]==ref["initial"]
        assert result["flows"][0]==ref["flows"][0] and result["latents"][0]==ref["latents"][0]
        assert result["counters"]["variant"]=="wide8192"
        if arm=="A":
            assert equal_trace(result,ref)
        item=dict(arm=arm,source=source,record=result,attention=dict(counts),trace_tensors=trajectories)
        if anchors:
            assert len(anchors)==80
            item["attention_anchor_identity"]=anchors
        if events:
            item["attention_core_and_recovery_ms"]=[sum(a.elapsed_time(b) for s,a,b in events if s==step) for step in range(4)]
        return item

    if phase=="quality":
        assets=Assets(panels,fit/"plan_assets","cuda")
        for arm in ("B","C"):
            audit=PanelAudit(active,assets,SOURCES[0])
            with active.bind(arms[arm],audit=audit):
                ref=exp.render(SOURCES[0],f"audit_{arm}",wide)
            assert equal_trace(ref,control[SOURCES[0]]["wide8192"]) and len(audit.rows)==8
            receipt["audit"][arm]=dict(A_matches_archive=True,rows=audit.rows)
            save(path,receipt)
        for i,source in enumerate(SOURCES):
            for arm in (("B","C") if i==0 else ("C","B")):
                receipt["renders"].append(render(source,arm,f"candidate_{arm}",save_states=True))
                save(path,receipt)
    else:
        assert phase=="timing"
        receipt["warmup"]=[render(SOURCES[0],arm,f"warm_{arm}",profile=arm!="A") for arm in arms]
        save(path,receipt)
        orders=(("A","B","C"),("B","C","A"),("C","A","B"))
        for repeat in range(3):
            for src,source in enumerate(SOURCES):
                for arm in orders[(repeat+src)%3]:
                    row=render(source,arm,f"timing_{repeat}_{arm}")
                    receipt["renders"].append(dict(repeat=repeat,**row))
                    save(path,receipt)
    receipt["status"]="complete"
    save(path,receipt)


if __name__=="__main__":
    main(*(Path(p).resolve() for p in sys.argv[1:4]),sys.argv[4])
