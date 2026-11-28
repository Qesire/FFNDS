"""Fit the frozen phase-0 basis and render a single matched 0392 candidate."""
from collections import defaultdict
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file

from attention_panel_fit import CAL, apply_transform, schur_fit, sufficient_stats, transform_cal
from attention_online_structured import Assets, Experiment, OnlineAttention, digest, load, save, save_trajectory, checked_controls


PLAN = "5c77fc9a4901232e7ae3129948bb304763472fa7"
SOURCE = "vbi_0392"


def endpoints(assets, site):
    path = site.root / site.row["tensor_file"]
    assert digest(path) == site.row["tensor_file_sha256"]
    z = assets.load(str(path), device=assets.device)
    y, selected = (z[k][0].transpose(0,1).float() for k in ("o_native","o_selected"))
    assert y.shape == selected.shape == (40,512,128)
    return y,selected,selected@assets.pv


def fit_basis(assets):
    stats = defaultdict(list)
    fit_sites = []
    for site in assets.groups["self","high"]:
        if site.source not in CAL:
            continue
        assert site.row["anchor_step"] == 0
        _,_,raw = endpoints(assets,site)
        panel = assets.prepare(site)
        panel["endpoints"]["out_current"] = (None,raw)
        stats[site.source].append(sufficient_stats(panel))
        fit_sites.append(dict(source=site.source,background=site.row["background"],layer=site.row["layer"],
                              panel_sha256=site.row["tensor_file_sha256"]))
    assert len(fit_sites)==24 and all(len(stats[s])==8 for s in CAL)
    matrices,_,_ = transform_cal(stats,CAL)
    return matrices["out_current"],fit_sites


def panel_diagnostics(assets, old, phase0, lam, scale):
    results=[]
    for site in assets.groups["self","high"]:
        if site.source not in ("vbi_0392","vbi_1013"):
            continue
        y,selected,raw = endpoints(assets,site)
        target=(y[:,assets.p]-selected[:,assets.p])/scale
        scores={}
        for tag,basis in (("step1_basis",old),("phase0_basis",phase0)):
            design=apply_transform(raw,basis)
            beta,_,_=schur_fit(design[:,assets.p],target,None,None,len(assets.p)*lam)
            predicted=(selected[:,assets.h]+(design[:,assets.h]@beta)*scale).bfloat16().float()
            scores[tag]=float((predicted-y[:,assets.h]).square().sum(dtype=torch.float64))
        results.append(dict(source=site.source,background=site.row["background"],layer=site.row["layer"],
                            phase0_to_step1_ratio=scores["phase0_basis"]/scores["step1_basis"],point_error=scores))
    assert len(results)==16
    return results


@torch.inference_mode()
def main(fit,panels,config_path):
    c=load(config_path)
    assert c["plan_commit"]==PLAN and digest(panels/"RESULTS.sha256")==c["panel_results_sha256"]
    assert c["source"]==SOURCE
    control=checked_controls(load(Path("base.json")),load(Path("cohort.json")),c)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assets=Assets(panels,fit/"plan_assets","cuda")
    phase0,sites=fit_basis(assets)
    dest=Path("output")
    dest.mkdir(exist_ok=True)
    basis_file=dest/"PHASE0_BASIS.safetensors"
    save_file({k:v.cpu().contiguous() for k,v in phase0.items()}, str(basis_file))
    exp=Experiment()
    online=OnlineAttention(exp,fit)
    matrices,lam,scale_r,scale_m=online.fits["high"]
    assert lam["OUT-ABS"]==0.1
    old=matrices["out_current"]["basis"]
    new=phase0["basis"]
    assert new.shape==old.shape==(64,31) and not torch.equal(old,new)
    diagnostic=panel_diagnostics(assets,old,new,lam["OUT-ABS"],scale_r)
    details=dict(status="complete",job=exp.job,plan_commit=PLAN,cal_sources=list(CAL),fit_sites=sites,
                 endpoint="anchor / step 0 only",fit_basis_sha256=digest(basis_file),calibration=online.calibration,
                 fixed_outabs_lambda=lam["OUT-ABS"],fixed_scale_r=scale_r,fixed_scale_m=scale_m,
                 held_metric="ungated H448 pre-W_O head-space point energy on original trajectories",held_diagnostics=diagnostic)
    save(dest/"PHASE0_FIT.json",details)
    print("PHASE0_FIT_READY",flush=True)
    wide=exp.build("wide8192")
    group="candidate_C0_phase0CAL"
    with online.bind("OUT-D",high_step0=True,phase0_basis=new,archive_routes=True) as (counts,_,routes), \
            save_trajectory(exp,dest/group/(SOURCE+"_trajectory")) as trace:
        row=exp.render(SOURCE,group,wide)
    assert row["initial"]==control[SOURCE]["wide8192"]["initial"]
    assert row["counters"]["variant"]=="wide8192" and row["counters"]["anchors"]==row["counters"]["hits"]==80
    assert len(routes)==80 and counts["native_probe_queries"]==7680
    details.update(record=row,trace=trace,attention=dict(counts),routes=routes,one_dev_video=True)
    save(dest/"PHASE0_RESULT.json",details)


if __name__=="__main__":
    main(*(Path(a).resolve() for a in sys.argv[1:4]))
