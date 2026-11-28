"""CPU replay of matched current-S128 PFRR on the sealed legacy DEV panels."""
import json
import math
from pathlib import Path
from statistics import geometric_mean
import sys

import numpy as np
import torch

from attention_panel_fit import Assets, DEV, digest, design, metrics, read, readout, save
from attention_paired_value_ops import pfrr


PLAN = "9dec5f1baf85c089b65dcbc142d0115fdf2308ae"
LOSSES = ("head", "point", "spatial", "temporal")
PFRR = "PFRR-S128"


@torch.inference_mode()
def run(panels, fit, config_file):
    config = read(config_file)
    assert config["plan_commit"] == PLAN and digest(panels/"RESULTS.sha256") == config["panel_results_sha256"]
    torch.set_num_threads(4)
    assets = Assets(panels,fit/"plan_assets","cpu")
    recipe = read(fit/"fit_full/self_low_recipe.json")
    transform = fit/"fit_full/self_low_transform.npz"
    assert digest(transform) == recipe["transform_sha256"]
    with np.load(transform,allow_pickle=False) as z:
        matrices={key:{"basis":torch.tensor(z[f"{key}_basis"],dtype=torch.float32)} for key in ("out_current","out_delta","finite")}
        sr,sm = float(z["output_rms"])*math.sqrt(128),float(z["mass_rms"])
    legacy = fit/"fit_full/site_metrics.jsonl"
    by_site = {(r["source_id"],r["layer"]):r for r in map(json.loads,legacy.read_text().splitlines())
               if r["source_id"] in DEV and (r["background"],r["module"],r["expert"])==("wide8192","self","low")}
    assert len(by_site)==12
    rows=[]
    for site in assets.groups["self","low"]:
        s, layer = site.source,site.row["layer"]
        if s not in DEV or site.row["background"]!="wide8192":
            continue
        t = site.root/site.row["tensor_file"]
        assert digest(t)==site.row["tensor_file_sha256"]
        d = assets.prepare(site)
        assert d["selected"].shape == d["y"].shape == (40,512,128) and d["gate"] is not None
        x = design(d,matrices)
        labels = d["y"].index_select(1,assets.p)
        predicted = pfrr(d["selected"].double(),labels.double(),assets.p).float()
        assert torch.equal(predicted.index_select(1,assets.p),labels)
        w = assets.weight(site)
        scores={PFRR:metrics(predicted,d,w,assets,d["selected"])}
        mismatch={}
        for arm in ("OUT-ABS","OUT-D"):
            z,_ = readout(d,x,arm,assets.p,recipe["lambda_values"][arm],sr,sm,assets.perm)
            scores[arm]=metrics(z,d,w,assets,d["selected"] if arm=="OUT-ABS" else d["anchor"])
            old=by_site[s,layer]["arms"][arm]["P64_estimate"]
            mismatch[arm]={k:abs(scores[arm][k]/old[k]-1) for k in LOSSES}
            assert max(mismatch[arm].values())<0.01,(s,layer,arm,mismatch[arm])
        rows.append(dict(source=s,layer=layer,tensor_sha256=site.row["tensor_file_sha256"],
                         scores={a:{k:y[k] for k in LOSSES} for a,y in scores.items()},legacy_relative_deviation=mismatch))
        print("PFRR_PANEL_VERIFIED",s,layer,flush=True)
    assert len(rows)==12
    summary={}
    for denom in ("OUT-ABS","OUT-D"):
        summary[denom]=dict()
        for loss in LOSSES:
            per_source={s:geometric_mean(r["scores"][PFRR][loss]/r["scores"][denom][loss] for r in rows if r["source"]==s) for s in DEV}
            summary[denom][loss]=dict(ratio=geometric_mean(per_source.values()),by_source=per_source,
                winning_sites=sum(r["scores"][PFRR][loss]<r["scores"][denom][loss] for r in rows))
    save(Path("PFRR_PANEL_RESULT.json"),dict(status="complete",plan_commit=PLAN,sites=rows,ratios=summary,
         coverage="uniform on supplied P64; no author RCP groups in the old run",basis_sha256=recipe["transform_sha256"],
         standardization_epsilon=1e-6,ridge=1,panels_sha256=config["panel_results_sha256"], legacy_metric_file_sha256=digest(legacy)))


if __name__=="__main__":
    run(*(Path(x).resolve() for x in sys.argv[1:]))
