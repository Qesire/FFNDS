"""Run frozen C versus A once on each of six independently selected clusters."""
from contextlib import nullcontext
from hashlib import sha256
import math
from pathlib import Path
from statistics import mean
import sys

import torch

from attention_online_structured import Experiment, OnlineAttention, digest, equal_trace, load, save


PLAN = "5c77fc9a4901232e7ae3129948bb304763472fa7"


@torch.inference_mode()
def main(fit, config_path):
    config=load(config_path)
    method_path=Path("CONFIRM_METHOD.json")
    manifest_path=Path("CONFIRM_SOURCES.json")
    method,manifest=load(method_path),load(manifest_path)
    method_sha=digest(method_path)
    assert method["status"]=="FROZEN" and method["plan_commit"]==config["plan_commit"]==PLAN
    assert digest(Path("attention_online_structured.py"))==method["attention_source_sha256"]
    assert digest(fit/"FIT.sha256")==method["frozen_fit_manifest_sha256"]
    assert digest(manifest_path)==config["confirm_sources_sha256"]
    assert manifest["status"]=="FROZEN_INPUT_SELECTION" and manifest["method_recipe_sha256"]==method_sha
    assert (manifest["candidate_id"],manifest["reference_id"])==(method["candidate_id"],method["reference_id"])
    archive=Path(config["reference_quality_path"])
    assert digest(archive)==method["quality_receipt_sha256"]
    historical=load(archive)
    people=manifest["sources"]
    excluded=set(manifest["exclusion_registry"]["excluded_cluster_ids"])
    assert len(people)==len({r["source_cluster_id"] for r in people})==6
    assert not (excluded & {r["source_cluster_id"] for r in people})
    for row in people:
        assert digest(Path(row["image_path"]))==row["image_sha256"]
        assert sha256(row["prompt"].encode()).hexdigest()==row["prompt_sha256"]

    exp=Experiment()
    for row in people:
        assert row["source_id"] not in exp.rows
        exp.rows[row["source_id"]]=dict(row,record_id=row["source_id"],seed=42)
    online=OnlineAttention(exp,fit)
    assert online.calibration==method["calibration"]
    wide=exp.build("wide8192")
    state=dict(status="running",job=exp.job,plan_commit=PLAN,method_recipe_sha256=method_sha,
               sources_sha256=manifest["sources_sha256"],manifest_sha256=digest(manifest_path),gpu=exp.state["gpu"],
               init_seconds=exp.state["init_seconds"],setup=exp.state["setup"],warmup={},pairs=[])
    path=exp.output/"C_CONFIRM_RESULT.json"

    def render(source,arm,group):
        method=online.bind("OUT-D") if arm=="C" else nullcontext(({},[],[]))
        with method as (counts,_,_):
            record=exp.render(source,group,wide)
        counter=record["counters"]
        assert (counter["variant"],counter["tile_rows"],counter["anchors"],counter["hits"])==("wide8192",8192,80,80)
        if arm=="C":
            assert (counts["attention_sites"],counts["anchors"],counts["hits"],counts["native_probe_queries"])==(160,80,80,5120)
        return dict(record=record,attention=dict(counts))

    dev="vbi_0392"
    for arm in ("A","C"):
        reference=historical["control"][dev]["wide8192"] if arm=="A" else next(
            x["record"] for x in historical["renders"] if x["source"]==dev and x["arm"]=="C")
        warm=render(dev,arm,f"warm_{arm}")
        assert equal_trace(warm["record"],reference),(arm,"frozen DEV warmup mismatch")
        state["warmup"][arm]=warm
        save(path,state)

    for i,row in enumerate(people):
        sid,cluster=row["source_id"],row["source_cluster_id"]
        order=("A","C") if i%2==0 else ("C","A")
        arms={a:render(sid,a,f"paired_{a}") for a in order}
        reference,candidate=(arms[a]["record"] for a in ("A","C"))
        assert reference["initial"]==candidate["initial"]
        assert all(reference[k][0]==candidate[k][0] for k in ("flows","latents"))
        flip=int(sha256(f"{method_sha}|{cluster}".encode()).hexdigest(),16)%2
        left,right=("A","C") if flip==0 else ("C","A")
        pair=dict(source_id=sid,cluster_id=cluster,stratum=row["stratum"],order=order,arms=arms,
                  wall_ratio=candidate["wall"]/reference["wall"],
                  gpu_ratio=sum(candidate["step_ms"])/sum(reference["step_ms"]),
                  step_delta_ms=[c-a for c,a in zip(candidate["step_ms"],reference["step_ms"],strict=True)],
                  review_order=dict(left=left,right=right))
        state["pairs"].append(pair)
        save(path,state)
    state["wall_geomean_ratio"]=math.exp(mean(math.log(r["wall_ratio"]) for r in state["pairs"]))
    state["gpu_geomean_ratio"]=math.exp(mean(math.log(r["gpu_ratio"]) for r in state["pairs"]))
    state["status"]="complete"
    save(path,state)


if __name__ == "__main__":
    main(*(Path(p).resolve() for p in sys.argv[1:]))
