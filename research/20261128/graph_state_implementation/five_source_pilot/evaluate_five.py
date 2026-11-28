"""Verify per-profile receipts, summarize matched runtime and prepare VBench binding."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, geometric_mean

import numpy as np
from safetensors.torch import load_file

import eval_anchor10 as evaluator


ARMS=("00_frozen","01_staged_ffn","02_partition_anchor","03_attention_stream","04_block_stream","05_high_output_fold",
      "06_projection_precompose","07_ffn_factor","08_attention_joint","09_combined_graph","10_combined_state","11_ffn_gauge")
ASSET_ARMS={"07_ffn_factor","08_attention_joint","10_combined_state","11_ffn_gauge"}


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def checked(p,role):
    result = p/(role+".json")
    if not (p/(role+".sha256")).is_file():
        return None
    sha,name=(p/(role+".sha256")).read_text().split()
    assert name==result.name and digest(result)==sha
    data=read(result)
    assert data["status"]=="complete" and len(data["records"])==5
    return data


def timed(proposed, reference, sources):
    assert proposed["job"]==reference["job"] and proposed["gpu"]==reference["gpu"]
    timing=[]
    for source in sources:
        m=proposed["records"][source]["record"]; b=reference["records"][source]["record"]
        assert m["initial"]==b["initial"]
        timing.append(dict(source=source,candidate_wall_s=m["wall"],reference_wall_s=b["wall"],
            wall_speedup=b["wall"]/m["wall"],candidate_steps_ms=m["step_ms"],reference_steps_ms=b["step_ms"],
            gpu_speedup=sum(b["step_ms"])/sum(m["step_ms"]),reference_peak_gib=b["peak_allocated"]/2**30,
            candidate_peak_gib=m["peak_allocated"]/2**30))
    return dict(wall_speedup=geometric_mean(x["wall_speedup"] for x in timing),
                gpu_speedup=geometric_mean(x["gpu_speedup"] for x in timing),
                candidate_mean_seconds=fmean(x["candidate_wall_s"] for x in timing),
                reference_mean_seconds=fmean(x["reference_wall_s"] for x in timing),
                peak_candidate_gib=max(x["candidate_peak_gib"] for x in timing),
                peak_reference_gib=max(x["reference_peak_gib"] for x in timing),sources=timing)


def relative_latent(a,b):
    if not all(x["final_latent"] is not None for x in (a[1],b[1])):
        return None
    data=[]
    for root,receipt in (a,b):
        tag=receipt["final_latent"]
        path=root/tag["file"]
        assert path.parent==root and digest(path)==tag["sha256"]
        assert tag["latent_trace_sha256"]==receipt["record"]["latents"][-1]
        data.append(load_file(str(path),device="cpu")["latent"].double())
    return float((data[0]-data[1]).norm()/data[1].norm())


def run(args):
    prior=read(args.previous_binding)
    sources=read(args.core/"SOURCES.json")["sources"]
    ids=[r["source_id"] for r in sources]
    old={r["source"]:r for r in prior["records"]}
    results={}; missing=[]; binding_rows={s:dict(old[s],videos={}) for s in ids}
    baseline=checked(args.core/"jobs/00_frozen","candidate")
    assert baseline is not None and all(r["exact_206224_centered_trace"] for r in baseline["records"].values())
    base_dir=args.core/"jobs/00_frozen/candidate/runtime/output/quality"
    geometry=(832,464,81,16)
    base_clips={}
    for arm in ARMS:
        directory=(args.asset if arm in ASSET_ARMS else args.core)/"jobs"/arm
        candidate=checked(directory,"candidate")
        if candidate is None:
            missing.append(arm)
            continue
        reference=checked(directory,"same_gpu_reference") if arm!="00_frozen" else None
        out=dict(status="complete",profile=candidate["profile"],profile_sha256=candidate["profile_sha256"],
                 semantics=candidate["semantics"],quality=[],timing=timed(candidate,reference,ids) if reference else None)
        for source in ids:
            item=candidate["records"][source]; r=item["record"]
            b=baseline["records"][source]["record"]
            assert r["initial"]==b["initial"]
            profile=item["profile"]
            if profile["anchor_partition"]=="triton":
                assert item["attention"]["partition_anchors"]==40
            if profile["block_streaming"]:
                assert r["counters"]["streamed_blocks"]==80
            if profile["attention_basis_dir"]:
                assert item["attention"]["joint_encoded_anchors"]==40
            if profile["ffn_staged"]:
                assert r["counters"]["staged_hits"]==80
            if profile["ffn_factor_dir"]:
                assert r["counters"]["factor_cache"]
            if profile["ffn_gauge_dir"]:
                assert r["counters"]["ffn_paired_gauge_active"]
            name=source+".mp4"
            root=directory/"candidate/runtime/output/quality"
            path=root/name
            assert digest(path)==r["video"]["mp4_sha256"]
            video=dict(path=str(path),sha256=r["video"]["mp4_sha256"],rgb_sha256=r["video"]["decoded_rgb24_sha256"])
            binding_rows[source]["videos"][arm]=video
            clip=evaluator.decode(video,geometry)
            if arm=="00_frozen":
                base_clips[source]=clip
            p=evaluator.pixel_distance(clip,base_clips[source])
            adjacent=fmean(float(np.abs(np.subtract(a,b,dtype=np.int16)).mean()) for a,b in zip(clip[1:],clip[:-1]))/255
            out["quality"].append(dict(source_id=source,exact_frozen_trace=item["exact_206224_centered_trace"],
                decoded_rgb24_sha256=video["rgb_sha256"],mp4_sha256=video["sha256"],
                rgb_rmse_0to1=math.sqrt(p["mse"])/255,temporal_delta_rgb_mae_0to1=p["temporal_delta_mae"]/255,
                mean_adjacent_rgb_mae_0to1=adjacent,
                final_latent_relative_l2=relative_latent((root,item),(base_dir,baseline["records"][source])),
                attention=item["attention"],ffn=r["counters"]))
        results[arm]=out
    evaluator.save(args.out/"PILOT_RESULTS.json",dict(status="complete" if not missing else "partial",sources=ids,
        missing_profiles=missing,scope="five fixed stress/median sources; no population quality inference",profiles=results))
    if not args.no_scores:
        cohort=args.core/"SOURCES.json"
        result=dict(schema="FFNDS_GRAPH_OPTIM_5_V1",arms=list(results),sources=ids,records=list(binding_rows.values()),
                    geometry=list(geometry),cohort=str(cohort),cohort_sha256=digest(cohort),
                    metric_scope=prior["metric_scope"],metric_scope_sha256=prior["metric_scope_sha256"],
                    evaluator_sha256=digest(Path(evaluator.__file__)))
        evaluator.save(args.out/"PILOT_SCORE_BINDING.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("core","asset","previous-binding","out"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--no-scores",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    run(args)
