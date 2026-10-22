#!/usr/bin/env python3
"""FFNDS measurement ontology / evidence claim guard (offline; never runs model/GPU)."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
CAT=ROOT/"research/metric_system/catalog.json"
RECORDS=ROOT/"research/metric_system/records"
def catalog():
    return json.loads(CAT.read_text(encoding="utf-8"))
def validate_catalog(data):
    e=[]
    levels={t.get("id") for t in data.get("tiers",[])}
    if levels!={"L0","L1","L2","L3","L4","L5"}:e.append("Six distinct metric grains L0-L5 required")
    metrics=data.get("metrics",[])
    ids=[m.get("id") for m in metrics]
    if len(ids)!=len(set(ids)):e.append("duplicate metric ids")
    for m in metrics:
        for key in ("id","unit","direction","statistic_unit","reference","status"):
            if not isinstance(m.get(key),str) or not m[key]:e.append(f"{m.get('id')} missing {key}")
        if m.get("level") not in levels:e.append(f"{m.get('id')} unrecognized grain")
    mandatory={"post_writeback_sse","flow_native_sse","visual_collapse_review",
               "free_rollout_completed","vbench_i2v_subject","vbench_i2v_background",
               "vbench_i2v_motion_smoothness","vbench_i2v_dynamic_degree",
               "vbench_i2v_aesthetic_quality","vbench_i2v_imaging_quality",
               "complete_ffn_latency_ms","generation_gpu_time_s","synthetic_backend_speedup"}
    if not mandatory.issubset(set(ids)):e.append("missing essential metric families")
    if data.get("history",{}).get("R14N0",{}).get("level")!="L1":
        e.append("R14N0 historical local metric must stay L1")
    if data.get("history",{}).get("R14",{}).get("metric_id")!="synthetic_backend_speedup":
        e.append("R14 synthetic hardware evidence must not be native model speed")
    for key,item in data.get("claims",{}).items():
        if not item.get("required_levels"):e.append("claim missing levels: "+key)
        if not set(item.get("required_levels",[])).issubset(levels):e.append("claim has invalid level: "+key)
        for m in item.get("required_metrics",[]):
            if m not in ids:e.append(f"claim {key} requires unknown metric {m}")
    return e
def validate_record(r,data):
    e=[]
    if not isinstance(r,dict):return ["record is not an object"]
    if not isinstance(r.get("record_id"),str) or not r["record_id"].strip():e.append("record_id missing")
    if r.get("evaluation_status")=="PLANNED_NOT_MEASURED":
        if r.get("observations"):e.append("planned record cannot contain measured observations")
        return e
    if r.get("evaluation_status")!="MEASURED":return e+["unknown status"]
    entries=r.get("observations",[])
    if not isinstance(entries,list) or not entries:return e+["measured record has no observations"]
    catalogMap={m["id"]:m for m in data["metrics"]}
    seen=set()
    for m in entries:
        if not isinstance(m,dict):e.append("observation is not object");continue
        id=m.get("metric_id")
        if id not in catalogMap:e.append("unknown metric "+str(id));continue
        if id in seen:e.append("duplicate observation metric "+id)
        seen.add(id)
        ref=catalogMap[id]
        if m.get("level")!=ref["level"]:e.append(id+" evidence level does not match catalog")
        if m.get("unit")!=ref["unit"]:e.append(id+" incompatible unit")
        if m.get("value") is None:e.append(id+" marked measured but has null value")
        for key in ("source", "sample_unit","spatial_grain","temporal_grain","population","aggregation",
                    "comparison_reference","execution_mode","source_exposure","implementation_sha",
                    "evidence_locator"):
            if not isinstance(m.get(key),str) or not m[key].strip():e.append(id+" missing "+key)
        if m.get("execution_mode") not in data["evidence_modes"]:e.append(id+" invalid execution_mode")
        if id=="post_writeback_sse" and m.get("semantic_claim") in ("video_quality","video_noncollapse_smoke"):
            e.append("local SSE masquerades as video quality")
        if id=="synthetic_backend_speedup" and m.get("semantic_claim") in ("native_speedup","video_speed_quality_pareto"):
            e.append("synthetic backend speedup masquerades as native deployed speed")
    claim=r.get("claim")
    if claim not in data["claims"]:e.append("unknown claim")
    else:
        cl=data["claims"][claim]
        present={m.get("level") for m in entries if isinstance(m,dict)}
        presentids={m.get("metric_id") for m in entries if isinstance(m,dict)}
        if not set(cl["required_levels"]).issubset(present):e.append("claim exceeds actually measured evidence levels")
        if not set(cl.get("required_metrics",[])).issubset(presentids):e.append("claim missing required metrics")
        if claim in ("video_noncollapse_smoke","video_speed_quality_pareto","independent_video_quality"):
            if not r.get("full_student_free_running") or not r.get("paired_baseline_full_video"):
                e.append("video claim lacks true full free-running paired student evidence")
        if claim=="video_speed_quality_pareto" and not r.get("matched_prior_art_or_strong_fixed"):
            e.append("competitive claim lacks matched strong comparison")
    return e
def main():
    data=catalog()
    errors=validate_catalog(data)
    files=sorted(RECORDS.glob("*.json")) if RECORDS.exists() else []
    for f in files:
        try:
            errors += [f"{f.name}: {s}" for s in validate_record(json.loads(f.read_text(encoding="utf-8")),data)]
        except (ValueError,OSError) as exc:
            errors.append(f"{f.name}: {exc}")
    for e in errors:print(e,file=sys.stderr)
    print(f"FFNDS metric levels: 6, metrics: {len(data.get('metrics',[]))}, declared result records: {len(files)}, errors: {len(errors)}")
    return 1 if errors else 0
if __name__=="__main__":raise SystemExit(main())
