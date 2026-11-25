
"""Recompute the frozen C/A six-source confirmation, preserving measured timing boundaries.

Only existing receipts, timestamps and aggregates are read. No model inference,
video decoding or synthetic model data is used.
"""
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json, math, statistics as st

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def geomean(values):
    return math.exp(st.mean(math.log(x) for x in values))

def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    source_default=here if (here/"C_CONFIRM_RESULT_196504.json").exists() else here/"confirm_source"
    p.add_argument("--source-dir",type=Path,default=source_default)
    p.add_argument("--output",type=Path,default=here/"confirm_evidence_recomputed.json")
    a=p.parse_args()
    r=read(a.source_dir/"C_CONFIRM_RESULT_196504.json")
    summary=read(a.source_dir/"C_CONFIRM_SUMMARY.json")
    times=read(a.source_dir/"C_CONFIRM_SAVE_TIMESTAMPS_196504.json")
    method=read(a.source_dir/"C_CONFIRM_METHOD.json")
    manifest=read(a.source_dir/"C_CONFIRM_SOURCES.archive.json")
    observed=read(a.source_dir/"C_CONFIRM_OBSERVED.json")
    assert r["status"]=="complete"
    assert digest(a.source_dir/"C_CONFIRM_RESULT_196504.json")==summary["execution_receipt_sha256"]
    assert digest(a.source_dir/"C_CONFIRM_METHOD.json")==r["method_recipe_sha256"]==manifest["method_recipe_sha256"]
    assert digest(a.source_dir/"C_CONFIRM_OBSERVED.json")==manifest["inputs"]["observed_clusters"]["sha256"]
    assert r["manifest_sha256"]==manifest["original_frozen_manifest_sha256"]==summary["source_manifest_sha256"]
    assert r["sources_sha256"]==manifest["sources_sha256"]
    assert method["status"]=="FROZEN"
    selected=manifest["sources"]
    assert len(selected)==len(set(x["source_cluster_id"] for x in selected))==6
    excluded=set(manifest["exclusion_registry"]["excluded_cluster_ids"])
    assert not {x["source_cluster_id"] for x in selected}&excluded
    assert set(observed["observed_clusters"])<=excluded
    assert all(sum(x["stratum"]==k for x in selected)==2 for k in ("natural","structured","subject"))
    aggregates={x["source"]:x for x in summary["pairs"]}
    out=[]
    for i,(pair,selection) in enumerate(zip(r["pairs"],selected,strict=True)):
        sid=pair["source_id"]
        assert (sid,pair["cluster_id"])==(selection["source_id"],selection["source_cluster_id"])
        expected_order=["A","C"] if i%2==0 else ["C","A"]
        assert pair["order"]==expected_order
        arms={k:pair["arms"][k]["record"] for k in ("A","C")}
        assert arms["A"]["initial"]==arms["C"]["initial"]
        assert all(arms["A"][k][0]==arms["C"][k][0] for k in ("flows","latents"))
        for rec in arms.values():
            assert rec["experts"]==[0,0,1,1]
            assert (rec["counters"]["variant"],rec["counters"]["tile_rows"],
                    rec["counters"]["anchors"],rec["counters"]["hits"])==("wide8192",8192,80,80)
            assert (rec["video"]["frames"],rec["video"]["width"],rec["video"]["height"])==(81,832,464)
        counts=pair["arms"]["C"]["attention"]
        assert (counts["attention_sites"],counts["anchors"],counts["hits"],counts["native_probe_queries"])==(160,80,80,5120)
        stage={}
        for arm,rec in arms.items():
            row=times["rows"][f"paired_{arm}/{sid}"]
            duration=(datetime.fromisoformat(row["end"])-datetime.fromisoformat(row["start"])).total_seconds()
            assert math.isclose(duration,row["seconds"],abs_tol=1e-9)
            wall=rec["wall"]
            gpu=sum(rec["step_ms"])/1000
            stage[arm]={"wall_s":wall,"gpu_s":gpu,"save_path_interval_s":duration,
                        "wall_minus_save_s":wall-duration,
                        "residual_wall_minus_save_minus_gpu_s":wall-duration-gpu,
                        "peak_allocated_gib":rec["peak_allocated"]/2**30}
        A,C=stage["A"],stage["C"]
        delta={k:C[k]-A[k] for k in ("wall_s","gpu_s","save_path_interval_s","wall_minus_save_s","residual_wall_minus_save_minus_gpu_s")}
        ratios={"wall":C["wall_s"]/A["wall_s"],"gpu":C["gpu_s"]/A["gpu_s"],
                "wall_minus_save":C["wall_minus_save_s"]/A["wall_minus_save_s"]}
        assert math.isclose(ratios["wall"],pair["wall_ratio"],rel_tol=1e-12)
        assert math.isclose(ratios["gpu"],pair["gpu_ratio"],rel_tol=1e-12)
        assert math.isclose(ratios["wall_minus_save"],aggregates[sid]["without_save_ratio"],rel_tol=1e-12)
        motion=aggregates[sid]["motion"]
        luma=motion["C"]["mean_adjacent_luma_mae"]/motion["A"]["mean_adjacent_luma_mae"]
        flow=motion["C"]["median_flow_spatial_p90_px"]/motion["A"]["median_flow_spatial_p90_px"]
        assert math.isclose(luma,aggregates[sid]["luma_ratio"],rel_tol=1e-12)
        assert math.isclose(flow,aggregates[sid]["flow_ratio"],rel_tol=1e-12)
        out.append({"source":sid,"cluster":pair["cluster_id"],"stratum":pair["stratum"],
                    "order":pair["order"],"arms":stage,"ratios":ratios,"delta_C_minus_A":delta,
                    "step_delta_s":[(c-d)/1000 for c,d in zip(arms["C"]["step_ms"],arms["A"]["step_ms"])],
                    "motion_from_logged_aggregates":{"luma_ratio":luma,"flow_ratio":flow,
                        "A_flow_p90_time_median":motion["A"]["median_flow_spatial_p90_px"],
                        "C_flow_p90_time_median":motion["C"]["median_flow_spatial_p90_px"]}})
    geo={k:geomean(x["ratios"][k] for x in out) for k in ("wall","gpu","wall_minus_save")}
    assert math.isclose(geo["wall"],summary["wall_ratio"],rel_tol=1e-12)
    assert math.isclose(geo["gpu"],summary["gpu_ratio"],rel_tol=1e-12)
    assert math.isclose(geo["wall_minus_save"],summary["diagnostic_without_save_geomean"],rel_tol=1e-12)
    mean_delta={k:st.mean(x["delta_C_minus_A"][k] for x in out) for k in out[0]["delta_C_minus_A"]}
    mean_steps=[st.mean(x["step_delta_s"][s] for x in out) for s in range(4)]
    assert math.isclose(sum(mean_steps),mean_delta["gpu_s"],abs_tol=1e-9)
    assert math.isclose(mean_delta["save_path_interval_s"]+mean_delta["wall_minus_save_s"],mean_delta["wall_s"],abs_tol=1e-9)
    result={
      "source_commit":"c5e4ee8168708a8e57cdb981103c4dbf715e3bfa",
      "scope":"Receipt and timestamp recomputation only; not new model execution or video decoding",
      "identity":{"method_sha256":r["method_recipe_sha256"],"original_full_manifest_sha256":r["manifest_sha256"],
        "raw_execution_sha256":digest(a.source_dir/"C_CONFIRM_RESULT_196504.json"),
        "archived_redacted_manifest_sha256":digest(a.source_dir/"C_CONFIRM_SOURCES.archive.json"),
        "source_table_sha256":r["sources_sha256"]},
      "units":{"clusters":6,"pairs_per_cluster":1,"seed":42,"strata":3,"clusters_per_stratum":2,
        "scope":"fresh relative to the archived historical/existing-output/CAL-DEV registry; no universal claim about every prior contact"},
      "pairs":out,"source_equal_geomean":geo,"mean_C_minus_A_s":mean_delta,"mean_step_C_minus_A_s":mean_steps,
      "count_C_faster":{"wall":sum(x["ratios"]["wall"]<1 for x in out),
                      "gpu":sum(x["ratios"]["gpu"]<1 for x in out),
                      "wall_minus_save":sum(x["ratios"]["wall_minus_save"]<1 for x in out)},
      "warm_model_setup_excluded":{"init_s":r["init_seconds"],"wide_setup":r["setup"]},
      "memory":{"A_gib":out[0]["arms"]["A"]["peak_allocated_gib"],"C_gib":out[0]["arms"]["C"]["peak_allocated_gib"],
                "delta_gib":out[0]["arms"]["C"]["peak_allocated_gib"]-out[0]["arms"]["A"]["peak_allocated_gib"]},
      "visual_review":summary["visual_review"],
      "limits":["The full measured wall result includes every saved video and is the prespecified endpoint.",
                "Subtracting observed save intervals is a phase diagnostic, not a newly measured end-to-end speed.",
                "The save interval includes encoding and other save-path overhead; its causal origin and recurrence were not measured.",
                "Motion ratios are recomputed from logged aggregates; complete visual quality remains pending.",
                "Source selection was verified against the published redacted archive and execution checks, not re-run against the full private catalogue."],
      "descriptive_intervals_as_published":{"wall":summary["wall_cluster_bootstrap_95"],"gpu":summary["gpu_cluster_bootstrap_95"],
                "note":"These deterministic descriptive bootstrap bounds were read, not independently recomputed; no extra model replication."}
    }
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(a.output),"geomean":geo,"mean_delta":mean_delta,"mean_step":mean_steps,"visual_review":summary["visual_review"]},ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
