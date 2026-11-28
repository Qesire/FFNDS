#!/usr/bin/env python3
"""Recompute schedule and motion receipts with Python's standard library.

No model inference, tensor decoding, video decoding, or external dependencies.
Relative L2 values are checked between receipts, not independently recomputed
from unpublished trace tensors. Motion arrays are the recorded decoder outputs.
"""
import argparse
import gzip
import hashlib
import json
import math
import pathlib
import statistics as st


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    data = pathlib.Path(path).read_bytes()
    if str(path).endswith(".gz"):
        data = gzip.decompress(data)
    return json.loads(data), data


def main():
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    source_default = here if (here / "C_SCHEDULE_196295.json.gz").exists() else here / "numeric_source"
    reference_candidates = [here / "ONLINE_QUALITY_194188.json.gz", here / "ONLINE_QUALITY_194188.json",
                            here.parent / "round5_online/numeric_source/ONLINE_QUALITY_194188.json"]
    reference_default = next((p for p in reference_candidates if p.exists()), reference_candidates[0])
    parser.add_argument("--source-dir", type=pathlib.Path, default=source_default)
    parser.add_argument("--reference-quality", type=pathlib.Path, default=reference_default)
    parser.add_argument("--output", type=pathlib.Path, default=here / "schedule_evidence_recomputed.json")
    args = parser.parse_args()
    raw, raw_bytes = load(args.source_dir / "C_SCHEDULE_196295.json.gz")
    motion, motion_bytes = load(args.source_dir / "C_SCHEDULE_MOTION_196295.json")
    summary, _ = load(args.source_dir / "C_SCHEDULE_SUMMARY.json")
    human, _ = load(args.source_dir / "C_SCHEDULE_HUMAN_REVIEW_20261125.json")
    reference, reference_bytes = load(args.reference_quality)
    checks = []

    def check(label, condition):
        checks.append({"name": label, "passed": bool(condition)})
        if not condition:
            raise AssertionError(label)

    def close(label, actual, expected, tol=1e-10):
        check(label, math.isclose(actual, expected, rel_tol=tol, abs_tol=tol))

    check("raw schedule payload sha256", sha256(raw_bytes) == summary["raw_schedule_receipt_sha256"])
    check("motion raw bytes sha256", sha256(motion_bytes) == human["motion_receipt_sha256"])
    check("reference raw quality sha256", sha256(reference_bytes) == raw["reference_quality_sha256"])
    check("completed schedule", raw["status"] == "complete")
    check("formal render and pair counts", len(raw["renders"]) == 12 and len(raw["pairs"]) == 6)
    ref_c = {x["source"]: x for x in reference["renders"] if x["arm"] == "C"}
    index = {(x["source"], x["repeat"], x["arm"]): x for x in raw["renders"]}
    check("unique formal renders", len(index) == 12)
    pairs = []
    for published in raw["pairs"]:
        src, repeat = published["source"], published["repeat"]
        c = index[src, repeat, "C"]["record"]
        c0 = index[src, repeat, "C0"]["record"]
        wall_ratio = c0["wall"] / c["wall"]
        gpu_ratio = sum(c0["step_ms"]) / sum(c["step_ms"])
        delta = [(b - a) / 1000 for a, b in zip(c["step_ms"], c0["step_ms"])]
        close(f"{src} r{repeat} wall ratio", wall_ratio, published["wall_ratio"])
        close(f"{src} r{repeat} gpu ratio", gpu_ratio, published["gpu_ratio"])
        for i, (value, expected) in enumerate(zip(delta, published["step_delta_ms"])):
            close(f"{src} r{repeat} step{i} delta", value, expected / 1000)
        actual_order = [x["arm"] for x in raw["renders"] if x["source"] == src and x["repeat"] == repeat]
        check(f"{src} r{repeat} order", actual_order == published["order"])
        pairs.append({"source": src, "repeat": repeat, "order": actual_order,
                      "wall_C_s": c["wall"], "wall_C0_s": c0["wall"],
                      "gpu_C_s": sum(c["step_ms"]) / 1000,
                      "gpu_C0_s": sum(c0["step_ms"]) / 1000,
                      "wall_ratio": wall_ratio, "gpu_ratio": gpu_ratio,
                      "wall_saving_s": c["wall"] - c0["wall"],
                      "gpu_saving_s": -sum(delta), "step_C0_minus_C_s": delta})
    timing = {"pairs": pairs,
              "paired_median_wall_ratio": st.median(p["wall_ratio"] for p in pairs),
              "paired_median_gpu_ratio": st.median(p["gpu_ratio"] for p in pairs),
              "paired_median_wall_saving_s": st.median(p["wall_saving_s"] for p in pairs),
              "paired_median_gpu_saving_s": st.median(p["gpu_saving_s"] for p in pairs),
              "wall_ratio_range": [min(p["wall_ratio"] for p in pairs), max(p["wall_ratio"] for p in pairs)],
              "gpu_ratio_range": [min(p["gpu_ratio"] for p in pairs), max(p["gpu_ratio"] for p in pairs)],
              "mean_step_C0_minus_C_s": [st.mean(p["step_C0_minus_C_s"][i] for p in pairs) for i in range(4)],
              "source_median_wall_ratio": {s: st.median(p["wall_ratio"] for p in pairs if p["source"] == s) for s in raw["sources"]},
              "order_median_wall_ratio": {order: st.median(p["wall_ratio"] for p in pairs if ",".join(p["order"]) == order) for order in ["C,C0", "C0,C"]},
              "all_6_wall_and_gpu_pairs_faster": all(p["wall_ratio"] < 1 and p["gpu_ratio"] < 1 for p in pairs),
              "arm_medians": {}}
    for arm in ["C", "C0"]:
        records = [x["record"] for x in raw["renders"] if x["arm"] == arm]
        timing["arm_medians"][arm] = {"wall_s": st.median(r["wall"] for r in records),
                                              "gpu_s": st.median(sum(r["step_ms"]) / 1000 for r in records),
                                              "peak_allocated_gib": st.median(r["peak_allocated"] / 2**30 for r in records)}
        close(f"{arm} peak allocated summary", timing["arm_medians"][arm]["peak_allocated_gib"], summary["peak_allocated_gib"][arm])
    close("paired median wall summary", timing["paired_median_wall_ratio"], summary["paired_wall_ratio"])
    close("paired median gpu summary", timing["paired_median_gpu_ratio"], summary["paired_gpu_ratio"])
    close("paired median wall raw", timing["paired_median_wall_ratio"], raw["paired_wall_ratio"])
    close("paired median gpu raw", timing["paired_median_gpu_ratio"], raw["paired_gpu_ratio"])
    for i, value in enumerate(timing["mean_step_C0_minus_C_s"]):
        close(f"step{i} mean delta summary", value, summary["mean_step_delta_s"][i])

    identity = []
    for r in raw["renders"]:
        src, arm = r["source"], r["arm"]
        q = ref_c[src] if arm == "C" else raw["quality"][src]
        rec, expected = r["record"], q["record"]
        fields = {k: rec[k] == expected[k] for k in ["initial", "flows", "latents", "video", "experts"]}
        check(f"{src} {arm} r{r['repeat']} quality/timing identity", all(fields.values()))
        ff = rec["counters"]
        check(f"{src} {arm} r{r['repeat']} wide8192 identity", ff["variant"] == "wide8192" and ff["tile_rows"] == 8192 and ff["anchors"] == ff["hits"] == 80)
        expected_att = {"attention_sites": 160, "anchors": 80 if arm == "C" else 40,
                        "hits": 80 if arm == "C" else 120,
                        "native_probe_queries": 5120 if arm == "C" else 7680}
        check(f"{src} {arm} r{r['repeat']} attention counters", all(r["attention"][k] == v for k, v in expected_att.items()))
        if arm == "C0":
            check(f"{src} C0 r{r['repeat']} route builds", r["attention"]["route_builds"] == 80)
        identity.append({"source": src, "arm": arm, "repeat": r["repeat"], "matched_fields": fields})

    trajectory = {}
    motion_out = {}
    for src in raw["sources"]:
        q0, qc = raw["quality"][src], ref_c[src]
        check(f"{src} C/C0 same initial", q0["record"]["initial"] == qc["record"]["initial"])
        check(f"{src} relative L2 summary agreement", q0["relative_l2_vs_C"] == summary["quality"][src]["relative_l2"])
        ids = {(r["step"], r["layer"]): r["support_sha256"] for r in qc["attention_anchor_identity"]}
        matched = {str(step): [r["layer"] for r in q0["attention_anchor_identity"]
                              if r["step"] == step and r["support_sha256"] == ids[step, r["layer"]]] for step in [0, 2]}
        check(f"{src} independently matched route hashes", {step: len(v) for step, v in matched.items()} == q0["route_matches"])
        trajectory[src] = {"same_initial": True, "matching_route_layers": matched,
                           "recorded_relative_L2_vs_C_by_step": q0["relative_l2_vs_C"],
                           "reported_video_RGB": summary["quality"][src]["video"],
                           "interpretation": "Relative-L2 denominators vary with step. The sequence is not an absolute norm growth estimate or operator amplification factor. Route hash inequality is not support overlap distance."}
        motion_out[src] = {"clips": {}, "C0_over_C": {}, "C_over_A": {}}
        videos = {"A": reference["control"][src]["wide8192"]["video"], "C": qc["record"]["video"], "C0": q0["record"]["video"]}
        for arm, mm in motion["sources"][src]["metrics"].items():
            check(f"{src} {arm} motion mp4 receipt identity", videos[arm]["mp4_sha256"] == motion["sources"][src]["mp4_sha256"][arm])
            luma, flow = mm["adjacent_luma_mae_by_transition"], mm["flow_p90_px_by_transition"]
            check(f"{src} {arm} 81 frames and 80 transitions", mm["frames"] == 81 and len(luma) == len(flow) == 80)
            check(f"{src} {arm} geometry/fps", mm["geometry"] == [832, 464] and mm["fps"] == 16)
            computed = {"mean_adjacent_luma_mae": st.mean(luma), "late_adjacent_luma_mae": st.mean(luma[-20:]),
                        "median_flow_spatial_p90_px": st.median(flow), "late_median_flow_spatial_p90_px": st.median(flow[-20:]),
                        "mean_adjacent_luma_mae_by_quarter": [st.mean(luma[i:i+20]) for i in range(0, 80, 20)],
                        "median_flow_p90_by_quarter": [st.median(flow[i:i+20]) for i in range(0, 80, 20)],
                        "first_last_luma_mae_as_reported": mm["first_last_luma_mae"]}
            for k in ["mean_adjacent_luma_mae", "late_adjacent_luma_mae", "median_flow_spatial_p90_px", "late_median_flow_spatial_p90_px"]:
                close(f"{src} {arm} {k}", computed[k], mm[k])
            for i, value in enumerate(computed["mean_adjacent_luma_mae_by_quarter"]):
                close(f"{src} {arm} quarter{i+1} luma", value, mm["mean_adjacent_luma_mae_by_quarter"][i])
            motion_out[src]["clips"][arm] = computed
        for label, numerator, denominator in [("C0_over_C", "C0", "C"), ("C_over_A", "C", "A")]:
            a, b = motion_out[src]["clips"][numerator], motion_out[src]["clips"][denominator]
            for k in a:
                motion_out[src][label][k] = [v / w for v, w in zip(a[k], b[k])] if isinstance(a[k], list) else a[k] / b[k]

    result = {
        "scope": "Stdlib recomputation of logged timings, hashes and recorded per-transition motion arrays. No model or video rerun.",
        "plan_commit": raw["plan_commit"], "job": raw["job"], "reference_quality_job": raw["reference_quality_job"],
        "sha256": {"schedule_payload": sha256(raw_bytes), "motion_raw_bytes": sha256(motion_bytes), "reference_quality": sha256(reference_bytes)},
        "statistical_units": {"sources": 2, "repeats_per_source_per_arm": 3, "timing_pairs": 6,
                              "quality_videos_per_arm": 2, "transitions_per_video": 80,
                              "note": "Repeated timings and adjacent video transitions are nested observations, not additional independent sources."},
        "timing": timing, "quality_timing_identity": identity, "trajectory": trajectory,
        "motion": motion_out, "human_review": human,
        "checks_passed": len(checks), "checks": checks,
        "interpretive_limits": ["Recorded relative L2 was checked for receipt agreement; raw tensor norms were not re-evaluated.",
                                "First/last luma and RGB RMSE are logged aggregates, not reconstructed from transition arrays.",
                                "Flow is a classical optical-flow proxy, not a direct semantic-motion verdict.",
                                "Whole-step intervention identifies its total trajectory effect; the receipts do not decompose attention, FFN-anchor and solver mediation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks_passed": len(checks),
                      "wall_ratio": timing["paired_median_wall_ratio"], "gpu_ratio": timing["paired_median_gpu_ratio"],
                      "motion_C0_over_C": {s: {k: v for k, v in motion_out[s]["C0_over_C"].items() if not isinstance(v, list)} for s in raw["sources"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
