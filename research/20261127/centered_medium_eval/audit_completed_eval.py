"""Recompute paired statistics and decoded-video diagnostics from a collected run."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, median

import numpy as np

from centered_runtime import file_hash
from summarize_medium_eval import ARMS, DIMS, MARGIN, SEED, inspect_metric_rows, paired_indices, quality_summary, summarize_timing


def load(path):
    return json.loads(path.read_bytes())


def matches(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(matches(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(matches(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)
    return a == b


def rms(values):
    return math.sqrt(fmean(x*x for x in values))


def audit(root, cohort):
    summary, run = (load(root/name) for name in ("SUMMARY.json", "runs.json"))
    run_sha = file_hash(root/"runs.json")
    assert summary["generation_run_sha256"] == run_sha and run["status"] == "complete"
    image_ids = [r["source_id"] for r in cohort["sources"]]
    strata = [r["stratum"] for r in cohort["sources"]]
    assert len(set(image_ids)) == len(set(r["source_cluster_id"] for r in cohort["sources"])) == 50
    row_map = {r["source_id"]:r for r in run["quality"]}
    assert set(row_map) == set(image_ids) and len(run["quality"]) == len(row_map)
    for row in run["quality"]:
        arms = row["arms"]
        assert set(arms) == set(ARMS)
        tuples = [(v["record"]["initial"], v["record"]["flows"][0], v["record"]["latents"][0]) for v in arms.values()]
        assert len(set(tuples)) == 1 and all(v["final_latent_saved"] for v in arms.values())
    rng = np.random.default_rng(SEED)
    indices = paired_indices(strata, rng)
    quality, effects = {}, {}
    for dimension in DIMS:
        score = root/"scores"/(dimension+".json")
        raw = load(score)
        old = summary["dimensions"][dimension]
        assert raw["dimension"] == dimension and raw["binding_sha256"] == summary["binding_sha256"]
        assert file_hash(score) == old["raw_score_sha256"] and old["status"] == "COMPLETE"
        table, counts, missing, invalid = inspect_metric_rows(raw, image_ids)
        assert all(not item for item in (*missing.values(), *invalid.values())) and matches(counts, old["score_audit"])
        arrays = {a:np.asarray([table[a][sid] for sid in image_ids]) for a in ARMS}
        assert matches({a:s.tolist() for a,s in arrays.items()}, old["source_scores"])
        assert image_ids == old["sources"]
        effects[dimension] = quality_summary(arrays, indices)
        assert matches(effects[dimension], old["comparisons"]), dimension
        quality[dimension] = table
    decisions = [t["centered/dense"]["noninferiority"] for t in effects.values()]
    combined = "FAIL" if "FAIL" in decisions else "PASS" if all(x=="PASS" for x in decisions) else "INCONCLUSIVE"
    assert combined == summary["metric_noninferiority"]
    assert matches(summarize_timing(run,cohort,rng), summary["timing"])
    latent = {r["source_id"]:r["relative_l2"] for r in summary["latents"]["rows"]}
    assert set(latent) == set(image_ids) and summary["latents"]["status"] == "COMPLETE"
    assert matches(summary["latents"]["means"], {k:fmean(v[k] for v in latent.values()) for k in next(iter(latent.values()))})
    motion = load(root/"motion/MOTION_INDEX.json")
    video = load(root/"metric_binding.video_index.json")
    assert motion["generation_run_sha256"] == run_sha
    assert motion["metric_binding_sha256"] == video["metric_binding_sha256"] == summary["binding_sha256"]
    files = {(r["source_id"],r["arm"]):r for r in video["records"]}
    checks = {(r["source_id"],r["arm"]):r for r in motion["records"]}
    assert len(files) == len(checks) == 150
    by_source, public_media = [], []
    for source in cohort["sources"]:
        sid = source["source_id"]
        item = dict(source_id=sid,stratum=source["stratum"],category=source["category"],
                    blind_review=source["blind_review_selected"],timing=source["timing_selected"],
                    vbench={d:{a:quality[d][a][sid] for a in ARMS} for d in DIMS},
                    latent_relative_l2=latent[sid],metrics={})
        for arm in ARMS:
            check, clip = checks[sid,arm], files[sid,arm]
            path = root/"motion"/arm/(sid+".json")
            assert file_hash(path) == check["report_sha256"]
            diag = load(path)
            assert diag["arm"] == arm and diag["source_id"] == sid
            assert diag["generation_run_sha256"] == run_sha and diag["metric_binding_sha256"] == summary["binding_sha256"]
            confirmed = row_map[sid]["arms"][arm]["record"]["video"]
            assert clip["video_sha256"] == diag["candidate"]["sha256"] == confirmed["mp4_sha256"]
            assert clip["decoded_rgb24_sha256"] == confirmed["decoded_rgb24_sha256"]
            frame = diag["per_frame"]
            assert [f["frame"] for f in frame] == list(range(81))
            flows = [f["candidate_downsampled_Farneback_flow_mean_px"] for f in frame[1:]]
            pixel = rms(f["paired_RGB_RMSE_0to1"] for f in frame) if arm!="dense" else 0.0
            temporal = rms(f["paired_temporal_delta_RGB_RMSE_0to1"] for f in frame[1:]) if arm!="dense" else 0.0
            flow = fmean(flows)
            assert matches(flow, check["metrics"]["temporal_motion"]["downsampled_Farneback_mean_flow_px"]["mean"])
            item["metrics"][arm] = dict(rgb_rmse_to_dense_0_1=pixel,temporal_delta_rgb_rmse_to_dense_0_1=temporal,mean_flow_px_per_frame=flow)
            public_media.append(dict(source_id=sid,arm=arm,mp4_sha256=clip["video_sha256"],
                                     decoded_rgb24_sha256=clip["decoded_rgb24_sha256"],file=f"videos/{sid}/{arm}.mp4"))
        by_source.append(item)
    tail = sorted(by_source, key=lambda r:r["vbench"]["imaging_quality"]["centered"]-r["vbench"]["imaging_quality"]["dense"])
    means = {a:{k:fmean(r["metrics"][a][k] for r in by_source) for k in by_source[0]["metrics"][a]} for a in ARMS}
    ratios = [r["metrics"]["centered"]["mean_flow_px_per_frame"]/r["metrics"]["dense"]["mean_flow_px_per_frame"] for r in by_source]
    return dict(status="COMPLETE",job=run["job"],generation_run_sha256=run_sha,metric_binding_sha256=summary["binding_sha256"],
        quality_noninferiority=combined,quality_margin=MARGIN,quality_sources=len(image_ids),vbench_scores_verified=1050,
        effects=effects,imaging_tail_four=[r["source_id"] for r in tail[:4]],
        image_and_motion_means=means,median_centered_to_dense_flow=median(ratios),
        centered_lower_flow_sources=sum(k<1 for k in ratios),timing=summary["timing"],latents=summary["latents"]["means"],
        blind_review_status="awaiting human review",per_source=by_source,media=public_media)


def save_csv(path, rows):
    table = []
    for r in rows:
        vals = {"source_id":r["source_id"],"stratum":r["stratum"],"category":r["category"],"blind_review":r["blind_review"]}
        vals.update({f"{dim}_{arm}":v for dim,data in r["vbench"].items() for arm,v in data.items()})
        vals.update({f"{dim}_centered_minus_dense_pp":100*(data["centered"]-data["dense"]) for dim,data in r["vbench"].items()})
        vals.update({f"{metric}_{arm}":v for arm,data in r["metrics"].items() for metric,v in data.items()})
        vals.update({"latent_relative_l2_"+k.replace("/","_vs_"):x for k,x in r["latent_relative_l2"].items()})
        table.append(vals)
    with path.open("w",encoding="utf-8",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("cohort",type=Path)
    args = parser.parse_args()
    data = audit(args.root,load(args.cohort))
    (args.root/"QUALITY_VERIFIED.json").write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    save_csv(args.root/"PER_SOURCE_METRICS.csv",data["per_source"])
    print("Verified",data["quality_sources"],"source triplets and",data["vbench_scores_verified"],"scores",data["quality_noninferiority"])
    print(data["image_and_motion_means"])
