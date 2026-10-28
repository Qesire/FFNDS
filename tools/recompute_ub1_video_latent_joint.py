#!/usr/bin/env python3
"""Recompute 20-source UB1 video/state evidence from FOUR original result JSONs.

Reads actual repo reports; no model, GPU, video decoding or repository mutation.
The image cluster is the independent unit. All reported CIs are exploratory.
"""
import argparse
import json
import math
from pathlib import Path

NATIVE = "research/20261027/UB1_CONSUMER_DISJOINT20_NATIVE_RESULTS_20261027.json"
LATENT = "research/20261027/UB1_DISJOINT20_LATENT_INCREMENT_RESULTS_20261027.json"
QUALITY = "research/20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.json"
VISUAL = "research/20261027/UB1_DISJOINT20_INDEPENDENT_VISUAL_REVIEW_20261027.json"


def read(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def mean(items):
    assert items
    return sum(items) / len(items)


def pctile(x, p):
    ordered = sorted(x)
    index = (len(ordered) - 1) * p
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


class LCG:
    """Replicate the stored 20-source bootstrap draw stream."""
    def __init__(self, seed):
        self.state = seed

    def index(self, n):
        self.state = (1664525 * self.state + 1013904223) & 0xffffffff
        return int((self.state / 4294967296) * n)


def bootstrap(rows, measure, rng, samples=10000):
    v = [measure([rows[rng.index(len(rows))] for _ in rows]) for _ in range(samples)]
    return [pctile(v, 0.025), pctile(v, 0.975)]


def ranks(arr):
    indexed = sorted(enumerate(arr), key=lambda t: t[1])
    out = [None] * len(arr)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and indexed[j][1] == indexed[i][1]:
            j += 1
        for k in range(i, j):
            out[indexed[k][0]] = (i + j + 1) / 2
        i = j
    return out


def pearson(x, y):
    mx, my = mean(x), mean(y)
    sx, sy = [z - mx for z in x], [z - my for z in y]
    den = math.sqrt(sum(t * t for t in sx) * sum(t * t for t in sy))
    return sum(a * b for a, b in zip(sx, sy)) / den if den else None


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def cohort_stats(rows, rng):
    means = []
    for k in range(4):
        means.append({
            "step": k,
            "latent_error": mean([r["steps"][k]["E"] for r in rows]),
            "temporal_difference_error": mean([r["steps"][k]["T"] for r in rows]),
            "increment": mean([r["steps"][k]["I"] for r in rows]),
            "signed_cross": mean([r["steps"][k]["C"] for r in rows]),
            "temporal_increment": mean([r["steps"][k]["IT"] for r in rows]),
            "temporal_signed_cross": mean([r["steps"][k]["CT"] for r in rows]),
            "positive_cross_sources": sum(r["steps"][k]["C"] > 0 for r in rows),
        })
    ratio = lambda z: mean([r["raft_student"] for r in z]) / mean([r["raft_dense"] for r in z])
    img = lambda z: mean([r["imaging_student"] - r["imaging_dense"] for r in z])
    wall = lambda z: mean([r["wall_dense"] - r["wall_student"] for r in z])
    raft_delta = lambda z: mean([r["raft_student"] - r["raft_dense"] for r in z])
    subject_delta = lambda z: mean([r["subject_consistency_student"] - r["subject_consistency_dense"] for r in z])
    ratio_ci = bootstrap(rows, ratio, rng)
    imaging_ci = bootstrap(rows, img, rng)
    raft_ci = bootstrap(rows, raft_delta, rng)
    subject_ci = bootstrap(rows, subject_delta, rng)
    wall_ci = bootstrap(rows, wall, rng)
    return {
        "n_image_clusters": len(rows),
        "raft_ratio_of_means": ratio(rows),
        "raft_ratio_bootstrap_95": ratio_ci,
        "raft_ratio_per_source_median": pctile([r["raft_student"] / r["raft_dense"] for r in rows], .5),
        "raft_drop_sources": sum(r["raft_student"] < r["raft_dense"] for r in rows),
        "imaging_delta": img(rows),
        "imaging_delta_bootstrap_95": imaging_ci,
        "raft_delta_bootstrap_95": raft_ci,
        "subject_consistency_delta_bootstrap_95": subject_ci,
        "wall_saved_seconds_bootstrap_95": wall_ci,
        "imaging_up_but_raft_down": sum(r["imaging_student"] > r["imaging_dense"] and r["raft_student"] < r["raft_dense"] for r in rows),
        "steps": means,
        "latent_error_monotone_sources": sum(all(r["steps"][k]["E"] > r["steps"][k-1]["E"] for k in range(1, 4)) for r in rows),
        "temporal_error_monotone_sources": sum(all(r["steps"][k]["T"] > r["steps"][k-1]["T"] for k in range(1, 4)) for r in rows),
        "step3_signed_cross_fraction": means[3]["signed_cross"] / (means[3]["increment"] + means[3]["signed_cross"]),
        "state_temporal_vs_raft_ratio_spearman": spearman([r["steps"][3]["T"] for r in rows], [r["raft_student"] / r["raft_dense"] for r in rows]),
        "state_temporal_vs_dense_raft_spearman": spearman([r["steps"][3]["T"] for r in rows], [r["raft_dense"] for r in rows]),
        "wall_saved_seconds_instrumented_mean": wall(rows),
        "wall_saved_seconds_instrumented_median": pctile([r["wall_dense"] - r["wall_student"] for r in rows], .5),
        "wall_faster_candidate_sources": sum(r["wall_student"] < r["wall_dense"] for r in rows),
    }


def compute(root):
    native, lat, qual, vis = [read(root, p) for p in (NATIVE, LATENT, QUALITY, VISUAL)]
    assert len(native["pairs"]) == len(lat["cases"]) == len(vis["sources"]) == 20
    metrics = {}
    for key, result in qual["reports"]["dense"]["metrics"].items():
        records = {}
        for x in result["source_results"]:
            source_group = ":".join(x["source_conditions"][0]["source_id"].split(":")[-2:])
            assert source_group not in records
            records[source_group] = x
        metrics[key] = records
    rows = []
    for idx, (n, l, v) in enumerate(zip(native["pairs"], lat["cases"], vis["sources"])):
        group = n["source_group"]
        assert idx == n["source_index"] == l["index"] == v["index"]
        assert n["source_id"] == l["source_id"] == v["source_id"]
        assert group == l["source_group"] == v["source_group"]
        assert n["seed"] == l["seed"] == v["seed"]
        assert n["initial_noise_actual_file_byte_equal"]
        assert len(l["steps"]) == 4
        for arm in ("dense", "candidate"):
            assert n["arms"][arm]["unique_FFN_step_block_positions"] == 160
            assert n["arms"][arm]["full_flows"] == 4
            assert n["arms"][arm]["recorded_frames"] == 81
        state = []
        for k, s in enumerate(l["steps"]):
            assert s["step"] == k
            e, t = s["statistics"]["identity"], s["statistics"]["latent_time_first_difference"]
            for item in (e, t):
                resid = item["previous_error_energy"] + item["increment_difference_energy"] + item["twice_previous_increment_dot"] - item["current_error_energy"]
                assert abs(resid) <= item["closure_tolerance"] + 1e-9
            state.append({
                "E": e["current_error_energy"],
                "T": t["current_error_energy"],
                "I": e["increment_difference_energy"],
                "C": e["twice_previous_increment_dot"],
                "IT": t["increment_difference_energy"],
                "CT": t["twice_previous_increment_dot"],
            })
        row = {
            "source_group": group,
            "stratum": "subject" if group.startswith("i2v_subject:") else "background",
            "seed": n["seed"],
            "raft_dense": metrics["raft_continuous_mean"][group]["dense"],
            "raft_student": metrics["raft_continuous_mean"][group]["candidate"],
            "imaging_dense": metrics["imaging_quality"][group]["dense"],
            "imaging_student": metrics["imaging_quality"][group]["candidate"],
            "subject_consistency_dense": metrics["subject_consistency"][group]["dense"],
            "subject_consistency_student": metrics["subject_consistency"][group]["candidate"],
            "wall_dense": n["arms"]["dense"]["generation_wall_seconds_including_video_latent_save_and_hash"],
            "wall_student": n["arms"]["candidate"]["generation_wall_seconds_including_video_latent_save_and_hash"],
            "steps": state,
            "contact_sheet_observation": v["observation"],
        }
        assert row["raft_dense"] > 0
        rows.append(row)
    assert len(set(r["source_group"] for r in rows)) == 20
    assert sum(r["stratum"] == "subject" for r in rows) == 10
    assert sum(r["stratum"] == "background" for r in rows) == 10
    rng = LCG(20261029)
    return {
        "schema": "ffnds.ub1_full_depth_video_recompute_from_original_json.v1",
        "status": "ACTUAL_RAW_JSON_RECALCULATION_NO_NEW_MODEL_OR_VIDEO",
        "input_reports": [NATIVE, LATENT, QUALITY, VISUAL],
        "methodology": "paired 20 historical exposed image clusters, one seed; 10k image-cluster bootstraps using saved LCG(20261029) stream and cohort/metric order; all 160-site 4-step generation; contact sheets are not human full-play",
        "all": cohort_stats(rows, rng),
        "subject": cohort_stats([r for r in rows if r["stratum"] == "subject"], rng),
        "background": cohort_stats([r for r in rows if r["stratum"] == "background"], rng),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = compute(args.root)
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
