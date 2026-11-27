"""Independently reaggregate uploaded 50-source scores and diagnostic rows.

Uses per-video scores already produced by the official metric functions. No
model execution, video decoding, or optical-flow estimation takes place here.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ARMS = ("dense", "wide8192", "centered")
PAIRS = (("centered", "dense"), ("wide8192", "dense"), ("centered", "wide8192"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input-dir", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    names = ("QUALITY_50_SUMMARY.json", "COHORT_50_PUBLIC.json", "MEDIUM_EVALUATION_SPEC.json",
             "PER_SOURCE_METRICS_50.csv", "SCHOOL_EXECUTION_50.json")
    summary, cohort, spec = [json.loads((args.input_dir / name).read_text()) for name in names[:3]]
    execution = json.loads((args.input_dir / names[-1]).read_text())
    rows = list(csv.DictReader((args.input_dir / names[3]).open(newline="")))
    sources = [r["source_id"] for r in cohort["sources"]]
    assert len(sources) == len(set(sources)) == 50
    assert len({r["source_cluster_id"] for r in cohort["sources"]}) == 50
    assert [r["source_id"] for r in rows] == sources
    by_source = {r["source_id"]: r for r in rows}
    strata = np.asarray([r["stratum"] for r in cohort["sources"]])
    assert Counter(strata) == spec["quality"]["strata"]
    rng = np.random.default_rng(spec["metrics"]["bootstrap_seed"])
    reps = spec["metrics"]["bootstrap_replicates"]
    # One shared paired sample matrix for all seven dimensions and all arms.
    selections = []
    for stratum in sorted(set(strata)):
        eligible = np.flatnonzero(strata == stratum)
        selections.append(rng.choice(eligible, (reps, len(eligible)), replace=True))
    boot = np.concatenate(selections, axis=1)
    checks, dimensions = [], {}
    margin = spec["metrics"]["noninferiority_delta"]
    epsilon = spec["metrics"]["decision_numerical_tolerance"]
    for name in spec["metrics"]["dimensions"]:
        published = summary["dimensions"][name]
        assert published["sources"] == sources and published["status"] == "COMPLETE"
        scores = {arm: np.asarray(published["source_scores"][arm], dtype=float) for arm in ARMS}
        assert all(s.shape == (50,) and np.isfinite(s).all() for s in scores.values())
        for arm in ARMS:
            csv_values = np.asarray([float(by_source[s][name + "_" + arm]) for s in sources])
            checks.extend(abs(csv_values - scores[arm]).tolist())
        comparisons = {}
        for candidate, reference in PAIRS:
            key = candidate + "/" + reference
            delta = scores[candidate] - scores[reference]
            lo, hi = np.quantile(delta[boot].mean(axis=1), [.025, .975])
            status = ("PASS" if lo > margin + epsilon else
                      "FAIL" if hi < margin - epsilon else "INCONCLUSIVE")
            current = dict(candidate_mean=float(scores[candidate].mean()),
                           reference_mean=float(scores[reference].mean()),
                           delta=float(delta.mean()), ci95=[float(lo), float(hi)], n=50,
                           noninferiority=status, worst_delta=float(delta.min()),
                           best_delta=float(delta.max()), p10_delta=float(np.quantile(delta, .1)),
                           median_delta=float(np.median(delta)), better_sources=int((delta > 0).sum()),
                           worse_than_margin_sources=int((delta < margin).sum()))
            original = published["comparisons"][key]
            for field in ("candidate_mean", "reference_mean", "delta", "worst_delta", "best_delta", "p10_delta"):
                checks.append(abs(current[field] - original[field]))
            checks.extend(abs(np.asarray(current["ci95"]) - original["ci95"]).tolist())
            assert current["noninferiority"] == original["noninferiority"]
            current["strata"] = {stratum: dict(
                n=int((strata == stratum).sum()),
                delta=float(delta[strata == stratum].mean())) for stratum in sorted(set(strata))}
            comparisons[key] = current
        dimensions[name] = comparisons
    assert max(checks) < 1e-12, max(checks)
    md = np.asarray([float(r["imaging_quality_centered"]) - float(r["imaging_quality_dense"]) for r in rows])
    wd = np.asarray([float(r["imaging_quality_wide8192"]) - float(r["imaging_quality_dense"]) for r in rows])
    worst = np.argsort(md)[:4]
    tail = []
    for i in worst:
        row = rows[int(i)]
        tail.append(dict(source_id=row["source_id"], stratum=row["stratum"], category=row["category"],
                         preselected_blind_review=row["blind_review"].lower() == "true",
                         centered_minus_dense_pp=float(100 * md[i]), wide_minus_dense_pp=float(100 * wd[i]),
                         centered_minus_wide_pp=float(100 * (md[i] - wd[i]))))
    flow = {arm: np.asarray([float(r["mean_flow_px_per_frame_" + arm]) for r in rows]) for arm in ARMS}
    motion = {}
    for candidate, reference in PAIRS:
        ratio = flow[candidate] / flow[reference]
        delta = flow[candidate] - flow[reference]
        motion[candidate + "/" + reference] = dict(
            candidate_mean=float(flow[candidate].mean()), reference_mean=float(flow[reference].mean()),
            ratio_of_means=float(flow[candidate].mean() / flow[reference].mean()),
            median_source_ratio=float(np.median(ratio)), lower_sources=int((delta < 0).sum()),
            delta_ci95=np.quantile(delta[boot].mean(1), [.025, .975]).tolist(),
            role="existing_Farneback_diagnostic_not_monotone_quality")
    distances = {}
    for arm in ARMS:
        distances[arm] = {metric: float(np.mean([float(r[metric + "_" + arm]) for r in rows]))
                          for metric in ("rgb_rmse_to_dense_0_1", "temporal_delta_rgb_rmse_to_dense_0_1")}
    latent = {}
    for candidate, reference in PAIRS:
        field = "latent_relative_l2_" + candidate + "_vs_" + reference
        values = np.asarray([float(r[field]) for r in rows])
        original = summary["latents"]["means"][candidate + "/" + reference]
        assert abs(float(values.mean()) - original) < 1e-12
        latent[candidate + "/" + reference] = dict(mean=float(values.mean()), median=float(np.median(values)),
                                                   max=float(values.max()), n=50)
    for arm in ARMS:
        reported = execution["quality_result"]["image_and_motion_means"][arm]
        for metric, value in distances[arm].items():
            assert abs(reported[metric] - value) < 1e-12
        assert abs(reported["mean_flow_px_per_frame"] - flow[arm].mean()) < 1e-12
    result = dict(
        status="UPLOADED_50_SOURCE_SCORES_INDEPENDENTLY_REAGGREGATED",
        evidence_commit="a1eb64d63910b4d90b94ea4c641ef461c83d0d92",
        input_sha256={name: digest(args.input_dir / name) for name in names},
        source_clusters=50, score_count=7 * 3 * 50, numeric_checks=len(checks),
        maximum_absolute_recomputed_difference=max(checks),
        bootstrap_replicates=reps, bootstrap_unit="paired_source_within_fixed_stratum",
        strata=dict(Counter(strata)), categories=dict(Counter(r["category"] for r in rows)),
        selected_prior_exposure=sum(r["prior_exposure"] for r in cohort["sources"]),
        dimensions=dimensions,
        imaging_decomposition=dict(
            wide_fraction_of_signed_mean_decline=float(wd.mean() / md.mean()),
            centered_increment_fraction=float((md - wd).mean() / md.mean()),
            median_centered_dense_pp=float(100 * np.median(md)), tail_four=tail,
            tail_four_fraction_of_signed_net_decline=float(md[worst].sum() / md.sum()),
            tail_four_fraction_of_all_negative_declines=float(md[worst].sum() / md[md < 0].sum()),
            remaining46_mean_delta_pp_exploratory=float(100 * np.delete(md, worst).mean()),
            scope="posthoc_concentration_diagnosis_all_50_remain_in_primary_result"),
        flow=motion, decoded_distances=distances, latent_distances=latent,
        model_forwards=0, gpu_jobs=0, actual_video_decodes=0,
        human_review="repository_frozen_24_source_review_pending; user's current overall positive assessment is separate")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "numeric_checks", "maximum_absolute_recomputed_difference")}, indent=2))


if __name__ == "__main__":
    main()
