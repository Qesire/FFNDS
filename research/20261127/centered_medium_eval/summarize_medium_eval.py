"""Paired source-cluster summaries for the frozen centered medium evaluation.

Reads generated receipts and official per-video scores; does not run a model.
Missing dimensions remain missing. Latent distances are diagnostic, separate
from the seven prespecified quality functions and the human motion review.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from centered_runtime import file_hash


ARMS = ("dense", "wide8192", "centered")
COMPARISONS = (("centered", "dense"), ("wide8192", "dense"), ("centered", "wide8192"))
DIMS = ("subject_consistency", "background_consistency", "aesthetic_quality",
        "motion_smoothness", "imaging_quality", "i2v_subject", "i2v_background")
SEED, REPLICATES, MARGIN = 20261127, 10000, -0.005
EXPECTED_STRATA = {"structured": 33, "natural": 33, "subject": 34}
RANGE_EPSILON, DECISION_EPSILON = 1e-6, 1e-12


def read(path):
    return json.loads(Path(path).read_text())


def paired_indices(strata, rng):
    """Preserve the preregistered stratum counts and all paired arms."""
    labels = np.asarray(strata)
    groups = [np.flatnonzero(labels == label) for label in sorted(set(strata))]
    return np.concatenate([group[rng.integers(0, len(group), (REPLICATES, len(group)))]
                           for group in groups], axis=1)


def quality_summary(scores, indices):
    result = {}
    for candidate, reference in COMPARISONS:
        delta = scores[candidate] - scores[reference]
        lo, hi = np.quantile(delta[indices].mean(1), [.025, .975])
        # Keep a mathematically exact boundary inconclusive despite floating
        # subtraction (e.g. 0.795 - 0.8 is slightly below -0.005 in float64).
        verdict = ("PASS" if lo > MARGIN + DECISION_EPSILON else
                   "FAIL" if hi < MARGIN - DECISION_EPSILON else "INCONCLUSIVE")
        result[f"{candidate}/{reference}"] = dict(
            candidate_mean=float(scores[candidate].mean()), reference_mean=float(scores[reference].mean()),
            delta=float(delta.mean()), delta_percentage_points=float(100 * delta.mean()),
            ci95=[float(lo), float(hi)], n=len(delta), noninferiority=verdict,
            worst_delta=float(delta.min()), best_delta=float(delta.max()),
            p10_delta=float(np.quantile(delta, .1)))
    return result


def descriptive_pairs(scores, positions):
    """Means on an explicit paired subset; no confirmatory decision or CI."""
    result = {}
    for candidate, reference in COMPARISONS:
        a, b = scores[candidate][positions], scores[reference][positions]
        result[f"{candidate}/{reference}"] = dict(
            n=len(a), candidate_mean=float(a.mean()) if len(a) else None,
            reference_mean=float(b.mean()) if len(b) else None,
            delta=float((a - b).mean()) if len(a) else None,
            delta_percentage_points=float(100 * (a - b).mean()) if len(a) else None)
    return result


def inspect_metric_rows(data, ids):
    """Preserve partial/invalid score evidence without filling or clamping it."""
    expected = set(ids)
    lookup, audit, missing, invalid = {}, {}, {}, {}
    for arm in ARMS:
        rows = data.get("arms", {}).get(arm, {}).get("rows", [])
        if len({r["source"] for r in rows}) != len(rows):
            raise ValueError("Duplicate metric source rows: " + arm)
        if any(r["source"] not in expected for r in rows):
            raise ValueError("Unexpected metric source pairing: " + arm)
        missing[arm] = [sid for sid in ids if sid not in {r["source"] for r in rows}]
        lookup[arm], invalid[arm] = {}, []
        finite_values = []
        nonnumeric, nonfinite, out_of_range = [], [], []
        for row in rows:
            sid = row["source"]
            try:
                value = float(row["score"])
            except (KeyError, ValueError, TypeError, OverflowError):
                nonnumeric.append(sid)
                invalid[arm].append(sid)
                continue
            if not np.isfinite(value):
                nonfinite.append(sid)
                invalid[arm].append(sid)
                continue
            finite_values.append(value)
            if value < -RANGE_EPSILON or value > 1 + RANGE_EPSILON:
                out_of_range.append(sid)
                invalid[arm].append(sid)
                continue
            # Values in the numerical tolerance band are retained unchanged.
            lookup[arm][sid] = value
        audit[arm] = dict(
            n_rows=len(rows), n_finite=len(finite_values), n_valid=len(lookup[arm]),
            finite_min=min(finite_values) if finite_values else None,
            finite_max=max(finite_values) if finite_values else None,
            nonnumeric_source_ids=nonnumeric, nonfinite_source_ids=nonfinite,
            out_of_range_source_ids=out_of_range)
    return lookup, audit, missing, invalid


def available_pairs(lookup, ids):
    result = {}
    for candidate, reference in COMPARISONS:
        available = [sid for sid in ids if sid in lookup[candidate] and sid in lookup[reference]]
        a = np.asarray([lookup[candidate][sid] for sid in available], dtype=float)
        b = np.asarray([lookup[reference][sid] for sid in available], dtype=float)
        result[f"{candidate}/{reference}"] = dict(
            n=len(available), source_ids=available,
            candidate_mean=float(a.mean()) if len(a) else None,
            reference_mean=float(b.mean()) if len(b) else None,
            delta=float((a - b).mean()) if len(a) else None,
            delta_percentage_points=float(100 * (a - b).mean()) if len(a) else None,
            quality_decision_role="available_paired_mean_only_no_noninferiority_decision")
    return result


def summarize_timing(run, cohort, rng):
    sources = cohort["timing_source_ids"]
    people = {p["source_id"]: p for p in cohort["sources"]}
    if len(sources) != 12 or len(set(sources)) != 12 or len(run["timing"]) != 36:
        raise ValueError("Expected 12 timing sources, each with three rounds")
    if any(sid not in people for sid in sources):
        raise ValueError("Timing sources differ from the frozen quality cohort")
    if len({people[s]["source_cluster_id"] for s in sources}) != 12:
        raise ValueError("Timing requires 12 independent source clusters")
    if Counter(people[s]["stratum"] for s in sources) != {s: 4 for s in EXPECTED_STRATA}:
        raise ValueError("Timing requires four independent sources per frozen stratum")
    grouped = {sid: {} for sid in sources}
    for row in run["timing"]:
        sid, repeat = row["source_id"], row["repeat"]
        if sid not in grouped or repeat not in range(3) or repeat in grouped[sid]:
            raise ValueError("Unexpected, duplicate, or unpaired timing round")
        if set(row["arms"]) != set(ARMS):
            raise ValueError("Timing must have all three arms")
        for output in row["arms"].values():
            if output.get("final_latent_saved") or output.get("quality_capture_in_timing_window"):
                raise ValueError("Quality tensor capture cannot enter formal timing")
        grouped[sid][repeat] = row
    if any(set(rounds) != {0, 1, 2} for rounds in grouped.values()):
        raise ValueError("Incomplete timing rounds")
    indices = paired_indices([people[s]["stratum"] for s in sources], rng)
    result = dict(n_source_clusters=12, rounds_per_source=3,
                  aggregation="within-source arithmetic mean seconds; across-source geometric mean ratio",
                  memory_scope=run["memory_scope"], by_interval={})
    for interval, getter in (("wall_s", lambda r: r["wall"]),
                             ("gpu_s", lambda r: sum(r["step_ms"]) / 1000)):
        values = {a: np.asarray([[getter(grouped[s][j]["arms"][a]["record"])
                                  for j in range(3)] for s in sources], dtype=float) for a in ARMS}
        if any(not np.isfinite(v).all() or (v <= 0).any() for v in values.values()):
            raise ValueError("Timing contains nonfinite or nonpositive values")
        means = {a: values[a].mean(1) for a in ARMS}
        table = {}
        for candidate, reference in COMPARISONS:
            logs = np.log(means[reference] / means[candidate])
            speedup = float(np.exp(logs.mean()))
            lo, hi = np.quantile(np.exp(logs[indices].mean(1)), [.025, .975])
            table[f"{candidate}/{reference}"] = dict(
                reference_mean_seconds=float(means[reference].mean()),
                candidate_mean_seconds=float(means[candidate].mean()), speedup=speedup,
                time_saving_fraction=1 - 1 / speedup, speedup_ci95=[float(lo), float(hi)],
                by_source=[dict(source_id=s, reference_seconds=float(means[reference][i]),
                                candidate_seconds=float(means[candidate][i]),
                                speedup=float(np.exp(logs[i]))) for i, s in enumerate(sources)])
        result["by_interval"][interval] = table
    speed = result["by_interval"]["wall_s"]["centered/dense"]
    result["engineering_target_met"] = speed["speedup"] >= 1.10 and speed["speedup_ci95"][0] > 1.00
    result["peak_allocated_gib_by_arm"] = {
        a: max(grouped[s][r]["arms"][a]["record"]["peak_allocated"]
               for s in sources for r in range(3)) / 1024**3 for a in ARMS}
    return result


def latent_distances(run):
    from safetensors.torch import load_file
    results = []
    for source in run["quality"]:
        tensors = {}
        for arm in ARMS:
            record = source["arms"][arm].get("final_latent")
            if not record:
                raise ValueError("Missing final latent: " + source["source_id"] + "/" + arm)
            if file_hash(record["path"]) != record["sha256"]:
                raise ValueError("Final latent fingerprint mismatch")
            tensors[arm] = load_file(record["path"], device="cpu")[record["tensor_name"]].double()
        if len({tuple(t.shape) for t in tensors.values()}) != 1:
            raise ValueError("Unpaired final latent geometry")
        comparisons = {}
        for candidate, reference in COMPARISONS:
            value = (tensors[candidate] - tensors[reference]).norm() / tensors[reference].norm().clamp_min(1e-30)
            comparisons[f"{candidate}/{reference}"] = float(value)
        results.append(dict(source_id=source["source_id"], relative_l2=comparisons))
    return dict(status="COMPLETE", quality_decision_role="diagnostic_trajectory_distance",
                rows=results, means={f"{a}/{b}": float(np.mean([r["relative_l2"][f"{a}/{b}"] for r in results]))
                                     for a, b in COMPARISONS})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--metric-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--with-latents", action="store_true")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a new summary path")
    run, binding = read(args.run), read(args.binding)
    if run["status"] != "complete":
        raise ValueError("Generation and timing run is incomplete")
    if binding["generation_run_sha256"] != file_hash(args.run):
        raise ValueError("Metric binding refers to a different generation run")
    cohort = read(run["cohort"])
    if file_hash(run["cohort"]) != run["cohort_sha256"] or binding["cohort_sha256"] != run["cohort_sha256"]:
        raise ValueError("Source cohort fingerprint mismatch")
    ids = [s["source_id"] for s in cohort["sources"]]
    if (len(ids) != 100 or len(set(ids)) != 100 or len(run["quality"]) != 100
            or {r["source_id"] for r in run["quality"]} != set(ids)):
        raise ValueError("Expected all 100 quality source clusters")
    clusters = [s["source_cluster_id"] for s in cohort["sources"]]
    if any(not isinstance(c, str) or not c for c in clusters) or len(set(clusters)) != 100:
        raise ValueError("Quality requires 100 independent source_cluster_id values")
    if Counter(s["stratum"] for s in cohort["sources"]) != EXPECTED_STRATA:
        raise ValueError("Quality stratum counts must be structured33/natural33/subject34")
    rng = np.random.default_rng(SEED)
    indices = paired_indices([s["stratum"] for s in cohort["sources"]], rng)
    summary = dict(schema="FFNDS_CENTERED_MEDIUM_SUMMARY_V1", generation_run_sha256=file_hash(args.run),
                   binding_sha256=file_hash(args.binding), scorer_sha256=file_hash(__file__),
                   mode="CUSTOM_PAIRED", source_count=100, seed=42,
                   bootstrap=dict(replicates=REPLICATES, seed=SEED, unit="paired_cluster_within_fixed_strata",
                                  interval="two_sided_95_percentile_pointwise"),
                   study_noninferiority_margin=MARGIN,
                   score_range=dict(expected=[0, 1], numerical_tolerance=RANGE_EPSILON,
                                    clamp=False),
                   decision_numerical_tolerance=DECISION_EPSILON, dimensions={})
    dimension_status = []
    for dim in DIMS:
        path = args.metric_dir / (dim + ".json")
        if not path.exists():
            summary["dimensions"][dim] = dict(
                status="NOT_RUN", reason="Missing official dimension output", planned_n=100,
                actual_n_by_arm={arm: 0 for arm in ARMS},
                missing_source_ids_by_arm={arm: ids for arm in ARMS})
            dimension_status.append("INCONCLUSIVE")
            continue
        data = read(path)
        if data["dimension"] != dim or data["binding_sha256"] != file_hash(args.binding):
            raise ValueError("Metric dimension or input binding mismatch")
        if data["scope_sha256"] != binding["metric_scope_sha256"]:
            raise ValueError("Metric scope differs from frozen binding")
        lookup, audit, missing, invalid = inspect_metric_rows(data, ids)
        if any(invalid.values()) or any(missing.values()):
            summary["dimensions"][dim] = dict(
                status="ERROR" if any(invalid.values()) else "INCOMPLETE", planned_n=100,
                actual_n_by_arm={a: audit[a]["n_rows"] for a in ARMS},
                valid_n_by_arm={a: audit[a]["n_valid"] for a in ARMS},
                score_audit=audit, missing_source_ids_by_arm=missing,
                invalid_score_source_ids_by_arm=invalid,
                available_paired_comparisons=available_pairs(lookup, ids),
                noninferiority="INCONCLUSIVE", raw_score_sha256=file_hash(path))
            dimension_status.append("INCONCLUSIVE")
            continue
        scores = {arm: np.asarray([lookup[arm][sid] for sid in ids], dtype=float) for arm in ARMS}
        table = quality_summary(scores, indices)
        by_stratum = {
            stratum: dict(n_source_clusters=count, comparisons=descriptive_pairs(
                scores, np.asarray([i for i, row in enumerate(cohort["sources"])
                                    if row["stratum"] == stratum], dtype=int)))
            for stratum, count in EXPECTED_STRATA.items()}
        tagged_positions = np.asarray([
            i for i, row in enumerate(cohort["sources"])
            if dim in row.get("original_dimension_tags", [])], dtype=int)
        summary["dimensions"][dim] = dict(status="COMPLETE", comparisons=table,
                                          source_scores={a: scores[a].tolist() for a in ARMS}, sources=ids,
                                          score_audit=audit, by_stratum=by_stratum,
                                          original_tag_subset=dict(
                                              n_source_clusters=len(tagged_positions),
                                              quality_decision_role="descriptive_CUSTOM_PAIRED_appendix_not_official_total",
                                              comparisons=descriptive_pairs(scores, tagged_positions)),
                                          raw_score_sha256=file_hash(path))
        dimension_status.append(table["centered/dense"]["noninferiority"])
    summary["metric_noninferiority"] = ("FAIL" if "FAIL" in dimension_status else
                                           "PASS" if all(s == "PASS" for s in dimension_status) else "INCONCLUSIVE")
    summary["timing"] = summarize_timing(run, cohort, rng)
    summary["latents"] = latent_distances(run) if args.with_latents else dict(status="NOT_RUN")
    summary["motion_and_human_review"] = "Attach per-source existing flow diagnostics and blinded review; not inferred from metric PASS"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
