"""Source-cluster paired metric comparison with explicit n, tails and bootstrap uncertainty.

Accepts historical FFNDS paired CSV columns (source_group,dimension,D_Q,candidate),
or new CSV (source_id,image_id,seed,metric,dense,candidate). A row is a pair;
seeds are nested within image clusters. Official metrics are never synthesized.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def quantile(values, q):
    a = sorted(values)
    if not a:
        return None
    pos = (len(a) - 1) * q
    low = int(pos)
    return a[low] + (a[min(low + 1, len(a) - 1)] - a[low]) * (pos - low)


def paired_report(records, *, replicates=2000, seed=42):
    if not records:
        raise ValueError("No actual paired metric records")
    # image cluster -> source/prompt -> repeated seeds/samples; each level has equal weight.
    by_dim_cluster = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    source_images, seen = {}, set()
    for row in records:
        dim = row.get("metric") or row.get("dimension")
        sid = row.get("source_id") or row.get("source_group")
        cluster = row.get("image_id") or sid
        seed_id = row.get("seed")
        sample_id = row.get("sample_index")
        if not dim or not sid or not cluster:
            raise ValueError("Require dimension, source_id and an image-level cluster")
        if sid in source_images and source_images[sid] != cluster:
            raise ValueError("A source ID maps to multiple image clusters: " + str(sid))
        source_images[sid] = cluster
        # A sample index is nested within a seed; neither field alone identifies a pair.
        key = (str(dim), str(sid), str(seed_id if seed_id not in (None, "") else 0),
               str(sample_id if sample_id not in (None, "") else 0))
        if key in seen:
            raise ValueError("Duplicated paired source/seed/sample/metric " + str(key))
        seen.add(key)
        dense = float(row.get("dense", row.get("D_Q")))
        candidate = float(row["candidate"])
        if not math.isfinite(dense) or not math.isfinite(candidate):
            raise ValueError("Actual metric values must be finite; missing remains NOT_RUN")
        by_dim_cluster[dim][cluster][sid].append((dense, candidate))
    result = {"schema": "ffnds.cluster.paired.metrics.v1",
              "independent_unit": "image cluster, equal-weight source conditions within each image; seeds and samples nested",
              "metrics": {}, "external_quality": "ACTUAL_SUBMITTED_VALUES_ONLY"}
    for dimension, clusters in sorted(by_dim_cluster.items()):
        pairs = []
        for cluster, sources in sorted(clusters.items()):
            source_means = [(sid, mean(v[0] for v in rows),
                             mean(v[1] for v in rows), len(rows))
                            for sid, rows in sorted(sources.items())]
            a = mean(item[1] for item in source_means)
            b = mean(item[2] for item in source_means)
            pairs.append({"cluster": cluster,
                          "n_source_conditions": len(source_means),
                          "n_sample_repetitions": sum(item[3] for item in source_means),
                          "source_conditions": [{"source_id": sid, "dense": d, "candidate": q,
                                                 "n_repetitions": count}
                                                for sid, d, q, count in source_means],
                          "dense": a, "candidate": b, "delta": b - a})
        diffs = [p["delta"] for p in pairs]
        baseline, candidate = mean(p["dense"] for p in pairs), mean(p["candidate"] for p in pairs)
        source_drops = [(p["dense"] - p["candidate"]) / p["dense"]
                        if p["dense"] else None for p in pairs]
        present = [v for v in source_drops if v is not None]
        n = len(pairs)
        bounds = None
        if n >= 2 and replicates > 0:
            rng = random.Random(seed)
            draws = [mean(rng.choices(diffs, k=n)) for _ in range(replicates)]
            bounds = [quantile(draws, .025), quantile(draws, .975)]
        result["metrics"][dimension] = {
            "independent_source_clusters": n,
            "actual_sample_records": sum(p["n_sample_repetitions"] for p in pairs),
            "baseline_cluster_mean": baseline,
            "candidate_cluster_mean": candidate,
            "absolute_candidate_minus_dense": candidate - baseline,
            "relative_percent": 100 * (candidate / baseline - 1) if baseline else None,
            "relative_percent_status": "DEFINED" if baseline else "UNDEFINED_ZERO_BASELINE",
            "delta_bootstrap_95_ci": bounds,
            "bootstrap_unit": "image_cluster" if bounds is not None else "DESCRIPTIVE_ONLY",
            "source_relative_drop_median": median(present) if present else None,
            "source_relative_drop_p90": quantile(present, .9),
            "sources_relative_drop_over_2pct": sum(v > .02 for v in present),
            "source_results": pairs,
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.bootstrap < 0:
        raise ValueError("Nonnegative bootstrap repetitions required")
    with args.pairs.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    report = paired_report(records, replicates=args.bootstrap, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps({"status": "SOURCE_CLUSTER_PAIRED_STATISTICS_ONLY",
                      "metric_names": sorted(report["metrics"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
