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
    by_dim_cluster = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in records:
        dim = row.get("metric") or row.get("dimension")
        sid = row.get("source_id") or row.get("source_group")
        cluster = row.get("image_id") or sid
        sample = row.get("sample_index", row.get("seed", 0))
        if not dim or not sid or not cluster:
            raise ValueError("Require dimension, source_id and an image-level cluster")
        key = (str(dim), str(sid), str(sample))
        if key in seen:
            raise ValueError("Duplicated paired source/seed/metric " + str(key))
        seen.add(key)
        dense = float(row.get("dense", row.get("D_Q")))
        candidate = float(row["candidate"])
        if not math.isfinite(dense) or not math.isfinite(candidate):
            raise ValueError("Actual metric values must be finite; missing remains NOT_RUN")
        by_dim_cluster[dim][cluster].append((dense, candidate))
    result = {"schema": "ffnds.cluster.paired.metrics.v1",
              "independent_unit": "image/source cluster; seeds and samples are repeated measures",
              "metrics": {}, "external_quality": "ACTUAL_SUBMITTED_VALUES_ONLY"}
    for dimension, clusters in sorted(by_dim_cluster.items()):
        pairs = []
        for cluster, rows in sorted(clusters.items()):
            a, b = mean(v[0] for v in rows), mean(v[1] for v in rows)
            pairs.append({"cluster": cluster, "n_sample_repetitions": len(rows),
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
