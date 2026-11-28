"""Recompute summaries from the published 204740 receipt and score JSON files.

This checks reported statistics and identities; original tensors/MP4s are not
present here, so their metric extraction is not rerun.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def geometric(values):
    return math.exp(sum(map(math.log, values))/len(values))


def geometry(error_base, error_new, update):
    """RMSE inputs share the same coordinate count. e=reference-base."""
    eb, ec, dd = error_base**2, error_new**2, update**2
    return {
        "base_error_rmse": error_base,
        "new_error_rmse": error_new,
        "update_rmse": update,
        "cosine_toward_reference": (eb+dd-ec)/(2*math.sqrt(eb*dd)),
        "positive_gain_cosine_threshold": math.sqrt(dd/eb)/2,
        "normalized_gain": (eb-ec)/eb,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    source = args.experiments
    filenames = ["E2E_RECEIPT_204740.json", "E2E_SCORES_204740.json", "E2E_EXECUTION.json"]
    receipt, scores, execution = [json.loads((source/name).read_bytes()) for name in filenames]
    assert receipt["status"] == "complete" and receipt["job"] == "204740"
    assert execution["formal_timing"] is False
    source_hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in filenames}
    code_hashes = {}
    for name in ["attention_paired_value_end_to_end.py", "attention_paired_value_ops.py"]:
        h = hashlib.sha256((source/name).read_bytes()).hexdigest()
        assert h == execution["school_verification"]["inputs"][name]
        code_hashes[name] = h
    rows = {}
    metric_ratios = {}
    video_hashes = []
    for sid, item in receipt["records"].items():
        b, c = [item["candidates"][arm]["record"] for arm in ["diagonal", "paired"]]
        assert b["initial"] == c["initial"]
        assert b["flows"][:3] == c["flows"][:3]
        assert b["latents"][:3] == c["latents"][:3]
        score = scores["sources"][sid]
        assert score["tensor_errors"] == item["errors"]
        assert score["paired_vs_diagonal"] == item["paired_vs_diagonal"]
        assert all(x[field]["relative_l2"] == 0 for x in item["paired_vs_diagonal"][:3] for field in ["flow", "latent"])
        for arm in ["diagonal", "paired"]:
            candidate = item["candidates"][arm]
            assert candidate["attention"]["attention_sites"] == 160
            assert candidate["attention"]["native_probe_queries"] == 5120
            assert candidate["record"]["counters"]["tile_rows"] == 8192
            video_hashes.append(candidate["record"]["video"]["mp4_sha256"])
        video_hashes.extend(x["video"]["mp4_sha256"] for x in item["controls"].values())
        row = {"fold": item["fold"], "order": item["order"], "metrics": {}, "geometry": {}}
        for ref in ["Dense", "wide8192"]:
            for field in ["flow", "latent"]:
                eb, ec = [item["errors"][arm][ref][-1][field] for arm in ["diagonal", "paired"]]
                name = field+"_vs_"+ref
                ratio = ec["relative_l2"]/eb["relative_l2"]
                metric_ratios.setdefault(name, {})[sid] = ratio
                row["metrics"][name] = {"diagonal": eb["relative_l2"], "paired": ec["relative_l2"], "paired_over_diagonal": ratio,
                    "difference_percentage_points": 100*(ec["relative_l2"]-eb["relative_l2"])}
                row["geometry"][name] = geometry(eb["rmse"], ec["rmse"], item["paired_vs_diagonal"][-1][field]["rmse"])
            for field in ["rmse_0_1", "temporal_difference_rmse_0_1"]:
                eb, ec = [score["decoded_pixel_errors"][arm+"_to_"+ref][field] for arm in ["diagonal", "paired"]]
                name = field+"_vs_"+ref
                metric_ratios.setdefault(name, {})[sid] = ec/eb
                row["metrics"][name] = {"diagonal": eb, "paired": ec, "paired_over_diagonal": ec/eb, "difference": ec-eb}
                row["geometry"][name] = geometry(eb, ec, score["decoded_pixel_errors"]["paired_to_diagonal"][field])
        row["paired_vs_diagonal_final"] = item["paired_vs_diagonal"][-1]
        row["paired_vs_diagonal_pixels"] = score["decoded_pixel_errors"]["paired_to_diagonal"]
        row["motion_proxies"] = score["within_video_motion_proxies"]
        row["execution_only_timing"] = {
            "wall_seconds": {"diagonal": b["wall"], "paired": c["wall"]},
            "step_ms": {"diagonal": b["step_ms"], "paired": c["step_ms"]},
            "low_hit_delta_ms": c["step_ms"][3]-b["step_ms"][3],
            "low_hit_ratio": c["step_ms"][3]/b["step_ms"][3],
            "new_packed_sites": {"diagonal": b["counters"]["new_packed_sites"], "paired": c["counters"]["new_packed_sites"]},
            "peak_allocated_GiB": {"diagonal": b["peak_allocated"]/2**30, "paired": c["peak_allocated"]/2**30},
            "peak_allocated_delta_MiB": (c["peak_allocated"]-b["peak_allocated"])/2**20,
        }
        row["reference_jobs"] = {k: v["job"] for k, v in item["controls"].items()}
        rows[sid] = row
    assert len(video_hashes) == len(set(video_hashes)) == 16
    aggregation = {}
    for metric, ratios in metric_ratios.items():
        dev = [v for sid, v in ratios.items() if sid != "vbi_0128"]
        aggregation[metric] = {"improved_sources": sum(v<1 for v in ratios.values()),
            "source_count": len(ratios), "DEV_geometric_ratio": geometric(dev),
            "all_four_descriptive_geometric_ratio": geometric(list(ratios.values())),
            "paired_mechanism_holdout_ratio": ratios["vbi_0128"]}
    out = {"evidence_commit": "b26f644923646dcd9d853849cf58fdd122c55a54", "job": "204740",
        "scope": "Recomputed published scalar statistics and identity checks; no raw tensors/videos downloaded or new GPU execution",
        "source_sha256": source_hashes, "runtime_source_sha256_verified": code_hashes,
        "candidate_videos": 8, "distinct_video_hashes": 16, "identical_prefix_steps": 3,
        "sources": rows, "aggregation": aggregation,
        "missing": ["Accessible original-video URLs or exact media directory", "Raw video/tensor files in this environment", "Formal repeated warm timing", "New-method human visual review", "End-to-end SparsePR v2 arm"],
        "geometry_definition": "e=reference-diagonal; d=paired-diagonal. All polarizations use a common coordinate count and RMS scale. They describe paired Euclidean error, not semantic quality."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"prefix_and_runtime_hash_checks": "passed", "aggregation": aggregation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
