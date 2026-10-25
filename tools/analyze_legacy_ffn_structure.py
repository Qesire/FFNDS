"""Describe hash-bound CAL24 operands without loading a model or assuming Q states.

Rows are repeated measurements within sources. Different scopes used different
row samples, so this tool deliberately makes no cross-step row comparisons.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_ffnds_observations import load_arrays, quantiles, ratio
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def profile(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] % 64 or not np.isfinite(x).all():
        raise ValueError("Need finite full-width group64 operand")
    groups = x.reshape(len(x), -1, 64)
    energy = np.sum(groups * groups, axis=2)
    total = energy.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("Zero row needs a separate profile population")
    p = energy / total[:, None]
    template = p.mean(axis=0)
    reconstruction = total[:, None] * template
    order = np.argsort(total, kind="stable")
    a, b = order[::2], order[1::2]
    a = a[:len(b)]
    cos = np.sum(p[a] * p[b], axis=1) / (np.linalg.norm(p[a], axis=1) * np.linalg.norm(p[b], axis=1))
    rms = np.sqrt(total / x.shape[1])
    # This is a descriptive fit, not held-out prediction or a routing oracle.
    return {
        "rows": len(x), "groups": energy.shape[1],
        "zero_element_fraction": float(np.mean(x == 0)),
        "negative_element_fraction": float(np.mean(x < 0)),
        "all_zero_group64_fraction": float(np.mean(np.all(groups == 0, axis=2))),
        "all_zero_leaf4_fraction": float(np.mean(np.all(x.reshape(len(x), -1, 4) == 0, axis=2))),
        "row_RMS_quantiles": quantiles(rms),
        "row_group_energy_participation_quantiles": quantiles(1 / np.sum(p * p, axis=1)),
        "static_template_effective_groups": float(1 / np.sum(template * template)),
        "row_amplitude_times_static_profile_relative_SSE": ratio(np.sum((energy - reconstruction) ** 2), np.sum(energy ** 2)),
        "RMS_adjacent_pair_count": len(a),
        "RMS_adjacent_pair_relative_RMS_gap_quantiles": quantiles((rms[b] - rms[a]) / rms[a]),
        "RMS_adjacent_pair_profile_cosine_quantiles": quantiles(cos),
        "nonzero_support_is_not_safe_skip": True,
    }


def analyze(collection_paths, indices, blocks):
    selected = set(indices)
    result = {"status": "COMPLETE_HASH_BOUND_SAVED_OPERAND_DESCRIPTIVE_ANALYSIS",
              "source_clusters": len(selected), "source_indices": sorted(selected),
              "scope": "Existing exposed CAL24 dense trajectories; post-QDQ operands, no pre-QDQ or native scale/code claims",
              "row_pairing": "Within-scope RMS-adjacent rows only. Cross-scope rows are NOT aligned.",
              "model_calls": 0, "GPU_calls": 0, "raw_bytes_read": 0, "bindings": [], "scopes": []}
    result["analysis_source_sha256"] = {p.name: sha256(p) for p in [Path(__file__),
        ROOT / "tools/analyze_ffnds_observations.py", ROOT / "tools/ffnds_observation_common.py"]}
    found = set()
    audit_path = ROOT / "reports/worker_20261020/calibration24_full_capture_audit_RUN.json"
    audit = json.loads(audit_path.read_text())
    expected_collections = {str(Path(r["path"]).resolve()): r["sha256"] for r in audit["collections"]}
    result["bindings"].append({"path": str(audit_path.relative_to(ROOT)), "sha256": sha256(audit_path)})
    for path in collection_paths:
        if sha256(path) != expected_collections.get(str(path.resolve())):
            raise ValueError("Collection differs from the original CAL24 audit")
        collection = json.loads(path.read_text())
        if collection["status"] != "PASS_CALIBRATION24_SHARD_FULL160_OPERANDS_AND_NOOP":
            raise ValueError("Unaccepted CAL24 collection")
        result["bindings"].append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        for entry in collection["completed"]:
            if entry["index"] not in selected:
                continue
            report_path = Path(entry["input_report"])
            if sha256(report_path) != entry["input_report_sha256"]:
                raise ValueError("Original input receipt changed")
            report = json.loads(report_path.read_text())
            if report["status"] != "PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP" or report["index"] in found:
                raise ValueError("Unaccepted or duplicate CAL24 source")
            if report["trajectories"]["plain"] != report["trajectories"]["observed"]:
                raise ValueError("Original no-op failed")
            found.add(report["index"])
            for scope in report["scopes"]:
                if scope["block"] not in blocks:
                    continue
                payload = Path(scope["path"])
                if payload.stat().st_size != scope["bytes"] or sha256(payload) != scope["sha256"]:
                    raise ValueError("Original operand changed")
                arrays = load_arrays(payload)
                if set(arrays) != {"V1", "V2", "Y", "b2", "gate"} or arrays["V2"].shape != (256, 13824):
                    raise ValueError("CAL24 operand schema changed")
                item = {"source_index": report["index"], "step": scope["step"],
                        "block": scope["block"], "expert": scope["expert"],
                        "payload_sha256": scope["sha256"], "bytes": scope["bytes"],
                        "V1": profile(arrays["V1"]), "V2": profile(arrays["V2"])}
                y = arrays["Y"].astype(np.float64) * arrays["gate"].astype(np.float64)
                item["gated_output_RMS_quantiles"] = quantiles(np.sqrt(np.mean(y * y, axis=1)))
                result["raw_bytes_read"] += scope["bytes"]
                result["scopes"].append(item)
                print(json.dumps({"source_index": report["index"], "step": scope["step"], "block": scope["block"]}), flush=True)
    if found != selected or len(result["scopes"]) != len(selected) * 4 * len(blocks):
        raise ValueError("Incomplete selected source/scope coverage")
    result["source_summary"] = []
    for index in sorted(selected):
        scopes = [s for s in result["scopes"] if s["source_index"] == index]
        result["source_summary"].append({"source_index": index,
            "V2_element_zero_fraction_range": [min(s["V2"]["zero_element_fraction"] for s in scopes), max(s["V2"]["zero_element_fraction"] for s in scopes)],
            "V2_zero_group64_fraction_range": [min(s["V2"]["all_zero_group64_fraction"] for s in scopes), max(s["V2"]["all_zero_group64_fraction"] for s in scopes)],
            "V2_profile_model_relative_SSE_quantiles": quantiles([s["V2"]["row_amplitude_times_static_profile_relative_SSE"] for s in scopes]),
            "V2_matched_RMS_profile_cosine_median_range": [min(s["V2"]["RMS_adjacent_pair_profile_cosine_quantiles"][2] for s in scopes), max(s["V2"]["RMS_adjacent_pair_profile_cosine_quantiles"][2] for s in scopes)]})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indices", nargs="+", type=int, default=[0, 4, 8, 12, 16, 20])
    ap.add_argument("--blocks", nargs="+", type=int, default=[0, 10, 20, 30, 39])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve original descriptive result")
    paths = [ROOT / f"runs/worker_restart_20261020/calibration24_ffn_operands/shard{s}/RUN.json" for s in [0, 1]]
    write_json(args.out, analyze(paths, args.indices, args.blocks))


if __name__ == "__main__":
    main()
