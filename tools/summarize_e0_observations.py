"""Publish compact, source-bound descriptions only after every native no-op passes."""
import argparse
import json
from pathlib import Path
import statistics

from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
INDICES = [0, 4, 8, 12, 16, 20]


def summarize(raw_root):
    private = ROOT / "runs/observations_20261025"
    inputs_path = ROOT / ".runtime/inputs/calibration24/canonical.json"
    inputs = json.loads(inputs_path.read_text())
    old_audit_path = ROOT / "reports/worker_20261020/calibration24_full_capture_audit_RUN.json"
    old_audit = json.loads(old_audit_path.read_text())
    old_videos = {r["index"]: r["sha256"] for r in old_audit["videos"] if r["arm"] == "plain"}
    first, replication = [json.loads((ROOT / "config" / name).read_text()) for name in [
        "e0_observation_native_20261025.json", "e0_observation_replication_20261025.json"]]
    for key in ["steps", "blocks", "uniform_rows", "weight_panel_groups", "native_source_sha256", "input_default"]:
        if first[key] != replication[key]:
            raise ValueError("A1 and replication capture settings differ")
    result = {"id": "E0_NATIVE_SIX_SOURCE_DISCOVERY_20261025",
              "status": "COMPLETE_ACCEPTED_PASSIVE_NATIVE_OBSERVATIONS",
              "independent_source_clusters": 6, "source_indices": INDICES,
              "source_strata": {"subject": 3, "background": 3, "camera": 0},
              "strata_correction": "The frozen replication question incorrectly described two sources per stratum; fixed actual CAL24 indices are three subject/three background and no camera sources. No sources were substituted.",
              "capture_settings_compatibility": "A1 source0 and replication use identical steps/blocks/rows/panels/native-source settings; full protocol SHA differs and both are retained.",
              "quality_or_speed_acceptance": False, "training_updates": 0,
              "model_or_operator_changed": False, "sources": [], "scopes": [],
              "original_CAL24_video_audit_sha256": sha256(old_audit_path),
              "analysis_source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
                  Path(__file__), ROOT / "tools/analyze_e0_native_mechanism.py",
                  ROOT / "tools/check_e0_quantizer_reference.py"]}}
    scopes, bytes_total = [], 0
    for index in INDICES:
        run_path = raw_root / f"source{index:02d}_attempt1/RUN.json"
        run = json.loads(run_path.read_text())
        protocol = ROOT / "config" / ("e0_observation_native_20261025.json" if index == 0 else "e0_observation_replication_20261025.json")
        if (run["status"] != "PASS_PASSIVE_CAPTURE_NOOP" or run["source_index"] != index
                or run["source_id"] != inputs[index]["input_provenance"]["image_sha256"]
                or run["inputs_sha256"] != sha256(inputs_path)
                or run["protocol_sha256"] != sha256(protocol)
                or run["generations_started"] != 2 or len(run["scopes"]) != 20
                or not run["assigned_GPU_resources"]["sampled_isolation_pass"]):
            raise ValueError("Unaccepted, mismatched or resource-contaminated source")
        a, b = run["phases"]["plain"], run["phases"]["observed"]
        if a["trajectory"] != b["trajectory"] or a["video_sha256"] != b["video_sha256"]:
            raise ValueError("Native no-op identity failed")
        for phase in (a, b):
            c = phase["native_counts"]
            if c != {"linear_forward_calls": 1600, "activation_qdq_calls": 1600,
                     "weight_transfers": 1600, "cache_misses": 0}:
                raise ValueError("Original dense call counts changed")
        mp = private / f"SOURCE{index:02d}_MECHANISM_V2_PRIVATE.json"
        qp = private / f"SOURCE{index:02d}_QUANT_REFERENCE_V2_PRIVATE.json"
        mechanism, quant = json.loads(mp.read_text()), json.loads(qp.read_text())
        if mechanism["run_sha256"] != sha256(run_path) or quant["run_sha256"] != sha256(run_path):
            raise ValueError("Analysis belongs to another native run")
        for name, expected in mechanism["analysis_source_sha256"].items():
            if sha256(ROOT / "tools" / name) != expected:
                raise ValueError("Mechanism analysis source drift")
        for name, expected in quant["source_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError("Quantizer comparison source drift")
        snapshot = run_path.parent / "EXECUTED_SOURCES"
        snapshot.mkdir(exist_ok=True)
        for name, expected in run["collector_sha256"].items():
            saved = snapshot / name
            current = ROOT / "tools" / name
            if saved.exists():
                if sha256(saved) != expected:
                    raise ValueError("Executed collector snapshot changed")
            elif sha256(current) == expected:
                saved.write_bytes(current.read_bytes())
            else:
                raise ValueError("Exact executed collector source is missing")
        atlas = {(s["step"], s["block"]): s for s in mechanism["atlas"]["scopes"]}
        selected = []
        for s in mechanism["scopes"]:
            item = {k: v for k, v in s.items() if k != "profiles"}
            item["source_index"] = index
            item["profiles"] = {name: {key: p[key] for key in [
                "zero_element_fraction", "all_zero_group64_fraction",
                "row_amplitude_times_static_profile_relative_SSE",
                "RMS_adjacent_pair_profile_cosine_quantiles"]} for name, p in s["profiles"].items()}
            obs = atlas[s["step"], s["block"]]
            item["Q1_relative_quantization_SSE"] = obs["quantization_pairs"]["q1"]["relative_SSE"]
            item["Q2_relative_quantization_SSE"] = obs["quantization_pairs"]["q2"]["relative_SSE"]
            item["pre_Q2_group64_amax_min"] = obs["fields"]["q2_pre"]["group64_amax_quantiles"][0]
            selected.append(item)
        size = sum(p.stat().st_size for p in run_path.parent.rglob("*") if p.is_file())
        bytes_total += size
        result["sources"].append({"index": index, "source_group": inputs[index]["source_group"],
            "source_id": run["source_id"], "seed": run["seed"],
            "HF_relative": "ffnds-derived/e0_observations_20261025/" + run_path.parent.name,
            "run_sha256": sha256(run_path), "protocol": str(protocol.relative_to(ROOT)),
            "protocol_sha256": run["protocol_sha256"], "collector_sha256": run["collector_sha256"],
            "analysis_sha256": sha256(mp), "quantizer_reference_sha256": sha256(qp),
            "trajectory": a["trajectory"], "video_sha256": a["video_sha256"],
            "matches_original_CAL24_dense_video_SHA": a["video_sha256"] == old_videos[index],
            "native_counts_each_phase": a["native_counts"], "full_generations": 2,
            "payloads": [{k: s[k] for k in ["step", "block", "path", "sha256"]} for s in run["scopes"]],
            "new_output_bytes": size, "isolation_pass": True,
            "Q2_profile_SSE_increment_mean": statistics.mean(
                s["profiles"]["q2_post"]["row_amplitude_times_static_profile_relative_SSE"] -
                s["profiles"]["q2_pre"]["row_amplitude_times_static_profile_relative_SSE"] for s in selected),
            "reference_cases": len(quant["cases"]),
            "reference_signed_zero_bit_mismatches": sum(c["BF16_bit_mismatches"] for c in quant["cases"]),
            "reference_nonzero_value_mismatches": sum(c["nonzero_value_mismatches"] for c in quant["cases"]),
            "reference_bits_equal_after_input_sign_preservation": quant["all_bits_equal_after_preserving_input_sign"]})
        scopes.extend(selected)
    result.update(scopes=scopes, captured_scopes=len(scopes), full_generations=12,
                  peak_simultaneous_GPU_cards=2, new_output_bytes=bytes_total)
    if len({s["source_id"] for s in result["sources"]}) != 6 or len({s["source_group"] for s in result["sources"]}) != 6:
        raise ValueError("Six distinct source clusters were not established")
    result["interpretation"] = "Existing pre-Q2 variation versus Q2 increments is a local descriptive mechanism comparison. Coarse energy profiles can conceal important directional/threshold effects; no full-video speed or method claim."
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Preserve first six-source summary")
    write_json(args.out, summarize(args.raw_root))


if __name__ == "__main__":
    main()
