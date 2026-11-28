"""CPU motion adapter around the unchanged FFNDS video diagnostic functions.

This exports motion/paired observations only. It does not claim the older full
quality-report schema, whose CLI also requires independently measured FFN cost.
No cost value is imputed and no motion/flow equation is reimplemented here.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

from build_metric_binding import file_hash, load_json, require


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def finite(value):
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def run(args):
    binding_path, output = args.binding.resolve(), args.out.resolve()
    binding_sha = file_hash(binding_path)
    binding = load_json(binding_path)
    require(binding["schema"] == "FFNDS_CENTERED_MEDIUM_METRIC_BINDING_V1", "Unexpected binding")
    require(file_hash(binding["generation_run"]) == binding["generation_run_sha256"], "Generation run changed")
    require(file_hash(binding["cohort"]) == binding["cohort_sha256"], "Source manifest changed")
    implementation = args.implementation.resolve()
    require(file_hash(implementation) == args.implementation_sha256, "Motion implementation changed")
    spec = importlib.util.spec_from_file_location("ffnds_bound_video_diagnostics", implementation)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    method = load_json(binding["method_binding"])
    require(file_hash(binding["method_binding"]) == binding["method_binding_sha256"], "Method binding changed")
    geometry = binding["geometry"]
    records = []
    for source in binding["records"]:
        sid = source["source_id"]
        require(Path(sid).name == sid, "Unexpected source filename")
        dense = source["videos"]["dense"]
        for arm in binding["arms"]:
            video = source["videos"][arm]
            for item in (video, dense):
                require(file_hash(item["path"]) == item["sha256"], "Video bytes changed")
                require(file_hash(item["full_method_receipt"]) == item["full_method_receipt_sha256"], "Receipt changed")
            path = output / arm / (sid + ".json")
            if path.exists():
                result = load_json(path)
                require(result["metric_binding_sha256"] == binding_sha and
                        result["implementation_sha256"] == args.implementation_sha256 and
                        result["adapter_sha256"] == file_hash(__file__), "Prior motion result uses another binding/implementation")
            else:
                def identity(item):
                    return dict(source_id=sid, seed=source["seed"], expected_frames=geometry[2],
                                input_prompt_digest=source["prompt_sha256"],
                                model_digest=method["original_release_manifest_sha256"],
                                protocol_sha256=binding_sha, run_sha256=item["full_method_receipt_sha256"])
                metadata = dict(identity=identity(video), cost={})
                if arm != "dense":
                    metadata["reference_identity"] = identity(dense)
                raw = module.evaluate(Path(video["path"]), None if arm == "dense" else Path(dense["path"]), metadata)
                artifact = raw["metrics"]["artifact"]
                require(artifact["decoded_frames"] == artifact["verified_full_decode_frames"] == geometry[2], "Incomplete decoded frames")
                require(artifact["frame_shape"] == [geometry[1], geometry[0], 3] and
                        abs(artifact["fps"] - geometry[3]) < 1e-3, "Motion decode geometry differs")
                require(len(raw["per_frame"]) == geometry[2], "Missing per-frame observations")
                selected = {key: value for key, value in raw["metrics"].items() if key != "runtime"}
                result = dict(
                    schema="FFNDS_MEDIUM_MOTION_DIAGNOSTICS_V1", status="CPU_DIAGNOSTICS_COMPLETE",
                    source_id=sid, source_cluster_id=source["source_cluster_id"], arm=arm,
                    reference_arm=None if arm == "dense" else "dense", metric_binding_sha256=binding_sha,
                    generation_run_sha256=binding["generation_run_sha256"],
                    implementation_path=str(implementation), implementation_sha256=args.implementation_sha256,
                    adapter_sha256=file_hash(__file__), implementation=raw["implementation"],
                    identity=metadata["identity"], reference_identity=metadata.get("reference_identity"),
                    candidate=raw["candidate"], reference=raw["reference"], metrics=selected,
                    per_frame=raw["per_frame"], windows16=raw["windows16"],
                    cost_reported=False, official_quality_scored=False, human_acceptance=False,
                    scope="Unchanged full-frame video diagnostics; timing and quality conclusions are separate",
                )
                require(finite(result), "Nonfinite motion observation")
                save(path, result)
            records.append(dict(source_id=sid, source_cluster_id=source["source_cluster_id"], arm=arm,
                                report_path=str(path), report_sha256=file_hash(path), metrics=result["metrics"]))
    summary = dict(schema="FFNDS_MEDIUM_MOTION_INDEX_V1", metric_binding_sha256=binding_sha,
                   generation_run_sha256=binding["generation_run_sha256"],
                   implementation_sha256=args.implementation_sha256, adapter_sha256=file_hash(__file__),
                   records=records, source_count=binding["source_count"], arms=binding["arms"])
    save(output / "MOTION_INDEX.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--implementation-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({"status": "CPU_MOTION_DIAGNOSTICS_COMPLETE", "records": len(result["records"])}))


if __name__ == "__main__":
    main()
