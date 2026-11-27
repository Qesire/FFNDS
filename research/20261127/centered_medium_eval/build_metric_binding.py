"""Bind FrozenCenteredRuntime outputs to the existing seven native metrics.

This is a CPU-only provenance adapter. It imports the archived eval_anchor10.py
helpers, checks source/image/receipt/video identities, and emits the binding
accepted by its `vbench` subcommand. It does not implement metric equations,
generate videos, choose sources, select the best timing repetition, or run a
model. It uses only the quality entries from run_medium_eval.py's run receipt.
"""

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path


EXPECTED_DIMS = (
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "motion_smoothness", "imaging_quality", "i2v_subject", "i2v_background",
)


def file_hash(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_evaluator(path, expected_sha256):
    path = Path(path).resolve()
    require(file_hash(path) == expected_sha256, "Evaluator source SHA256 differs")
    spec = importlib.util.spec_from_file_location("ffnds_frozen_eval_anchor10", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.DIMS) == EXPECTED_DIMS, "Expected the archived seven dimensions")
    return module


def build(args):
    evaluator_path = args.evaluator.resolve()
    evaluator = load_evaluator(evaluator_path, args.evaluator_sha256)
    cohort_path, scope_path, method_path, run_path = (
        path.resolve() for path in (args.cohort, args.metric_scope, args.method_binding, args.run)
    )
    cohort, scope, method = map(load_json, (cohort_path, scope_path, method_path))
    run = load_json(run_path)
    cohort_sha, method_sha = file_hash(cohort_path), file_hash(method_path)
    require(run["schema"] == "FFNDS_CENTERED_MEDIUM_RUN_V1" and run["status"] == "complete",
            "Expected the completed medium evaluation run")
    require(run["cohort_sha256"] == cohort_sha and run["method_binding_sha256"] == method_sha,
            "Run cohort or method identity differs")
    require(run["seed"] == 42, "Run seed differs")
    require(len(run["identity_checks"]) == 2 and all(r["matches_201500"] for r in run["identity_checks"]),
            "The production adapter has not matched the archived candidate")
    require(file_hash(scope_path) == args.metric_scope_sha256, "Current metric scope SHA256 differs")
    require(cohort.get("evaluation_mode", "CUSTOM_PAIRED") == "CUSTOM_PAIRED", "Only CUSTOM_PAIRED is bound here")
    require(cohort.get("synthetic", False) is False, "Synthetic cohorts cannot enter this binding")
    for key in (
        "metric_project_root", "quality_reference_dir", "quality_reference_sha256",
        "metric_package_versions", "metric_cache_root", "metric_site_packages",
        "metric_asset_sha256", "metric_code_sha256", "metric_seed",
    ):
        require(key in scope, "Current metric scope missing: " + key)
    geometry = [method["geometry"][key] for key in ("width", "height", "frames", "fps")]
    require(geometry == [832, 464, 81, 16] and method["seed"] == 42, "Frozen four-step geometry/seed differs")
    require(cohort.get("seed", 42) == 42, "The frozen runtime uses seed 42")
    arms = run["arms"]
    require(arms == ["dense", "wide8192", "centered"], "Expected the frozen three-arm run")
    output_root = Path(run["output_root"]).resolve()
    runner_sha = file_hash(args.runner)
    rows = cohort["sources"]
    ids = [row["source_id"] for row in rows]
    require(bool(rows) and len(set(ids)) == len(ids), "Expected unique source IDs")
    require(all("source_cluster_id" in row for row in rows), "Source clusters must be explicit")
    require(ids == [row["source_id"] for row in run["quality"]], "Quality rows do not match the frozen source order")
    quality = {row["source_id"]: row for row in run["quality"]}
    image_root = Path(scope["metric_project_root"]) / "vbench2_beta_i2v/data/crop/16-9"
    records, ledger = [], []
    seen_paths = set()
    for source in rows:
        sid = source["source_id"]
        require(isinstance(sid, str) and Path(sid).name == sid, "source_id must be a single filename component")
        image = Path(source["image_path"]).resolve()
        image_name = source.get("image_name", image.name)
        require(Path(image_name).name == image_name, "image_name must be a single filename")
        evaluator.verify(image, source["image_sha256"])
        evaluator.verify(image_root / image_name, source["image_sha256"])
        require(sha256(source["prompt"].encode()).hexdigest() == source["prompt_sha256"], "Prompt SHA256 differs: " + sid)
        # Native custom-input scores are valid on every bound image/video pair.
        # Preserve original benchmark dimension tags for a separate tagged table.
        original_tags = source.get("original_dimension_tags", source.get("dimensions", []))
        require(isinstance(original_tags, list), "Original dimension tags must be a list")
        row = dict(source=sid, source_id=sid, source_cluster_id=source["source_cluster_id"],
                   seed=42, sample_index=0, image_name=image_name,
                   image_sha256=source["image_sha256"], prompt=source["prompt"],
                   prompt_sha256=source["prompt_sha256"], category=source.get("category", "unspecified"),
                   dimensions=list(EXPECTED_DIMS), original_dimension_tags=original_tags,
                   exposure_history=source.get("exposure_history"), videos={})
        first_anchor = None
        require(quality[sid]["source_cluster_id"] == source["source_cluster_id"], "Run/source cluster differs")
        require(set(quality[sid]["arms"]) == set(arms), "Incomplete source quality arms")
        for arm in arms:
            run_wrapped = quality[sid]["arms"][arm]
            group = Path(run_wrapped["record"]["group"])
            require(not group.is_absolute() and ".." not in group.parts, "Invalid output group")
            directory = output_root / group
            wrapped_path = directory / (sid + ".full_method.json")
            receipt_path = directory / (sid + ".json")
            video_path = directory / (sid + ".mp4")
            wrapped, receipt = load_json(wrapped_path), load_json(receipt_path)
            require(wrapped == run_wrapped, "Run and full-method sidecar disagree")
            require(wrapped["arm"] == arm and wrapped["source_id"] == sid, "Wrapped source/arm differs")
            require(wrapped["source_manifest_sha256"] == cohort_sha, "Generation used another source manifest")
            require(wrapped["binding_sha256"] == method_sha, "Generation used another method binding")
            require(wrapped["runner_sha256"] == runner_sha, "Generation wrapper source differs")
            require(wrapped["runtime_generate_sha256"] == run["runtime_generate_sha256"], "Runtime generate.py differs")
            require(wrapped.get("diagnostic_replay") is False and wrapped.get("trajectory_saved") is False,
                    "Expected production renders without diagnostic replay/trajectory copies")
            if arm == "centered":
                require(wrapped["candidate_id"] == method["candidate_id"], "Centered method identity differs")
            record = wrapped["record"]
            for key in ("source", "initial", "flows", "latents", "experts", "video"):
                require(record[key] == receipt[key], "Wrapped and original receipt disagree: " + key)
            require(record["source"] == sid, "Original source differs")
            anchor = (record["initial"], record["flows"][0], record["latents"][0])
            require(first_anchor is None or first_anchor == anchor, "Paired initial noise/first native anchor differs")
            first_anchor = anchor
            require(wrapped.get("final_latent_saved") is True, "Quality final latent is missing")
            latent = wrapped["final_latent"]
            require(latent["capture_step"] == 3 and latent["tensor_name"] == "latent", "Unexpected latent capture")
            evaluator.verify(latent["path"], latent["sha256"])
            require(Path(latent["path"]).stat().st_size == latent["bytes"], "Final latent byte count differs")
            require(str(video_path) not in seen_paths, "Generated video path reused across samples")
            seen_paths.add(str(video_path))
            inputs = dict(source=sid, initial=record["initial"],
                          first_flow=record["flows"][0], first_latent=record["latents"][0])
            video = evaluator.checked_video(
                inputs, video_path, receipt_path, file_hash(receipt_path),
                record["video"]["mp4_sha256"], record["video"]["decoded_rgb24_sha256"], geometry,
            )
            if args.verify_decoded_rgb:
                evaluator.decode(video, geometry)
            video.update(full_method_receipt=str(wrapped_path),
                         full_method_receipt_sha256=file_hash(wrapped_path),
                         final_latent=latent,
                         review_url=wrapped.get("review_url"))
            row["videos"][arm] = video
            ledger.append(dict(source_id=sid, source_cluster_id=row["source_cluster_id"], arm=arm,
                               video_path=str(video_path), video_sha256=video["sha256"],
                               decoded_rgb24_sha256=video["rgb_sha256"],
                               review_url=video["review_url"],
                               review_link_status="PRESENT_UNCHECKED" if video["review_url"] else "PENDING"))
        records.append(row)
    binding = dict(
        schema="FFNDS_CENTERED_MEDIUM_METRIC_BINDING_V1", evaluation_mode="CUSTOM_PAIRED",
        scoring_population="custom_all_bound_sources", cohort=str(cohort_path), cohort_sha256=cohort_sha,
        method_binding=str(method_path), method_binding_sha256=method_sha,
        metric_scope=str(scope_path), metric_scope_sha256=args.metric_scope_sha256,
        evaluator_sha256=args.evaluator_sha256, evaluator_path=str(evaluator_path),
        builder_sha256=file_hash(__file__), runner_sha256=runner_sha,
        runtime_generate_sha256=run["runtime_generate_sha256"], arms=arms,
        generation_run=str(run_path), generation_run_sha256=file_hash(run_path),
        sources=ids, source_count=len(rows), source_cluster_count=len({r["source_cluster_id"] for r in rows}),
        seed=42, real_samples_per_source_per_arm=1, geometry=geometry,
        decoded_rgb_verified=bool(args.verify_decoded_rgb),
        original_tag_counts={dim: sum(dim in r["original_dimension_tags"] for r in records) for dim in EXPECTED_DIMS},
        records=records,
    )
    evaluator.save(args.binding, binding)
    evaluator.save(args.binding.with_suffix(".video_index.json"), dict(
        schema="FFNDS_REVIEW_VIDEO_INDEX_V1", metric_binding_sha256=file_hash(args.binding),
        status="LINKS_PENDING" if any(r["review_url"] is None for r in ledger) else "LINKS_PRESENT_UNCHECKED",
        records=ledger,
    ))
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "cohort", "metric-scope", "method-binding", "evaluator", "runner", "binding"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("metric-scope-sha256", "evaluator-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--verify-decoded-rgb", action="store_true",
                        help="CPU-decode every video and verify RGB hash, exact frame count and geometry")
    args = parser.parse_args()
    binding = build(args)
    print(json.dumps({"status": "BOUND_NO_METRIC_EXECUTION", "sources": binding["source_count"],
                      "clusters": binding["source_cluster_count"], "arms": binding["arms"],
                      "binding_sha256": file_hash(args.binding)}))


if __name__ == "__main__":
    main()
