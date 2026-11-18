"""Score and report the two independently matched full80 DEV3 video comparisons."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

try:
    from quality_paired import BASE, DEV3, DIMS, FFPROBE_SHA256, FLOOR, FLOW, read, sha, source_arm, write
    import quality_paired
except ModuleNotFoundError as exc:
    if exc.name != "quality_paired":
        raise
    from ffnds_state_operator import quality_paired
    from ffnds_state_operator.quality_paired import BASE, DEV3, DIMS, FFPROBE_SHA256, FLOOR, FLOW, read, sha, source_arm, write

PAIRS = {
    "readout-online-80": "readout-static-80",
    "modes-online-80": "modes-static-80",
}
NEW = tuple(arm for online, static in PAIRS.items() for arm in (static, online))
ARMS = BASE + NEW
COMPARISON = "OBSERVED_SECANT_FULL80_DEV3"
RECIPE_SHA256 = "ee780352c8cf1c1e4efc8aea7c354b3be225c525141942cbd9c15d24cabae5b6"
AMENDMENT = "58f68ab460d47f0f803fd3f2f1b6ec03b651ec21"
METRICS = (*DIMS, "raft_continuous_px_per_2_frames", *FLOW, "raft_amplitude_ratio_to_dense", "camera_motion")


# Scope and protocol identities from the pinned strict Stage 4 completions.
HISTORICAL = {
    'i2v_subject': ('684b38ffbcd6d15e4c63f4bf2edd0d07ba805b54c4c9ca0464fafb3331253997',
        '2d6bb45536a29b862a11e11c2b06e98db4937234f36382ec2c86f98ac3b87884'),
    'i2v_background': ('bfac8a90ad9fbe219eaae217be9154e7b5abbd0b105398ef70815cbe56ac7f52',
        '261fb7371c1af00b7da01577ccd095358a2834956dba91802c680f4f76e19212'),
    'imaging_quality': ('6491772e74e7f24909ff24230f98469612e06b6e6ab2bdbdf8941430ef34bbe6',
        '79f25ab6a703e07a8f01086a8e6dda1d73417c241065da62111cf3da4a97eda1'),
    'subject_consistency': ('86a66d2ce50c0da9f1e582ae8d38e1cbb0d978a7dfa6efe3ca5231d4075e1b6f',
        'c779104a012cf2acf147ffa02c16225a168d7dce9494d48b3d5ca482cfa81222'),
    'background_consistency': ('210470b6a790c4778349691814d0d1b7ae1d3cdc5561f9a8b5e6bb568f1a77ca',
        'f3f245d76ec7a513015df3c91d5fe13702bb774e908195028bdbbd15e888987a'),
    'motion_smoothness': ('7ebc748b47917e032571434e22c994454875d2d93ad2be72f16e6ff6768ace6b',
        '53ffa4de64b860897dbef4f8949812815573bea8e7b826e70f99d7ce8116f3fa'),
    'temporal_flickering': ('08631ae0fdb55db8f18c19a9b0c76d1a5595b6abf8673ddc3abd027ed3c1c5c8',
        '1f4169a23aceac9cb252a8fd06d47d49111b403796a461122feefca3f2143c82'),
    'aesthetic_quality': ('7dc9771f7b5bebea600be8c26bfe20a5b78f94d5347a766b2db0782f7026a238',
        '858133bdef8637916bf2a98d706518d14e5f2b82df33c7a920ffcefb143b064c'),
    'dynamic_degree': ('49f2ca2c1499c00d26a18dce2ce234b61d54ff551c60100ec304ec168fa183ff',
        'f419e302109ea33ddf1d8c18a273343d5771fec0273d6a98783b4e33cb08eb2f'),
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@contextmanager
def inherited_cohort():
    """Bind the exact new slots while using the previously run RAFT and VBench adapters."""
    initial = quality_paired.NEW, quality_paired.ARMS, quality_paired.COMPARISON
    quality_paired.NEW, quality_paired.ARMS, quality_paired.COMPARISON = NEW, ARMS, COMPARISON
    try:
        yield
    finally:
        quality_paired.NEW, quality_paired.ARMS, quality_paired.COMPARISON = initial


def stage_for(stage_map, source, arm):
    by_arm = stage_map[arm]
    return Path(by_arm[source] if isinstance(by_arm, dict) else by_arm)


def validate_matched_scopes(stage_map):
    """Bind each matched pair and verify shared bases across the two families."""
    fields = ("bindings", "parameters", "strict_effective_config",
              "v2_reference_dir", "reference_sha256", "dense_initial_noise", "schedule",
              "fixed64_token_ids", "native_transformer_sha256")
    stages = {stage_for(stage_map, source, arm) for source in DEV3 for arm in NEW}
    shared = None
    releases = {}
    for stage in sorted(stages):
        scope = read(stage / "scope.json")
        binding = {key: scope[key] for key in fields}
        specification = scope["experiment_spec"]
        require(sha(specification["path"]) == specification["sha256"], f"Amended plan specification differs: {stage}")
        binding["experiment_spec_sha256"] = specification["sha256"]
        if shared is None:
            shared = binding
        require(scope["base_commit"] == AMENDMENT and binding == shared,
                f"Full80 arms must share the frozen base and execution recipe: {stage}")
        name = scope["arm"].split("-", 1)[0]
        release = scope["full80_parameters"]
        require(releases.setdefault(name, release) == release, f"Matched arms use different CAL release: {stage}")
    require(set(releases) == {"readout", "modes"}, "Both full80 families required")
    if releases["readout"] == releases["modes"]:
        return

    manifests = {}
    for name, ref in releases.items():
        require(sha(ref["path"]) == ref["sha256"], f"{name}: CAL release hash differs")
        manifest = manifests[name] = read(ref["path"])
        require(manifest["status"] == "PASS" and len(manifest["sites"]) == 80 and manifest["local_metric_video_gate"] is False
                and manifest["bindings"] == shared["bindings"] and manifest["strict_q1"] == shared["parameters"],
                f"{name}: restored CAL lineage differs")
    before, after = (manifests[name] for name in ("readout", "modes"))
    CAL_FIELDS = ("spec_sha256", "q1_cal_pairs", "sources", "producer")
    require(all(before[key] == after[key] for key in CAL_FIELDS) and before["sites"].keys() == after["sites"].keys(),
            "CAL releases do not share the same source, position, and shard identity")
    for site, original in before["sites"].items():
        repaired = after["sites"][site]
        if original != repaired:
            require(sha(original["path"]) == original["sha256"] and sha(repaired["path"]) == repaired["sha256"],
                    f"{site}: parameter values differ from release identity")
            first, second = read(original["path"]), read(repaired["path"])
            require(first["site"] == second["site"] == site and first["identity"] == second["identity"]
                    and first["observed_readout"] == second["observed_readout"],
                    f"{site}: numerical mode recovery changed base or readout")


def generation_proof(stage, baseline, arm):
    """Admit only videos with matching native and outer execution identities."""
    stage = Path(stage).resolve()
    src = baseline["source"]
    output = f"output/{src}"
    names = (
        "scope.json", "ACTIVE_RESEARCH.json", "REVIEW.json", "RUN.json", "EXIT.json", "ATTEMPT_EXIT.json",
        "receipt/RUN.json", "output/BINDINGS.json", "output/SUMMARY.json", "output/RECIPE_PARITY.json",
        f"{output}.json", f"{output}.mp4",
    )
    files = {name: sha(stage / name) for name in names}
    scope, active_doc, review, run, exit_, attempt, native, binding, summary, recipe, produced = (
        read(stage / name) for name in names[:-1]
    )
    scope_sha = files["scope.json"]
    job = str(run["slurm_job_id"])
    # The native guard indexes the record while an outer scope may carry the same fields at top level.
    active = active_doc.get(scope["scope_id"].lower(), active_doc)
    sources = tuple(scope["sources"])
    require(sources in (DEV3, (src,)) and src in sources
            and scope["method"] == "WAN_FREE_GENERATION" and scope["arm"] == arm and scope["force_dense"] is False
            and (scope["seed"], scope["max_d4"], scope["max_nfe"], scope["retries"]) ==
            (42, len(sources), 4 * len(sources), 0), f"{stage}: wrong full80 scope")
    require(review["recheck_disposition"] == "APPROVE" and review["reviewer_agent_id"]
            and review["reviewer_agent_id"] != review["author_agent_id"]
            and active["runnable"] is True and active["review_sha256"] == files["REVIEW.json"]
            and run["review_sha256"] == files["REVIEW.json"]
            and all(r["scope_sha256"] == scope_sha for r in (review, active, run, attempt, native, binding, recipe))
            and (run["scope_id"], exit_["scope_id"]) == (scope["scope_id"], scope["scope_id"]),
            f"{stage}: scope, review, and execution receipts differ")
    require(exit_["exit_code"] == attempt["exit_code"] == 0
            and all(str(r["slurm_job_id"]) == job for r in (exit_, attempt, binding, recipe))
            and str(native["allocation"]["job_id"]) == job, f"{stage}: job did not exit successfully")
    archived = exit_["output_sha256"]
    for name in ("receipt/RUN.json", "output/BINDINGS.json", "output/SUMMARY.json", "output/RECIPE_PARITY.json",
                 f"{output}.json", f"{output}.mp4"):
        require(archived.get(name) == files[name], f"{stage}: completed output is unbound: {name}")
    require(binding["schema"] == "FFNDS_OBSERVED_FULL80_BINDINGS_V1"
            and (binding["arm"], binding["sources"]) == (arm, list(sources))
            and binding["strict_q1"] == scope["parameters"] and binding["p0"] == scope["bindings"]
            and binding["parameters"] == scope["full80_parameters"] and binding["producer_sha256"] == scope["bundle_sha256"]
            and recipe["strict"] == scope["strict_effective_config"]
            and recipe["exact_bytes_equal"] is True and recipe["nfe_at_check"] == 0
            and recipe["live"]["sha256"] == recipe["strict"]["sha256"]
            and binding["recipe_parity"]["sha256"] == files["output/RECIPE_PARITY.json"],
            f"{stage}: effective recipe or full80 parameter proof differs")

    ref_dir = Path(scope["v2_reference_dir"])
    for name in ("scope.json", "v2_conditions.json"):
        require(sha(ref_dir / name) == scope["reference_sha256"][name], f"{stage}: capture identity changed: {name}")
    prior = read(ref_dir / "scope.json")
    conditions = [row for row in read(ref_dir / "v2_conditions.json") if row["record_id"] == src]
    cond, = conditions
    identity = baseline["input_identity"]
    require(all(cond[key] == baseline[key] for key in ("image_sha256", "prompt_sha256", "source_cluster_id", "seed"))
            and prior["recipe_binding"]["recipe_sha256"] == RECIPE_SHA256
            and {k: v["sha256"] for k, v in binding["checkpoints"].items()} == identity["checkpoint_sha256"]
            and produced["initial_noise"] == identity["initial_noise"] and produced["schedule"] == identity["scheduler"],
            f"{stage}: source, recipe, weights, scheduler, or noise differs from Dense")
    require((summary["status"], summary["arm"], summary["nfe"]) == ("COMPLETE", arm, 4 * len(sources))
            and [r["source"] for r in summary["records"]] == list(sources)
            and next(r["receipt_sha256"] for r in summary["records"] if r["source"] == src) == files[f"{output}.json"],
            f"{stage}: summary does not bind completed source receipt")
    video = produced["video"]
    require((produced["source"], produced["arm"], produced["seed"]) == (src, arm, 42)
            and produced["dispatch"] == [{"step": i, "expert": i // 2} for i in range(4)]
            and produced["ffn_calls_by_step"] == [40] * 4
            and tuple(video[k] for k in ("frames", "fps", "height", "width")) == (81, 16, 464, 832)
            and video["mp4_sha256"] == files[f"{output}.mp4"] and digest(video["decoded_rgb24_sha256"]),
            f"{stage}: source video receipt or dispatch differs")
    validate_sites(produced["sites"], produced["gate_witness"], stage, src, arm, archived)
    proof_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {**baseline, "arm": arm, "video_path": str(stage / f"{output}.mp4"), "video_sha256": files[f"{output}.mp4"],
            "generation_proof_key": f"observed_full80__{arm}", "generation_proof_sha256": proof_sha}


def validate_sites(rows, gates, stage, source, arm, archived):
    require(len(rows) == len(gates) == 160, f"{stage}: source is not full80")
    for i, (row, gate) in enumerate(zip(rows, gates, strict=True)):
        step, layer = divmod(i, 40)
        expert, kind = ("high" if step < 2 else "low"), ("hit" if step % 2 else "anchor")
        require((row["step"], row["layer"], row["expert"], row["operator"], row["kind"]) ==
                (step, layer, expert, arm, kind)
                and row["probe_calls"] == int(kind == "hit" and arm in PAIRS), f"{stage}: site contract at {step}/{layer}")
        require(gate["full_write_bitwise_equal"] is True and gate["gate_dtype"] == "torch.bfloat16"
                and all(gate[k] == v for k, v in row.items()), f"{stage}: gate witness differs at {step}/{layer}")
        if kind == "hit":
            query = row["query"]
            expected = f"output/{source}_query/{expert}{layer:02d}.pt"
            require(Path(query["path"]).is_absolute() and Path(query["path"]).samefile(stage / expected)
                    and digest(query["sha256"]) and query["bytes"] > 0
                    and archived.get(expected) == query["sha256"],
                    f"{stage}: unbound query receipt at {step}/{layer}")


def verify(record, scope):
    import quality_wan_proof

    source, arm = source_arm(record)
    if arm in BASE:
        return quality_wan_proof.verify(record, {**scope, "comparison": "main"})
    require(arm in NEW, "Unexpected full80 quality arm")
    historical = scope["historical_scope"]
    require(sha(historical["path"]) == historical["sha256"], "Historical scope changed")
    dense, = [r for r in read(historical["path"])["records"] if source_arm(r) == (source, "Dense")]
    stage = stage_for(scope["producer_stages"], source, arm)
    require(generation_proof(stage, dense, arm) == record, "Full80 generation proof changed")
    video = Path(record["video_path"])
    expected = read(video.with_suffix(".json"))["video"]["decoded_rgb24_sha256"]
    require(sha(scope["ffmpeg_path"]) == scope["ffmpeg_sha256"]
            and quality_wan_proof.decoded(video, scope["ffmpeg_path"]) == expected,
            "Full80 video decoded RGB differs")
    return {"status": "PASS", "proof_sha256": record["generation_proof_sha256"]}


def load_pinned_baseline(root, dim):
    stage = root / "batches" / f"main__{dim}"
    scope_hash, protocol_hash = HISTORICAL[dim]
    require(sha(stage / "scope.json") == scope_hash, f"{dim}: historical Stage 4 scope changed")
    scope, scores, _ = quality_paired.load_batch(stage, dim, paired=False)
    require(scope["comparison"] == "main" and scope["metric_protocol_sha256"] == protocol_hash
            and sha(scope["metric_protocol_path"]) == protocol_hash
            and quality_paired.exact_keys(BASE) <= scores.keys(), f"{dim}: historical baseline lineage changed")
    return scope, scores


def prepare(strict_root, new_root, producer_stages, author, ffprobe):
    """Materialize independently reviewable scopes from verified generation and pinned metrics."""
    require(set(producer_stages) == set(NEW), "Exactly four full80 candidate arms are required")
    for value in producer_stages.values():
        require(isinstance(value, str) or (isinstance(value, dict) and set(value) == set(DEV3)),
                "Producer map must bind all DEV3 sources")
        paths = [value] if isinstance(value, str) else value.values()
        require(all(Path(path).is_absolute() for path in paths), "Producer locators must be absolute")
    ffprobe = ffprobe.resolve()
    require(sha(ffprobe) == FFPROBE_SHA256, "Frozen ffprobe executable differs")

    validate_matched_scopes(producer_stages)
    historical = {dim: load_pinned_baseline(strict_root, dim)[0] for dim in DIMS}
    dense = {r["source"]: r for r in historical["dynamic_degree"]["records"]
             if r["arm"] == "Dense" and r["source"] in DEV3}
    candidates = {
        (source, arm): generation_proof(stage_for(producer_stages, source, arm), dense[source], arm)
        for source in DEV3 for arm in NEW
    }
    require(len({row["video_path"] for row in candidates.values()}) == len(candidates),
            "Candidate generation reuses one video path")
    original_names = ("quality_guard.py", "quality_pair_report.py", "quality_vbench_batch.py",
                      "quality_wan_proof.py", "quality_dense_proof.py", "runtime_guard.py")
    for dim in DIMS:
        old_stage = strict_root / "batches" / f"main__{dim}"
        old = historical[dim]
        base = {source_arm(row): row for row in old["records"]}
        for source in DEV3:
            require(base[source, "Dense"] == dense[source], f"{dim}: historical Dense record differs for {source}")
        stage = new_root / dim
        stage.mkdir(parents=True, exist_ok=False)
        for name in original_names:
            path = old_stage / name
            require(old["implementation_sha256"][str(path.resolve())] == sha(path), f"{dim}: {name} changed")
            shutil.copyfile(path, stage / name)
        wrapper = stage / "quality_full80.py"
        adapter = stage / "quality_paired.py"
        shutil.copyfile(__file__, wrapper)
        shutil.copyfile(quality_paired.__file__, adapter)
        code = {str(stage / name): sha(stage / name) for name in (*original_names, wrapper.name, adapter.name)}
        arms = ARMS if dim == "dynamic_degree" else NEW
        records = [(base if arm in BASE else candidates)[source, arm] for source in DEV3 for arm in arms]
        scope_id = f"FFNDS_OBSERVED_SECANT_FULL80_QUALITY_{dim.upper()}_V1"
        scope = {
            **old, "status": "REVIEW_CANDIDATE", "comparison": COMPARISON, "scope_id": scope_id,
            "active_key": scope_id.lower(), "authors": [author], "remote_stage": str(stage), "records": records,
            "review_path": str(stage / "REVIEW.json"), "active_path": str(stage / "ACTIVE_RESEARCH.json"),
            "receipt_dir": str(stage / "receipt"), "output": str(stage / "output"),
            "runtime_guard_path": str(stage / "runtime_guard.py"),
            "runtime_guard_sha256": code[str(stage / "runtime_guard.py")],
            "generation_verifier_path": str(wrapper), "generation_verifier_sha256": code[str(wrapper)],
            "implementation_sha256": code, "producer_stages": producer_stages,
            "ffprobe": {"path": str(ffprobe), "sha256": FFPROBE_SHA256},
            "historical_scope": {"path": str(old_stage / "scope.json"), "sha256": HISTORICAL[dim][0]},
            "amendment_commit": AMENDMENT, "registered_extra_d4": 0, "registered_extra_nfe": 0,
        }
        scope.pop("bundle_sha256", None)
        with inherited_cohort():
            quality_paired.validate_records(scope)
        write(stage / "scope.json", scope)


def run_guard(scope_path, scope_sha):
    import quality_guard
    import quality_vbench_batch

    scope = read(scope_path)
    own, adapter = (Path(path).resolve() for path in (__file__, quality_paired.__file__))
    require(all(scope["implementation_sha256"].get(str(path)) == sha(path) for path in (own, adapter))
            and scope["amendment_commit"] == AMENDMENT, "Reviewed amended quality implementations differ")
    original = quality_vbench_batch.run
    quality_paired.run_worker.strict_run = original
    quality_vbench_batch.run = quality_paired.run_worker
    try:
        with inherited_cohort():
            quality_paired.validate_records(scope)
            quality_guard.main(scope_path, scope_sha)
    finally:
        quality_vbench_batch.run = original
        del quality_paired.run_worker.strict_run


def paired_report(strict_root, new_root, out):
    metrics = {key: {name: None for name in METRICS} for key in quality_paired.exact_keys(ARMS)}
    videos, lineage = {}, {}
    camera_reasons = {}
    for dim in DIMS:
        previous, baseline = load_pinned_baseline(strict_root, dim)
        new_stage = new_root / dim
        current, fresh, flow = None, {}, {}
        exit_path = new_stage / "receipt/EXIT.json"
        if exit_path.is_file() and read(exit_path)["exit_code"] == 0:
            with inherited_cohort():
                current, fresh, flow = quality_paired.load_batch(new_stage, dim, paired=True)
            require(current["metric_protocol_sha256"] == previous["metric_protocol_sha256"]
                    and current["amendment_commit"] == AMENDMENT, f"{dim}: metric protocol or amendment drift")
        if dim == "dynamic_degree":
            contract = read(previous["paired_contract_path"])
            require(sha(previous["paired_contract_path"]) == previous["paired_contract_sha256"],
                    "Historical camera applicability contract changed")
            for source in DEV3:
                camera_reasons[source] = ("NO_PINNED_CAMERA_METRIC_BATCH"
                    if source in contract["explicit_camera_instruction_by_source"] else "NO_REGISTERED_EXPLICIT_CAMERA_INSTRUCTION")
        previous_rows = {source_arm(r): r for r in previous["records"]}
        new_rows = {source_arm(r): r for r in current["records"]} if current else {}
        for key, values in metrics.items():
            source, arm = key
            if arm in BASE:
                record, score = previous_rows[key], baseline[key]
                if key in fresh:
                    require(new_rows[key] == record and dim == "dynamic_degree"
                            and score["value"] == fresh[key]["value"]
                            and abs(score["continuous_motion"] - fresh[key]["continuous_motion"]) <= 1e-5,
                            f"{dim}: baseline RAFT replay drift at {key}")
            elif key in fresh:
                record, score = new_rows[key], fresh[key]
                require(record["input_identity"] == previous_rows[source, "Dense"]["input_identity"],
                        f"{dim}: sampled video input differs from Dense at {key}")
            else:
                continue
            video = {name: record[name] for name in ("source", "arm", "video_path", "video_sha256",
                                                       "generation_proof_sha256")}
            require(videos.setdefault(key, video) == video, f"{dim}: video changes across dimensions at {key}")
            values[dim] = float(score["value"])
            if dim == "dynamic_degree":
                amplitude = float(score["continuous_motion"])
                require(math.isfinite(amplitude) and amplitude >= 0, f"{key}: invalid RAFT amplitude")
                values["raft_continuous_px_per_2_frames"] = amplitude
                for name in FLOW:
                    value = flow.get(key, {}).get(name)
                    require(value is None or math.isfinite(value), f"{key}: nonfinite RAFT {name}")
                    values[name] = value
        status = "COMPLETE" if current else "FAILED" if exit_path.exists() else "PENDING"
        lineage[dim] = {"status": status, "historical_scope_sha256": HISTORICAL[dim][0],
                        "metric_protocol_sha256": HISTORICAL[dim][1],
                        "new_scope_sha256": sha(new_stage / "scope.json") if current else None,
                        "new_exit_sha256": sha(exit_path) if exit_path.is_file() else None}
    lineage["camera_motion"] = {"status": "NOT_MEASURED", "reasons_by_source": camera_reasons}

    for (source, _), values in metrics.items():
        amplitude = values["raft_continuous_px_per_2_frames"]
        dense_amp = metrics[source, "Dense"]["raft_continuous_px_per_2_frames"]
        if amplitude is not None and dense_amp is not None and dense_amp > FLOOR:
            values["raft_amplitude_ratio_to_dense"] = amplitude / dense_amp

    report = []
    for source in DEV3:
        for arm in ARMS:
            values = metrics[source, arm]
            for name in METRICS:
                value = values[name]
                status = "COMPLETE" if value is not None else "MISSING"
                if name == "camera_motion":
                    status = "NOT_MEASURED"
                elif name == "raft_direction_error_deg" and values["raft_direction_valid_fraction"] == 0:
                    status = "UNDEFINED_DIRECTION"
                elif name == "raft_amplitude_ratio_to_dense":
                    dense_amp = metrics[source, "Dense"]["raft_continuous_px_per_2_frames"]
                    if dense_amp is not None and dense_amp <= FLOOR:
                        status = "UNDEFINED_DENSE_FLOOR"
                row = {"source": source, "arm": arm, "metric": name, "value": value, "status": status,
                       "detail": camera_reasons[source] if name == "camera_motion" else "",
                       "matched_static_arm": PAIRS.get(arm), "delta_vs_matched_static": None}
                for reference in BASE:
                    other = metrics[source, reference][name]
                    row[f"delta_vs_{reference}"] = value - other if value is not None and other is not None else None
                if arm in PAIRS:
                    other = metrics[source, PAIRS[arm]][name]
                    if value is not None and other is not None:
                        row["delta_vs_matched_static"] = value - other
                report.append(row)

    summary = []
    for arm in NEW:
        for name in METRICS:
            for reference in (*BASE, *((PAIRS[arm],) if arm in PAIRS else ())):
                column = "delta_vs_matched_static" if reference == PAIRS.get(arm) else f"delta_vs_{reference}"
                deltas = {r["source"]: r[column] for r in report if r["arm"] == arm and r["metric"] == name}
                present = [value for value in deltas.values() if value is not None]
                summary.append({"arm": arm, "metric": name, "reference": reference,
                                "delta_by_source": deltas, "n_sources": len(present),
                                "equal_source_mean_delta": statistics.mean(present) if present else None})

    out.mkdir(parents=True, exist_ok=False)
    with (out / "QUALITY_PAIRED.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    write(out / "QUALITY_LINEAGE.json", lineage)
    write(out / "DEV3_SUMMARY.json", summary)
    write(out / "PLAYBACK_MANIFEST.json", {
        "status": "HUMAN_REVIEW_PENDING", "frames": 81, "fps": 16,
        "automated_score_gate": False, "cohort": "DEVELOPMENT_EXPOSED_DEV3",
        "raft_semantics": "EPE and direction use the same pixel grid relative to Dense. Pair deltas compare distances to Dense.",
        "sources": [{"source": source, "videos": [videos.get((source, arm), {"arm": arm, "status": "MISSING"})
                                                  for arm in ARMS]} for source in DEV3],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("scope", type=Path)
    run.add_argument("scope_sha256")
    prep = commands.add_parser("prepare")
    prep.add_argument("strict_stage4_root", type=Path)
    prep.add_argument("new_stage4_root", type=Path)
    prep.add_argument("producer_stage_map", type=Path)
    prep.add_argument("author")
    prep.add_argument("--ffprobe", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("strict_stage4_root", type=Path)
    report.add_argument("new_stage4_root", type=Path)
    report.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run_guard(args.scope, args.scope_sha256)
    elif args.command == "prepare":
        prepare(args.strict_stage4_root.resolve(), args.new_stage4_root.resolve(),
                read(args.producer_stage_map), args.author, args.ffprobe)
    else:
        paired_report(args.strict_stage4_root.resolve(), args.new_stage4_root.resolve(), args.out)


if __name__ == "__main__":
    main()
