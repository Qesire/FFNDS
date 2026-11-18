"""Scope-six video proof, fixed-cohort Stage 4 preparation and matched scores."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

SOURCE = "vbi_0392"
ARMS = ("readout-static-6", "readout-online-6", "Global-P64-6")
LAYERS = (4, 19, 34)
SITES = {f"{expert}{layer:02d}" for expert in ("high", "low") for layer in LAYERS}
AMENDMENT = "58f68ab460d47f0f803fd3f2f1b6ec03b651ec21"
COMPARISON = "OBSERVED_SECANT_SCOPE6_0392"
BASES = ("Dense", "Q1r2", "Global-P64", "L0", "N32", "readout-static-80", "readout-online-80")
CONTROL = dict(zip(ARMS, ("readout-static-80", "readout-online-80", "Global-P64"), strict=True))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, doc):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def bound(path, expected):
    require(sha(path) == expected, f"{path}: SHA256 differs")
    return read(path)


def proof(stage, arm, exit_sha=None):
    """Verify completed native generation and return its six bound query locators."""
    stage = Path(stage).resolve()
    paths = ("scope.json", "RUN.json", "EXIT.json", "ATTEMPT_EXIT.json", "REVIEW.json",
             "ACTIVE_RESEARCH.json", "receipt/RUN.json", "output/BINDINGS.json",
             "output/SUMMARY.json", "output/RECIPE_PARITY.json", f"output/{SOURCE}.json", f"output/{SOURCE}.mp4")
    hashes = {p: sha(stage / p) for p in paths}
    if exit_sha is not None:
        require(hashes["EXIT.json"] == exit_sha, f"{stage}: frozen EXIT differs")
    scope, run, done, attempt, review, active, native, binding, summary, recipe, video = (
        read(stage / p) for p in paths[:-1])
    job = str(run["slurm_job_id"])
    scope_sha = hashes["scope.json"]
    active = active[scope["scope_id"].lower()]
    require(arm in ARMS and scope["arm"] == arm and scope["base_commit"] == AMENDMENT
            and scope["method"] == "WAN_FREE_GENERATION"
            and (scope["sources"], scope["seed"], scope["max_d4"], scope["max_nfe"], scope["retries"])
            == ([SOURCE], 42, 1, 4, 0) and scope["force_dense"] is False,
            f"{stage}: wrong scope-six video arm")
    require(done["exit_code"] == attempt["exit_code"] == 0
            and (run["scope_id"], done["scope_id"]) == (scope["scope_id"], scope["scope_id"])
            and all(str(row["slurm_job_id"]) == job for row in (done, attempt, binding, recipe))
            and str(native["allocation"]["job_id"]) == job
            and all(row["scope_sha256"] == scope_sha for row in (run, attempt, review, active, native, binding, recipe))
            and active["runnable"] is True and review["recheck_disposition"] == "APPROVE"
            and review["reviewer_agent_id"] != review["author_agent_id"]
            and active["review_sha256"] == run["review_sha256"] == hashes["REVIEW.json"],
            f"{stage}: scope-six review or completed job differs")
    archive = done["output_sha256"]
    require(all(archive.get(p) == hashes[p] for p in paths[6:]), f"{stage}: output not covered by EXIT")
    require(binding["schema"] == "FFNDS_OBSERVED_SCOPE6_BINDINGS_V1"
            and binding["arm"] == arm and binding["sources"] == [SOURCE]
            and set(binding["candidate_sites"]) == SITES and binding["strict_q1"] == scope["parameters"]
            and binding["strict_q1_sites_outside_candidate"] == 74
            and set(binding["l64_sites"]) == (set() if arm == ARMS[2] else SITES)
            and binding["parameters"] == scope["full80_parameters"]
            and binding["global_p64_scope"] == scope["global_p64_scope"]
            and binding["producer_sha256"] == scope["bundle_sha256"]
            and binding["p0"] == scope["bindings"], f"{stage}: candidate/Q1 binding differs")
    require(recipe["schema"] == "FFNDS_OBSERVED_SCOPE6_RECIPE_PARITY_V1"
            and recipe["strict"] == scope["strict_effective_config"]
            and recipe["exact_bytes_equal"] is True and recipe["nfe_at_check"] == 0
            and recipe["live"]["sha256"] == recipe["strict"]["sha256"]
            and archive["output/EFFECTIVE_CONFIG.json"] == recipe["live"]["sha256"]
            and binding["recipe_parity"]["sha256"] == hashes["output/RECIPE_PARITY.json"],
            f"{stage}: scope-six native recipe differs")
    require((summary["status"], summary["arm"], summary["nfe"]) == ("COMPLETE", arm, 4)
            and summary["records"] == [{"source": SOURCE, "receipt_sha256": hashes[f"output/{SOURCE}.json"]}]
            and (video["source"], video["arm"], video["seed"]) == (SOURCE, arm, 42)
            and video["dispatch"] == [{"step": s, "expert": s // 2} for s in range(4)]
            and video["ffn_calls_by_step"] == [40] * 4 and video["schedule"] == scope["schedule"]
            and video["video"]["mp4_sha256"] == hashes[f"output/{SOURCE}.mp4"]
            and tuple(video["video"][k] for k in ("frames", "fps", "height", "width")) == (81, 16, 464, 832),
            f"{stage}: invalid completed source receipt")
    noise_ref = scope["dense_initial_noise"]
    require(video["initial_noise"] == bound(noise_ref["path"], noise_ref["sha256"])[SOURCE],
            f"{stage}: student initial noise differs from Dense")
    gates = {(row["step"], row["layer"]): row for row in video["gate_witness"]}
    require(len(gates) == len(video["gate_witness"]) == 12 and len(video["sites"]) == 160, "Wrong scope-six topology")
    queries = {}
    for index, row in enumerate(video["sites"]):
        step, layer = divmod(index, 40)
        expert = "high" if step < 2 else "low"
        require((row["step"], row["layer"], row["expert"]) == (step, layer, expert), "Scope-six order differs")
        if layer not in LAYERS:
            require("operator" not in row and "query" not in row and (step, layer) not in gates
                    and row.get("baseline") == "Q1r2" and row.get("probe_calls") == 0,
                    f"{stage}: noncandidate site not strict Q1")
            continue
        gate = gates[step, layer]
        require(row["operator"] == arm and row["kind"] == ("hit" if step % 2 else "anchor")
                and gate["gate_dtype"] == "torch.bfloat16" and gate["full_write_bitwise_equal"] is True
                and all(gate[k] == v for k, v in row.items()), f"{stage}: wrong candidate current gate or site")
        require(row.get("probe_calls", 0) == int(step % 2 == 1 and arm != ARMS[0]), "Wrong candidate probe count")
        if step % 2 == 0:
            require("query" not in row, f"{stage}: an anchor must not carry a student hit query")
        if arm == "Global-P64-6" and step % 2:
            require(row["selected_k"] == 5120 and row["probe_kind"] == "native_w1_only",
                    f"{stage}: wrong Global-P64 query producer")
        if step % 2:
            key = f"{expert}{layer:02d}"
            rel = f"output/{SOURCE}_query/{key}.pt"
            ref = row["query"]
            path = stage / rel
            require(Path(ref["path"]) == path and ref["sha256"] == archive[rel]
                    and path.is_file() and path.stat().st_size == ref["bytes"], f"{stage}: missing query proof at {key}")
            queries[key] = {"asset": ref, "gate_sha256": gate["gate_sha256"]}
    require(set(queries) == SITES, f"{stage}: incomplete six query hits")
    token = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return scope, binding, video, queries, job, token


def quality_modules():
    try:
        import quality_full80 as full
    except ModuleNotFoundError as exc:
        if exc.name != "quality_full80":
            raise
        from ffnds_observed_operator import quality_full80 as full
    return full, full.quality_paired


def candidate_record(stage, baseline, arm):
    scope, binding, video, _, _, proof_sha = proof(stage, arm)
    v2 = Path(scope["v2_reference_dir"])
    records = bound(v2 / "v2_conditions.json", scope["reference_sha256"]["v2_conditions.json"])
    condition, = (r for r in records if r["record_id"] == SOURCE)
    identity = baseline["input_identity"]
    require(all(condition[k] == baseline[k] for k in ("prompt_sha256", "image_sha256", "source_cluster_id", "seed"))
            and {k: v["sha256"] for k, v in binding["checkpoints"].items()} == identity["checkpoint_sha256"]
            and video["schedule"] == identity["scheduler"] and video["initial_noise"] == identity["initial_noise"],
            f"{stage}: scope-six input differs from Dense")
    return {**baseline, "arm": arm, "video_path": str(Path(stage).resolve() / f"output/{SOURCE}.mp4"),
            "video_sha256": video["video"]["mp4_sha256"], "generation_proof_key": f"observed_scope6__{arm}",
            "generation_proof_sha256": proof_sha}

DYNAMIC_BASES = ("Dense", "Q1r2", "Global-P64", "readout-static-80")


def archived(main_root, p2_root, dim):
    full, paired = quality_modules()
    with full.inherited_cohort():
        main = paired.load_batch(main_root / dim, dim, paired=True)
    stage = p2_root / dim
    historical = paired.load_batch(stage, dim, paired=True) if (stage / "receipt/EXIT.json").is_file() else None
    return main, historical


def verify(record, scope):
    import quality_wan_proof

    full, _ = quality_modules()
    arm = record["arm"]
    if arm in DYNAMIC_BASES[:3]:
        return quality_wan_proof.verify(record, {**scope, "comparison": "main"})
    if arm == "readout-static-80":
        old = scope["main_quality_scope"]
        old_scope = bound(old["path"], old["sha256"])
        require(record in old_scope["records"], "Archived readout-static-80 metric record changed")
        return full.verify(record, old_scope)
    require(arm in ARMS, "Unexpected scope-six verifier arm")
    old = scope["historical_scope"]
    dense, = (r for r in bound(old["path"], old["sha256"])["records"]
              if (r["source"], r["arm"]) == (SOURCE, "Dense"))
    stage = scope["scope6_producer_stages"][arm]
    require(candidate_record(stage, dense, arm) == record, "Scope-six video proof changed")
    video = Path(record["video_path"])
    expected = read(video.with_suffix(".json"))["video"]["decoded_rgb24_sha256"]
    require(sha(scope["ffmpeg_path"]) == scope["ffmpeg_sha256"]
            and quality_wan_proof.decoded(video, scope["ffmpeg_path"]) == expected, "Scope-six decoded video differs")
    return {"status": "PASS", "proof_sha256": record["generation_proof_sha256"]}


def cohort_call(worker, *args, **kwargs):
    _, paired = quality_modules()
    saved = paired.DEV3, paired.NEW, paired.ARMS, paired.COMPARISON
    paired.DEV3, paired.NEW, paired.ARMS, paired.COMPARISON = (SOURCE,), ARMS, DYNAMIC_BASES + ARMS, COMPARISON
    try:
        return worker(*args, **kwargs)
    finally:
        paired.DEV3, paired.NEW, paired.ARMS, paired.COMPARISON = saved


def prepare(strict_root, p2_root, main_root, new_root, stages, author, ffprobe):
    full, paired = quality_modules()
    require(set(stages) == set(ARMS) and all(Path(v).is_absolute() for v in stages.values())
            and len(set(stages.values())) == 3, "Expected three separate scope-six producer stages")
    require(sha(ffprobe) == paired.FFPROBE_SHA256, "Pinned ffprobe differs")
    gen = {arm: read(Path(stages[arm]) / "scope.json") for arm in ARMS}
    fields = ("bindings", "parameters", "strict_effective_config", "schedule", "dense_initial_noise",
              "native_transformer_sha256", "global_p64_scope", "reference_sha256")
    common = {key: gen[ARMS[0]][key] for key in fields}
    require(all({key: entry[key] for key in fields} == common for entry in gen.values())
            and gen[ARMS[0]]["full80_parameters"] == gen[ARMS[1]]["full80_parameters"],
            "Scope-six generation arms do not share frozen base, query coordinates and matched readout parameters")
    spec_sha = gen[ARMS[0]]["experiment_spec"]["sha256"]
    for arm, entry in gen.items():
        spec = Path(stages[arm]) / "experiment_spec.json"
        require(entry["experiment_spec"] == {"path": str(spec), "sha256": spec_sha} and sha(spec) == spec_sha,
                f"{arm}: the frozen experiment specification differs")
    old = {dim: full.load_pinned_baseline(strict_root, dim)[0] for dim in paired.DIMS}
    dense, = (r for r in old["dynamic_degree"]["records"] if (r["source"], r["arm"]) == (SOURCE, "Dense"))
    candidates = {arm: candidate_record(stages[arm], dense, arm) for arm in ARMS}
    require(len({r["video_path"] for r in candidates.values()}) == 3, "Scope-six clip path aliases")
    originals = ("quality_guard.py", "quality_pair_report.py", "quality_vbench_batch.py",
                 "quality_wan_proof.py", "quality_dense_proof.py", "runtime_guard.py")
    for dim in paired.DIMS:
        old_stage = strict_root / "batches" / f"main__{dim}"
        stage = new_root / dim
        scope = old[dim]
        main, p2 = archived(main_root, p2_root, dim)
        main_scope, _, _ = main
        require(main_scope["metric_protocol_sha256"] == scope["metric_protocol_sha256"]
                and (p2 is None or p2[0]["metric_protocol_sha256"] == scope["metric_protocol_sha256"]),
                f"{dim}: metric lineage changed")
        historic = {r["arm"]: r for r in scope["records"] if r["source"] == SOURCE}
        require(historic["Dense"] == dense, f"{dim}: historical Dense scope differs")
        records = list(candidates.values())
        if dim == "dynamic_degree":
            archived_readout, = (r for r in main_scope["records"]
                                 if (r["source"], r["arm"]) == (SOURCE, "readout-static-80"))
            records = [*(historic[arm] for arm in DYNAMIC_BASES[:3]), archived_readout, *records]
        stage.mkdir(parents=True, exist_ok=False)
        for name in originals:
            path = old_stage / name
            require(sha(path) == scope["implementation_sha256"][str(path.resolve())], f"Changed pinned {name}")
            shutil.copyfile(path, stage / name)
        adapters = {"quality_full80.py": full.__file__, "quality_paired.py": paired.__file__,
                    "quality_scope6.py": __file__}
        for name, target in adapters.items():
            shutil.copyfile(target, stage / name)
        code = {str(stage / name): sha(stage / name) for name in (*originals, *adapters)}
        scope_id = f"FFNDS_OBSERVED_SECANT_SCOPE6_0392_QUALITY_{dim.upper()}_V1"
        main_path = main_root / dim / "scope.json"
        new_scope = {
            **scope, "comparison": COMPARISON, "scope_id": scope_id, "active_key": scope_id.lower(),
            "authors": [author], "status": "REVIEW_CANDIDATE", "records": records, "remote_stage": str(stage),
            "review_path": str(stage / "REVIEW.json"), "active_path": str(stage / "ACTIVE_RESEARCH.json"),
            "receipt_dir": str(stage / "receipt"), "output": str(stage / "output"),
            "runtime_guard_path": str(stage / "runtime_guard.py"),
            "runtime_guard_sha256": code[str(stage / "runtime_guard.py")],
            "generation_verifier_path": str(stage / "quality_scope6.py"),
            "generation_verifier_sha256": code[str(stage / "quality_scope6.py")],
            "implementation_sha256": code, "scope6_producer_stages": stages,
            "ffprobe": {"path": str(ffprobe), "sha256": paired.FFPROBE_SHA256},
            "historical_scope": {"path": str(old_stage / "scope.json"), "sha256": sha(old_stage / "scope.json")},
            "main_quality_scope": {"path": str(main_path), "sha256": sha(main_path)},
            "amendment_commit": AMENDMENT, "registered_extra_nfe": 0, "registered_extra_d4": 0,
        }
        new_scope.pop("bundle_sha256", None)
        cohort_call(paired.validate_records, new_scope)
        write(stage / "scope.json", new_scope)


def run_guard(scope_path, scope_sha):
    import quality_guard
    import quality_vbench_batch

    full, paired = quality_modules()
    scope = read(scope_path)
    require(scope["amendment_commit"] == AMENDMENT
            and all(scope["implementation_sha256"].get(str(Path(p).resolve())) == sha(p)
                    for p in (__file__, full.__file__, paired.__file__)), "Reviewed metric adapter changed")
    original = quality_vbench_batch.run
    paired.run_worker.strict_run = original
    quality_vbench_batch.run = paired.run_worker
    try:
        cohort_call(quality_guard.main, scope_path, scope_sha)
    finally:
        quality_vbench_batch.run = original
        del paired.run_worker.strict_run


def report(strict_root, p2_root, main_root, new_root, out):
    full, paired = quality_modules()
    context = BASES + ARMS
    values = {arm: {metric: None for metric in full.METRICS} for arm in context}
    lineage, videos = {}, {}
    for dim in paired.DIMS:
        old, baseline = full.load_pinned_baseline(strict_root, dim)
        main, p2 = archived(main_root, p2_root, dim)
        stage = new_root / dim
        fresh = cohort_call(paired.load_batch, stage, dim, paired=True)
        batches = [(old, baseline, {}), main, fresh]
        if p2 is not None:
            batches.insert(1, p2)
        for scope, scores, flows in batches:
            require(scope["metric_protocol_sha256"] == old["metric_protocol_sha256"], f"{dim}: score protocol changed")
            for r in scope["records"]:
                src, arm = r["source"], r["arm"]
                if src != SOURCE or arm not in context:
                    continue
                old_video = {name: r[name] for name in ("source", "arm", "video_path", "video_sha256")}
                require(videos.setdefault(arm, old_video) == old_video, f"{dim}: {arm} video differs")
                cell, score = values[arm], scores[(src, arm)]
                value = score["value"]
                require(cell[dim] is None or cell[dim] == value, f"{dim}: {arm} score replay drift")
                cell[dim] = value
                if dim == "dynamic_degree":
                    motion = score["continuous_motion"]
                    amp = "raft_continuous_px_per_2_frames"
                    require(cell[amp] is None or abs(cell[amp] - motion) <= 1e-5, f"{arm}: RAFT replay drift")
                    cell[amp] = motion
                    for name in paired.FLOW:
                        if (src, arm) in flows:
                            observed = flows[src, arm][name]
                            require(cell[name] is None or observed is not None and abs(cell[name] - observed) <= 1e-5,
                                    f"{arm}: matched Dense flow replay drift at {name}")
                            cell[name] = observed
        lineage[dim] = {"scope6_scope_sha256": sha(stage / "scope.json"),
                        "scope6_exit_sha256": sha(stage / "receipt/EXIT.json"),
                        "metric_protocol_sha256": old["metric_protocol_sha256"],
                        "main_scope_sha256": sha(main_root / dim / "scope.json"),
                        "strict_scope_sha256": full.HISTORICAL[dim][0],
                        "p2_scope_sha256": sha(p2_root / dim / "scope.json") if p2 else None}
    lineage["camera_motion"] = {"status": "NOT_MEASURED", "reason": "NO_REGISTERED_EXPLICIT_CAMERA_INSTRUCTION"}
    amp = "raft_continuous_px_per_2_frames"
    dense_amp = values["Dense"][amp]
    for cell in values.values():
        if cell[amp] is not None and dense_amp > paired.FLOOR:
            cell["raft_amplitude_ratio_to_dense"] = cell[amp] / dense_amp
    rows = []
    for arm, cell in values.items():
        for metric, value in cell.items():
            status = "COMPLETE" if value is not None else "MISSING"
            if metric == "camera_motion":
                status = "NOT_MEASURED"
            elif metric == "raft_direction_error_deg" and cell["raft_direction_valid_fraction"] == 0:
                status = "UNDEFINED_DIRECTION"
            elif metric == "raft_amplitude_ratio_to_dense" and dense_amp <= paired.FLOOR:
                status = "UNDEFINED_DENSE_FLOOR"
            row = {"source": SOURCE, "arm": arm, "metric": metric, "value": value, "status": status,
                   "same_method_full_scope_arm": CONTROL.get(arm)}
            for reference in (*BASES, ARMS[0]):
                other = values[reference][metric]
                row[f"delta_vs_{reference}"] = value - other if arm != reference and value is not None and other is not None else None
            rows.append(row)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "QUALITY_PAIRED.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(out / "QUALITY_LINEAGE.json", lineage)
    write(out / "PLAYBACK_MANIFEST.json", {"source": SOURCE, "status": "HUMAN_REVIEW_PENDING", "automated_score_gate": False,
                                          "videos": [videos.get(arm, {"arm": arm, "status": "MISSING"}) for arm in context]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    summary = commands.add_parser("report")
    for command in (prep, summary):
        for key in ("strict_root", "p2_quality_root", "main_quality_root", "scope6_quality_root"):
            command.add_argument(key, type=Path)
    prep.add_argument("producer_stage_map", type=Path)
    prep.add_argument("author")
    prep.add_argument("--ffprobe", type=Path, required=True)
    summary.add_argument("out", type=Path)
    run = commands.add_parser("run")
    run.add_argument("scope", type=Path)
    run.add_argument("scope_sha256")
    args = parser.parse_args()
    if args.command == "run":
        run_guard(args.scope, args.scope_sha256)
        return
    roots = tuple(getattr(args, key).resolve() for key in
                  ("strict_root", "p2_quality_root", "main_quality_root", "scope6_quality_root"))
    if args.command == "prepare":
        prepare(*roots, read(args.producer_stage_map), args.author, args.ffprobe.resolve())
    else:
        report(*roots, args.out)


if __name__ == "__main__":
    main()
