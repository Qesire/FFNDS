"""Fixed DEV3 metric orchestration and source-paired summaries.

Use the strict Stage 4 worker/guard in each reviewed batch directory. This file
extends cohort admission and observes the existing RAFT inference. It does not
launch Slurm jobs or produce videos.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from fractions import Fraction
from pathlib import Path

DEV3 = ("vbi_0392", "vbi_1013", "vbi_0800")
BASE = ("Dense", "Q1r2", "Global-P64")
NEW = ("L0", "L+", "Q32", "N32")
ARMS = BASE + NEW
DIMS = ("i2v_subject", "i2v_background", "imaging_quality", "subject_consistency",
        "background_consistency", "motion_smoothness", "temporal_flickering",
        "aesthetic_quality", "dynamic_degree")
FLOW = ("raft_endpoint_error_px", "raft_magnitude_mae_px", "raft_direction_error_deg",
        "raft_direction_valid_fraction")
FLOOR = 0.001
COMPARISON = "STATE_OPERATOR_DEV3"
FFPROBE_SHA256 = "040709d6ae5123136c2c397e7164e4d2cc909b81e1357263795feb3c09a07d7d"
COLOR_FIELDS = ("color_range", "color_space", "color_transfer", "color_primaries", "chroma_location")
FRAME_GRID = tuple(range(81))
RAFT_FRAME_IDS = FRAME_GRID[::2]
RAFT_PAIRS = tuple(zip(RAFT_FRAME_IDS, RAFT_FRAME_IDS[1:]))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def exact_keys(arms):
    return {(source, arm) for source in DEV3 for arm in arms}


def source_arm(row):
    return row["source"], row["arm"]


def validate_records(scope):
    dim, records = scope["dimension"], scope["records"]
    allowed_arms = ARMS if dim == "dynamic_degree" else NEW
    expected = [(source, arm) for source in DEV3 for arm in allowed_arms]
    if dim not in DIMS or scope["comparison"] != COMPARISON:
        raise ValueError("Invalid state-operator DEV3 metric scope")
    if list(map(source_arm, records)) != expected:
        raise ValueError("Expected frozen source-major records for this metric")
    return records


def color_tags(record):
    unknown = {"", "unknown", "unspecified", "reserved", None}
    return {key: None if record.get(key) in unknown else record[key] for key in COLOR_FIELDS}


def validate_probe(payload):
    """Compare exact rational presentation times, independent of container time base."""
    streams, frames = payload["streams"], payload["frames"]
    if len(streams) != 1 or len(frames) != len(FRAME_GRID):
        raise ValueError("Expected one video stream and exactly 81 decoded video frames")
    stream, = streams
    time_base = Fraction(stream["time_base"])
    if (stream["width"], stream["height"]) != (832, 464) or time_base <= 0:
        raise ValueError("Video grid or declared time base mismatch")
    if Fraction(stream["avg_frame_rate"]) != 16 or Fraction(stream["r_frame_rate"]) != 16:
        raise ValueError("Expected CFR at declared average and nominal 16 fps")
    if int(stream["start_pts"]) != 0 or (stream.get("start_time") is not None and
                                             Fraction(stream["start_time"]) != 0):
        raise ValueError("Expected source frame zero at presentation time zero")
    frame_tags = []
    for index, frame in zip(FRAME_GRID, frames):
        pts = int(frame["pts"])
        if (pts * time_base != Fraction(index, 16) or int(frame["best_effort_timestamp"]) != pts
                or (frame["width"], frame["height"], frame["pix_fmt"]) !=
                (832, 464, stream["pix_fmt"])):
            raise ValueError(f"Decoded frame {index} differs in PTS, frame order, grid or pixel format")
        if "duration" in frame and int(frame["duration"]) * time_base != Fraction(1, 16):
            raise ValueError(f"Decoded frame {index} has unequal duration")
        frame_tags.append(color_tags(frame))
    if any(tags != frame_tags[0] for tags in frame_tags):
        raise ValueError("Frame color tags vary across presentation times")
    stream_tags = color_tags(stream)
    if any(stream_tags[k] is not None and frame_tags[0][k] is not None and
           stream_tags[k] != frame_tags[0][k] for k in COLOR_FIELDS):
        raise ValueError("Stream and frame color declarations disagree")
    return {"time_base": str(time_base), "codec": stream["codec_name"],
            "pixel_format": stream["pix_fmt"], "stream_color_tags": stream_tags,
            "frame_color_tags": frame_tags[0], "frame_index_ticks_1_over_16": list(FRAME_GRID)}


def probe_videos(records, ffprobe, check_resources):
    if ffprobe["sha256"] != FFPROBE_SHA256 or sha(ffprobe["path"]) != FFPROBE_SHA256:
        raise ValueError("School ffprobe identity differs")
    evidence = []
    for record in records:
        media = record["video_path"]
        if sha(media) != record["video_sha256"]:
            raise ValueError("Video differs before ffprobe")
        result = subprocess.run([
            ffprobe["path"], "-v", "error", "-select_streams", "v",
            "-show_streams", "-show_frames", "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,time_base,start_pts,start_time,"
            + ",".join(COLOR_FIELDS) + ":frame=pts,best_effort_timestamp,duration,width,height,pix_fmt,"
            + ",".join(COLOR_FIELDS), "-of", "json", media,
        ], check=True, capture_output=True, text=True, timeout=120)
        if result.stderr.strip():
            raise ValueError(f"ffprobe reported video decode errors for {media}")
        evidence.append({"source": record["source"], "arm": record["arm"],
                         "video_sha256": record["video_sha256"], **validate_probe(json.loads(result.stdout))})
        check_resources()
    contract = [{key: cell[key] for key in ("codec", "pixel_format", "stream_color_tags", "frame_color_tags",
                                            "frame_index_ticks_1_over_16")} for cell in evidence]
    if any(row != contract[0] for row in contract):
        raise ValueError("Codec, pixel format or declared color parity differs across the scoped clips")
    return evidence


def generation_proof(stage, base, arm):
    """Bind one completed state-operator generation to the strict Dense source."""
    stage = Path(stage).resolve()
    src = base["source"]
    files = {name: sha(stage / name) for name in (
        "scope.json", "REVIEW.json", "ACTIVE_RESEARCH.json", "receipt/RUN.json",
        "receipt/EXIT.json", "output/BINDINGS.json", "output/SUMMARY.json",
        f"output/{src}.json", f"output/{src}.mp4")}
    scope = read(stage / "scope.json")
    review = read(stage / "REVIEW.json")
    active = read(stage / "ACTIVE_RESEARCH.json")[scope["scope_id"].lower()]
    run = read(stage / "receipt/RUN.json")
    exit_ = read(stage / "receipt/EXIT.json")
    binding = read(stage / "output/BINDINGS.json")
    summary = read(stage / "output/SUMMARY.json")
    produced = read(stage / "output" / f"{src}.json")
    identity = base["input_identity"]
    reference = Path(scope["v2_reference_dir"])
    for name in ("scope.json", "v2_conditions.json"):
        if sha(reference / name) != scope["reference_sha256"][name]:
            raise ValueError("State-operator V2 source identity changed")
    prior = read(reference / "scope.json")
    condition = next(row for row in read(reference / "v2_conditions.json") if row["record_id"] == src)
    condition_fields = ("image_sha256", "prompt_sha256", "source_cluster_id", "seed")
    if any(condition[key] != base[key] for key in condition_fields):
        raise ValueError("State-operator source differs from historical Dense")
    scope_id, job = files["scope.json"], str(binding["slurm_job_id"])
    if (scope["method"] != "WAN_FREE_GENERATION" or tuple(scope["sources"]) != DEV3
            or scope["arm"] != arm or scope["seed"] != 42 or scope["max_nfe"] != 12
            or (scope["force_dense"] is not False)
            or review["recheck_disposition"] != "APPROVE" or not review["reviewer_agent_id"]
            or review["reviewer_agent_id"] == review.get("author_agent_id")
            or any(row["scope_sha256"] != scope_id for row in (review, active, run, binding))
            or active["review_sha256"] != files["REVIEW.json"] or active["runnable"] is not True
            or exit_["exit_code"] != 0 or str(exit_["slurm_job_id"]) != job
            or str(run["allocation"]["job_id"]) != job or binding["arm"] != arm
            or (summary["status"], summary["arm"], summary["nfe"]) != ("COMPLETE", arm, 12)
            or [row["source"] for row in summary["records"]] != list(DEV3)
            or next(row["receipt_sha256"] for row in summary["records"] if row["source"] == src) != files[f"output/{src}.json"]
            or any(binding[key] != scope[parent] for key, parent in
                   (("fit", "fit"), ("p0", "bindings"), ("p1", "bf16_acceptance"), ("strict_q1", "parameters")))
            or prior["recipe_binding"]["recipe_sha256"] != "ee780352c8cf1c1e4efc8aea7c354b3be225c525141942cbd9c15d24cabae5b6"
            or {key: value["sha256"] for key, value in binding["checkpoints"].items()} != identity["checkpoint_sha256"]
            or (produced["source"], produced["arm"], produced["seed"]) != (src, arm, 42)
            or produced["dispatch"] != [{"step": i, "expert": i // 2} for i in range(4)]
            or len(produced["sites"]) != 160
            or produced["initial_noise"] != identity["initial_noise"]
            or produced["schedule"] != identity["scheduler"]
            or produced["video"]["mp4_sha256"] != files[f"output/{src}.mp4"]
            or tuple(produced["video"][k] for k in ("frames", "fps", "height", "width")) != (81, 16, 464, 832)
            or not produced["video"]["decoded_rgb24_sha256"]):
        raise ValueError(f"Failed same-source state operator generation admission at {stage}")
    token = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    record = {**base, "arm": arm, "video_path": str(stage / "output" / f"{src}.mp4"),
              "video_sha256": files[f"output/{src}.mp4"],
              "generation_proof_key": f"state_operator__{arm}",
              "generation_proof_sha256": token}
    return record


def verify(record, scope):
    """Generation proof callback used by the strict Stage 4 quality guard."""
    key = source_arm(record)
    if key[1] in NEW:
        stage = scope["p2_producer_stages"][key[1]]
        historical_scope = Path(scope["p2_historical_scope"]["path"])
        if sha(historical_scope) != scope["p2_historical_scope"]["sha256"]:
            raise ValueError("Strict origin metric scope changed")
        base = next(x for x in read(historical_scope)["records"] if source_arm(x) == (key[0], "Dense"))
        if generation_proof(stage, base, key[1]) != record:
            raise ValueError("State-operator generation proof changed")
        import quality_wan_proof

        video = Path(record["video_path"])
        expected = read(video.with_suffix(".json"))["video"]["decoded_rgb24_sha256"]
        if sha(scope["ffmpeg_path"]) != scope["ffmpeg_sha256"] or quality_wan_proof.decoded(video, scope["ffmpeg_path"]) != expected:
            raise ValueError("State-operator decoded RGB identity differs")
        return {"status": "PASS", "proof_sha256": record["generation_proof_sha256"]}
    import quality_wan_proof

    return quality_wan_proof.verify(record, {**scope, "comparison": "main"})


def prepare(strict_root, new_root, producer_stages, author, *, ffprobe=None):
    """Materialize exact-input scopes as REVIEW_CANDIDATE for real review."""
    if ffprobe is None:
        ffprobe = Path.home() / "miniconda3/envs/vbench_i2v/bin/ffprobe"
    probe_path = Path(ffprobe).resolve()
    if sha(probe_path) != FFPROBE_SHA256:
        raise ValueError("FFprobe does not match the frozen executable")
    probe_binding = {"path": str(probe_path), "sha256": FFPROBE_SHA256}
    for dim in DIMS:
        old_stage = strict_root / "batches" / f"main__{dim}"
        old_scope, _, _ = load_batch(old_stage, dim, paired=False)
        samples = {source_arm(row): row for row in old_scope["records"]}
        stage = (new_root / dim).resolve()
        stage.mkdir(parents=True, exist_ok=False)
        originals = ("quality_guard.py", "quality_pair_report.py", "quality_vbench_batch.py",
                     "quality_wan_proof.py", "quality_dense_proof.py", "runtime_guard.py")
        for name in originals:
            source = old_stage / name
            if old_scope["implementation_sha256"][str(source.resolve())] != sha(source):
                raise ValueError(f"Strict metric implementation changed for {name}")
            shutil.copyfile(source, stage / name)
        wrapper = stage / Path(__file__).name
        shutil.copyfile(__file__, wrapper)
        code = {str(stage / name): sha(stage / name) for name in (*originals, wrapper.name)}
        arms = ARMS if dim == "dynamic_degree" else NEW
        records = []
        for src in DEV3:
            for arm in arms:
                records.append(samples[src, arm] if arm in BASE else
                               generation_proof(producer_stages[arm], samples[src, "Dense"], arm))
        scope_id = f"FFNDS_STATE_OPERATOR_P2_QUALITY_{dim.upper()}_V1"
        scope = {**old_scope, "status": "REVIEW_CANDIDATE", "comparison": COMPARISON,
                 "scope_id": scope_id, "active_key": scope_id.lower(), "authors": [author],
                 "remote_stage": str(stage), "records": records,
                 "review_path": str(stage / "REVIEW.json"), "active_path": str(stage / "ACTIVE_RESEARCH.json"),
                 "receipt_dir": str(stage / "receipt"), "output": str(stage / "output"),
                 "runtime_guard_path": str(stage / "runtime_guard.py"),
                 "runtime_guard_sha256": code[str(stage / "runtime_guard.py")],
                 "generation_verifier_path": str(wrapper), "generation_verifier_sha256": sha(wrapper),
                 "implementation_sha256": code, "p2_producer_stages": producer_stages,
                 "ffprobe": probe_binding,
                 "p2_historical_scope": {"path": str((old_stage / "scope.json").resolve()),
                                         "sha256": sha(old_stage / "scope.json")}}
        scope.pop("bundle_sha256", None)
        validate_records(scope)
        write(stage / "scope.json", scope)


def flow_statistics(blocks, dense_blocks):
    """Pixel-aligned native RAFT statistics over the same 40 2-frame pairs."""
    import numpy as np

    if len(blocks) != 40 or len(dense_blocks) != 40:
        raise ValueError("Expected 40 RAFT comparisons")
    epe, amplitude, angles, valid, total = 0., 0., 0., 0, 0
    for flow, dense in zip(blocks, dense_blocks):
        if flow.shape != (2, 464, 832) or dense.shape != flow.shape:
            raise ValueError("Mismatch on the fixed 464x832 RAFT grid")
        flow, dense = flow.astype("float64"), dense.astype("float64")
        f_norm, d_norm = np.linalg.norm(flow, axis=0), np.linalg.norm(dense, axis=0)
        epe += np.linalg.norm(flow - dense, axis=0).sum()
        amplitude += np.abs(f_norm - d_norm).sum()
        mask = (f_norm > FLOOR) & (d_norm > FLOOR)
        if mask.any():
            dot = (flow * dense).sum(0)[mask]
            cos = dot / (f_norm[mask] * d_norm[mask])
            angles += np.degrees(np.arccos(np.clip(cos, -1., 1.))).sum()
        valid += int(mask.sum())
        total += mask.size
    return dict(zip(FLOW, (float(epe / total), float(amplitude / total),
                           float(angles / valid) if valid else None, valid / total)))


def run_worker(scope, *, scope_sha256, authorize, verify_generation, check_resources):
    import quality_vbench_batch as strict

    authorize(scope)
    records = validate_records(scope)
    probes = probe_videos(records, scope["ffprobe"], check_resources)
    current = [None]
    arrays = {}
    pair_order = {}
    sampled = [None]
    strict_run = run_worker.strict_run
    old_dev3 = strict.DEV3
    strict.DEV3 = DEV3
    strict.COMPARISONS[COMPARISON] = ARMS if scope["dimension"] == "dynamic_degree" else NEW
    dynamic = None
    bench_class = None
    try:
        if scope["dimension"] == "dynamic_degree":
            import vbench.dynamic_degree as dynamic_module
            from vbench2_beta_i2v import VBenchI2V

            infer, forward = dynamic_module.DynamicDegree.infer, dynamic_module.RAFT.forward
            extract_frame = dynamic_module.DynamicDegree.extract_frame
            dynamic = dynamic_module
            bench_class = VBenchI2V
            build_info = bench_class.build_full_info_json

            def extended_full_info(self, videos_path, *args, **kwargs):
                """Append the two reviewed slots beyond the standard 0–4 inventory."""
                path = Path(build_info(self, videos_path, *args, **kwargs))
                content = read(path)
                prompts = {row["prompt"] for row in records}
                if len(content) != len(DEV3) or {row["prompt_en"] for row in content} != prompts:
                    raise ValueError("Unexpected native VBench RAFT source inventory")
                for row in content:
                    prompt = row["prompt_en"]
                    prefix = [str(Path(videos_path) / f"{prompt}-{i}.mp4") for i in range(5)]
                    if row["video_list"] != prefix:
                        raise ValueError("Unexpected ordering of native VBench RAFT slots")
                    suffix = [Path(videos_path) / f"{prompt}-{i}.mp4" for i in (5, 6)]
                    if not all(file.is_symlink() and file.is_file() for file in suffix):
                        raise ValueError("Missing reviewed extra RAFT video slots")
                    row["video_list"].extend(map(str, suffix))
                path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
                return str(path)

            def observed_infer(self, path, *args, **kwargs):
                resolved = str(Path(path).resolve())
                if resolved in arrays:
                    raise ValueError("Duplicate RAFT inference for one video")
                arrays[resolved] = []
                pair_order[resolved] = []
                current[0] = resolved
                try:
                    result = infer(self, path, *args, **kwargs)
                    if tuple(pair_order[resolved]) != RAFT_PAIRS:
                        raise ValueError("RAFT did not evaluate the ordered stride-two frame pairs")
                    return result
                finally:
                    current[0] = None
                    sampled[0] = None

            def observed_extract(self, frame_list, interval=1):
                frames = extract_frame(self, frame_list, interval)
                if (current[0] is None or sampled[0] is not None or len(frame_list) != 81
                        or interval != 2 or len(frames) != 41
                        or any(frame is not frame_list[i] for frame, i in zip(frames, RAFT_FRAME_IDS))):
                    raise ValueError("Native RAFT did not sample decoded frames 0:81:2")
                sampled[0] = frames
                return frames

            def observed_forward(self, *args, **kwargs):
                import torch

                key = current[0]
                if key is None or sampled[0] is None or len(args) != 2:
                    raise ValueError("RAFT inference lacks an observed native sample list")
                slot = len(pair_order[key])
                if slot >= len(RAFT_PAIRS) or any(
                    actual.shape != (1, 3, 464, 832) or not torch.equal(actual, expected)
                    for actual, expected in zip(args, sampled[0][slot:slot + 2])
                ):
                    raise ValueError(f"RAFT forward inputs disagree with native sampled frame-pair slot {slot}")
                result = forward(self, *args, **kwargs)
                flow = result[1]
                if tuple(flow.shape) != (1, 2, 464, 832):
                    raise ValueError("Unexpected native RAFT flow shape or lifecycle")
                arrays[key].append(flow[0].detach().cpu().float().numpy().copy())
                pair_order[key].append(RAFT_PAIRS[slot])
                return result

            dynamic.DynamicDegree.infer, dynamic.RAFT.forward = observed_infer, observed_forward
            dynamic.DynamicDegree.extract_frame = observed_extract
            bench_class.build_full_info_json = extended_full_info
        result = strict_run(scope, scope_sha256=scope_sha256, authorize=lambda _: None,
                            verify_generation=verify_generation, check_resources=check_resources)
    finally:
        strict.DEV3 = old_dev3
        strict.COMPARISONS.pop(COMPARISON, None)
        if dynamic is not None:
            dynamic.DynamicDegree.infer, dynamic.RAFT.forward = infer, forward
            dynamic.DynamicDegree.extract_frame = extract_frame
        if bench_class is not None:
            bench_class.build_full_info_json = build_info
    probe_path = Path(scope["output"]) / "VIDEO_FRAME_CONTRACT.json"
    write(probe_path, probes)
    result["video_frame_contract_sha256"] = sha(probe_path)
    if scope["dimension"] == "dynamic_degree":
        lookup = {source_arm(row): row for row in records}
        stats = []
        for record in records:
            expected_dense = lookup[record["source"], "Dense"]
            flow = arrays[str(Path(record["video_path"]).resolve())]
            dense = arrays[str(Path(expected_dense["video_path"]).resolve())]
            item = flow_statistics(flow, dense)
            check_resources()
            stats.append({"source": record["source"], "arm": record["arm"],
                          "video_sha256": record["video_sha256"],
                          "dense_video_sha256": expected_dense["video_sha256"],
                          "scope_sha256": scope_sha256,
                          "metric_protocol_sha256": scope["metric_protocol_sha256"],
                          "pairs": 40, "pair_stride_frames": 2, "grid": [464, 832],
                          "verified_frame_index_pairs": pair_order[str(Path(record["video_path"]).resolve())],
                          **item})
        path = Path(scope["output"]) / "paired_raft.json"
        write(path, stats)
        result["paired_raft_sha256"] = sha(path)
    return result


def run_guard(scope_path, scope_sha):
    import quality_guard
    import quality_vbench_batch

    scope = read(scope_path)
    code = scope["implementation_sha256"]
    own = str(Path(__file__).resolve())
    if code.get(own) != sha(own):
        raise ValueError("The reviewed scope must bind this wrapper")
    validate_records(scope)
    run_worker.strict_run = quality_vbench_batch.run
    quality_vbench_batch.run = run_worker
    try:
        quality_guard.main(scope_path, scope_sha)
    finally:
        quality_vbench_batch.run = run_worker.strict_run


def load_batch(stage, dimension, *, paired):
    stage = Path(stage)
    scope_path, exit_path = stage / "scope.json", stage / "receipt/EXIT.json"
    scope, receipt = read(scope_path), read(exit_path)
    if (scope["dimension"] != dimension or receipt["scope_sha256"] != sha(scope_path)
            or receipt["exit_code"] != 0 or receipt["result"]["status"] != "COMPLETE"):
        raise ValueError(f"Unverified metric completion at {stage}")
    records = validate_records(scope) if paired else scope["records"]
    expected = {source_arm(record): record for record in records}
    output = Path(scope["output"])
    score_path = output / "per_video_scores.json"
    if sha(score_path) != receipt["result"]["per_video_sha256"]:
        raise ValueError(f"Per-video artifact digest mismatch at {stage}")
    rows = read(score_path)
    scores = {}
    for row in rows:
        key = source_arm(row)
        reference = expected[key]
        if (key in scores or row["dimension"] != dimension or row["status"] != "COMPLETE"
                or row["scope_sha256"] != receipt["scope_sha256"]
                or row["metric_protocol_sha256"] != scope["metric_protocol_sha256"]
                or any(row[field] != reference[field] for field in
                       ("video_sha256", "input_identity", "source_cluster_id", "generation_proof_sha256"))
                or not math.isfinite(row["value"])):
            raise ValueError(f"Invalid per-video record at {stage}")
        scores[key] = row
    if set(scores) != set(expected):
        raise ValueError(f"Missing per-video records at {stage}")
    if paired:
        probe_path = output / "VIDEO_FRAME_CONTRACT.json"
        if sha(probe_path) != receipt["result"]["video_frame_contract_sha256"]:
            raise ValueError("Per-video frame contract digest mismatch")
        contract = read(probe_path)
        if ([source_arm(row) for row in contract] != list(expected)
                or any(row["video_sha256"] != expected[source_arm(row)]["video_sha256"]
                       or row["frame_index_ticks_1_over_16"] != list(FRAME_GRID) for row in contract)):
            raise ValueError("Frame contracts fail the frozen video identity or PTS grid")
    flow = {}
    if paired and dimension == "dynamic_degree":
        vector_path = output / "paired_raft.json"
        if not vector_path.exists():
            raise ValueError("Dynamic batch did not export paired RAFT vectors")
        if sha(vector_path) != receipt["result"]["paired_raft_sha256"]:
            raise ValueError("Paired RAFT artifact digest mismatch")
        for row in read(vector_path):
            key = source_arm(row)
            if (key in flow or row["video_sha256"] != expected[key]["video_sha256"]
                    or row["dense_video_sha256"] != expected[(key[0], "Dense")]["video_sha256"]
                    or row["metric_protocol_sha256"] != scope["metric_protocol_sha256"]
                    or row["scope_sha256"] != receipt["scope_sha256"]
                    or row["grid"] != [464, 832] or row["pairs"] != 40 or row["pair_stride_frames"] != 2
                    or row["verified_frame_index_pairs"] != [list(pair) for pair in RAFT_PAIRS]):
                raise ValueError("Paired RAFT origin or grid mismatch")
            flow[key] = row
        if set(flow) != set(expected):
            raise ValueError("Incomplete paired RAFT vectors")
    return scope, scores, flow


def paired_report(strict_root, new_root, out):
    metrics = {key: {} for key in exact_keys(ARMS)}
    lineage = {}
    videos = {}
    for dim in DIMS:
        strict_stage = strict_root / "batches" / f"main__{dim}"
        old_path = strict_stage / "scope.json"
        scope = read(old_path) if old_path.is_file() else None
        baseline = {}
        if scope and (strict_stage / "receipt/EXIT.json").is_file():
            if (Path(scope["output"]) / "per_video_scores.json").is_file():
                scope, baseline, _ = load_batch(strict_stage, dim, paired=False)
        new_stage = new_root / dim
        paired = (new_stage / "receipt/EXIT.json").exists()
        current, fresh, flow = load_batch(new_stage, dim, paired=True) if paired else (None, {}, {})
        if current and scope and scope["metric_protocol_sha256"] != current["metric_protocol_sha256"]:
            raise ValueError(f"Historical and new metric protocols differ for {dim}")
        base_records = {source_arm(row): row for row in scope["records"]} if scope else {}
        new_records = {source_arm(row): row for row in current["records"]} if current else {}
        for key in metrics:
            source, arm = key
            if key in base_records:
                record = base_records[key]
                if key in new_records and new_records[key] != record:
                    raise ValueError("Historical baseline record changed")
                score = baseline.get(key)
                if score is not None and key in fresh and (abs(score["continuous_motion"] - fresh[key]["continuous_motion"]) > 1e-5
                                     or score["value"] != fresh[key]["value"]):
                    raise ValueError("Historical RAFT scalar replay drift")
            elif key in new_records:
                record, score = new_records[key], fresh[key]
                dense_record = base_records.get((source, "Dense"))
                if dense_record and record["input_identity"] != dense_record["input_identity"]:
                    raise ValueError("New video and Dense have different input identity")
            else:
                record = score = None
            if record:
                video = {"source": source, "arm": arm, "path": record["video_path"],
                         "sha256": record["video_sha256"]}
                if key in videos and videos[key] != video:
                    raise ValueError("Video identity changed across metric dimensions")
                videos[key] = video
            values = metrics[key]
            values[dim] = float(score["value"]) if score else None
            if dim == "dynamic_degree":
                values["raft_continuous_px_per_2_frames"] = score["continuous_motion"] if score else None
                values.update({name: flow.get(key, {}).get(name) for name in FLOW})
        lineage[dim] = {"historical_scope_sha256": sha(old_path) if scope else None,
                        "historical_score_availability": "VERIFIED" if baseline else "MISSING",
                        "new_scope_sha256": sha(new_stage / "scope.json") if paired else None,
                        "metric_protocol_sha256": scope["metric_protocol_sha256"] if scope else None}

    report = []
    stationary = {"dynamic_degree"}
    for source in DEV3:
        for arm in ARMS:
            values, dense = metrics[source, arm], metrics[source, "Dense"]
            for dim, value in values.items():
                references = ("Dense", "Q1r2", "Global-P64", "L0")
                status = ("COMPLETE" if value is not None else
                          "UNDEFINED_DIRECTION" if dim == "raft_direction_error_deg"
                          and values["raft_direction_valid_fraction"] == 0 else "MISSING")
                row = {"source": source, "arm": arm, "metric": dim,
                       "value": value, "status": status,
                       "selection_metric": dim not in stationary}
                for ref in references:
                    other = metrics[source, ref][dim]
                    row[f"delta_vs_{ref}"] = value - other if value is not None and other is not None else None
                report.append(row)
            amplitude = values["raft_continuous_px_per_2_frames"]
            dense_amp = dense["raft_continuous_px_per_2_frames"]
            if amplitude is not None and dense_amp is not None and dense_amp > FLOOR:
                ratio = amplitude / dense_amp
            else:
                ratio = None
            status = ("COMPLETE" if ratio is not None else
                      "UNDEFINED_DENSE_FLOOR" if dense_amp is not None and dense_amp <= FLOOR else "MISSING")
            report.append({"source": source, "arm": arm, "metric": "raft_amplitude_ratio_to_dense",
                           "value": ratio, "status": status,
                           "selection_metric": True})
    out.mkdir(parents=True, exist_ok=False)
    with (out / "QUALITY_PAIRED.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    write(out / "QUALITY_LINEAGE.json", lineage)
    summary = []
    for arm in NEW:
        for dim in report_metrics():
            for reference in (*BASE, "L0"):
                if arm == reference:
                    continue
                rows = [r for r in report if r["arm"] == arm and r["metric"] == dim]
                deltas = {r["source"]: r.get(f"delta_vs_{reference}") for r in rows}
                valid = {source: value for source, value in deltas.items() if value is not None}
                summary.append({"arm": arm, "metric": dim, "reference": reference, "n_sources": len(valid),
                                "delta_by_source": deltas, "equal_source_mean_delta": statistics.mean(valid.values()) if valid else None,
                                "min_delta_source": min(valid, key=valid.get) if valid else None,
                                "max_delta_source": max(valid, key=valid.get) if valid else None})
    write(out / "DEV3_SUMMARY.json", summary)
    write(out / "PLAYBACK_MANIFEST.json", {
        "status": "PENDING_HUMAN_REVIEW", "frames": 81, "fps": 16,
        "sources": [{"source": source, "videos": [videos.get((source, arm), {"arm": arm, "status": "MISSING"})
                                                   for arm in ARMS]} for source in DEV3],
    })


def report_metrics():
    return (*DIMS, "raft_continuous_px_per_2_frames", *FLOW)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("scope", type=Path)
    run.add_argument("scope_sha256")
    preparation = commands.add_parser("prepare")
    preparation.add_argument("strict_stage4_root", type=Path)
    preparation.add_argument("new_stage4_root", type=Path)
    preparation.add_argument("producer_stage_map", type=Path,
                             help="JSON mapping L0, L+, Q32, N32 to reviewed producer stage directories")
    preparation.add_argument("author")
    preparation.add_argument("--ffprobe", type=Path, help="Path to the frozen ffprobe executable")
    report = commands.add_parser("report")
    report.add_argument("strict_stage4_root", type=Path)
    report.add_argument("new_stage4_root", type=Path)
    report.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run_guard(args.scope, args.scope_sha256)
    elif args.command == "prepare":
        stages = read(args.producer_stage_map)
        if set(stages) != set(NEW):
            raise ValueError("Exactly four reviewed producer stages are required")
        prepare(args.strict_stage4_root.resolve(), args.new_stage4_root.resolve(),
                {arm: str(Path(stage).resolve()) for arm, stage in stages.items()}, args.author,
                ffprobe=args.ffprobe)
    else:
        paired_report(args.strict_stage4_root, args.new_stage4_root, args.out)


if __name__ == "__main__":
    main()
