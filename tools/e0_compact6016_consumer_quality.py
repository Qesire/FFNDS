"""Four new A0 clips, two same-byte Dense score reuses; fixed VBench subset.

Import, inspect and draft are stdlib only. Native scoring requires a frozen
protocol, completed six-video producer and an independently exited byte-only
CPU preflight. Original evaluator equations and frame sampling are retained.
"""
import argparse
import collections
import copy
import filecmp
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_COMPACT6016_CONSUMER00_12_QUALITY_20261028"
PARENT_ID = "E0_COMPACT6016_CONSUMER00_12_SIX_VIDEOS_OFFLOAD_AUX_V2_20261028"
PARENT_STATUS = "COMPLETE_MATCHED_SIX_NATIVE_VIDEOS_CORE_METRICS_QUALITY_PENDING"
LEGACY_ID = "E0_UB1_CONSUMER_DISJOINT_QUALITY_HISTORY_20261027"
LEGACY_PROTOCOL_SHA = "d5e72fda97cf405666bdd21fa5828b8a38339b65717057367430a037c94adc49"
PARENT_PROTOCOL = "config/e0_compact6016_six_videos_20261028_v2.json"
LEGACY_PROTOCOL = "config/e0_ub1_consumer_disjoint_quality_history_20261027_v1.json"
LEGACY_REPORT = "research/20261027/UB1_CONSUMER_DISJOINT20_QUALITY_RESULTS_20261027.json"
SOURCES = (0, 12)
MODES = ("dense", "energy94", "random94")
DIMENSIONS = ("imaging_quality", "aesthetic_quality", "motion_smoothness", "dynamic_degree",
              "subject_consistency", "background_consistency", "i2v_subject", "i2v_background")
COMMON = DIMENSIONS[:6]
COSTS = {"imaging_quality": {"MUSIQ": 324},
         "aesthetic_quality": {"CLIP_L14": 12, "aesthetic_head": 12},
         "motion_smoothness": {"AMT": 160},
         "dynamic_degree": {"RAFT": 160, "RAFT_iterations": 3200},
         "subject_consistency": {"DINO": 324},
         "background_consistency": {"CLIP_B32": 4},
         "i2v_subject": {"DINO": 164}, "i2v_background": {"DreamSim_embed": 164}}
LIMITS = {"official_dimension_calls": 8, "new_video_dimension_results": 28,
          "reused_Dense_video_dimension_results": 14, "reused_core_reports": 6,
          "RAFT_forwards": 160, "RAFT_iterations": 3200,
          "video_generations": 0, "training_updates": 0, "decompositions": 0,
          "CPU_metric_recomputations": 0}
RESOURCES = {"GPU": 1, "session_max_GPUs": 2, "repository_GPU_policy": "DYNAMIC_MAX_ACTUALLY_CALLABLE",
             "CPU_threads": 2, "wall_seconds": 1800, "outer_seconds": 2100,
             "new_HF_bytes": 256 << 20, "GPU_reserved_bytes": 8 << 30,
             "free_disk_bytes": 10 << 30, "automatic_retry": False}
ENVIRONMENT_KEYS = ("asset_manifest", "helper", "v8_protocol", "v8_directory", "site", "overlay", "source_root", "v8_receipt", "nvidia_smi")
CPU_STATUS = "PASS_FOUR_NEW_TWO_REUSED_QUALITY_BYTES_NO_NUMPY_NO_TORCH_NO_GPU"
REQUIRED_SOURCES = {
    "tools/e0_compact6016_consumer_quality.py", "tests/test_e0_compact6016_consumer_quality.py",
    "tools/e0_ub1_consumer_official_quality.py", "tools/e0_ub1_consumer_disjoint_quality_history.py",
    "tools/e0_official_metric_environment_v8.py", "tools/offline_i2v_metric_loaders.py",
    "tools/rollout_contract.py", "tools/ffnds_video_metrics.py", "tools/ffnds_stage_registered_i2v.py",
    "tools/ffnds_cohort_preflight.py", "tools/ffnds_experiment_plans.py", "tools/worker_identity.py",
    "tools/startup_gpu_reservation.py", "tools/assigned_gpu_resources.py",
    "config/e0_official_metric_environment_20261027_v8.json",
    "config/e0_official_metric_environment_lock_20261026_v1.json", ".runtime/R17_metric_assets.json"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def binding(path, base):
    path, base = Path(path).resolve(), Path(base).resolve()
    require(path.is_relative_to(base), "Binding escaped root")
    return {"path": str(path.relative_to(base)), "bytes": path.stat().st_size, "sha256": sha(path)}


def bound(base, item):
    base = Path(base).resolve()
    require(isinstance(item, dict) and type(item.get("bytes")) is int and item["bytes"] >= 0
            and isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64, "Asset NOT_BOUND")
    path = (base / item["path"]).resolve()
    require(path.is_relative_to(base) and path.is_file() and path.stat().st_size == item["bytes"]
            and sha(path) == item["sha256"], "Asset bytes/SHA differ: " + str(item.get("path")))
    return path


def read_bound(base, item):
    return read(bound(base, item))


def clip_binding(video, hf):
    return {"path": str(Path(video["path"]).resolve().relative_to(Path(hf).resolve())),
            "bytes": video["bytes"], "sha256": video["sha256"]}


def validate_plan(plan, *, frozen=False):
    require(plan.get("id") == ID and plan.get("dimensions") == list(DIMENSIONS)
            and plan.get("costs") == COSTS and plan.get("limits") == LIMITS
            and plan.get("resources") == RESOURCES and plan.get("automatic_retry") is False,
            "Fixed quality scope/budget differs")
    require(len(plan.get("videos", [])) == 6
            and {(x["source"], x["mode"]) for x in plan["videos"]} == {(s, m) for s in SOURCES for m in MODES},
            "Exactly six source/mode records required")
    require(len(plan.get("images", [])) == 2 and {x["source"] for x in plan["images"]} == set(SOURCES)
            and len({x["binding"]["sha256"] for x in plan["images"]}) == 2, "Two distinct image clusters required")
    require(plan["parent"]["protocol"]["path"] == PARENT_PROTOCOL
            and plan["legacy"]["protocol"]["path"] == LEGACY_PROTOCOL
            and plan["legacy"]["protocol"]["sha256"] == LEGACY_PROTOCOL_SHA
            and plan["legacy"]["published_results"]["path"] == LEGACY_REPORT, "Fixed producer/score identities differ")
    require(len(plan["legacy"]["parts"]) == 2 and {p["part"] for p in plan["legacy"]["parts"]} == {0, 1},
            "Both exited legacy score parts required")
    paths = [x["path"] for x in plan["repo_sources"]]
    require(len(paths) == len(set(paths)) and REQUIRED_SOURCES <= set(paths), "Missing or duplicate immutable software sources")
    require(set(plan["environment"]) == set(ENVIRONMENT_KEYS), "Metric environment closure differs")
    require(len(plan["hf_bindings"]) == len({x["path"] for x in plan["hf_bindings"]}), "Duplicate evaluator asset binding")
    output = Path(plan["output_HF_relative"])
    require(not output.is_absolute() and ".." not in output.parts and output.parts[:1] == ("ffnds-derived",), "Output path escaped HF")
    if frozen:
        require(plan.get("assets_finalized") is True and plan.get("binding_status") == "BOUND_READY_FOR_REGISTRATION",
                "Quality draft is not frozen/bound")


def exited(receipt, run_binding, protocol_sha, identity, *, legacy_part=None):
    if legacy_part is None:
        require(receipt.get("id") == identity and receipt.get("protocol_sha256") == protocol_sha
                and type(receipt.get("child_returncode")) is int and receipt["child_returncode"] == 0
                and receipt.get("process_exited") is True and receipt.get("timeout") is False
                and receipt.get("automatic_retry") is False and receipt.get("run_sha256") == run_binding["sha256"],
                "Actual exit0/exact RUN SHA receipt required")
    else:
        # Historical launcher did not write a RUN SHA; its independently
        # published RUN SHA is required by verify_legacy instead of invented.
        require(receipt.get("id") == identity and receipt.get("protocol_sha256") == protocol_sha
                and receipt.get("part") == legacy_part and type(receipt.get("launcher_returncode")) is int
                and receipt["launcher_returncode"] == 0 and receipt.get("process_exited") is True
                and receipt.get("automatic_retry") is False and receipt.get("timeout") is not True,
                "Legacy actual exit0/protocol/part receipt required")


def verify_generation(plan, hf, *, repo=ROOT):
    parent = read_bound(repo, plan["parent"]["protocol"])
    run = read_bound(hf, plan["parent"]["RUN"])
    receipt = read_bound(hf, plan["parent"]["exit"])
    digest = plan["parent"]["protocol"]["sha256"]
    require(parent.get("id") == run.get("id") == PARENT_ID and run.get("status") == PARENT_STATUS
            and run.get("protocol_sha256") == digest and run.get("automatic_retry") is False,
            "Six-video producer incomplete or wrong identity")
    exited(receipt, plan["parent"]["RUN"], digest, PARENT_ID)
    require(not run.get("assigned_GPU_resources", {}).get("violations"), "Producer GPU monitor violation")
    for phase in ("predebited", "observed"):
        require(all(run.get("operation_ledger", {}).get(phase, {}).get(k, 0) == v for k, v in parent["limits"].items()),
                "Six-video operation ledger incomplete")
    phases = run.get("phases", [])
    require(len(phases) == 6 and {(x["source_index"], x["mode"]) for x in phases} == {(s, m) for s in SOURCES for m in MODES},
            "Six unique completed producer phases required")
    phase_by_key = {(x["source_index"], x["mode"]): x for x in phases}
    cases = {x["index"]: x for x in parent["cases"]}
    require(set(cases) == set(SOURCES), "Producer image cohort differs")
    paths, runs, cores = [], set(), []
    for item in plan["videos"]:
        key = item["source"], item["mode"]
        phase, case = phase_by_key[key], cases[item["source"]]
        generation = read_bound(hf, item["generation"])
        registered = read_bound(hf, item["registered"])
        require(phase.get("status") == generation.get("status") == "PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE"
                and phase.get("completed_native_flow_steps") == generation.get("completed_native_flow_steps") == 4
                and phase["generation_receipt"]["sha256"] == item["generation"]["sha256"]
                and phase["registered_receipt"]["sha256"] == item["registered"]["sha256"], "Immutable generation registration differs")
        require(phase["run_id"] == generation["run_id"] == registered["job"]["run_id"]
                and phase["run_id"] not in runs, "Duplicate or wrong real generation run")
        runs.add(phase["run_id"])
        require(registered.get("status") == "COMPLETE_NATIVE_GENERATION_ONLY"
                and registered.get("immutable_generation_receipt") == phase["generation_receipt"]
                and registered.get("native_ffn_sites") == 160 and registered.get("scheduler_generator_reset") is True,
                "Registered generation incomplete")
        require(all(item[k] == case[k] for k in ("source_id", "image_sha256", "prompt_sha256", "prompt"))
                and item["seed"] == case["seed"] == 42 and hashlib.sha256(item["prompt"].encode()).hexdigest() == item["prompt_sha256"],
                "Source/prompt/seed identity differs")
        require(all(generation[k] == phase[k] for k in ("source_id", "image_sha256", "prompt_sha256", "seed", "mode", "source_index"))
                and all(generation[k] == item[k] for k in ("source_id", "image_sha256", "prompt_sha256", "seed", "mode"))
                and generation["source_index"] == item["source"] and generation["mode"] == item["mode"]
                and generation["protocol_sha256"] == digest, "Generation source/mode/protocol differs")
        video = bound(hf, item["video"])
        require(item["video"] == clip_binding(generation["video"], hf) == clip_binding(phase["video"], hf)
                and generation["video"]["frames"] == registered["decoded_frames"] == 81
                and generation["video"]["fps"] == registered["decoded_fps"] == 16
                and generation["video"]["size"] == registered["decoded_size"] == [832, 464]
                and Path(registered["video_path"]).resolve() == video, "Video bytes/geometry/registration differs")
        paths.append(video)
        for name, field in (("initial_noise", "initial_noise"), ("raw_noise", "raw_FP32_noise")):
            require(item[name] == generation[field] == phase[field], "Saved noise binding differs")
            bound(hf, item[name])
        require(registered["initial_noise"] == generation["initial_noise"], "Registered noise differs")
        core = read_bound(hf, item["core"])
        require(item["core"]["sha256"] == phase["core_metrics"]["sha256"]
                and phase["core_metrics"]["status"] == "ALL_81_FRAMES_CORE_METRICS_PASS_QUALITY_PENDING", "Core report binding differs")
        from ffnds_video_metrics import validate
        require(not validate(core) and core["candidate"]["sha256"] == item["video"]["sha256"]
                and core["identity"]["source_id"] == case["image_sha256"]
                and core["identity"]["seed"] == 42 and core["identity"]["expected_frames"] == 81
                and core["identity"]["input_prompt_digest"] == case["prompt_sha256"]
                and core["identity"]["protocol_sha256"] == digest
                and core["identity"]["run_sha256"] == item["generation"]["sha256"], "All-frame core report/source identity differs")
        dense = next(v for v in plan["videos"] if v["source"] == item["source"] and v["mode"] == "dense")
        if item["mode"] == "dense":
            require(not core.get("reference") and phase.get("historical_comparison", {}).get("passed") is True
                    and item["video"]["sha256"] == case["historical_dense"]["video"]["sha256"], "Dense historical/core identity differs")
        else:
            require(core.get("reference", {}).get("sha256") == dense["video"]["sha256"]
                    and core["reference_identity"]["run_sha256"] == dense["generation"]["sha256"], "Core paired Dense reference differs")
            require(phase.get("actual_initial_noise_files_byte_equal_to_fresh_dense") is True
                    and phase.get("actual_raw_FP32_noise_byte_equal_to_fresh_dense") is True, "Producer noise pairing was not completed")
            for name in ("initial_noise", "raw_noise"):
                require(filecmp.cmp(bound(hf, item[name]), bound(hf, dense[name]), shallow=False), "Actual Dense/candidate noise bytes differ")
        cores.append(item["core"])
    require(len(set(paths)) == len({(p.stat().st_dev, p.stat().st_ino) for p in paths}) == 6,
            "Six real video files required; path/inode aliases rejected")
    cohort_plan = read_bound(hf, plan["parent"]["cohort_plan"])
    verified = read_bound(hf, plan["parent"]["verified_cohort"])
    require(run["verified_generation_cohort"]["sha256"] == plan["parent"]["verified_cohort"]["sha256"], "Verified collection differs from producer RUN")
    from ffnds_stage_registered_i2v import collect
    producer_root = Path(bound(hf, plan["parent"]["RUN"])).parent
    require(verified == collect(cohort_plan, producer_root)
            and verified["phase"] == "COMPLETE_REGISTERED_GENERATION_VERIFIED"
            and len(verified["sources"]) == 2 and len({x["image_id"] for x in verified["sources"]}) == 2,
            "Actual registered collection/source clusters differ")
    for image in plan["images"]:
        path = bound(hf, image["binding"])
        source = next(s for s in verified["sources"] if s["source_id"] == cases[image["source"]]["source_id"])
        require(image["binding"]["sha256"] == cases[image["source"]]["image_sha256"] == source["image_id"]
                and image["name"] == source["image_name"] and Path(source["image_path"]).resolve() == path,
                "Original image staging/source cluster differs")
    return parent, run, cores


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def raft_summary(scores):
    require(isinstance(scores, list) and len(scores) == 40
            and all(finite_number(x) and x >= 0 for x in scores), "Exactly40 finite nonnegative RAFT pair values required")
    return {"mean_top5percent_flow_px_per_2frame_pair": statistics.fmean(scores),
            "five_fixed_8pair_windows": [
                {"pair_first": start, "pair_last": start + 7, "frame_first": 2 * start,
                 "frame_last": 2 * (start + 8), "mean_top5percent_flow_px": statistics.fmean(scores[start:start + 8])}
                for start in range(0, 40, 8)],
            "pair_scores": list(scores), "independent_samples": 1,
            "identity": "original RAFT get_score top5% flow, not mean whole-image flow or object velocity"}


def validate_raft(row):
    summary = raft_summary(row.get("scores"))
    require(row.get("params") == {"thres": 10.875, "count_num": 10}, "Original RAFT binary threshold/sampling differs")
    expected = sum(x > 10.875 for x in row["scores"]) >= 10
    require(type(row.get("official_binary")) is bool and row["official_binary"] == expected, "Official binary does not match original40pair rule")
    return summary


def rows_for_dimension(videos, dimension, *, new_only=True):
    require(dimension in DIMENSIONS, "Unknown quality dimension")
    return [x for x in videos if (not new_only or x["mode"] != "dense")
            and (dimension in COMMON or x["source"] == (0 if dimension == "i2v_subject" else 12))]


def verify_legacy(plan, parent, hf, *, repo=ROOT):
    old = read_bound(repo, plan["legacy"]["protocol"])
    published = read_bound(repo, plan["legacy"]["published_results"])
    require(old.get("id") == LEGACY_ID and old.get("dimensions") == list(DIMENSIONS)
            and published.get("protocol_sha256") == LEGACY_PROTOCOL_SHA,
            "Original score protocol/published identity differs")
    require(plan["environment"] == {k: old[k] for k in ENVIRONMENT_KEYS}
            and plan["hf_bindings"] == old["hf_bindings"], "Evaluator source/model/environment differs from reused Dense scores")
    original_generation = read_bound(repo, old["generation_protocol"])
    original_cases = {c["index"]: c for c in original_generation["cases"]}
    cases = {c["index"]: c for c in parent["cases"]}
    reused, raft, precisions = {}, {}, []
    for part in plan["legacy"]["parts"]:
        index = part["part"]
        require(part["RUN"]["sha256"] == published["artifact_sha256"][f"part{index}_run"]
                and part["exit"]["sha256"] == published["artifact_sha256"][f"part{index}_launch"], "Legacy published RUN/exit SHA differs")
        run, receipt = read_bound(hf, part["RUN"]), read_bound(hf, part["exit"])
        require(run.get("id") == LEGACY_ID and run.get("status") == "COMPLETE_SEVEN_PART_DIMENSIONS"
                and run.get("part") == index and run.get("protocol_sha256") == LEGACY_PROTOCOL_SHA
                and not run.get("assigned_gpu_resources", {}).get("violations"), "Legacy scoring part incomplete")
        exited(receipt, part["RUN"], LEGACY_PROTOCOL_SHA, LEGACY_ID, legacy_part=index)
        precisions.append(run["runtime_precision"])
        source = 0 if index == 0 else 12
        item = next(x for x in old["videos"] if x["source"] == source and x["mode"] == "dense")
        original = original_cases[source]
        entries = [s for s in original_generation["source_bank"]["sources"] if s["source_id"] == original["semantic_source_id"]]
        require(len(entries) == 1, "Original source-bank semantic mapping is not unique")
        semantic = entries[0]
        require(item["source_id"] == original["source_id"] == original["image"]["sha256"] == cases[source]["image_sha256"]
                and original["semantic_source_id"] == cases[source]["source_id"] == semantic["source_id"]
                and semantic["legacy_image_fingerprint"] == cases[source]["image_sha256"]
                and semantic["prompt"] == item["prompt"] and semantic["image_name"] == item["image_name"]
                and original["dimension"] == ("i2v_subject" if source == 0 else "i2v_background"),
                "Legacy fingerprint/semantic source-bank mapping differs")
        require(item["video"]["sha256"] == cases[source]["historical_dense"]["video"]["sha256"]
                and item["generation"]["sha256"] == cases[source]["historical_dense"]["generation"]["sha256"]
                and item["image_sha256"] == cases[source]["image_sha256"]
                and item["prompt_sha256"] == cases[source]["prompt_sha256"] and item["seed"] == 42,
                "Reused Dense video/source/image/prompt differs")
        historical = read_bound(hf, item["generation"])
        require(historical["video"]["sha256"] == item["video"]["sha256"] and historical["mode"] == "dense"
                and historical["source_id"] == item["source_id"]
                and historical["status"] == "PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE", "Legacy Dense generation differs")
        video = str(bound(hf, item["video"]))
        expected = list(COMMON) + ["i2v_subject" if source == 0 else "i2v_background"]
        require(len(run["completed"]) == 7 and {x["dimension"] for x in run["completed"]} == set(expected), "Legacy score dimension coverage differs")
        reused[source] = {}
        for entry in run["completed"]:
            dim = entry["dimension"]
            input_path = Path(bound(hf, part["RUN"])).parent / (dim + "_INPUT.json")
            require(sha(input_path) == entry["input_sha256"], "Original score INPUT SHA differs")
            inputs = [x for x in read(input_path) if x.get("video_list") == [video]]
            require(len(inputs) == 1 and inputs[0]["prompt_en"] == item["prompt"]
                    and inputs[0]["image_name"] == item["image_name"] and inputs[0]["dimension"] == [dim], "Original score image/prompt INPUT differs")
            values = [x for x in entry["result"][-1] if x.get("video_path") == video]
            require(len(values) == 1, "Missing/duplicate reused Dense score")
            value = values[0]["video_results"]
            require(type(value) is bool if dim == "dynamic_degree" else finite_number(value), "Nonfinite/wrong legacy score type")
            reused[source][dim] = value
            if dim == "dynamic_degree":
                rows = [x for x in entry["RAFT_observations"] if x["video_path"] == video]
                require(len(rows) == 1 and rows[0]["official_binary"] == value, "Reused Dense RAFT observations differ")
                validate_raft(rows[0]); raft[source] = rows[0]
    require(precisions[0] == precisions[1] == {"matmul_TF32": False, "float32_matmul_precision": "highest",
                                             "bf16_reduced_precision_reduction": True}, "Legacy numeric runtime flags differ")
    return reused, raft, precisions[0]


def verify_assets(plan, hf, *, repo=ROOT):
    for item in plan["repo_sources"]:
        bound(repo, item)
    parent, run, cores = verify_generation(plan, hf, repo=repo)
    reused, raft, precision = verify_legacy(plan, parent, hf, repo=repo)
    for item in plan["hf_bindings"]:
        bound(hf, item)
    env = plan["environment"]
    v8 = read(Path(repo) / env["v8_protocol"])
    receipt = read_bound(hf, env["v8_receipt"])
    require(receipt.get("status") == "PASS_IMPORTS_ONLY" and receipt.get("protocol_sha256") == sha(Path(repo) / env["v8_protocol"]),
            "Original metric import environment receipt differs")
    return {"parent": run, "reused": reused, "raft": raft, "precision": precision, "cores": cores}


def authorize(plan, digest, active):
    validate_plan(plan, frozen=True)
    user, policy = active.get("user_continuation_authorization_20261027", {}), active.get("resource_scheduling_policy_20261028", {})
    require(user.get("status") == "ACTIVE_CURRENT_USER_DIRECTED_SESSION" and user.get("new_scope_registration_allowed") is True
            and policy.get("mode") == "DYNAMIC_MAX_ACTUALLY_CALLABLE" and policy.get("global_gpu_card_cap") is None
            and active.get("current_session_resource_constraint_20261028", {}).get("max_simultaneously_assigned_idle_GPUs") == 2,
            "Current session/repository GPU authority differs")
    scopes = [s for s in active["experiment_authority"]["current_user_authorized_scopes"] if s.get("id") == ID]
    require(len(scopes) == 1, "Unique quality registration required")
    scope = scopes[0]
    require(scope.get("protocol_sha256") == digest and scope.get("execution_status") in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
            and scope.get("max_gpu_cards") == 1 and scope.get("limits") == LIMITS and scope.get("resources") == RESOURCES,
            "Exact registered quality budget differs")
    return scope


def cpu_report(plan, digest):
    return {"id": ID, "status": CPU_STATUS, "protocol_sha256": digest, "sources_sha256": state_sha(plan["repo_sources"]),
            "producer_RUN_sha256": plan["parent"]["RUN"]["sha256"], "producer_exit_sha256": plan["parent"]["exit"]["sha256"],
            "new_scored_videos": 4, "reused_Dense_videos": 2, "reused_core_reports": 6,
            "independent_image_clusters": 2, "all_CUDA_devices_hidden": True, "NumPy_or_Torch_imported": False,
            "model_calls": 0, "GPU_calls": 0, "official_dimension_calls": 0, "training_updates": 0}


def require_cpu_receipt(scope, plan, digest, hf):
    cpu = scope.get("cpu_preflight")
    require(isinstance(cpu, dict) and {"RUN", "launch_exit"} <= set(cpu), "Exited independent CPU preflight NOT_BOUND")
    report, receipt = read_bound(hf, cpu["RUN"]), read_bound(hf, cpu["launch_exit"])
    expected = cpu_report(plan, digest)
    require(all(report.get(k) == v for k, v in expected.items()), "CPU preflight proof differs")
    exited(receipt, cpu["RUN"], digest, ID)
    return cpu


def precision_environment():
    require(not any(os.environ.get(k) == "1" for k in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")),
            "Forced TF32 environment conflicts with reused Dense scores")


def preflight(protocol, hf):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules and "numpy" not in sys.modules,
            "Byte-only CPU preflight requires hidden CUDA/no numerical imports")
    precision_environment()
    plan = read(protocol); validate_plan(plan, frozen=True)
    verify_assets(plan, hf)
    verify_environment(plan, hf)
    require("torch" not in sys.modules and "numpy" not in sys.modules, "Numerical import during byte-only preflight")
    return cpu_report(plan, sha(protocol))


class Ledger:
    def __init__(self, report, *, limits=None, persist=lambda: None):
        self.report, self.limits, self.persist = report, dict(LIMITS if limits is None else limits), persist
        for phase in ("predebited", "invoked", "completed"):
            report[phase] = dict.fromkeys(self.limits, 0)

    @contextmanager
    def __call__(self, key, count=1):
        require(key in self.limits and type(count) is int and count > 0
                and self.report["predebited"][key] + count <= self.limits[key], "Quality operation budget exceeded: " + key)
        self.report["predebited"][key] += count; self.persist()
        self.report["invoked"][key] += count; self.report["last_invoked"] = key; self.persist()
        yield
        self.report["completed"][key] += count; self.persist()

    def finish(self):
        require(all(self.report[k] == self.limits for k in ("predebited", "invoked", "completed")), "Incomplete exact quality operation budget")


def observed_raft_forward(original, ledger):
    def forward(self, *args, **kwargs):
        require(len(args) == 2 and set(kwargs) == {"iters", "test_mode"}
                and kwargs["iters"] == 20 and kwargs["test_mode"] is True, "Original RAFT invocation differs")
        with ledger("RAFT_forwards"), ledger("RAFT_iterations", 20):
            return original(self, *args, **kwargs)
    return forward


@contextmanager
def raft_observer(module, observations, ledger, expected_paths):
    original_forward = module.RAFT.forward
    original_score, original_infer = module.DynamicDegree.get_score, module.DynamicDegree.infer
    current = [None]
    def score(self, *args, **kwargs):
        require(current[0] is not None and len(current[0]["scores"]) < 40, "Unexpected RAFT score observer")
        value = original_score(self, *args, **kwargs)
        require(finite_number(value) and value >= 0, "Nonfinite RAFT score")
        current[0]["scores"].append(float(value))
        return value
    def infer(self, path, *args, **kwargs):
        require(path in expected_paths and not any(x["video_path"] == path for x in observations), "Wrong/duplicate new RAFT video")
        row = {"video_path": path, "scores": [], "status": "INVOKED"}
        observations.append(row); current[0] = row
        try:
            value = original_infer(self, path, *args, **kwargs)
            row.update(params=dict(self.params), official_binary=bool(value), status="COMPLETE")
            validate_raft(row)
            return value
        finally:
            current[0] = None
    module.RAFT.forward = observed_raft_forward(original_forward, ledger)
    module.DynamicDegree.get_score, module.DynamicDegree.infer = score, infer
    try:
        yield
    finally:
        module.RAFT.forward = original_forward
        module.DynamicDegree.get_score, module.DynamicDegree.infer = original_score, original_infer


def motion_comparison(candidate, dense):
    c, d = validate_raft(candidate), validate_raft(dense)
    def ratio(left, right):
        return {"value": left / right, "status": "DEFINED"} if right > 0 else {"value": None, "status": "ZERO_DENSE_MOTION"}
    return {"candidate": c, "Dense": d,
            "candidate_to_Dense_mean_ratio": ratio(c["mean_top5percent_flow_px_per_2frame_pair"], d["mean_top5percent_flow_px_per_2frame_pair"]),
            "fixed_window_candidate_to_Dense_ratios": [ratio(x["mean_top5percent_flow_px"], y["mean_top5percent_flow_px"])
                                                       for x, y in zip(c["five_fixed_8pair_windows"], d["five_fixed_8pair_windows"])],
            "official_binary_separate": {"candidate": candidate["official_binary"], "Dense": dense["official_binary"]},
            "scope": "image-flow magnitude on all five fixed windows; not semantic object motion or prompt correctness"}


def summarize(plan, reused, reused_raft, completed):
    values = {(s, "dense"): dict(v) for s, v in reused.items()}
    flows = {(s, "dense"): v for s, v in reused_raft.items()}
    paths = {str((Path(plan["hf_root_for_report"]) / v["video"]["path"]).resolve()): (v["source"], v["mode"])
             for v in plan["videos"] if v["mode"] != "dense"}
    observed = set()
    require(len(completed) == 8 and {x["dimension"] for x in completed} == set(DIMENSIONS), "Eight completed dimensions required")
    for entry in completed:
        dimension = entry["dimension"]
        expected = {key for path, key in paths.items() if dimension in COMMON or key[0] == (0 if dimension == "i2v_subject" else 12)}
        rows = entry["result"][-1]
        require(len(rows) == len(expected), "Missing per-video dimension result")
        for row in rows:
            require(row["video_path"] in paths, "Result outside four new clips")
            key = paths[row["video_path"]]
            require(key in expected and (*key, dimension) not in observed, "Duplicate/wrong source dimension result")
            observed.add((*key, dimension)); values.setdefault(key, {})[dimension] = row["video_results"]
        if dimension == "dynamic_degree":
            for row in entry["RAFT_observations"]:
                require(row["video_path"] in paths and paths[row["video_path"]] not in flows, "Duplicate/wrong RAFT observation")
                flows[paths[row["video_path"]]] = row
    require(len(observed) == 28 and len(flows) == 6, "Four new / two reused coverage incomplete")
    sources = []
    for source in SOURCES:
        dense_item = next(v for v in plan["videos"] if v["source"] == source and v["mode"] == "dense")
        sources.append({"source": source, "source_id": dense_item["source_id"], "image_cluster": dense_item["image_sha256"],
                        "seed": 42, "actual_generations_per_mode": 1, "score_observations_per_mode": 1,
                        "scores": {mode: values[source, mode] for mode in MODES},
                        "new_minus_Dense": {mode: {dim: float(values[source, mode][dim]) - float(values[source, "dense"][dim])
                                                   for dim in values[source, "dense"]} for mode in MODES[1:]},
                        "motion": {mode: motion_comparison(flows[source, mode], flows[source, "dense"]) for mode in MODES[1:]}})
    require(len({x["image_cluster"] for x in sources}) == 2, "Source image clusters collapsed")
    means = {dim: {mode: statistics.fmean(values[s, mode][dim] for s in SOURCES) for mode in MODES} for dim in COMMON}
    return {"per_source": sources, "equal_source_descriptive_means_common_six_only": means,
            "independent_image_clusters": 2, "source_role": "PREVIOUSLY_EXPOSED_DEVELOPMENT",
            "seed_repetitions_per_source_mode": 1, "Dense_score_reuse_adds_independent_samples": 0,
            "source_specific_I2V_axes": "source0 subject and source12 background shown separately, each n1",
            "inference": "descriptive paired sources only; no bootstrap/significance or population/leaderboard claim",
            "quality_accepted": False}


def verify_environment(plan, hf):
    env = plan["environment"]
    helper = importlib.import_module("e0_official_metric_environment_v8")
    require(Path(helper.__file__).resolve() == ROOT / env["helper"], "Unexpected metric environment helper origin")
    v8 = read(ROOT / env["v8_protocol"])
    lock = read(ROOT / v8["lock_path"])
    require(sha(ROOT / v8["lock_path"]) == v8["lock_sha256"]
            and sha(Path(sys.executable)) == v8["installed_environment"]["python_sha256"], "Metric interpreter/package lock differs")
    helper.verify_original_sources(hf, lock)
    helper.verify_shared(lock["shared_runtime"])
    require(helper.verify_installed_environment(hf, v8, lock).resolve() == (hf / env["site"]).resolve()
            and helper.verify_overlay(hf / env["v8_directory"], v8, hf).resolve() == (hf / env["overlay"]).resolve(),
            "Original installed metric environment differs")
    helper.verify_bound_helpers(v8)
    return helper, v8


def check_result(result, expected_paths, dimension):
    from e0_ub1_consumer_official_quality import validate_results
    require(isinstance(result, (list, tuple)) and len(result) == 2, "Original evaluator result shape differs")
    rows = validate_results(result[-1], expected_paths)
    for row in rows:
        require(type(row.get("video_results")) is bool if dimension == "dynamic_degree" else finite_number(row.get("video_results")),
                "Nonfinite/wrong new score type")
    return rows


def run(args):
    precision_environment()
    require("torch" not in sys.modules and "numpy" not in sys.modules, "Fresh quality worker required before numeric imports")
    hf, protocol = args.hf_root.resolve(), args.protocol.resolve()
    plan, digest = read(protocol), sha(protocol)
    scope = authorize(plan, digest, read(ROOT / "research/ACTIVE_RESEARCH.json"))
    cpu = require_cpu_receipt(scope, plan, digest, hf)
    from worker_identity import require_worker
    require_worker()
    out = (hf / plan["output_HF_relative"]).resolve()
    require(out.is_relative_to(hf) and out != hf and not out.exists(), "Fresh confined output required; no retry/overwrite")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "ATTEMPT.json", {"id": ID, "protocol_sha256": digest, "automatic_retry": False})
    started = time.monotonic()
    report = {"id": ID, "status": "STARTED_FIXED_FOUR_NEW_TWO_REUSED", "protocol_sha256": digest,
              "prior_independent_CPU_preflight": cpu, "producer": plan["parent"],
              "legacy_score_bindings": plan["legacy"], "core_reports_reused": [v["core"] for v in plan["videos"]],
              "declared_outer_model_work_budgets": COSTS, "predebited_outer_model_work": {},
              "model_work_accounting": "Original frame/batch upper budgets; only RAFT calls/iterations observed exactly. Other internal forward/construction counts UNKNOWN.",
              "opaque_internal_model_forwards": "UNKNOWN", "opaque_model_constructions": "UNKNOWN",
              "opaque_checkpoint_loads": "UNKNOWN", "completed_dimensions": [], "RAFT_observations": [],
              "new_scored_videos": 4, "reused_Dense_videos": 2, "independent_image_clusters": 2,
              "legacy_exit_RUN_SHA_field": "NOT_PRESENT; original published RUN and launcher SHA both bound",
              "temporal_flickering": "NOT_RUN_STATIC_ELIGIBILITY_UNESTABLISHED", "camera_motion": "NOT_RUN_NOT_IN_SCOPE",
              "official_five_independent_samples": False, "human_fullplay_review": "NOT_RUN", "quality_accepted": False}
    monitor, reservation = None, None
    ledger = Ledger(report, persist=lambda: write(out / "RUN.json", report))
    def check():
        authorize(plan, digest, read(ROOT / "research/ACTIVE_RESEARCH.json"))
        if monitor:
            monitor.check()
        require(time.monotonic() - started <= RESOURCES["wall_seconds"], "Quality wall budget exceeded")
        require(shutil.disk_usage(out).free >= RESOURCES["free_disk_bytes"], "Quality free disk reserve exceeded")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink()) <= RESOURCES["new_HF_bytes"],
                "Quality output byte budget exceeded")
    def timeout(*_):
        raise TimeoutError("Fixed1800second quality attempt exhausted; no retry")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCES["wall_seconds"])
    try:
        verified = verify_assets(plan, hf)
        helper, v8 = verify_environment(plan, hf)
        helper.two_cpus()
        check()
        with ledger("reused_core_reports", 6):
            report["core_reuse_status"] = "BOUND_ALL81FRAME_REPORTS_NO_RECOMPUTATION"
        with ledger("reused_Dense_video_dimension_results", 14):
            report["reused_Dense_scores"] = verified["reused"]
            report["reused_Dense_RAFT"] = verified["raft"]
            report["verified_legacy_source_schema_mapping"] = {
                str(v["source"]): {"legacy_source_id_image_fingerprint": v["image_sha256"],
                                   "semantic_source_id": v["source_id"], "image_cluster": v["image_sha256"]}
                for v in plan["videos"] if v["mode"] == "dense"}
        env = plan["environment"]
        for name in ("tmp", "xdg", "mpl", "torch", "hf", "inductor", "debug"):
            (out / name).mkdir()
        os.environ.update(TMPDIR=str(out / "tmp"), XDG_CACHE_HOME=str(out / "xdg"), MPLCONFIGDIR=str(out / "mpl"),
                          TORCH_HOME=str(out / "torch"), HF_HOME=str(out / "hf"), TORCHINDUCTOR_CACHE_DIR=str(out / "inductor"),
                          TORCH_COMPILE_DEBUG_DIR=str(out / "debug"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                          NUMEXPR_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
        sys.dont_write_bytecode = True
        sys.path.append(str(hf / env["site"])); sys.path.insert(0, str(hf / env["overlay"]))
        helper.disable_local_ipv6_probe(hf / env["site"], v8)
        import torch
        from startup_gpu_reservation import reserve_before_assets
        from assigned_gpu_resources import AssignedGPUResources
        reservation, monitor, receipt = reserve_before_assets(torch); report["reservation"] = receipt
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        report["runtime_precision"] = {"matmul_TF32": torch.backends.cuda.matmul.allow_tf32,
                                      "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                      "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        require(report["runtime_precision"] == verified["precision"], "New/reused score precision differs")
        require(torch.cuda.device_count() == 1 and "A100" in torch.cuda.get_device_name(0), "Single assigned A100 required")
        snapshot = out / "EXECUTED_SOURCES"; snapshot.mkdir()
        for item in plan["repo_sources"]:
            target = snapshot / item["path"]; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / item["path"], target)
        shutil.copyfile(protocol, snapshot / "protocol.json")
        common = importlib.import_module("e0_ub1_consumer_official_quality")
        require(Path(common.__file__).resolve() == ROOT / "tools/e0_ub1_consumer_official_quality.py", "Unexpected frozen compatibility helper origin")
        binary = Path(shutil.which("nvidia-smi") or "").resolve()
        require(str(binary) == env["nvidia_smi"]["path"] and sha(binary) == env["nvidia_smi"]["sha256"], "GPU monitor binary differs")
        sys.addaudithook(common.audit_scope(out, AssignedGPUResources.query.__code__, binary, os.environ["FFNDS_PHYSICAL_GPU"]))
        source = (hf / env["source_root"]).resolve(); sys.path.insert(0, str(source))
        from offline_i2v_metric_loaders import submodules
        assets = read(ROOT / env["asset_manifest"]); base = hf / assets["HF_relative_root"]
        device = torch.device("cuda:0")
        oldhub = torch.hub.load
        def no_download(*_args, **_kwargs):
            raise RuntimeError("Network/implicit metric download forbidden")
        torch.hub.load = common.make_local_hub_loader(oldhub, base / "source/dino", {"facebookresearch/dino:main", str(base / "source/dino")})
        def state_load(url, *_args, **kwargs):
            require(url == "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth", "Unregistered metric checkpoint URL")
            return torch.load(base / "dino_model/dino_vitbase16_pretrain.pth", map_location=kwargs.get("map_location"), weights_only=kwargs.get("weights_only", False))
        torch.hub.load_state_dict_from_url = state_load; torch.hub.download_url_to_file = no_download
        import socket
        socket.socket.connect = no_download; socket.create_connection = no_download
        work = out / "work"; images = work / "vbench2_beta_i2v/data/crop/16-9"; images.mkdir(parents=True)
        for image in plan["images"]:
            (images / image["name"]).symlink_to(bound(hf, image["binding"]))
        os.chdir(work)
        for dimension in DIMENSIONS:
            check()
            selected = rows_for_dimension(plan["videos"], dimension)
            expected_paths = [str(bound(hf, x["video"])) for x in selected]
            require(len(selected) == (4 if dimension in COMMON else 2), "Four-new-only dimension dispatch differs")
            doc = [{"prompt_en": x["prompt"], "dimension": [dimension], "image_name": x["image_name"], "video_list": [path]}
                   for x, path in zip(selected, expected_paths)]
            inp = out / (dimension + "_INPUT.json"); write(inp, doc)
            report["active_dimension"] = dimension
            require(dimension not in report["predebited_outer_model_work"], "Duplicate dimension predebit")
            report["predebited_outer_model_work"][dimension] = COSTS[dimension].copy()
            write(out / "RUN.json", report)
            module = importlib.import_module(("vbench2_beta_i2v." if dimension.startswith("i2v_") else "vbench.") + dimension)
            require(Path(module.__file__).resolve() == source / ("vbench2_beta_i2v" if dimension.startswith("i2v_") else "vbench") / (dimension + ".py"),
                    "Unexpected original metric source origin")
            observations = report["RAFT_observations"]
            if dimension == "i2v_background":
                original_dream = module.dreamsim
                def dream(*args_, **kwargs_):
                    kwargs_.update(cache_dir=str(base / "dreamsim"), device=device)
                    return original_dream(*args_, **kwargs_)
                module.dreamsim = dream
            try:
                with ledger("official_dimension_calls"):
                    if dimension == "dynamic_degree":
                        with raft_observer(module, observations, ledger, expected_paths), torch.inference_mode():
                            result = getattr(module, "compute_" + dimension)(str(inp), device, submodules(dimension, assets, hf))
                    elif dimension == "i2v_background":
                        from peft import PeftModel, LoraConfig
                        adapter = base / "dreamsim/ensemble_lora"
                        adapter_sha = next(x["sha256"] for x in plan["hf_bindings"] if x["path"].endswith("/dreamsim/ensemble_lora/adapter_config.json"))
                        with common.dreamsim_adapter_compat(PeftModel, LoraConfig, adapter, adapter_sha), torch.inference_mode():
                            result = getattr(module, "compute_" + dimension)(str(inp), device, submodules(dimension, assets, hf))
                        report["dreamsim_adapter_compatibility"] = {"config_sha256": adapter_sha, "original_five_lora_keys_only": True,
                                                                  "inactive_extensions_verified": True, "checkpoint_bytes_changed": False}
                    else:
                        with torch.inference_mode():
                            result = getattr(module, "compute_" + dimension)(str(inp), device, submodules(dimension, assets, hf))
                torch.cuda.synchronize()
                result = common.plain(result)
                rows = check_result(result, expected_paths, dimension)
                with ledger("new_video_dimension_results", len(rows)):
                    entry = {"dimension": dimension, "result": result, "input_sha256": sha(inp)}
                    if dimension == "dynamic_degree":
                        require(len(observations) == 4 and {x["video_path"] for x in observations} == set(expected_paths), "Missing four new RAFT observations")
                        for row in observations:
                            validate_raft(row)
                        require(report["completed"]["RAFT_forwards"] == 160 and report["completed"]["RAFT_iterations"] == 3200,
                                "Actual160/3200 RAFT counts differ")
                        entry["RAFT_observations"] = copy.deepcopy(observations)
                    report["completed_dimensions"].append(entry)
                gc.collect(); torch.cuda.empty_cache()
                require(torch.cuda.max_memory_reserved() <= RESOURCES["GPU_reserved_bytes"], "Quality GPU reservation budget exceeded")
            finally:
                if dimension == "i2v_background":
                    module.dreamsim = original_dream
        ledger.finish()
        require(report["predebited_outer_model_work"] == COSTS, "Incomplete eight-dimension outer budgets")
        report["source_summary"] = summarize({**plan, "hf_root_for_report": str(hf)}, verified["reused"], verified["raft"], report["completed_dimensions"])
        report["active_dimension"] = None
        report["status"] = "COMPLETE_EIGHT_DIMENSIONS_FOUR_NEW_TWO_DENSE_REUSED_QUALITY_NOT_ACCEPTED"
        report["peak_GPU_reserved_bytes"] = torch.cuda.max_memory_reserved()
        check()
    except BaseException as error:
        report.update(status="FAILED_FIXED_QUALITY_PRESERVED_NO_RETRY", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        original_error = sys.exc_info()[1]
        signal.alarm(0)
        report["wall_seconds"] = time.monotonic() - started
        if monitor:
            try:
                monitor.close(); report["assigned_GPU_resources"] = monitor.report()
                require(not report["assigned_GPU_resources"].get("violations"), "Quality GPU monitor violation")
            except BaseException as error:
                report.update(status="FAILED_GPU_MONITOR_PRESERVED_NO_RETRY", monitor_error=str(error))
                write(out / "RUN.json", report)
                if original_error is None:
                    raise
        write(out / "RUN.json", report)
    return report


def make_draft(hf, *, repo=ROOT):
    """Metadata-only extraction: copy original weight SHA records, never load weights.

    A running/incomplete parent leaves explicit NOT_BOUND slots. Root freezes
    only after tests/review and an exact independently exited CPU preflight.
    """
    hf, repo = Path(hf).resolve(), Path(repo).resolve()
    old, parent = read(repo / LEGACY_PROTOCOL), read(repo / PARENT_PROTOCOL)
    out = hf / parent["output_HF_relative"]
    parent_run = read(out / "RUN.json") if (out / "RUN.json").exists() else {}
    complete = parent_run.get("status") == PARENT_STATUS and (out.parent / "EXIT_RECEIPT.json").exists()
    def optional(path, base, *, ready=True):
        path = Path(path)
        return binding(path, base) if ready and path.is_file() else {"path": str(path.relative_to(base)), "bytes": None, "sha256": "NOT_BOUND"}
    videos = []
    for mode in MODES:
        for case in parent["cases"]:
            source = case["index"]
            phase = next((x for x in parent_run.get("phases", []) if x.get("source_index") == source and x.get("mode") == mode), None)
            previous = next(v for v in old["videos"] if v["source"] == source and v["mode"] == "dense")
            item = {"source": source, "mode": mode, "seed": 42,
                    **{k: case[k] for k in ("source_id", "image_sha256", "prompt_sha256", "prompt")}, "image_name": previous["image_name"]}
            if phase:
                item.update(video=clip_binding(phase["video"], hf), generation=phase["generation_receipt"],
                            registered=optional(out / phase["registered_receipt"]["path"], hf),
                            core=optional(out / phase["core_metrics"]["path"], hf),
                            initial_noise=phase["initial_noise"], raw_noise=phase["raw_FP32_noise"])
            else:
                for k in ("video", "generation", "registered", "core", "initial_noise", "raw_noise"):
                    item[k] = {"path": "NOT_BOUND/" + f"source{source:02d}/{mode}/{k}", "bytes": None, "sha256": "NOT_BOUND"}
            videos.append(item)
    legacy_root = hf / old["output_HF_relative"]
    sources = sorted(REQUIRED_SOURCES)
    return {"id": ID, "status": "DRAFT_FIXED_FOUR_NEW_TWO_REUSED_NO_EXECUTION", "assets_finalized": False,
            "binding_status": "BOUND_ASSETS_REVIEW_AND_FREEZE_PENDING" if complete else "NOT_BOUND_PARENT_COMPLETION_PENDING",
            "dimensions": list(DIMENSIONS), "costs": copy.deepcopy(COSTS), "limits": copy.deepcopy(LIMITS),
            "resources": copy.deepcopy(RESOURCES), "automatic_retry": False,
            "parent": {"protocol": binding(repo / PARENT_PROTOCOL, repo),
                       "RUN": optional(out / "RUN.json", hf, ready=complete),
                       "exit": optional(out.parent / "EXIT_RECEIPT.json", hf, ready=complete),
                       "cohort_plan": optional(out / "EXECUTED_SOURCES/registered_video_plan.json", hf),
                       "verified_cohort": optional(out / "VERIFIED_GENERATIONS.json", hf, ready=complete)},
            "legacy": {"protocol": binding(repo / LEGACY_PROTOCOL, repo), "published_results": binding(repo / LEGACY_REPORT, repo),
                       "parts": [{"part": part, "RUN": binding(legacy_root / f"part{part}/RUN.json", hf),
                                  "exit": binding(legacy_root / f"part{part}/LAUNCH_EXIT.json", hf)} for part in (0, 1)]},
            "videos": videos, "images": [x for x in old["images"] if x["source"] in SOURCES],
            "environment": {k: old[k] for k in ENVIRONMENT_KEYS}, "hf_bindings": old["hf_bindings"],
            "repo_sources": [binding(repo / s, repo) for s in sources],
            "output_HF_relative": "ffnds-derived/e0_compact6016_consumer00_12_quality_20261028/attempt1/results",
            "estimand": "Two exposed development image clusters, seed42, one genuine generation/mode/source. Dense old score reuse adds zero samples. Six common dimensions source-equal descriptive; subject/background I2V each one source separately.",
            "motion_primary": "Original RAFT40 top5% pair values: mean, five preregistered consecutive8pair windows, candidate/Dense ratios; original binary separately. Not object velocity, ROI tracking or prompt correctness.",
            "core_metrics": "Bind/reuse all six actual81frame reports including paired Farneback, RGB/luma, FB/warp diagnostics; no new Farneback or CPU metric recomputation.",
            "model_work_identity": "Declared conservative frame/batch budgets from original source. Actual RAFT160 forwards/3200iterations observed; opaque internal forwards/constructions/checkpoint loads UNKNOWN.",
            "evidence_limits": {"source_role": "PREVIOUSLY_EXPOSED_DEVELOPMENT", "official_leaderboard": False,
                                "official_five_independent_samples": False, "temporal_flickering": "NOT_RUN",
                                "camera_motion": "NOT_RUN", "human_fullplay_review": "NOT_RUN", "quality_accepted": False},
            "pending_freeze": "Independent review, refreshed software SHA bindings, frozen exact protocol, ACTIVE registration and exited CUDA-hidden byte-only CPU preflight required before model/GPU calls."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("draft", "inspect", "preflight", "run"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--hf-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if str(ROOT / "tools") not in sys.path:
        sys.path.append(str(ROOT / "tools"))
    if args.command == "draft":
        require(args.hf_root is not None and args.out is not None and not args.out.exists(), "Fresh draft output required")
        plan = make_draft(args.hf_root); validate_plan(plan)
        with args.out.open("x") as stream:
            json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
        print(json.dumps({"id": ID, "binding_status": plan["binding_status"], "model_calls_executed": 0}))
    elif args.command == "inspect":
        require(args.protocol is not None, "Protocol required")
        plan = read(args.protocol); validate_plan(plan)
        print(json.dumps({"id": ID, "binding_status": plan["binding_status"], "limits": LIMITS, "model_calls_executed": 0}, indent=2))
    elif args.command == "preflight":
        require(args.protocol is not None and args.hf_root is not None and args.out is not None and not args.out.exists(), "Fresh CPU proof path required")
        report = preflight(args.protocol, args.hf_root.resolve())
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
        print(json.dumps({"status": report["status"], "model_calls": 0}))
    else:
        require(args.protocol is not None and args.hf_root is not None, "Frozen protocol/HF root required")
        result = run(args)
        print(json.dumps({"status": result["status"], "actual_completed_operations": result["completed"]}))


if __name__ == "__main__":
    main()
