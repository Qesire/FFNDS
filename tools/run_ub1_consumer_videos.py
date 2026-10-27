"""One fixed formal source: fresh dense, neutral UB1, fitted fixed-RR UB1.

The two registered sources run in separate externally assigned A100 jobs.
Every mode is a single full-depth native generation, followed by CPU core
metrics.  A neutral latent/video mismatch prevents the fitted generation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from ub1_consumer_runtime import (ConsumerResidency, UB1ConsumerRuntime, bytes_of_tensors,
    confined, drop_dense_ffn_records, expected_expert, load_original_biases,
    prepare_positions, replace_consumers, sha256, tensor_sha, verify_binding)

ROOT = Path(__file__).resolve().parents[1]
ID = "UB1_FIXED_RR_CONSUMER_VIDEOS_20261027"
MODES = ("dense", "neutral", "candidate")
PER_CASE_LIMITS = {"video_generations": 3, "native_full_model_flows": 12,
                   "native_main_FFN_linears": 960, "native_RR_linears": 640,
                   "native_FFN_activation_QDQ": 960, "native_non_FFN_linears": 3840,
                   "native_non_FFN_activation_QDQ": 3840, "core_video_reports": 3,
                   "new_fits": 0, "optimizer_updates": 0, "official_quality_calls": 0}
VERSIONS = ("torch", "torchvision", "transformers", "diffusers", "safetensors", "numpy", "triton")
REQUIRED_SOURCES = {
    "tools/run_ub1_consumer_videos.py", "tools/ub1_consumer_runtime.py",
    "tests/test_ub1_consumer_runtime.py", "tools/e0_ub1_native_consistency.py",
    "tools/step_budget_runtime.py", "tools/trained_packed_runtime.py", "tools/reduced_rank_residual.py",
    "tools/ffnds_all_weight_residency.py", "tools/run_e0_all_weight_residency.py",
    "tools/restored_assets.py", "tools/historical_source_snapshot.py", "tools/worker_identity.py",
    "tools/startup_gpu_reservation.py", "tools/assigned_gpu_resources.py", "tools/ffnds_video_metrics.py",
    "tools/generate_formal_suite.py", "reports/worker_20261020/UB1_policy.json",
    ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py",
    ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json",
    ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py",
    ".runtime/code/LightX2V/lightx2v/common/ops/mm/mm_weight.py",
    ".runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py",
    ".runtime/code/LightX2V/lightx2v/models/schedulers/wan/scheduler.py",
    ".runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py",
}


def object_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    os.replace(temp, path)


def bound_path(binding, roots):
    root = binding.get("root", "repo")
    if root not in roots:
        raise ValueError("Unknown binding root")
    return verify_binding(roots[root], binding)


def validate_plan(plan):
    if (plan.get("id") != ID or plan.get("assets_finalized") is not True or
            plan.get("modes") != list(MODES) or plan.get("limits_per_case") != PER_CASE_LIMITS or
            plan.get("max_full_generations_total") != 6 or plan.get("max_gpu_cards") != 4 or
            plan.get("planned_simultaneous_video_GPU_cards") not in (1, 2) or
            plan.get("automatic_retry") is not False or plan.get("official_quality_calls") != 0 or
            plan.get("neutral_requires_exact_latents_and_video") is not True):
        raise ValueError("Unfrozen or changed two-source/six-video native scope")
    cases = plan.get("cases", [])
    if (len(cases) != 2 or [c["index"] for c in cases] != [0, 10] or
            [c["id"] for c in cases] != ["input00_seed42", "input10_seed42"] or
            any(c["seed"] != 42 or c["exposure"] != "ALREADY_DEVELOPMENT_EXPOSED" for c in cases)):
        raise ValueError("Only the two fixed development-exposed formal examples are registered")
    if (plan.get("video_contract") != {"frames": 81, "width": 832, "height": 464, "fps": 16,
            "steps": 4, "blocks": 40, "denoising_steps": [1000, 750, 500, 250],
            "latent_shape": [16, 21, 58, 104]}):
        raise ValueError("Native full-depth/video contract differs")
    resources = plan["resource_limits"]
    if (resources["per_case_HF_max_bytes"] > 512 << 20 or resources["total_HF_max_bytes"] > 1 << 30 or
            resources["per_case_HF_max_bytes"] * 2 > resources["total_HF_max_bytes"] or
            not 0 < resources["compute_wall_seconds"] <= 14400 or
            not 0 < resources["resident_tensor_max_bytes"] <= 64 << 30 or
            resources["cpu_threads"] not in range(1, 9)):
        raise ValueError("Native resource budget differs")
    return cases


def validate_authorization(active, protocol_sha):
    scopes = active["experiment_authority"]["current_user_authorized_scopes"]
    return any(s.get("id") == ID and s.get("protocol_sha256") == protocol_sha and
               s.get("execution_status") in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE") for s in scopes)


def verify_coefficient_chains(bank_plan, private_root, recipe):
    """Read-only validation; no CheckpointStore constructor or legacy runtime."""
    assets = {a["path"]: a for a in bank_plan["private_assets"]}
    if len(assets) != len(bank_plan["private_assets"]):
        raise ValueError("Duplicate private coefficient binding")
    keys = ("main_npz", "main_json", "main_identity", "main_report",
            "rr_npz", "rr_json", "rr_identity", "rr_report")
    needed = {p[k] for p in bank_plan["positions"] for k in keys}
    needed.update(bank_plan.get("private_common", []))
    if not needed <= set(assets):
        raise ValueError("Original coefficient source-chain binding missing")
    files = {p: verify_binding(private_root, assets[p]) for p in sorted(needed)}
    cache_sha = recipe["cache_manifest"]["sha256"]
    parents, base_fingerprints, residual_base = set(), set(), set()
    reports_checked = {}
    for p in bank_plan["positions"]:
        for prefix in ("main", "rr"):
            identity_path, report_path = p[prefix + "_identity"], p[prefix + "_report"]
            identity = json.loads(files[identity_path].read_text())
            source_report = json.loads(files[report_path].read_text())
            sidecar = json.loads(files[p[prefix + "_json"]].read_text())
            fp = p[prefix + "_fingerprint"]
            if (identity.get("fingerprint") != fp or object_sha(identity["identity"]) != fp or
                    source_report.get("fingerprint") != fp or sidecar.get("fingerprint") != fp or
                    sidecar.get("status") != "COMPLETE" or
                    sidecar.get("payload_sha256") != assets[p[prefix + "_npz"]]["sha256"]):
                raise ValueError("Original coefficient report/identity/sidecar/payload chain differs")
            native_id = identity["identity"]
            if native_id.get("cache_manifest_sha256") != cache_sha:
                raise ValueError("Original coefficients use a different native cache identity")
            status = source_report.get("status")
            if p["step"] in (0, 3):
                if (status != "COMPLETED_UB1_STEP40_FIT" or source_report.get("step") != p["step"] or
                        source_report.get("groups") != (54, 108, 108, 162)[p["step"]] or
                        len(source_report.get("positions", [])) != 40 or
                        native_id.get("parent_policy_sha256") != recipe["parent_policy"]["sha256"] or
                        native_id.get("protocol_sha256") != recipe["protocol"]["sha256"]):
                    raise ValueError("Original changed-step parent/protocol identity differs")
                changed = recipe["changed_fits"][str(p["step"])]
                if fp != changed["fingerprint"] or sha256(files[report_path]) != changed["sha256"]:
                    raise ValueError("Original UB1 changed-fit recipe binding differs")
                parents.add(native_id["parent_policy_sha256"])
            elif prefix == "main":
                if status != "COMPLETED_CALIBRATION_PARAMETER_FIT" or source_report.get("expert") != p["expert"]:
                    raise ValueError("Original centered main-fit identity differs")
                base_fingerprints.add(fp)
            else:
                if status != "COMPLETED_RR1_FULL160_CALIBRATION_FIT" or len(source_report.get("positions", [])) != 160:
                    raise ValueError("Original full160 RR identity differs")
                residual_base.update(x["fingerprint"] for x in native_id["base_fits"])
            reports_checked[report_path] = {"sha256": sha256(files[report_path]), "fingerprint": fp, "status": status}
    if parents != {recipe["parent_policy"]["sha256"]} or len(base_fingerprints) != 2 or residual_base != base_fingerprints:
        raise ValueError("Original main/RR fitted-base chain differs")
    parent_files = [files[name] for name in needed if assets[name]["sha256"] == recipe["parent_policy"]["sha256"]]
    if not parent_files:
        raise ValueError("Original immutable C24 parent policy bytes missing")
    parent = json.loads(parent_files[0].read_text())
    if (parent.get("support") != "C" or parent.get("mode") != "centered_gain_bias" or
            parent.get("cache_manifest") != recipe["cache_manifest"] or
            {v["fingerprint"] for v in parent["fit_reports"].values()} != base_fingerprints):
        raise ValueError("Original C24 parent fit/cache/support binding differs")
    for expert, binding in parent["fit_reports"].items():
        if not any(r["sha256"] == binding["sha256"] and r["fingerprint"] == binding["fingerprint"] for r in reports_checked.values()):
            raise ValueError("Original parent centered-fit report bytes missing")
    if not any(r["sha256"] == parent["residual_fit"]["sha256"] for r in reports_checked.values()):
        raise ValueError("Original parent RR report bytes missing")
    return {"reports": list(reports_checked.values()), "original_parent_policy_sha256": recipe["parent_policy"]["sha256"],
            "recorded_historical_protocol_sha256": recipe["protocol"]["sha256"],
            "public_protocol_hash_is_not_substituted_for_historical_identity": True}


def compare_historical(trajectory, video_sha, reference):
    actual = [t["raw_tensor_sha256"] for t in trajectory]
    expected = reference["latent_sha256"]
    if len(actual) != 4 or len(expected) != 4:
        raise ValueError("Historical comparison requires four native latents")
    matched = [a == b for a, b in zip(actual, expected)]
    video_match = video_sha == reference["video_sha256"]
    return {"four_raw_FP32_latents_match": all(matched), "latent_match_by_step": matched,
            "video_bytes_match": video_match, "passed": all(matched) and video_match,
            "actual_latent_sha256": actual, "historical_latent_sha256": expected,
            "actual_video_sha256": video_sha, "historical_video_sha256": reference["video_sha256"]}


def verify_historical_case(case, roots, sample):
    for arm in ("dense", "ub1"):
        reference = case["historical"][arm]
        record = json.loads(bound_path(reference["input"], roots).read_text())
        original = record["input"]
        if (record["status"] != "PASS_FORMAL_INPUT_GENERATION_ONLY" or record["index"] != case["index"] or
                original["seed"] != 42 or original["source_group"] != sample["source_group"] or
                original["prompt"] != sample["prompt"] or
                record["video"]["sha256"] != reference["video_sha256"] or
                [x["raw_tensor_sha256"] for x in record["trajectory"]] != reference["latent_sha256"]):
            raise ValueError("Historical formal input/latent/video record differs")
        if (original["input_provenance"]["image_sha256"] != case["source_id"] or
                original["input_provenance"].get("historical_image_sha256") is not None):
            raise ValueError("Historical recovered-image provenance differs")


def preflight(protocol, hf_root, private_root, *, root=ROOT, active=None):
    plan = json.loads(Path(protocol).read_text())
    cases = validate_plan(plan)
    roots = {"repo": Path(root).resolve(), "hf": Path(hf_root).resolve(), "private_hf": Path(private_root).resolve()}
    bound = {p["path"] for p in plan["repo_sources"]}
    if (not REQUIRED_SOURCES <= bound or
            not any("hifxg_quant" in p and p.endswith(".so") for p in bound)):
        raise ValueError("Required native source or QDQ extension binding missing")
    for binding in plan["repo_sources"]:
        verify_binding(roots["repo"], binding)
    for name in ("model_view", "local_model_view", "cache_manifest", "input_manifest", "input_recovery", "original_recipe", "bank_protocol"):
        bound_path(plan[name], roots)
    logical = json.loads(bound_path(plan["model_view"], roots).read_text())
    if set(plan["original_checkpoint_files"]) != {"high_noise_model", "low_noise_model"}:
        raise ValueError("Both original expert checkpoint identities required")
    for expert, binding in plan["original_checkpoint_files"].items():
        original = [r for r in logical["files"] if r["filename"] == binding["filename"]]
        if (len(original) != 1 or any(original[0][k] != binding[k] for k in ("bytes", "sha256")) or
                not binding["filename"].endswith(".safetensors") or
                expert.split("_noise_model")[0] + "_noise" not in binding["filename"]):
            raise ValueError("Original expert checkpoint SHA/model-view identity differs")
    samples = json.loads(bound_path(plan["input_manifest"], roots).read_text())
    recovered = json.loads(bound_path(plan["input_recovery"], roots).read_text())
    if (len(samples) != 20 or recovered["status"] != "PASS_NEW_FORMAL20_INPUT_RECOVERY" or
            recovered["canonical_sha256"] != plan["input_manifest"]["sha256"]):
        raise ValueError("Recovered canonical formal20 input manifest differs")
    for case in cases:
        sample = samples[case["index"]]
        image = bound_path(case["image"], roots)
        if (sample["seed"] != 42 or sample["source_group"] != case["source_group"] or
                sample["input_provenance"]["image_sha256"] != case["source_id"] or
                case["image"]["sha256"] != case["source_id"] or
                hashlib.sha256(sample["prompt"].encode()).hexdigest() != case["prompt_sha256"] or
                sample["prompt_sha256"] != case["prompt_sha256"]):
            raise ValueError("Actual current formal image/prompt/seed differs")
        verify_historical_case(case, roots, sample)
    recipe = json.loads(bound_path(plan["original_recipe"], roots).read_text())
    if recipe["groups_by_step"] != [54, 108, 108, 162] or recipe["identity_control"] is not False or recipe["cache_manifest"]["sha256"] != plan["cache_manifest"]["sha256"]:
        raise ValueError("Original UB1 recipe/cache object differs")
    bank_plan = json.loads(bound_path(plan["bank_protocol"], roots).read_text())
    if bank_plan.get("id") != "UB1_FIXED_RR_CONSUMER_BANK_20261027" or bank_plan.get("assets_finalized") is not True:
        raise ValueError("Consumer training protocol not frozen")
    chain = verify_coefficient_chains(bank_plan, roots["private_hf"], recipe)
    manifest = json.loads(bound_path(plan["bank_manifest"], roots).read_text())
    if (manifest.get("status") != "COMPLETE_UB1_FIXED_RR_CONSUMER_BANK" or
            manifest.get("input_count") != 24 or manifest.get("bank_protocol") != {
                "path": plan["bank_protocol"]["path"], "sha256": plan["bank_protocol"]["sha256"]} or
            len(manifest.get("packs", [])) != 160 or
            {(p["step"], p["block"]) for p in manifest["packs"]} != {(s, b) for s in range(4) for b in range(40)}):
        raise ValueError("Complete frozen 160-consumer bank manifest required")
    parts = manifest.get("part_RUNs", [])
    if len(parts) != 4 or {p["part"] for p in parts} != set(range(4)):
        raise ValueError("All four original bank execution receipts required")
    for binding in parts:
        receipt = json.loads(verify_binding(roots["hf"], binding).read_text())
        if (receipt.get("status") != "COMPLETE_TRAINING_CONSUMER_BANK_PART" or receipt.get("part") != binding["part"] or
                receipt.get("protocol_sha256") != plan["bank_protocol"]["sha256"] or len(receipt.get("sites", [])) != 40):
            raise ValueError("Original bank part execution identity differs")
    for item in manifest["packs"]:
        if item["expert"] != expected_expert(item["step"]):
            raise ValueError("Consumer expert/step differs")
        verify_binding(roots["hf"], item)
    versions = {n: importlib.metadata.version(n) for n in VERSIONS}
    if versions != plan["runtime_versions"]:
        raise ValueError("Frozen historical software versions differ")
    if active is None:
        active = json.loads((Path(root) / "research/ACTIVE_RESEARCH.json").read_text())
    authorized = validate_authorization(active, sha256(protocol))
    return plan, samples, bank_plan, manifest, roots, {"status": "STATIC_NATIVE_CONSUMER_PREFLIGHT_PASS",
        "authorized": authorized, "GPU_initialized": False, "new_generations": 0,
        "versions": versions, "coefficient_chain": chain}


class ExecutionLedger:
    """One canonical attempt per source; a failed neutral cannot advance."""
    def __init__(self, path, case, protocol_sha):
        self.path, self.case = Path(path), case
        self.record = {"id": ID, "status": "STARTED_NOT_ACCEPTED", "case": case,
            "protocol_sha256": protocol_sha, "attempt": 1, "automatic_retry": False,
            "counts": dict.fromkeys(PER_CASE_LIMITS, 0), "phases": [],
            "count_semantics": "generation upper bounds debited before dispatch; actual native observations and completed flow steps recorded per phase",
            "quality_or_speed_acceptance": False, "official_quality_status": "NOT_RUN_SEPARATE_SCOPE_REQUIRED",
            "human_fullplay_review": "PENDING"}

    def start(self):
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / "ATTEMPT.json").open("x") as stream:
            json.dump({"id": ID, "case": self.case["id"], "attempt": 1,
                       "protocol_sha256": self.record["protocol_sha256"]}, stream)
        self.save()

    def save(self):
        write_json(self.path / "RUN.json", self.record)

    def charge(self, key, count=1):
        target = self.record["counts"][key] + count
        if target > PER_CASE_LIMITS[key]:
            raise RuntimeError("Native call budget exceeded before " + key)
        self.record["counts"][key] = target
        self.save()

    def reserve_generation(self, mode):
        index = self.record["counts"]["video_generations"]
        if index >= 3 or mode != MODES[index] or len(self.record["phases"]) != index:
            raise RuntimeError("Generation order/attempt changed")
        if mode == "candidate" and not self.record["phases"][1].get("historical_comparison", {}).get("passed"):
            raise RuntimeError("Neutral exact latent/video reproduction must pass before candidate")
        costs = {"video_generations": 1, "native_full_model_flows": 4,
                 "native_main_FFN_linears": 320, "native_RR_linears": 0 if mode == "dense" else 320,
                 "native_FFN_activation_QDQ": 320, "native_non_FFN_linears": 1280,
                 "native_non_FFN_activation_QDQ": 1280}
        if any(self.record["counts"][key] + amount > PER_CASE_LIMITS[key] for key, amount in costs.items()):
            raise RuntimeError("Native generation call budget exceeded before dispatch")
        for key, amount in costs.items():
            self.record["counts"][key] += amount
        self.save()

    def complete_phase(self, phase):
        if phase["mode"] != MODES[len(self.record["phases"])]:
            raise RuntimeError("Native phase ledger order differs")
        self.record["phases"].append(phase)
        self.save()


def validate_native_counts(counts, timings, consumer=None):
    if (counts.get("linear_forward_calls") != 1600 or counts.get("activation_qdq_calls") != 1600 or
            counts.get("cache_misses") != 0 or counts.get("weight_transfers") != 0):
        raise RuntimeError("Native 320 FFN + 1280 non-FFN resident call ledger differs")
    if len(timings) != 160 or {(t["step"], t["block"]) for t in timings} != {(s, b) for s in range(4) for b in range(40)}:
        raise RuntimeError("Full 40-block/four-step FFN timing boundary differs")
    if consumer is not None:
        expected = {"main_linear": 320, "RR_linear": 320, "Q1": 160, "Q2": 160, "up": 160, "down": 160}
        if consumer["counts"] != expected or consumer["pending_RR"] or len(consumer["main_coverage"]) != 320 or len(consumer["RR_coverage"]) != 160:
            raise RuntimeError("Native main/RR/QDQ coverage differs")


def inspect_video(path, contract):
    import imageio.v2 as imageio
    with imageio.get_reader(path) as reader:
        count, info = reader.count_frames(), reader.get_meta_data()
    if (count != contract["frames"] or abs(float(info["fps"]) - contract["fps"]) > .01 or
            list(info["size"]) != [contract["width"], contract["height"]]):
        raise RuntimeError("Full video frame/fps/resolution contract differs")
    return {"path": str(path), "sha256": sha256(path), "frames": count,
            "fps": float(info["fps"]), "size": list(info["size"]), "bytes": Path(path).stat().st_size}


def generate_phase(pipe, base, runtime, mode, case, sample, image, ledger, timer, plan,
                   monitor, check_resources, setup_seconds):
    import torch
    from safetensors.torch import save_file
    contract = plan["video_contract"]
    ledger.reserve_generation(mode)
    parent = ledger.path / mode
    parent.mkdir(exist_ok=False)
    phase = {"status": "STARTED_NOT_ACCEPTED", "mode": mode, "trajectory": [],
             "source_id": case["source_id"], "seed": 42, "prompt_sha256": case["prompt_sha256"],
             "setup_seconds": setup_seconds, "scheduler_generator_reset": True,
             "quality_or_speed_acceptance": False}
    write_json(parent / "PHASE.json", phase)
    scheduler = pipe.runner.scheduler
    original_post = scheduler.step_post
    before = dict(base.counters)
    if runtime is not base:
        runtime.begin_request()
    timer.events.clear()
    scheduler.generator = None
    torch.cuda.reset_peak_memory_stats()

    def post():
        result = original_post()
        # This observation is common to all modes, paid inside generation wall.
        latent = scheduler.latents.detach().float().cpu().contiguous()
        step = int(scheduler.step_index)
        if (step != len(phase["trajectory"]) or step not in range(4) or
                list(latent.shape) != contract["latent_shape"] or not bool(torch.isfinite(latent).all())):
            raise RuntimeError("Native scheduler latent shape/order/finite contract differs")
        denoising_step = int(scheduler.denoising_step_list[step])
        if denoising_step != contract["denoising_steps"][step]:
            raise RuntimeError("Native scheduler denoising step differs")
        check_resources(projected_bytes=latent.numel() * latent.element_size() + (1 << 20))
        path = parent / f"latent_step{step}.safetensors"
        if path.exists():
            raise FileExistsError("Refusing latent replacement")
        save_file({"latents": latent}, str(path))
        phase["trajectory"].append({"step": step, "denoising_step": denoising_step,
            "sigma": float(scheduler.sigmas[step]), "shape": list(latent.shape), "dtype": "float32",
            "raw_tensor_sha256": tensor_sha(latent), "file_sha256": sha256(path), "path": path.name,
            "bytes": path.stat().st_size})
        write_json(parent / "TRAJECTORY.json", phase["trajectory"])
        check_resources()
        return result

    video = parent / "output.mp4"
    scheduler.step_post = post
    started = time.perf_counter()
    try:
        check_resources()
        with torch.inference_mode():
            pipe.generate(seed=42, image_path=str(image), prompt=sample["prompt"], negative_prompt="", save_result_path=str(video))
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        monitor.check()
        consumer = runtime.end_request() if runtime is not base else None
        counts = {k: base.counters[k] - before[k] for k in before}
        timings = timer.report()
        validate_native_counts(counts, timings, consumer)
        if [t["step"] for t in phase["trajectory"]] != [0, 1, 2, 3]:
            raise RuntimeError("Native four-step trajectory incomplete")
        phase.update(status="PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE",
            generation_wall_seconds_including_video_save_latent_save_and_hash=wall,
            FFN_boundary_seconds=sum(t["elapsed_ms"] for t in timings) / 1000,
            FFN_boundary_timings=timings, native_counter_delta=counts,
            completed_native_flow_steps=len(phase["trajectory"]),
            consumer_runtime=consumer, video=inspect_video(video, contract),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            allocated_bytes_after_generation=torch.cuda.memory_allocated())
        if mode in ("dense", "neutral"):
            phase["historical_comparison"] = compare_historical(phase["trajectory"], phase["video"]["sha256"],
                case["historical"]["dense" if mode == "dense" else "ub1"])
        write_json(parent / "PHASE.json", phase)
        if mode == "neutral" and not phase["historical_comparison"]["passed"]:
            raise RuntimeError("Neutral UB1 four-latent/video reproduction failed; candidate is blocked")
        return phase
    except BaseException as exc:
        phase.update(status="FAILED_PRESERVED_NO_RETRY", error_type=type(exc).__name__, error=str(exc),
                     completed_native_flow_steps=len(phase["trajectory"]),
                     observed_native_counter_delta={k: base.counters[k] - before[k] for k in before})
        if runtime is not base:
            phase["failed_consumer_runtime"] = runtime.report()
        write_json(parent / "PHASE.json", phase)
        raise
    finally:
        scheduler.step_post = original_post


def attach_core_metrics(phase, ledger, assets, check_resources):
    from ffnds_video_metrics import evaluate, validate
    parent = ledger.path / phase["mode"]
    # Stable generation receipt is written before the CPU report references it.
    receipt = parent / "GENERATION.json"
    with receipt.open("x") as stream:
        json.dump(phase, stream, ensure_ascii=False, indent=2, allow_nan=False)
    identity = {"source_id": ledger.case["source_id"], "seed": 42, "expected_frames": 81,
                "input_prompt_digest": ledger.case["prompt_sha256"], "model_digest": object_sha(assets),
                "protocol_sha256": ledger.record["protocol_sha256"], "run_sha256": sha256(receipt)}
    reference = None
    inputs = {"identity": identity, "cost": {
        "generation_wall_seconds": phase["generation_wall_seconds_including_video_save_latent_save_and_hash"],
        "ffn_boundary_seconds": phase["FFN_boundary_seconds"], "setup_seconds": phase["setup_seconds"],
        "peak_gpu_memory_mib": phase["peak_cuda_allocated_bytes"] / (1 << 20)}}
    if phase["mode"] != "dense":
        dense = ledger.record["phases"][0]
        reference = Path(dense["video"]["path"])
        inputs["reference_identity"] = {"source_id": identity["source_id"], "seed": 42,
            "input_prompt_digest": identity["input_prompt_digest"],
            "run_sha256": sha256(ledger.path / "dense/GENERATION.json")}
    ledger.charge("core_video_reports")
    started = time.perf_counter()
    report = evaluate(Path(phase["video"]["path"]), reference, inputs)
    errors = validate(report)
    write_json(parent / "CORE_METRICS.json", report)
    if errors:
        raise RuntimeError("All-frame core metric report failed: " + "; ".join(errors))
    phase["core_metrics"] = {"path": str(parent / "CORE_METRICS.json"),
                             "sha256": sha256(parent / "CORE_METRICS.json"),
                             "CPU_seconds": time.perf_counter() - started,
                             "status": "ALL_81_FRAMES_CORE_METRICS_PASS_QUALITY_PENDING"}
    write_json(parent / "PHASE.json", phase)
    check_resources()


def execute(args, prepared):
    plan, samples, bank_plan, manifest, roots, readiness = prepared
    if not args.execute_native or not readiness["authorized"]:
        raise RuntimeError("Exact protocol SHA must be registered before native execution")
    case = next(c for c in plan["cases"] if c["id"] == args.case)
    from worker_identity import private_config_path, require_worker
    require_worker()
    allowed_workers = plan["private_worker_config_sha256"]
    if sha256(private_config_path()) not in allowed_workers:
        raise ValueError("Executor private profile is not frozen in this scope")
    out = confined(roots["hf"], plan["output_HF_relative"]) / case["id"]
    ledger = ExecutionLedger(out, case, sha256(args.protocol))
    ledger.start()
    limits = plan["resource_limits"]
    monitor = None
    base = runtime = residency = adapter = None
    started_total = time.monotonic()
    old_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError("Registered native video compute wall budget reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(limits["compute_wall_seconds"])

    def check_resources(projected_bytes=0):
        if monitor is not None:
            monitor.check()
        used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink())
        if used + projected_bytes > limits["per_case_HF_max_bytes"]:
            raise RuntimeError("Registered video output budget reached")
        if shutil.disk_usage(out).free - projected_bytes < limits["HF_free_min_bytes"]:
            raise RuntimeError("Registered HF free-space reserve reached")
        available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024
        if available < limits["host_available_min_bytes"]:
            raise RuntimeError("Registered host available-memory reserve reached")

    try:
        check_resources()
        snapshot = out / "EXECUTED_SOURCES"
        snapshot.mkdir()
        for binding in plan["repo_sources"]:
            target = snapshot / binding["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            check_resources(projected_bytes=(ROOT / binding["path"]).stat().st_size)
            shutil.copyfile(ROOT / binding["path"], target)
        shutil.copyfile(args.protocol, snapshot / "protocol.json")
        shutil.copyfile(bound_path(plan["bank_protocol"], roots), snapshot / "bank_protocol.json")
        shutil.copyfile(bound_path(plan["bank_manifest"], roots), snapshot / "BANK_MANIFEST.json")
        for name in ("HiF4_Sp", "LightX2V", "Wan2.2-I2V-A14B-W4A4"):
            sys.path.insert(0, str(ROOT / ".runtime/code" / name))
        for name in ("HiFloat4", "HiFloat4/hif4_gpu"):
            sys.path.insert(0, str(ROOT / ".runtime/code/HiF4_Sp" / name))
        import torch
        from ffnds_all_weight_residency import AllWeightResidency
        from historical_source_snapshot import HistoricalSources
        from restored_assets import verified_restored_assets
        from run_e0_all_weight_residency import FFNStageTimer
        from startup_gpu_reservation import reserve_before_assets
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        extension = importlib.import_module("HiFloat4.hif4_gpu.quant_cy.base.cusrc.hifxg_quant")
        bound_extensions = {(ROOT / p["path"]).resolve() for p in plan["repo_sources"]
                            if p["path"].endswith(".so") and "hifxg_quant" in p["path"]}
        if Path(extension.__file__).resolve() not in bound_extensions:
            raise ValueError("Actual loaded native QDQ extension is not frozen")
        reservation, monitor, receipt = reserve_before_assets(torch)
        ledger.record["CUDA_reservation"] = receipt
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("Exactly one externally assigned A100 required")
        if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
            raise ValueError("TF32 override conflicts with the frozen native arithmetic")
        torch.set_num_threads(limits["cpu_threads"])
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        ledger.record["environment"] = {"versions": readiness["versions"], "CUDA": torch.version.cuda,
            "GPU": torch.cuda.get_device_name(0), "TF32": False,
            "BF16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE")}
        setup_started = time.perf_counter()
        history = HistoricalSources()
        kwargs, cache, assets = verified_restored_assets(history)
        ledger.record["assets"] = assets
        ledger.record["historical_source_verifications"] = history.receipts
        ledger.record["coefficient_chain"] = readiness["coefficient_chain"]
        # Bind checkpoint keys/bytes once in setup; no online bias hashing.
        native_biases = load_original_biases({"high_noise_model": kwargs["high_noise_ckpt"],
                                             "low_noise_model": kwargs["low_noise_ckpt"]})
        ledger.record["original_checkpoint_biases"] = {f"{e}_block{b:02d}": {
            name: value for name, value in values.items() if not torch.is_tensor(value)}
            for (e, b), values in native_biases.items()}
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        base = adapter.HIF4CacheRuntime(cache["path"])
        adapter.activate_lightx2v_hooks(runtime=base)
        image = bound_path(case["image"], roots)
        sample = samples[case["index"]]
        original_bytes = sum(bytes_of_tensors(r["weight_qdq"]) for records in base.records.values() for r in records.values() if torch.is_tensor(r.get("weight_qdq")))
        total_memory = torch.cuda.get_device_properties(0).total_memory
        dense_estimate = original_bytes + limits["reference_original_peak_physical_MiB"] * (1 << 20) + limits["memory_safety_bytes"]
        if original_bytes > limits["resident_tensor_max_bytes"] or dense_estimate > total_memory:
            raise RuntimeError("Dense all-weight residency exceeds frozen A100 headroom")
        shared_setup_seconds = time.perf_counter() - setup_started
        ledger.record["memory_precheck"] = {"original_decoded_weight_bytes": original_bytes,
            "dense_estimated_peak_bytes": dense_estimate, "device_total_bytes": total_memory}
        ledger.save()
        with torch.inference_mode(), FFNStageTimer(adapter) as timer:
            with AllWeightResidency(base, "all_resident_no_flush", limits["resident_tensor_max_bytes"]) as dense_residency:
                preload_started = time.perf_counter()
                for expert, records in base.records.items():
                    for name, record in records.items():
                        if not torch.is_tensor(record.get("weight_qdq")):
                            continue
                        base._weight(expert=expert, block_index=int(name.split(".")[1]), module_name=name,
                                     record=record, device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.bfloat16)
                torch.cuda.synchronize()
                preload_seconds = time.perf_counter() - preload_started
                ledger.record["dense_weight_preload_seconds"] = preload_seconds
                ledger.record["dense_residency_before_generation"] = dense_residency.snapshot()
                phase = generate_phase(pipe, base, base, "dense", case, sample, image, ledger, timer, plan,
                    monitor, check_resources, shared_setup_seconds + preload_seconds)
                attach_core_metrics(phase, ledger, assets, check_resources)
                ledger.complete_phase(phase)
                torch.cuda.synchronize()
            # AllWeightResidency has dropped every decoded dense GPU reference.
            base._resident_weights.clear()
            base._resident_key = None
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ledger.record["dense_residency_after_release"] = dense_residency.snapshot()
            ledger.record["CUDA_allocated_after_dense_release"] = torch.cuda.memory_allocated()
            pack_setup_started = time.perf_counter()
            positions = prepare_positions(base, bank_plan, roots["private_hf"], native_biases)
            ledger.record["released_original_dense_FFN_CPU_bytes"] = drop_dense_ffn_records(base)
            del native_biases
            gc.collect()
            non_ffn_bytes = sum(bytes_of_tensors(r["weight_qdq"]) for records in base.records.values() for r in records.values() if torch.is_tensor(r.get("weight_qdq")))
            packed_bytes = sum(sum(bytes_of_tensors(p[k]) for k in ("w1", "b1", "s1", "s2", "channels", "rr", "w2", "bias")) for p in positions.values())
            packed_estimate = non_ffn_bytes + packed_bytes + limits["reference_original_peak_physical_MiB"] * (1 << 20) + limits["memory_safety_bytes"]
            if non_ffn_bytes + packed_bytes > limits["resident_tensor_max_bytes"] or packed_estimate > total_memory:
                raise RuntimeError("Full160 original-RR packed residency exceeds A100 headroom")
            ledger.record["memory_precheck"].update(non_FFN_weight_bytes=non_ffn_bytes,
                packed_all160_tensor_bytes=packed_bytes, packed_estimated_peak_bytes=packed_estimate)
            device = torch.device("cuda", torch.cuda.current_device())
            with ConsumerResidency(base, positions, device, limits["resident_tensor_max_bytes"]) as residency:
                packed_preload_seconds = residency.preload()
                neutral_setup_seconds = time.perf_counter() - pack_setup_started
                ledger.record["neutral_pack_prepare_and_preload_seconds"] = neutral_setup_seconds
                ledger.record["neutral_packed_preload_seconds"] = packed_preload_seconds
                runtime = UB1ConsumerRuntime(base, positions, residency, adapter.native_hif4_activation_qdq, mode="neutral")
                adapter.activate_lightx2v_hooks(runtime=runtime)
                phase = generate_phase(pipe, base, runtime, "neutral", case, sample, image, ledger, timer, plan,
                    monitor, check_resources, shared_setup_seconds + neutral_setup_seconds)
                attach_core_metrics(phase, ledger, assets, check_resources)
                ledger.complete_phase(phase)
                # The new fitted consumer is loaded only after the strict replay.
                candidate_setup_started = time.perf_counter()
                residency.unload_consumers()
                ledger.record["consumer_residency_after_neutral_W2_release"] = residency.snapshot()
                torch.cuda.empty_cache()
                replace_consumers(positions, manifest, roots["hf"])
                residency.load_consumers()
                candidate_setup_seconds = time.perf_counter() - candidate_setup_started
                ledger.record["candidate_consumer_replace_and_preload_seconds"] = candidate_setup_seconds
                runtime = UB1ConsumerRuntime(base, positions, residency, adapter.native_hif4_activation_qdq, mode="candidate")
                adapter.activate_lightx2v_hooks(runtime=runtime)
                phase = generate_phase(pipe, base, runtime, "candidate", case, sample, image, ledger, timer, plan,
                    monitor, check_resources, shared_setup_seconds + neutral_setup_seconds + candidate_setup_seconds)
                attach_core_metrics(phase, ledger, assets, check_resources)
                ledger.complete_phase(phase)
                ledger.record["packed_residency_end"] = residency.snapshot()
        if ledger.record["counts"] != PER_CASE_LIMITS or len(ledger.record["phases"]) != 3:
            raise RuntimeError("Exact fixed native/source ledger incomplete")
        check_resources()
        ledger.record["status"] = "PASS_FIXED_FULL_DEPTH_CONSUMER_EXECUTION_CORE_METRICS_QUALITY_PENDING"
    except BaseException as exc:
        ledger.record.update(status="FAILED_PRESERVED_NO_RETRY", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        if adapter is not None:
            adapter.deactivate_lightx2v_hooks()
        if monitor is not None:
            monitor.close()
            ledger.record["assigned_GPU_resources"] = monitor.report()
            if ledger.record["assigned_GPU_resources"].get("violations"):
                ledger.record["status"] = "FAILED_GPU_RESOURCE_MONITOR_PRESERVED"
        ledger.record["elapsed_seconds_including_setup_and_core_metrics"] = time.monotonic() - started_total
        ledger.record["HF_output_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink())
        ledger.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--private-hf-root", type=Path, required=True)
    parser.add_argument("--case", choices=("input00_seed42", "input10_seed42"))
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()
    prepared = preflight(args.protocol, args.hf_root, args.private_hf_root)
    if args.execute_native:
        if not args.case:
            parser.error("--case is required for native execution")
        execute(args, prepared)
    else:
        print(json.dumps(prepared[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
