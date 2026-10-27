"""Four independent five-source native dense/fixed-consumer video parts.

The immutable consumer bank and the previously completed neutral replays are
inputs.  Each part keeps one native pipeline and CPU weight store, alternates
dense/candidate GPU residency, and saves ten complete newly paired videos.
"""
from __future__ import annotations

import argparse
import filecmp
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

from run_ub1_consumer_videos import (PER_CASE_LIMITS as CONTROL_LIMITS,
    REQUIRED_SOURCES as NATIVE_SOURCES, VERSIONS, bound_path, inspect_video,
    object_sha, validate_native_counts, verify_coefficient_chains, write_json)
from ub1_consumer_runtime import (ConsumerResidency, UB1ConsumerRuntime,
    bytes_of_tensors, confined, expected_expert, load_original_biases,
    prepare_positions, replace_consumers, sha256, tensor_sha, verify_binding)
from ffnds_cohort_preflight import audit as audit_cohort

ROOT = Path(__file__).resolve().parents[1]
ID = "UB1_FIXED_RR_CONSUMER_DISJOINT20_20261027"
MODES = ("dense", "candidate")
VIDEO_CONTRACT = {"frames": 81, "width": 832, "height": 464, "fps": 16,
    "steps": 4, "blocks": 40, "denoising_steps": [1000, 750, 500, 250],
    "latent_shape": [16, 21, 58, 104]}
PART_LIMITS = {"video_generations": 10, "native_full_model_flows": 40,
    "native_main_FFN_linears": 3200, "native_RR_linears": 1600,
    "native_FFN_activation_QDQ": 3200, "native_non_FFN_linears": 12800,
    "native_non_FFN_activation_QDQ": 12800, "core_video_reports": 10,
    "actual_initial_noise_files": 10, "post_step_FP32_latent_files": 40,
    "new_fits": 0, "optimizer_updates": 0, "official_quality_calls": 0,
    "neutral_generations": 0}
REQUIRED_SOURCES = NATIVE_SOURCES | {
    "tools/run_ub1_consumer_disjoint20.py", "tests/test_run_ub1_consumer_disjoint20.py",
    "tools/ffnds_experiment_plans.py", "tools/ffnds_cohort_preflight.py",
    "tools/ffnds_stage_registered_i2v.py", ".runtime/code/LightX2V/lightx2v/pipeline.py",
    ".runtime/code/LightX2V/lightx2v/models/runners/default_runner.py",
    ".runtime/code/LightX2V/lightx2v/models/runners/wan/wan_runner.py"}
AUTHORIZATION = "USER_RESUME_OFFLINE_IFC_20261027"


def validate_plan(plan, part):
    if (plan.get("id") != ID or plan.get("assets_finalized") is not True or
            plan.get("modes") != list(MODES) or plan.get("limits_per_part") != PART_LIMITS or
            plan.get("video_contract") != VIDEO_CONTRACT or plan.get("automatic_retry") is not False or
            plan.get("max_full_generations_total") != 40 or plan.get("max_gpu_cards") != 4 or
            plan.get("planned_simultaneous_GPU_cards") != 4 or
            plan.get("cohort_exposure") != "PREVIOUSLY_EXPOSED_DEVELOPMENT_COHORT" or
            plan.get("new_neutral_generations") != 0 or type(part) is not int or part not in range(4)):
        raise ValueError("Unfrozen fixed four-part/40-video development-cohort scope")
    cases = plan["cases"]
    if (len(cases) != 20 or [c["index"] for c in cases] != list(range(20)) or
            [c["id"] for c in cases] != [f"source{i:02d}_seed42" for i in range(20)] or
            any(c["seed"] != 42 or c["dimension"] != ("i2v_subject" if i < 10 else "i2v_background") for i, c in enumerate(cases))):
        raise ValueError("Fixed enumeration of ten subject and ten background sources required")
    expected_parts = [{"part": n, "source_indices": list(range(n * 5, n * 5 + 5))} for n in range(4)]
    if plan.get("parts") != expected_parts:
        raise ValueError("Fixed nonoverlapping five-source partitions required")
    resources = plan["resource_limits"]
    if (resources["cpu_threads"] != 8 or resources["per_part_HF_max_bytes"] != 768 << 20 or
            resources["total_HF_max_bytes"] != 3 << 30 or resources["compute_wall_seconds"] != 1800 or
            resources["external_launcher_total_wall_seconds"] != 2100 or
            resources["host_available_min_bytes"] < 128 << 30 or
            not 0 < resources["resident_tensor_max_bytes"] <= 64 << 30):
        raise ValueError("Fixed resource limits differ")
    validate_registered_jobs(plan)
    return cases[part * 5:part * 5 + 5]


def validate_registered_jobs(plan):
    cohort = plan["cohort_plan"]
    checked = audit_cohort(cohort)
    if (checked["errors"] or cohort.get("schema") != "ffnds.cohort.v1" or
            cohort.get("mode") != "CUSTOM_PAIRED" or cohort.get("experiment") != ID or
            cohort.get("phase") != "REGISTERED_REAL_GENERATION_PLAN" or
            cohort.get("required_variants") != list(MODES) or cohort.get("calibration") != [] or
            len(cohort["sources"]) != 20 or len(cohort["samples"]) != 40 or
            plan["source_bank"] != {"schema": "ffnds.source.bank.v1", "sources": cohort["sources"],
                "scope": "original source records with declared historical exposure"}):
        raise ValueError("Frozen custom paired source bank / forty real registered jobs differ")
    for case, source in zip(plan["cases"], cohort["sources"]):
        if (case["semantic_source_id"] != source["source_id"] or source["role"] != "DEVELOPMENT" or
                source["legacy_image_fingerprint"] != case["source_id"] or
                source["source_group"] != case["source_group"] or source["scene_stratum"] != case["dimension"] or
                source["native_source_ordinal"] != case["official_source_index"] or
                hashlib.sha256(source["prompt"].encode()).hexdigest() != case["prompt_sha256"] or
                source["exposure_history"] != {"fitting": False, "method_selection": True, "prior_evaluation": True}):
            raise ValueError("Source identity or prior development exposure differs")
        for offset, mode in enumerate(MODES):
            job = cohort["samples"][case["index"] * 2 + offset]
            run_id = f"{ID}:{source['source_id']}:0:{mode}"
            if (job["source_id"] != source["source_id"] or job["variant"] != mode or job["seed"] != 42 or
                    job["sample_index"] != 0 or job["run_id"] != run_id or job["asset_id"] != "video:" + run_id or
                    job["conditioning_id"] != source["conditioning_id"] or
                    job["initial_noise_id"] != f"{ID}:{source['source_id']}:noise:0" or
                    job["scheduler_id"] != plan["scheduler_id"] or job["status"] != "PLANNED" or
                    job["video_path"] is not None or job["run_receipt_path"] is not None or
                    any(job[k] != VIDEO_CONTRACT[k] for k in ("frames", "fps", "width", "height"))):
                raise ValueError("Registered run identity / paired noise / native geometry differs")
    scope = plan["native_registration"]
    if (scope.get("schema") != "ffnds.native.scope.v1" or scope.get("experiment") != ID or
            scope.get("status") != "REGISTERED_BOUNDED_GPU_EXECUTION" or
            scope.get("max_visible_GPUs") != 1 or scope.get("user_authorization") != AUTHORIZATION or
            scope.get("allowed_run_ids") != [j["run_id"] for j in cohort["samples"]]):
        raise ValueError("Exact forty-run native registration differs")
    return checked


def registration_authorized(plan, protocol_sha, active):
    latest = active.get("user_continuation_authorization_20261027", {})
    if latest.get("status") != "ACTIVE_CURRENT_USER_DIRECTED_SESSION" or latest.get("id") != AUTHORIZATION:
        return False
    expected = plan["native_registration"]["allowed_run_ids"]
    return any(s.get("id") == ID and s.get("protocol_sha256") == protocol_sha and
        s.get("allowed_run_ids") == expected and
        s.get("execution_status") in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
        for s in active.get("experiment_authority", {}).get("current_user_authorized_scopes", []))


def overlap_table(cases, training_sources, formal_sources):
    return [{"index": c["index"], "source_group": c["source_group"], "image_sha256": c["source_id"],
        "training24_indices": [s["index"] for s in training_sources if s["source_id"] == c["source_id"]],
        "formal20_indices": [i for i, s in enumerate(formal_sources) if s["input_provenance"]["image_sha256"] == c["source_id"]]}
        for c in cases]


def validate_cohort(plan, samples, recovery, bank, formal):
    cases = plan["cases"]
    if (len(samples) != 20 or recovery["status"] != "PASS_FROZEN_SOURCE_DISJOINT20_INPUTS" or
            recovery["source_count"] != 20 or recovery["canonical_sha256"] != plan["input_manifest"]["sha256"] or
            len({c["source_id"] for c in cases}) != 20 or len({c["source_group"] for c in cases}) != 20):
        raise ValueError("Original disjoint20 source/recovery identity differs")
    recovered = {r["index"]: r for r in recovery["records"]}
    if set(recovered) != set(range(20)):
        raise ValueError("Recovery source enumeration differs")
    for index, (case, sample) in enumerate(zip(cases, samples)):
        if (case["index"] != index or case["seed"] != sample["seed"] or sample["seed"] != 42 or
                sample["negative_prompt"] != "" or sample["role"] != "frozen_source_disjoint_supplement" or
                sample["source_group"] != case["source_group"] or sample["dimension"] != case["dimension"] or
                sample["source_index"] != case["official_source_index"] or
                sample["prompt_sha256"] != case["prompt_sha256"] or
                hashlib.sha256(sample["prompt"].encode()).hexdigest() != case["prompt_sha256"] or
                sample["input_provenance"]["image_sha256"] != case["source_id"] or
                case["image"]["sha256"] != case["source_id"] or
                recovered[index]["source_group"] != case["source_group"] or
                recovered[index]["image_sha256"] != case["source_id"]):
            raise ValueError("Current source image/prompt/seed/enumeration differs")
    actual = overlap_table(cases, bank["sources"], formal)
    if actual != plan["source_overlap_table"] or any(r["training24_indices"] or r["formal20_indices"] for r in actual):
        raise ValueError("Frozen image-SHA disjointness table differs")
    return actual


def verify_controls(plan, roots):
    """Verify the old exact controls as inputs; no neutral replay is added."""
    old_path = bound_path(plan["prior_video_protocol"], roots)
    old_plan = json.loads(old_path.read_text())
    current_sources = {x["path"]: x["sha256"] for x in plan["repo_sources"]}
    if (old_plan["bank_manifest"]["sha256"] != plan["bank_manifest"]["sha256"] or
            old_plan["bank_protocol"]["sha256"] != plan["bank_protocol"]["sha256"] or
            any(current_sources.get(x["path"]) != x["sha256"] for x in old_plan["repo_sources"])):
        raise ValueError("Native method/bank differs from the completed exact controls")
    controls = plan["neutral_controls"]
    if len(controls) != 2 or [c["source_index"] for c in controls] != [0, 10]:
        raise ValueError("Both original neutral controls required")
    for control in controls:
        record = json.loads(bound_path(control["RUN"], roots).read_text())
        launch = json.loads(bound_path(control["launch_exit"], roots).read_text())
        if (record["status"] != "PASS_FIXED_FULL_DEPTH_CONSUMER_EXECUTION_CORE_METRICS_QUALITY_PENDING" or
                record["protocol_sha256"] != plan["prior_video_protocol"]["sha256"] or
                record["counts"] != CONTROL_LIMITS or record["case"]["index"] != control["source_index"] or
                launch["launcher_returncode"] != 0 or launch["process_exited"] is not True or
                launch["protocol_sha256"] != record["protocol_sha256"] or
                [v["mode"] for v in record["phases"]] != ["dense", "neutral", "candidate"]):
            raise ValueError("Prior successful native control receipt differs")
        neutral = record["phases"][1]
        comparison = neutral["historical_comparison"]
        if (comparison.get("passed") is not True or comparison.get("four_raw_FP32_latents_match") is not True or
                comparison.get("video_bytes_match") is not True or comparison["latent_match_by_step"] != [True] * 4 or
                [r["raw_tensor_sha256"] for r in neutral["trajectory"]] != control["neutral_latent_sha256"] or
                neutral["video"]["sha256"] != control["neutral_video"]["sha256"] or
                neutral["consumer_runtime"]["source_sha256"] != current_sources["tools/ub1_consumer_runtime.py"]):
            raise ValueError("Prior exact four-latent/video neutral identity differs")
        bound_path(control["neutral_video"], roots)
        validate_native_counts(neutral["native_counter_delta"], neutral["FFN_boundary_timings"], neutral["consumer_runtime"])
    return {"source_indices": [0, 10], "exact_original_neutral_replays": 2,
            "new_neutral_replays_in_this_scope": 0, "protocol_sha256": plan["prior_video_protocol"]["sha256"]}


def preflight(protocol, hf_root, private_root, part, *, active=None):
    plan = json.loads(Path(protocol).read_text())
    chosen = validate_plan(plan, part)
    roots = {"repo": ROOT.resolve(), "hf": Path(hf_root).resolve(), "private_hf": Path(private_root).resolve()}
    sources = {x["path"] for x in plan["repo_sources"]}
    if not REQUIRED_SOURCES <= sources or not any("hifxg_quant" in x and x.endswith(".so") for x in sources):
        raise ValueError("Required native source/extension bindings missing")
    for row in plan["repo_sources"]:
        verify_binding(ROOT, row)
    names = ("model_view", "local_model_view", "cache_manifest", "input_manifest", "input_recovery",
             "formal20_manifest", "original_recipe", "bank_protocol", "bank_manifest", "metric_contract",
             "prior_development_comparison")
    paths = {name: bound_path(plan[name], roots) for name in names}
    bank = json.loads(paths["bank_protocol"].read_text())
    samples = json.loads(paths["input_manifest"].read_text())
    recovery = json.loads(paths["input_recovery"].read_text())
    formal = json.loads(paths["formal20_manifest"].read_text())
    overlaps = validate_cohort(plan, samples, recovery, bank, formal)
    if recovery["original_quality_comparison_sha256"] != plan["prior_development_comparison"]["sha256"]:
        raise ValueError("Prior development exposure evidence differs")
    for case, source in zip(plan["cases"], plan["cohort_plan"]["sources"]):
        actual_image = bound_path(case["image"], roots)
        if Path(source["image_path"]).resolve() != actual_image or source["prompt"] != samples[case["index"]]["prompt"]:
            raise ValueError("Registered condition does not reference the actual frozen input")
    logical = json.loads(paths["model_view"].read_text())
    for expert, binding in plan["original_checkpoint_files"].items():
        matched = [r for r in logical["files"] if r["filename"] == binding["filename"]]
        if expert not in ("high_noise_model", "low_noise_model") or len(matched) != 1 or any(matched[0][k] != binding[k] for k in ("sha256", "bytes")):
            raise ValueError("Original native expert checkpoint identity differs")
    if set(plan["original_checkpoint_files"]) != {"high_noise_model", "low_noise_model"}:
        raise ValueError("Both native expert checkpoint bindings required")
    recipe = json.loads(paths["original_recipe"].read_text())
    if (bank["id"] != "UB1_FIXED_RR_CONSUMER_BANK_20261027" or not bank["assets_finalized"] or
            recipe["groups_by_step"] != [54, 108, 108, 162] or recipe["identity_control"] is not False or
            recipe["cache_manifest"]["sha256"] != plan["cache_manifest"]["sha256"]):
        raise ValueError("Fixed original UB1 support/RR/native cache identity differs")
    chain = verify_coefficient_chains(bank, roots["private_hf"], recipe)
    manifest = json.loads(paths["bank_manifest"].read_text())
    if (manifest["status"] != "COMPLETE_UB1_FIXED_RR_CONSUMER_BANK" or manifest["input_count"] != 24 or
            manifest["bank_protocol"] != {"path": plan["bank_protocol"]["path"], "sha256": plan["bank_protocol"]["sha256"]} or
            len(manifest["packs"]) != 160 or {(p["step"], p["block"]) for p in manifest["packs"]} != {(s, b) for s in range(4) for b in range(40)}):
        raise ValueError("Immutable full160 consumer bank differs")
    if len(manifest["part_RUNs"]) != 4 or {p["part"] for p in manifest["part_RUNs"]} != set(range(4)):
        raise ValueError("All four original bank construction receipts required")
    for row in manifest["part_RUNs"]:
        receipt = json.loads(verify_binding(roots["hf"], row).read_text())
        if (receipt["status"] != "COMPLETE_TRAINING_CONSUMER_BANK_PART" or receipt["part"] != row["part"] or
                receipt["protocol_sha256"] != plan["bank_protocol"]["sha256"] or len(receipt["sites"]) != 40):
            raise ValueError("Original bank part receipt differs")
    for row in manifest["packs"]:
        if row["expert"] != expected_expert(row["step"]):
            raise ValueError("Consumer expert/step identity differs")
        verify_binding(roots["hf"], row)
    controls = verify_controls(plan, roots)
    if "prior_quality_RUN" in plan:
        quality = json.loads(bound_path(plan["prior_quality_RUN"], roots).read_text())
        if quality["status"] != "COMPLETE_EIGHT_OFFICIAL_DIMENSIONS" or len(quality["completed"]) != 8:
            raise ValueError("Completed development quality receipt differs")
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != plan["runtime_versions"]:
        raise ValueError("Frozen native software versions differ")
    if active is None:
        active = json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text())
    authorized = registration_authorized(plan, sha256(protocol), active)
    return plan, chosen, samples, bank, manifest, roots, {"status": "STATIC_DISJOINT20_NATIVE_PREFLIGHT_PASS",
        "authorized": authorized, "part": part, "source_indices": [c["index"] for c in chosen],
        "GPU_initialized": False, "new_generations": 0, "versions": versions,
        "coefficient_chain": chain, "neutral_controls": controls, "source_overlap_table": overlaps,
        "cohort_audit": validate_registered_jobs(plan),
        "registered_run_ids": [j["run_id"] for j in plan["cohort_plan"]["samples"] if j["source_id"] in {c["semantic_source_id"] for c in chosen}]}


class NativeRecordsView:
    """Borrow CPU record references; restore the identical dense store on exit."""
    def __init__(self, base, mode):
        if mode not in MODES:
            raise ValueError("Only dense and fixed candidate are allowed")
        self.base, self.mode = base, mode

    def __enter__(self):
        if self.base._resident_weights:
            raise RuntimeError("A prior block GPU weight cache remains")
        self.original = self.base.records
        if self.mode == "candidate":
            self.base.records = {e: {n: r for n, r in rows.items() if not n.endswith((".ffn.0", ".ffn.2"))}
                                 for e, rows in self.original.items()}
        return self

    def __exit__(self, *error):
        self.base.records = self.original
        self.base._resident_key = None


def generation_costs(mode):
    if mode not in MODES:
        raise ValueError("New neutral or unknown generation is outside this scope")
    return {"video_generations": 1, "native_full_model_flows": 4, "native_main_FFN_linears": 320,
        "native_RR_linears": 320 if mode == "candidate" else 0, "native_FFN_activation_QDQ": 320,
        "native_non_FFN_linears": 1280, "native_non_FFN_activation_QDQ": 1280,
        "actual_initial_noise_files": 1, "post_step_FP32_latent_files": 4}


class PartLedger:
    def __init__(self, path, part, cases, protocol_sha, plan, hf_root):
        self.path, self.cases = Path(path), cases
        self.plan, self.hf_root = plan, Path(hf_root).resolve()
        self.order = [(c["index"], m) for c in cases for m in MODES]
        self.record = {"id": ID, "part": part, "protocol_sha256": protocol_sha,
            "status": "STARTED_NOT_ACCEPTED", "attempt": 1, "automatic_retry": False,
            "source_indices": [c["index"] for c in cases], "predebited_work": dict.fromkeys(PART_LIMITS, 0),
            "count_semantics": "whole-video upper bounds debited before dispatch; actual native observations and completed flow steps recorded per phase",
            "phases": [], "completed_pairs": [], "quality_or_speed_acceptance": False,
            "official_quality": "NOT_RUN_SEPARATE_SCOPE", "human_fullplay_review": "PENDING"}

    def job(self, index, mode):
        if (index, mode) not in self.order:
            raise RuntimeError("Registered job is outside this independent part")
        offset = index * 2 + MODES.index(mode)
        job = self.plan["cohort_plan"]["samples"][offset]
        if job["run_id"] not in self.plan["native_registration"]["allowed_run_ids"]:
            raise RuntimeError("Unregistered native job")
        return offset, job, self.plan["cohort_plan"]["sources"][index]

    def start(self):
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / "ATTEMPT.json").open("x") as stream:
            json.dump({"id": ID, "part": self.record["part"], "protocol_sha256": self.record["protocol_sha256"], "attempt": 1}, stream)
        self.save()

    def save(self):
        write_json(self.path / "RUN.json", self.record)

    def charge(self, key, count=1):
        if type(count) is not int or count < 0 or self.record["predebited_work"][key] + count > PART_LIMITS[key]:
            raise RuntimeError("Fixed part budget exceeded before " + key)
        self.record["predebited_work"][key] += count
        self.save()

    def reserve_generation(self, index, mode):
        sequence = self.record["predebited_work"]["video_generations"]
        if sequence >= 10 or len(self.record["phases"]) != sequence or self.order[sequence] != (index, mode):
            raise RuntimeError("Source/mode order or incomplete prior generation differs")
        costs = generation_costs(mode)
        if any(self.record["predebited_work"][k] + v > PART_LIMITS[k] for k, v in costs.items()):
            raise RuntimeError("Fixed native video budget exceeded before dispatch")
        for key, amount in costs.items():
            self.record["predebited_work"][key] += amount
        job_index, job, _ = self.job(index, mode)
        self.record["current_attempt"] = {"sequence": sequence, "source_index": index, "mode": mode,
            "registered_job_index": job_index, "registered_run_id": job["run_id"]}
        self.save()

    def complete_phase(self, phase):
        sequence = len(self.record["phases"])
        if (sequence >= len(self.order) or self.order[sequence] != (phase["source_index"], phase["mode"]) or
                phase.get("status") != "PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE" or
                phase.get("core_metrics", {}).get("status") != "ALL_81_FRAMES_CORE_METRICS_PASS_QUALITY_PENDING"):
            raise RuntimeError("Completed paired phase identity/core evidence differs")
        if phase["mode"] == "candidate":
            dense = self.record["phases"][-1]
            verify_noise_pair(phase["initial_noise"], dense["initial_noise"], self.hf_root)
            pair = {"source_index": phase["source_index"], "source_id": phase["source_id"], "seed": 42,
                "status": "PASS_FRESH_DENSE_CANDIDATE_PAIR_CORE_METRICS_QUALITY_PENDING",
                "fresh_dense_reference": dense["generation_receipt"], "candidate": phase["generation_receipt"],
                "dense_core_metrics": dense["core_metrics"], "candidate_core_metrics": phase["core_metrics"],
                "actual_initial_noise_files_byte_equal": True,
                "dense_initial_noise": dense["initial_noise"], "candidate_initial_noise": phase["initial_noise"],
                "quality_or_speed_acceptance": False}
            path = self.path / f"source{phase['source_index']:02d}_seed42/PAIR.json"
            with path.open("x") as stream:
                json.dump(pair, stream, ensure_ascii=False, indent=2, allow_nan=False)
            self.record["completed_pairs"].append({"source_index": phase["source_index"], "path": str(path), "sha256": sha256(path)})
        self.record["phases"].append(phase)
        self.save()


def save_initial_noise(latents, path, hf_root, check_resources, expected_shape):
    """Save the actual pre-update state without casting its dtype or values."""
    import torch
    from safetensors.torch import save_file
    value = latents.detach().cpu().contiguous()
    if list(value.shape) != list(expected_shape) or not bool(torch.isfinite(value).all()):
        raise RuntimeError("Actual initial-noise shape or finite values differ")
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing actual initial-noise replacement")
    check_resources(projected_bytes=value.numel() * value.element_size() + 1024)
    save_file({"latents": value}, str(path))
    return {"path": str(path.resolve().relative_to(Path(hf_root).resolve())), "sha256": sha256(path),
        "raw_tensor_sha256": tensor_sha(value), "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."), "bytes": path.stat().st_size,
        "capture": "scheduler.step_post step0 before original post; native generator reset per request"}


def verify_noise_pair(candidate, dense, hf_root):
    left, right = verify_binding(hf_root, candidate), verify_binding(hf_root, dense)
    if (left.samefile(right) or not filecmp.cmp(left, right, shallow=False) or
            any(candidate[k] != dense[k] for k in ("raw_tensor_sha256", "shape", "dtype"))):
        raise RuntimeError("Fresh Dense/candidate actual initial-noise bytes differ or alias")
    return True


def registered_receipt(phase, job, source, assets):
    validate_native_counts(phase["native_counter_delta"], phase["FFN_boundary_timings"], phase["consumer_runtime"])
    return {"schema": "ffnds.registered.native.video.v1", "status": "COMPLETE_NATIVE_GENERATION_ONLY",
        "job": job, "source": source, "actual_initial_noise_file": phase["actual_initial_noise_file"],
        "initial_noise_shape": phase["initial_noise"]["shape"], "initial_noise": phase["initial_noise"],
        "video_path": phase["video"]["path"], "decoded_frames": phase["video"]["frames"],
        "decoded_fps": phase["video"]["fps"], "decoded_size": phase["video"]["size"],
        "native_step_trace": phase["trajectory"], "native_ffn_counters": phase["native_counter_delta"],
        "native_ffn_sites": 160, "native_ffn_site_receipts": phase["FFN_boundary_timings"],
        "model_assets": assets, "generation_wall_seconds": phase["generation_wall_seconds_including_video_save_latent_save_and_hash"],
        "peak_cuda_allocated_bytes": phase["peak_cuda_allocated_bytes"], "scheduler_generator_reset": True,
        "method": "dense" if phase["mode"] == "dense" else "UB1_FIXED_RR_FITTED_CONSUMER",
        "official_quality": "NOT_RUN_SEPARATE_SCOPE", "quality_accepted": False,
        "immutable_generation_receipt": phase["generation_receipt"]}


def generate_video(pipe, base, runtime, mode, case, sample, image, ledger, timer,
                   monitor, check_resources, preload_seconds, shared_setup_seconds, contract=VIDEO_CONTRACT, dense=None):
    import torch
    from safetensors.torch import save_file
    ledger.reserve_generation(case["index"], mode)
    parent = ledger.path / case["id"] / mode
    parent.mkdir(parents=True, exist_ok=False)
    job_index, job, source = ledger.job(case["index"], mode)
    phase = {"status": "STARTED_NOT_ACCEPTED", "source_index": case["index"],
        "id": ID, "protocol_sha256": ledger.record["protocol_sha256"],
        "bank_manifest_sha256": ledger.plan["bank_manifest"]["sha256"] if mode == "candidate" else None,
        "official_source_index": case["official_source_index"], "source_group": case["source_group"],
        "dimension": case["dimension"], "mode": mode, "source_id": case["source_id"], "seed": 42,
        "prompt_sha256": case["prompt_sha256"], "trajectory": [], "scheduler_generator_reset": True,
        "registered_job_index": job_index, "registered_run_id": job["run_id"],
        "semantic_source_id": source["source_id"], "image_id": source["image_id"],
        "conditioning_id": job["conditioning_id"], "scheduler_id": job["scheduler_id"],
        "initial_noise_id": job["initial_noise_id"], "image_sha256": case["source_id"],
        "setup_seconds": preload_seconds, "shared_part_setup_seconds": shared_setup_seconds,
        "setup_cost_scope": "per-video setup is exclusive weight preloading; shared initialization is recorded once in the part RUN",
        "cohort_exposure": "PREVIOUSLY_EXPOSED_DEVELOPMENT_COHORT", "quality_or_speed_acceptance": False}
    scheduler = pipe.runner.scheduler
    original_post = scheduler.step_post
    before = dict(base.counters)
    if runtime is not base:
        runtime.begin_request()
    timer.events.clear()
    scheduler.generator = None
    torch.cuda.reset_peak_memory_stats()

    def post():
        step = int(scheduler.step_index)
        if step == 0:
            if "initial_noise" in phase:
                raise RuntimeError("Initial native noise recorded more than once")
            phase["initial_noise"] = save_initial_noise(scheduler.latents, parent / "initial_noise.safetensors",
                ledger.hf_root, check_resources, contract["latent_shape"])
            phase["actual_initial_noise_file"] = str(parent / "initial_noise.safetensors")
            if mode == "candidate":
                if dense is None or any(dense[k] != phase[k] for k in ("source_index", "source_id", "seed", "conditioning_id", "scheduler_id", "initial_noise_id")):
                    raise RuntimeError("Candidate has no fresh paired native initial state")
                verify_noise_pair(phase["initial_noise"], dense["initial_noise"], ledger.hf_root)
                phase["actual_initial_noise_files_byte_equal"] = True
            write_json(parent / "PHASE.json", phase)
        value = original_post()
        latent = scheduler.latents.detach().float().cpu().contiguous()
        if (step != len(phase["trajectory"]) or step not in range(4) or list(latent.shape) != contract["latent_shape"] or
                not bool(torch.isfinite(latent).all()) or int(scheduler.denoising_step_list[step]) != contract["denoising_steps"][step]):
            raise RuntimeError("Native latent shape/order/finite/scheduler contract differs")
        check_resources(projected_bytes=latent.numel() * latent.element_size() + (1 << 20))
        path = parent / f"latent_step{step}.safetensors"
        if path.exists():
            raise FileExistsError("Refusing latent replacement")
        save_file({"latents": latent}, str(path))
        phase["trajectory"].append({"step": step, "denoising_step": int(scheduler.denoising_step_list[step]),
            "sigma": float(scheduler.sigmas[step]), "shape": list(latent.shape), "dtype": "float32",
            "raw_tensor_sha256": tensor_sha(latent), "file_sha256": sha256(path), "path": path.name, "bytes": path.stat().st_size})
        write_json(parent / "TRAJECTORY.json", phase["trajectory"])
        check_resources()
        return value

    scheduler.step_post = post
    write_json(parent / "PHASE.json", phase)
    started = time.perf_counter()
    try:
        check_resources()
        with torch.inference_mode():
            pipe.generate(seed=42, image_path=str(image), prompt=sample["prompt"], negative_prompt="", save_result_path=str(parent / "output.mp4"))
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        monitor.check()
        consumer = runtime.end_request() if runtime is not base else None
        counts = {k: base.counters[k] - before[k] for k in before}
        timings = timer.report()
        validate_native_counts(counts, timings, consumer)
        if "initial_noise" not in phase or [r["step"] for r in phase["trajectory"]] != [0, 1, 2, 3]:
            raise RuntimeError("Native trajectory incomplete")
        phase.update(status="PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE",
            generation_wall_seconds_including_video_save_latent_save_and_hash=wall,
            FFN_boundary_seconds=sum(r["elapsed_ms"] for r in timings) / 1000,
            FFN_boundary_timings=timings, native_counter_delta=counts, consumer_runtime=consumer,
            completed_native_flow_steps=len(phase["trajectory"]), video=inspect_video(parent / "output.mp4", contract),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        receipt = parent / "GENERATION.json"
        with receipt.open("x") as stream:
            json.dump(phase, stream, ensure_ascii=False, indent=2, allow_nan=False)
        phase["generation_receipt"] = {"path": str(receipt), "sha256": sha256(receipt)}
        write_json(parent / "PHASE.json", phase)
        return phase
    except BaseException as exc:
        phase.update(status="FAILED_NATIVE_PHASE_PRESERVED_NO_RETRY", error_type=type(exc).__name__, error=str(exc),
            generation_wall_seconds_before_failure=time.perf_counter() - started,
            completed_native_flow_steps=len(phase["trajectory"]),
            observed_native_counter_delta={k: base.counters[k] - before[k] for k in before})
        if runtime is not base:
            phase["failed_consumer_runtime"] = runtime.report()
        write_json(parent / "PHASE.json", phase)
        raise
    finally:
        scheduler.step_post = original_post


def metric_inputs(phase, case, protocol_sha, assets, dense=None):
    receipt = phase["generation_receipt"]
    if sha256(receipt["path"]) != receipt["sha256"] or sha256(phase["video"]["path"]) != phase["video"]["sha256"]:
        raise ValueError("Current immutable generation/video evidence differs")
    identity = {"source_id": case["source_id"], "seed": 42, "expected_frames": 81,
        "input_prompt_digest": case["prompt_sha256"], "model_digest": object_sha(assets),
        "protocol_sha256": protocol_sha, "run_sha256": receipt["sha256"]}
    inputs = {"identity": identity, "cost": {
        "generation_wall_seconds": phase["generation_wall_seconds_including_video_save_latent_save_and_hash"],
        "ffn_boundary_seconds": phase["FFN_boundary_seconds"], "setup_seconds": phase["setup_seconds"],
        "peak_gpu_memory_mib": phase["peak_cuda_allocated_bytes"] / (1 << 20)}}
    if phase["mode"] == "candidate":
        if dense is None or dense["mode"] != "dense" or any(dense[k] != phase[k] for k in ("source_index", "source_id", "seed", "prompt_sha256")):
            raise ValueError("Candidate requires its fresh same-source/seed/prompt dense reference")
        if sha256(dense["generation_receipt"]["path"]) != dense["generation_receipt"]["sha256"] or sha256(dense["video"]["path"]) != dense["video"]["sha256"]:
            raise ValueError("Fresh dense evidence changed")
        inputs["reference_identity"] = {"source_id": identity["source_id"], "seed": 42,
            "input_prompt_digest": identity["input_prompt_digest"], "run_sha256": dense["generation_receipt"]["sha256"]}
    return inputs


def attach_core_metrics(phase, case, ledger, assets, check_resources, dense=None):
    from ffnds_video_metrics import evaluate, validate
    inputs = metric_inputs(phase, case, ledger.record["protocol_sha256"], assets, dense)
    ledger.charge("core_video_reports")
    started = time.perf_counter()
    parent = ledger.path / case["id"] / phase["mode"]
    try:
        report = evaluate(Path(phase["video"]["path"]), Path(dense["video"]["path"]) if dense else None, inputs)
        write_json(parent / "CORE_METRICS.json", report)
        errors = validate(report)
        if errors:
            raise RuntimeError("Incomplete all-frame core diagnostics: " + "; ".join(errors))
        phase["core_metrics"] = {"path": str(parent / "CORE_METRICS.json"), "sha256": sha256(parent / "CORE_METRICS.json"),
            "CPU_seconds": time.perf_counter() - started, "status": "ALL_81_FRAMES_CORE_METRICS_PASS_QUALITY_PENDING"}
        write_json(parent / "PHASE.json", phase)
        check_resources()
    except BaseException as exc:
        phase.update(status="FAILED_CORE_METRICS_PRESERVED_NO_RETRY", error_type=type(exc).__name__, error=str(exc))
        write_json(parent / "PHASE.json", phase)
        raise


def release_gpu_cache(base, torch_module):
    torch_module.cuda.synchronize()
    base._resident_weights.clear()
    base._resident_key = None
    gc.collect()
    torch_module.cuda.empty_cache()


def execute(args, prepared):
    plan, cases, samples, bank, manifest, roots, readiness = prepared
    if not args.execute_native or not readiness["authorized"]:
        raise RuntimeError("Exact frozen part protocol SHA must be registered before execution")
    from worker_identity import private_config_path, require_worker
    require_worker()
    if sha256(private_config_path()) not in plan["private_worker_config_sha256"]:
        raise ValueError("Private worker profile is not frozen for this scope")
    out = confined(roots["hf"], plan["output_HF_relative"]) / f"part{args.part}"
    ledger = PartLedger(out, args.part, cases, sha256(args.protocol), plan, roots["hf"])
    ledger.start()
    ledger.record["static_preflight_seconds"] = readiness["static_preflight_seconds"]
    ledger.record["registered_run_ids"] = readiness["registered_run_ids"]
    ledger.record["cohort_audit"] = readiness["cohort_audit"]
    limits = plan["resource_limits"]
    monitor = adapter = base = None
    start_total = time.monotonic()
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError("Fixed disjoint20 part wall budget reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(limits["compute_wall_seconds"])
    def check_resources(projected_bytes=0):
        if monitor:
            monitor.check()
        used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink())
        if used + projected_bytes > limits["per_part_HF_max_bytes"] or shutil.disk_usage(out).free - projected_bytes < limits["HF_free_min_bytes"]:
            raise RuntimeError("Fixed part HF output/free-space budget reached")
        available = int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))) * 1024
        if available < limits["host_available_min_bytes"]:
            raise RuntimeError("Fixed part host-memory reserve reached")
    try:
        check_resources()
        snapshot = out / "EXECUTED_SOURCES"
        snapshot.mkdir()
        for binding in plan["repo_sources"]:
            destination = snapshot / binding["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            check_resources(projected_bytes=(ROOT / binding["path"]).stat().st_size)
            shutil.copyfile(ROOT / binding["path"], destination)
        for name, source in (("protocol.json", args.protocol), ("bank_protocol.json", bound_path(plan["bank_protocol"], roots)),
                             ("BANK_MANIFEST.json", bound_path(plan["bank_manifest"], roots)),
                             ("prior_video_protocol.json", bound_path(plan["prior_video_protocol"], roots))):
            shutil.copyfile(source, snapshot / name)
        write_json(snapshot / "source_bank.json", plan["source_bank"])
        write_json(snapshot / "registered_video_plan.json", plan["cohort_plan"])
        write_json(snapshot / "native_registration.json", plan["native_registration"])
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
        allowed = {(ROOT / p["path"]).resolve() for p in plan["repo_sources"] if "hifxg_quant" in p["path"] and p["path"].endswith(".so")}
        if Path(extension.__file__).resolve() not in allowed:
            raise ValueError("Loaded native QDQ extension is not frozen")
        reservation, monitor, receipt = reserve_before_assets(torch)
        ledger.record["CUDA_reservation"] = receipt
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0) or os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
            raise ValueError("Exactly one assigned A100 with original arithmetic required")
        torch.set_num_threads(limits["cpu_threads"])
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        ledger.record["environment"] = {"versions": readiness["versions"], "CUDA": torch.version.cuda,
            "GPU": torch.cuda.get_device_name(0), "TF32": False,
            "BF16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        setup_start = time.perf_counter()
        history = HistoricalSources()
        kwargs, cache, assets = verified_restored_assets(history)
        ledger.record.update(assets=assets, historical_source_verifications=history.receipts,
            coefficient_chain=readiness["coefficient_chain"], neutral_controls=readiness["neutral_controls"],
            source_overlap_table=readiness["source_overlap_table"])
        biases = load_original_biases({"high_noise_model": kwargs["high_noise_ckpt"], "low_noise_model": kwargs["low_noise_ckpt"]})
        ledger.record["original_checkpoint_biases"] = {f"{e}_block{b:02d}": {k: v for k, v in row.items() if not torch.is_tensor(v)} for (e, b), row in biases.items()}
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        base = adapter.HIF4CacheRuntime(cache["path"])
        original_records = base.records
        positions = prepare_positions(base, bank, roots["private_hf"], biases)
        replace_consumers(positions, manifest, roots["hf"])
        del biases
        gc.collect()
        shared_setup = time.perf_counter() - setup_start
        ledger.record["shared_setup_seconds"] = shared_setup
        dense_bytes = sum(bytes_of_tensors(r["weight_qdq"]) for rows in base.records.values() for r in rows.values() if torch.is_tensor(r.get("weight_qdq")))
        non_ffn_bytes = sum(bytes_of_tensors(r["weight_qdq"]) for rows in base.records.values() for n, r in rows.items() if torch.is_tensor(r.get("weight_qdq")) and not n.endswith((".ffn.0", ".ffn.2")))
        pack_bytes = sum(sum(bytes_of_tensors(p[k]) for k in ("w1", "b1", "s1", "s2", "channels", "rr", "w2", "bias")) for p in positions.values())
        largest = max(dense_bytes, non_ffn_bytes + pack_bytes)
        estimated = largest + limits["reference_original_peak_physical_MiB"] * (1 << 20) + limits["memory_safety_bytes"]
        if largest > limits["resident_tensor_max_bytes"] or estimated > torch.cuda.get_device_properties(0).total_memory:
            raise RuntimeError("Alternating dense/candidate residency exceeds A100 headroom")
        ledger.record["memory_precheck"] = {"dense_weight_bytes": dense_bytes, "candidate_non_FFN_weight_bytes": non_ffn_bytes,
            "fixed_candidate_pack_bytes": pack_bytes, "maximum_resident_tensor_bytes": largest, "estimated_peak_bytes": estimated,
            "dense_CPU_records_preserved_by_reference": True, "dense_and_candidate_GPU_residency_overlap": False}
        ledger.save()
        device = torch.device("cuda", torch.cuda.current_device())
        with torch.inference_mode(), FFNStageTimer(adapter) as timer:
            for case in cases:
                dense = None
                image = bound_path(case["image"], roots)
                sample = samples[case["index"]]
                for mode in MODES:
                    release_gpu_cache(base, torch)
                    if base.records is not original_records:
                        raise RuntimeError("Original CPU dense records were not restored")
                    with NativeRecordsView(base, mode):
                        policy = AllWeightResidency(base, "all_resident_no_flush", limits["resident_tensor_max_bytes"]) if mode == "dense" else ConsumerResidency(base, positions, device, limits["resident_tensor_max_bytes"])
                        with policy:
                            preload_start = time.perf_counter()
                            if mode == "dense":
                                for expert, rows in base.records.items():
                                    for name, record in rows.items():
                                        if torch.is_tensor(record.get("weight_qdq")):
                                            base._weight(expert=expert, block_index=int(name.split(".")[1]), module_name=name, record=record, device=device, dtype=torch.bfloat16)
                                torch.cuda.synchronize()
                                runtime = base
                            else:
                                policy.preload()
                                runtime = UB1ConsumerRuntime(base, positions, policy, adapter.native_hif4_activation_qdq, mode="candidate")
                            preload_seconds = time.perf_counter() - preload_start
                            adapter.activate_lightx2v_hooks(runtime=runtime)
                            phase = generate_video(pipe, base, runtime, mode, case, sample, image, ledger, timer,
                                monitor, check_resources, preload_seconds, shared_setup, dense=dense)
                            job_index, job, source = ledger.job(case["index"], mode)
                            registered_path = out / case["id"] / mode / "RUN.json"
                            with registered_path.open("x") as stream:
                                json.dump(registered_receipt(phase, job, source, assets), stream,
                                    indent=2, ensure_ascii=False, allow_nan=False)
                            phase["registered_receipt"] = {"path": str(registered_path), "sha256": sha256(registered_path)}
                            view = out.parent / f"job{job_index:06d}"
                            view.symlink_to(registered_path.parent.relative_to(out.parent), target_is_directory=True)
                            phase["residency"] = policy.snapshot()
                            attach_core_metrics(phase, case, ledger, assets, check_resources, dense)
                            ledger.complete_phase(phase)
                            if mode == "dense":
                                dense = phase
                            torch.cuda.synchronize()
                    adapter.deactivate_lightx2v_hooks()
                    release_gpu_cache(base, torch)
                    if policy.snapshot()["retained_bytes"] != 0 or base.records is not original_records:
                        raise RuntimeError("GPU residency or CPU record restoration incomplete")
        if ledger.record["predebited_work"] != PART_LIMITS or len(ledger.record["completed_pairs"]) != 5:
            raise RuntimeError("Fixed ten-video/five-pair part incomplete")
        check_resources()
        ledger.record["status"] = "PASS_FIXED_DISJOINT20_PART_FULL_DEPTH_CORE_METRICS_QUALITY_PENDING"
    except BaseException as exc:
        ledger.record.update(status="FAILED_PART_PRESERVED_NO_RETRY", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if adapter:
            adapter.deactivate_lightx2v_hooks()
        if monitor:
            monitor.close()
            ledger.record["assigned_GPU_resources"] = monitor.report()
            if ledger.record["assigned_GPU_resources"].get("violations"):
                ledger.record["status"] = "FAILED_ASSIGNED_GPU_MONITOR_PRESERVED"
        ledger.record["elapsed_seconds_including_shared_setup_preload_generation_and_core_metrics"] = time.monotonic() - start_total
        ledger.record["HF_output_bytes"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink())
        ledger.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--private-hf-root", type=Path, required=True)
    parser.add_argument("--part", type=int, choices=range(4), required=True)
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    prepared = preflight(args.protocol, args.hf_root, args.private_hf_root, args.part)
    prepared[-1]["static_preflight_seconds"] = time.monotonic() - started
    if args.execute_native:
        execute(args, prepared)
    else:
        print(json.dumps(prepared[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
