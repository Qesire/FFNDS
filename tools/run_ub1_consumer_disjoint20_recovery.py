"""One separately registered replacement part for disjoint20 sources 0..4.

The frozen twenty-source module supplies native execution and asset validation.
Its forty-job protocol is a read-only asset template, never this entrypoint's
executable authority. This scope has ten new jobs, one part and one GPU.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
ID = "UB1_FIXED_RR_CONSUMER_DISJOINT20_PART0_RECOVERY_20261027"
TEMPLATE_ID = "UB1_FIXED_RR_CONSUMER_DISJOINT20_20261027"
AUTHORIZATION = "USER_RESUME_OFFLINE_IFC_20261027"
TEMPLATE = {"root": "repo", "path": "config/ub1_consumer_disjoint20_20261027_v1.json",
    "sha256": "ad81fffb73256c14e65c4221d255080072bcc9902cf420b541e4da4f61254ca6", "bytes": 146392}
ENGINE = {"path": "tools/run_ub1_consumer_disjoint20.py",
    "sha256": "8a5fa265fe585e944bbace3b7f351a43225dd8b1adf2489eaac1be14abf40748", "bytes": 49708}
WORKER_B = "f2d4d146e019bb15aee904e2c5b175c44179558626044038301ab44e7244d289"
OUTPUT = "ffnds-derived/ub1_fixed_rr_consumer_disjoint20_part0_recovery_20261027_v1"
MODES = ("dense", "candidate")
INDICES = [0, 1, 2, 3, 4]
EXTRA_SOURCES = {"tools/run_ub1_consumer_disjoint20_recovery.py",
    "tests/test_run_ub1_consumer_disjoint20_recovery.py", TEMPLATE["path"]}
OVERRIDES = {"id", "status", "scope", "max_full_generations_total", "max_gpu_cards",
    "planned_simultaneous_GPU_cards", "limits_total", "private_worker_config_sha256",
    "parts", "output_HF_relative", "resource_limits", "storage_accounting", "cases",
    "source_bank", "cohort_plan", "native_registration", "source_overlap_table",
    "repo_sources", "execution_readiness_conditions", "pending_freeze_items", "independent_review",
    "quality_protocol"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def verify_file(root, binding):
    root = Path(root).resolve()
    path = (root / binding["path"]).resolve()
    if (not path.is_relative_to(root) or not path.is_file() or
            path.stat().st_size != binding["bytes"] or sha256(path) != binding["sha256"]):
        raise ValueError("Frozen recovery input differs: " + binding["path"])
    return path


def load_engine():
    """Load a private module object; never patch the old importable module."""
    path = verify_file(ROOT, ENGINE)
    spec = importlib.util.spec_from_file_location("_ub1_single_part_recovery_native_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_cohort(template):
    cohort = copy.deepcopy(template["cohort_plan"])
    cohort["experiment"] = ID
    cohort["sources"] = cohort["sources"][:5]
    cohort["samples"] = cohort["samples"][:10]
    for job in cohort["samples"]:
        job["run_id"] = f"{ID}:{job['source_id']}:0:{job['variant']}"
        job["asset_id"] = "video:" + job["run_id"]
        job["initial_noise_id"] = f"{ID}:{job['source_id']}:noise:0"
    return cohort


def validate_plan(plan, template, engine, part=0):
    if type(part) is not int or part != 0:
        raise ValueError("Only logical part0 recovery is executable")
    engine.validate_plan(template, 0)
    for name, value in template.items():
        if name not in OVERRIDES and plan.get(name) != value:
            raise ValueError("Recovery changes frozen native method/assets: " + name)
    expected_resource = {**template["resource_limits"], "total_HF_max_bytes": 768 << 20}
    if (plan.get("id") != ID or plan.get("engine_template") != TEMPLATE or
            plan.get("engine_source") != ENGINE or plan.get("assets_finalized") is not True or
            plan.get("parts") != [{"part": 0, "source_indices": INDICES}] or
            plan.get("cases") != template["cases"][:5] or
            plan.get("source_overlap_table") != template["source_overlap_table"][:5] or
            plan.get("limits_per_part") != engine.PART_LIMITS or plan.get("limits_total") != engine.PART_LIMITS or
            plan.get("max_full_generations_total") != 10 or plan.get("max_gpu_cards") != 1 or
            plan.get("planned_simultaneous_GPU_cards") != 1 or plan.get("modes") != list(MODES) or
            plan.get("automatic_retry") is not False or plan.get("new_neutral_generations") != 0 or
            plan.get("official_quality_calls") != 0 or plan.get("output_HF_relative") != OUTPUT or
            plan.get("private_worker_config_sha256") != [WORKER_B] or
            plan.get("resource_limits") != expected_resource or
            plan.get("recovery_policy") != {"logical_part": 0, "source_indices": INDICES,
                "old_part0_quarantined": True, "overwrite_original_outputs": False,
                "primary_twenty_source_assembly": "new sources0..4 plus original completed sources5..19",
                "replayed_sources_increase_independent_n": False,
                "new_fitting_or_parameter_changes": False, "native_monitor_timeout_seconds": 10}):
        raise ValueError("Fixed single-part/ten-video recovery scope differs")
    cohort = expected_cohort(template)
    if (plan.get("cohort_plan") != cohort or plan.get("source_bank") != {
            **template["source_bank"], "sources": cohort["sources"]}):
        raise ValueError("Recovery must contain only the five original sources and ten new jobs")
    registration = {**template["native_registration"], "experiment": ID,
        "allowed_run_ids": [j["run_id"] for j in cohort["samples"]]}
    if plan.get("native_registration") != registration:
        raise ValueError("Exact ten new run IDs required; old forty IDs grant no authority")
    if plan.get("quality_protocol") != {**template["quality_protocol"],
            "sampling_mode": "CUSTOM_PAIRED: replacement sources0..4 only, five previously exposed image-prompt clusters, one true seed42 generation per arm; combine with original completed sources5..19 for primary20 without increasing independent n."}:
        raise ValueError("Recovery exposure and independent-source accounting differ")
    inherited = {x["path"]: x for x in template["repo_sources"]}
    actual = {x["path"]: x for x in plan["repo_sources"]}
    if (len(actual) != len(plan["repo_sources"]) or set(actual) != set(inherited) | EXTRA_SOURCES or
            any(actual[name] != value for name, value in inherited.items())):
        raise ValueError("Original native source bindings must remain byte-identical")
    storage = plan["storage_accounting"]
    if (storage["per_part_videos"] != 10 or storage["per_part_post_step_FP32_latent_tensors"] != 40 or
            storage["per_part_actual_initial_noise_files"] != 10 or
            storage["total_saved_post_step_latent_tensors"] != 40 or
            storage["total_actual_initial_noise_files"] != 10 or
            storage["additional_full_flows_or_FFN_states_persisted"] != 0 or
            storage["global_job_directory_views"] != "ten relative directory symlinks job000000..job000009 inside the new output root, pointing into part0/source00..04_seed42/mode; no copied videos or alias samples" or
            storage["per_part_initial_noise_FP32_raw_bytes"] != 81070080 or
            storage["per_part_post_step_FP32_raw_bytes"] != 324280320 or
            storage["per_part_executed_source_snapshot_bytes"] != sum(x["bytes"] for x in plan["repo_sources"]) or
            storage["per_part_post_step_FP32_safetensor_bytes_expected"] +
            storage["per_part_initial_noise_FP32_safetensor_bytes_expected"] +
            storage["per_part_executed_source_snapshot_bytes"] +
            storage["per_part_video_trace_report_reserve_bytes"] > 768 << 20):
        raise ValueError("Recovery storage bounds differ or exceed three-quarter GiB")
    audited = engine.audit_cohort(cohort)
    if audited["errors"]:
        raise ValueError("Invalid actual five-source recovery cohort")
    return plan["cases"], audited


def read_active():
    return json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text())


def registration_authorized(plan, protocol_sha, active):
    latest = active.get("user_continuation_authorization_20261027", {})
    if latest.get("status") != "ACTIVE_CURRENT_USER_DIRECTED_SESSION" or latest.get("id") != AUTHORIZATION:
        return False
    expected = plan["native_registration"]["allowed_run_ids"]
    if len(expected) != 10 or len(set(expected)) != 10 or any(not x.startswith(ID + ":") for x in expected):
        return False
    return any(scope.get("id") == ID and scope.get("protocol_sha256") == protocol_sha and
        scope.get("allowed_run_ids") == expected and scope.get("max_gpu_cards") == 1 and
        scope.get("total_limits") == plan["limits_total"] and
        scope.get("execution_status") in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
        for scope in active.get("experiment_authority", {}).get("current_user_authorized_scopes", []))


def verify_quarantine(plan, roots, engine):
    old = json.loads(engine.bound_path(plan["quarantined_part0_RUN"], roots).read_text())
    launch = json.loads(engine.bound_path(plan["quarantined_part0_launch"], roots).read_text())
    if (old.get("id") != TEMPLATE_ID or old.get("part") != 0 or old.get("protocol_sha256") != TEMPLATE["sha256"] or
            old.get("status") != "FAILED_ASSIGNED_GPU_MONITOR_PRESERVED" or old.get("source_indices") != INDICES or
            old.get("predebited_work", {}).get("video_generations") != 3 or
            len(old.get("phases", [])) != 2 or len(old.get("completed_pairs", [])) != 1 or
            old.get("assigned_GPU_resources", {}).get("violations") != ["Assigned GPU monitor failed: TimeoutExpired"] or
            launch.get("launcher_returncode") != 1 or launch.get("process_exited") is not True or
            launch.get("protocol_sha256") != TEMPLATE["sha256"]):
        raise ValueError("Original failed part0/launcher quarantine identity differs")
    return {"old_part0_run": plan["quarantined_part0_RUN"], "old_part0_launch": plan["quarantined_part0_launch"],
        "original_status": old["status"], "original_predebited_work": old["predebited_work"],
        "old_part0_excluded_from_primary_twenty": True, "old_outputs_modified": False}


def preflight(protocol, hf_root, private_hf_root, part=0, *, active=None):
    plan = json.loads(Path(protocol).read_text())
    if plan.get("engine_template") != TEMPLATE:
        raise ValueError("Frozen original template identity required")
    template_path = verify_file(ROOT, TEMPLATE)
    template = json.loads(template_path.read_text())
    engine = load_engine()
    cases, audit = validate_plan(plan, template, engine, part)
    for row in plan["repo_sources"]:
        engine.verify_binding(ROOT, row)
    actual_active = read_active() if active is None else active
    # Original authorization remains exactly what the real ACTIVE says. It is
    # never set true, borrowed, or passed as this recovery's authority.
    old = engine.preflight(template_path, hf_root, private_hf_root, 0, active=actual_active)
    _, old_cases, samples, bank, manifest, roots, native_readiness = old
    if old_cases != cases:
        raise ValueError("Asset template selected a different source set")
    quarantine = verify_quarantine(plan, roots, engine)
    if (roots["hf"] / OUTPUT).resolve().is_relative_to((roots["hf"] / template["output_HF_relative"]).resolve()):
        raise ValueError("Recovery output must not enter quarantined original scope")
    readiness = {**native_readiness, "id": ID, "status": "STATIC_PART0_RECOVERY_NATIVE_PREFLIGHT_PASS",
        "authorized": registration_authorized(plan, sha256(protocol), actual_active),
        "asset_template_authorized": native_readiness["authorized"],
        "asset_template_is_execution_authority": False, "asset_template": TEMPLATE,
        "registered_run_ids": plan["native_registration"]["allowed_run_ids"],
        "cohort_audit": audit, "source_overlap_table": plan["source_overlap_table"], "quarantine": quarantine,
        "part": 0, "source_indices": INDICES, "maximum_new_generations": 10, "maximum_assigned_GPUs": 1}
    return plan, cases, samples[:5], bank, manifest, roots, readiness


def execution_engine(plan, protocol_sha, hf_root):
    engine = load_engine()
    original_ledger = engine.PartLedger
    frozen_plan = copy.deepcopy(plan)
    output = (Path(hf_root).resolve() / OUTPUT / "part0").resolve()

    class RecoveryLedger(original_ledger):
        def __init__(self, path, part, cases, digest, payload, root):
            if (type(part) is not int or part != 0 or cases != frozen_plan["cases"] or
                    [c["index"] for c in cases] != INDICES or payload != frozen_plan or digest != protocol_sha or
                    Path(root).resolve() != Path(hf_root).resolve() or Path(path).resolve() != output):
                raise RuntimeError("Recovery ledger permits only the new registered ten jobs in part0")
            super().__init__(path, part, cases, digest, payload, root)
            self.record.update(executable_scope="SINGLE_PART0_TEN_NEW_VIDEOS", asset_template=TEMPLATE,
                recovery_policy=frozen_plan["recovery_policy"], quarantined_part0_RUN=frozen_plan["quarantined_part0_RUN"],
                original_part0_not_relabelled=True)

        def job(self, index, mode):
            if type(index) is not int or index not in INDICES or mode not in MODES or self.plan != frozen_plan:
                raise RuntimeError("Job outside frozen five-source recovery scope")
            result = super().job(index, mode)
            if result[1] != frozen_plan["cohort_plan"]["samples"][index * 2 + MODES.index(mode)]:
                raise RuntimeError("Recovery job identity changed")
            return result

        def reserve_generation(self, index, mode):
            self.job(index, mode)
            if not registration_authorized(frozen_plan, protocol_sha, read_active()):
                raise RuntimeError("Actual ACTIVE no longer authorizes the exact ten-job recovery")
            return super().reserve_generation(index, mode)

    # These are scope/ledger adaptations in a newly loaded module object. Native
    # execution, parameter tensors, generation callbacks and monitoring are the
    # exact original functions; the old module and source bytes stay untouched.
    engine.ID = ID
    engine.PartLedger = RecoveryLedger
    return engine


def execute(args, prepared):
    plan, cases, samples, bank, manifest, roots, readiness = prepared
    if type(args.part) is not int or args.part != 0 or not args.execute_native:
        raise RuntimeError("Only explicit part0 native recovery can execute")
    digest = sha256(args.protocol)
    if json.loads(Path(args.protocol).read_text()) != plan or not registration_authorized(plan, digest, read_active()):
        raise RuntimeError("Exact recovery protocol and ten allowed run IDs must be registered in actual ACTIVE")
    template = json.loads(verify_file(ROOT, TEMPLATE).read_text())
    engine = execution_engine(plan, digest, roots["hf"])
    # The unmodified validator belongs to the old scope, so use the frozen asset
    # engine for template validation and this wrapper for actual execution scope.
    validate_plan(plan, template, load_engine(), args.part)
    if (cases != plan["cases"] or len(samples) != 5 or roots["hf"] != Path(args.hf_root).resolve() or
            roots["private_hf"] != Path(args.private_hf_root).resolve() or not readiness.get("authorized")):
        raise RuntimeError("Recovery prepared payload or actual authorization differs")
    engine.execute(args, prepared)
    run = json.loads((roots["hf"] / OUTPUT / "part0/RUN.json").read_text())
    if (run.get("status") != "PASS_FIXED_DISJOINT20_PART_FULL_DEPTH_CORE_METRICS_QUALITY_PENDING" or
            run.get("id") != ID or run.get("protocol_sha256") != digest or run.get("source_indices") != INDICES or
            run.get("predebited_work") != engine.PART_LIMITS or len(run.get("completed_pairs", [])) != 5):
        raise RuntimeError("Recovery did not complete its exact registered part; original outputs preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--private-hf-root", type=Path, required=True)
    parser.add_argument("--part", type=int, choices=[0], default=0)
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
