"""Six real native videos from one immutable mixed-lineage ordinary6016 bank.

The default command verifies bytes on CPU. Native execution additionally needs
an exact ACTIVE registration and an isolated A100. Dense is reproduced first;
then energy94/random94 each use their own two folded GEMMs and native Q2.
Importing this module loads neither Torch nor a model.
"""
from contextlib import ExitStack, contextmanager
import argparse
import filecmp
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import time
import traceback

from ub1_consumer_runtime import confined, sha256, verify_binding
from b_final_block_capture import patch_method, read_state_bytes

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_COMPACT6016_CONSUMER00_12_SIX_VIDEOS_20261028"
AUTHORIZATION = "USER_RESUME_OFFLINE_IFC_20261027"
MODES = ("dense", "energy94", "random94")
LINEAR_SUFFIXES = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2")
INDICES = (0, 12)
RUN_IDS = tuple(f"compact6016_video_source{s:02d}_seed42_{m}" for m in MODES for s in INDICES)
BANK_SCHEMA = "ffnds.compact6016.deployment_bank.mixed_lineage.v1"
BANK_SHA = "e3c4f4020951913f9cffb2e2cf244137e94db2031ea64aabea7bc65a7aff6a50"
VIDEO = {"frames": 81, "width": 832, "height": 464, "fps": 16, "steps": 4, "blocks": 40,
         "denoising_steps": [1000, 750, 500, 250], "latent_shape": [16, 21, 58, 104]}
LIMITS = {"video_generations": 6, "input_encoder_requests": 6, "T5_positive_prompt_encodes": 6,
          "VAE_encodes": 6, "CLIP_encodes": 0, "scheduler_prepare": 6,
          "FP32_initial_noise_draws": 6, "native_full_model_flows": 24, "native_blocks": 960,
          "native_FFN_calls": 960, "native_FFN_linears": 1920, "native_FFN_activation_QDQ": 1920,
          "native_original_width_FFN_linears": 640, "native_narrow6016_FFN_linears": 1280,
          "native_non_FFN_linears": 7680, "native_non_FFN_activation_QDQ": 7680,
          "pre_infer_calls": 24, "pre_Default_MM_calls": 120, "patch_Conv3D_calls": 24,
          "head_Default_MM_calls": 24, "final_LayerNorm_calls": 24, "post_unpatchify_calls": 24,
          "scheduler_step_pre": 24, "scheduler_step_post": 24, "VAE_decodes": 6,
          "core_CPU_video_reports": 6, "actual_initial_noise_files": 6,
          "actual_raw_FP32_noise_files": 6, "matched_offload_buffer_resets": 12,
          "original_checkpoint_block_buffer_loads": 972,
          "post_step_FP32_latent_files": 24, "FFN_boundary_timer_records": 960,
          "narrow160_site_complete_reports": 4, "native_decoded_cache_matrix_transfers_setup": 2400,
          "native_decoded_cache_matrix_H2D_bytes_setup": 143025766400,
          "narrow_bias_smooth_tensor_transfers_setup": 640, "narrow_bias_smooth_H2D_bytes_setup": 10690560,
          "offline_training_updates": 0, "online_LoRA_GEMM": 0, "new_weight_quantizations": 0,
          "weight_decompositions": 0, "official_quality_calls": 0, "head_sensitivity_probes": 0,
          "extra_teacher_FFN_calls": 0}
REQUIRED_SOURCES = {"tools/e0_compact6016_six_videos.py", "tests/test_e0_compact6016_six_videos.py",
    "tools/compact6016_native_runtime.py", "tools/compact6016_remaining_runtime.py",
    "tools/e0_compact6016_remaining66.py", "tools/compact6016_remaining_contract.py",
    "tools/e0_compact6016_full80_completion.py", "tools/audit_compact6016_completion_partial.py", "tools/ub1_consumer_runtime.py",
    "tools/b_final_block_capture.py", "tools/ffnds_all_weight_residency.py",
    "tools/run_e0_all_weight_residency.py", "tools/run_ub1_consumer_videos.py",
    "tools/restored_assets.py", "tools/historical_source_snapshot.py", "tools/ffnds_video_metrics.py",
    "tools/ffnds_cohort_preflight.py", "tools/ffnds_stage_registered_i2v.py",
    "tools/worker_identity.py", "tools/startup_gpu_reservation.py",
    "tools/check_e0_full_depth_interface.py",
    "tools/assigned_gpu_resources.py", "tools/e0_b_final_block_capture_local_package.py",
    "tools/e0_b_final_block_capture.py", "tools/local_python_package_mirror.py",
    ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/pre_infer.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/post_infer.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/offload/transformer_infer.py",
    ".runtime/code/LightX2V/lightx2v/common/offload/manager.py",
    ".runtime/code/LightX2V/lightx2v/common/modules/weight_module.py",
    ".runtime/code/LightX2V/lightx2v/common/ops/norm/layer_norm_weight.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/model.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/distill_model.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/weights/pre_weights.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/weights/transformer_weights.py",
    ".runtime/code/LightX2V/lightx2v/models/runners/wan/wan_distill_runner.py",
    ".runtime/code/LightX2V/lightx2v/models/schedulers/scheduler.py"}


def digest(path):
    return sha256(path)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bind_file(root, binding):
    if not isinstance(binding, dict) or any(k not in binding for k in ("path", "bytes", "sha256")):
        raise ValueError("NOT_BOUND: immutable asset requires path, bytes and SHA")
    return verify_binding(root, binding)


def bound(binding, roots):
    return bind_file(roots[binding.get("root", "repo")], binding)


def verify_folded_payload(path, cell, *, dim=5120, hidden=13824, width=6016, group=64, rank=32):
    """Stream the actual entire export and every tensor payload without Torch.

    This proves stored bytes and whole-group coordinates. Original native
    bias/smooth selection and device load remain load_bank gates at execution.
    No tensor arithmetic or fictitious 'loaded bank' evidence is produced.
    """
    shapes = {"W1": (width, dim), "W2": (dim, width), "b1": (width,), "b2": (dim,),
        "smooth1": (dim,), "smooth2": (width,), "groups": (width // group,), "channels": (width,),
        "training_lora_A1": (rank, dim), "training_lora_B1": (width, rank),
        "training_lora_A2": (rank, width), "training_lora_B2": (dim, rank)}
    dtype_width = {"BF16": 2, "F32": 4, "I64": 8}
    path = Path(path)
    expected = cell["export"]
    if path.stat().st_size != expected["bytes"] or width % group or hidden % group:
        raise ValueError("Actual folded export byte/geometry mismatch")
    file_hash, tensor_hashes, coordinates = hashlib.sha256(), {}, {}
    with path.open("rb") as stream:
        length_raw = stream.read(8)
        if len(length_raw) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", length_raw)[0]
        if not 0 < length <= 1 << 20 or length + 8 > expected["bytes"]:
            raise ValueError("Invalid safetensors header length")
        raw_header = stream.read(length)
        header = json.loads(raw_header)
        header.pop("__metadata__", None)
        if set(header) != set(shapes) or set(expected["tensors"]) != set(shapes):
            raise ValueError("Exactly eight deploy plus four preserved training fields required")
        file_hash.update(length_raw); file_hash.update(raw_header)
        ordered = sorted(header, key=lambda k: header[k]["data_offsets"][0])
        position = 0
        for name in ordered:
            item, recorded = header[name], expected["tensors"][name]
            dtype = "I64" if name in ("groups", "channels") else "F32" if name.startswith(("smooth", "training_")) else "BF16"
            torch_dtype = {"I64": "torch.int64", "F32": "torch.float32", "BF16": "torch.bfloat16"}[dtype]
            start, end = item["data_offsets"]
            elements = math.prod(shapes[name])
            if (item["dtype"] != dtype or item["shape"] != list(shapes[name])
                    or recorded["dtype"] != torch_dtype or recorded["shape"] != list(shapes[name])
                    or start != position or end - start != elements * dtype_width[dtype]):
                raise ValueError("Folded payload header/shape/offset differs: " + name)
            digest = hashlib.sha256()
            remaining, small = end - start, bytearray()
            while remaining:
                raw = stream.read(min(8 << 20, remaining))
                if not raw:
                    raise ValueError("Truncated folded tensor payload")
                digest.update(raw); file_hash.update(raw); remaining -= len(raw)
                if name in ("groups", "channels"):
                    small.extend(raw)
            tensor_hashes[name] = digest.hexdigest()
            if tensor_hashes[name] != recorded["sha256"]:
                raise ValueError("Actual folded tensor payload SHA mismatch: " + name)
            if name in ("groups", "channels"):
                coordinates[name] = list(struct.unpack("<" + "q" * elements, small))
            position = end
        if stream.read(1) or 8 + length + position != expected["bytes"] or file_hash.hexdigest() != expected["sha256"]:
            raise ValueError("Actual complete folded export SHA/length mismatch")
    groups = coordinates["groups"]
    if (groups != cell["groups"] or groups != sorted(set(groups))
            or any(g < 0 or g >= hidden // group for g in groups)
            or coordinates["channels"] != [g * group + j for g in groups for j in range(group)]):
        raise ValueError("Actual stored whole-group row/channel coordinates differ")
    return {"run_id": cell["run_id"], "bytes": expected["bytes"], "sha256": file_hash.hexdigest(),
            "tensor_payloads_verified": len(tensor_hashes), "groups_and_channels_payload_verified": True,
            "native_loaded": False}


def memory_envelope(physical, resource):
    dense = physical["dense_decoded_matrix_bytes"]
    candidate = physical["candidate_attention_decoded_bytes"] + physical["candidate_FFN_matrix_bytes_per_init"] + physical["candidate_bias_smooth_bytes_per_init"]
    if candidate != physical["candidate_total_resident_tensor_bytes_per_init"]:
        raise ValueError("Non-FFN plus single-bank physical memory sum differs")
    maximum = max(dense, candidate)
    if maximum > resource["resident_cache_max_bytes"]:
        raise ValueError("Disjoint bank residency exceeds cap")
    return {"dense_resident_bytes": dense, "candidate_resident_bytes": candidate,
            "maximum_disjoint_resident_bytes": maximum,
            "estimated_peak_with_original_buffers_and_safety": maximum + resource["reference_original_peak_physical_MiB"] * (1 << 20) + resource["GPU_memory_safety_bytes"],
            "dense_plus_candidate_simultaneously": False,
            "candidate_original_decoded_FFN_matrices": 0,
            "original_checkpoint_FFN_prefetch": "retained inside original block buffers; separately measured, never removed from wall/physical memory"}


def validate_plan(plan, frozen=False):
    if (plan.get("id") != ID or plan.get("allowed_run_ids") != list(RUN_IDS)
            or plan.get("modes") != list(MODES) or plan.get("source_indices") != list(INDICES)
            or plan.get("seed") != 42 or plan.get("automatic_retry") is not False
            or plan.get("bank_schema") != BANK_SCHEMA or plan.get("video_contract") != VIDEO
            or plan.get("limits") != LIMITS):
        raise ValueError("Exact six-video identity, geometry or experiment budget differs")
    resources = plan["resource_limits"]
    if (resources.get("GPU") != 1 or resources.get("session_max_idle_GPUs") != 2
            or resources.get("repository_GPU_policy") != "DYNAMIC_MAX_ACTUALLY_CALLABLE"
            or resources.get("CPU_threads") != 8 or resources.get("new_HF_bytes") != 1 << 30
            or not 0 < resources.get("compute_wall_seconds", 0) <= 7200
            or resources.get("external_wall_seconds", 0) <= resources["compute_wall_seconds"]
            or resources.get("resident_cache_max_bytes") != 64 << 30
            or resources.get("host_available_min_bytes", 0) < 128 << 30):
        raise ValueError("Fixed one-GPU resource limits differ")
    if len(plan["cases"]) != 2 or [c["index"] for c in plan["cases"]] != list(INDICES):
        raise ValueError("Only two fixed exposed source identities are allowed")
    for case in plan["cases"]:
        if (case["seed"] != 42 or hashlib.sha256(case["prompt"].encode()).hexdigest() != case["prompt_sha256"]
                or case["image"]["sha256"] != case["image_sha256"]
                or case.get("exposure") != "PREVIOUSLY_EXPOSED_DEVELOPMENT_N2_SHARED_SEED42"):
            raise ValueError("Source conditioning or historical exposure differs")
        historical = case["historical_dense"]
        if len(historical["post"]) != 4 or len(historical["post_raw_FP32_sha256"]) != 4:
            raise ValueError("Four real historical Dense post-state bindings required")
    if frozen and (plan.get("assets_finalized") is not True or plan.get("bank_binding_status") != "BOUND_COMPLETE_MIXED160"):
        raise ValueError("NOT_BOUND: complete mixed160 bank/assets have not been frozen")
    return True


def authorize(plan, protocol_sha, active, run_id=None):
    validate_plan(plan, frozen=True)
    authority = active.get("user_continuation_authorization_20261027", {})
    scopes = [s for s in active.get("experiment_authority", {}).get("current_user_authorized_scopes", []) if s.get("id") == ID]
    policy = active.get("resource_scheduling_policy_20261028", {})
    session = active.get("current_session_resource_constraint_20261028", {})
    if (authority.get("id") != AUTHORIZATION or authority.get("status") != "ACTIVE_CURRENT_USER_DIRECTED_SESSION"
            or authority.get("new_scope_registration_allowed") is not True or len(scopes) != 1
            or policy.get("mode") != "DYNAMIC_MAX_ACTUALLY_CALLABLE" or policy.get("global_gpu_card_cap") is not None
            or session.get("max_simultaneously_assigned_idle_GPUs") != 2):
        raise ValueError("Active standing authority and unique video scope required")
    scope = scopes[0]
    if (scope.get("protocol_sha256") != protocol_sha or scope.get("allowed_run_ids") != list(RUN_IDS)
            or scope.get("execution_status") not in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
            or scope.get("limits") != LIMITS or scope.get("max_simultaneously_assigned_idle_GPUs") != 1
            or scope.get("session_max_simultaneously_assigned_idle_GPUs") != 2
            or run_id is not None and run_id not in RUN_IDS):
        raise ValueError("Exact six-video registration/run/budget differs")
    return scope


def validate_prior_cpu(scope, plan, protocol_sha, roots):
    entry = scope.get("cpu_preflight")
    if not isinstance(entry, dict):
        raise ValueError("Independent actual exited CPU preflight NOT_BOUND in ACTIVE")
    report = json.loads(bind_file(roots["hf"], entry["binding"]).read_text())
    receipt = json.loads(bind_file(roots["hf"], entry["launch_exit"]).read_text())
    source_sha = hashlib.sha256(json.dumps(plan["repo_sources"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (report.get("id") != ID or report.get("status") != "PASS_CPU_BYTES_NATIVE_VIDEO_NOT_EXECUTED"
            or report.get("protocol_sha256") != protocol_sha or report.get("source_bindings_sha256") != source_sha
            or report.get("bank_manifest_sha256") != BANK_SHA or report.get("mixed_bank_cells") != 160
            or len(report.get("actual_folded_export_payload_verifications", [])) != 160
            or report.get("NumPy_or_Torch_imported") is not False or report.get("all_CUDA_devices_hidden") is not True
            or report.get("bank_loaded_for_native") is not False or report.get("model_calls") != 0 or report.get("GPU_calls") != 0
            or receipt.get("id") != ID or receipt.get("protocol_sha256") != protocol_sha
            or receipt.get("launcher_returncode") != 0 or receipt.get("process_exited") is not True
            or receipt.get("RUN_sha256") != entry["binding"]["sha256"]):
        raise ValueError("Prior independent CPU byte preflight / actual exit differs")
    return entry


def registered_cohort(plan, hf_root):
    """Construct the six actual jobs with the shared cohort audit semantics."""
    from ffnds_cohort_preflight import audit
    sources, samples = [], []
    for case in plan["cases"]:
        sources.append({"source_id": case["source_id"], "image_id": case["image_sha256"],
            "prompt_id": case["prompt_sha256"], "role": "DEVELOPMENT", "prompt": case["prompt"],
            "image_path": str(confined(hf_root, case["image"]["path"])), "conditioning_id": case["conditioning_id"],
            "image_name": Path(case["image"]["path"]).name,
            "scene_stratum": case["source_group"].split(":")[0], "source_group": case["source_group"],
            "exposure_history": {"fitting": False, "method_selection": True, "prior_evaluation": True}})
    for mode in MODES:
        for case in plan["cases"]:
            run_id = f"compact6016_video_source{case['index']:02d}_seed42_{mode}"
            samples.append({"source_id": case["source_id"], "variant": mode, "seed": 42, "sample_index": 0,
                "run_id": run_id, "asset_id": "video:" + run_id, "conditioning_id": case["conditioning_id"],
                "initial_noise_id": ID + ":" + case["source_id"] + ":noise:0",
                "scheduler_id": "Wan22StepDistillScheduler:1000,750,500,250:high,high,low,low",
                **{k: VIDEO[k] for k in ("frames", "fps", "width", "height")},
                "status": "PLANNED", "run_receipt_path": None, "video_path": None})
    cohort = {"schema": "ffnds.cohort.v1", "mode": "CUSTOM_PAIRED", "experiment": ID,
              "phase": "REGISTERED_REAL_GENERATION_PLAN", "sources": sources, "samples": samples,
              "calibration": [], "required_variants": list(MODES)}
    checked = audit(cohort)
    if checked["errors"] or checked["summary"]["independent_image_clusters"] != 2 or checked["summary"]["unique_generation_run_ids"] != 6:
        raise ValueError("Six real paired jobs fail cohort audit: " + "; ".join(checked["errors"]))
    return cohort, checked


def verify_local_package(plan, roots, local_root):
    """Verify the successful mirror producer and complete bytes; import nothing."""
    from local_python_package_mirror import canonical_sha, inventory
    contract = plan["local_transformers"]
    producer = json.loads(bound(contract["producer_RUN"], roots).read_text())
    exited = json.loads(bound(contract["producer_exit"], roots).read_text())
    if (contract.get("transport") != "BYTE_IDENTICAL_NODE_LOCAL_TRANSFORMERS_NO_SOURCE_OR_MODEL_CHANGE"
            or contract.get("version") != "5.6.2" or producer.get("status") != "PASS_LOCAL_MIRROR_IMPORT_NO_MODEL"
            or producer.get("protocol_sha256") != contract["producer_protocol_sha256"]
            or producer.get("inventory_sha256") != contract["inventory_sha256"]
            or producer.get("archive_sha256") != contract["archive_sha256"]
            or producer.get("cuda_initialized") is not False or producer.get("cuda_device_count") != 0
            or producer.get("GPU_model_calls") != 0 or exited.get("launcher_returncode") != 0
            or exited.get("process_exited") is not True or exited.get("RUN_sha256") != contract["producer_RUN"]["sha256"]
            or exited.get("protocol_sha256") != producer["protocol_sha256"]):
        raise ValueError("Exact successful CPU-only local package producer/exit required")
    local_root = Path(local_root).resolve(strict=True)
    if hashlib.sha256(str(local_root).encode()).hexdigest() != contract["local_root_path_sha256"]:
        raise ValueError("Node-local mirrored package path identity differs")
    rows = json.loads(bound(contract["inventory"], roots).read_text())["files"]
    if canonical_sha(rows) != contract["inventory_sha256"] or inventory(local_root / "transformers") != rows:
        raise ValueError("Complete byte-identical local package inventory differs")
    if any(n == "transformers" or n.startswith("transformers.") for n in sys.modules):
        raise ValueError("Transformers imported before its verified local path")
    return local_root, rows


def preflight(protocol, hf_root, local_root, *, root=ROOT):
    started = time.monotonic()
    torch_before = "torch" in sys.modules
    if torch_before or "numpy" in sys.modules or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Independent CPU byte preflight requires hidden GPUs and no numeric imports")
    plan = json.loads(Path(protocol).read_text())
    validate_plan(plan, frozen=True)
    roots = {"repo": Path(root).resolve(), "hf": Path(hf_root).resolve()}
    source_paths = [p["path"] for p in plan["repo_sources"]]
    if len(source_paths) != len(set(source_paths)) or not REQUIRED_SOURCES <= set(source_paths):
        raise ValueError("Exact producer/native source bindings incomplete")
    for item in plan["repo_sources"]:
        bound(item, roots)
    from e0_compact6016_remaining66 import assemble_manifest
    manifest = json.loads(bound(plan["bank_manifest"], roots).read_text())
    if (manifest.get("schema") != BANK_SCHEMA or manifest.get("training_protocol") != plan["training_protocol"]
            or plan["bank_manifest"]["sha256"] != BANK_SHA or manifest.get("reused_complete_students") != 94
            or manifest.get("new_complete_students") != 66
            or manifest.get("parts") != plan["producer_part_receipts"]
            or manifest != assemble_manifest(plan["training_protocol"], plan["producer_part_receipts"], roots["hf"], repo_root=roots["repo"])):
        raise ValueError("Complete mixed160 manifest/actual completed receipts differ")
    if manifest["asset_view"] != plan["asset_view"]:
        raise ValueError("Original full80 native parent asset view differs")
    payloads = []
    for cell in manifest["cells"]:
        path = confined(roots["hf"], cell["export"]["path"])
        payloads.append(verify_folded_payload(path, cell))
    envelope = memory_envelope(plan["physical_cache"], plan["resource_limits"])
    canonical = json.loads(bound(plan["canonical_inputs"], roots).read_text())
    original = json.loads(bound(plan["original_cohort_protocol"], roots).read_text())
    bound(plan["native_logical_model_view"], roots)
    pointer_path = roots["repo"] / ".runtime/hif4_rebuilt_cache.json"
    pointer = json.loads(pointer_path.read_text())
    if ({k: pointer[k] for k in ("bytes", "sha256")} != plan["native_cache_identity"]
            or pointer["identity"]["model_view_sha256"] != plan["native_logical_model_view"]["sha256"]):
        raise ValueError("Worker pointer differs from bound original D-Q cache/model")
    historical_cases = []
    for case in plan["cases"]:
        historical = case["historical_dense"]
        producer = json.loads(bound(historical["producer_RUN"], roots).read_text())
        launch = json.loads(bound(historical["producer_exit"], roots).read_text())
        historical_protocol = bound(historical["producer_protocol"], roots)
        if (producer.get("status") != "PASS_FIXED_DISJOINT20_PART_FULL_DEPTH_CORE_METRICS_QUALITY_PENDING"
                or producer.get("protocol_sha256") != sha256(historical_protocol)
                or launch.get("launcher_returncode") != 0 or launch.get("process_exited") is not True
                or launch.get("protocol_sha256") != producer["protocol_sha256"]):
            raise ValueError("Historical Dense actual producer completion/exit differs")
        sample = canonical[case["index"]]
        source = original["cohort_plan"]["sources"][case["index"]]
        overlap = next(r for r in original["source_overlap_table"] if r["index"] == case["index"])
        if (sample["seed"] != 42 or sample["negative_prompt"] != "" or sample["prompt"] != case["prompt"]
                or sample["input_provenance"]["image_sha256"] != case["image_sha256"]
                or source["source_id"] != case["source_id"] or source["conditioning_id"] != case["conditioning_id"]
                or overlap["image_sha256"] != case["image_sha256"] or overlap["training24_indices"]):
            raise ValueError("Canonical actual image/prompt/seed/condition identity differs")
        bound(case["image"], roots)
        old = case["historical_dense"]
        generation = json.loads(bound(old["generation"], roots).read_text())
        initial = read_state_bytes(bound(old["initial"], roots), "BF16")
        posts = [read_state_bytes(bound(binding, roots), "F32") for binding in old["post"]]
        bound(old["video"], roots)
        if (generation.get("status") != "PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE"
                or generation.get("mode") != "dense" or generation.get("source_index") != case["index"]
                or generation.get("seed") != 42 or generation.get("image_sha256") != case["image_sha256"]
                or generation.get("semantic_source_id") != case["source_id"]
                or generation.get("protocol_sha256") != producer["protocol_sha256"]
                or generation.get("prompt_sha256") != case["prompt_sha256"]
                or generation.get("conditioning_id") != case["conditioning_id"]
                or generation.get("native_counter_delta") != {"linear_forward_calls": 1600,
                    "activation_qdq_calls": 1600, "weight_transfers": 0, "cache_misses": 0}
                or generation.get("completed_native_flow_steps") != 4
                or initial["raw_tensor_sha256"] != old["initial_raw_BF16_sha256"]
                or initial["raw_tensor_sha256"] != generation["initial_noise"]["raw_tensor_sha256"]
                or [x["raw_tensor_sha256"] for x in posts] != old["post_raw_FP32_sha256"]
                or [x["raw_tensor_sha256"] for x in posts] != [x["raw_tensor_sha256"] for x in generation["trajectory"]]
                or old["video"]["sha256"] != generation["video"]["sha256"]):
            raise ValueError("Historical Dense generation/state/video identity differs")
        historical_cases.append({"index": case["index"], "initial": initial, "post": posts})
    versions = {name: importlib.metadata.version(name) for name in plan["runtime_versions"]}
    if versions != plan["runtime_versions"]:
        raise ValueError("Frozen native package versions differ")
    local, rows = verify_local_package(plan, roots, local_root)
    cohort, checked = registered_cohort(plan, roots["hf"])
    readiness = {"id": ID, "status": "PASS_CPU_BYTES_NATIVE_VIDEO_NOT_EXECUTED", "protocol_sha256": sha256(protocol),
        "source_bindings_sha256": hashlib.sha256(json.dumps(plan["repo_sources"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "versions": versions, "historical_cases": historical_cases, "cohort_audit": checked,
        "mixed_bank_cells": len(manifest["cells"]), "local_package_files_checked": len(rows),
        "actual_folded_export_payload_verifications": payloads, "bank_loaded_for_native": False,
        "bank_manifest_sha256": plan["bank_manifest"]["sha256"],
        "memory_envelope": envelope, "all_CUDA_devices_hidden": True,
        "NumPy_or_Torch_imported": "numpy" in sys.modules or "torch" in sys.modules,
        "native_cache_pointer_sha256": sha256(pointer_path), "seconds": time.monotonic() - started,
        "Torch_or_CUDA_imported_by_preflight": not torch_before and "torch" in sys.modules,
        "model_calls": 0, "GPU_calls": 0}
    if readiness["Torch_or_CUDA_imported_by_preflight"]:
        raise RuntimeError("Byte-only preflight unexpectedly imported Torch")
    return plan, roots, manifest, cohort, local, rows, readiness


class Ledger:
    """Reserve before dispatch; exceptions retain reservations and journal events.

    Production uses a small append-only event journal before each real call.
    RUN snapshots occur at native step boundaries, avoiding thousands of full
    JSON rewrites on a network filesystem. Failed partial native counters are
    saved separately from successful-return observations.
    """
    def __init__(self, limits, persist=lambda record: None, journal=lambda event: None):
        self.limits, self.persist, self.journal = dict(limits), persist, journal
        self.record = {"predebited": {}, "observed": {}, "operation_serial": 0,
                       "journal_dispatch_wall_seconds": 0.}

    def _event(self, kind, run_id, costs):
        self.record["operation_serial"] += 1
        event = {"serial": self.record["operation_serial"], "kind": kind, "run_id": run_id, "costs": dict(costs)}
        started = time.perf_counter()
        self.journal(event)
        self.record["journal_dispatch_wall_seconds"] += time.perf_counter() - started
        self.record["last_operation"] = event
        self.persist(self.record)

    def reserve(self, run_id, costs):
        if run_id not in RUN_IDS and run_id != "setup":
            raise ValueError("Unregistered operation identity")
        used = self.record["predebited"]
        if not costs or any(k not in self.limits or type(v) is not int or v < 0
                            or used.get(k, 0) + v > self.limits[k] for k, v in costs.items()):
            raise ValueError("Operation exceeds budget")
        for key, value in costs.items():
            used[key] = used.get(key, 0) + value
        self._event("reserve_before_call", run_id, costs)

    def observe(self, costs, run_id="setup"):
        observed, used = self.record["observed"], self.record["predebited"]
        if not costs or any(k not in self.limits or type(v) is not int or v < 0
                            or observed.get(k, 0) + v > used.get(k, 0) for k, v in costs.items()):
            raise ValueError("Observed work was not reserved")
        for key, value in costs.items():
            observed[key] = observed.get(key, 0) + value
        self._event("observe_successful_return", run_id, costs)

    def call(self, run_id, costs, function, *args, **kwargs):
        self.reserve(run_id, costs)
        value = function(*args, **kwargs)
        self.observe(costs, run_id)
        return value

    def complete(self):
        return all(self.record[k].get(name, 0) == value for k in ("predebited", "observed") for name, value in self.limits.items())


class NativeCallTap:
    """Count the actual unchanged native path, not a synthetic full-flow budget."""
    def __init__(self, pipe, adapter, base, runtime, ledger, run_id, mode, persist=lambda: None,
                 after_step_pre=lambda: None, after_raw_draw=None):
        self.pipe, self.adapter, self.base, self.runtime = pipe, adapter, base, runtime
        self.ledger, self.run_id, self.mode, self.persist = ledger, run_id, mode, persist
        self.after_step_pre = after_step_pre
        self.after_raw_draw = after_raw_draw
        self.raw_draw_calls = 0
        self.stack = ExitStack()
        self.slot = None
        self.lines, self.blocks, self.ffns, self.flows = set(), [], [], []
        self.wall = {}
        self.checkpoint_buffer_loads = []
        self.attempted = {}
        self.qdq_slots = set()

    def counted(self, owner, name, costs, check=None):
        original = getattr(owner, name)
        def call(*args, **kwargs):
            if check:
                check(*args, **kwargs)
            for key, amount in costs.items():
                self.attempted[key] = self.attempted.get(key, 0) + amount
            started = time.perf_counter()
            try:
                return self.ledger.call(self.run_id, costs, original, *args, **kwargs)
            finally:
                for key in costs:
                    self.wall[key] = self.wall.get(key, 0.) + time.perf_counter() - started
        self.stack.enter_context(patch_method(owner, name, call))

    def step(self, expert_index=None):
        step = int(self.pipe.runner.scheduler.step_index)
        if step not in range(4) or expert_index is not None and expert_index != (0 if step < 2 else 1):
            raise ValueError("Actual native step/expert differs")
        return step

    def __enter__(self):
        try:
            self._install()
        except BaseException:
            self.stack.close()
            raise
        return self

    def _install(self):
        runner, scheduler = self.pipe.runner, self.pipe.runner.scheduler
        for name, label in (("run_input_encoder", "input_encoder_requests"), ("run_text_encoder", "T5_positive_prompt_encodes"),
                            ("run_vae_encoder", "VAE_encodes"), ("run_vae_decoder", "VAE_decodes")):
            self.counted(runner, name, {label: 1})
        self.counted(scheduler, "prepare", {"scheduler_prepare": 1})
        def check_noise(seed, latent_shape, dtype=None):
            if seed != 42 or list(latent_shape) != VIDEO["latent_shape"] or str(dtype) != "torch.float32" or scheduler.generator is not None:
                raise ValueError("Exactly one reset original FP32 scheduler noise draw required")
        original_prepare_latents = scheduler.prepare_latents
        def prepare_latents(seed, latent_shape, dtype=None):
            check_noise(seed, latent_shape, dtype)
            self.attempted["FP32_initial_noise_draws"] = self.attempted.get("FP32_initial_noise_draws", 0) + 1
            if self.after_raw_draw is None:
                return self.ledger.call(self.run_id, {"FP32_initial_noise_draws": 1}, original_prepare_latents, seed, latent_shape, dtype=dtype)
            import torch
            original_randn = torch.randn
            def randn(*args, **kwargs):
                if self.raw_draw_calls or list(args) != VIDEO["latent_shape"] or kwargs.get("dtype") != torch.float32 or kwargs.get("generator") is not scheduler.generator:
                    raise ValueError("Unexpected random draw inside original prepare_latents")
                value = original_randn(*args, **kwargs)
                self.raw_draw_calls += 1
                self.after_raw_draw(value)
                return value
            with patch_method(torch, "randn", randn):
                result = self.ledger.call(self.run_id, {"FP32_initial_noise_draws": 1}, original_prepare_latents, seed, latent_shape, dtype=dtype)
            if self.raw_draw_calls != 1:
                raise ValueError("Original raw FP32 noise was not captured exactly once")
            return result
        self.stack.enter_context(patch_method(scheduler, "prepare_latents", prepare_latents))
        def forbid(*args, **kwargs):
            raise RuntimeError("CLIP/streaming/extra generation is outside the fixed six-video scope")
        self.stack.enter_context(patch_method(runner, "run_image_encoder", forbid))
        if hasattr(runner, "run_vae_decoder_stream"):
            self.stack.enter_context(patch_method(runner, "run_vae_decoder_stream", forbid))
        original_pre = scheduler.step_pre
        def step_pre(*args, **kwargs):
            self.attempted["scheduler_step_pre"] = self.attempted.get("scheduler_step_pre", 0) + 1
            value = self.ledger.call(self.run_id, {"scheduler_step_pre": 1}, original_pre, *args, **kwargs)
            self.after_step_pre()
            return value
        self.stack.enter_context(patch_method(scheduler, "step_pre", step_pre))
        self.counted(runner.model, "infer", {"native_full_model_flows": 1})
        original_apply = self.runtime.apply
        def apply(mm, value, *, expert, step_index, block_index, module_name):
            step, block = self.step(), int(block_index)
            target = (step, block, module_name)
            ordinal = len(self.lines)
            expected_target = (ordinal // 400, ordinal % 400 // 10, f"blocks.{ordinal % 400 // 10}.{LINEAR_SUFFIXES[ordinal % 10]}") if ordinal < 1600 else None
            if (self.slot is not None or type(step_index) is not int or step_index != step or block not in range(40)
                    or expert != ("high_noise_model" if step < 2 else "low_noise_model")
                    or module_name not in self.base.records[expert] or not module_name.startswith(f"blocks.{block}.")
                    or target in self.lines or target != expected_target):
                raise ValueError("Unknown, duplicate or interleaved native linear dispatch")
            ffn = module_name.endswith((".ffn.0", ".ffn.2"))
            costs = {"native_FFN_linears": 1,
                     "native_original_width_FFN_linears" if self.mode == "dense" else "native_narrow6016_FFN_linears": 1} if ffn else {"native_non_FFN_linears": 1}
            self.slot = target
            try:
                self.ledger.reserve(self.run_id, costs)
                result = original_apply(mm, value, expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
                if result is None:
                    raise RuntimeError("Native runtime returned fallback instead of registered GEMM")
                self.ledger.observe(costs, self.run_id)
                self.lines.add(target)
                return result
            finally:
                self.slot = None
        self.stack.enter_context(patch_method(self.runtime, "apply", apply))
        original_qdq = self.adapter.native_hif4_activation_qdq
        def qdq(value, *, force_fp32=True):
            if self.slot is None or force_fp32 is not True:
                raise ValueError("Native activation QDQ escaped a registered linear")
            if self.slot in self.qdq_slots:
                raise ValueError("Duplicate activation QDQ slot in actual native linear")
            self.qdq_slots.add(self.slot)
            field = "native_FFN_activation_QDQ" if self.slot[2].endswith((".ffn.0", ".ffn.2")) else "native_non_FFN_activation_QDQ"
            return self.ledger.call(self.run_id, {field: 1}, original_qdq, value, force_fp32=True)
        self.stack.enter_context(patch_method(self.adapter, "native_hif4_activation_qdq", qdq))
        if self.runtime is not self.base:
            self.stack.enter_context(patch_method(self.runtime, "quantizer", qdq))
        for index, model in enumerate(runner.model.model):
            infer, weights = model.transformer_infer, model.transformer_weights
            def required(*args, _index=index, **kwargs):
                self.step(_index)
            for owner, name, field in ((model.pre_infer, "infer", "pre_infer_calls"),
                    (weights.norm, "apply", "final_LayerNorm_calls"), (weights.head, "apply", "head_Default_MM_calls"),
                    (model.post_infer, "infer", "post_unpatchify_calls")):
                self.counted(owner, name, {field: 1}, required)
            for name in ("time_embedding_0", "time_embedding_2", "time_projection_1", "text_embedding_0", "text_embedding_2", "patch_embedding"):
                self.counted(getattr(model.pre_weight, name), "apply",
                    {"patch_Conv3D_calls" if name == "patch_embedding" else "pre_Default_MM_calls": 1}, required)
            for name, field, target in (("infer_block", "native_blocks", self.blocks), ("infer_ffn", "native_FFN_calls", self.ffns)):
                original = getattr(infer, name)
                def call(*args, _original=original, _index=index, _infer=infer, _field=field, _target=target, **kwargs):
                    step, block = self.step(_index), int(_infer.block_idx)
                    if (step, block) != (len(_target) // 40, len(_target) % 40):
                        raise ValueError("Exactly forty native blocks/FFNs per step in order required")
                    value = self.ledger.call(self.run_id, {_field: 1}, _original, *args, **kwargs)
                    _target.append((step, block))
                    return value
                self.stack.enter_context(patch_method(infer, name, call))
            manager = getattr(infer, "offload_manager", None)
            if manager is None or manager.offload_granularity != "block" or manager.lazy_load:
                raise ValueError("Original non-lazy block-offload path required")
            for buffer in manager.cuda_buffers:
                original_load = buffer.load_state_dict
                def load(destination, block_index, adapter_block_index=None, _original=original_load, _index=index):
                    import torch
                    self.step(_index)
                    tensors = {k: v for k, v in destination.items() if torch.is_tensor(v)}
                    if any(v.device.type != "cpu" for v in tensors.values()):
                        raise ValueError("Original block source is not a CPU checkpoint tensor")
                    observation = {"step": self.step(_index), "expert_index": _index, "block": int(block_index),
                        "source_tensor_bytes": sum(v.numel() * v.element_size() for v in tensors.values()),
                        "source_FFN_tensor_bytes": sum(v.numel() * v.element_size() for k, v in tensors.items() if ".ffn." in k),
                        "source_tensor_count": len(tensors), "status": "INVOKED_BEFORE_ORIGINAL_BLOCK_BUFFER_LOAD"}
                    self.checkpoint_buffer_loads.append(observation)
                    value = self.ledger.call(self.run_id, {"original_checkpoint_block_buffer_loads": 1}, _original, destination, block_index, adapter_block_index)
                    observation["status"] = "ORIGINAL_BLOCK_BUFFER_LOAD_RETURNED"
                    return value
                self.stack.enter_context(patch_method(buffer, "load_state_dict", load))

    def report(self):
        return {"native_linear_dispatches": len(self.lines), "native_block_coverage": self.blocks,
                "native_FFN_coverage": self.ffns, "invoked_method_counts": self.attempted,
                "method_dispatch_wall_seconds_including_nested_calls": self.wall,
                "checkpoint_block_buffer_loads": self.checkpoint_buffer_loads,
                "checkpoint_copy_scope": "actual source tensor bytes passed to unchanged full block buffer load; includes original FFN; not a hardware H2D profiler total"}

    def verify_complete(self):
        expected = dict.fromkeys(("input_encoder_requests", "T5_positive_prompt_encodes", "VAE_encodes",
                                  "VAE_decodes", "scheduler_prepare", "FP32_initial_noise_draws"), 1)
        expected.update(dict.fromkeys(("scheduler_step_pre", "native_full_model_flows", "pre_infer_calls",
            "final_LayerNorm_calls", "head_Default_MM_calls", "post_unpatchify_calls", "patch_Conv3D_calls"), 4))
        expected["pre_Default_MM_calls"] = 20
        wanted = {(s, b, name) for s in range(4) for b in range(40)
                  for name in self.base.records["high_noise_model" if s < 2 else "low_noise_model"]
                  if name.startswith(f"blocks.{b}.")}
        checkpoint_expected = [(s, b) for s in range(4) for b in (([0] if s in (0, 2) else []) + list(range(1, 40)) + [0])]
        if (self.attempted != expected or self.lines != wanted or len(wanted) != 1600
                or self.qdq_slots != wanted
                or [(x["step"], x["block"]) for x in self.checkpoint_buffer_loads] != checkpoint_expected
                or any(x["status"] != "ORIGINAL_BLOCK_BUFFER_LOAD_RETURNED" for x in self.checkpoint_buffer_loads)
                or self.blocks != [(s, b) for s in range(4) for b in range(40)] or self.ffns != self.blocks):
            raise ValueError("Per-video actual preparation/native operation coverage differs")

    def __exit__(self, *args):
        self.stack.close()


def validate_native_counts(counts, timings, compact=None):
    expected_sites = {(s, b) for s in range(4) for b in range(40)}
    if (counts != {"linear_forward_calls": 1600, "activation_qdq_calls": 1600, "weight_transfers": 0, "cache_misses": 0}
            or len(timings) != 160 or {(t["step"], t["block"]) for t in timings} != expected_sites
            or any(not math.isfinite(t["elapsed_ms"]) or t["elapsed_ms"] < 0 for t in timings)):
        raise ValueError("Actual full-depth native linear/QDQ/timing coverage differs")
    if compact is not None and (compact["native_counter_delta"] != counts
            or compact["counts"] != {"W1": 160, "W2": 160, "Q1": 160, "Q2": 160,
                "narrow_BF16_GEMM": 320, "online_LoRA_GEMM": 0, "weight_QDQ": 0}
            or len(compact["completed_native_sites"]) != 160 or len(compact["main_coverage"]) != 320
            or compact["original_attention_projection_calls"] != 1280):
        raise ValueError("Actual narrow6016 full160 coverage differs")


def compare_historical(phase, case):
    old = case["historical_dense"]
    states = [item["raw_tensor_sha256"] for item in phase["trajectory"]]
    result = {"actual_initial_BF16_matches": phase["initial_noise"]["raw_tensor_sha256"] == old["initial_raw_BF16_sha256"],
              "post_raw_FP32_match_by_step": [a == b for a, b in zip(states, old["post_raw_FP32_sha256"])],
              "video_bytes_match": phase["video"]["sha256"] == old["video"]["sha256"]}
    result["passed"] = len(states) == 4 and result["actual_initial_BF16_matches"] and all(result["post_raw_FP32_match_by_step"]) and result["video_bytes_match"]
    return result


def verify_noise_pair(candidate, dense, hf_root):
    left, right = bind_file(hf_root, candidate), bind_file(hf_root, dense)
    if (candidate["sha256"] != dense["sha256"] or left.samefile(right) or not filecmp.cmp(left, right, shallow=False)
            or any(candidate[k] != dense[k] for k in ("raw_tensor_sha256", "shape", "dtype"))):
        raise ValueError("Three real generations must have separate byte-identical actual noise files")
    return True


def generate_phase(pipe, adapter, base, runtime, mode, case, out, hf_root, ledger, timer, check, guard, persist,
                   dense=None):
    import torch
    from safetensors.torch import save_file
    from ub1_consumer_runtime import tensor_sha
    from run_ub1_consumer_videos import inspect_video
    run_id = f"compact6016_video_source{case['index']:02d}_seed42_{mode}"
    guard(run_id)
    ledger.reserve(run_id, {"video_generations": 1})
    parent = out / run_id
    parent.mkdir(exist_ok=False)
    phase = {"id": ID, "run_id": run_id, "source_index": case["index"], "source_id": case["source_id"],
        "image_sha256": case["image_sha256"], "prompt_sha256": case["prompt_sha256"], "conditioning_id": case["conditioning_id"],
        "seed": 42, "mode": mode, "status": "STARTED_NOT_ACCEPTED", "trajectory": [],
        "scheduler_generator_reset": True, "source_exposure": case["exposure"], "quality_or_speed_acceptance": False}
    write(parent / "PHASE.json", phase)
    scheduler = pipe.runner.scheduler
    scheduler.generator = None
    # All six measured requests pay the same two original first-buffer loads.
    # This preserves the original buffers/copy implementation and avoids giving
    # later banks a warm checkpoint-buffer advantage over the first Dense job.
    for model in pipe.runner.model.model:
        manager = model.transformer_infer.offload_manager
        ledger.reserve(run_id, {"matched_offload_buffer_resets": 1})
        manager.need_init_first_buffer = True
        ledger.observe({"matched_offload_buffer_resets": 1}, run_id)
    original_post = scheduler.step_post
    before = dict(base.counters)
    timer.events.clear()
    torch.cuda.reset_peak_memory_stats()
    if mode != "dense":
        runtime.begin_request()
    def save_state(value, name, field):
        ledger.reserve(run_id, {field: 1})
        value = value.detach().cpu().contiguous()
        if list(value.shape) != VIDEO["latent_shape"] or not bool(torch.isfinite(value).all()):
            raise ValueError("Actual native state shape or finite values differ")
        check(value.numel() * value.element_size() + (1 << 20))
        path = parent / (name + ".safetensors")
        if path.exists():
            raise FileExistsError("Refusing native trajectory replacement")
        save_file({"latents": value}, str(path))
        result = {"root": "hf", "path": str(path.relative_to(hf_root)), "bytes": path.stat().st_size,
            "sha256": sha256(path), "raw_tensor_sha256": tensor_sha(value), "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch.")}
        ledger.observe({field: 1}, run_id)
        return result
    def after_pre():
        step = int(scheduler.step_index)
        if step == 0:
            if scheduler.latents.dtype != torch.bfloat16 or "initial_noise" in phase:
                raise ValueError("One actual original BF16 inference initial state required")
            phase["initial_noise"] = save_state(scheduler.latents, "initial_noise", "actual_initial_noise_files")
            if mode == "dense":
                if phase["initial_noise"]["raw_tensor_sha256"] != case["historical_dense"]["initial_raw_BF16_sha256"]:
                    raise ValueError("Dense initial noise differs before first native flow")
            else:
                if dense is None or any(dense[k] != phase[k] for k in ("source_id", "image_sha256", "seed", "conditioning_id", "prompt_sha256")):
                    raise ValueError("Candidate requires its actual same-source Dense reference")
                verify_noise_pair(phase["initial_noise"], dense["initial_noise"], hf_root)
                phase["actual_initial_noise_files_byte_equal_to_fresh_dense"] = True
    def after_raw_draw(value):
        if value.dtype != torch.float32 or "raw_FP32_noise" in phase:
            raise ValueError("Exactly one original unconditioned FP32 noise tensor required")
        phase["raw_FP32_noise"] = save_state(value, "raw_noise_FP32", "actual_raw_FP32_noise_files")
        if mode != "dense":
            if dense is None:
                raise ValueError("Candidate raw noise needs fresh Dense")
            verify_noise_pair(phase["raw_FP32_noise"], dense["raw_FP32_noise"], hf_root)
            phase["actual_raw_FP32_noise_byte_equal_to_fresh_dense"] = True
    def post():
        step = int(scheduler.step_index)
        if step != len(phase["trajectory"]) or step not in range(4) or int(scheduler.denoising_step_list[step]) != VIDEO["denoising_steps"][step]:
            raise ValueError("Four original scheduler steps in order required")
        if "initial_noise" not in phase:
            raise ValueError("Actual pre-flow initial state was not captured")
        ledger.reserve(run_id, {"scheduler_step_post": 1})
        value = original_post()
        ledger.observe({"scheduler_step_post": 1}, run_id)
        native_dtype = str(scheduler.latents.dtype)
        stored = save_state(scheduler.latents.detach().float(), f"latent_step{step}", "post_step_FP32_latent_files")
        phase["trajectory"].append({**stored, "step": step, "native_dtype_before_FP32_storage": native_dtype,
            "denoising_step": int(scheduler.denoising_step_list[step]), "sigma": float(scheduler.sigmas[step])})
        write(parent / "PHASE.json", phase)
        persist()
        check()
        return value
    started = time.perf_counter()
    journal_before = ledger.record["journal_dispatch_wall_seconds"]
    tap = NativeCallTap(pipe, adapter, base, runtime, ledger, run_id, mode, persist, after_pre, after_raw_draw)
    try:
        with tap, patch_method(scheduler, "step_post", post), torch.inference_mode():
            pipe.generate(seed=42, image_path=str(confined(hf_root, case["image"]["path"])),
                prompt=case["prompt"], negative_prompt="", save_result_path=str(parent / "output.mp4"))
        torch.cuda.synchronize()
        tap.verify_complete()
        ledger.observe({"video_generations": 1}, run_id)
        if [p["step"] for p in phase["trajectory"]] != list(range(4)):
            raise ValueError("Native free-rollout trajectory incomplete")
        ledger.reserve(run_id, {"FFN_boundary_timer_records": 160})
        timings = timer.report()
        validate_native_counts({k: base.counters[k] - before[k] for k in before}, timings)
        ledger.observe({"FFN_boundary_timer_records": 160}, run_id)
        compact = None
        if mode != "dense":
            ledger.reserve(run_id, {"narrow160_site_complete_reports": 1})
            compact = runtime.end_request(FFN_boundary_timings=timings)
            ledger.observe({"narrow160_site_complete_reports": 1}, run_id)
        counts = {k: base.counters[k] - before[k] for k in before}
        validate_native_counts(counts, timings, compact)
        phase.update(status="PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE", native_counter_delta=counts,
            FFN_boundary_timings=timings, FFN_boundary_seconds=sum(t["elapsed_ms"] for t in timings) / 1000.,
            narrow_runtime=compact, native_actual_call_trace=tap.report(), completed_native_flow_steps=4,
            generation_wall_seconds_including_encoders_decode_save_observers=time.perf_counter() - started,
            operation_journal_dispatch_wall_seconds=ledger.record["journal_dispatch_wall_seconds"] - journal_before,
            timing_scope="matched instrumented native generation/FFN; includes paid call journal and tensor observers, not an uninstrumented deployment benchmark",
            video=inspect_video(parent / "output.mp4", VIDEO), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        if mode == "dense":
            phase["historical_comparison"] = compare_historical(phase, case)
            if not phase["historical_comparison"]["passed"]:
                raise ValueError("Fresh Dense does not reproduce historical initial/four-post/video bytes")
        return phase
    except BaseException as error:
        phase.update(status="FAILED_NATIVE_PHASE_PRESERVED_NO_RETRY", error_type=type(error).__name__, error=str(error),
            completed_native_flow_steps=len(phase["trajectory"]), actual_native_counter_delta={k: base.counters[k] - before[k] for k in before},
            actual_native_call_trace=tap.report(), failed_wall_seconds=time.perf_counter() - started)
        if mode != "dense":
            phase["failed_narrow_runtime"] = runtime.report()
        raise
    finally:
        write(parent / "PHASE.json", phase)


def attach_core(phase, dense, assets, out, ledger, check):
    import cv2
    from ffnds_video_metrics import evaluate, validate
    run_id = phase["run_id"]
    ledger.reserve(run_id, {"core_CPU_video_reports": 1})
    cv2.setNumThreads(8)
    identity = {"source_id": phase["image_sha256"], "seed": 42, "expected_frames": 81,
        "input_prompt_digest": phase["prompt_sha256"], "model_digest": hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest(),
        "protocol_sha256": phase["protocol_sha256"], "run_sha256": phase["generation_receipt"]["sha256"]}
    inputs = {"identity": identity, "cost": {
        "generation_wall_seconds": phase["generation_wall_seconds_including_encoders_decode_save_observers"],
        "ffn_boundary_seconds": phase["FFN_boundary_seconds"], "setup_seconds": phase["mode_setup_seconds"],
        "peak_gpu_memory_mib": phase["peak_cuda_allocated_bytes"] / (1 << 20)}}
    if dense is not None:
        inputs["reference_identity"] = {k: identity[k] for k in ("source_id", "seed", "input_prompt_digest")}
        inputs["reference_identity"]["run_sha256"] = dense["generation_receipt"]["sha256"]
    started = time.perf_counter()
    report = evaluate(Path(phase["video"]["path"]), Path(dense["video"]["path"]) if dense else None, inputs)
    errors = validate(report)
    path = out / run_id / "CORE_METRICS.json"
    write(path, report)
    if errors:
        raise ValueError("Full81-frame core metrics incomplete: " + "; ".join(errors))
    ledger.observe({"core_CPU_video_reports": 1}, run_id)
    phase["core_metrics"] = {"path": str(path.relative_to(out)), "sha256": sha256(path),
        "status": "ALL_81_FRAMES_CORE_METRICS_PASS_QUALITY_PENDING", "CPU_wall_seconds": time.perf_counter() - started}
    check()


def release_cache(base, torch):
    torch.cuda.synchronize()
    if getattr(base, "_compact6016_residency_active", False):
        raise RuntimeError("Narrow residency context was not released")
    base._resident_weights.clear()
    base._resident_key = None
    gc.collect()
    torch.cuda.empty_cache()


def require_hook0(adapter, runtime):
    if adapter._ACTIVE_OBSERVER is not None or adapter._ACTIVE_RUNTIME is not runtime:
        raise RuntimeError("Only the registered runtime, with zero teacher observer, may own native target MM")


def execute(args, prepared):
    plan, roots, manifest, cohort, local, rows, readiness = prepared
    digest = sha256(args.protocol)
    def guard(run_id=None):
        return authorize(plan, digest, json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text()), run_id)
    prior_cpu = validate_prior_cpu(guard(), plan, digest, roots)
    from worker_identity import private_config_path, require_worker
    require_worker()
    if sha256(private_config_path()) not in plan["private_worker_config_sha256"]:
        raise ValueError("Exact private assigned worker identity is not bound")
    out = confined(roots["hf"], plan["output_HF_relative"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ATTEMPT.json").open("x") as stream:
        json.dump({"id": ID, "protocol_sha256": digest, "automatic_retry": False}, stream)
    record = {"schema": "ffnds.compact6016.six_native_videos.v1", "id": ID, "protocol_sha256": digest,
        "status": "RUNNING_FIXED_SIX_VIDEOS_NOT_ACCEPTED", "automatic_retry": False, "CPU_preflight": readiness,
        "phases": [], "modes": [], "quality_or_speed_acceptance": False, "official_quality": "NOT_RUN",
        "human_fullplay_review": "PENDING", "independent_image_clusters": 2, "shared_seed": 42}
    record["prior_independent_CPU_preflight"] = prior_cpu
    persist = lambda: write(out / "RUN.json", record)
    journal = (out / "OPERATIONS.jsonl").open("x", buffering=1)
    def append_event(event):
        journal.write(json.dumps(event, separators=(",", ":")) + "\n")
        journal.flush()
    ledger = Ledger(LIMITS, journal=append_event)
    record["operation_ledger"] = ledger.record
    persist()
    started = time.monotonic()
    monitor = adapter = base = None
    finder = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*args):
        raise TimeoutError("Fixed six-video compute wall budget reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(plan["resource_limits"]["compute_wall_seconds"])
    def check(projected=0):
        if monitor:
            monitor.check()
        resource = plan["resource_limits"]
        used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink())
        if used + projected > resource["new_HF_bytes"] or shutil.disk_usage(out).free - projected < resource["HF_free_min_bytes"]:
            raise RuntimeError("HF output/free-space budget reached")
        available = int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))) * 1024
        if available < resource["host_available_min_bytes"]:
            raise RuntimeError("Host-memory reserve reached")
    try:
        check()
        snapshot = out / "EXECUTED_SOURCES"
        snapshot.mkdir(exist_ok=False)
        for binding in plan["repo_sources"]:
            source = bound(binding, roots)
            destination = snapshot / binding["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            check(source.stat().st_size)
            shutil.copyfile(source, destination)
        shutil.copyfile(args.protocol, snapshot / "protocol.json")
        shutil.copyfile(bound(plan["bank_manifest"], roots), snapshot / "BANK_MANIFEST.json")
        write(snapshot / "registered_video_plan.json", cohort)
        from e0_b_final_block_capture_local_package import CheckedPackageFinder
        finder = CheckedPackageFinder(local, rows)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(local))
        sys.meta_path.insert(0, finder)
        for name in ("HiF4_Sp", "LightX2V", "Wan2.2-I2V-A14B-W4A4"):
            sys.path.insert(0, str(ROOT / ".runtime/code" / name))
        for name in ("HiFloat4", "HiFloat4/hif4_gpu"):
            sys.path.insert(0, str(ROOT / ".runtime/code/HiF4_Sp" / name))
        import torch
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, receipt = reserve_before_assets(torch)
        record["CUDA_reservation"] = receipt
        if (torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0)
                or any(os.environ.get(key) == "1" for key in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"))):
            raise ValueError("One isolated A100 with frozen native arithmetic required")
        torch.set_num_threads(plan["resource_limits"]["CPU_threads"])
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        from compact6016_remaining_runtime import load_bank, Compact6016Residency, Compact6016Runtime
        from ffnds_all_weight_residency import AllWeightResidency
        from run_e0_all_weight_residency import FFNStageTimer
        from historical_source_snapshot import HistoricalSources
        from restored_assets import verified_restored_assets
        from ub1_consumer_runtime import load_original_biases, bytes_of_tensors
        from e0_b_final_block_capture import verify_native_quantizer
        from check_e0_full_depth_interface import REQUIRED_CONFIG
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        history = HistoricalSources()
        setup_start = time.perf_counter()
        kwargs, cache, assets = verified_restored_assets(history)
        if {k: cache[k] for k in ("sha256", "bytes")} != plan["native_cache_identity"]:
            raise ValueError("Original complete D-Q cache identity differs")
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        for key, value in REQUIRED_CONFIG.items():
            if key not in ("scheduler", "attention") and pipe.runner.config.get(key) != value:
                raise ValueError("Frozen full native pipeline branch differs: " + key)
        for key in ("lazy_load", "unload_modules", "use_compile", "compile", "use_stream_vae", "lora_configs", "lora_dynamic_apply"):
            if pipe.runner.config.get(key):
                raise ValueError("Unsupported native branch: " + key)
        for model in pipe.runner.model.model:
            infer = model.transformer_infer
            if (infer.has_post_adapter or infer.seq_p_group is not None or infer.infer_dtype != torch.bfloat16
                    or infer.sensitive_layer_dtype != torch.bfloat16 or len(model.transformer_weights.blocks) != 40):
                raise ValueError("Original BF16 forty-block model required")
        base = adapter.HIF4CacheRuntime(cache["path"])
        import importlib
        record["native_quantizer_setup"] = verify_native_quantizer(importlib.import_module("quant.hif4_backend"), plan)
        biases = load_original_biases({"high_noise_model": kwargs["high_noise_ckpt"], "low_noise_model": kwargs["low_noise_ckpt"]})
        record.update(model_assets=assets, historical_source_verifications=history.receipts,
            shared_pipeline_cache_bias_setup_seconds=time.perf_counter() - setup_start,
            BF16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            actual_BF16_CUDA_reduction_flag_is_inherited=True)
        device = torch.device("cuda", 0)
        physical = plan["physical_cache"]
        dense_bytes = sum(bytes_of_tensors(r["weight_qdq"]) for rows_ in base.records.values() for r in rows_.values())
        attention_bytes = sum(bytes_of_tensors(r["weight_qdq"]) for rows_ in base.records.values() for name, r in rows_.items() if not name.endswith((".ffn.0", ".ffn.2")))
        if dense_bytes != physical["dense_decoded_matrix_bytes"] or attention_bytes != physical["candidate_attention_decoded_bytes"]:
            raise ValueError("Actual decoded original cache geometry/size differs")
        envelope = memory_envelope(physical, plan["resource_limits"])
        largest = envelope["maximum_disjoint_resident_bytes"]
        estimate = envelope["estimated_peak_with_original_buffers_and_safety"]
        if largest > plan["resource_limits"]["resident_cache_max_bytes"] or estimate > torch.cuda.get_device_properties(0).total_memory:
            raise RuntimeError("Original buffers plus disjoint decoded residency exceed A100 headroom")
        record["memory_precheck"] = {"maximum_decoded_residency_bytes": largest, "estimated_original_buffers_plus_residency_bytes": estimate,
            "decoded_original_FFN_candidate_cache_matrices": 0, "original_block_buffers_and_full_FFN_prefetch": "RETAINED_AND_MEASURED_INSIDE_NATIVE_WALL"}
        persist()
        dense = {}
        with torch.inference_mode(), FFNStageTimer(adapter) as timer:
            for mode in MODES:
                guard()
                release_cache(base, torch)
                mode_record = {"mode": mode, "status": "SETUP_STARTED"}
                record["modes"].append(mode_record)
                persist()
                mode_start = time.perf_counter()
                positions = None if mode == "dense" else load_bank(manifest, mode, roots["hf"], base, biases, repo_root=ROOT)
                policy = AllWeightResidency(base, "all_resident_no_flush", plan["resource_limits"]["resident_cache_max_bytes"]) if mode == "dense" else Compact6016Residency(base, positions, device, plan["resource_limits"]["resident_cache_max_bytes"])
                actual_matrix_bytes = dense_bytes if mode == "dense" else attention_bytes + sum(bytes_of_tensors(p[k]) for p in positions.values() for k in ("W1", "W2"))
                auxiliary = 0 if mode == "dense" else sum(bytes_of_tensors(p[k]) for p in positions.values() for k in ("b1", "b2", "smooth1", "smooth2"))
                expected_matrix = physical["dense_decoded_matrix_bytes"] if mode == "dense" else physical["candidate_attention_decoded_bytes"] + physical["candidate_FFN_matrix_bytes_per_init"]
                if actual_matrix_bytes != expected_matrix or auxiliary != (0 if mode == "dense" else physical["candidate_bias_smooth_bytes_per_init"]):
                    raise ValueError("Actual folded-bank physical size differs")
                setup_cost = {"native_decoded_cache_matrix_transfers_setup": 800, "native_decoded_cache_matrix_H2D_bytes_setup": actual_matrix_bytes}
                if mode != "dense":
                    setup_cost.update(narrow_bias_smooth_tensor_transfers_setup=320, narrow_bias_smooth_H2D_bytes_setup=auxiliary)
                ledger.reserve("setup", setup_cost)
                before = dict(base.counters)
                with policy:
                    if mode == "dense":
                        for expert, rows_ in base.records.items():
                            for name, item in rows_.items():
                                base._weight(expert=expert, block_index=int(name.split(".")[1]), module_name=name, record=item, device=device, dtype=torch.bfloat16)
                        torch.cuda.synchronize()
                        runtime = base
                    else:
                        policy.preload()
                        runtime = Compact6016Runtime(base, positions, policy, adapter.native_hif4_activation_qdq, initialization=mode)
                    if {k: base.counters[k] - before[k] for k in before} != {"linear_forward_calls": 0, "activation_qdq_calls": 0, "weight_transfers": 800, "cache_misses": 0}:
                        raise ValueError("Mode setup performed unexpected real model work or transfers")
                    ledger.observe(setup_cost)
                    mode_record.update(status="PRELOADED", setup_seconds_including_bank_load=time.perf_counter() - mode_start, residency_after_preload=policy.snapshot())
                    persist()
                    adapter.activate_lightx2v_hooks(observer=None, runtime=runtime)
                    require_hook0(adapter, runtime)
                    try:
                        for case in plan["cases"]:
                            phase = generate_phase(pipe, adapter, base, runtime, mode, case, out, roots["hf"], ledger, timer, check, guard, persist,
                                dense=dense.get(case["index"]) if mode != "dense" else None)
                            phase.update(protocol_sha256=digest, bank_manifest_sha256=plan["bank_manifest"]["sha256"],
                                mode_setup_seconds=mode_record["setup_seconds_including_bank_load"],
                                shared_setup_seconds=record["shared_pipeline_cache_bias_setup_seconds"],
                                setup_cost_scope="one preload per two-source mode; reported once in RUN, shared across its two videos",
                                actual_original_checkpoint_buffer_prefetch="unchanged full native buffers including bypassed original FFN matrices")
                            if mode == "dense":
                                dense[case["index"]] = phase
                            else:
                                reference = dense[case["index"]]
                                verify_noise_pair(phase["initial_noise"], reference["initial_noise"], roots["hf"])
                                phase["actual_initial_noise_files_byte_equal_to_fresh_dense"] = True
                            receipt_path = out / phase["run_id"] / "GENERATION.json"
                            if receipt_path.exists():
                                raise FileExistsError("Immutable generation receipt already exists")
                            write(receipt_path, phase)
                            phase["generation_receipt"] = {"root": "hf", "path": str(receipt_path.relative_to(roots["hf"])), "bytes": receipt_path.stat().st_size, "sha256": sha256(receipt_path)}
                            attach_core(phase, dense.get(case["index"]) if mode != "dense" else None, assets, out, ledger, check)
                            job = next(j for j in cohort["samples"] if j["run_id"] == phase["run_id"])
                            source = next(s for s in cohort["sources"] if s["source_id"] == job["source_id"])
                            registered = {"schema": "ffnds.registered.native.video.v1", "status": "COMPLETE_NATIVE_GENERATION_ONLY",
                                "job": job, "source": source, "actual_initial_noise_file": str(confined(roots["hf"], phase["initial_noise"]["path"])),
                                "initial_noise_shape": phase["initial_noise"]["shape"], "initial_noise": phase["initial_noise"],
                                "video_path": phase["video"]["path"], "decoded_frames": phase["video"]["frames"],
                                "decoded_fps": phase["video"]["fps"], "decoded_size": phase["video"]["size"],
                                "native_step_trace": phase["trajectory"], "native_ffn_counters": phase["native_counter_delta"],
                                "native_ffn_sites": 160, "model_assets": assets, "scheduler_generator_reset": True,
                                "generation_wall_seconds": phase["generation_wall_seconds_including_encoders_decode_save_observers"],
                                "peak_cuda_allocated_bytes": phase["peak_cuda_allocated_bytes"], "method": mode,
                                "official_quality": "NOT_RUN", "quality_accepted": False, "immutable_generation_receipt": phase["generation_receipt"]}
                            registered_path = out / phase["run_id"] / "RUN.json"
                            write(registered_path, registered)
                            phase["registered_receipt"] = {"path": str(registered_path.relative_to(out)), "sha256": sha256(registered_path)}
                            job_index = next(i for i, item in enumerate(cohort["samples"]) if item["run_id"] == phase["run_id"])
                            (out / f"job{job_index:06d}").symlink_to(phase["run_id"], target_is_directory=True)
                            write(out / phase["run_id"] / "PHASE.json", phase)
                            record["phases"].append(phase)
                            persist()
                            check()
                    finally:
                        adapter.deactivate_lightx2v_hooks()
                    mode_record["residency_before_release"] = policy.snapshot()
                mode_record.update(status="COMPLETE_TWO_SOURCE_MODE_RELEASED", residency_after_release=policy.snapshot())
                del runtime, positions, policy
                release_cache(base, torch)
                persist()
        if not ledger.complete() or [p["run_id"] for p in record["phases"]] != list(RUN_IDS):
            raise ValueError("Exact six-video actual operation ledger is incomplete")
        from ffnds_stage_registered_i2v import collect
        complete_cohort = collect(cohort, out)
        write(out / "VERIFIED_GENERATIONS.json", complete_cohort)
        record["verified_generation_cohort"] = {"path": "VERIFIED_GENERATIONS.json", "sha256": sha256(out / "VERIFIED_GENERATIONS.json"),
            "status": complete_cohort["phase"], "original_quality_evaluation": "NOT_RUN"}
        record.update(status="COMPLETE_MATCHED_SIX_NATIVE_VIDEOS_CORE_METRICS_QUALITY_PENDING", actual_native_counters=dict(base.counters))
    except BaseException as error:
        record.update(status="FAILED_FIXED_SIX_VIDEOS_PRESERVED_NO_RETRY", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        original_error = sys.exc_info()[1]
        cleanup_errors = []
        if base is not None:
            record["actual_native_counters_including_failed_work"] = dict(base.counters)
        if adapter is not None:
            try:
                adapter.deactivate_lightx2v_hooks()
            except BaseException as error:
                cleanup_errors.append(error)
        if monitor is not None:
            try:
                monitor.close()
                record["assigned_GPU_resources"] = monitor.report()
                if record["assigned_GPU_resources"].get("violations"):
                    raise RuntimeError("Assigned GPU monitor recorded a violation")
            except BaseException as error:
                cleanup_errors.append(error)
        if finder is not None:
            try:
                write(out / "LOCAL_PACKAGE_IMPORT.json", {"id": ID, "protocol_sha256": digest,
                    "inventory_sha256": plan["local_transformers"]["inventory_sha256"],
                    "checked_import_origins": finder.origins, "source_or_model_changes": 0})
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                if finder in sys.meta_path:
                    sys.meta_path.remove(finder)
        try:
            journal.close()
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            record.update(status="FAILED_SIX_VIDEO_CLEANUP_OR_GPU_MONITOR_PRESERVED", cleanup_errors=[{"type": type(e).__name__, "message": str(e)} for e in cleanup_errors])
        try:
            record.update(wall_seconds_including_all_setup_and_observers=time.monotonic() - started,
                HF_output_bytes_before_final_RUN=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
                operation_journal={"path": "OPERATIONS.jsonl", "sha256": sha256(out / "OPERATIONS.jsonl")})
            persist()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm)
        if cleanup_errors and original_error is None:
            raise RuntimeError("Six-video cleanup/GPU monitor failed; receipt preserved") from cleanup_errors[0]
        if cleanup_errors and original_error is not None:
            original_error.add_note("Additional cleanup errors preserved in RUN.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, default=Path.home() / ".cache/huggingface")
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--preflight-out", type=Path)
    parser.add_argument("--inspect-draft", action="store_true")
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()
    if args.inspect_draft:
        plan = json.loads(args.protocol.read_text())
        validate_plan(plan)
        print(json.dumps({"id": ID, "status": plan["status"], "assets_finalized": plan["assets_finalized"],
            "bank_binding_status": plan["bank_binding_status"], "allowed_run_ids": list(RUN_IDS), "limits": LIMITS,
            "memory_envelope": memory_envelope(plan["physical_cache"], plan["resource_limits"]), "model_calls": 0, "GPU_calls": 0}, indent=2))
        return
    if args.local_root is None:
        raise ValueError("Actual byte preflight requires the exact local Transformers mirror")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.execute_native:
        if not visible or len(visible.split(",")) != 1:
            raise ValueError("One explicitly assigned GPU required")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        prepared = preflight(args.protocol, args.hf_root, args.local_root)
    finally:
        if args.execute_native:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
    if args.execute_native:
        execute(args, prepared)
    else:
        if args.preflight_out:
            with args.preflight_out.open("x") as stream:
                json.dump(prepared[-1], stream, indent=2, allow_nan=False)
                stream.write("\n")
        else:
            print(json.dumps(prepared[-1], indent=2))


if __name__ == "__main__":
    main()
