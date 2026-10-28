"""Reproduce CAL24's96 Dense steps and add same-row terminal r/u/head labels.

One frozen part contains six original image sources, each with its original
seed42+i and a continuous four-step native chain. Historical post-state SHA
and V1/Y/gate bytes must match before a label is accepted. No video decoding,
full-u storage, student fitting, decomposition or extra head/FFN forwards.
The default invocation is byte-only CPU preflight; it imports no Torch.
"""
import argparse
from contextlib import ExitStack
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

import b_cal24_label_contract as contract
from b_final_block_capture import cpu_copy, patch_method, raw_sha, state_sha, tensor_record, tree_state
from ub1_consumer_runtime import confined, sha256, verify_binding

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_B_CAL24_TERMINAL_ALIGNED_LABELS_20261028"
AUTHORIZATION = "USER_RESUME_OFFLINE_IFC_20261027"
EXPERTS = ("high_noise_model", "low_noise_model")
D, M, ROWS, OUT = contract.D, contract.M, contract.ROWS, 64
GRID, SHAPE = (21, 29, 52), (16, 21, 58, 104)
LINEAR_SUFFIXES = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                   "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2")
PRE_MM_ORDER = ("time_embedding_0", "time_embedding_2", "time_projection_1", "text_embedding_0", "text_embedding_2")
LABEL_PAYLOAD_BYTES = sum(math.prod(shape) * (8 if dtype == "I64" else 2) for dtype, shape in contract.PACK_SCHEMA.values())
RESOURCES = {"GPU_per_part": 1, "global_max_simultaneously_assigned_idle_GPUs": 2,
    "CPU_threads_per_part": 2, "compute_wall_seconds_per_part": 2400,
    "external_wall_seconds_per_part": 2700, "new_HF_bytes_per_part": 512 << 20,
    "new_HF_bytes_total": 2 << 30, "HF_free_min_bytes": 20 << 30,
    "host_available_min_bytes": 128 << 30, "resident_weight_max_bytes": 64 << 30,
    "reference_original_peak_physical_MiB": 21390, "GPU_memory_safety_bytes": 2 << 30}
SOURCE_COST = {"encoder_requests": 1, "noise_preparations": 1, "T5_positive_prompt_encodes": 1,
               "VAE_encodes": 1, "scheduler_prepare": 1, "FP32_initial_noise_draws": 1}
FLOW_COST = {"dense_flows": 1, "native_blocks": 40, "native_ffn_calls": 40,
    "native_ffn_linears": 80, "native_ffn_qdq": 80, "native_non_ffn_linears": 320,
    "native_non_ffn_qdq": 320, "head_calls": 1, "layer_norm_calls": 1,
    "scheduler_pre": 1, "scheduler_post": 1, "pre_infer_calls": 1,
    "pre_Default_MM_calls": 5, "patch_Conv3D_calls": 1, "post_unpatchify_calls": 1,
    "CPU_BF16_gate_mul": 1, "CPU_BF16_gate_add": 1,
    "CPU_BF16_gate_mul_elements": ROWS * D, "CPU_BF16_gate_add_elements": ROWS * D,
    "historical_operand_byte_comparisons": 3, "label_packs": 1, "label_header_validations": 1,
    "CPU_saved_rows_payload_validations": 1, "CPU_saved_tensor_payload_validations": 7,
    "CPU_saved_tensor_payload_bytes_validated": LABEL_PAYLOAD_BYTES}
SETUP_COST = {"setup_decoded_weight_GPU_transfers": 800, "setup_decoded_weight_H2D_bytes": 56203673600,
    "CPU_original_head_view_file_loads": 2, "CPU_original_head_tensor_validations": 6,
    "CPU_head_weight_transpose_copies": 2, "CPU_head_weight_transpose_elements": 655360,
    "original_head_asset_packs": 2}
PART_LIMITS = {**contract.budget(0)["limits"], **{k: v * 6 for k, v in SOURCE_COST.items() if k not in contract.TOTAL},
    **{k: v * 24 for k, v in FLOW_COST.items() if k not in contract.TOTAL}, **SETUP_COST,
    "CLIP_encodes": 0, "full_u_files": 0, "raw_trajectory_tensor_files": 0,
    "head_replays": 0, "sensitivity_probes": 0, "shadow_FFN_calls": 0, "video_generations": 0,
    "weight_quantizations": 0, "decompositions": 0, "official_quality_calls": 0}
REQUIRED_SOURCES = {"tools/e0_b_cal24_terminal_labels.py", "tests/test_e0_b_cal24_terminal_labels.py",
    "tools/b_cal24_label_contract.py", "tests/test_b_cal24_label_contract.py", "tools/b_final_block_capture.py",
    "tools/b_head_asset_export.py", "tools/e0_b_final_block_capture.py",
    "tools/e0_b_final_block_capture_local_package.py", "tools/local_python_package_mirror.py",
    "tools/e0_whole_flow_counterfactual.py", "tools/ffnds_all_weight_residency.py",
    "tools/restored_assets.py", "tools/historical_source_snapshot.py", "tools/worker_identity.py",
    "tools/startup_gpu_reservation.py", "tools/assigned_gpu_resources.py", "tools/check_e0_full_depth_interface.py",
    ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py",
    ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/model.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/distill_model.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/pre_infer.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/offload/transformer_infer.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/post_infer.py",
    ".runtime/code/LightX2V/lightx2v/models/runners/default_runner.py",
    ".runtime/code/LightX2V/lightx2v/models/runners/wan/wan_runner.py",
    ".runtime/code/LightX2V/lightx2v/models/runners/wan/wan_distill_runner.py",
    ".runtime/code/LightX2V/lightx2v/models/schedulers/scheduler.py",
    ".runtime/code/LightX2V/lightx2v/models/schedulers/wan/scheduler.py",
    ".runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py",
    ".runtime/code/LightX2V/lightx2v/common/ops/mm/mm_weight.py",
    ".runtime/code/LightX2V/lightx2v/common/ops/norm/layer_norm_weight.py"}


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def bound(binding, roots):
    if not isinstance(binding, dict) or any(k not in binding for k in ("path", "bytes", "sha256")):
        raise ValueError("NOT_BOUND: exact path/bytes/SHA required")
    return verify_binding(roots[binding.get("root", "repo")], binding)


def run_ids(part=None):
    if part is not None and (type(part) is not int or part not in range(4)):
        raise ValueError("Four fixed six-source parts required")
    indices = range(24) if part is None else range(part * 6, part * 6 + 6)
    return [f"cal24_{i:02d}_step{s}" for i in indices for s in range(4)]


def expected_linear_events(step):
    return [(step, block, f"blocks.{block}.{suffix}") for block in range(40) for suffix in LINEAR_SUFFIXES]


def validate_plan(plan, part, frozen=False):
    run_ids(part)
    if (plan.get("id") != ID or plan.get("automatic_retry") is not False
            or plan.get("allowed_run_ids") != run_ids() or plan.get("limits_per_part") != PART_LIMITS
            or plan.get("limits_total") != {k: v * 4 for k, v in PART_LIMITS.items()}
            or plan.get("resource_limits") != RESOURCES
            or plan.get("parts") != [{"part": p, "source_indices": list(range(p * 6, p * 6 + 6))} for p in range(4)]
            or plan.get("label_tensor_schema") != {k: {"dtype": v[0], "shape": v[1]} for k, v in contract.PACK_SCHEMA.items()}
            or plan.get("source_split", {}).get("training") != list(range(16))
            or plan.get("source_split", {}).get("diagnostic_heldout") != list(range(16, 24))
            or plan.get("source_exposure") != "ALL_24_HISTORICALLY_EXPOSED_NOT_BLIND_CONFIRMATION"
            or plan.get("reuse_historical_post_as_tensor") is not False):
        raise ValueError("Exact CAL24 continuous-four-step label scope/resources differ")
    if frozen and (plan.get("assets_finalized") is not True or plan.get("binding_status") != "BOUND_READY_FOR_REGISTRATION"):
        raise ValueError("NOT_BOUND: reviewed producer assets have not been frozen")
    if len(plan.get("inputs", [])) != 24 or len(plan.get("sources", [])) != 24:
        raise ValueError("Exactly24 old source/INPUT bindings required")
    return True


def authorize(plan, digest, active, part, case_id=None):
    validate_plan(plan, part, frozen=True)
    current = active.get("user_continuation_authorization_20261027", {})
    scopes = [s for s in active.get("experiment_authority", {}).get("current_user_authorized_scopes", []) if s.get("id") == ID]
    if (current.get("id") != AUTHORIZATION or current.get("status") != "ACTIVE_CURRENT_USER_DIRECTED_SESSION"
            or current.get("max_simultaneously_assigned_idle_GPUs") != 2 or len(scopes) != 1):
        raise ValueError("Current two-idle-GPU authority and unique exact scope required")
    scope = scopes[0]
    if (scope.get("protocol_sha256") != digest or scope.get("allowed_run_ids") != run_ids()
            or scope.get("limits_per_part") != PART_LIMITS or scope.get("max_simultaneously_assigned_idle_GPUs") != 2
            or scope.get("execution_status") not in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
            or case_id is not None and case_id not in run_ids(part)):
        raise ValueError("Registered CAL24 part/run/protocol/budget differs")


def validate_cpu_preflight(plan, digest, active, part, hf_root, worker_config_sha256):
    """Bind a prior independent CUDA-hidden CPU process and its real exit0.

    Bindings live in ACTIVE after the protocol is frozen, avoiding a protocol
    hash/receipt cycle. The inline entry preflight is an additional check.
    """
    scopes = [s for s in active.get("experiment_authority", {}).get("current_user_authorized_scopes", []) if s.get("id") == ID]
    entries = [r for r in scopes[0].get("CPU_preflights", []) if r.get("part") == part] if len(scopes) == 1 else []
    if len(entries) != 1:
        raise ValueError("Exactly one independently exited CPU preflight is required for this part")
    roots = {"hf": Path(hf_root).resolve()}
    entry = entries[0]
    report = json.loads(bound({**entry["binding"], "root": "hf"}, roots).read_text())
    exited = json.loads(bound({**entry["launch_exit"], "root": "hf"}, roots).read_text())
    if (report.get("id") != ID or report.get("part") != part or report.get("protocol_sha256") != digest
            or report.get("status") != "PASS_CAL24_LABEL_CPU_ENTRY_NO_TORCH_NO_GPU"
            or report.get("source_bindings_sha256") != state_sha(plan["repo_sources"])
            or report.get("source_bindings_verified") != len(plan["repo_sources"])
            or report.get("expected_case_identity_sha256") != plan["expected_case_identity_sha256"]
            or report.get("source_indices") != list(range(part * 6, part * 6 + 6))
            or report.get("original_operand_files_full_SHA_verified") != 24
            or report.get("Torch_or_CUDA_imported") is not False or report.get("all_CUDA_devices_hidden") is not True
            or report.get("native_model_calls") != 0 or report.get("GPU_calls") != 0
            or worker_config_sha256 not in plan["private_worker_config_sha256"]
            or report.get("worker_config_sha256") != worker_config_sha256
            or exited.get("id") != ID or exited.get("part") != part or exited.get("protocol_sha256") != digest
            or exited.get("launcher_returncode") != 0 or exited.get("process_exited") is not True
            or exited.get("RUN_sha256") != entry["binding"]["sha256"]):
        raise ValueError("Prior independent CPU preflight source/worker/part/protocol/actual exit differs")
    return entry


def safetensors_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length <= 1 << 20:
            raise ValueError("Safetensors header exceeds fixed limit")
        header = json.loads(stream.read(length))
    return header, length


def validate_operand(path):
    header, length = safetensors_header(path)
    expected = {"V1": (ROWS, D), "V2": (ROWS, 13824), "Y": (ROWS, D), "b2": (D,), "gate": (1, D)}
    if set(header) - {"__metadata__"} != set(expected):
        raise ValueError("Original five-tensor CAL24 operand schema differs")
    spans = []
    for name, shape in expected.items():
        item = header[name]
        start, end = item["data_offsets"]
        if item["dtype"] != "BF16" or item["shape"] != list(shape) or end - start != math.prod(shape) * 2:
            raise ValueError("Original CAL24 operand header differs")
        spans.append((start, end))
    position = 0
    for start, end in sorted(spans):
        if start != position:
            raise ValueError("Original operand payload gaps/overlap")
        position = end
    if position + length + 8 != Path(path).stat().st_size:
        raise ValueError("Original operand total bytes differ")
    return {"tensor_shapes": {k: list(v) for k, v in expected.items()}, "payload_bytes": position}


def local_package(plan, roots, local_root):
    from local_python_package_mirror import canonical_sha, inventory
    p = plan["local_transformers"]
    producer = json.loads(bound(p["producer_RUN"], roots).read_text())
    exited = json.loads(bound(p["producer_exit"], roots).read_text())
    if (p["transport"] != "BYTE_IDENTICAL_NODE_LOCAL_TRANSFORMERS_NO_SOURCE_OR_MODEL_CHANGE"
            or p["version"] != "5.6.2" or producer.get("status") != "PASS_LOCAL_MIRROR_IMPORT_NO_MODEL"
            or producer.get("protocol_sha256") != p["producer_protocol_sha256"]
            or producer.get("inventory_sha256") != p["inventory_sha256"] or producer.get("archive_sha256") != p["archive_sha256"]
            or producer.get("cuda_initialized") is not False or producer.get("cuda_device_count") != 0
            or producer.get("GPU_model_calls") != 0 or exited.get("launcher_returncode") != 0
            or exited.get("process_exited") is not True or exited.get("RUN_sha256") != p["producer_RUN"]["sha256"]
            or exited.get("protocol_sha256") != producer["protocol_sha256"]):
        raise ValueError("Successful no-model mirror producer/exit differs")
    root = Path(local_root).resolve(strict=True)
    if hashlib.sha256(str(root).encode()).hexdigest() != p["local_root_path_sha256"]:
        raise ValueError("Private local package path identity differs")
    receipt = json.loads(bound(p["inventory"], roots).read_text())
    rows = receipt["files"]
    if (receipt["archive_sha256"] != p["archive_sha256"] or canonical_sha(rows) != p["inventory_sha256"]
            or inventory(root / "transformers") != rows):
        raise ValueError("Complete exact mirror inventory differs")
    if any(n == "transformers" or n.startswith("transformers.") for n in sys.modules):
        raise ValueError("Transformers imported before its checked local path")
    return root, rows


def validate_head_view(plan, roots):
    view = json.loads(bound(plan["head_view_manifest"], roots).read_text())
    path = bound(plan["head_view_file"], roots)
    producer = json.loads(bound(plan["head_view_RUN"], roots).read_text())
    exited = json.loads(bound(plan["head_view_exit"], roots).read_text())
    if (view.get("status") != "COMPLETE" or producer.get("status") != "COMPLETE"
            or view.get("schema") != "ffnds.b_terminal_head_asset_view.v1"
            or producer.get("manifest_sha256") != plan["head_view_manifest"]["sha256"]
            or exited.get("launcher_returncode") != 0 or exited.get("process_exited") is not True
            or exited.get("RUN_sha256") != plan["head_view_RUN"]["sha256"]):
        raise ValueError("Original CPU head view completion/actual exit differs")
    header, length = safetensors_header(path)
    raw = path.read_bytes()[8 + length:]
    if set(header) - {"__metadata__"} != {t["view_key"] for t in view["tensors"]} or len(view["tensors"]) != 6:
        raise ValueError("Six original head tensor fields required")
    for item in view["tensors"]:
        h = header[item["view_key"]]
        start, end = h["data_offsets"]
        key = {"head.head.weight": "weight_raw_sha256", "head.head.bias": "bias_raw_sha256", "head.modulation": "modulation_raw_sha256"}[item["key"]]
        if (h["dtype"] != item["dtype"] or h["shape"] != item["shape"] or end - start != item["bytes"]
                or hashlib.sha256(raw[start:end]).hexdigest() != item["tensor_sha256"]
                or plan["head_assets"][item["expert"]][key] != item["tensor_sha256"]):
            raise ValueError("Original head view tensor/header/source identity differs")
    return view


def preflight(protocol, hf_root, local_root, part, *, root=ROOT):
    started = time.monotonic()
    before = "torch" in sys.modules
    plan = json.loads(Path(protocol).read_text())
    validate_plan(plan, part)
    roots = {"repo": Path(root).resolve(), "hf": Path(hf_root).resolve()}
    paths = [p["path"] for p in plan["repo_sources"]]
    if len(paths) != len(set(paths)) or not REQUIRED_SOURCES <= set(paths):
        raise ValueError("Required original producer/native source bindings incomplete")
    for item in plan["repo_sources"]:
        bound(item, roots)
    prior = json.loads(bound(plan["cal24_source_binding_protocol"], roots).read_text())
    if any(prior[k] != plan[k] for k in ("inputs", "sources", "source_split")):
        raise ValueError("Frozen old CAL24 cohort/row/source binding differs")
    cases = contract.build_cases(plan, roots["hf"])
    if state_sha(cases) != plan["expected_case_identity_sha256"]:
        raise ValueError("Exact96 case/row/state identities differ")
    selected = [c for c in cases if c["part"] == part]
    operands = []
    for case in selected:
        path = bound({"root": "hf", "path": case["operand_path"], "bytes": case["operand_bytes"], "sha256": case["operand_sha256"]}, roots)
        operands.append({"case_id": case["case_id"], **validate_operand(path)})
    head_view = validate_head_view(plan, roots)
    logical = json.loads(bound(plan["native_logical_model_view"], roots).read_text())
    for parent in head_view["parents"]:
        found = [r for r in logical["files"] if r["filename"] == parent["filename"]]
        if len(found) != 1 or found[0]["sha256"] != parent["existing_verified_file_sha256"]:
            raise ValueError("Head original parent differs from actual native model view")
    pointer_path = roots["repo"] / ".runtime/hif4_rebuilt_cache.json"
    pointer = json.loads(pointer_path.read_text())
    if ({k: pointer[k] for k in ("sha256", "bytes")} != plan["native_cache_identity"]
            or pointer["identity"]["model_view_sha256"] != plan["native_logical_model_view"]["sha256"]):
        raise ValueError("Actual worker cache/model pointer differs")
    versions = {n: importlib.metadata.version(n) for n in plan["runtime_versions"]}
    if versions != plan["runtime_versions"]:
        raise ValueError("Original native package versions differ")
    local, rows = local_package(plan, roots, local_root)
    from worker_identity import private_config_path
    worker_path = private_config_path()
    readiness = {"id": ID, "status": "PASS_CAL24_LABEL_CPU_ENTRY_NO_TORCH_NO_GPU", "part": part,
        "protocol_sha256": sha256(protocol), "cases": selected, "verified_original_operand_headers": operands,
        "source_bindings_sha256": state_sha(plan["repo_sources"]), "source_bindings_verified": len(plan["repo_sources"]),
        "expected_case_identity_sha256": plan["expected_case_identity_sha256"],
        "source_indices": list(range(part * 6, part * 6 + 6)), "original_operand_files_full_SHA_verified": len(operands),
        "worker_config_sha256": sha256(worker_path) if worker_path.is_file() else "NOT_BOUND",
        "versions": versions, "mirror_files": len(rows), "historical_raw_noise_and_conditions": "NOT_SAVED",
        "raw_historical_poststate_tensors": "NOT_SAVED_ONLY_SHA_AVAILABLE",
        "Torch_imported_by_preflight": not before and "torch" in sys.modules,
        "Torch_or_CUDA_imported": "torch" in sys.modules or "torch.cuda" in sys.modules,
        "all_CUDA_devices_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "model_calls": 0, "native_model_calls": 0, "GPU_calls": 0, "seconds": time.monotonic() - started}
    if readiness["Torch_imported_by_preflight"]:
        raise RuntimeError("Byte-only preflight unexpectedly imported Torch")
    return plan, roots, cases, selected, local, rows, readiness


class WorkLedger:
    """Reserve exact source/flow bounds before native dispatch; retain failures."""
    def __init__(self, record, cases, part, persist=lambda: None):
        self.record, self.persist = record, persist
        self.native = contract.Ledger(cases, part)
        record["predebited"] = dict.fromkeys(PART_LIMITS, 0)
        record["observed"] = dict.fromkeys(PART_LIMITS, 0)

    def reserve(self, costs, identity):
        if identity not in run_ids(self.native.part) and identity not in ("setup", *(f"source{i:02d}" for i in range(self.native.part * 6, self.native.part * 6 + 6))):
            raise ValueError("Work outside registered part identity")
        if any(k not in PART_LIMITS or type(v) is not int or v < 0 or self.record["predebited"][k] + v > PART_LIMITS[k] for k, v in costs.items()):
            raise ValueError("Exact bounded label work exceeded")
        for key, value in costs.items():
            self.record["predebited"][key] += value
            if key in self.native.limits:
                self.native.predebit(key, value)
        self.record["last_reserved_operation"] = identity
        self.persist()

    def observe(self, costs):
        if any(k not in PART_LIMITS or type(v) is not int or v < 0 or self.record["observed"][k] + v > self.record["predebited"][k] for k, v in costs.items()):
            raise ValueError("Actual work lacked a predebit")
        for key, value in costs.items():
            self.record["observed"][key] += value
            if key in self.native.limits:
                self.native.observe(key, value)
        self.persist()

    def finish(self):
        if self.record["predebited"] != self.record["observed"] or self.record["observed"] != PART_LIMITS:
            raise ValueError("Fixed part source/flow/setup budget incomplete")
        return self.native.finish()


def label_metadata(case, evidence):
    meta = {k: str(case[k]) for k in contract.META_KEYS}
    meta["rows_sha256"] = contract.rows_sha256(case["rows"])
    for key in contract.EVIDENCE_SHA_KEYS:
        meta[key] = evidence[key]
    for key in contract.EVIDENCE_BOOL_KEYS:
        meta[key] = "true" if evidence[key] is True else "false"
    for key in contract.EVIDENCE_STRIDE_KEYS:
        meta[key] = json.dumps(evidence[key], separators=(",", ":"))
    return meta


def validate_saved_payload(path, case, tensor_hashes):
    """Check actual serialized bytes, including ordered I64 rows, without Torch."""
    validated_header = contract.validate_label_header(path, case)
    header, length = safetensors_header(path)
    expected_rows = struct.pack("<" + "q" * len(case["rows"]), *case["rows"])
    checked = {}
    with Path(path).open("rb") as stream:
        for key in contract.PACK_SCHEMA:
            start, end = header[key]["data_offsets"]
            stream.seek(8 + length + start)
            raw = stream.read(end - start)
            actual_sha = hashlib.sha256(raw).hexdigest()
            if (len(raw) != end - start or actual_sha != tensor_hashes[key]
                    or key == "rows" and raw != expected_rows):
                raise ValueError("Saved label tensor payload differs from accepted rows/memory: " + key)
            checked[key] = actual_sha
    return checked, validated_header


def write_label(path, case, pack, evidence):
    import torch
    from safetensors.torch import save_file
    path = Path(path)
    if path.exists() or set(pack) != set(contract.PACK_SCHEMA):
        raise ValueError("New exact seven-tensor label pack required")
    for key, (dtype, shape) in contract.PACK_SCHEMA.items():
        value = pack[key]
        expected_dtype = torch.int64 if dtype == "I64" else torch.bfloat16
        if value.dtype != expected_dtype or value.device.type != "cpu" or tuple(value.shape) != tuple(shape) or not bool(torch.isfinite(value).all()):
            raise ValueError("Actual label tensor shape/dtype/device/finite differs: " + key)
    if pack["rows"].tolist() != case["rows"]:
        raise ValueError("Captured ordered row coordinates differ from original CAL24")
    temp = path.with_suffix(path.suffix + ".tmp")
    save_file({k: v.contiguous() for k, v in pack.items()}, str(temp), metadata=label_metadata(case, evidence))
    temp.replace(path)
    tensor_records = {k: tensor_record(v) for k, v in pack.items()}
    checked_payload, header = validate_saved_payload(path, case, {k: v["raw_tensor_sha256"] for k, v in tensor_records.items()})
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "header": header,
            "tensors": tensor_records, "serialized_payload_sha256": checked_payload,
            "all_serialized_tensor_bytes_match_memory": True, "saved_rows_payload_matches_case": True}


class Cal24LabelTap:
    """Nonmutating original pre→40 blocks→head→post; copy selected rows only."""
    def __init__(self, runtime, adapter, models, original_heads, load_operand, save_label, check=lambda: None):
        self.runtime, self.adapter, self.models, self.original_heads = runtime, adapter, models, original_heads
        self.load_operand, self.save_label, self.check = load_operand, save_label, check
        self.active, self.slot = None, None
        self.pending = {}
        self.events = {k: [] for k in ("pre", "pre_mm", "patch", "block", "ffn", "linear", "qdq", "norm", "head", "post")}
        self.stack = ExitStack()

    def begin(self, case):
        import torch
        if self.active is not None or self.pending or case["step"] not in range(4) or case["expert"] != EXPERTS[0 if case["step"] < 2 else 1]:
            raise ValueError("Fresh exact CAL24 source/step/expert required")
        contract.rows_sha256(case["rows"])
        self.active = {"case": case, "starts": {k: len(v) for k, v in self.events.items()}}
        self.expected_linears = expected_linear_events(case["step"])
        self.original = self.load_operand(case)
        self.cpu_rng, self.cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        self.last_u = self.head_output = None

    def required(self, index=None):
        if self.active is None or index is not None and index != (0 if self.active["case"]["step"] < 2 else 1):
            raise RuntimeError("Native call outside active CAL24 expert/step")
        return self.active["case"]["step"]

    def selected(self, value, width=None):
        import torch
        width = D if width is None else width
        if value.dtype != torch.bfloat16 or tuple(value.shape) != (M, width):
            raise ValueError("Actual full-M BF16 native operand geometry differs")
        rows = torch.tensor(self.active["case"]["rows"], device=value.device, dtype=torch.int64)
        return cpu_copy(value.index_select(0, rows))

    def add(self, key, value):
        if key in self.pending:
            raise RuntimeError("Duplicate actual terminal tap: " + key)
        self.pending[key] = cpu_copy(value)

    def __enter__(self):
        old_apply, old_qdq = self.runtime.apply, self.adapter.native_hif4_activation_qdq
        def qdq(value, *, force_fp32=True):
            if self.slot is None or force_fp32 is not True:
                raise ValueError("Native QDQ must belong to one registered linear")
            ordinal = len(self.events["qdq"]) - self.active["starts"]["qdq"]
            if ordinal not in range(400) or self.slot != self.expected_linears[ordinal]:
                raise ValueError("Exact original QDQ slot order differs before execution")
            result = old_qdq(value, force_fp32=True)
            self.events["qdq"].append(self.slot)
            if self.slot[1:] == (39, "blocks.39.ffn.0"):
                if result.dtype != value.dtype:
                    raise ValueError("Original Q1 following cast must be an identity")
                self.add("V1", self.selected(result.reshape(-1, D)))
            return result
        def apply(mm, value, *, expert, step_index, block_index, module_name):
            step = self.required()
            if self.slot is not None or step_index != step or expert != self.active["case"]["expert"]:
                raise ValueError("Actual native expert/step/dispatch differs")
            self.slot = (int(step_index), int(block_index), module_name)
            try:
                ordinal = len(self.events["linear"]) - self.active["starts"]["linear"]
                if ordinal not in range(400) or self.slot != self.expected_linears[ordinal]:
                    raise ValueError("Exact original linear module order differs before execution")
                result = old_apply(mm, value, expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
                if result is None:
                    raise ValueError("Original native cache returned fallback")
                self.events["linear"].append(self.slot)
                return result
            finally:
                self.slot = None
        self.stack.enter_context(patch_method(self.runtime, "apply", apply))
        self.stack.enter_context(patch_method(self.adapter, "native_hif4_activation_qdq", qdq))
        try:
            for index, model in enumerate(self.models):
                self.install_model(index, model)
        except BaseException:
            self.stack.close()
            raise
        return self

    def install_model(self, index, model):
        infer, weights = model.transformer_infer, model.transformer_weights
        old_pre, old_block, old_ffn = model.pre_infer.infer, infer.infer_block, infer.infer_ffn
        old_non, old_norm, old_head, old_post = infer.infer_non_blocks, weights.norm.apply, weights.head.apply, model.post_infer.infer
        old_writeback = infer.post_process
        native_non_code = old_non.__func__.__code__
        for name in ("time_embedding_0", "time_embedding_2", "time_projection_1", "text_embedding_0", "text_embedding_2", "patch_embedding"):
            op = getattr(model.pre_weight, name)
            def counted(value, _name=name, _old=op.apply):
                step = self.required(index)
                key = "patch" if _name == "patch_embedding" else "pre_mm"
                ordinal = len(self.events[key]) - self.active["starts"][key]
                names = ("patch_embedding",) if key == "patch" else PRE_MM_ORDER
                if ordinal not in range(len(names)) or _name != names[ordinal]:
                    raise ValueError("Exact original pre-infer operation order differs")
                result = _old(value)
                self.events[key].append((step, _name))
                return result
            self.stack.enter_context(patch_method(op, "apply", counted))
        def pre(*args, **kwargs):
            step = self.required(index)
            result = old_pre(*args, **kwargs)
            if tuple(result.grid_sizes.tuple) != GRID or tuple(result.x.shape) != (M, D):
                raise ValueError("Actual pre-infer native grid differs")
            self.events["pre"].append(step)
            return result
        def block(current, x, pre_out):
            step, block_index = self.required(index), int(infer.block_idx)
            if block_index != len(self.events["block"]) - self.active["starts"]["block"]:
                raise ValueError("Forty native blocks in order required")
            if ((hasattr(current.compute_phases[0], "before_proj") and current.compute_phases[0].before_proj.weight is not None)
                    or hasattr(current.compute_phases[2], "after_proj")):
                raise ValueError("Unexpected native block adapter projection")
            result = old_block(current, x, pre_out)
            self.events["block"].append((step, block_index))
            if block_index == 39:
                self.last_u = result
                self.add("u_rows", self.selected(result))
            return result
        def ffn(phase, x, attn, shift, scale, gate=None):
            step, block_index = self.required(index), int(infer.block_idx)
            if block_index != len(self.events["ffn"]) - self.active["starts"]["ffn"]:
                raise ValueError("One original FFN per block in order required")
            if block_index == 39:
                if gate is None or gate.numel() != D:
                    raise ValueError("Original row-constant e5 required")
                self.add("gate", gate.reshape(1, D))
            result = old_ffn(phase, x, attn, shift, scale, gate)
            self.events["ffn"].append((step, block_index))
            if block_index == 39:
                self.add("Y", self.selected(result))
            return result
        def writeback(x, y, gate, pre_out=None):
            self.required(index)
            if int(infer.block_idx) == 39:
                self.add("r_rows", self.selected(x))
            return old_writeback(x, y, gate, pre_out)
        def non(weights_arg, x, e):
            self.required(index)
            if weights_arg is not weights or x is not self.last_u or e.dim() != 2:
                raise ValueError("Original head must consume actual post-write residual")
            self.active["actual_u_layout"] = {"shape": list(x.shape), "dtype": str(x.dtype), "stride": list(x.stride())}
            self.active["actual_head_modulation"] = tensor_record(weights.head_modulation.tensor)
            if self.active["actual_head_modulation"]["raw_tensor_sha256"] != raw_sha(self.original_heads[index]["head_modulation"]):
                raise ValueError("Actual forward head modulation differs from original CPU head view")
            result = old_non(weights_arg, x, e)
            if result is not self.head_output:
                raise ValueError("Original head output object changed")
            return result
        def norm(value):
            step = self.required(index)
            caller = sys._getframe(1)
            if caller.f_code is not native_non_code or caller.f_locals.get("x") is not self.last_u or value is not self.last_u:
                raise ValueError("Original head normalization caller/object changed")
            e = caller.f_locals.get("e")
            if not isinstance(e, tuple) or len(e) != 2 or any(t.numel() != D for t in e):
                raise ValueError("Actual head shift/scale chunk differs")
            self.active["actual_head_shift_stride"], self.active["actual_head_scale_stride"] = list(e[0].stride()), list(e[1].stride())
            self.add("head_shift", e[0])
            self.add("head_scale", e[1])
            self.events["norm"].append(step)
            return old_norm(value)
        def head(value):
            step = self.required(index)
            if (weights.norm._get_actual_weight() is not None or weights.norm._get_actual_bias() is not None
                    or weights.norm.eps != 1e-6 or weights.head.has_lora_branch or getattr(weights.head, "has_diff", False)):
                raise ValueError("Original unaffined LN and unadapted Default head required")
            actual_weight, actual_bias = weights.head._get_actual_weight(), weights.head._get_actual_bias()
            original = self.original_heads[index]
            weight, bias = tensor_record(actual_weight), tensor_record(actual_bias)
            if (weight["shape"] != [D, OUT] or bias["shape"] != [OUT]
                    or weight["raw_tensor_sha256"] != raw_sha(original["head_weight_runtime_KN"])
                    or bias["raw_tensor_sha256"] != raw_sha(original["head_bias"])):
                raise ValueError("Actual materialized forward head weight/bias differs from CPU view")
            self.active["actual_head_operator"] = {"weight_runtime_KN": weight, "bias": bias,
                "norm_eps": weights.norm.eps, "norm_weight": None, "norm_bias": None,
                "head_class": type(weights.head).__module__ + "." + type(weights.head).__qualname__,
                "norm_class": type(weights.norm).__module__ + "." + type(weights.norm).__qualname__}
            result = old_head(value)
            self.add("head_rows", self.selected(result, OUT))
            self.head_output = result
            self.events["head"].append(step)
            return result
        def post(value, pre_out):
            step = self.required(index)
            if value is not self.head_output:
                raise ValueError("Original unpatchify head object changed")
            result = old_post(value, pre_out)
            self.events["post"].append(step)
            self.last_u = self.head_output = None
            return result
        for owner, name, replacement in ((model.pre_infer, "infer", pre), (infer, "infer_block", block),
                (infer, "infer_ffn", ffn), (infer, "post_process", writeback), (infer, "infer_non_blocks", non),
                (weights.norm, "apply", norm), (weights.head, "apply", head), (model.post_infer, "infer", post)):
            self.stack.enter_context(patch_method(owner, name, replacement))

    def finish(self, actual_post_sha):
        import torch
        state, case = self.active, self.active["case"]
        step = case["step"]
        calls = {k: v[state["starts"][k]:] for k, v in self.events.items()}
        if ({k: len(v) for k, v in calls.items()} != {"pre": 1, "pre_mm": 5, "patch": 1,
                "block": 40, "ffn": 40, "linear": 400, "qdq": 400, "norm": 1, "head": 1, "post": 1}
                or calls["block"] != [(step, b) for b in range(40)] or calls["ffn"] != calls["block"]
                or calls["linear"] != self.expected_linears or calls["qdq"] != self.expected_linears
                or calls["pre_mm"] != [(step, name) for name in PRE_MM_ORDER]
                or calls["patch"] != [(step, "patch_embedding")]
                or sum(name.endswith((".ffn.0", ".ffn.2")) for _, _, name in calls["linear"]) != 80
                or {b: sum(block == b for _, block, _ in calls["linear"]) for b in range(40)} != dict.fromkeys(range(40), 10)):
            raise ValueError("Actual original pre/all40/head/post native graph incomplete")
        if not torch.equal(self.cpu_rng, torch.get_rng_state()) or not torch.equal(self.cuda_rng, torch.cuda.get_rng_state()):
            raise ValueError("Observed native flow changed global RNG state")
        if actual_post_sha != case["historical_post_sha256"]:
            raise ValueError("Historical CAL24 post-state raw FP32 SHA mismatch")
        required = {"V1", "Y", "r_rows", "u_rows", "head_rows", "gate", "head_shift", "head_scale"}
        if set(self.pending) != required or any(t.dtype != torch.bfloat16 or not bool(torch.isfinite(t).all()) for t in self.pending.values()):
            raise ValueError("Complete finite BF16 terminal selected rows required")
        equal = {}
        for key in ("V1", "Y", "gate"):
            left, right = self.pending[key], self.original[key]
            equal[key] = (left.dtype == right.dtype and left.shape == right.shape
                          and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))
            if not equal[key]:
                raise ValueError("Original CAL24 same-row operand bytes differ: " + key)
        reconstructed = self.pending["r_rows"].clone()
        reconstructed.add_(self.pending["Y"] * self.pending["gate"])
        if not torch.equal(reconstructed.view(torch.uint8), self.pending["u_rows"].view(torch.uint8)):
            raise ValueError("CPU BF16 gate multiply then add did not reproduce actual u rows")
        op, modulation = state["actual_head_operator"], state["actual_head_modulation"]
        evidence = {"head_weight_raw_sha256": op["weight_runtime_KN"]["raw_tensor_sha256"],
            "head_bias_raw_sha256": op["bias"]["raw_tensor_sha256"], "head_modulation_raw_sha256": modulation["raw_tensor_sha256"],
            "actual_post_sha256": actual_post_sha, "original_v1_byte_equal": equal["V1"],
            "original_y_byte_equal": equal["Y"], "original_gate_byte_equal": equal["gate"],
            "head_weight_stride": op["weight_runtime_KN"]["stride"], "head_bias_stride": op["bias"]["stride"],
            "head_modulation_stride": modulation["stride"], "head_shift_stride": state["actual_head_shift_stride"],
            "head_scale_stride": state["actual_head_scale_stride"]}
        pack = {k: self.pending[k] for k in ("r_rows", "u_rows", "head_rows", "gate", "head_shift", "head_scale")}
        pack["rows"] = torch.tensor(case["rows"], dtype=torch.int64)
        saved = self.save_label(case, pack, evidence)
        result = {"case": case, "label": saved, "evidence": evidence, "actual_head_operator": op,
            "actual_head_modulation": modulation, "actual_u_layout": state["actual_u_layout"],
            "native_boundary_counts": {k: len(v) for k, v in calls.items()},
            "native_noop": {"wrappers_return_original_objects": True, "global_CPU_CUDA_RNG_unchanged": True,
                "extra_FFN_head_norm_QDQ_forwards": 0, "CPU_BF16_gate_mul_then_add_byte_equal": True},
            "historical_raw_initial_noise_and_conditions": "NOT_SAVED_ONLY_SOURCE_RECONSTRUCTION_AND_POST_SHA_CHECKS"}
        self.pending.clear()
        self.original = None
        self.active = self.last_u = self.head_output = None
        return result

    def __exit__(self, *error):
        self.stack.close()


class ContinuousSourceSegment:
    """Original step_pre→wrapper.infer→step_post, no historical-state injection."""
    def __init__(self, runner, tap, cases, source_actual, ledger, guard, record, persist, check,
                 save_case=lambda case: None):
        self.runner, self.tap, self.cases, self.source_actual = runner, tap, cases, source_actual
        self.ledger, self.guard, self.record, self.persist, self.check = ledger, guard, record, persist, check
        self.completed = []
        self.save_case = save_case

    def run_segment(self, segment_idx=0):
        import torch
        scheduler = self.runner.scheduler
        if (segment_idx != 0 or type(scheduler).__name__ != "Wan22StepDistillScheduler"
                or scheduler._uses_conditioned_latent_prefix() or scheduler.keep_latents_dtype_in_scheduler
                or not torch.equal(scheduler.sigmas, torch.tensor([1., .9375, .8333333134651184, .625]))
                or self.source_actual != SOURCE_COST):
            raise ValueError("Exact original source preparation and continuous four-step scheduler required")
        index = self.cases[0]["source_index"]
        self.record["actual_preparation_calls"] = dict(self.source_actual)
        self.record["prepared_FP32_noise"] = tensor_record(scheduler.latents)
        self.record["condition_state"] = tree_state(self.runner.inputs)
        self.record["condition_state_sha256"] = state_sha(self.record["condition_state"])
        self.ledger.observe(SOURCE_COST)
        self.ledger.native.mark_source(index)
        self.persist()
        for case in self.cases:
            step = case["step"]
            self.guard(case["case_id"])
            self.check()
            self.ledger.reserve(FLOW_COST, case["case_id"])
            self.record["current_attempted_case"] = case["case_id"]
            self.persist()
            scheduler.step_pre(step_index=step)
            if scheduler.latents.dtype != torch.bfloat16 or tuple(scheduler.latents.shape) != SHAPE:
                raise ValueError("Native original step_pre BF16 state geometry differs")
            pre = tensor_record(scheduler.latents)
            self.tap.begin(case)
            before = dict(self.tap.runtime.counters)
            started = time.perf_counter()
            self.runner.model.infer(self.runner.inputs)
            if self.runner.model.cur_model_index != (0 if step < 2 else 1):
                raise ValueError("Original wrapper selected incorrect native expert")
            if scheduler.noise_pred.dtype != torch.float32 or tuple(scheduler.noise_pred.shape) != SHAPE:
                raise ValueError("Actual native post-unpatchify FP32 flow geometry differs")
            scheduler.step_post()
            native_post = cpu_copy(scheduler.latents)
            if native_post.dtype != torch.bfloat16 or tuple(native_post.shape) != SHAPE:
                raise ValueError("Original native post-state BF16 geometry differs")
            actual_post_sha = raw_sha(native_post.float())
            result = self.tap.finish(actual_post_sha)
            torch.cuda.synchronize()
            delta = {k: self.tap.runtime.counters[k] - v for k, v in before.items()}
            if delta != {"linear_forward_calls": 400, "activation_qdq_calls": 400, "weight_transfers": 0, "cache_misses": 0}:
                raise ValueError("Actual complete native linear/QDQ graph differs")
            result.update(schema="ffnds.b_cal24.terminal_label_case.v1", status="PASS_CAL24_POST_AND_OPERAND_BYTES_ALIGNED",
                actual_pre_state=pre, actual_native_post_dtype=str(native_post.dtype), actual_post_raw_FP32_sha256=actual_post_sha,
                native_counter_delta=delta, wall_seconds_with_capture_hash_and_CPU_gate_check=time.perf_counter() - started)
            self.completed.append(result)
            self.save_case(result)
            self.record["cases"] = self.completed
            self.ledger.observe(FLOW_COST)
            self.ledger.native.mark_case(case)
            self.persist()
            self.check()
        if [r["case"]["step"] for r in self.completed] != list(range(4)):
            raise ValueError("Four continuous native state-matched steps incomplete")
        del self.runner.inputs
        return scheduler.latents


def source_preparation(stack, runner, source, actual):
    """Observe original encoder/prepare calls; each expected request exactly once."""
    import torch
    scheduler = runner.scheduler
    for owner, name, field in ((runner, "run_input_encoder", "encoder_requests"),
            (runner, "run_text_encoder", "T5_positive_prompt_encodes"), (runner, "run_vae_encoder", "VAE_encodes"),
            (scheduler, "prepare", "scheduler_prepare")):
        original = getattr(owner, name)
        def counted(*args, _original=original, _field=field, **kwargs):
            if actual[_field] != 0:
                raise ValueError("Duplicate original source preparation: " + _field)
            actual[_field] += 1
            return _original(*args, **kwargs)
        stack.enter_context(patch_method(owner, name, counted))
    original_noise = scheduler.prepare_latents
    def noise(seed, latent_shape, dtype=torch.float32):
        if (seed != source["seed"] or tuple(latent_shape) != SHAPE or dtype != torch.float32
                or scheduler.generator is not None or actual["FP32_initial_noise_draws"] != 0):
            raise ValueError("Fresh original FP32 generator reset and source-specific seed required")
        actual["FP32_initial_noise_draws"] += 1
        actual["noise_preparations"] += 1
        result = original_noise(seed, latent_shape, dtype=dtype)
        if scheduler.latents.dtype != torch.float32 or tuple(scheduler.latents.shape) != SHAPE:
            raise ValueError("Actual prepared FP32 noise geometry differs")
        return result
    stack.enter_context(patch_method(scheduler, "prepare_latents", noise))
    def forbid(*args, **kwargs):
        raise RuntimeError("Decode, CLIP and streaming are outside this label-only scope")
    for name in ("run_image_encoder", "run_vae_decoder", "run_vae_decoder_stream"):
        if hasattr(runner, name):
            stack.enter_context(patch_method(runner, name, forbid))


def execute(args, prepared):
    plan, roots, cases, chosen, local, rows, readiness = prepared
    digest = sha256(args.protocol)
    def guard(case_id=None):
        authorize(plan, digest, json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text()), args.part, case_id)
    guard()
    from worker_identity import private_config_path, require_worker
    require_worker()
    worker_sha = sha256(private_config_path())
    if worker_sha not in plan["private_worker_config_sha256"]:
        raise ValueError("Actual private worker profile has not been frozen")
    cpu_gate = validate_cpu_preflight(plan, digest, json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text()),
        args.part, roots["hf"], worker_sha)
    out = confined(roots["hf"], plan["output_HF_relative"]) / f"part{args.part}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ATTEMPT.json").open("x") as stream:
        json.dump({"id": ID, "part": args.part, "protocol_sha256": digest, "automatic_retry": False}, stream)
    record = {"schema": "ffnds.b_cal24.terminal_label_run.v1", "id": ID, "part": args.part,
        "protocol_sha256": digest, "status": "RUNNING_FIXED_CAL24_TERMINAL_LABELS_NOT_ACCEPTED",
        "automatic_retry": False, "CPU_preflight": readiness, "sources": [], "label_only": True,
        "prior_independently_exited_CPU_preflight": cpu_gate,
        "independent_source_images_in_part": 6, "historical_exposure": plan["source_exposure"],
        "quality_or_speed_acceptance": False, "count_semantics": "source and complete-flow upper bounds debited before dispatch; successful actual graph counts verified; failed native counters retained separately"}
    persist = lambda: write(out / "RUN.json", record)
    ledger = WorkLedger(record, cases, args.part, persist)
    persist()
    started = time.monotonic()
    monitor = adapter = runtime = finder = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError("Fixed six-source label wall budget reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(RESOURCES["compute_wall_seconds_per_part"])
    def check(projected=0):
        if monitor:
            monitor.check()
        used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if used + projected > RESOURCES["new_HF_bytes_per_part"] or shutil.disk_usage(out).free - projected < RESOURCES["HF_free_min_bytes"]:
            raise RuntimeError("Bounded output/free-HF-space budget reached")
        available = int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))) * 1024
        if available < RESOURCES["host_available_min_bytes"]:
            raise RuntimeError("Host-memory reserve reached")
    try:
        check()
        snapshot = out / "EXECUTED_SOURCES"
        snapshot.mkdir(exist_ok=False)
        for item in plan["repo_sources"]:
            source = bound(item, roots)
            target = snapshot / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            check(source.stat().st_size)
            shutil.copyfile(source, target)
        shutil.copyfile(args.protocol, snapshot / "protocol.json")
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
        from safetensors import safe_open
        from safetensors.torch import save_file
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, receipt = reserve_before_assets(torch)
        record["CUDA_reservation"] = receipt
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0) or os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
            raise ValueError("One isolated A100 with original arithmetic required")
        torch.set_num_threads(RESOURCES["CPU_threads_per_part"])
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        from historical_source_snapshot import HistoricalSources
        from restored_assets import verified_restored_assets
        from ffnds_all_weight_residency import AllWeightResidency
        from e0_whole_flow_counterfactual import LatentOnlyMain
        from e0_b_final_block_capture import verify_native_quantizer
        from check_e0_full_depth_interface import REQUIRED_CONFIG
        from b_head_asset_export import export
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        import importlib
        history = HistoricalSources()
        kwargs, cache, assets = verified_restored_assets(history)
        if {k: cache[k] for k in ("sha256", "bytes")} != plan["native_cache_identity"]:
            raise ValueError("Exact original full D-Q cache differs")
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        for key, value in REQUIRED_CONFIG.items():
            if key not in ("scheduler", "attention") and pipe.runner.config.get(key) != value:
                raise ValueError("Original native configuration differs: " + key)
        for key in ("lazy_load", "unload_modules", "use_compile", "compile", "use_stream_vae", "lora_configs", "lora_dynamic_apply"):
            if pipe.runner.config.get(key):
                raise ValueError("Unsupported native label branch: " + key)
        for model in pipe.runner.model.model:
            infer = model.transformer_infer
            if (infer.has_post_adapter or infer.seq_p_group is not None or infer.infer_dtype != torch.bfloat16
                    or infer.sensitive_layer_dtype != torch.bfloat16 or len(model.transformer_weights.blocks) != 40):
                raise ValueError("Original unadapted BF16 forty-block model required")
        runtime = adapter.HIF4CacheRuntime(cache["path"])
        record["native_quantizer_setup"] = verify_native_quantizer(importlib.import_module("quant.hif4_backend"), plan)
        if any(runtime.counters.values()):
            raise ValueError("Quantizer resolution performed unexpected model work")
        record.update(model_assets=assets, historical_source_verifications=history.receipts,
            BF16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        adapter.activate_lightx2v_hooks(runtime=runtime)
        # The original GPU head is materialized only inside model.infer. Startup
        # exports use the bound CPU view; they never force an extra head prefetch.
        ledger.reserve(SETUP_COST, "setup")
        head_path = bound(plan["head_view_file"], roots)
        original_heads = []
        record["CPU_head_export_attempts"] = []
        for expert in EXPERTS:
            attempt = {"expert": expert, "status": "CPU_VIEW_EXPORT_INVOKED"}
            record["CPU_head_export_attempts"].append(attempt)
            persist()
            original_heads.append(export(head_path, plan["head_view_file"]["sha256"], expert, plan["head_assets"][expert]))
            attempt["status"] = "CPU_VIEW_EXPORT_RETURNED_THREE_TENSORS_VALIDATED"
        record["original_head_assets"] = []
        for expert, tensors in zip(EXPERTS, original_heads):
            folder = out / expert
            folder.mkdir()
            path = folder / "original_head.safetensors"
            check(sum(t.numel() * t.element_size() for t in tensors.values()) + (1 << 20))
            save_file(tensors, str(path))
            record["original_head_assets"].append({"expert": expert, "path": str(path.relative_to(out)),
                "bytes": path.stat().st_size, "sha256": sha256(path), "tensors": {k: tensor_record(v) for k, v in tensors.items()},
                "actual_runtime_layout_checked_only_inside_forward": True})
        def load_operand(case):
            path = bound({"root": "hf", "path": case["operand_path"], "bytes": case["operand_bytes"], "sha256": case["operand_sha256"]}, roots)
            validate_operand(path)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                return {k: handle.get_tensor(k).contiguous() for k in ("V1", "Y", "gate")}
        def save_label(case, pack, evidence):
            folder = out / case["case_id"]
            folder.mkdir(exist_ok=False)
            check(sum(t.numel() * t.element_size() for t in pack.values()) + (1 << 20))
            result = write_label(folder / "labels.safetensors", case, pack, evidence)
            result["path"] = str((folder / result["path"]).relative_to(out))
            return result
        with torch.inference_mode(), AllWeightResidency(runtime, "all_resident_no_flush", RESOURCES["resident_weight_max_bytes"]) as policy:
            total = sum(v["weight_qdq"].numel() * v["weight_qdq"].element_size() for rows_ in runtime.records.values() for v in rows_.values())
            estimate = total + RESOURCES["reference_original_peak_physical_MiB"] * (1 << 20) + RESOURCES["GPU_memory_safety_bytes"]
            if total != SETUP_COST["setup_decoded_weight_H2D_bytes"] or estimate > torch.cuda.get_device_properties(0).total_memory:
                raise RuntimeError("Full decoded residency plus original buffers exceed A100 headroom")
            for expert, rows_ in runtime.records.items():
                for name, item in rows_.items():
                    runtime._weight(expert=expert, block_index=int(name.split(".")[1]), module_name=name,
                                    record=item, device=torch.device("cuda", 0), dtype=torch.bfloat16)
            if runtime.counters != {"linear_forward_calls": 0, "activation_qdq_calls": 0, "weight_transfers": 800, "cache_misses": 0}:
                raise ValueError("Setup weight transfers or unintended native work differs")
            ledger.observe(SETUP_COST)
            persist()
            with Cal24LabelTap(runtime, adapter, pipe.runner.model.model, original_heads, load_operand, save_label, check) as tap:
                for index in range(args.part * 6, args.part * 6 + 6):
                    source = next(s for s in plan["sources"] if s["index"] == index)
                    input_binding = next(s for s in plan["inputs"] if s["index"] == index)
                    original_input = json.loads(verify_binding(roots["hf"], input_binding).read_text())["input"]
                    actual = dict.fromkeys(SOURCE_COST, 0)
                    source_record = {"source_index": index, "seed": source["seed"], "source_id": source["source_id"],
                        "semantic_source_id": source["semantic_source_id"], "input_sha256": input_binding["sha256"], "cases": [],
                        "actual_preparation_invocations_including_failed_calls": actual}
                    record["sources"].append(source_record)
                    ledger.reserve(SOURCE_COST, f"source{index:02d}")
                    pipe.runner.scheduler.generator = None
                    with ExitStack() as stack:
                        source_preparation(stack, pipe.runner, source, actual)
                        segment = ContinuousSourceSegment(pipe.runner, tap, [c for c in chosen if c["source_index"] == index],
                            actual, ledger, guard, source_record, persist, check,
                            save_case=lambda item: write(out / item["case"]["case_id"] / "CASE.json", item))
                        stack.enter_context(patch_method(pipe.runner, "run_segment", segment.run_segment))
                        stack.enter_context(LatentOnlyMain(pipe.runner))
                        image = contract._inside(roots["hf"], original_input["img_paths"][0])
                        pipe.generate(seed=source["seed"], image_path=str(image),
                            prompt=original_input["prompt"], negative_prompt="", save_result_path=str(out / "NO_VIDEO_WILL_BE_WRITTEN.mp4"))
                    if actual != SOURCE_COST or len(segment.completed) != 4:
                        raise ValueError("Actual original source encoder/noise/four-step graph incomplete")
                    source_record["status"] = "COMPLETE_FOUR_CONTINUOUS_STEPS_CAL24_BYTES_ALIGNED"
                    persist()
                    check()
            record["contract_case_ledger"] = ledger.finish()
            record["producer_setup_binding"] = {k: record["observed"][k] for k in SETUP_COST}
            record["residency_policy"] = policy.snapshot()
            expected = {"linear_forward_calls": 9600, "activation_qdq_calls": 9600, "weight_transfers": 800, "cache_misses": 0}
            if runtime.counters != expected:
                raise ValueError("Whole-part actual native counters differ")
            record.update(status="COMPLETE_CAL24_TERMINAL_LABEL_PART_POST_AND_OPERAND_BYTES_MATCH", actual_native_counters=dict(runtime.counters))
    except BaseException as error:
        record.update(status="FAILED_FIXED_CAL24_LABEL_PART_NO_AUTOMATIC_RETRY", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if "tap" in locals():
            record["failed_active_case"] = tap.active
            record["failed_native_tap_event_counts"] = {key: len(values) for key, values in tap.events.items()}
            record["failed_pending_capture_names"] = sorted(tap.pending)
        raise
    finally:
        original_error = sys.exc_info()[1]
        cleanup = []
        if runtime is not None:
            record["actual_native_counters_including_failed_work"] = dict(runtime.counters)
        if adapter is not None:
            try:
                adapter.deactivate_lightx2v_hooks()
            except BaseException as error:
                cleanup.append(error)
        if monitor is not None:
            try:
                monitor.close()
                record["assigned_GPU_resources"] = monitor.report()
                if record["assigned_GPU_resources"].get("violations"):
                    raise RuntimeError("Assigned GPU monitor recorded violations")
            except BaseException as error:
                cleanup.append(error)
        if finder is not None:
            try:
                write(out / "LOCAL_PACKAGE_IMPORT.json", {"id": ID, "part": args.part,
                    "protocol_sha256": digest, "inventory_sha256": plan["local_transformers"]["inventory_sha256"],
                    "checked_import_origins": finder.origins, "source_or_model_changes": 0})
            except BaseException as error:
                cleanup.append(error)
            finally:
                if finder in sys.meta_path:
                    sys.meta_path.remove(finder)
        if cleanup:
            record.update(status="FAILED_CAL24_LABEL_CLEANUP_OR_GPU_MONITOR_PRESERVED",
                          cleanup_errors=[{"type": type(e).__name__, "message": str(e)} for e in cleanup])
        try:
            record.update(wall_seconds_including_setup=time.monotonic() - started,
                HF_output_bytes_before_final_RUN=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()))
            persist()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm)
        if cleanup and original_error is None:
            raise RuntimeError("CAL24 cleanup/monitor failed; receipt retained") from cleanup[0]
        if cleanup and original_error is not None:
            original_error.add_note("Additional cleanup failures preserved in RUN.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, default=Path.home() / ".cache/huggingface")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--part", type=int, choices=range(4), required=True)
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()
    prepared = preflight(args.protocol, args.hf_root, args.local_root, args.part)
    if args.execute_native:
        execute(args, prepared)
    else:
        print(json.dumps(prepared[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
