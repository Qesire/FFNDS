"""Two original-BF16 Wan four-step videos; stdlib draft/preflight, no HiF4 hook."""
import argparse
import copy
import filecmp
import hashlib
import importlib.metadata
import importlib.abc
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import struct
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

from b_final_block_capture import patch_method
from e0_compact6016_six_videos import (original_rms_auxiliaries, checkpoint_source_metadata,
    verify_local_package, verify_noise_pair, VIDEO, LINEAR_SUFFIXES)

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_ORIGINAL_BF16_CONSUMER00_12_VIDEOS_20261028"
PARENT = "config/e0_compact6016_six_videos_20261028_v2.json"
PARENT_RUN_SHA = "23a34c360ec07df9644a7a8491edc77f4b8164f837aa71e8af78a06a74517a3a"
RUN_IDS = tuple(f"original_bf16_source{s:02d}_seed42_dense" for s in (0, 12))
FULL_READ_BYTES = 69_025_362_058
RESOURCE = {"GPU": 1, "CPU_threads": 4, "host_available_min_bytes": 128 << 30,
            "GPU_reserved_max_bytes": 64 << 30, "wall_seconds": 1200, "outer_seconds": 1500,
            "new_HF_bytes": 1 << 30, "free_disk_min_bytes": 20 << 30,
            "session_max_GPUs": 2, "repository_GPU_policy": "DYNAMIC_MAX_ACTUALLY_CALLABLE", "automatic_retry": False}
CPU_RESOURCE = {"GPU": 0, "CPU_threads": 4, "hash_streams": 2, "full_model_payload_bytes": FULL_READ_BYTES,
                "model_metadata_bytes": 21_454_628,
                "wall_seconds": 1200, "outer_seconds": 1500, "automatic_retry": False}
LIMITS = {"original_expert_loads": 2, "raw_Default_MM_loads": 812, "buffer_Default_MM_loads": 40,
          "T5_raw_Default_MM_loads": 168, "T5_buffer_Default_MM_loads": 7,
          "T5_Default_MM_calls": 336, "T5_checkpoint_block_buffer_loads": 48,
          "video_generations": 2, "input_encoder_requests": 2, "T5_positive_prompt_encodes": 2,
          "VAE_encodes": 2, "VAE_decodes": 2, "scheduler_prepare": 2, "FP32_initial_noise_draws": 2,
          "scheduler_step_pre": 8, "scheduler_step_post": 8, "native_full_model_flows": 8,
          "native_blocks": 320, "native_FFN_calls": 320, "native_FFN_linears": 640,
          "FFN_CUDA_event_pairs": 320,
          "native_non_FFN_linears": 2560, "pre_Default_MM_calls": 40, "head_Default_MM_calls": 8,
          "patch_Conv3D_calls": 8, "pre_infer_calls": 8, "final_LayerNorm_calls": 8, "post_unpatchify_calls": 8,
          "original_checkpoint_block_buffer_loads": 324, "offload_buffer_resets": 4,
          "raw_FP32_noise_files": 2, "initial_BF16_noise_files": 2, "native_FP32_flow_files": 8,
          "native_BF16_post_files": 8, "core_CPU_reports": 2,
          "activation_QDQ": 0, "weight_quantizations": 0, "quantized_expert_loads": 0,
          "quantized_MM_constructions": 0, "CLIP_encodes": 0, "training_updates": 0,
          "decompositions": 0, "official_quality_calls": 0, "extra_teacher_flows": 0}
PRE_NAMES = ("time_embedding_0", "time_embedding_2", "time_projection_1", "text_embedding_0", "text_embedding_2")
T5_NAMES = ("attn.q", "attn.k", "attn.v", "attn.o", "ffn.fc1", "ffn.gate.0", "ffn.fc2")
CONTRACT = ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json"
TOOLS = {"tools/e0_original_bf16_videos.py", "tests/test_e0_original_bf16_videos.py",
    "tools/e0_compact6016_six_videos.py", "tools/b_final_block_capture.py", "tools/ub1_consumer_runtime.py",
    "tools/e0_b_final_block_capture_local_package.py", "tools/local_python_package_mirror.py",
    "tools/run_ub1_consumer_videos.py", "tools/ffnds_video_metrics.py", "tools/ffnds_cohort_preflight.py",
    "tools/ffnds_experiment_plans.py", "tools/ffnds_stage_registered_i2v.py", "tools/worker_identity.py",
    "tools/startup_gpu_reservation.py", "tools/assigned_gpu_resources.py", CONTRACT,
    ".runtime/model_view.json", ".runtime/node_local_model_view.json"}


def need(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def canonical(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def binding(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    need(path.is_relative_to(root), "Binding escaped root")
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path)}


def bound(root, item):
    root = Path(root).resolve(); path = (root / item["path"]).resolve()
    need(path.is_relative_to(root) and path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Bound input bytes/SHA differ: " + item["path"])
    return path


def stat_record(path):
    x = Path(path).stat()
    return {"device": x.st_dev, "inode": x.st_ino, "bytes": x.st_size, "mtime_ns": x.st_mtime_ns, "ctime_ns": x.st_ctime_ns}


def model_metadata(plan):
    originals = {x["filename"]: x for x in read(ROOT / ".runtime/model_view.json")["files"]
                 if x["filename"].endswith(("config.json", "configuration.json")) or x["filename"].startswith("google/")}
    items = plan["model_view_metadata"]
    need(len(items) == len(originals) and {x["filename"] for x in items} == set(originals)
         and sum(x["bytes"] for x in items) == CPU_RESOURCE["model_metadata_bytes"], "Original tokenizer/config metadata closure differs")
    for x in items:
        original = originals[x["filename"]]
        need(all(x[k] == original[k] for k in ("bytes", "sha256", "repo", "revision")), "Original metadata identity differs")
        bound(Path(plan["model_view_root"]), {"path": x["filename"], "bytes": x["bytes"], "sha256": x["sha256"]})


def header(path):
    """Read only the safetensors JSON header, never tensor payloads."""
    with Path(path).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; need(0 < n <= 4 << 20, "Raw checkpoint header length differs")
        raw = f.read(n)
    rows = {k: v for k, v in json.loads(raw).items() if k != "__metadata__"}
    need(rows and all(v["dtype"] == "BF16" for v in rows.values()), "Original BF16 checkpoint required")
    offsets = sorted((v["data_offsets"][0], v["data_offsets"][1]) for v in rows.values())
    cursor = 0
    for start, end in offsets:
        need(start == cursor and end >= start, "Raw payload offsets overlap or have gaps"); cursor = end
    need(cursor + 8 + n == Path(path).stat().st_size
         and all(v["data_offsets"][1] - v["data_offsets"][0] == 2 * math.prod(v["shape"]) for v in rows.values()), "Raw payload geometry differs")
    return {"header_sha256": hashlib.sha256(raw).hexdigest(), "header_bytes": n, "tensor_count": len(rows), "rows": rows}


def raw_header_check(value):
    rows = value["rows"]
    need(len(rows) == 1095 and not any(any(t in k.lower() for t in ("scale", "smooth", "quant", "lora")) for k in rows), "Original unquantized DiT schema differs")
    for b in range(40):
        need(rows[f"blocks.{b}.ffn.0.weight"]["shape"] == [13824, 5120]
             and rows[f"blocks.{b}.ffn.2.weight"]["shape"] == [5120, 13824], "Raw full-width FFN geometry differs")
    return value


def recipe():
    c = read(ROOT / CONTRACT)
    ignored = {"schema_version", "artifact", "attention", "scheduler", "high_noise_filename", "low_noise_filename", "high_noise_sha256", "low_noise_sha256", "image_encoder_offload"}
    result = {k: v for k, v in c.items() if k not in ignored}
    result.update(self_attn_1_type="torch_sdpa", cross_attn_1_type="torch_sdpa", cross_attn_2_type="torch_sdpa",
        clip_encoder_offload=False, dit_quantized=False, dit_quant_scheme="Default", quant_method=None,
        t5_quantized=False, clip_quantized=False, vae_quantized=False, weight_auto_quant=False,
        feature_caching="NoCaching", lazy_load=False, unload_modules=False, use_compile=False,
        lora_configs=None, lora_dynamic_apply=False, use_stream_vae=False, use_tae=False, dummy_model=False)
    return result


def validate(plan, frozen=False):
    need(plan.get("id") == ID and plan.get("numeric_path") == "BF16_ORIGINAL" and plan.get("limits") == LIMITS
         and plan.get("resources") == RESOURCE and plan.get("CPU_preflight_resources") == CPU_RESOURCE
         and plan.get("runtime_overlay") == recipe(), "Fixed original BF16 scope/recipe differs")
    need([c["index"] for c in plan["cases"]] == [0, 12] and len({c["image_sha256"] for c in plan["cases"]}) == 2, "Two exposed source clusters required")
    need(len(plan["models"]) == 4 and sum(x["bytes"] for x in plan["models"]) == FULL_READ_BYTES
         and {x["kind"] for x in plan["models"]} == {"high", "low", "T5", "VAE"}, "Four raw model payloads required")
    originals = {x["filename"]: x for x in read(ROOT / ".runtime/model_view.json")["files"]}
    need(all(x["filename"] in originals and all(x[k] == originals[x["filename"]][k] for k in ("sha256", "bytes", "repo", "revision"))
             and Path(x["path"]).is_relative_to(plan["model_view_root"]) for x in plan["models"]), "Exact public original model identities/view required")
    paths = [x["path"] for x in plan["repo_sources"]]
    need(len(paths) == len(set(paths)) and TOOLS <= set(paths), "Immutable native source closure incomplete")
    need(Path(plan["output_HF_relative"]).parts[:1] == ("ffnds-derived",) and ".." not in Path(plan["output_HF_relative"]).parts, "Output escaped HF")
    if frozen: need(plan.get("assets_finalized") is True and plan.get("binding_status") == "BOUND_READY_FOR_REGISTRATION", "Original BF16 draft is not frozen")


def history(plan, hf):
    parent = read(bound(ROOT, plan["parent"]["protocol"])); run = read(bound(hf, plan["parent"]["RUN"])); exit = read(bound(hf, plan["parent"]["exit"]))
    need(plan["parent"]["RUN"]["sha256"] == PARENT_RUN_SHA and run.get("status") == "COMPLETE_MATCHED_SIX_NATIVE_VIDEOS_CORE_METRICS_QUALITY_PENDING"
         and run.get("protocol_sha256") == plan["parent"]["protocol"]["sha256"]
         and exit.get("id") == run["id"] == parent["id"] and exit.get("protocol_sha256") == run["protocol_sha256"]
         and exit.get("child_returncode") == 0 and type(exit["child_returncode"]) is int and exit.get("process_exited") is True
         and exit.get("timeout") is False and exit.get("automatic_retry") is False
         and exit.get("run_sha256") == PARENT_RUN_SHA, "Actual completed historical producer/RUN/exit0 required")
    for case in plan["cases"]:
        original = next(c for c in parent["cases"] if c["index"] == case["index"])
        need(all(case[k] == original[k] for k in ("source_id", "image_sha256", "prompt_sha256", "prompt", "conditioning_id", "image", "source_group")), "Historical conditioning differs")
        need(hashlib.sha256(case["prompt"].encode()).hexdigest() == case["prompt_sha256"], "Prompt digest differs")
        phase = next(p for p in run["phases"] if p["source_index"] == case["index"] and p["mode"] == "dense")
        generation = read(bound(hf, case["control_generation"]))
        need(case["control_generation"] == {k: v for k, v in phase["generation_receipt"].items() if k != "root"}
             and generation.get("status") == "PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE"
             and generation.get("completed_native_flow_steps") == 4 and generation.get("seed") == 42
             and generation.get("initial_noise") == case["control_initial"] and generation.get("raw_FP32_noise") == case["control_raw"], "Historical exact noise producer differs")
        for key in ("image", "control_initial", "control_raw"): bound(hf, case[key])
    return True


def cohort(plan, hf):
    from ffnds_cohort_preflight import audit
    sources, samples = [], []
    for c, rid in zip(plan["cases"], RUN_IDS):
        sources.append({"source_id": c["source_id"], "image_id": c["image_sha256"], "prompt_id": c["prompt_sha256"], "prompt": c["prompt"],
            "image_path": str((hf / c["image"]["path"]).resolve()), "image_name": Path(c["image"]["path"]).name,
            "conditioning_id": c["conditioning_id"], "role": "DEVELOPMENT", "exposure_history": {"fitting": False, "method_selection": True, "prior_evaluation": True}})
        samples.append({"source_id": c["source_id"], "variant": "dense", "run_id": rid, "asset_id": "video:" + rid,
            "seed": 42, "sample_index": 0, "conditioning_id": c["conditioning_id"], "initial_noise_id": ID + ":" + c["source_id"],
            "scheduler_id": "Wan22StepDistillScheduler:shift5:sigma1,.9375,.8333333134651184,.625:high,high,low,low",
            **{k: VIDEO[k] for k in ("frames", "fps", "width", "height")}, "status": "PLANNED"})
    value = {"schema": "ffnds.cohort.v1", "mode": "CUSTOM_PAIRED", "experiment": ID, "phase": "REGISTERED_REAL_GENERATION_PLAN", "sources": sources, "samples": samples, "calibration": [], "required_variants": ["dense"]}
    checked = audit(value); need(not checked["errors"] and checked["summary"]["independent_image_clusters"] == 2, "Original BF16 cohort invalid")
    return value


def authorize(plan, digest, run_id=None):
    validate(plan, True); active = read(ROOT / "research/ACTIVE_RESEARCH.json")
    policy = active.get("resource_scheduling_policy_20261028", {})
    user = active.get("user_continuation_authorization_20261027", {})
    need(user.get("status") == "ACTIVE_CURRENT_USER_DIRECTED_SESSION" and user.get("new_scope_registration_allowed") is True
         and policy.get("mode") == "DYNAMIC_MAX_ACTUALLY_CALLABLE" and policy.get("global_gpu_card_cap") is None
         and active.get("current_session_resource_constraint_20261028", {}).get("max_simultaneously_assigned_idle_GPUs") == 2, "Repository/session authority differs")
    rows = [x for x in active["experiment_authority"]["current_user_authorized_scopes"] if x.get("id") == ID]
    need(len(rows) == 1 and rows[0].get("protocol_sha256") == digest and rows[0].get("execution_status") in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
         and rows[0].get("limits") == LIMITS and rows[0].get("resources") == RESOURCE and rows[0].get("max_gpu_cards") == 1
         and (run_id is None or run_id in RUN_IDS and run_id in rows[0].get("run_ids", [])), "Exact original BF16 ACTIVE registration required")
    return rows[0]


def full_hash(item, stop=None):
    stop = threading.Event() if stop is None else stop
    result = {"path": item["path"], "expected_sha256": item["sha256"], "predebited_bytes": item["bytes"], "actual_read_bytes": 0}
    try:
        need(not stop.is_set(), "Stopped after another raw payload failure")
        path = Path(item["path"]); before = stat_record(path); need(before["bytes"] == item["bytes"], "Raw model file size differs")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                need(not stop.is_set(), "Stopped after another raw payload failure")
                data = stream.read(8 << 20)
                if not data: break
                result["actual_read_bytes"] += len(data); h.update(data)
        result.update(sha256=h.hexdigest(), stat=stat_record(path))
        need(before == result["stat"] and result["actual_read_bytes"] == item["bytes"] and result["sha256"] == item["sha256"], "Fresh full original model SHA/stat differs")
        result["status"] = "PASS_FRESH_FULL_SHA"
    except BaseException as e:
        stop.set(); result.update(status="FAILED_OR_STOPPED_READ_PRESERVED_NO_RETRY", error_type=type(e).__name__, error=str(e))
    return result


def CPU_expected(plan, digest):
    return {"id": ID, "protocol_sha256": digest, "status": "PASS_FRESH_RAW_PAYLOAD_SHA_NO_MODEL_NO_NUMPY_NO_TORCH",
        "source_closure_sha256": canonical(plan["repo_sources"]), "models_sha256": canonical(plan["models"]),
        "model_metadata_sha256": canonical(plan["model_view_metadata"]),
        "all_CUDA_devices_hidden": True, "NumPy_or_Torch_imported": False, "model_calls": 0, "GPU_calls": 0,
        "predebited_full_payload_bytes": FULL_READ_BYTES, "actual_full_payload_read_bytes": FULL_READ_BYTES}


def preflight(args):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules and "numpy" not in sys.modules, "Independent CPU proof requires hidden CUDA/no numeric imports")
    started = time.monotonic(); plan = read(args.protocol); validate(plan, True)
    out = args.out.resolve(); need(not out.exists() and out.is_relative_to(args.hf_root.resolve()), "Fresh confined CPU proof output required")
    out.mkdir(parents=True); report = CPU_expected(plan, sha(args.protocol)); report.update(status="STARTED_RAW_HASH_NO_RETRY", model_payloads=[])
    write(out / "RUN.json", report)
    def timeout(*_): raise TimeoutError("Raw CPU hash wall1200 reached; no retry")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(CPU_RESOURCE["wall_seconds"])
    try:
        need(len(os.sched_getaffinity(0)) <= 4, "CPU proof affinity must be at most4")
        for x in plan["repo_sources"]: bound(ROOT, x)
        model_metadata(plan)
        history(plan, args.hf_root)
        verify_local_package(plan, {"repo": ROOT, "hf": args.hf_root}, args.local_root)
        stop = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(lambda x: full_hash(x, stop), plan["models"]):
                report["model_payloads"].append(result); write(out / "RUN.json", report)
        need(all(x["status"] == "PASS_FRESH_FULL_SHA" for x in report["model_payloads"])
             and sum(x["actual_read_bytes"] for x in report["model_payloads"]) == FULL_READ_BYTES, "Full69GB payload read incomplete or failed; no retry")
        for x in plan["models"]:
            if x["kind"] in ("high", "low"): need(raw_header_check(header(x["path"]))["header_sha256"] == x["header_sha256"], "Raw DiT header differs")
        need("torch" not in sys.modules and "numpy" not in sys.modules, "Numerical import in CPU proof")
        report.update(CPU_expected(plan, sha(args.protocol)))
    except BaseException as e:
        report.update(status="FAILED_RAW_HASH_PRESERVED_NO_RETRY", error_type=type(e).__name__, error=str(e)); raise
    finally:
        signal.alarm(0); report["wall_seconds"] = time.monotonic() - started
        report["actual_full_payload_read_bytes"] = sum(x["actual_read_bytes"] for x in report["model_payloads"])
        write(out / "RUN.json", report)
    return report


def require_CPU(scope, plan, digest, hf):
    proof = scope.get("cpu_preflight"); need(isinstance(proof, dict) and {"RUN", "launch_exit"} <= set(proof), "Exited raw CPU proof NOT_BOUND")
    run, exit = read(bound(hf, proof["RUN"])), read(bound(hf, proof["launch_exit"]))
    need(all(run.get(k) == v for k, v in CPU_expected(plan, digest).items()) and 0 <= run.get("wall_seconds", -1) <= 1200, "Raw CPU proof identity/count differs")
    need(exit.get("id") == ID and exit.get("protocol_sha256") == digest and exit.get("process_exited") is True and exit.get("child_returncode") == 0
         and type(exit["child_returncode"]) is int and exit.get("timeout") is False and exit.get("automatic_retry") is False and exit.get("run_sha256") == proof["RUN"]["sha256"], "Raw CPU actual exit0/RUN SHA required")
    need(len(run["model_payloads"]) == 4 and {x["path"] for x in run["model_payloads"]} == {x["path"] for x in plan["models"]}
         and sum(x["actual_read_bytes"] for x in run["model_payloads"]) == FULL_READ_BYTES, "Four fresh full SHA proofs required")
    for item in plan["models"]:
        row = next(x for x in run["model_payloads"] if x["path"] == item["path"])
        need(row.get("status") == "PASS_FRESH_FULL_SHA" and row.get("sha256") == item["sha256"] and row["actual_read_bytes"] == row["predebited_bytes"] == item["bytes"] and stat_record(item["path"]) == row["stat"], "Raw payload changed after independent hash proof")
    return proof


class Ledger:
    def __init__(self, record, journal):
        self.record, self.journal = record, journal
        record.update(predebited={}, observed={}, forbidden_attempts=[])

    def call(self, rid, costs, fn, *args, **kwargs):
        need(rid in RUN_IDS or rid == "setup", "Unregistered operation")
        need(all(k in LIMITS and type(n) is int and n > 0 and self.record["predebited"].get(k, 0) + n <= LIMITS[k] for k, n in costs.items()), "Original BF16 operation budget exceeded")
        for k, n in costs.items(): self.record["predebited"][k] = self.record["predebited"].get(k, 0) + n
        self.journal({"kind": "reserved_before_call", "run_id": rid, "costs": costs})
        value = fn(*args, **kwargs)
        for k, n in costs.items(): self.record["observed"][k] = self.record["observed"].get(k, 0) + n
        self.journal({"kind": "observed_return", "run_id": rid, "costs": costs})
        return value

    def forbidden(self, kind):
        def fail(*a, **kw):
            self.record["forbidden_attempts"].append(kind)
            self.journal({"kind": "forbidden_before_dispatch", "operation": kind})
            raise RuntimeError("Outside BF16 original scope: " + kind)
        return fail

    def finish(self):
        need(not self.record["forbidden_attempts"] and all(self.record[p].get(k, 0) == n for p in ("predebited", "observed") for k, n in LIMITS.items()), "Incomplete actual BF16 operation budget")


class NoHiF4(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "HiFloat4" or fullname.startswith(("HiFloat4.", "quant.hif4", "quant.hifx")):
            raise ImportError("Legacy HiF4 adapter/quantizer is outside BF16_ORIGINAL")


def walk(owner):
    yield owner
    for child in getattr(owner, "_modules", {}).values():
        if child is not None: yield from walk(child)


def operand_check(torch, mm, value):
    need(type(mm).__name__ == "MMWeight" and type(mm).__module__ == "lightx2v.common.ops.mm.mm_weight"
         and not mm.has_lora_branch and not mm.has_diff and not hasattr(mm, "weight_diff") and not hasattr(mm, "bias_diff"), "Exact original Default MM without adapters required")
    weight, bias = mm.weight, getattr(mm, "bias", None)
    need(torch.is_tensor(value) and torch.is_tensor(weight) and value.dtype == weight.dtype == torch.bfloat16
         and value.device.type == weight.device.type == "cuda" and value.device.index in (None, 0) and weight.device.index in (None, 0)
         and value.ndim == weight.ndim == 2 and value.shape[1] == weight.shape[0], "Actual Default MM operand dtype/device/shape differs")
    need(bias is None or torch.is_tensor(bias) and bias.dtype == torch.bfloat16 and bias.device.type == "cuda" and tuple(bias.shape) == (weight.shape[1],), "Actual original BF16 bias differs")
    need(tuple(weight.stride()) == (1, weight.shape[0]), "Original transposed K-by-N weight stride differs")
    return {"input": list(value.shape), "weight": list(weight.shape), "input_stride": list(value.stride()),
            "weight_stride": list(weight.stride()), "bias_stride": None if bias is None else list(bias.stride()),
            "dtype": str(value.dtype), "device": str(value.device)}


def loaded_raw_check(torch, result, rows):
    need(set(result) == set(rows) and all(torch.is_tensor(t) and t.dtype == torch.bfloat16 and t.device.type == "cpu" and list(t.shape) == rows[k]["shape"] for k, t in result.items()), "Actual original loader tensors differ from raw BF16 header")
    return result


class OriginalGuard:
    """Guard original loaders and Default MM; never replace their arithmetic."""
    def __init__(self, torch, plan, ledger):
        self.torch, self.plan, self.ledger, self.active = torch, plan, ledger, None
        self.stack = ExitStack(); self.loaded = []; self.t5_loads = {}

    def __enter__(self):
        from lightx2v.models.networks.base_model import BaseTransformerModel
        from lightx2v.common.ops.mm.mm_weight import MMWeight, MMWeightQuantTemplate
        source_load, source_apply, source_mm_load = BaseTransformerModel._load_ckpt, MMWeight.apply, MMWeight.load
        rows = {str(Path(x["path"]).resolve()): raw_header_check(header(x["path"]))["rows"] for x in self.plan["models"] if x["kind"] in ("high", "low")}
        def load(model, *a, **kw):
            path = str(Path(model.model_path).resolve())
            need(path in rows and path not in self.loaded and model.device.type == "cpu" and not model.dit_quantized, "Wrong/duplicate original expert loader")
            def original():
                result = source_load(model, *a, **kw)
                loaded_raw_check(self.torch, result, rows[path])
                self.loaded.append(path); return result
            return self.ledger.call("setup", {"original_expert_loads": 1}, original)
        def mm_load(mm, weights):
            need(type(mm) is MMWeight and not mm.lazy_load and not mm.create_cpu_buffer, "Unexpected original MM loader type")
            need(self.torch.is_tensor(weights[mm.weight_name]) and weights[mm.weight_name].dtype == self.torch.bfloat16, "Original MM source is not raw BF16")
            t5 = re.fullmatch(r"blocks\.(\d+)\.(attn\.[qkvo]|ffn\.(?:fc[12]|gate\.0))\.weight", mm.weight_name)
            if t5:
                need(int(t5[1]) in range(24) and id(mm) not in self.t5_loads, "Wrong/duplicate original T5 MM loader")
                self.t5_loads[id(mm)] = (bool(mm.create_cuda_buffer), int(t5[1]), t5[2])
            key = ("T5_" if t5 else "") + ("buffer_Default_MM_loads" if mm.create_cuda_buffer else "raw_Default_MM_loads")
            return self.ledger.call("setup", {key: 1}, source_mm_load, mm, weights)
        def apply(mm, value):
            need(self.active is not None, "Unregistered/warmup Default MM outside a native generation")
            meta = operand_check(self.torch, mm, value)
            return self.active.mm_call(mm, value, meta, source_apply)
        self.stack.enter_context(patch_method(BaseTransformerModel, "_load_ckpt", load))
        self.stack.enter_context(patch_method(BaseTransformerModel, "_load_quant_ckpt", self.ledger.forbidden("quantized_expert_loads")))
        self.stack.enter_context(patch_method(MMWeightQuantTemplate, "__init__", self.ledger.forbidden("quantized_MM_constructions")))
        self.stack.enter_context(patch_method(MMWeight, "load", mm_load))
        self.stack.enter_context(patch_method(MMWeight, "apply", apply))
        return self

    def __exit__(self, *args): self.stack.close()


def build_pipeline(plan):
    """Same lossless config merge as the historical builder, without its import."""
    from lightx2v import LightX2VPipeline
    from lightx2v.utils.set_config import auto_calc_config, get_default_config
    from lightx2v.utils.utils import validate_config_paths
    models = {x["kind"]: x["path"] for x in plan["models"]}
    pipe = LightX2VPipeline(model_path=plan["model_view_root"], model_cls="wan2.2_moe_distill", task="i2v",
        high_noise_original_ckpt=models["high"], low_noise_original_ckpt=models["low"])
    pipe.enable_offload(cpu_offload=True, offload_granularity="block", text_encoder_offload=True, image_encoder_offload=False, vae_offload=False)
    pipe.resize_mode = None
    pipe.set_infer_config(attn_mode="torch_sdpa", rope_type="torch", infer_steps=4, num_frames=81, height=480, width=832,
        guidance_scale=1., sample_shift=5., fps=16, aspect_ratio="16:9", boundary=.9, boundary_step_index=2,
        denoising_step_list=[1000, 750, 500, 250], audio_fps=24000, double_precision_rope=True,
        norm_modulate_backend="torch", distilled_sigma_values=None)
    config = get_default_config(); config.update({k: copy.deepcopy(v) for k, v in vars(pipe).items() if v is not None})
    config.update(copy.deepcopy(plan["runtime_overlay"])); config = auto_calc_config(config)
    config.update(copy.deepcopy(plan["runtime_overlay"]))
    need(all(config.get(k) == v for k, v in plan["runtime_overlay"].items()), "Original BF16 config merge differs")
    validate_config_paths(config); pipe.runner = pipe._init_runner(config)
    need(all(pipe.runner.config.get(k) == v for k, v in plan["runtime_overlay"].items()), "Actual original BF16 runner config differs")
    return pipe


class NativeTap:
    def __init__(self, pipe, guard, rid, save, check):
        self.runner, self.guard, self.rid, self.save, self.check = pipe.runner, guard, rid, save, check
        self.stack, self.block, self.lines, self.other, self.shapes = ExitStack(), None, set(), set(), []
        self.mapping = {}; self.raw_draws = 0; self.events = []
        self.t5_mapping, self.t5_lines, self.t5_blocks, self.t5_encoding = {}, [], [], False

    def step(self, expert=None):
        s = int(self.runner.scheduler.step_index)
        need(s in range(4) and (expert is None or expert == (0 if s < 2 else 1)), "Actual step/expert differs")
        return s

    def counted(self, owner, name, key, check=lambda *a, **kw: None):
        original = getattr(owner, name)
        def called(*a, **kw):
            check(*a, **kw)
            return self.guard.ledger.call(self.rid, {key: 1}, original, *a, **kw)
        self.stack.enter_context(patch_method(owner, name, called))

    def mm_call(self, mm, value, meta, original):
        if id(mm) in self.t5_mapping:
            suffix = self.t5_mapping[id(mm)]; target = (int(self.t5_model.block_idx), suffix)
            expected = (len(self.t5_lines) // 7, T5_NAMES[len(self.t5_lines) % 7])
            need(self.t5_encoding and self.block is None and target == expected, "Wrong phase/duplicate/out-of-order original T5 MM")
            self.t5_lines.append(target)
            result = self.guard.ledger.call(self.rid, {"T5_Default_MM_calls": 1}, original, mm, value)
            self.shapes.append({"role": "T5", "block": target[0], "module": suffix, **meta})
            return result
        need(id(mm) in self.mapping, "Default MM outside bound original model objects")
        role, expert, suffix = self.mapping[id(mm)]; step = self.step(expert)
        if role == "block":
            need(self.block is not None and self.block[0] == expert, "Block MM outside original active block")
            b = self.block[1]; target = (step, b, suffix)
            expected = (step, len([x for x in self.lines if x[0] == step]) // 10,
                        LINEAR_SUFFIXES[len([x for x in self.lines if x[0] == step]) % 10])
            need(target == expected and target not in self.lines, "Duplicate/out-of-order original full-depth MM")
            if suffix.startswith("ffn."): need(meta["input"][0] == 31668, "Actual full-M FFN population differs")
            key = "native_FFN_linears" if suffix.startswith("ffn.") else "native_non_FFN_linears"
            self.lines.add(target)
        else:
            target = (step, role, suffix); need(target not in self.other and self.block is None, "Duplicate/pre/head MM in wrong phase")
            self.other.add(target); key = "pre_Default_MM_calls" if role == "pre" else "head_Default_MM_calls"
        result = self.guard.ledger.call(self.rid, {key: 1}, original, mm, value)
        self.shapes.append({"step": step, "expert": expert, "role": role, "module": suffix, **meta})
        return result

    def __enter__(self):
        try:
            self.install(); self.guard.active = self; return self
        except BaseException:
            self.stack.close(); raise

    def install(self):
        torch, runner, scheduler = self.guard.torch, self.runner, self.runner.scheduler
        for name, key in (("run_input_encoder", "input_encoder_requests"), ("run_vae_encoder", "VAE_encodes"), ("run_vae_decoder", "VAE_decodes")):
            self.counted(runner, name, key)
        need(len(runner.text_encoders) == 1, "One original T5 encoder required")
        encoder = runner.text_encoders[0]; self.t5_model = t5 = encoder.model
        need(encoder.cpu_offload and t5.cpu_offload and t5.num_layers == 24 and t5.dim == 4096 and t5.dim_ffn == 10240
             and len(t5.blocks_weights.offload_block_buffers) == 1, "Original T5 offload geometry differs")
        for mm in walk(t5.blocks_weights.offload_block_buffers[0]):
            if type(mm).__name__ == "MMWeight":
                need(id(mm) in self.guard.t5_loads and self.guard.t5_loads[id(mm)][0], "Unbound T5 CUDA buffer MM")
                self.t5_mapping[id(mm)] = self.guard.t5_loads[id(mm)][2]
        need(len(self.t5_mapping) == 7 and set(self.t5_mapping.values()) == set(T5_NAMES), "Original T5 seven-MM buffer differs")
        original_text = runner.run_text_encoder
        def text(*a, **kw):
            need(not self.t5_encoding and not self.t5_lines, "Repeated T5 prompt encoder")
            self.t5_encoding = True
            try: return self.guard.ledger.call(self.rid, {"T5_positive_prompt_encodes": 1}, original_text, *a, **kw)
            finally: self.t5_encoding = False
        self.stack.enter_context(patch_method(runner, "run_text_encoder", text))
        buffer = t5.blocks_weights.offload_block_buffers[0]; original_t5_load = buffer.load_state_dict
        def t5_load(destination, block_index, adapter_block_index=None):
            need(self.t5_encoding and block_index == len(self.t5_blocks) == int(t5.block_idx) and block_index in range(24)
                 and len(self.t5_lines) == block_index * 7 and isinstance(destination, dict), "Wrong/duplicate/out-of-order T5 block load")
            self.t5_blocks.append(block_index)
            return self.guard.ledger.call(self.rid, {"T5_checkpoint_block_buffer_loads": 1}, original_t5_load, destination, block_index, adapter_block_index)
        self.stack.enter_context(patch_method(buffer, "load_state_dict", t5_load))
        self.stack.enter_context(patch_method(runner, "run_image_encoder", self.guard.ledger.forbidden("CLIP_encodes")))
        if hasattr(runner, "run_vae_decoder_stream"): self.stack.enter_context(patch_method(runner, "run_vae_decoder_stream", self.guard.ledger.forbidden("stream_VAE")))
        def prepared(*a, **kw):
            value = prepare(*a, **kw)
            need(type(scheduler).__name__ == "Wan22StepDistillScheduler" and scheduler.sigmas.tolist() == [1., .9375, .8333333134651184, .625]
                 and scheduler.denoising_step_list == [1000, 750, 500, 250] and scheduler.boundary_step_index == 2
                 and not scheduler._uses_conditioned_latent_prefix() and not scheduler.keep_latents_dtype_in_scheduler, "Original shifted timetable/scheduler branch differs")
            self.timetable = {"sigmas": scheduler.sigmas.tolist(), "actual_timesteps": scheduler.timesteps.detach().cpu().tolist(), "denoising_indices": list(scheduler.denoising_step_list)}
            return value
        prepare = scheduler.prepare; self.stack.enter_context(patch_method(scheduler, "prepare", prepared)); self.counted(scheduler, "prepare", "scheduler_prepare")
        original_noise = scheduler.prepare_latents
        def noise(seed, shape, dtype=None):
            need(seed == 42 and list(shape) == VIDEO["latent_shape"] and dtype == torch.float32 and scheduler.generator is None, "One original seed42 raw FP32 draw required")
            original_randn = torch.randn
            def randn(*a, **kw):
                need(self.raw_draws == 0 and list(a) == VIDEO["latent_shape"] and kw.get("dtype") == torch.float32 and kw.get("generator") is scheduler.generator, "Unexpected raw random draw")
                self.raw_draws += 1; result = original_randn(*a, **kw); self.save(result, "raw_noise_FP32", "raw_FP32_noise_files"); return result
            with patch_method(torch, "randn", randn): return self.guard.ledger.call(self.rid, {"FP32_initial_noise_draws": 1}, original_noise, seed, shape, dtype=dtype)
        self.stack.enter_context(patch_method(scheduler, "prepare_latents", noise))
        original_pre, original_post = scheduler.step_pre, scheduler.step_post
        def pre(*a, **kw):
            result = self.guard.ledger.call(self.rid, {"scheduler_step_pre": 1}, original_pre, *a, **kw)
            if self.step() == 0: self.save(scheduler.latents, "initial_noise", "initial_BF16_noise_files")
            return result
        def post(*a, **kw):
            s = self.step(); self.save(scheduler.noise_pred, f"flow_step{s}", "native_FP32_flow_files")
            result = self.guard.ledger.call(self.rid, {"scheduler_step_post": 1}, original_post, *a, **kw)
            self.save(scheduler.latents, f"latent_step{s}", "native_BF16_post_files"); self.check(); return result
        self.stack.enter_context(patch_method(scheduler, "step_pre", pre)); self.stack.enter_context(patch_method(scheduler, "step_post", post))
        self.counted(runner.model, "infer", "native_full_model_flows")
        for expert, model in enumerate(runner.model.model):
            weights, infer = model.transformer_weights, model.transformer_infer
            need(len(weights.blocks) == 40 and not infer.has_post_adapter and infer.infer_dtype == infer.sensitive_layer_dtype == torch.bfloat16, "Original forty-block BF16 branch required")
            for name in PRE_NAMES: self.mapping[id(getattr(model.pre_weight, name))] = ("pre", expert, name)
            self.mapping[id(weights.head)] = ("head", expert, "head")
            for owner, name, key in ((model.pre_infer, "infer", "pre_infer_calls"), (model.pre_weight.patch_embedding, "apply", "patch_Conv3D_calls"), (weights.norm, "apply", "final_LayerNorm_calls"), (model.post_infer, "infer", "post_unpatchify_calls")):
                self.counted(owner, name, key, lambda *a, _e=expert, **kw: self.step(_e))
            original_block, original_ffn = infer.infer_block, infer.infer_ffn
            def block(*a, _e=expert, _i=infer, _fn=original_block, **kw):
                self.step(_e); need(self.block is None, "Nested native blocks")
                self.block = (_e, int(_i.block_idx))
                try: return self.guard.ledger.call(self.rid, {"native_blocks": 1}, _fn, *a, **kw)
                finally: self.block = None
            self.stack.enter_context(patch_method(infer, "infer_block", block))
            def ffn(*a, _e=expert, _fn=original_ffn, **kw):
                self.step(_e)
                def timed():
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    self.events.append((start, end)); start.record()
                    try: return _fn(*a, **kw)
                    finally: end.record()
                return self.guard.ledger.call(self.rid, {"native_FFN_calls": 1, "FFN_CUDA_event_pairs": 1}, timed)
            self.stack.enter_context(patch_method(infer, "infer_ffn", ffn))
            manager = infer.offload_manager
            need(manager.offload_granularity == "block" and not manager.lazy_load and len(manager.cuda_buffers) == 2, "Original double block buffer offload required")
            self.guard.ledger.call(self.rid, {"offload_buffer_resets": 1}, setattr, manager, "need_init_first_buffer", True)
            aux = {b: original_rms_auxiliaries(owner, b) for b, owner in enumerate(weights.blocks)}
            for buffer in manager.cuda_buffers:
                for mm in walk(buffer):
                    if type(mm).__name__ == "MMWeight": self.mapping[id(mm)] = ("block", expert, ".".join(mm.weight_name.split(".")[2:]).removesuffix(".weight"))
                original_load = buffer.load_state_dict
                def load(destination, block_index, adapter_block_index=None, _fn=original_load, _aux=aux, _e=expert):
                    self.step(_e)
                    def checked():
                        checkpoint_source_metadata(torch, destination, block_index, _aux[block_index])
                        return _fn(destination, block_index, adapter_block_index)
                    return self.guard.ledger.call(self.rid, {"original_checkpoint_block_buffer_loads": 1}, checked)
                self.stack.enter_context(patch_method(buffer, "load_state_dict", load))

    def finish(self):
        need(self.t5_lines == [(b, n) for b in range(24) for n in T5_NAMES] and self.t5_blocks == list(range(24)), "Actual original T5 offload coverage incomplete")
        need(self.raw_draws == 1 and self.lines == {(s, b, k) for s in range(4) for b in range(40) for k in LINEAR_SUFFIXES}
             and self.other == {(s, role, n) for s in range(4) for role, names in (("pre", PRE_NAMES), ("head", ("head",))) for n in names}, "Actual 160 FFN/1600 block MM/24pre-head coverage incomplete")

    def __exit__(self, *args):
        self.guard.active = None; self.stack.close()


def native_phase(pipe, policy, plan, case, rid, out, hf, check, digest, report, persist):
    import torch
    from safetensors.torch import save_file
    from ub1_consumer_runtime import tensor_sha
    from run_ub1_consumer_videos import inspect_video
    authorize(plan, digest, rid)
    folder = out / rid; folder.mkdir()
    phase = {"id": ID, "numeric_path": "BF16_ORIGINAL", "mode": "dense", "run_id": rid,
        "source_index": case["index"], "source_id": case["source_id"], "image_sha256": case["image_sha256"],
        "prompt_sha256": case["prompt_sha256"], "conditioning_id": case["conditioning_id"], "seed": 42,
        "protocol_sha256": digest, "status": "STARTED_NO_RETRY", "states": {}, "source_exposure": "PREVIOUSLY_EXPOSED_DEVELOPMENT",
        "quality_or_speed_acceptance": False}
    report["phases"].append(phase); persist()
    def save(value, name, key):
        need(name not in phase["states"], "Duplicate native state output")
        expected = torch.float32 if key in ("raw_FP32_noise_files", "native_FP32_flow_files") else torch.bfloat16
        def output():
            need(value.dtype == expected and list(value.shape) == VIDEO["latent_shape"], "Actual native state shape/dtype differs")
            cpu = value.detach().cpu().contiguous(); need(bool(torch.isfinite(cpu).all()), "Nonfinite native state")
            path = folder / (name + ".safetensors"); need(not path.exists(), "State overwrite forbidden")
            save_file({"flow" if key == "native_FP32_flow_files" else "latents": cpu}, str(path))
            item = {"root": "hf", **binding(path, hf), "raw_tensor_sha256": tensor_sha(cpu), "shape": list(cpu.shape), "dtype": str(cpu.dtype).removeprefix("torch.")}
            phase["states"][name] = item
            if name in ("raw_noise_FP32", "initial_noise"):
                reference = case["control_raw"] if name == "raw_noise_FP32" else case["control_initial"]
                verify_noise_pair(item, reference, hf)
                phase[name + "_same_bytes_as_old_QDQ_input_control"] = True
            persist(); return item
        return policy.ledger.call(rid, {key: 1}, output)
    pipe.runner.scheduler.generator = None
    tap = NativeTap(pipe, policy, rid, save, check)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    try:
        def generate():
            with tap, torch.inference_mode():
                pipe.generate(seed=42, image_path=str((hf / case["image"]["path"]).resolve()), prompt=case["prompt"], negative_prompt="", save_result_path=str(folder / "output.mp4"))
            torch.cuda.synchronize(); tap.finish()
        policy.ledger.call(rid, {"video_generations": 1}, generate)
        phase.update(status="PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE", completed_native_flow_steps=4,
            scheduler_generator_reset=True, timetable=tap.timetable,
            native_counter_delta={"linear_forward_calls": len(tap.lines), "activation_qdq_calls": 0, "weight_quantizations": 0},
            Default_MM_actual_operand_metadata=tap.shapes, initial_noise=phase["states"]["initial_noise"],
            raw_FP32_noise=phase["states"]["raw_noise_FP32"], video=inspect_video(folder / "output.mp4", VIDEO),
            generation_wall_seconds_including_encoders_decode_observers=time.perf_counter() - started,
            FFN_boundary_CUDA_seconds=sum(a.elapsed_time(b) for a, b in tap.events) / 1000,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            post_comparison_to_old_QDQ="DIFFERENT_NUMERIC_PATH_NO_NEUTRAL_EQUALITY_REQUIREMENT")
        need(phase["peak_cuda_reserved_bytes"] <= RESOURCE["GPU_reserved_max_bytes"], "BF16 GPU reserved memory limit exceeded")
        path = folder / "GENERATION.json"; write(path, phase); phase["generation_receipt"] = binding(path, hf)
        return phase
    except BaseException as e:
        phase.update(status="FAILED_ORIGINAL_BF16_PHASE_PRESERVED_NO_RETRY", error_type=type(e).__name__, error=str(e)); raise
    finally:
        write(folder / "PHASE.json", phase); persist()


def run(args):
    started = time.monotonic(); plan = read(args.protocol); digest = sha(args.protocol)
    scope = authorize(plan, digest); proof = require_CPU(scope, plan, digest, args.hf_root)
    need("torch" not in sys.modules and "numpy" not in sys.modules, "Fresh BF16 worker before numerical imports required")
    from worker_identity import require_worker, private_config_path
    require_worker(); need(sha(private_config_path()) in plan["private_worker_config_sha256"], "Assigned worker identity differs")
    need(len(os.sched_getaffinity(0)) <= 4, "Original BF16 worker CPU affinity must be at most4")
    need(not any(os.environ.get(k) == "1" for k in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")), "Forced TF32 environment forbidden")
    for x in plan["repo_sources"]: bound(ROOT, x)
    model_metadata(plan)
    history(plan, args.hf_root)
    local, rows = verify_local_package(plan, {"repo": ROOT, "hf": args.hf_root}, args.local_root)
    need({k: importlib.metadata.version(k) for k in plan["runtime_versions"]} == plan["runtime_versions"], "Native dependency versions differ")
    out = (args.hf_root / plan["output_HF_relative"]).resolve()
    need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), "Fresh confined native output required")
    out.mkdir(parents=True); report = {"id": ID, "numeric_path": "BF16_ORIGINAL", "protocol_sha256": digest, "status": "STARTED_FIXED_TWO_NATIVE_VIDEOS",
        "automatic_retry": False, "CPU_preflight": proof, "models": plan["models"], "phases": [], "official_quality": "NOT_RUN",
        "independent_image_clusters": 2, "source_role": "PREVIOUSLY_EXPOSED_DEVELOPMENT", "quality_or_speed_acceptance": False}
    persist = lambda: write(out / "RUN.json", report)
    stream = (out / "OPERATIONS.jsonl").open("x", buffering=1)
    ledger = Ledger(report.setdefault("operation_ledger", {}), lambda event: stream.write(json.dumps(event, separators=(",", ":")) + "\n"))
    persist(); monitor = None
    def check():
        authorize(plan, digest)
        if monitor: monitor.check()
        available = int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))) * 1024
        need(available >= RESOURCE["host_available_min_bytes"] and shutil.disk_usage(out).free >= RESOURCE["free_disk_min_bytes"], "Original BF16 host/disk reserve exceeded")
        need(sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink()) <= RESOURCE["new_HF_bytes"], "Original BF16 HF output budget exceeded")
        need(time.monotonic() - started <= RESOURCE["wall_seconds"], "Original BF16 compute wall exceeded")
    def timeout(*_): raise TimeoutError("Original BF16 wall1200 exhausted; no retry")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCE["wall_seconds"])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder, no_hif4 = CheckedPackageFinder(local, rows), NoHiF4()
    try:
        check(); snapshot = out / "EXECUTED_SOURCES"; snapshot.mkdir()
        for item in plan["repo_sources"]:
            dest = snapshot / item["path"]; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / item["path"], dest)
        shutil.copyfile(args.protocol, snapshot / "protocol.json")
        sys.dont_write_bytecode = True; sys.path[:0] = [str(local), str(ROOT / ".runtime/code/LightX2V")]
        sys.meta_path[:0] = [no_hif4, finder]
        os.environ.update(DTYPE="BF16", SENSITIVE_LAYER_DTYPE="BF16", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import torch
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, report["CUDA_reservation"] = reserve_before_assets(torch)
        need(torch.cuda.device_count() == 1 and "A100" in torch.cuda.get_device_name(0), "One isolated A100 required")
        torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision("highest"); torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        report["runtime_precision"] = {"matmul_TF32": False, "float32_matmul_precision": "highest", "bf16_reduced_precision_reduction": True}
        with OriginalGuard(torch, plan, ledger) as policy:
            setup = time.perf_counter(); pipe = build_pipeline(plan); setup_seconds = time.perf_counter() - setup
            report.update(raw_model_setup_seconds=setup_seconds, original_loader_paths=policy.loaded,
                BF16_matrix_residency="raw CPU checkpoint pin tensors and original four CUDA block buffers; no decoded cache or student bank")
            need(len(policy.loaded) == 2 and not any(k == "HiFloat4" or k.startswith("HiFloat4.") for k in sys.modules), "Original BF16 loader/legacy hook identity differs")
            registered = cohort(plan, args.hf_root); write(snapshot / "registered_video_plan.json", registered); persist(); check()
            for index, (case, rid) in enumerate(zip(plan["cases"], RUN_IDS)):
                phase = native_phase(pipe, policy, plan, case, rid, out, args.hf_root, check, digest, report, persist)
                from ffnds_video_metrics import evaluate, validate as validate_core
                import cv2
                cv2.setNumThreads(4)
                identity = {"source_id": case["image_sha256"], "seed": 42, "expected_frames": 81, "input_prompt_digest": case["prompt_sha256"],
                    "model_digest": canonical(plan["models"]), "protocol_sha256": digest, "run_sha256": phase["generation_receipt"]["sha256"]}
                cost = {"generation_wall_seconds": phase["generation_wall_seconds_including_encoders_decode_observers"], "ffn_boundary_seconds": phase["FFN_boundary_CUDA_seconds"],
                    "setup_seconds": setup_seconds, "peak_gpu_memory_mib": phase["peak_cuda_allocated_bytes"] / (1 << 20)}
                core = ledger.call(rid, {"core_CPU_reports": 1}, evaluate, Path(phase["video"]["path"]), None, {"identity": identity, "cost": cost})
                need(not validate_core(core), "Native81frame CORE incomplete")
                cp = out / rid / "CORE_METRICS.json"; write(cp, core); phase["core_metrics"] = binding(cp, args.hf_root)
                folder = out / f"job{index:06d}"; folder.mkdir()
                write(folder / "RUN.json", {"schema": "ffnds.registered.native.video.v1", "status": "COMPLETE_NATIVE_GENERATION_ONLY", "job": registered["samples"][index],
                    "source": registered["sources"][index], "native_ffn_sites": 160, "scheduler_generator_reset": True,
                    "native_ffn_counters": phase["native_counter_delta"], "actual_initial_noise_file": str(args.hf_root / phase["initial_noise"]["path"]),
                    "video_path": phase["video"]["path"], "decoded_frames": 81, "decoded_size": [832, 464], "decoded_fps": 16,
                    "immutable_generation_receipt": phase["generation_receipt"], "numeric_path": "BF16_ORIGINAL"})
                persist(); check()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified = collect(registered, out); write(out / "VERIFIED_GENERATIONS.json", verified)
            report.update(status="COMPLETE_TWO_ORIGINAL_BF16_NATIVE_VIDEOS_CORE_QUALITY_PENDING", verified_cohort=binding(out / "VERIFIED_GENERATIONS.json", args.hf_root), checked_transformers_imports=finder.origins)
            check()
    except BaseException as e:
        report.update(status="FAILED_ORIGINAL_BF16_PRESERVED_NO_RETRY", error_type=type(e).__name__, error=str(e)); raise
    finally:
        signal.alarm(0)
        monitor_error = None
        if monitor:
            try:
                monitor.close(); report["assigned_GPU_resources"] = monitor.report()
                need(not report["assigned_GPU_resources"].get("violations"), "Assigned GPU final resource violations")
            except BaseException as e:
                monitor_error = e
                report.update(status="FAILED_ORIGINAL_BF16_PRESERVED_NO_RETRY", resource_error_type=type(e).__name__, resource_error=str(e))
        for item in (finder, no_hif4):
            if item in sys.meta_path: sys.meta_path.remove(item)
        report["wall_seconds"] = time.monotonic() - started; persist(); stream.close()
        if monitor_error is not None and sys.exc_info()[0] is None: raise monitor_error
    return report


def draft(hf):
    parent = read(ROOT / PARENT); folder = hf / parent["output_HF_relative"]; producer = read(folder / "RUN.json")
    logical, node = read(ROOT / ".runtime/model_view.json"), read(ROOT / ".runtime/node_local_model_view.json")
    by_name = {x["filename"]: x for x in logical["files"]}; models = []
    for kind, filename in (("high", "wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors"), ("low", "wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors"), ("T5", "models_t5_umt5-xxl-enc-bf16.pth"), ("VAE", "Wan2.1_VAE.pth")):
        original = by_name[filename]; current = next(x for x in node["files"] if x["filename"] == filename)
        need(all(current[k] == original[k] for k in ("repo", "revision", "sha256", "bytes")), "Node-local and HF raw model identities differ")
        row = {"kind": kind, "path": str(Path(current["view_path"]).resolve()), **{k: original[k] for k in ("filename", "repo", "revision", "bytes", "sha256")}}
        if kind in ("high", "low"): row["header_sha256"] = raw_header_check(header(original["view_path"]))["header_sha256"]
        models.append(row)
    cases = []
    for case in parent["cases"]:
        phase = next(p for p in producer["phases"] if p["mode"] == "dense" and p["source_index"] == case["index"])
        cases.append({**{k: case[k] for k in ("index", "source_id", "image_sha256", "prompt_sha256", "prompt", "conditioning_id", "image", "source_group")},
            "control_generation": {k: v for k, v in phase["generation_receipt"].items() if k != "root"}, "control_initial": phase["initial_noise"], "control_raw": phase["raw_FP32_noise"]})
    sources = TOOLS | {str(p.relative_to(ROOT)) for top in ("lightx2v", "lightx2v_platform") for p in (ROOT / ".runtime/code/LightX2V" / top).rglob("*.py")}
    return {"id": ID, "numeric_path": "BF16_ORIGINAL", "status": "DRAFT_NO_MODEL_NO_GPU", "assets_finalized": False, "binding_status": "DECLARED_NODE_LOCAL_ASSETS_FRESH_CPU_SHA_PENDING",
        "limits": LIMITS, "resources": RESOURCE, "CPU_preflight_resources": CPU_RESOURCE, "runtime_overlay": recipe(),
        "model_view_root": str(Path(node["view"]).resolve()), "models": models, "cases": cases,
        "model_view_metadata": [{k: x[k] for k in ("filename", "bytes", "sha256", "repo", "revision")} for x in logical["files"]
                                if x["filename"].endswith(("config.json", "configuration.json")) or x["filename"].startswith("google/")],
        "parent": {"protocol": binding(ROOT / PARENT, ROOT), "RUN": binding(folder / "RUN.json", hf), "exit": binding(folder.parent / "EXIT_RECEIPT.json", hf)},
        "local_transformers": parent["local_transformers"], "runtime_versions": parent["runtime_versions"], "private_worker_config_sha256": parent["private_worker_config_sha256"],
        "repo_sources": [binding(ROOT / p, ROOT) for p in sorted(sources)],
        "output_HF_relative": "ffnds-derived/e0_original_bf16_consumer00_12_videos_20261028/attempt1/results",
        "estimand": "Two exposed image clusters, same seed42/conditioning/timetable as old QDQ input control; original BF16 DiT matrices and ordinary native Default mm/addmm, no activation or weight quantization. New numeric trajectory, no old-post neutrality or pure-quantization causal attribution.",
        "physical_state": "CPU raw high/low pinned matrices plus original two block buffers per expert; no all-resident duplicate decoded cache.",
        "input_noise_control": "FP32 raw draw before conditioning and BF16 inference initial state after original step_pre0 must each be separate actual files byte-equal to the historical control.",
        "official_quality": "NOT_RUN", "training_updates": 0, "decompositions": 0, "automatic_retry": False}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("command", choices=("draft", "inspect", "preflight", "run"))
    p.add_argument("--protocol", type=Path); p.add_argument("--hf-root", type=Path, required=True); p.add_argument("--local-root", type=Path); p.add_argument("--out", type=Path)
    a = p.parse_args(); a.hf_root = a.hf_root.resolve()
    if a.command == "draft":
        need(a.out is not None and not a.out.exists(), "Fresh draft file required"); value = draft(a.hf_root); validate(value); write(a.out, value)
        print(json.dumps({"id": ID, "status": value["status"], "model_calls": 0}))
    elif a.command == "inspect":
        validate(read(a.protocol)); print(json.dumps({"id": ID, "limits": LIMITS, "resources": RESOURCE, "model_calls": 0}, indent=2))
    elif a.command == "preflight":
        need(a.local_root is not None and a.out is not None and a.protocol is not None, "Protocol/local package/fresh CPU output required")
        value = preflight(a); print(json.dumps({"status": value["status"], "actual_full_payload_read_bytes": value["actual_full_payload_read_bytes"], "model_calls": 0}))
    else:
        need(a.local_root is not None and a.protocol is not None, "Frozen protocol/local package required")
        print(json.dumps({"status": run(a)["status"]}))


if __name__ == "__main__": main()
