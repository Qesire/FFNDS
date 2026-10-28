"""Four registered parts build two ordinary 6016 FFNs for every expert/block.

The completed four-site pilot is imported read-only for stateless byte/path,
native-library and training-dispatch helpers. Its globals are never patched.
No full model, video, teacher forward, weight QDQ or prior fitted bank is run.
"""

import argparse
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

import e0_compact6016_recovery_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_COMPACT6016_FULL80_BANK_20261028"
EXPERTS = ("high_noise_model", "low_noise_model")
INITIALIZATIONS = ("energy94", "random94")
TRAIN, HELD = tuple(range(16)), tuple(range(16, 24))
FUNCTIONS = tuple((e, b) for e in EXPERTS for b in range(40))
RESOURCE_LIMITS = {"GPU_per_part": 1, "global_max_GPU": 4, "CPU_threads_per_part": 2,
                   "compute_wall_seconds_per_part": 2400, "external_wall_seconds_per_part": 2700,
                   "new_HF_bytes_per_part": 6 << 30, "new_HF_bytes_total": 24 << 30,
                   "free_disk_bytes": 30 << 30, "peak_GPU_reserved_bytes_per_part": 32 << 30}
TRAINING = dict(pilot.TRAINING, steps_per_cell=256,
                case_order="source_index_0_to_15_outer_then_expert_steps_ascending_inner",
                shared_expert_block_weights_across_both_steps=True)


def steps_for(expert):
    if expert not in EXPERTS:
        raise ValueError("Unknown native expert")
    return (0, 1) if expert == EXPERTS[0] else (2, 3)


def functions_for(part):
    if type(part) is not int or part not in range(4):
        raise ValueError("One of four fixed parts required")
    return tuple((e, b) for e in EXPERTS for b in range(part * 10, (part + 1) * 10))


def function_id(expert, block):
    return f"compact6016bank_{'high' if expert == EXPERTS[0] else 'low'}_b{block:02d}"


def run_ids(part=None):
    functions = FUNCTIONS if part is None else functions_for(part)
    return [function_id(e, b) + "_" + init for e, b in functions for init in INITIALIZATIONS]


def part_limits():
    cells, forwards, backwards, merges = 40, 354, 257, 258
    return {"student_forwards": cells * forwards, "native_Q2_calls": cells * forwards,
            "BF16_forward_GEMM": cells * 2 * forwards,
            "FP32_merge_GEMM": cells * 2 * merges, "merged_weight_pairs": cells * merges,
            "backward_calls": cells * backwards, "BF16_backward_GEMM": cells * 3 * backwards,
            "FP32_LoRA_backward_GEMM": cells * 4 * backwards,
            "loss_evaluations": cells * (1 + 256 + 96),
            "finite_gradient_tensor_checks": cells * 4 * backwards,
            "finite_updated_parameter_tensor_checks": cells * 4 * 256,
            "Adam_steps": cells * 256, "Adam_parameter_tensor_updates": cells * 4 * 256,
            "training_source_batches": cells * 256, "diagnostic_source_batches": cells * 96,
            "GPU_gate_reference_forwards": cells, "GPU_gate_STE_forwards": cells,
            "LoRA_random_factor_draws": cells * 2,
            "exported_students": cells, "exported_final_BF16_matrices": cells * 2,
            "support_source_step_calls": 20 * 16 * 2,
            "support_FP32_batched_bmm": 20 * 16 * 2 * 27,
            "support_logical_group_products": 20 * 16 * 2 * 216,
            "weight_QDQ_calls": 0, "Q1_calls": 0, "teacher_FFN_calls": 0,
            "teacher_V2_student_forwards": 0, "weight_decompositions": 0,
            "full_model_calls": 0, "video_generations": 0, "official_quality_calls": 0}


PART_LIMITS = part_limits()
TOTAL_LIMITS = {k: v * 4 for k, v in PART_LIMITS.items()}
sha, write, bound = pilot.sha, pilot.write, pilot.bound


def authorize(plan, digest, active, part, run_id=None):
    if (plan.get("id") != ID or plan.get("assets_finalized") is not True
            or plan.get("initializations") != list(INITIALIZATIONS)
            or plan.get("limits_per_part") != PART_LIMITS or plan.get("limits_total") != TOTAL_LIMITS
            or plan.get("resource_limits") != RESOURCE_LIMITS or plan.get("training") != TRAINING
            or plan.get("allowed_run_ids") != run_ids() or plan.get("automatic_retry") is not False):
        raise ValueError("Exact frozen full80 scope differs")
    functions_for(part)
    authority = active.get("user_continuation_authorization_20261027", {})
    if (authority.get("id") != "USER_RESUME_OFFLINE_IFC_20261027"
            or authority.get("status") != "ACTIVE_CURRENT_USER_DIRECTED_SESSION"
            or authority.get("max_simultaneously_assigned_idle_GPUs") != 4
            or authority.get("new_scope_registration_allowed") is not True):
        raise ValueError("Current user continuation authority inactive")
    scopes = [s for s in active.get("experiment_authority", {}).get("current_user_authorized_scopes", [])
              if s.get("id") == ID]
    if len(scopes) != 1:
        raise ValueError("Unique full80 scope registration required")
    scope = scopes[0]
    if (scope.get("protocol_sha256") != digest
            or scope.get("execution_status") not in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
            or scope.get("allowed_run_ids") != run_ids()
            or scope.get("limits_per_part") != PART_LIMITS or scope.get("limits_total") != TOTAL_LIMITS
            or scope.get("max_simultaneously_assigned_idle_GPUs") != 4):
        raise ValueError("Exact protocol/run IDs/part budgets not registered")
    if run_id is not None and run_id not in run_ids(part):
        raise ValueError("Cell run ID belongs outside this independent part")


class Ledger:
    def __init__(self, record, persist=lambda: None, guard=lambda run_id: None):
        self.predebited = record.setdefault("predebited", dict.fromkeys(PART_LIMITS, 0))
        self.observed = record.setdefault("observed", dict.fromkeys(PART_LIMITS, 0))
        self.persist, self.guard = persist, guard

    def reserve(self, cost, run_id):
        self.guard(run_id)
        if any(k not in PART_LIMITS or type(v) is not int or v < 0
               or self.predebited[k] + v > PART_LIMITS[k] for k, v in cost.items()):
            raise ValueError("Operation exceeds part budget before execution")
        for key, value in cost.items():
            self.predebited[key] += value
        self.persist()

    def observe(self, key, amount=1):
        if key not in PART_LIMITS or self.observed[key] + amount > self.predebited[key]:
            raise ValueError("Unpaid actual operation: " + key)
        self.observed[key] += amount


def random_supports():
    import numpy as np
    rng = np.random.default_rng(42)
    return {(e, b): sorted(rng.choice(216, 94, replace=False).tolist()) for e, b in FUNCTIONS}


def select_support(records, expert, block):
    """Equal two steps within source, then equal sixteen training sources."""
    import numpy as np
    selected = [r for r in records if r["expert"] == expert and r["block"] == block
                and r["index"] in TRAIN]
    expected = {(i, s) for i in TRAIN for s in steps_for(expert)}
    if len(selected) != 32 or {(r["index"], r["step"]) for r in selected} != expected:
        raise ValueError("Exactly sixteen training sources times two steps required")
    selected.sort(key=lambda r: (r["index"], r["step"]))
    energy = np.asarray([r["group_energy"] for r in selected], dtype=np.float64)
    if energy.shape != (32, 216) or not np.isfinite(energy).all() or (energy < 0).any():
        raise ValueError("Invalid training-only native group energy")
    mean = energy.reshape(16, 2, 216).mean(1).mean(0)
    groups = sorted(np.lexsort((np.arange(216), -mean))[:94].tolist())
    return groups, mean.tolist()


def training_case(cases, index, step, expert):
    if type(index) is not int or index not in TRAIN or type(step) is not int or step not in steps_for(expert):
        raise ValueError("Held-out source/wrong expert step cannot train")
    return cases[index, step]


def operand_header(path):
    """Check real safetensors headers using bytes only, without Torch/CUDA."""
    expected = {"V1": [256, 5120], "V2": [256, 13824], "Y": [256, 5120],
                "gate": [1, 5120], "b2": [5120]}
    size = path.stat().st_size
    with path.open("rb") as handle:
        length = handle.read(8)
        if len(length) != 8:
            raise ValueError("Truncated safetensors length")
        n = struct.unpack("<Q", length)[0]
        if not 0 < n <= min(1 << 20, size - 8):
            raise ValueError("Invalid operand header length")
        header = json.loads(handle.read(n))
    if set(header) - {"__metadata__"} != set(expected):
        raise ValueError("Exact original five-tensor payload schema required")
    regions = []
    for name, shape in expected.items():
        record = header[name]
        lo, hi = record["data_offsets"]
        if (record["shape"] != shape or record["dtype"] != "BF16"
                or type(lo) is not int or type(hi) is not int
                or not 0 <= lo <= hi <= size - 8 - n or hi - lo != math.prod(shape) * 2):
            raise ValueError("Original operand BF16 shape/storage region differs")
        regions.append((lo, hi))
    regions.sort()
    if regions[0][0] != 0 or regions[-1][1] != size - 8 - n or any(a[1] != b[0] for a, b in zip(regions, regions[1:])):
        raise ValueError("Operand payload data regions overlap or leave gaps")
    return {name: {"shape": shape, "dtype": "BF16"} for name, shape in expected.items()}


def expand_cases(plan, hf, part):
    """The 24 byte-bound original INPUT manifests supply all real row IDs."""
    sources = {s["index"]: s for s in plan["sources"]}
    if (len(sources) != 24 or set(sources) != set(range(24))
            or len({s["source_id"] for s in sources.values()}) != 24
            or len(plan["inputs"]) != 24 or {i["index"] for i in plan["inputs"]} != set(range(24))):
        raise ValueError("Original CAL24 source/input identity differs")
    chosen = set(functions_for(part))
    cases = []
    for item in plan["inputs"]:
        record = json.loads(bound(hf, item).read_text())
        index, source = item["index"], sources[item["index"]]
        if record.get("status") != "PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP" or record.get("index") != index:
            raise ValueError("Original operand producer/input status differs")
        native = record["input"]
        if (native["input_provenance"]["image_sha256"] != source["source_id"]
                or native["source_group"] != source["source_group"] or native["calibration_seed"] != source["seed"]):
            raise ValueError("Source image/prompt/seed identity differs")
        scopes = record["scopes"]
        if len(scopes) != 160 or {(s["step"], s["block"]) for s in scopes} != {(s, b) for s in range(4) for b in range(40)}:
            raise ValueError("Original 160 scopes per source required")
        for scope in scopes:
            expert = EXPERTS[0] if scope["step"] < 2 else EXPERTS[1]
            if scope["expert"] != expert or scope.get("grid") != [21, 29, 52]:
                raise ValueError("Native grid/expert differs")
            if (expert, scope["block"]) not in chosen:
                continue
            rows = scope["selected_rows"]
            if (scope["rows"] != 256 or len(rows) != 256 or len(set(rows)) != 256
                    or any(type(r) is not int or not 0 <= r < 31668 for r in rows)):
                raise ValueError("Actual original 256 row coordinates differ")
            path = Path(scope["path"]).resolve()
            if not path.is_relative_to(hf.resolve()):
                raise ValueError("Original operand path escaped shared HF")
            cases.append({"index": index, "step": scope["step"], "block": scope["block"],
                          "expert": expert, "path": str(path.relative_to(hf.resolve())),
                          "sha256": scope["sha256"], "bytes": scope["bytes"],
                          "rows": rows, "sampling_seed": scope["sampling_seed"],
                          "input_report": item["path"], "input_report_sha256": item["sha256"]})
    expected = {(i, s, b) for i in range(24) for e, b in chosen for s in steps_for(e)}
    if len(cases) != 960 or {(c["index"], c["step"], c["block"]) for c in cases} != expected:
        raise ValueError("This part requires exactly 960 unique source-step-block operands")
    return cases


def validate_plan(plan, view, part):
    expected_parts = [{"part": p, "functions": [{"expert": e, "block": b} for e, b in functions_for(p)],
                       "allowed_run_ids": run_ids(p)} for p in range(4)]
    if (plan.get("parts") != expected_parts
            or plan.get("source_split") != {"training": list(TRAIN), "diagnostic_heldout": list(HELD),
                  "exposure": "ALL_24_HISTORICALLY_EXPOSED_NOT_BLIND_CONFIRMATION"}
            or plan.get("state_distribution") != "saved_native_teacher_post_Q1; student_own_W1_GELU_Smooth2_native_Q2"
            or view.get("schema") != "ffnds.compact6016.full80_asset_view.v1"
            or view.get("status") != "COMPLETE" or view.get("provenance", {}).get("CUDA_initialized") is not False):
        raise ValueError("Full80 design/CPU asset view differs")
    shards = view["shards"]
    if len(shards) != 80 or {(s["expert"], s["block"]) for s in shards} != set(FUNCTIONS):
        raise ValueError("Exactly eighty native expert/block shards required")
    biases = view["bias_tensors"]
    expected_biases = {f"{e}_block{b:02d}_b{n}" for e, b in FUNCTIONS for n in (1, 2)}
    if len(biases) != 160 or {s["tensor_key"] for s in biases} != expected_biases:
        raise ValueError("Exactly 160 original raw native biases required")
    for s in shards:
        for name, dtype, shape in (("W1", "BF16", [13824, 5120]), ("W2", "BF16", [5120, 13824]),
                                    ("smooth1", "F32", [5120]), ("smooth2", "F32", [13824])):
            entry = s["tensors"][name]
            if entry["dtype"] != dtype or entry["shape"] != shape or len(entry["sha256"]) != 64:
                raise ValueError("Original tensor dtype/header/SHA contract differs")
        if set(s["gate_sha_by_step"]) != {str(step) for step in steps_for(s["expert"])}:
            raise ValueError("Both native steps require separate gate byte identities")
    expected_random = [{"expert": e, "block": b, "groups": groups}
                       for (e, b), groups in random_supports().items()]
    if plan.get("random_supports") != expected_random:
        raise ValueError("Seed42 independent random support draws differ")
    functions_for(part)


def _support_observer(torch, ledger):
    from torch.utils._python_dispatch import TorchDispatchMode
    class SupportObserver(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.bmm.default:
                if any(t.dtype != torch.float32 for t in args[:2]):
                    raise RuntimeError("Original contribution BMM must be FP32")
                ledger.observe("support_FP32_batched_bmm")
                ledger.observe("support_logical_group_products", int(args[0].shape[0]))
            elif func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
                raise RuntimeError("Unexpected non-BMM support computation")
            return func(*args, **(kwargs or {}))
    return SupportObserver()


def source_cluster_means(records, expert, metric):
    """Step averages are repeated observations inside each of 24 sources."""
    result = []
    for index in range(24):
        rows = [r for r in records if r["index"] == index]
        if len(rows) != 2 or {r["step"] for r in rows} != set(steps_for(expert)):
            raise ValueError("Exactly two within-source step observations required")
        values = [r[metric] for r in rows]
        value = None if any(v is None for v in values) else sum(values) / 2
        result.append({"index": index, "value": value})
    return result


def run(args):
    plan, digest = json.loads(args.protocol.read_text()), sha(args.protocol)
    active_path = ROOT / "research/ACTIVE_RESEARCH.json"
    authorize(plan, digest, json.loads(active_path.read_text()), args.part)
    hf, private = args.hf_root.resolve(), args.weight_hf_root.resolve()
    from worker_identity import require_worker
    require_worker()
    output_root = (hf / plan["output_HF_relative"]).resolve()
    out = output_root / f"part{args.part}"
    if output_root == hf or not output_root.is_relative_to(hf):
        raise ValueError("Output escaped shared HF")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ATTEMPT.json").open("x") as handle:
        json.dump({"id": ID, "part": args.part, "protocol_sha256": digest, "automatic_retry": False}, handle)
    report = {"id": ID, "part": args.part, "protocol_sha256": digest, "status": "STARTED",
              "functions": [], "cells": [], "source_split": plan["source_split"],
              "source_independent_N": 24, "training_independent_N": 16, "diagnostic_heldout_independent_N": 8,
              "quality_or_speed_acceptance": False, "model_scope": "independent FFNs only; no model rollout or video"}
    budget = {"id": ID, "part": args.part, "protocol_sha256": digest}
    ledger = Ledger(budget, persist=lambda: write(out / "BUDGET.json", budget),
                    guard=lambda run_id: authorize(plan, digest, json.loads(active_path.read_text()), args.part, run_id))
    started, torch, monitor = time.monotonic(), None, None
    current_row, current_path = None, None

    def check():
        if monitor is not None:
            monitor.check()
            if any(s.get("physical_GPU_used_memory_MiB", 0) > 32768 for s in monitor.samples):
                raise RuntimeError("Physical GPU memory budget")
        if torch is not None and torch.cuda.is_initialized() and torch.cuda.max_memory_reserved() > 32 << 30:
            raise RuntimeError("Torch reserved GPU memory budget")
        if shutil.disk_usage(out).free < RESOURCE_LIMITS["free_disk_bytes"]:
            raise RuntimeError("HF disk reserve")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink()) > 6 << 30:
            raise RuntimeError("Part HF output budget")

    def timeout(*_):
        raise TimeoutError("2400 second compute wall")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(2400)
    try:
        versions = {k: importlib.metadata.version(k) for k in ("torch", "numpy", "safetensors")}
        if versions != plan["runtime_versions"]:
            raise ValueError("Actual runtime versions differ")
        required = {"tools/e0_compact6016_full80_bank.py", "tests/test_e0_compact6016_full80_bank.py",
                    "tools/e0_compact6016_recovery_pilot.py", "tools/compact6016_adapter.py",
                    "tools/generate_proxy_pr2.py", "tools/worker_identity.py",
                    "tools/startup_gpu_reservation.py", "tools/assigned_gpu_resources.py"}
        if not required <= {x["path"] for x in plan["repo_sources"]}:
            raise ValueError("Required frozen implementation bindings missing")
        for item in plan["repo_sources"]:
            bound(ROOT, item)
        view = json.loads(bound(hf, plan["asset_view"]).read_text())
        validate_plan(plan, view, args.part)
        producer = json.loads(bound(hf, plan["asset_producer_RUN"]).read_text())
        if producer.get("status") != "COMPLETE" or producer.get("manifest_sha256") != plan["asset_view"]["sha256"] or producer.get("CUDA_initialized") is not False:
            raise ValueError("CPU asset producer receipt differs")
        independent = json.loads(bound(hf, plan["asset_independent_validation"]).read_text())
        if (independent.get("status") != "PASS_INDEPENDENT_METADATA_AND_BIAS_BYTE_VALIDATION"
                or independent.get("manifest_sha256") != plan["asset_view"]["sha256"]
                or independent.get("bias_file_sha256") != view["bias_file"]["sha256"]
                or independent.get("original_weight_entries") != 80
                or independent.get("original_bias_tensors_byte_verified") != 160
                or independent.get("torch_or_CUDA_imported") is not False
                or independent.get("model_QDQ_GEMM_calls") != 0):
            raise ValueError("Independent actual metadata/bias validation receipt differs")
        bound(hf, view["bias_file"])
        chosen_functions = functions_for(args.part)
        specs = [s for s in view["shards"] if (s["expert"], s["block"]) in chosen_functions]
        for spec in specs:
            bound(private, spec)
        cases = expand_cases(plan, hf, args.part)
        for case in cases:
            operand_header(bound(hf, case))
        snap = out / "EXECUTED_SOURCES"
        snap.mkdir()
        shutil.copyfile(args.protocol, snap / "protocol.json")
        shutil.copyfile(hf / plan["asset_view"]["path"], snap / "asset_view.json")
        for item in plan["repo_sources"]:
            path = snap / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / item["path"], path)
        report["asset_preflight"] = {"part_original_shards_SHA_verified": 20, "part_payload_SHA_verified": 960,
            "original_INPUT_manifests_SHA_verified": 24, "all_payload_headers": "ALL_960_VERIFIED_BEFORE_FIRST_FFN",
            "asset_producer_process_exit_evidence": plan["asset_producer_process_exit_evidence"]}
        check()
        import torch as torch_api
        torch = torch_api
        import numpy as np
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        from startup_gpu_reservation import reserve_before_assets
        from compact6016_adapter import CompactAdapters, compact_forward, error_metrics, gated_loss, same_bits
        reservation, monitor, receipt = reserve_before_assets(torch)
        report["reservation"] = receipt
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("One exclusively assigned A100 per part required")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
            raise ValueError("TF32 override conflicts with original FP32 computations")
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
        for name in ("Wan2.2-I2V-A14B-W4A4", "HiF4_Sp"):
            sys.path.insert(0, str(ROOT / ".runtime/code" / name))
        for name in ("HiFloat4", "HiFloat4/hif4_gpu"):
            sys.path.insert(0, str(ROOT / ".runtime/code/HiF4_Sp" / name))
        from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq, get_wan_hif4_runtime_counters
        import quant.hif4_backend as native_backend
        observer = pilot._matmul_observer(torch, ledger)
        before_setup = {"adapter": get_wan_hif4_runtime_counters(), "backend": native_backend.get_hif4_stats()}
        with observer:
            report["actual_native_quantizer"] = pilot.verify_native_quantizer(native_backend, plan)
        after_setup = {"adapter": get_wan_hif4_runtime_counters(), "backend": native_backend.get_hif4_stats()}
        if before_setup != after_setup or any(v for counts in after_setup.values() for v in counts.values()):
            raise RuntimeError("Unpaid native API setup model operation")
        report["native_API_setup"] = {"after_GPU_reservation": True, "counters_before": before_setup,
            "counters_after": after_setup, "CUDA_context_dependency": "original quant package optional T5 class default current_device; no FFN"}
        counters_before, backend_before = get_wan_hif4_runtime_counters(), native_backend.get_hif4_stats()
        from generate_proxy_pr2 import contribution_scores
        support_observer = _support_observer(torch, ledger)
        report["runtime"] = {"versions": versions, "CUDA": torch.version.cuda, "GPU": torch.cuda.get_device_name(0),
                             "TF32": False, "float32_matmul_precision": torch.get_float32_matmul_precision()}

        def tensor_sha(tensor):
            import hashlib
            return hashlib.sha256(memoryview(tensor.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()

        def quantizer(value, *, force_fp32):
            ledger.observe("native_Q2_calls")
            return native_hif4_activation_qdq(value, force_fp32=force_fp32)

        def merge(model):
            ledger.observe("merged_weight_pairs")
            with observer.during("merge"):
                return model.merged_weights()

        def forward(case, weights, b1, b2, smooth, *, training=False):
            ledger.observe("student_forwards")
            with observer.during("forward"):
                return compact_forward(case["V1"].to("cuda"), weights[0], b1, weights[1], b2, smooth,
                                       quantizer, training=training, return_trace=True)

        def loss(output, case):
            ledger.observe("loss_evaluations")
            return gated_loss(output, case["Y"].to("cuda"), case["gate"].to("cuda"))

        def backward(value, model, *, nonzero_B=False):
            ledger.observe("backward_calls")
            with observer.during("backward"):
                value.backward()
            norms = {}
            for name, parameter in model.named_parameters():
                ledger.observe("finite_gradient_tensor_checks")
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("Missing/nonfinite adapter gradient")
                norms[name] = float(parameter.grad.float().square().sum().sqrt())
                if nonzero_B and "lora_B" in name and norms[name] == 0:
                    raise RuntimeError("Dead W1/W2 LoRA gradient")
            return norms

        def reserve_forward(run_id, **extra):
            ledger.reserve({"student_forwards": 1, "native_Q2_calls": 1, "BF16_forward_GEMM": 2, **extra}, run_id)

        bias_values = load_file(str(hf / view["bias_file"]["path"]), device="cpu")
        if set(bias_values) != {b["tensor_key"] for b in view["bias_tensors"]}:
            raise ValueError("Exact 160 original biases required")
        for spec in view["bias_tensors"]:
            value = bias_values[spec["tensor_key"]]
            if value.dtype != torch.bfloat16 or list(value.shape) != spec["shape"] or tensor_sha(value) != spec["tensor_sha256"]:
                raise ValueError("Raw native bias bytes differ")
        sources = {s["index"]: s for s in plan["sources"]}
        random_map = random_supports()
        for expert, block in chosen_functions:
            check()
            spec = next(s for s in specs if (s["expert"], s["block"]) == (expert, block))
            raw = torch.load(private / spec["path"], map_location="cpu", weights_only=True, mmap=True)[expert]
            up, down = raw[f"blocks.{block}.ffn.0"], raw[f"blocks.{block}.ffn.2"]
            for name, value in (("W1", up["weight_qdq"]), ("W2", down["weight_qdq"]),
                                ("smooth1", up["smooth_scale"]), ("smooth2", down["smooth_scale"])):
                expected = spec["tensors"][name]
                dtype = torch.bfloat16 if name.startswith("W") else torch.float32
                if value.dtype != dtype or list(value.shape) != expected["shape"] or tensor_sha(value) != expected["sha256"] or not bool(torch.isfinite(value).all()):
                    raise ValueError("Original full80 weight/smooth tensor identity differs")
                if name.startswith("smooth") and not bool((value > 0).all()):
                    raise ValueError("Original positive smooth required")
            if any(m.get("activation_qdq") is not True or m.get("force_fp32") is not True for m in (up, down)):
                raise ValueError("Original Smooth/native QDQ module contract differs")
            b1_full = bias_values[f"{expert}_block{block:02d}_b1"]
            b2 = bias_values[f"{expert}_block{block:02d}_b2"].to("cuda")
            local_cases, local_refs = {}, {}
            for case_ref in [c for c in cases if (c["expert"], c["block"]) == (expert, block)]:
                with safe_open(str(hf / case_ref["path"]), framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != {"V1", "V2", "Y", "gate", "b2"}:
                        raise ValueError("Original operand payload schema differs")
                    data = {name: handle.get_tensor(name) for name in ("V1", "Y", "gate", "b2")}
                    v2_header = handle.get_slice("V2")
                    if v2_header.get_shape() != [256, 13824] or v2_header.get_dtype() != "BF16":
                        raise ValueError("Native teacher V2 header differs")
                for name, shape in (("V1", (256, 5120)), ("Y", (256, 5120)), ("gate", (1, 5120)), ("b2", (5120,))):
                    value = data[name]
                    if value.dtype != torch.bfloat16 or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                        raise ValueError("Real V1/Y/gate/b2 header/finiteness differs")
                if tensor_sha(data["gate"]) != spec["gate_sha_by_step"][str(case_ref["step"])] or not same_bits(data["b2"], b2.cpu()):
                    raise ValueError("Step-specific gate/native b2 differs")
                key = case_ref["index"], case_ref["step"]
                local_cases[key], local_refs[key] = data, case_ref
            # The support observer is independent of the frozen pilot training
            # observer, whose original no-BMM rule remains unmodified.
            support_records = []
            support_run_id = function_id(expert, block) + "_energy94"
            support_path = out / (function_id(expert, block) + "_SUPPORT.json")
            full_w2 = down["weight_qdq"].to("cuda")
            for index in TRAIN:
                for step in steps_for(expert):
                    check()
                    case_ref = local_refs[index, step]
                    with safe_open(str(hf / case_ref["path"]), framework="pt", device="cpu") as handle:
                        v2 = handle.get_tensor("V2")
                    if not bool(torch.isfinite(v2).all()):
                        raise ValueError("Nonfinite original training V2")
                    ledger.reserve({"support_source_step_calls": 1, "support_FP32_batched_bmm": 27,
                                    "support_logical_group_products": 216}, support_run_id)
                    ledger.observe("support_source_step_calls")
                    before = ledger.observed["support_FP32_batched_bmm"]
                    with torch.no_grad(), support_observer:
                        energy = contribution_scores(v2.to("cuda"), full_w2, local_cases[index, step]["gate"].to("cuda"))
                    if ledger.observed["support_FP32_batched_bmm"] - before != 27 or energy.shape != (256, 216) or not np.isfinite(energy).all():
                        raise RuntimeError("Original 27-BMM source contribution differs")
                    support_records.append({"index": index, "source_id": sources[index]["semantic_source_id"],
                        "expert": expert, "step": step, "block": block, "source_state": "teacher_input",
                        "payload_sha256": case_ref["sha256"], "group_energy": energy.mean(0).tolist()})
                    write(support_path, {"schema": "ffnds.compact6016.train_source_support.v1",
                        "status": "COLLECTING_NO_RETRY", "records": support_records, "heldout_used_for_support": False})
                    del v2, energy
            groups, mean_energy = select_support(support_records, expert, block)
            write(support_path, {"schema": "ffnds.compact6016.train_source_support.v1", "status": "COMPLETE", "records": support_records,
                "energy_groups": groups, "source_then_step_equal_mean_energy": mean_energy,
                "heldout_used_for_support": False, "native_FP32_batched_bmm": 864})
            report["functions"].append({"expert": expert, "block": block,
                "steps": list(steps_for(expert)), "support_file": {"path": str(support_path.relative_to(hf)),
                "sha256": sha(support_path), "bytes": support_path.stat().st_size},
                "energy_groups": groups, "random_groups": random_map[expert, block]})
            write(out / "RUN.json", report)
            del full_w2, support_records
            for init in INITIALIZATIONS:
                run_id = function_id(expert, block) + "_" + init
                selected = groups if init == "energy94" else random_map[expert, block]
                row = {"run_id": run_id, "expert": expert, "block": block, "steps": list(steps_for(expert)),
                       "initialization": init, "groups": selected, "status": "STARTED", "diagnostics": {},
                       "epoch_source_step_loss_means": []}
                cell_dir = out / "cells" / run_id
                cell_dir.mkdir(parents=True)
                current_row, current_path = row, cell_dir / "RESULT.json"
                write(current_path, row)
                ids = torch.tensor([64 * g + j for g in selected for j in range(64)], dtype=torch.int64)
                w1 = up["weight_qdq"].index_select(0, ids).contiguous().to("cuda")
                w2 = down["weight_qdq"].index_select(1, ids).contiguous().to("cuda")
                b1 = b1_full.index_select(0, ids).contiguous().to("cuda")
                smooth = down["smooth_scale"].index_select(0, ids).contiguous().to("cuda")
                with observer:
                    ledger.reserve({"LoRA_random_factor_draws": 2}, run_id)
                    model = CompactAdapters(w1, w2, rank=32, seed=42,
                                            on_random=lambda: ledger.observe("LoRA_random_factor_draws"))
                    ledger.reserve({"merged_weight_pairs": 1, "FP32_merge_GEMM": 2}, run_id)
                    zero_weights = merge(model)
                    row["zero_merge_base_weight_bitwise"] = [same_bits(z, w) for z, w in zip(zero_weights, (w1, w2))]
                    gate_case = local_cases[0, steps_for(expert)[0]]
                    row["GPU_gate_case"] = {"index": 0, "step": steps_for(expert)[0]}
                    reserve_forward(run_id, GPU_gate_reference_forwards=1)
                    ledger.observe("GPU_gate_reference_forwards")
                    with torch.no_grad():
                        reference, reference_trace = forward(gate_case, (w1, w2), b1, b2, smooth)
                    reserve_forward(run_id, GPU_gate_STE_forwards=1, loss_evaluations=1, backward_calls=1,
                                    BF16_backward_GEMM=3, FP32_LoRA_backward_GEMM=4, finite_gradient_tensor_checks=4)
                    ledger.observe("GPU_gate_STE_forwards")
                    output, trace = forward(gate_case, zero_weights, b1, b2, smooth, training=True)
                    byte_checks = {name: same_bits(trace[name], reference_trace[name]) for name in trace}
                    byte_checks["pre_gate_output"] = same_bits(output, reference)
                    row["GPU_zero_LoRA_and_native_STE_byte_checks"] = byte_checks
                    if not all(byte_checks.values()):
                        raise RuntimeError("Original selected zero-LoRA/native STE GPU byte gate failed")
                    value = loss(output, gate_case)
                    if not bool(torch.isfinite(value)):
                        raise RuntimeError("Nonfinite GPU gate loss")
                    row["GPU_gate_gradient_norms"] = backward(value, model, nonzero_B=True)
                    model.zero_grad(set_to_none=True)
                    initial_weights = tuple(w.detach() for w in zero_weights)
                    del zero_weights, reference, reference_trace, output, trace, value

                    def diagnose(label, weights):
                        records = []
                        row["diagnostics"][label] = records
                        for index in range(24):
                            for step in steps_for(expert):
                                check()
                                case = local_cases[index, step]
                                reserve_forward(run_id, diagnostic_source_batches=1, loss_evaluations=1)
                                ledger.observe("diagnostic_source_batches")
                                with torch.no_grad():
                                    output, trace = forward(case, weights, b1, b2, smooth)
                                    value = loss(output, case)
                                    if not bool(torch.isfinite(value)):
                                        raise RuntimeError("Nonfinite diagnostic loss")
                                    metrics = error_metrics(output, case["Y"].to("cuda"), case["gate"].to("cuda"))
                                records.append({"index": index, "step": step,
                                    "source_id": sources[index]["semantic_source_id"],
                                    "role": "TRAIN" if index in TRAIN else "HELDOUT_EXPOSED_DIAGNOSTIC",
                                    "optimization_loss": float(value), **metrics,
                                    "output_tensor_sha256": tensor_sha(output),
                                    "student_pre_Q2_sha256": tensor_sha(trace["pre_Q2"]),
                                    "student_post_Q2_sha256": tensor_sha(trace["post_Q2"]),
                                    "student_Q2_shape": [256, 6016]})
                        row["diagnostics"][label] = records
                        row.setdefault("source_cluster_means", {})[label] = {
                            metric: source_cluster_means(records, expert, metric) for metric in (
                                "gated_pre_gate_relative_SSE", "actual_BF16_gate_relative_SSE",
                                "source_row_centered_relative_SSE", "source_row_mean_error_fraction")}
                        write(current_path, row)

                    diagnose("initial", initial_weights)
                    del initial_weights
                    optimizer = torch.optim.Adam(model.parameters(), lr=.0005, betas=(.9, .999),
                                                 eps=1e-8, weight_decay=0, foreach=False, fused=False)
                    for epoch in range(8):
                        losses = []
                        for index in TRAIN:
                            for step in steps_for(expert):
                                check()
                                case = training_case(local_cases, index, step, expert)
                                ledger.reserve({"training_source_batches": 1, "merged_weight_pairs": 1,
                                    "FP32_merge_GEMM": 2, "student_forwards": 1, "native_Q2_calls": 1,
                                    "BF16_forward_GEMM": 2, "loss_evaluations": 1, "backward_calls": 1,
                                    "BF16_backward_GEMM": 3, "FP32_LoRA_backward_GEMM": 4,
                                    "finite_gradient_tensor_checks": 4, "finite_updated_parameter_tensor_checks": 4,
                                    "Adam_steps": 1, "Adam_parameter_tensor_updates": 4}, run_id)
                                ledger.observe("training_source_batches")
                                optimizer.zero_grad(set_to_none=True)
                                weights = merge(model)
                                output, trace = forward(case, weights, b1, b2, smooth, training=True)
                                value = loss(output, case)
                                if not bool(torch.isfinite(value)):
                                    raise RuntimeError("Nonfinite training loss")
                                backward(value, model)
                                ledger.observe("Adam_steps")
                                ledger.observe("Adam_parameter_tensor_updates", 4)
                                optimizer.step()
                                for parameter in model.parameters():
                                    ledger.observe("finite_updated_parameter_tensor_checks")
                                    if not bool(torch.isfinite(parameter).all()):
                                        raise RuntimeError("Nonfinite updated adapter parameter")
                                losses.append(float(value.detach()))
                                del weights, output, trace, value
                        row["epoch_source_step_loss_means"].append(sum(losses) / 32)
                        write(current_path, row)
                        print(run_id, "epoch", epoch + 1, "loss", row["epoch_source_step_loss_means"][-1], flush=True)
                    ledger.reserve({"merged_weight_pairs": 1, "FP32_merge_GEMM": 2}, run_id)
                    with torch.no_grad():
                        final_weights = tuple(w.detach().contiguous() for w in merge(model))
                    ledger.reserve({"exported_students": 1, "exported_final_BF16_matrices": 2}, run_id)
                    export = {"W1": final_weights[0].cpu(), "W2": final_weights[1].cpu(), "b1": b1.cpu(),
                              "b2": b2.cpu(), "smooth1": up["smooth_scale"].contiguous(),
                              "smooth2": smooth.cpu(), "groups": torch.tensor(selected, dtype=torch.int64), "channels": ids}
                    export.update({"training_" + k: v.detach().cpu().contiguous() for k, v in model.named_parameters()})
                    export_path = cell_dir / "folded.safetensors"
                    save_file(export, str(export_path), metadata={"id": ID, "run_id": run_id, "expert": expert,
                        "block": str(block), "steps": ",".join(map(str, steps_for(expert))),
                        "deployment": "two folded BF16 matrices; native Smooth/QDQ; no online LoRA",
                        "pilot_input_boundary": "native saved post-Q1 BF16; upstream smooth1 retained, not repeated"})
                    reloaded = load_file(str(export_path), device="cpu")
                    if set(reloaded) != set(export) or any(not same_bits(reloaded[k], export[k]) for k in export):
                        raise RuntimeError("Folded pack byte-reload gate failed")
                    ledger.observe("exported_students")
                    ledger.observe("exported_final_BF16_matrices", 2)
                    row["export"] = {"path": str(export_path.relative_to(hf)), "sha256": sha(export_path),
                        "bytes": export_path.stat().st_size, "reloaded_all_tensor_bytes_equal": True,
                        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": tensor_sha(v)} for k, v in export.items()},
                        "online_LoRA_GEMM": 0, "online_weight_QDQ": 0}
                    del final_weights
                    final_weights = (reloaded["W1"].to("cuda"), reloaded["W2"].to("cuda"))
                    diagnose("final_folded_BF16", final_weights)
                    row["status"] = "COMPLETE_256_UPDATES_AND_INITIAL_FINAL_48_SOURCE_STEP_DIAGNOSTICS"
                    write(current_path, row)
                    report["cells"].append({"run_id": run_id, "expert": expert, "block": block,
                        "initialization": init, "status": row["status"], "result": {
                            "path": str(current_path.relative_to(hf)), "sha256": sha(current_path), "bytes": current_path.stat().st_size},
                        "export": row["export"]})
                    write(out / "RUN.json", report)
                    del model, optimizer, final_weights, export, reloaded, w1, w2, b1, smooth
                    current_row, current_path = None, None
                    check()
            del raw, up, down, local_cases, local_refs, b2
        if ledger.predebited != PART_LIMITS or ledger.observed != PART_LIMITS:
            raise RuntimeError("Actual part operation ledger incomplete/differs")
        counter_delta = {k: get_wan_hif4_runtime_counters()[k] - counters_before[k] for k in counters_before}
        backend_delta = {k: native_backend.get_hif4_stats()[k] - backend_before[k] for k in backend_before}
        if any(d["weight_qdq_calls"] != 0 or d["activation_qdq_calls"] != PART_LIMITS["native_Q2_calls"] for d in (counter_delta, backend_delta)):
            raise RuntimeError("Independent native Q2 counters disagree")
        report.update(status="COMPLETE_FULL80_BANK_PART", elapsed_seconds=time.monotonic() - started,
            native_quantizer_counter_delta=counter_delta, native_backend_counter_delta=backend_delta,
            predebited=ledger.predebited, observed=ledger.observed,
            peak_torch_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
            student_quantization={"every_forward_recomputed_from_student_GELU": True, "group_hierarchy": [64, 8, 4],
                                  "internal_codes_scales": "NOT_EXPOSED_BY_ORIGINAL_NATIVE_QDQ_API"})
        check()
    except BaseException as error:
        report.update(status="FAILED_NO_RETRY", error_type=type(error).__name__, error=str(error))
        if current_row is not None:
            current_row.update(status="FAILED_NO_RETRY", error_type=type(error).__name__, error=str(error))
            write(current_path, current_row)
        raise
    finally:
        original_error = sys.exc_info()[0] is not None
        signal.alarm(0)
        monitor_error = None
        if monitor is not None:
            try:
                monitor.close()
                report["assigned_gpu_resources"] = monitor.report()
                if report["assigned_gpu_resources"].get("violations"):
                    raise RuntimeError("Assigned GPU monitor violation")
            except BaseException as error:
                report["status"] = "FAILED_GPU_MONITOR"
                monitor_error = error
        report["predebited"], report["observed"] = ledger.predebited, ledger.observed
        report["outer_process_completion"] = "REQUIRES_LAUNCHER_PROCESS_EXIT_RECEIPT"
        write(out / "BUDGET.json", budget)
        write(out / "RUN.json", report)
        if monitor_error is not None and not original_error:
            raise monitor_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--part", type=int, choices=range(4), required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--weight-hf-root", type=Path, required=True)
    run(parser.parse_args())
