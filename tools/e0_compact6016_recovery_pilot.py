"""Registered four-site, two-initialization ordinary 6016 FFN LoRA pilot.

Preparation and CPU tests do not authorize execution. The exact protocol SHA,
eight run IDs, assets, budgets and current user authority must be in ACTIVE.
There is one exclusive attempt, no resume, fallback, retry, or checkpoint
selection. Historical Whole94/UB1 fitting banks are never opened.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_COMPACT6016_RECOVERY_PILOT_20261028"
POSITIONS = ((0, 0), (0, 39), (3, 0), (3, 39))
INITIALIZATIONS = ("energy94", "random94")
TRAIN = tuple(range(16))
HELD = tuple(range(16, 24))
RESOURCE_LIMITS = {"GPU": 1, "global_max_GPU": 4, "CPU_threads": 2,
                   "compute_wall_seconds": 1200, "external_wall_seconds": 1500,
                   "new_HF_bytes": 5 * (1 << 28), "free_disk_bytes": 10 << 30,
                   "peak_GPU_reserved_bytes": 32 << 30}
TRAINING = {"rank": 32, "seed": 42, "epochs": 8, "steps_per_cell": 128,
            "rows_per_batch": 256, "source_order": list(TRAIN),
            "optimizer": "Adam", "lr": .0005, "betas": [.9, .999],
            "eps": 1e-8, "weight_decay": 0, "foreach": False, "fused": False,
            "gradient_clipping": False, "checkpoint_selection": False,
            "weight_requantization": False, "train_bias": False,
            "train_smooth": False, "lora_scale": 1.0}


def expected_run_ids():
    return [f"compact6016_s{s}_b{b:02d}_{init}" for s, b in POSITIONS
            for init in INITIALIZATIONS]


def total_limits():
    # Per cell: 2 GPU-gate forwards + 24 initial + 128 train + 24 final;
    # 1 gate backward + 128 update backwards; initial/128/final merge pairs.
    cells, forwards, backwards, merges = 8, 178, 129, 130
    return {"student_forwards": cells * forwards,
            "native_Q2_calls": cells * forwards,
            "BF16_forward_GEMM": cells * 2 * forwards,
            "FP32_merge_GEMM": cells * 2 * merges,
            "merged_weight_pairs": cells * merges,
            "backward_calls": cells * backwards,
            "BF16_backward_GEMM": cells * 3 * backwards,
            "FP32_LoRA_backward_GEMM": cells * 4 * backwards,
            "loss_evaluations": cells * (1 + 128 + 48),
            "finite_gradient_tensor_checks": cells * 4 * backwards,
            "finite_updated_parameter_tensor_checks": cells * 4 * 128,
            "Adam_steps": cells * 128,
            "Adam_parameter_tensor_updates": cells * 4 * 128,
            "training_source_batches": cells * 128,
            "diagnostic_source_batches": cells * 48,
            "GPU_gate_reference_forwards": cells,
            "GPU_gate_STE_forwards": cells,
            "LoRA_random_factor_draws": cells * 2,
            "exported_students": cells,
            "exported_final_BF16_matrices": cells * 2,
            "weight_QDQ_calls": 0, "Q1_calls": 0,
            "teacher_FFN_calls": 0, "teacher_V2_forwards": 0,
            "weight_decompositions": 0, "full_model_calls": 0,
            "video_generations": 0, "official_quality_calls": 0}


LIMITS = total_limits()


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for value in iter(lambda: handle.read(4 << 20), b""):
            digest.update(value)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def bound(root, item):
    root = Path(root).resolve()
    path = (root / item["path"]).resolve()
    if (path == root or not path.is_relative_to(root) or not path.is_file()
            or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]):
        raise ValueError("Bound asset differs: " + item["path"])
    return path


def authorize(plan, digest, active, run_id=None):
    if (plan.get("id") != ID or plan.get("assets_finalized") is not True
            or plan.get("positions") != [list(p) for p in POSITIONS]
            or plan.get("initializations") != list(INITIALIZATIONS)
            or plan.get("limits") != LIMITS or plan.get("training") != TRAINING
            or plan.get("resource_limits") != RESOURCE_LIMITS
            or plan.get("allowed_run_ids") != expected_run_ids()
            or plan.get("automatic_retry") is not False):
        raise ValueError("Exact finalized scope/resource/training contract differs")
    authority = active.get("user_continuation_authorization_20261027", {})
    if (authority.get("id") != "USER_RESUME_OFFLINE_IFC_20261027"
            or authority.get("status") != "ACTIVE_CURRENT_USER_DIRECTED_SESSION"
            or authority.get("max_simultaneously_assigned_idle_GPUs") != 4
            or authority.get("new_scope_registration_allowed") is not True):
        raise ValueError("Current user authority inactive")
    scopes = [s for s in active.get("experiment_authority", {}).get(
        "current_user_authorized_scopes", []) if s.get("id") == ID]
    if len(scopes) != 1:
        raise ValueError("Unique registered scope required")
    scope = scopes[0]
    if (scope.get("protocol_sha256") != digest
            or scope.get("execution_status") not in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
            or scope.get("allowed_run_ids") != expected_run_ids()
            or scope.get("limits") != LIMITS
            or scope.get("max_simultaneously_assigned_idle_GPUs") != 1):
        raise ValueError("Exact protocol/run IDs/budget not registered")
    if run_id is not None and run_id not in scope["allowed_run_ids"]:
        raise ValueError("Unregistered cell run ID")


class BudgetLedger:
    """Atomic predebit; actual entered operations are counted separately."""
    def __init__(self, report, persist=lambda: None, guard=lambda run_id: None):
        self.report, self.persist, self.guard = report, persist, guard
        self.predebited = report.setdefault("predebited", dict.fromkeys(LIMITS, 0))
        self.observed = report.setdefault("observed", dict.fromkeys(LIMITS, 0))

    def reserve(self, cost, run_id):
        self.guard(run_id)
        if any(k not in LIMITS or type(v) is not int or v < 0
               or self.predebited[k] + v > LIMITS[k] for k, v in cost.items()):
            raise ValueError("Operation exceeds frozen budget before execution")
        for key, value in cost.items():
            self.predebited[key] += value
        self.persist()

    def observe(self, key, value=1):
        if key not in LIMITS or self.observed[key] + value > self.predebited[key]:
            raise ValueError("Unpaid actual operation: " + key)
        self.observed[key] += value


def select_energy_groups(profile, step, block):
    """Read energy arrays from the sixteen training sources only."""
    import numpy as np
    records = [r for r in profile["records"]
               if (r["step"], r["block"]) == (step, block) and r["index"] in TRAIN]
    if len(records) != 16 or {r["index"] for r in records} != set(TRAIN):
        raise ValueError("Exactly sixteen training source vectors required")
    records.sort(key=lambda r: r["index"])
    vectors = np.asarray([r["group_energy"] for r in records], dtype=np.float64)
    if vectors.shape != (16, 216) or not np.isfinite(vectors).all() or (vectors < 0).any():
        raise ValueError("Invalid training group statistics")
    energy = vectors.mean(0)
    return sorted(np.lexsort((np.arange(216), -energy))[:94].tolist())


def random_groups_by_position():
    import numpy as np
    generator = np.random.default_rng(42)
    return {p: sorted(generator.choice(216, 94, replace=False).tolist()) for p in POSITIONS}


def validate_plan(plan, profile):
    sources, cases = plan["sources"], plan["cases"]
    expected = {(i, s, b) for i in range(24) for s, b in POSITIONS}
    if (plan.get("source_split") != {"training": list(TRAIN), "diagnostic_heldout": list(HELD),
                                    "exposure": "ALL_24_HISTORICALLY_EXPOSED_NOT_BLIND_CONFIRMATION"}
            or len(sources) != 24 or {s["index"] for s in sources} != set(range(24))
            or len({s["source_id"] for s in sources}) != 24
            or len(cases) != 96 or {(c["index"], c["step"], c["block"]) for c in cases} != expected
            or profile.get("schema") != "ffnds.calibration.source_statistics.v2"
            or profile.get("evidence_kind") != "ACTUAL_NATIVE_OBSERVATION"):
        raise ValueError("Source split/96 real operands/profile differs")
    by_index = {s["index"]: s for s in sources}
    records = profile["records"]
    if len(records) != 96 or {(r["index"], r["step"], r["block"]) for r in records} != expected:
        raise ValueError("Exact source-profile coverage differs")
    case_map = {(c["index"], c["step"], c["block"]): c for c in cases}
    for record in records:
        source = by_index[record["index"]]
        case = case_map[record["index"], record["step"], record["block"]]
        if (record["source_id"] != source["semantic_source_id"]
                or record["image_id"] != source["source_id"]
                or record["seed"] != source["seed"] or record["source_state"] != "teacher_input"
                or record["payload_sha256"] != case["sha256"]
                or record["statistic_semantics"] != "native_gate_times_W2_group_contribution_l2_squared"
                or record["normalization"] != "mean_over_256_saved_rows"):
            raise ValueError("Actual profile/source/payload identity differs")
    for case in cases:
        if (case["expert"] != ("high_noise_model" if case["step"] == 0 else "low_noise_model")
                or len(case["rows"]) != 256 or len(set(case["rows"])) != 256
                or any(type(r) is not int or r < 0 or r >= 31668 for r in case["rows"])):
            raise ValueError("Actual native rows/expert differ")
    cells = plan["cells"]
    if len(cells) != 8 or {c["run_id"] for c in cells} != set(expected_run_ids()):
        raise ValueError("Exact eight cells required")
    random_groups = random_groups_by_position()
    for cell in cells:
        position = cell["step"], cell["block"]
        init = cell["initialization"]
        if position not in POSITIONS or init not in INITIALIZATIONS:
            raise ValueError("Unknown cell")
        expected_groups = (select_energy_groups(profile, *position) if init == "energy94"
                           else random_groups[position])
        if (cell["groups"] != expected_groups
                or cell["run_id"] != f"compact6016_s{position[0]}_b{position[1]:02d}_{init}"):
            raise ValueError("Frozen training-only/random support differs")
    if (len(plan["weights"]) != 4 or {(w["step"], w["block"]) for w in plan["weights"]} != set(POSITIONS)
            or len(plan["bias_tensors"]) != 8):
        raise ValueError("Original weight/bias identity incomplete")


def training_case(cases, index):
    if type(index) is not int or index not in TRAIN:
        raise ValueError("Held-out source cannot enter training")
    return cases[index]


def verify_native_quantizer(backend, plan):
    """Bind the actually imported compiled implementation; reject fallback."""
    function, qtype, api_name = backend.get_hif4_api()
    if api_name != "HiFloat4.main":
        raise ValueError("Original HiFloat4.main API required; no fallback")
    module = sys.modules[function.__module__]
    if not getattr(module, "CUDA_KERNELS", ()):
        raise ValueError("Original compiled native HiF4 kernels unavailable")
    libraries = {}
    for name, imported in list(sys.modules.items()):
        filename = getattr(imported, "__file__", None)
        if filename and filename.endswith(".so") and any(t in name for t in ("hifxg_quant", "lowbit_quant")):
            path = Path(filename).resolve()
            matches = [v for v in plan["repo_sources"] if (ROOT / v["path"]).resolve() == path]
            if len(matches) != 1 or sha(path) != matches[0]["sha256"]:
                raise ValueError("Actually imported native library not uniquely bound")
            libraries[str(path.relative_to(ROOT))] = matches[0]["sha256"]
    if len(libraries) != 2:
        raise ValueError("Two exact original native quantizer libraries required")
    return {"API": api_name, "function_module": function.__module__,
            "compiled_CUDA_kernels_available": True, "libraries": libraries}


def _matmul_observer(torch, ledger):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Observer(TorchDispatchMode):
        phase = None

        @contextmanager
        def during(self, name):
            previous, self.phase = self.phase, name
            try:
                yield
            finally:
                self.phase = previous

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
                        torch.ops.aten.bmm.default):
                if func == torch.ops.aten.bmm.default:
                    raise RuntimeError("Unexpected BMM in ordinary narrow FFN")
                matrix = args[1] if func == torch.ops.aten.addmm.default else args[0]
                key = {("merge", torch.float32): "FP32_merge_GEMM",
                       ("forward", torch.bfloat16): "BF16_forward_GEMM",
                       ("backward", torch.bfloat16): "BF16_backward_GEMM",
                       ("backward", torch.float32): "FP32_LoRA_backward_GEMM"}.get((self.phase, matrix.dtype))
                if key is None:
                    raise RuntimeError("Unregistered matmul phase/dtype")
                ledger.observe(key)
            return func(*args, **(kwargs or {}))

    return Observer()


def run(args):
    plan = json.loads(args.protocol.read_text())
    digest = sha(args.protocol)
    active_path = ROOT / "research/ACTIVE_RESEARCH.json"
    authorize(plan, digest, json.loads(active_path.read_text()))
    hf, weight_root = args.hf_root.resolve(), args.weight_hf_root.resolve()
    sys.path.insert(0, str(ROOT / "tools"))
    from worker_identity import require_worker
    require_worker()
    out = (hf / plan["output_HF_relative"]).resolve()
    if out == hf or not out.is_relative_to(hf):
        raise ValueError("Output escaped HF")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ATTEMPT.json").open("x") as handle:
        json.dump({"id": ID, "protocol_sha256": digest, "automatic_retry": False}, handle)
    report = {"id": ID, "status": "STARTED", "protocol_sha256": digest,
              "cells": [], "quality_or_speed_acceptance": False,
              "source_split": plan["source_split"],
              "scope": "four local teacher-state sites; two ordinary 6016 initializations; no rollout or motion claim"}
    monitor, reservation, torch = None, None, None
    started = time.monotonic()
    ledger = BudgetLedger(report, persist=lambda: write(out / "RUN.json", report),
                          guard=lambda run_id: authorize(plan, digest, json.loads(active_path.read_text()), run_id))

    def check():
        if monitor is not None:
            monitor.check()
            if any(s.get("physical_GPU_used_memory_MiB", 0) > 32768 for s in monitor.samples):
                raise RuntimeError("Physical GPU memory budget")
        if torch is not None and torch.cuda.is_initialized() and torch.cuda.max_memory_reserved() > 32 << 30:
            raise RuntimeError("Torch reserved GPU memory budget")
        if (shutil.disk_usage(out).free < RESOURCE_LIMITS["free_disk_bytes"]
                or sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink()) > RESOURCE_LIMITS["new_HF_bytes"]):
            raise RuntimeError("HF disk/output budget")

    def timeout(*_):
        raise TimeoutError("1200 second compute wall")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(RESOURCE_LIMITS["compute_wall_seconds"])
    try:
        versions = {k: importlib.metadata.version(k) for k in ("torch", "numpy", "safetensors")}
        if versions != plan["runtime_versions"]:
            raise ValueError("Actual runtime versions differ")
        required = {"tools/e0_compact6016_recovery_pilot.py", "tools/compact6016_adapter.py",
                    "tests/test_compact6016_adapter.py", "tests/test_e0_compact6016_recovery_pilot.py",
                    "tools/e0_ub1_matched_recovery.py", "tools/worker_identity.py",
                    "tools/startup_gpu_reservation.py", "tools/assigned_gpu_resources.py",
                    ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py",
                    ".runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py"}
        if not required <= {r["path"] for r in plan["repo_sources"]}:
            raise ValueError("Required source bindings missing")
        for item in plan["repo_sources"]:
            bound(ROOT, item)
        for item in plan["hf_assets"]:
            bound(hf, item)
        for item in plan["weight_assets"]:
            bound(weight_root, item)
        paths = {v["path"]: v["sha256"] for v in plan["hf_assets"]}
        wpaths = {v["path"]: v["sha256"] for v in plan["weight_assets"]}
        for entry in (plan["profile"], plan["profile_RUN"], plan["profile_exit"], plan["bias_view"], plan["bias_file"]):
            if paths.get(entry["path"]) != entry["sha256"]:
                raise ValueError("Profile/bias binding missing")
        producer = json.loads((hf / plan["profile_RUN"]["path"]).read_text())
        producer_exit = json.loads((hf / plan["profile_exit"]["path"]).read_text())
        if (producer.get("id") != "E0_UB1_SOURCE_ENERGY_PROFILE_20261027"
                or producer_exit.get("id") != producer.get("id")
                or producer.get("status") != "COMPLETE_LEGACY_CAL24_SOURCE_PROFILE"
                or producer_exit.get("launcher_returncode") != 0 or producer_exit.get("process_exited") is not True
                or producer_exit.get("protocol_sha256") != producer.get("protocol_sha256")):
            raise ValueError("Source profile producer did not complete")
        profile = json.loads((hf / plan["profile"]["path"]).read_text())
        validate_plan(plan, profile)
        from e0_ub1_matched_recovery import validate_input_scope
        sources = {s["index"]: s for s in plan["sources"]}
        for case in plan["cases"]:
            if paths.get(case["path"]) != case["sha256"] or paths.get(case["input_report"]) != case["input_report_sha256"]:
                raise ValueError("Actual operand/input report not bound")
            validate_input_scope(case, json.loads((hf / case["input_report"]).read_text()), sources[case["index"]])
        for weight in plan["weights"]:
            if wpaths.get(weight["path"]) != weight["sha256"]:
                raise ValueError("Original native shard not bound")
        snap = out / "EXECUTED_SOURCES"
        snap.mkdir()
        shutil.copyfile(args.protocol, snap / "protocol.json")
        for item in plan["repo_sources"]:
            target = snap / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / item["path"], target)
        check()
        import torch as torch_api
        torch = torch_api
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        from startup_gpu_reservation import reserve_before_assets
        from compact6016_adapter import CompactAdapters, compact_forward, error_metrics, gated_loss, same_bits
        reservation, monitor, receipt = reserve_before_assets(torch)
        report["reservation"] = receipt
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("One exclusively assigned A100 required")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
            raise ValueError("TF32 override conflicts with FP32 merge")
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
        for name in ("Wan2.2-I2V-A14B-W4A4", "HiF4_Sp"):
            sys.path.insert(0, str(ROOT / ".runtime/code" / name))
        for name in ("HiFloat4", "HiFloat4/hif4_gpu"):
            sys.path.insert(0, str(ROOT / ".runtime/code/HiF4_Sp" / name))
        from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq, get_wan_hif4_runtime_counters
        import quant.hif4_backend as native_backend
        setup_before = {"adapter": get_wan_hif4_runtime_counters(),
                        "backend": native_backend.get_hif4_stats()}
        observer = _matmul_observer(torch, ledger)
        # quant/__init__ optionally imports Wan T5, whose class default calls
        # torch.cuda.current_device(). That CUDA setup is deliberately after
        # reservation. API import is required to execute no unpaid GEMM/QDQ.
        with observer:
            report["actual_native_quantizer"] = verify_native_quantizer(native_backend, plan)
        setup_after = {"adapter": get_wan_hif4_runtime_counters(),
                       "backend": native_backend.get_hif4_stats()}
        if setup_before != setup_after or any(v for counts in setup_after.values() for v in counts.values()):
            raise RuntimeError("Native API setup executed unexpected QDQ/linear")
        report["native_API_setup"] = {"after_registered_GPU_reservation": True,
            "before_counters": setup_before, "after_counters": setup_after,
            "unpaid_GEMM_rejected_by_dispatch_guard": True,
            "CUDA_context_import_dependency": "quant/__init__.py -> wan_smooth -> wan/modules/t5.py T5EncoderModel class default torch.cuda.current_device(); setup only"}
        report["runtime"] = {"versions": versions, "CUDA": torch.version.cuda,
                             "GPU": torch.cuda.get_device_name(0), "TF32": False,
                             "float32_matmul_precision": torch.get_float32_matmul_precision(),
                             "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        counters_before = get_wan_hif4_runtime_counters()
        backend_before = native_backend.get_hif4_stats()

        def tensor_sha(tensor):
            return hashlib.sha256(memoryview(tensor.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()

        def quantizer(value, *, force_fp32):
            ledger.observe("native_Q2_calls")
            return native_hif4_activation_qdq(value, force_fp32=force_fp32)

        def merged(model):
            ledger.observe("merged_weight_pairs")
            with observer.during("merge"):
                return model.merged_weights()

        def forward(case, weights, b1, b2, smooth, *, training=False):
            ledger.observe("student_forwards")
            with observer.during("forward"):
                return compact_forward(case["V1"].to("cuda"), weights[0], b1, weights[1], b2,
                                       smooth, quantizer, training=training, return_trace=True)

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
                    raise RuntimeError("Dead W1/W2 LoRA gradient at GPU gate")
            return norms

        def reserve_forward(run_id, **extra):
            ledger.reserve({"student_forwards": 1, "native_Q2_calls": 1,
                            "BF16_forward_GEMM": 2, **extra}, run_id)

        bias_values = load_file(str(hf / plan["bias_file"]["path"]), device="cpu")
        if set(bias_values) != {b["tensor_key"] for b in plan["bias_tensors"]}:
            raise ValueError("Exact eight extracted native bias tensors required")
        for spec in plan["bias_tensors"]:
            value = bias_values[spec["tensor_key"]]
            if value.dtype != torch.bfloat16 or list(value.shape) != spec["shape"] or tensor_sha(value) != spec["tensor_sha256"]:
                raise ValueError("Original raw checkpoint bias differs")
        with observer:
            for step, block in POSITIONS:
                check()
                wref = next(w for w in plan["weights"] if (w["step"], w["block"]) == (step, block))
                raw = torch.load(weight_root / wref["path"], map_location="cpu", weights_only=True, mmap=True)[wref["expert"]]
                up, down = raw[f"blocks.{block}.ffn.0"], raw[f"blocks.{block}.ffn.2"]
                for slot, module, shape in (("W1", up, (13824, 5120)), ("W2", down, (5120, 13824))):
                    value = module["weight_qdq"]
                    if (value.dtype != torch.bfloat16 or tuple(value.shape) != shape
                            or tensor_sha(value) != wref[slot + "_tensor_sha256"] or not bool(torch.isfinite(value).all())
                            or module.get("activation_qdq") is not True or module.get("force_fp32") is not True):
                        raise ValueError("Original native W1/W2 identity differs")
                smooth_full = down["smooth_scale"]
                if (smooth_full.dtype != torch.float32 or tuple(smooth_full.shape) != (13824,)
                        or tensor_sha(smooth_full) != wref["smooth2_tensor_sha256"]
                        or not bool(torch.isfinite(smooth_full).all()) or not bool((smooth_full > 0).all())):
                    raise ValueError("Original fixed smooth2 differs")
                smooth1 = up["smooth_scale"]
                if (smooth1.dtype != torch.float32 or tuple(smooth1.shape) != (5120,)
                        or tensor_sha(smooth1) != wref["smooth1_tensor_sha256"]
                        or not bool(torch.isfinite(smooth1).all()) or not bool((smooth1 > 0).all())):
                    raise ValueError("Original upstream smooth1 differs")
                b1_full = bias_values[f"step{step}_block{block:02d}_b1"]
                b2 = bias_values[f"step{step}_block{block:02d}_b2"].to("cuda")
                cases = {}
                for index in range(24):
                    case = next(c for c in plan["cases"] if (c["index"], c["step"], c["block"]) == (index, step, block))
                    with safe_open(str(hf / case["path"]), framework="pt", device="cpu") as handle:
                        if set(handle.keys()) != {"V1", "V2", "Y", "gate", "b2"}:
                            raise ValueError("Native operand payload schema differs")
                        # Teacher V2 is not decoded or passed into the student.
                        data = {name: handle.get_tensor(name) for name in ("V1", "Y", "gate", "b2")}
                    for name, shape in (("V1", (256, 5120)), ("Y", (256, 5120)), ("gate", (1, 5120)), ("b2", (5120,))):
                        if data[name].dtype != torch.bfloat16 or tuple(data[name].shape) != shape or not bool(torch.isfinite(data[name]).all()):
                            raise ValueError("Actual native operand header/dtype/finiteness differs")
                    if tensor_sha(data["gate"]) != wref["gate_tensor_sha256"] or not same_bits(data["b2"], b2.cpu()):
                        raise ValueError("Native gate/b2 differs")
                    cases[index] = data
                for init in INITIALIZATIONS:
                    cell = next(c for c in plan["cells"] if (c["step"], c["block"], c["initialization"]) == (step, block, init))
                    run_id = cell["run_id"]
                    row = {"run_id": run_id, "step": step, "block": block,
                           "initialization": init, "groups": cell["groups"], "status": "STARTED",
                           "diagnostics": {}, "epoch_source_loss_means": []}
                    report["cells"].append(row)
                    ids = torch.tensor([64 * g + j for g in cell["groups"] for j in range(64)], dtype=torch.int64)
                    w1 = up["weight_qdq"].index_select(0, ids).contiguous().to("cuda")
                    w2 = down["weight_qdq"].index_select(1, ids).contiguous().to("cuda")
                    b1 = b1_full.index_select(0, ids).contiguous().to("cuda")
                    smooth = smooth_full.index_select(0, ids).contiguous().to("cuda")
                    ledger.reserve({"LoRA_random_factor_draws": 2}, run_id)
                    model = CompactAdapters(w1, w2, rank=32, seed=42,
                                            on_random=lambda: ledger.observe("LoRA_random_factor_draws"))
                    # A real CUDA control tests zero LoRA, exact Q2 forward and
                    # finite gradients before the first Adam update.
                    ledger.reserve({"merged_weight_pairs": 1, "FP32_merge_GEMM": 2}, run_id)
                    zero_weights = merged(model)
                    row["zero_merge_base_weight_bitwise"] = [same_bits(z, w) for z, w in zip(zero_weights, (w1, w2))]
                    reserve_forward(run_id, GPU_gate_reference_forwards=1)
                    ledger.observe("GPU_gate_reference_forwards")
                    with torch.no_grad():
                        reference, reference_trace = forward(cases[0], (w1, w2), b1, b2, smooth)
                    reserve_forward(run_id, GPU_gate_STE_forwards=1, loss_evaluations=1,
                                    backward_calls=1, BF16_backward_GEMM=3,
                                    FP32_LoRA_backward_GEMM=4, finite_gradient_tensor_checks=4)
                    ledger.observe("GPU_gate_STE_forwards")
                    initial_output, initial_trace = forward(cases[0], zero_weights, b1, b2, smooth, training=True)
                    byte_checks = {name: same_bits(initial_trace[name], reference_trace[name]) for name in initial_trace}
                    byte_checks["pre_gate_output"] = same_bits(initial_output, reference)
                    row["GPU_zero_LoRA_and_native_STE_byte_checks"] = byte_checks
                    if not all(byte_checks.values()):
                        raise RuntimeError("Zero-LoRA original function/native STE GPU byte gate failed")
                    gate_loss = loss(initial_output, cases[0])
                    if not bool(torch.isfinite(gate_loss)):
                        raise RuntimeError("Nonfinite GPU gate loss")
                    row["GPU_gate_gradient_norms"] = backward(gate_loss, model, nonzero_B=True)
                    model.zero_grad(set_to_none=True)
                    initial_weights = tuple(w.detach() for w in zero_weights)
                    del zero_weights, initial_output, initial_trace, reference, reference_trace, gate_loss

                    def diagnose(label, weights):
                        rows = []
                        for index in range(24):
                            check()
                            reserve_forward(run_id, diagnostic_source_batches=1, loss_evaluations=1)
                            ledger.observe("diagnostic_source_batches")
                            with torch.no_grad():
                                output, trace = forward(cases[index], weights, b1, b2, smooth)
                                value = loss(output, cases[index])
                                if not bool(torch.isfinite(value)):
                                    raise RuntimeError("Nonfinite diagnostic loss")
                                metrics = error_metrics(output, cases[index]["Y"].to("cuda"), cases[index]["gate"].to("cuda"))
                            rows.append({"index": index, "source_id": sources[index]["semantic_source_id"],
                                         "role": "TRAIN" if index in TRAIN else "HELDOUT_EXPOSED_DIAGNOSTIC",
                                         "optimization_loss": float(value), **metrics,
                                         "output_tensor_sha256": tensor_sha(output),
                                         "student_pre_Q2_sha256": tensor_sha(trace["pre_Q2"]),
                                         "student_post_Q2_sha256": tensor_sha(trace["post_Q2"]),
                                         "student_Q2_shape": [256, 6016]})
                        row["diagnostics"][label] = rows
                        write(out / "RUN.json", report)

                    diagnose("initial", initial_weights)
                    del initial_weights
                    optimizer = torch.optim.Adam(model.parameters(), lr=.0005, betas=(.9, .999),
                                                 eps=1e-8, weight_decay=0, foreach=False, fused=False)
                    for epoch in range(8):
                        epoch_losses = []
                        for index in TRAIN:
                            check()
                            case = training_case(cases, index)
                            ledger.reserve({"training_source_batches": 1, "merged_weight_pairs": 1,
                                            "FP32_merge_GEMM": 2, "student_forwards": 1,
                                            "native_Q2_calls": 1, "BF16_forward_GEMM": 2,
                                            "loss_evaluations": 1, "backward_calls": 1,
                                            "BF16_backward_GEMM": 3, "FP32_LoRA_backward_GEMM": 4,
                                            "finite_gradient_tensor_checks": 4,
                                            "finite_updated_parameter_tensor_checks": 4,
                                            "Adam_steps": 1, "Adam_parameter_tensor_updates": 4}, run_id)
                            ledger.observe("training_source_batches")
                            optimizer.zero_grad(set_to_none=True)
                            current_weights = merged(model)
                            output, trace = forward(case, current_weights, b1, b2, smooth, training=True)
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
                            epoch_losses.append(float(value.detach()))
                            del current_weights, output, trace, value
                        row["epoch_source_loss_means"].append(sum(epoch_losses) / 16)
                        write(out / "RUN.json", report)
                        print(run_id, "epoch", epoch + 1, "loss", row["epoch_source_loss_means"][-1], flush=True)
                    ledger.reserve({"merged_weight_pairs": 1, "FP32_merge_GEMM": 2}, run_id)
                    with torch.no_grad():
                        final_weights = tuple(w.detach().contiguous() for w in merged(model))
                    ledger.reserve({"exported_students": 1, "exported_final_BF16_matrices": 2}, run_id)
                    export = {"W1": final_weights[0].cpu(), "W2": final_weights[1].cpu(),
                              "b1": b1.cpu(), "b2": b2.cpu(), "smooth2": smooth.cpu(),
                              "smooth1": smooth1.contiguous(),
                              "groups": torch.tensor(cell["groups"], dtype=torch.int64), "channels": ids}
                    export.update({"training_" + k: v.detach().cpu().contiguous() for k, v in model.named_parameters()})
                    export_path = out / (run_id + ".safetensors")
                    save_file(export, str(export_path), metadata={"id": ID, "run_id": run_id,
                              "deployment": "W1+b1 GELU nativeSmooth2/HiF4Q2 W2+b2; no online LoRA",
                              "pilot_input_boundary": "saved native post-Q1 BF16; smooth1 is retained for future native Q1 integration, not repeated here",
                              "teacher_V2_used": "false", "weight_requantization": "false"})
                    reloaded = load_file(str(export_path), device="cpu")
                    if set(reloaded) != set(export) or any(not same_bits(reloaded[k], export[k]) for k in export):
                        raise RuntimeError("Folded pack serialization byte gate failed")
                    ledger.observe("exported_students")
                    ledger.observe("exported_final_BF16_matrices", 2)
                    row["export"] = {"path": str(export_path.relative_to(hf)), "sha256": sha(export_path),
                                     "bytes": export_path.stat().st_size,
                                     "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": tensor_sha(v)} for k, v in export.items()},
                                     "reloaded_all_tensor_bytes_equal": True,
                                     "online_LoRA_GEMM": 0, "online_weight_QDQ": 0}
                    del final_weights
                    # Final diagnostics execute the serialized, reloaded BF16
                    # pack. No adapter or teacher is used by this path.
                    final_weights = (reloaded["W1"].to("cuda"), reloaded["W2"].to("cuda"))
                    diagnose("final_folded_BF16", final_weights)
                    row["status"] = "COMPLETE_128_UPDATES_AND_INITIAL_FINAL_24_SOURCE_DIAGNOSTICS"
                    write(out / "RUN.json", report)
                    del model, optimizer, final_weights, export, reloaded, w1, w2, b1, smooth
                    check()
                del raw, up, down, smooth_full, smooth1, cases, b2
        if ledger.predebited != LIMITS or ledger.observed != LIMITS:
            raise RuntimeError("Actual operation ledger incomplete/differs")
        counters_after = get_wan_hif4_runtime_counters()
        delta = {k: counters_after[k] - counters_before[k] for k in counters_before}
        if delta["weight_qdq_calls"] != 0 or delta["activation_qdq_calls"] != LIMITS["native_Q2_calls"]:
            raise RuntimeError("Native quantizer counter disagrees")
        report["native_quantizer_counter_delta"] = delta
        backend_after = native_backend.get_hif4_stats()
        backend_delta = {k: backend_after[k] - backend_before[k] for k in backend_before}
        if backend_delta["weight_qdq_calls"] != 0 or backend_delta["activation_qdq_calls"] != LIMITS["native_Q2_calls"]:
            raise RuntimeError("Native backend counter disagrees")
        report["native_backend_counter_delta"] = backend_delta
        report["student_quantization"] = {"every_forward_recomputed_from_student_GELU": True,
            "force_fp32": True, "group_hierarchy": [64, 8, 4],
            "pre_Q2_and_QDQ_tensor_SHA_saved_per_diagnostic": True,
            "internal_codes_scales": "NOT_EXPOSED_BY_ORIGINAL_NATIVE_QDQ_API"}
        report.update(status="COMPLETE_FOUR_SITE_TWO_INITIALIZATION_COMPACT6016_PILOT",
                      elapsed_seconds=time.monotonic() - started,
                      peak_torch_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
                      final_diagnostic_online_LoRA_GEMM=0)
        check()
    except BaseException as error:
        report.update(status="FAILED_NO_RETRY", error_type=type(error).__name__, error=str(error))
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
        report["outer_process_completion"] = "REQUIRES_LAUNCHER_PROCESS_EXIT_RECEIPT"
        write(out / "RUN.json", report)
        if monitor_error is not None and not original_error:
            raise monitor_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--hf-root", required=True, type=Path)
    parser.add_argument("--weight-hf-root", required=True, type=Path)
    run(parser.parse_args())
