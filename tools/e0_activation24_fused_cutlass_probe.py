"""Fixed synthetic C5: one-pass select/pack versus Torch CUTLASS conversion.

GPU byte conformance to the original converter is mandatory in this very run,
including fixed small zero/tie cases and each full shape. No backend fallback,
autotuning, model, fitting, or video calls are permitted.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from activation24_fused_cutlass_pack import (BLOCK_GROUPS, CONVERTER_SHA256, NUM_WARPS,
    edge_input, fused_pack, materialize_masked, padded_rows, wrap_cutlass)
from e0_activation24_cutlass_backend_probe import finish_monitor, raise_monitor_if_unmasked
from e0_activation24_spmm_probe import select_top2

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_ACTIVATION24_FUSED_CUTLASS_PROBE_20261027"
SHAPES = ((31668, 5120, 13824), (31668, 13824, 5120))
EDGE_SHAPES = ((35, 64), (33, 128))
LIMITS = {"random_tensor_init": 4, "edge_tensor_upload": 2,
    "finite_input_check": 6, "stable_reference_select": 4, "materialize_selector": 30,
    "fused_select_pack": 32, "torch_converter": 32, "cutlass_wrapper": 48,
    "packed_meta_byte_check": 4, "masked_byte_check": 2, "dense_gemm": 16,
    "cutlass_sparse_gemm": 44, "search_api": 0, "model_calls": 0,
    "fits": 0, "video_generations": 0, "official_scores": 0}
TIMING_COSTS = {
    "dense": {"dense_gemm": 1},
    "materialize_selector": {"materialize_selector": 1},
    "torch_converter": {"torch_converter": 1, "cutlass_wrapper": 1},
    "fused_select_pack": {"fused_select_pack": 1},
    "cutlass_prepacked_contiguous": {"cutlass_sparse_gemm": 1},
    "reference_online_contiguous": {"materialize_selector": 1, "torch_converter": 1,
        "cutlass_wrapper": 1, "cutlass_sparse_gemm": 1},
    "fused_online_contiguous": {"fused_select_pack": 1, "cutlass_wrapper": 1, "cutlass_sparse_gemm": 1}}
RESOURCES = {"simultaneously_assigned_idle_GPUs": 1, "global_user_GPU_max": 4,
    "cpu_threads": 2, "HF_free_min_bytes": 10 << 30, "HF_output_max_bytes": 100 << 20,
    "new_HF_bytes": 100 << 20, "compute_wall_seconds": 480, "external_total_wall_seconds": 600,
    "torch_peak_allocated_max_bytes": 64 << 30}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    os.replace(temporary, path)


def charge(counts, costs):
    if any(k not in LIMITS or type(v) is not int or v < 0 or counts[k] + v > LIMITS[k]
           for k, v in costs.items()):
        raise RuntimeError("Fixed C5 budget exceeded before dispatch")
    for key, amount in costs.items():
        counts[key] += amount


def validate_plan(plan):
    if (plan.get("id") != ID or plan.get("assets_finalized") is not True or plan.get("limits") != LIMITS or
            plan.get("shapes") != [list(x) for x in SHAPES] or plan.get("edge_shapes") != [list(x) for x in EDGE_SHAPES] or
            plan.get("warmups") != 2 or plan.get("repetitions") != 5 or plan.get("resource_limits") != RESOURCES or
            plan.get("timing_costs") != TIMING_COSTS or plan.get("converter_sha256") != CONVERTER_SHA256 or
            plan.get("block_groups") != BLOCK_GROUPS or plan.get("num_warps") != NUM_WARPS or
            plan.get("relative_l2_tolerance") != .02 or plan.get("automatic_retry") is not False or
            plan.get("byte_conformance_required_in_this_GPU_run") is not True or
            plan.get("selector") != "same stable descending absolute BF16 top2, lower-index ties" or
            plan.get("backend") != "CUTLASS_ONLY_NO_FALLBACK" or plan.get("autotuning") != "NOT_CALLED" or
            plan.get("output_HF_relative") != "ffnds-derived/e0_activation24_fused_cutlass_probe_20261027/attempt1"):
        raise ValueError("Unfrozen C5 mathematical/byte/shape/resource scope")


def authorize(plan, digest, active):
    validate_plan(plan)
    user = active.get("user_continuation_authorization_20261027", {})
    if (user.get("id") != "USER_RESUME_OFFLINE_IFC_20261027" or
            user.get("status") != "ACTIVE_CURRENT_USER_DIRECTED_SESSION" or
            user.get("new_scope_registration_allowed") is not True or
            user.get("max_simultaneously_assigned_idle_GPUs") != 4):
        raise ValueError("Actual current user continuation authority required")
    if not any(row.get("id") == ID and row.get("protocol_sha256") == digest and
            row.get("execution_status") in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
            for row in active.get("experiment_authority", {}).get("current_user_authorized_scopes", [])):
        raise ValueError("Exact new C5 scope is not registered")


def verify_bindings(plan):
    rows = plan["bindings"]
    if len({x["path"] for x in rows}) != len(rows):
        raise ValueError("Duplicate C5 source binding")
    required = {"tools/activation24_fused_cutlass_pack.py", "tools/e0_activation24_fused_cutlass_probe.py",
        "tests/test_e0_activation24_fused_cutlass_probe.py", "tools/e0_activation24_spmm_probe.py",
        "tools/e0_activation24_cutlass_backend_probe.py", "tools/worker_identity.py",
        "tools/startup_gpu_reservation.py", "tools/assigned_gpu_resources.py", "config/worker.env",
        "scripts/worker_wait_uuid_idle_exec.sh",
        ".venv/lib/python3.11/site-packages/torch/sparse/semi_structured.py",
        ".venv/lib/python3.11/site-packages/torch/sparse/_semi_structured_ops.py",
        ".venv/lib/python3.11/site-packages/torch/sparse/_semi_structured_conversions.py",
        ".venv/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so",
        ".venv/lib/python3.11/site-packages/triton/__init__.py",
        ".venv/lib/python3.11/site-packages/triton/runtime/jit.py",
        ".venv/lib/python3.11/site-packages/triton/language/core.py",
        ".venv/lib/python3.11/site-packages/triton/compiler/compiler.py",
        ".venv/lib/python3.11/site-packages/triton/_C/libtriton.so"}
    if not required <= {row["path"] for row in rows}:
        raise ValueError("Required C5 source/backend/launcher binding missing")
    for row in rows:
        path = (ROOT / row["path"]).resolve()
        if (Path(row["path"]).is_absolute() or not path.is_relative_to(ROOT.resolve()) or
                not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]):
            raise ValueError("Frozen C5 source/backend binding differs: " + row["path"])
    converter = [x for x in rows if x["path"].endswith("torch/sparse/_semi_structured_conversions.py")]
    if len(converter) != 1 or converter[0]["sha256"] != CONVERTER_SHA256:
        raise ValueError("Exact original converter binding required")


def byte_conformance(packed, meta, reference):
    import torch
    for actual, expected in ((packed, reference.packed), (meta, reference.meta)):
        if (actual.shape != expected.shape or actual.dtype != expected.dtype or
                actual.device != expected.device or not actual.is_contiguous() or
                not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))):
            raise RuntimeError("GPU packed/meta byte conformance failed; stop without fallback")
    return {"packed_bytes_equal": True, "metadata_bytes_equal": True, "device": str(packed.device)}


def require_gpu_gate(case):
    gate = case.get("gpu_converter_byte_gate", {})
    if (gate.get("packed_bytes_equal") is not True or gate.get("metadata_bytes_equal") is not True or
            not str(gate.get("device", "")).startswith("cuda")):
        raise RuntimeError("Actual GPU converter byte gate required before SpMM/timing")


def verify_loaded_native(plan):
    expected = {(ROOT / x["path"]).resolve().name: (ROOT / x["path"]).resolve()
                for x in plan["bindings"] if Path(x["path"]).name in ("libtorch_cuda.so", "libtriton.so")}
    if set(expected) != {"libtorch_cuda.so", "libtriton.so"}:
        raise ValueError("Torch CUTLASS dispatch and Triton library identities required")
    observed = {name: set() for name in expected}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        raw = fields[5]
        path = Path(raw.removesuffix(" (deleted)")).resolve()
        if path.name in observed:
            if raw.endswith(" (deleted)"):
                raise ValueError("Deleted loaded native library")
            observed[path.name].add(path)
    if any(observed[name] != {path} for name, path in expected.items()):
        raise ValueError("Loaded CUTLASS/Triton library differs from exact bindings")
    return {name: str(path.relative_to(ROOT.resolve())) for name, path in expected.items()}


def run(args):
    if not args.execute_gpu:
        raise RuntimeError("Explicit --execute-gpu and real registration required")
    if os.environ.get("TRITON_INTERPRET", "0") != "0":
        raise RuntimeError("Real GPU code generation required; CPU interpreter cannot benchmark")
    plan = json.loads(args.protocol.read_text())
    digest = sha(args.protocol)
    authorize(plan, digest, json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text()))
    verify_bindings(plan)
    from worker_identity import require_worker
    require_worker()
    hf = args.hf_root.resolve()
    prior = (hf / plan["prior_C4_RUN"]["path"]).resolve()
    if not prior.is_relative_to(hf) or sha(prior) != plan["prior_C4_RUN"]["sha256"]:
        raise ValueError("Actual successful C4 RUN binding differs")
    if json.loads(prior.read_text())["status"] != "COMPLETE_SYNTHETIC_DUAL_BACKEND_ONLY":
        raise ValueError("Completed C4 input required")
    out = (hf / plan["output_HF_relative"]).resolve()
    if out == hf or not out.is_relative_to(hf):
        raise ValueError("C5 output escaped HF")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ATTEMPT.json").open("x") as stream:
        json.dump({"id": ID, "protocol_sha256": digest, "automatic_retry": False}, stream)
    report = {"id": ID, "protocol_sha256": digest, "status": "STARTED",
        "predebited_work": dict.fromkeys(LIMITS, 0), "completed_operation_counts": dict.fromkeys(LIMITS, 0),
        "edge_cases": [], "cases": [], "new_model_fit_video_calls": 0,
        "counter_semantics": "public operation upper bounds prepaid before call, no refund; completed counts only after return; internal Torch converter kernels not individually counted",
        "search_or_autotuning": "NOT_CALLED", "scope": "synthetic BF16 encoding/execution only, no Wan numeric or video evidence"}
    monitor = None
    started = time.monotonic()
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError("Fixed C5 480-second compute limit")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(RESOURCES["compute_wall_seconds"])
    def check():
        if monitor:
            monitor.check()
        if (shutil.disk_usage(out).free < RESOURCES["HF_free_min_bytes"] or
                sum(x.stat().st_size for x in out.rglob("*") if x.is_file() and not x.is_symlink()) > RESOURCES["HF_output_max_bytes"]):
            raise RuntimeError("C5 storage budget reached")
    def dispatch(label, cost, function):
        check()
        charge(report["predebited_work"], cost)
        report["current_operation"] = {"label": label, "cost": cost}
        write(out / "RUN.json", report)
        result = function()
        for key, amount in cost.items():
            report["completed_operation_counts"][key] += amount
        return result
    try:
        check()
        snap = out / "EXECUTED_SOURCES"
        snap.mkdir()
        shutil.copyfile(args.protocol, snap / "protocol.json")
        for index, row in enumerate(plan["bindings"]):
            path = ROOT / row["path"]
            if path.suffix in (".py", ".sh"):
                if path.stat().st_size > 1 << 20:
                    raise ValueError("C5 source snapshot too large")
                shutil.copyfile(path, snap / f"{index:02d}_{path.name}")
        os.environ["TRITON_CACHE_DIR"] = str(out / "triton_cache")
        import torch
        import torch.nn.functional as F
        import torch.sparse.semi_structured as sm
        import triton
        from startup_gpu_reservation import reserve_before_assets
        bound = {(ROOT / x["path"]).resolve() for x in plan["bindings"]}
        for name in ("torch.sparse.semi_structured", "torch.sparse._semi_structured_ops",
                     "torch.sparse._semi_structured_conversions", "triton"):
            if Path(sys.modules[name].__file__).resolve() not in bound:
                raise ValueError("Imported C5 dependency differs: " + name)
        versions = {k: importlib.metadata.version(k) for k in plan["runtime_versions"]}
        if versions != plan["runtime_versions"]:
            raise ValueError("C5 runtime versions differ")
        reservation, monitor, receipt = reserve_before_assets(torch)
        report["reservation"] = receipt
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("One assigned A100 required")
        torch.set_num_threads(RESOURCES["cpu_threads"])
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        report["runtime"] = {"versions": versions, "CUDA": torch.version.cuda, "GPU": torch.cuda.get_device_name(0),
            "TF32": False, "BF16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        def finite(value):
            return dispatch("finite_input_check", {"finite_input_check": 1}, lambda: bool(torch.isfinite(value).all()))
        def reference(value):
            masked = dispatch("stable_reference_select", {"stable_reference_select": 1},
                lambda: F.pad(select_top2(value), (0, 0, 0, padded_rows(value.shape[0]) - value.shape[0])))
            packed = dispatch("original_torch_converter", {"torch_converter": 1, "cutlass_wrapper": 1},
                lambda: sm.SparseSemiStructuredTensorCUTLASS.from_dense(masked))
            if type(packed) is not sm.SparseSemiStructuredTensorCUTLASS:
                raise RuntimeError("Original converter changed backend")
            return masked, packed
        def gate(value, original):
            torch.cuda.synchronize()
            initial = time.perf_counter()
            packed, meta = dispatch("fused_select_pack_gate", {"fused_select_pack": 1}, lambda: fused_pack(value))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - initial
            proof = dispatch("packed_meta_byte_check", {"packed_meta_byte_check": 1},
                lambda: byte_conformance(packed, meta, original))
            return packed, meta, proof, seconds
        for m, k in EDGE_SHAPES:
            value = dispatch("fixed_edge_tensor_upload", {"edge_tensor_upload": 1}, lambda: edge_input(m, k).cuda())
            if not finite(value):
                raise RuntimeError("Nonfinite fixed edge input")
            masked, original = reference(value)
            packed, meta, proof, setup = gate(value, original)
            report["edge_cases"].append({"M": m, "K": k, "M_padded": padded_rows(m),
                "gpu_converter_byte_gate": proof, "initial_dispatch_including_compilation_seconds": setup})
            write(out / "RUN.json", report)
            del value, masked, original, packed, meta
        if len(report["edge_cases"]) != 2:
            raise RuntimeError("Both real GPU small byte gates must pass before full shapes")
        def random_tensor(shape, seed):
            def create():
                generator = torch.Generator(device="cuda")
                generator.manual_seed(seed)
                return torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
            return dispatch("random_tensor_init", {"random_tensor_init": 1}, create)
        def spmm(packed, right, m):
            return torch.mm(packed, right)[:m].contiguous()
        for index, (m, k, n) in enumerate(SHAPES):
            value = random_tensor((m, k), 20261027 + 2 * index)
            weight = random_tensor((n, k), 20261028 + 2 * index)
            if not finite(value) or not finite(weight):
                raise RuntimeError("Nonfinite synthetic input")
            stable, original = reference(value)
            packed, meta, proof, setup = gate(value, original)
            case = {"M": m, "K": k, "N": n, "M_padded": padded_rows(m),
                "input_seed": 20261027 + 2 * index, "weight_seed": 20261028 + 2 * index,
                "gpu_converter_byte_gate": proof, "initial_dispatch_including_compilation_seconds": setup,
                "packed_shape": list(packed.shape), "meta_shape": list(meta.shape), "timings": {}}
            require_gpu_gate(case)
            report["cases"].append(case)
            masked = dispatch("materialized_selector_setup", {"materialize_selector": 1}, lambda: materialize_masked(value))
            equal = dispatch("masked_byte_check", {"masked_byte_check": 1},
                lambda: torch.equal(masked.view(torch.uint8), stable.view(torch.uint8)))
            if not equal:
                raise RuntimeError("Unfused selector bytes differ from frozen stable top2")
            case["materialized_selector_byte_equal"] = True
            del stable, original
            candidate = dispatch("wrap_fused_setup", {"cutlass_wrapper": 1}, lambda: wrap_cutlass(packed, meta))
            right = weight.t()
            dense = dispatch("same_mask_dense_reference", {"dense_gemm": 1}, lambda: F.linear(masked[:m], weight))
            actual = dispatch("fused_packed_spmm_correctness", {"cutlass_sparse_gemm": 1}, lambda: spmm(candidate, right, m))
            torch.cuda.synchronize()
            delta, denominator = actual.float() - dense.float(), dense.float().norm().item()
            relative = delta.norm().item() / denominator if denominator else None
            if (relative is None or relative > plan["relative_l2_tolerance"] or not bool(torch.isfinite(actual).all()) or
                    tuple(actual.shape) != (m, n) or not actual.is_contiguous()):
                raise RuntimeError("CUTLASS versus same-masked dense numerical gate failed")
            case["same_mask_correctness"] = {"relative_l2": relative, "max_abs": delta.abs().max().item(),
                "output_contiguous": True, "output_shape": list(actual.shape)}
            del dense, actual, delta
            if index == 0:
                report["loaded_native_libraries"] = verify_loaded_native(plan)
            def ref_online():
                return spmm(sm.SparseSemiStructuredTensorCUTLASS.from_dense(materialize_masked(value)), right, m)
            def fused_online():
                return spmm(wrap_cutlass(*fused_pack(value)), right, m)
            functions = {"dense": lambda: F.linear(value, weight),
                "materialize_selector": lambda: materialize_masked(value),
                "torch_converter": lambda: sm.SparseSemiStructuredTensorCUTLASS.from_dense(masked),
                "fused_select_pack": lambda: fused_pack(value),
                "cutlass_prepacked_contiguous": lambda: spmm(candidate, right, m),
                "reference_online_contiguous": ref_online, "fused_online_contiguous": fused_online}
            for name, function in functions.items():
                require_gpu_gate(case)
                samples = []
                for repetition in range(plan["warmups"] + plan["repetitions"]):
                    def timed():
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        wall = time.monotonic()
                        begin.record()
                        result = function()
                        end.record()
                        torch.cuda.synchronize()
                        info = {"cuda_ms": begin.elapsed_time(end), "wall_ms": 1000 * (time.monotonic() - wall),
                            "allocated_before_bytes": before, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                        if info["peak_allocated_bytes"] > RESOURCES["torch_peak_allocated_max_bytes"]:
                            raise RuntimeError("C5 tensor allocation budget reached")
                        del result
                        return info
                    measured = dispatch(name, TIMING_COSTS[name], timed)
                    if repetition >= plan["warmups"]:
                        samples.append(measured)
                case["timings"][name] = samples
                write(out / "RUN.json", report)
            del value, weight, right, masked, packed, meta, candidate
        if report["predebited_work"] != LIMITS or report["completed_operation_counts"] != LIMITS:
            raise RuntimeError("C5 exact operation ledger incomplete")
        check()
        report.update(status="COMPLETE_SYNTHETIC_FUSED_CUTLASS_ONLY", elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        report.update(status="FAILED_NO_RETRY_NO_BACKEND_FALLBACK", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        original_error_active = sys.exc_info()[0] is not None
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        monitor_error = finish_monitor(report, monitor) if monitor else None
        report["elapsed_seconds_including_setup_and_compile"] = time.monotonic() - started
        write(out / "RUN.json", report)
        raise_monitor_if_unmasked(monitor_error, original_error_active)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    if args.execute_gpu:
        run(args)
    else:
        plan = json.loads(args.protocol.read_text())
        validate_plan(plan)
        verify_bindings(plan)
        print(json.dumps({"status": "STATIC_C5_BINDINGS_PASS_GPU_BYTE_GATES_NOT_RUN",
            "protocol_sha256": sha(args.protocol), "GPU_initialized": False,
            "model_calls": 0, "GPU_execution_authorized_by_this_preflight": False}, indent=2))


if __name__ == "__main__":
    main()
