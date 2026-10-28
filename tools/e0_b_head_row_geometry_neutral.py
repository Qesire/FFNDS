"""Four-case B0 full-M geometry neutral using saved rows, never saved full-u.

Two deterministic fills and two repeats give exactly sixteen original-shape
LN/head calls. Only the 256 genuine saved rows claim native byte equality.
Unselected rows are artificial; their output is never called a recovered state.
Import and byte-only preflight perform no Torch import or device discovery.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_B_HEAD_ROW_GEOMETRY_NEUTRAL_20261028"
B0_ID = "E0_B_FINAL_BLOCK_NATIVE_CAPTURE_HEAD_ASSET_20261028"
B0_STATUS = "COMPLETE_B_FINAL_BLOCK_CAPTURE_HISTORICAL_POSTSTATE_BYTES_MATCH"
HISTORICAL_PROBE_SHA = "1abe493a430fe592ebb2b450c0ee40a9557949df65f8c548c5cd470ebe14bb8d"
GEOMETRY = {"M": 31668, "D": 5120, "head_outputs": 64, "selected_rows": 256}
CASES = ((1, 0), (1, 3), (4, 0), (4, 3))
FILLS = ("zero", "fixed_channel_pattern")
FILL_DESCRIPTION = "BF16(((channel_index % 17) - 8) / 8); broadcast across all rows; no RNG"
RUNTIME = {"torch": "2.10.0+cu128", "numpy": "2.4.6", "safetensors": "0.8.0"}
RESOURCES = {"GPU": 1, "global_max_GPU": 2, "CPU_threads": 2,
             "compute_seconds": 600, "outer_seconds": 900,
             "output_bytes": 32 << 20, "GPU_reserved_bytes": 8 << 30,
             "free_disk_bytes": 1 << 30, "automatic_retry": False}
LIMITS = {"head_pack_loads": 2, "capture_pack_loads": 4, "tensor_identity_checks": 4,
          "case_layout_uploads": 4, "channel_pattern_builds": 4,
          "padded_states": 8, "selected_row_writes": 8,
          "padded_state_hashes": 24, "layer_norm_calls": 16, "head_calls": 16,
          "selected_input_row_checks": 16, "selected_head_input_checks": 16,
          "selected_head_row_checks": 16, "same_fill_full_head_repeat_checks": 8,
          "full_head_hashes": 16,
          "full_u_reads": 0, "FFN_calls": 0, "QDQ_calls": 0, "QR_calls": 0,
          "fit_updates": 0, "model_calls": 0, "video_generations": 0}
REQUIRED_SOURCES = {"tools/e0_b_head_row_geometry_neutral.py",
                    "tests/test_e0_b_head_row_geometry_neutral.py",
                    "tools/e0_b_terminal_head_probe.py", "tools/worker_identity.py",
                    "tools/startup_gpu_reservation.py", "tools/assigned_gpu_resources.py"}
CAPTURE_KEYS = ("rows", "u_rows", "head_output", "head_input_rows", "head_scale", "head_shift")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def bound(root, binding):
    require(isinstance(binding, dict) and {"path", "bytes", "sha256"} <= set(binding), "Asset NOT_BOUND")
    root = Path(root).resolve()
    relative = Path(binding["path"])
    path = (root / relative).resolve()
    require(not relative.is_absolute() and path != root and path.is_relative_to(root), "Asset escaped declared root")
    require(path.is_file() and path.stat().st_size == binding["bytes"] and sha(path) == binding["sha256"],
            "Bound asset bytes differ: " + binding["path"])
    return path


def read_bound(root, binding):
    return json.loads(bound(root, binding).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def validate_plan(plan, *, frozen=False):
    require(plan.get("id") == ID and plan.get("geometry") == GEOMETRY
            and plan.get("fills") == list(FILLS) and plan.get("fill_description") == FILL_DESCRIPTION
            and plan.get("repeats_per_fill") == 2 and plan.get("epsilon") == 1e-6
            and plan.get("limits") == LIMITS and plan.get("resources") == RESOURCES
            and plan.get("runtime_versions") == RUNTIME and plan.get("automatic_retry") is False,
            "Changed neutral geometry, numerical scope or resource budget")
    cases = plan.get("cases", [])
    require(len(cases) == 4 and [(c.get("source_index"), c.get("step")) for c in cases] == list(CASES),
            "Exactly four ordered B0 source01/04 step0/3 cases required")
    require(all(set(c) == {"source_index", "step", "capture", "case", "head"} for c in cases),
            "Only saved rows/capture, CASE and head bindings allowed; never bind full-u")
    require(plan.get("historical_probe_protocol", {}).get("sha256") == HISTORICAL_PROBE_SHA,
            "Exact historical B1 B0 receipt anchor required")
    if frozen:
        require(plan.get("assets_finalized") is True, "Draft is not executable")
        require(isinstance(plan.get("repo_sources"), list)
                and REQUIRED_SOURCES <= {b["path"] for b in plan["repo_sources"]}, "Missing frozen source/test bindings")


def validate_case_metadata(case, item, *, geometry=GEOMETRY):
    M, D, H, rows = (geometry[k] for k in ("M", "D", "head_outputs", "selected_rows"))
    source, step = item["source_index"], item["step"]
    expected_id = f"b_final_capture_source{source:02d}_seed42_step{step}"
    require(case.get("schema") == "ffnds.b_terminal.capture_case.v1" and case.get("run_id") == expected_id
            and case.get("source_index") == source and case.get("step") == step and case.get("seed") == 42,
            "Saved CASE source/seed/step identity differs")
    actual = case["actual_pre_head_u"]
    require(actual.get("shape") == [M, D] and actual.get("dtype") == "torch.bfloat16"
            and actual.get("stride") == [D, 1], "Recorded original full-M u layout differs")
    operator = case["actual_head_operator"]
    require(operator.get("head_class") == "lightx2v.common.ops.mm.mm_weight.MMWeight"
            and operator.get("norm_class") == "lightx2v.common.ops.norm.layer_norm_weight.LNWeight"
            and operator.get("head_has_lora_branch") is False and operator.get("norm_weight") is None
            and operator.get("norm_bias") is None and operator.get("norm_eps") == 1e-6
            and operator.get("infer_dtype") == operator.get("sensitive_dtype") == "torch.bfloat16",
            "Original row-local torch LN/DefaultMM branch differs")
    for name, shape, stride in (("weight", [D, H], [1, D]), ("bias", [H], [1])):
        spec = operator[name]
        require(spec.get("shape") == shape and spec.get("stride") == stride
                and spec.get("dtype") == "torch.bfloat16", "Original head tensor layout differs")
    for name in ("scale", "shift"):
        spec = case["actual_combined_head_modulation"][name]
        require(spec.get("shape") == [1, 1, D] and spec.get("stride") == [2 * D, D, 1]
                and spec.get("dtype") == "torch.bfloat16", "Actual modulation layout differs")
    chosen = case["row_draw"]["rows"]
    require(len(chosen) == rows and len(set(chosen)) == rows
            and all(type(i) is int and 0 <= i < M for i in chosen), "Recorded selected row coordinates differ")
    return operator


def check_capture_header(path, case, *, geometry=GEOMETRY):
    """Header only; no tensor import, allocation, or full-u path read."""
    M, D, H, rows = (geometry[k] for k in ("M", "D", "head_outputs", "selected_rows"))
    expected = {"rows": ("I64", [rows]), "u_rows": ("BF16", [rows, D]),
                "head_output": ("BF16", [M, H]), "head_input_rows": ("BF16", [rows, D]),
                "head_scale": ("BF16", [1, 1, D]), "head_shift": ("BF16", [1, 1, D])}
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        require(len(raw) == 8, "Truncated capture header")
        n = struct.unpack("<Q", raw)[0]
        require(0 < n <= 1 << 20 and n + 8 <= size, "Invalid capture header length")
        header = json.loads(handle.read(n))
    for name, (dtype, shape) in expected.items():
        spec = header.get(name, {})
        require(spec.get("dtype") == dtype and spec.get("shape") == shape, "Capture tensor header differs: " + name)
        lo, hi = spec.get("data_offsets", (-1, -1))
        count = 1
        for dim in shape:
            count *= dim
        require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= size - n - 8
                and hi - lo == count * (8 if dtype == "I64" else 2), "Capture storage extent differs")
        recorded = case["capture_file"]["tensors"][name]
        require(recorded.get("shape") == shape and recorded.get("dtype") == ("torch.int64" if dtype == "I64" else "torch.bfloat16"),
                "CASE capture tensor metadata differs")
    return len(expected)


def verify_assets(plan, hf):
    """Rebind only historical receipts and four small captures, never full-u."""
    historical = read_bound(ROOT, plan["historical_probe_protocol"])
    require(historical.get("id") == "E0_B_TERMINAL_HEAD_PROBE_20261028"
            and historical.get("assets_finalized") is True
            and historical.get("b0_run") == plan["b0_run"] and historical.get("b0_exit") == plan["b0_exit"],
            "Historical B1 does not bind these exact B0 receipts")
    run, receipt = read_bound(hf, plan["b0_run"]), read_bound(hf, plan["b0_exit"])
    require(run.get("id") == receipt.get("id") == B0_ID and run.get("status") == B0_STATUS
            and receipt.get("protocol_sha256") == run.get("protocol_sha256")
            and receipt.get("launcher_returncode") == 0 and receipt.get("process_exited") is True
            and receipt.get("automatic_retry") is False, "Original B0 completion/actual exit differs")
    if "RUN_sha256" in receipt:
        require(receipt["RUN_sha256"] == plan["b0_run"]["sha256"], "B0 exit RUN SHA differs")
    old_cases = {(c["source_index"], c["step"]): c for c in historical["cases"]}
    checked = []
    for item in plan["cases"]:
        key = item["source_index"], item["step"]
        old = old_cases[key]
        require(all(item[k] == old[k] for k in ("capture", "case", "head")), "Changed immutable B0 case/capture/head binding")
        case = read_bound(hf, item["case"])
        validate_case_metadata(case, item)
        owners = [c for c in run["cases"] if c["index"] == key[0]]
        captures = [c for owner in owners for c in owner["step_captures"] if c["step"] == key[1]]
        require(len(owners) == len(captures) == 1 and captures[0]["run_id"] == case["run_id"], "B0 parent case identity differs")
        for saved in (case["capture_file"], captures[0]["capture_file"]):
            require(saved["sha256"] == item["capture"]["sha256"] and saved["bytes"] == item["capture"]["bytes"],
                    "Parent capture bytes differ")
        require(case["original_head_file"]["sha256"] == item["head"]["sha256"], "Original head file differs")
        check_capture_header(bound(hf, item["capture"]), case)
        bound(hf, item["head"])
        checked.append(case)
    return run, checked, {"original_B0_exit_RUN_sha_field": "PRESENT" if "RUN_sha256" in receipt else "ABSENT_IN_ORIGINAL_RECEIPT",
        "exact_RUN_and_exit_bound_by_historical_B1_protocol": HISTORICAL_PROBE_SHA,
        "full_u_read": False, "capture_case_count": 4}


def preflight(protocol, hf):
    require("torch" not in sys.modules and "torch.cuda" not in sys.modules, "CPU preflight requires a fresh process without Torch")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide every CUDA device for independent CPU preflight")
    plan = json.loads(Path(protocol).read_text())
    validate_plan(plan, frozen=True)
    versions = {k: importlib.metadata.version(k) for k in RUNTIME}
    require(versions == RUNTIME, "Package versions differ")
    for item in plan["repo_sources"]:
        bound(ROOT, item)
    _, _, parent = verify_assets(plan, Path(hf).resolve())
    return {"id": ID, "status": "PASS_B_HEAD_ROW_GEOMETRY_CPU_PREFLIGHT_NO_TORCH_NO_GPU",
            "protocol_sha256": sha(protocol), "repo_sources_sha256": semantic_sha(plan["repo_sources"]),
            "cases_sha256": semantic_sha(plan["cases"]), "cases": 4,
            "Torch_or_CUDA_imported": "torch" in sys.modules or "torch.cuda" in sys.modules,
            "all_CUDA_devices_hidden": True, "native_calls": 0, "parent": parent}


def authorize(plan, digest, active):
    validate_plan(plan, frozen=True)
    user = active.get("user_continuation_authorization_20261027", {})
    require(user.get("id") == "USER_RESUME_OFFLINE_IFC_20261027"
            and user.get("status") == "ACTIVE_CURRENT_USER_DIRECTED_SESSION"
            and user.get("new_scope_registration_allowed") is True
            and user.get("max_simultaneously_assigned_idle_GPUs") == 2, "Current two-GPU session authority differs")
    scopes = [x for x in active.get("experiment_authority", {}).get("current_user_authorized_scopes", []) if x.get("id") == ID]
    require(len(scopes) == 1, "Unique new neutral scope registration required")
    scope = scopes[0]
    require(scope.get("protocol_sha256") == digest and scope.get("execution_status") in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
            and scope.get("limits") == LIMITS and scope.get("resources") == RESOURCES
            and scope.get("max_gpu_cards") == 1 and scope.get("max_simultaneously_assigned_idle_GPUs") == 2,
            "Exact neutral protocol/limits/resources not registered")
    return scope


def prior_preflight(plan, digest, scope, hf):
    proof = scope.get("cpu_preflight")
    require(isinstance(proof, dict) and {"RUN", "launch_exit"} <= set(proof), "Exited independent CPU preflight NOT_BOUND")
    report, receipt = read_bound(hf, proof["RUN"]), read_bound(hf, proof["launch_exit"])
    require(report.get("id") == ID and report.get("protocol_sha256") == digest
            and report.get("status") == "PASS_B_HEAD_ROW_GEOMETRY_CPU_PREFLIGHT_NO_TORCH_NO_GPU"
            and report.get("repo_sources_sha256") == semantic_sha(plan["repo_sources"])
            and report.get("cases_sha256") == semantic_sha(plan["cases"]) and report.get("cases") == 4
            and report.get("Torch_or_CUDA_imported") is False and report.get("all_CUDA_devices_hidden") is True
            and report.get("native_calls") == 0, "Prior CPU preflight version or result differs")
    require(receipt.get("id") == ID and receipt.get("protocol_sha256") == digest
            and receipt.get("launcher_returncode") == 0 and receipt.get("process_exited") is True
            and receipt.get("automatic_retry") is False and receipt.get("RUN_sha256") == proof["RUN"]["sha256"],
            "Prior CPU preflight actual exit/RUN SHA differs")
    return proof


class Ledger:
    def __init__(self, report, persist=lambda: None, guard=lambda: None, limits=LIMITS):
        self.report, self.persist, self.guard, self.limits = report, persist, guard, limits
        for phase in ("predebited", "invoked", "observed"):
            report[phase] = dict.fromkeys(limits, 0)

    def reserve(self, cost):
        self.guard()
        require(all(k in self.limits and type(v) is int and v >= 0
                    and self.report["predebited"][k] + v <= self.limits[k] for k, v in cost.items()),
                "Operation exceeds fixed budget before dispatch")
        for key, value in cost.items():
            self.report["predebited"][key] += value
        self.persist()

    def observe(self, key, phase):
        target = "invoked" if phase == "invoked" else "observed"
        require(key in self.limits and phase in ("invoked", "completed"), "Unknown operation observation")
        ceiling = self.report["predebited"][key] if target == "invoked" else self.report["invoked"][key]
        require(self.report[target][key] + 1 <= ceiling, "Unpaid operation observation")
        self.report[target][key] += 1
        self.report["last_operation"] = {"key": key, "phase": phase}
        self.persist()

    @contextmanager
    def operation(self, key):
        self.reserve({key: 1})
        self.observe(key, "invoked")
        yield
        self.observe(key, "completed")

    def finish(self):
        require(all(self.report[k] == self.limits for k in ("predebited", "invoked", "observed")), "Incomplete exact operation ledger")


def tensor_sha(torch, value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def channel_pattern(torch, D):
    """A fixed, exactly representable nonconstant pattern; no RNG or fitting."""
    return ((torch.arange(D, dtype=torch.int64) % 17 - 8).float() / 8).to(torch.bfloat16)


def padded_state(torch, u_rows, rows, fill, shape, stride, *, device, ledger):
    from e0_b_terminal_head_probe import strict_rows
    M, D = shape
    strict_rows(torch, rows, M, expected=u_rows.shape[0])
    require(u_rows.dtype == torch.bfloat16 and tuple(u_rows.shape)[1:] == (D,)
            and tuple(stride) == (D, 1) and fill in FILLS, "Padded state geometry differs")
    pattern = None
    if fill == "fixed_channel_pattern":
        with ledger.operation("channel_pattern_builds"):
            pattern = channel_pattern(torch, D).to(device)
    with ledger.operation("padded_states"):
        state = torch.empty_strided((M, D), tuple(stride), dtype=torch.bfloat16, device=device)
        if pattern is None:
            state.zero_()
        else:
            state.copy_(pattern.expand(M, D))
    with ledger.operation("selected_row_writes"):
        state.index_copy_(0, rows.to(device), u_rows.to(device))
    return state


def validate_loaded(torch, tensors, head, case, *, geometry=GEOMETRY):
    from e0_b_terminal_head_probe import strict_rows
    M, D, H, rows = (geometry[k] for k in ("M", "D", "head_outputs", "selected_rows"))
    require(set(tensors) == set(CAPTURE_KEYS), "Unexpected saved capture tensors")
    strict_rows(torch, tensors["rows"], M, expected=rows)
    require(tensors["rows"].tolist() == case["row_draw"]["rows"], "Saved rows differ from CASE coordinates")
    for name, value in tensors.items():
        spec = case["capture_file"]["tensors"][name]
        require(value.device.type == "cpu" and list(value.shape) == spec["shape"] and str(value.dtype) == spec["dtype"]
                and tensor_sha(torch, value) == spec["raw_tensor_sha256"] and bool(torch.isfinite(value).all()),
                "Saved tensor bytes differ: " + name)
    for name, key in (("head_weight_runtime_KN", "weight"), ("head_bias", "bias")):
        value, spec = head[name], case["actual_head_operator"][key]
        require(value.dtype == torch.bfloat16 and value.device.type == "cpu" and list(value.shape) == spec["shape"]
                and tensor_sha(torch, value) == spec["raw_tensor_sha256"] and bool(torch.isfinite(value).all()),
                "Original head bytes differ: " + name)
    for name in ("scale", "shift"):
        require(tensor_sha(torch, tensors["head_" + name]) == case["actual_combined_head_modulation"][name]["raw_tensor_sha256"],
                "Actual combined modulation bytes differ")
    require(tuple(tensors["u_rows"].shape) == (rows, D) and tuple(tensors["head_output"].shape) == (M, H), "Selected/full head geometry differs")


def neutral_case(torch, tensors, weight, bias, scale, shift, metadata, ledger, *, device,
                 geometry=GEOMETRY, synchronize=lambda: None, check=lambda: None, records=None):
    """Tiny-testable core: same original shape and stride, selected-row claims."""
    from e0_b_terminal_head_probe import native_head, same_bits
    M, D = geometry["M"], geometry["D"]
    rows = tensors["rows"].to(device)
    genuine = tensors["u_rows"].to(device)
    expected_head = tensors["head_output"].index_select(0, tensors["rows"]).to(device)
    expected_hidden = tensors["head_input_rows"].to(device)
    results = [] if records is None else records
    for fill in FILLS:
        check()
        u = padded_state(torch, genuine, rows, fill, (M, D), metadata["actual_pre_head_u"]["stride"], device=device, ledger=ledger)
        with ledger.operation("padded_state_hashes"):
            original_sha = tensor_sha(torch, u)
        previous = None
        for repeat in range(2):
            check()
            call_record = {"fill": fill, "repeat": repeat, "status": "STARTED"}
            results.append(call_record)
            ledger.reserve({"layer_norm_calls": 1, "head_calls": 1})
            def observe(key, phase):
                if phase == "completed":
                    synchronize()
                ledger.observe(key, phase)
            output, hidden = native_head(torch, u, weight, bias, scale, shift, observe=observe)
            synchronize()
            with ledger.operation("padded_state_hashes"):
                require(tensor_sha(torch, u) == original_sha, "Native LN/head mutated padded input")
            with ledger.operation("selected_input_row_checks"):
                require(same_bits(torch, u.index_select(0, rows), genuine), "Original selected u rows changed")
            with ledger.operation("selected_head_input_checks"):
                require(same_bits(torch, hidden.index_select(0, rows), expected_hidden), "Selected LN/modulation rows differ from original capture")
            with ledger.operation("selected_head_row_checks"):
                require(same_bits(torch, output.index_select(0, rows), expected_head), "Selected head rows differ from original full-M capture")
            if previous is not None:
                with ledger.operation("same_fill_full_head_repeat_checks"):
                    require(same_bits(torch, output, previous), "Same-fill full head repeat differs")
            with ledger.operation("full_head_hashes"):
                output_sha = tensor_sha(torch, output)
            call_record.update(status="COMPLETE", padded_input_sha256=original_sha,
                               full_head_sha256=output_sha, selected_u_unchanged=True,
                               selected_head_input_bytes_match=True, selected_head_output_bytes_match=True,
                               same_fill_full_head_repeat_bytes_match=True if previous is not None else "FIRST_REPEAT")
            previous = output
            del hidden
        del previous, output, u
    return results


def run(args):
    plan, digest = json.loads(args.protocol.read_text()), sha(args.protocol)
    active_path = ROOT / "research/ACTIVE_RESEARCH.json"
    scope = authorize(plan, digest, json.loads(active_path.read_text()))
    hf = args.hf_root.resolve()
    out = (hf / plan["output_HF_relative"]).resolve()
    require(out != hf and out.is_relative_to(hf), "Output escaped shared HF root")
    from worker_identity import require_worker
    require_worker()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ATTEMPT.json").open("x") as handle:
        json.dump({"id": ID, "protocol_sha256": digest, "automatic_retry": False}, handle)
    report = {"id": ID, "protocol_sha256": digest, "status": "STARTED", "cases": [],
              "evidence_scope": "FULL_M_GEOMETRY_SELECTED_ROW_REPLAY; no full-state/full-head historical reproduction"}
    persist = lambda: write(out / "RUN.json", report)
    ledger = Ledger(report, persist, lambda: authorize(plan, digest, json.loads(active_path.read_text())))
    torch, monitor, start = None, None, time.monotonic()
    def check():
        if monitor is not None:
            monitor.check()
            require(all(s.get("physical_GPU_used_memory_MiB", 0) <= 8192 for s in monitor.samples), "Physical GPU memory budget exceeded")
        if torch is not None and torch.cuda.is_initialized():
            require(torch.cuda.max_memory_reserved() <= RESOURCES["GPU_reserved_bytes"], "GPU reserved memory budget exceeded")
        require(shutil.disk_usage(out).free >= RESOURCES["free_disk_bytes"], "HF free disk reserve exhausted")
        require(sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.is_symlink()) <= RESOURCES["output_bytes"], "Output byte budget exceeded")
    def timeout(*_):
        raise TimeoutError("Neutral 600-second compute wall or termination signal")
    handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    for sig in handlers:
        signal.signal(sig, timeout)
    signal.alarm(RESOURCES["compute_seconds"])
    try:
        require("torch" not in sys.modules, "Native entry must bind authorization before Torch import")
        for item in plan["repo_sources"]:
            bound(ROOT, item)
        require({k: importlib.metadata.version(k) for k in RUNTIME} == RUNTIME, "Runtime versions differ")
        report["CPU_preflight_binding"] = prior_preflight(plan, digest, scope, hf)
        parent, metadata, report["parent_evidence"] = verify_assets(plan, hf)
        snapshot = out / "EXECUTED_SOURCES"
        snapshot.mkdir()
        shutil.copyfile(args.protocol, snapshot / "protocol.json")
        for item in plan["repo_sources"]:
            destination = snapshot / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / item["path"], destination)
        check()
        import torch as torch_api
        torch = torch_api
        from safetensors import safe_open
        from startup_gpu_reservation import reserve_before_assets
        from e0_b_terminal_head_probe import restore_layout
        reservation, monitor, report["reservation"] = reserve_before_assets(torch)
        require(torch.cuda.device_count() == 1 and "A100" in torch.cuda.get_device_name(0), "One exclusively assigned A100 required")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        require(os.environ.get("NVIDIA_TF32_OVERRIDE") != "1", "TF32 override conflicts with original head")
        require(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == parent["BF16_reduced_precision_reduction"], "Original BF16 reduction setting differs")
        report["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "GPU": torch.cuda.get_device_name(0),
            "BF16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "TF32": torch.backends.cuda.matmul.allow_tf32}
        heads = {}
        for item, case in zip(plan["cases"], metadata):
            check()
            expert = 0 if item["step"] == 0 else 1
            if expert not in heads:
                with ledger.operation("head_pack_loads"):
                    with safe_open(str(bound(hf, item["head"])), framework="pt", device="cpu") as handle:
                        require(set(handle.keys()) == {"head_weight_runtime_KN", "head_bias", "head_modulation"}, "Original head pack schema differs")
                        heads[expert] = {k: handle.get_tensor(k).contiguous() for k in ("head_weight_runtime_KN", "head_bias")}
            with ledger.operation("capture_pack_loads"):
                with safe_open(str(bound(hf, item["capture"])), framework="pt", device="cpu") as handle:
                    tensors = {k: handle.get_tensor(k).contiguous() for k in CAPTURE_KEYS}
            with ledger.operation("tensor_identity_checks"):
                validate_loaded(torch, tensors, heads[expert], case)
            op, mod = case["actual_head_operator"], case["actual_combined_head_modulation"]
            with ledger.operation("case_layout_uploads"):
                weight = restore_layout(torch, heads[expert]["head_weight_runtime_KN"], op["weight"]["stride"], "cuda")
                bias = restore_layout(torch, heads[expert]["head_bias"], op["bias"]["stride"], "cuda")
                scale = restore_layout(torch, tensors["head_scale"], mod["scale"]["stride"], "cuda")
                shift = restore_layout(torch, tensors["head_shift"], mod["shift"]["stride"], "cuda")
            row = {"source_index": item["source_index"], "step": item["step"], "run_id": case["run_id"],
                   "capture_binding": item["capture"], "case_binding": item["case"], "head_binding": item["head"],
                   "actual_original_u_stride": case["actual_pre_head_u"]["stride"], "calls": []}
            report["cases"].append(row)
            row["calls"] = neutral_case(torch, tensors, weight, bias, scale, shift, case, ledger,
                                        device="cuda", synchronize=torch.cuda.synchronize, check=check, records=row["calls"])
            persist()
            del tensors, weight, bias, scale, shift
        ledger.finish()
        report.update(status="COMPLETE_B_HEAD_FULL_M_GEOMETRY_SELECTED_ROW_NEUTRAL", elapsed_seconds=time.monotonic() - start,
            peak_torch_GPU_reserved_bytes=torch.cuda.max_memory_reserved(), full_state_reproduction=False,
            original_full_head_reproduction=False, selected_row_native_byte_reproduction=True,
            outer_process_completion="REQUIRES_DURABLE_WORKER_ACTUAL_EXIT_AND_EXACT_RUN_SHA")
        check()
    except BaseException as error:
        report.update(status="FAILED_NO_RETRY", error_type=type(error).__name__, error=str(error), exception_traceback=traceback.format_exc())
        raise
    finally:
        original = sys.exc_info()[0] is not None
        signal.alarm(0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        failure = None
        if monitor is not None:
            try:
                monitor.close()
                report["assigned_gpu_resources"] = monitor.report()
                require(not report["assigned_gpu_resources"].get("violations"), "Assigned GPU monitor violation")
            except BaseException as error:
                report["status"] = "FAILED_GPU_MONITOR"
                failure = error
        persist()
        if failure is not None and not original:
            raise failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--inspect", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.execute:
        run(args)
    elif args.preflight:
        require(args.report is not None, "CPU preflight requires an exclusive HF report path")
        root, destination = args.hf_root.resolve(), args.report.resolve()
        require(destination.is_relative_to(root) and destination != root, "Preflight output escaped HF root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as handle:
            try:
                result = preflight(args.protocol, root)
            except BaseException as error:
                json.dump({"id": ID, "protocol_sha256": sha(args.protocol), "status": "FAILED_NO_RETRY",
                           "error_type": type(error).__name__, "error": str(error)}, handle, indent=2)
                handle.write("\n")
                raise
            json.dump(result, handle, indent=2)
            handle.write("\n")
    else:
        plan = json.loads(args.protocol.read_text())
        validate_plan(plan)
        print(json.dumps({"id": ID, "assets_finalized": plan.get("assets_finalized"), "limits": LIMITS,
                          "resources": RESOURCES, "numeric_calls": 0, "full_u_reads": 0}, indent=2))


if __name__ == "__main__":
    main()
