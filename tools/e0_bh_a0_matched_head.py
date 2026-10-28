"""Same CAL24 teacher-state/head comparison: saved B-H versus terminal A0.

Default inspection is arithmetic-free. A separately frozen CPU byte preflight
and exact ACTIVE registration precede any Torch import in execution. The 96
full-M selected-row neutrals must all pass before the 192 A0 readouts. A0's FFN
uses M256 and saved post-Q1 V1; the following head uses original full-M geometry
with artificial zero-filled other rows. This is a function-error comparison.
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
import sys
import time
import traceback

import b_terminal_paired_discrimination as labels

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_BH_A0_MATCHED_TERMINAL_HEAD_20261028"
BH_ID = "E0_BH_RESIDUAL_ANCHORED_DIRECT65_H128_20261028"
BH_PROTOCOL_SHA = "2832d5be8c5d248c923c3a3b244c5d3c4f84e957efbf95bf36d9605af2a88ce1"
BANK_SHA = "e3c4f4020951913f9cffb2e2cf244137e94db2031ea64aabea7bc65a7aff6a50"
GEOMETRY = {"M": 31668, "D": 5120, "outputs": 64, "rows": 256, "width": 6016, "group": 64}
MODES = ("BH_saved", "energy94", "random94")
EXPERTS = ("high_noise_model", "low_noise_model")
CASE_IDS = tuple(f"cal24_{i:02d}_step{s}" for i in range(24) for s in range(4))
RESOURCES = {"GPU": 1, "session_max_GPUs": 2, "repository_GPU_policy": "DYNAMIC_MAX_ACTUALLY_CALLABLE",
             "CPU_threads": 2, "compute_seconds": 1200, "outer_seconds": 1800,
             "GPU_reserved_bytes": 8 << 30, "output_bytes": 64 << 20,
             "free_disk_bytes": 1 << 30, "automatic_retry": False}
REQUIRED_SOURCES = {"tools/e0_bh_a0_matched_head.py", "tests/test_e0_bh_a0_matched_head.py",
    "tools/b_terminal_paired_discrimination.py", "tools/b_cal24_label_contract.py",
    "tools/e0_compact6016_remaining66.py", "tools/compact6016_remaining_contract.py",
    "tools/e0_compact6016_full80_completion.py", "tools/audit_compact6016_completion_partial.py",
    "tools/e0_b_terminal_head_probe.py", "tools/compact6016_adapter.py",
    "tools/e0_b_final_block_capture.py", "tools/worker_identity.py",
    "tools/startup_gpu_reservation.py", "tools/assigned_gpu_resources.py",
    ".runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py"}
LABEL_KEYS = ("r_rows", "u_rows", "head_rows", "gate", "head_scale", "head_shift", "rows")
A0_KEYS = ("W1", "W2", "b1", "b2", "smooth1", "smooth2", "groups", "channels")


def require(value, message):
    if not value:
        raise ValueError(message)


def exact_limits():
    n, d, h, w = (GEOMETRY[k] for k in ("rows", "D", "outputs", "width"))
    transfer_bytes = (96 * (6*n*d + 6*n*h + 10*d + 16*n)
                      + 4 * (4*w*d + 6*w + 2*d) + 2 * (2*d*h + 2*h))
    return {"head_pack_loads": 2, "head_tensor_checks": 4, "A0_export_pack_loads": 4,
        "A0_deploy_tensor_checks": 32, "BH_prediction_pack_loads": 2,
        "label_pack_loads": 192, "label_tensor_checks": 1344, "V1_only_pack_loads": 96,
        "V1_tensor_checks": 96, "saved_BH_head_checks": 96, "neutral_cases": 96,
        "A0_W1_BF16_GEMM": 192, "A0_tanh_GELU": 192, "A0_FP32_Smooth2_divide": 192,
        "A0_native_Q2": 192, "A0_W2_BF16_GEMM": 192, "A0_diagnostic_output_SHA_checks": 192,
        "BF16_gate_multiply": 192, "BF16_residual_add": 192,
        "padded_zero_states": 288, "selected_row_writes": 288,
        "fullM_LN": 288, "BF16_head_gain_add": 288, "BF16_head_modulation": 288,
        "fullM_head_GEMM": 288, "selected_u_unchanged_checks": 288,
        "neutral_head_byte_checks": 96, "selected_head_hashes": 288,
        "paired_head_metrics": 288, "FP64_head_metric_reductions": 576,
        "GPU_original_layout_restores": 388,
        "CUDA_H2D_tensor_copies": 1272, "CUDA_H2D_payload_bytes": transfer_bytes,
        "Q1_calls": 0, "BH_student_forwards": 0, "teacher_Y_numeric_reads": 0,
        "full_u_reads": 0, "model_calls": 0, "training_updates": 0,
        "decompositions": 0, "video_generations": 0}


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_bound(root, binding):
    return json.loads(labels.bound(root, binding).read_text())


def precision_gate():
    require(all(os.environ.get(k) != "1" for k in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")),
            "Environment forces TF32")


def validate_plan(plan, frozen=False):
    require(plan.get("id") == ID and plan.get("geometry") == GEOMETRY and plan.get("limits") == exact_limits()
        and plan.get("resources") == RESOURCES and plan.get("modes") == list(MODES)
        and plan.get("allowed_case_ids") == list(CASE_IDS) and plan.get("epsilon") == 1e-6
        and plan.get("source_split") == {"training": list(range(16)), "diagnostic_heldout": list(range(16, 24))},
        "Changed matched-head numerical scope, split or budget")
    require(plan.get("BH_protocol", {}).get("sha256") == BH_PROTOCOL_SHA
        and plan.get("bank_manifest", {}).get("sha256") == BANK_SHA
        and plan.get("parent_protocol", {}).get("sha256") == labels.PARENT_SHA,
        "Only completed exact BH/A0/CAL24 identities accepted")
    if frozen:
        require(plan.get("assets_finalized") is True and plan.get("binding_status") == "BOUND_READY_FOR_REGISTRATION",
                "Draft assets NOT_BOUND")
        paths = [b["path"] for b in plan["repo_sources"]]
        require(len(paths) == len(set(paths)) and REQUIRED_SOURCES <= set(paths)
            and sum(p.endswith(".so") and any(k in p for k in ("hifxg_quant", "lowbit_quant")) for p in paths) == 2,
            "Missing immutable sources or two native Q2 extensions")


def validate_layout(item, geometry=GEOMETRY):
    d, h, m, n = (geometry[k] for k in ("D", "outputs", "M", "rows"))
    case, op, ev = item["case"], item["actual_head_operator"], item["evidence"]
    require(item["actual_u_layout"] == {"shape": [m, d], "dtype": "torch.bfloat16", "stride": [d, 1]},
            "Original fullM u shape/stride differs")
    require(op.get("norm_eps") == 1e-6 and op.get("norm_weight") is None and op.get("norm_bias") is None
        and op.get("head_class") == "lightx2v.common.ops.mm.mm_weight.MMWeight"
        and op.get("norm_class") == "lightx2v.common.ops.norm.layer_norm_weight.LNWeight",
        "Original row-local LN/DefaultMM branch differs")
    for key, shape, stride in (("weight_runtime_KN", [d,h], [1,d]), ("bias", [h], [1])):
        spec = op[key]
        require(spec["shape"] == shape and spec["stride"] == stride and spec["dtype"] == "torch.bfloat16",
                "Original head shape/stride differs")
    require(ev["head_weight_stride"] == [1,d] and ev["head_bias_stride"] == [1]
        and ev["head_scale_stride"] == ev["head_shift_stride"] == [2*d,d,1], "Original modulation strides differ")
    require(case["block"] == 39 and len(case["rows"]) == len(set(case["rows"])) == n
        and all(type(i) is int and 0 <= i < m for i in case["rows"]), "Original selected rows differ")


def verify_assets(plan, hf):
    """Byte-only bindings; only four A0 exports, never the full 20GB bank."""
    manifest = read_bound(hf, plan["label_manifest"])
    labels.verify_manifest(manifest, plan, hf)
    require({x["case"]["case_id"] for x in manifest["cases"]} == set(CASE_IDS), "CAL24 cases differ")
    for item in manifest["cases"]:
        validate_layout(item)
    import e0_compact6016_remaining66 as remaining
    bank = read_bound(hf, plan["bank_manifest"])
    require(bank == remaining.assemble_manifest(bank["training_protocol"], bank["parts"], hf, repo_root=ROOT),
            "A0 mixed-lineage completion receipts differ")
    bh_plan = read_bound(ROOT, plan["BH_protocol"])
    require(bh_plan["label_manifest"] == plan["label_manifest"], "BH used different CAL24 labels")
    run, receipt = read_bound(hf, plan["BH_RUN"]), read_bound(hf, plan["BH_exit"])
    require(run.get("id") == receipt.get("id") == BH_ID and run.get("protocol_sha256") == BH_PROTOCOL_SHA
        and run.get("status") == "COMPLETE_BH_DIRECT65_H128_FIXED256_UPDATES_PER_EXPERT_M256"
        and all(run.get(k) == bh_plan["limits"] for k in ("predebited", "invoked", "completed")),
        "BH fixed-budget completion differs")
    require(receipt.get("child_returncode") == 0 and receipt.get("process_exited") is True
        and receipt.get("timeout") is False and receipt.get("automatic_retry") is False
        and receipt.get("protocol_sha256") == BH_PROTOCOL_SHA and receipt.get("run_sha256") == plan["BH_RUN"]["sha256"],
        "BH actual exited worker/RUN identity differs")
    require(len(plan["BH_predictions"]) == len(plan["BH_exports"]) == 2, "Two BH final prediction/export bindings required")
    by_expert = {e["expert"]: e for e in run["experts"]}
    require(set(by_expert) == set(EXPERTS), "BH expert set differs")
    bh_root = Path(plan["BH_RUN"]["path"]).parent
    for expert, item, exported in zip(EXPERTS, plan["BH_predictions"], plan["BH_exports"]):
        old = by_expert[expert]
        require(item["expert"] == exported["expert"] == expert, "BH expert order differs")
        require(item["binding"] == {**old["final_prediction_pack"], "path": str(bh_root/old["final_prediction_pack"]["path"])},
                "Saved final BH prediction binding differs from actual RUN")
        expected_export = {k: old["export"][k] for k in ("bytes", "sha256")}
        expected_export["path"] = str(bh_root/expert/old["export"]["path"])
        require(exported["binding"] == expected_export, "BH export binding differs")
        labels.bound(hf, item["binding"])
        labels.bound(hf, exported["binding"])
        provenance = {x["case"]["case_id"]: x for x in read_bound(hf, {**old["provenance"], "path": str(bh_root/old["provenance"]["path"])})}
        require(all(provenance[x["case"]["case_id"]]["case"] == x["case"] for x in manifest["cases"] if x["case"]["expert"] == expert),
                "BH prediction source/state/row provenance differs")
    selected = [c for c in bank["cells"] if c["block"] == 39]
    require(len(selected) == 4 and {(c["expert"], c["initialization"]) for c in selected} == {(e,i) for e in EXPERTS for i in MODES[1:]},
            "Four completed terminal A0 cells required")
    for cell in selected:
        labels.bound(hf, cell["export"])
    return manifest, selected, run


def preflight(protocol, hf):
    require("torch" not in sys.modules and "numpy" not in sys.modules and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Fresh byte-only CPU preflight requires hidden CUDA and no numerical imports")
    precision_gate()
    plan = json.loads(Path(protocol).read_text()); validate_plan(plan, frozen=True)
    for binding in plan["repo_sources"]:
        labels.bound(ROOT, binding)
    versions = {k: importlib.metadata.version(k) for k in plan["runtime_versions"]}
    require(versions == plan["runtime_versions"], "Runtime versions differ")
    manifest, cells, _ = verify_assets(plan, hf)
    return {"id": ID, "status": "PASS_MATCHED_HEAD_BYTES_NO_NUMPY_NO_TORCH_NO_GPU",
        "protocol_sha256": labels.sha(protocol), "sources_sha256": labels.state_sha(plan["repo_sources"]),
        "label_manifest_sha256": plan["label_manifest"]["sha256"], "bank_manifest_sha256": BANK_SHA,
        "BH_RUN_sha256": plan["BH_RUN"]["sha256"], "cases": len(manifest["cases"]), "A0_exports": len(cells),
        "all_CUDA_devices_hidden": True, "NumPy_or_Torch_imported": "torch" in sys.modules or "numpy" in sys.modules,
        "model_calls": 0, "native_calls": 0, "training_updates": 0}


def authorize(plan, digest, active):
    validate_plan(plan, frozen=True)
    user = active.get("user_continuation_authorization_20261027", {})
    policy = active.get("resource_scheduling_policy_20261028", {})
    require(user.get("status") == "ACTIVE_CURRENT_USER_DIRECTED_SESSION" and user.get("new_scope_registration_allowed") is True
        and policy.get("mode") == "DYNAMIC_MAX_ACTUALLY_CALLABLE" and policy.get("global_gpu_card_cap") is None
        and active.get("current_session_resource_constraint_20261028", {}).get("max_simultaneously_assigned_idle_GPUs") == 2,
        "Current session/repository resource authority differs")
    scopes = [s for s in active["experiment_authority"]["current_user_authorized_scopes"] if s.get("id") == ID]
    require(len(scopes) == 1, "Unique matched-head registration required")
    scope = scopes[0]
    require(scope.get("protocol_sha256") == digest and scope.get("execution_status") in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
        and scope.get("max_gpu_cards") == 1 and scope.get("limits") == exact_limits() and scope.get("resources") == RESOURCES,
        "Exact registered matched-head scope differs")
    return scope


def require_cpu_receipt(scope, plan, digest, hf):
    cpu = scope.get("cpu_preflight")
    require(isinstance(cpu, dict) and {"RUN", "launch_exit"} <= set(cpu), "Exited independent CPU preflight NOT_BOUND")
    report, receipt = read_bound(hf, cpu["RUN"]), read_bound(hf, cpu["launch_exit"])
    require(report.get("id") == ID and report.get("status") == "PASS_MATCHED_HEAD_BYTES_NO_NUMPY_NO_TORCH_NO_GPU"
        and report.get("protocol_sha256") == digest and report.get("sources_sha256") == labels.state_sha(plan["repo_sources"])
        and report.get("label_manifest_sha256") == plan["label_manifest"]["sha256"] and report.get("bank_manifest_sha256") == BANK_SHA
        and report.get("BH_RUN_sha256") == plan["BH_RUN"]["sha256"] and report.get("cases") == 96 and report.get("A0_exports") == 4
        and report.get("all_CUDA_devices_hidden") is True and report.get("NumPy_or_Torch_imported") is False
        and all(report.get(k) == 0 for k in ("model_calls", "native_calls", "training_updates")), "CPU preflight proof differs")
    require(receipt.get("id") == ID and receipt.get("protocol_sha256") == digest and receipt.get("child_returncode") == 0
        and receipt.get("process_exited") is True and receipt.get("timeout") is False
        and receipt.get("automatic_retry") is False and receipt.get("run_sha256") == cpu["RUN"]["sha256"],
        "CPU preflight actual exit/RUN SHA differs")
    return cpu


class Ledger:
    def __init__(self, report, limits=None, persist=lambda: None, journal=lambda _:None):
        self.report, self.limits, self.persist, self.journal = report, exact_limits() if limits is None else limits, persist, journal
        report["operation_serial"]=0
        for key in ("predebited", "invoked", "completed"):
            report[key] = dict.fromkeys(self.limits, 0)

    def save(self,key,phase,count):
        self.report["operation_serial"]+=1
        self.journal({"serial":self.report["operation_serial"],"key":key,"phase":phase,"count":count,
                      "case":self.report.get("active_case")})
        self.persist()

    @contextmanager
    def __call__(self, key, count=1):
        require(key in self.limits and type(count) is int and count > 0 and self.report["predebited"][key]+count <= self.limits[key],
                "Exceeded fixed matched-head operation: " + key)
        self.report["predebited"][key] += count; self.save(key,"predebited",count)
        self.report["invoked"][key] += count; self.report["last_invoked"] = key; self.save(key,"invoked",count)
        yield
        self.report["completed"][key] += count; self.save(key,"completed",count)

    def finish(self):
        require(all(self.report[k] == self.limits for k in ("predebited", "invoked", "completed")), "Incomplete exact matched-head budget")


def tensor_sha(torch, value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def same_bits(torch, left, right):
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8))


def upload(torch, value, device, work):
    if value.device.type == "cpu" and str(device).startswith("cuda"):
        with work("CUDA_H2D_tensor_copies"), work("CUDA_H2D_payload_bytes", value.numel()*value.element_size()):
            return value.to(device)
    return value.to(device)


def load_tensors(path, names):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {k: handle.get_tensor(k).contiguous() for k in names}


def a0_postwrite(torch, v1, residual, gate, pack, quantizer, work, geometry=GEOMETRY):
    """Only legal post-Q1 V1 and residual; no teacher Y or repeated Q1."""
    n,d,w,g = (geometry[k] for k in ("rows", "D", "width", "group"))
    require(v1.dtype == residual.dtype == gate.dtype == torch.bfloat16 and v1.shape == residual.shape == (n,d)
        and gate.shape == (1,d) and w % g == 0 and g == geometry["group"], "A0 input/whole-group geometry differs")
    with work("A0_W1_BF16_GEMM"):
        hidden = torch.nn.functional.linear(v1, pack["W1"], pack["b1"])
    with work("A0_tanh_GELU"):
        hidden = torch.nn.functional.gelu(hidden, approximate="tanh")
    with work("A0_FP32_Smooth2_divide"):
        q2_input = (hidden.float()/pack["smooth2"]).to(torch.bfloat16)
    with work("A0_native_Q2"):
        q2 = quantizer(q2_input, force_fp32=True).to(torch.bfloat16)
        require(q2.shape == (n,w), "Student-own Q2 shape changed")
    with work("A0_W2_BF16_GEMM"):
        y = torch.nn.functional.linear(q2, pack["W2"], pack["b2"])
    with work("BF16_gate_multiply"):
        delta = y*gate
    with work("BF16_residual_add"):
        u = residual.clone(); u.add_(delta)
    require(y.dtype == u.dtype == torch.bfloat16 and y.shape == u.shape == (n,d), "A0 native BF16 write differs")
    return y,u


def fullM_readout(torch, u_rows, rows, head, scale, shift, work, geometry=GEOMETRY):
    m,d,n = (geometry[k] for k in ("M", "D", "rows"))
    require(u_rows.dtype == torch.bfloat16 and u_rows.shape == (n,d) and rows.dtype == torch.int64
        and rows.shape == (n,) and len(torch.unique(rows)) == n and bool(((rows >= 0) & (rows < m)).all()),
        "FullM selected rows invalid")
    with work("padded_zero_states"):
        state = torch.empty_strided((m,d), (d,1), dtype=u_rows.dtype, device=u_rows.device); state.zero_()
    with work("selected_row_writes"):
        state.index_copy_(0, rows, u_rows)
    with work("fullM_LN"):
        normalized = torch.nn.functional.layer_norm(state,(d,),None,None,1e-6)
    with work("BF16_head_gain_add"):
        gain = 1+scale.reshape(-1)
    with work("BF16_head_modulation"):
        normalized.mul_(gain).add_(shift.reshape(-1))
    with work("fullM_head_GEMM"):
        output = torch.empty((m,geometry["outputs"]),dtype=state.dtype,device=state.device)
        torch.addmm(head["bias"],normalized,head["weight"],out=output)
    with work("selected_u_unchanged_checks"):
        require(same_bits(torch,state.index_select(0,rows),u_rows), "LN/head changed selected u")
    chosen = output.index_select(0,rows)
    with work("selected_head_hashes"):
        digest = tensor_sha(torch,chosen)
    return chosen,digest


def head_metrics(torch, prediction, target, work):
    require(prediction.shape == target.shape and prediction.dtype == target.dtype == torch.bfloat16
        and bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(target).all()), "Head metric operands differ")
    with work("paired_head_metrics"), work("FP64_head_metric_reductions",2):
        sse = float((prediction.double()-target.double()).square().sum())
        energy = float(target.double().square().sum())
    require(energy > 0, "Zero captured head norm")
    return {"head_relative_SSE": sse/energy, "head_mean_row_SSE": sse/len(target), "head_SSE": sse, "head_target_energy": energy}


def check_a0_output(torch, value, expected, record, work):
    with work("A0_diagnostic_output_SHA_checks"):
        actual=tensor_sha(torch,value)
        record["original_final_diagnostic_output_sha256"]=expected
        record["actual_M256_output_sha256"]=actual
        record["original_A0_BF16_reduction_flag"]="NOT_RECORDED_IN_ORIGINAL_PRODUCER"
        record["A0_output_bytes_match_original_diagnostic"]=actual==expected
        require(actual==expected,"A0 M256 output differs from original final diagnostic bytes; exact hashes/case retained, no flag search or retry")


def summarize(records):
    result = {}
    for expert in EXPERTS:
        result[expert] = {}
        steps = {0,1} if expert == EXPERTS[0] else {2,3}
        for split,indices in (("train16",range(16)),("heldout8_exposed",range(16,24))):
            source = {}
            for index in indices:
                rows = [x for x in records if x["source_index"] == index and x["expert"] == expert]
                require(len(rows) == 2 and {x["step"] for x in rows} == steps, "Missing nested expert steps")
                source[str(index)] = {arm:sum(x["arms"][arm]["head_relative_SSE"] for x in rows)/2 for arm in MODES}
            result[expert][split] = {"independent_sources":len(source),"per_source":source,
                "equal_source_mean":{arm:sum(x[arm] for x in source.values())/len(source) for arm in MODES},
                "BH_lower_source_count":{arm:sum(x["BH_saved"] < x[arm] for x in source.values()) for arm in MODES[1:]}}
    return result


def neutral_gate(records, expected=CASE_IDS):
    require(len(records) == len(expected) and {x["case_id"] for x in records} == set(expected)
        and all(x.get("bitwise_equal") is True for x in records), "All96 neutrals must pass first")


def validate_label_tensors(torch, pack, item, work, geometry=GEOMETRY):
    from b_cal24_label_contract import PACK_SCHEMA
    for name in LABEL_KEYS:
        with work("label_tensor_checks"):
            spec = PACK_SCHEMA[name]
            dtype = torch.int64 if name == "rows" else torch.bfloat16
            require(pack[name].dtype == dtype and list(pack[name].shape) == spec[1] and tensor_sha(torch,pack[name]) == item["tensor_sha256"][name]
                and bool(torch.isfinite(pack[name]).all()), "Actual label tensor bytes differ: " + name)
    require(pack["rows"].tolist() == item["case"]["rows"], "Actual ordered row payload differs")


def run(args):
    protocol,hf = args.protocol.resolve(),args.hf_root.resolve()
    require("torch" not in sys.modules and "numpy" not in sys.modules, "Bind the exact execution before numerical imports")
    precision_gate(); plan = json.loads(protocol.read_text()); digest = labels.sha(protocol)
    active_path = ROOT/"research/ACTIVE_RESEARCH.json"
    guard = lambda: authorize(plan,digest,json.loads(active_path.read_text()))
    scope = guard(); cpu = require_cpu_receipt(scope,plan,digest,hf)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(visible is not None and visible.strip() and len(visible.split(",")) == 1, "One explicitly assigned GPU required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        preflight(protocol,hf)
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    manifest,cells,bh_run = verify_assets(plan,hf)
    from worker_identity import require_worker
    require_worker()
    out = (hf/plan["output_HF_relative"]).resolve()
    require(out != hf and out.is_relative_to(hf) and not out.exists(), "Fresh result path required; no retry")
    out.mkdir(parents=True,exist_ok=False)
    report = {"id":ID,"protocol_sha256":digest,"status":"RUNNING_MATCHED_HEAD","CPU_preflight":cpu,
        "neutral":[],"cases":[],"evidence":"M256_A0_FFN_THEN_FULLM_GEOMETRY_SELECTED_ROW_HEAD_VS_SAVED_BH",
        "speed_claim":"NOT_MEASURED","video_claim":"NOT_MEASURED","full_state_reconstruction":False}
    persist = lambda:write(out/"RUN.json",report)
    journal=(out/"OPERATIONS.jsonl").open("x",buffering=1)
    def event(value):
        journal.write(json.dumps(value,separators=(",",":"))+"\n");journal.flush()
    ledger = Ledger(report,journal=event); persist()
    started,monitor,torch = time.monotonic(),None,None
    def check():
        guard()
        if monitor is not None: monitor.check()
        if torch is not None and torch.cuda.is_initialized():
            require(torch.cuda.max_memory_reserved() <= RESOURCES["GPU_reserved_bytes"], "GPU memory budget exceeded")
        require(shutil.disk_usage(out).free >= RESOURCES["free_disk_bytes"] and
            sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= RESOURCES["output_bytes"], "Output/disk budget exceeded")
    def timeout(*_): raise TimeoutError("Matched-head compute wall/signal")
    handlers = {s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    for sig in handlers: signal.signal(sig,timeout)
    signal.alarm(RESOURCES["compute_seconds"])
    try:
        import torch as torch_api
        torch = torch_api
        from startup_gpu_reservation import reserve_before_assets
        from e0_b_terminal_head_probe import restore_layout
        reservation,monitor,report["reservation"] = reserve_before_assets(torch)
        require(torch.cuda.device_count() == 1 and "A100" in torch.cuda.get_device_name(0), "One assigned A100 required")
        torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("highest"); torch.backends.cuda.matmul.allow_tf32 = False
        parent_runs = [read_bound(hf,p["RUN"]) for p in manifest["parts"]]
        reduction_flags = {p["BF16_reduced_precision_reduction"] for p in parent_runs}
        require(len(reduction_flags) == 1, "Original head producer reduction flags differ")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduction_flags.pop()
        report["BF16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        for name in ("HiF4_Sp","Wan2.2-I2V-A14B-W4A4"):
            sys.path.insert(0,str(ROOT/".runtime/code"/name))
        for name in ("HiFloat4","HiFloat4/hif4_gpu"):
            sys.path.insert(0,str(ROOT/".runtime/code/HiF4_Sp"/name))
        import quant.hif4_backend as backend
        from e0_b_final_block_capture import verify_native_quantizer
        report["quantizer"] = verify_native_quantizer(backend,plan)
        _,qtype,_ = backend.get_hif4_api()
        require(qtype("hifx4").blk_size == 64, "Native Q2 group must be64")
        quantizer = lambda value,force_fp32=True:backend.hif4_qdq_tensor(value,dim=-1,force_fp32=force_fp32)
        heads = {}
        for item in manifest["head_assets"]:
            expert = item["expert"]
            with ledger("head_pack_loads"):
                pack = load_tensors(labels.bound(hf,item["binding"]),("head_weight_runtime_KN","head_bias"))
            for key,value in pack.items():
                with ledger("head_tensor_checks"):
                    require(value.dtype == torch.bfloat16 and list(value.shape) == item["tensors"][key]["shape"]
                        and tensor_sha(torch,value) == item["tensor_sha256"][key], "Actual original head bytes differ")
            heads[expert] = {}
            for name,key,stride in (("weight","head_weight_runtime_KN",(1,5120)),("bias","head_bias",(1,))):
                value=upload(torch,pack[key],"cuda",ledger)
                with ledger("GPU_original_layout_restores"):
                    heads[expert][name]=restore_layout(torch,value,stride,"cuda")
        def load_case(item):
            with ledger("label_pack_loads"):
                pack = load_tensors(labels.bound(hf,item["label"]),LABEL_KEYS)
            validate_label_tensors(torch,pack,item,ledger)
            return pack
        def conditioning(pack,item):
            values=[]
            for key in ("head_scale","head_shift"):
                value=upload(torch,pack[key],"cuda",ledger)
                with ledger("GPU_original_layout_restores"):
                    values.append(restore_layout(torch,value,item["evidence"][key+"_stride"],"cuda"))
            return values
        # Complete all96 independently captured native neutrals first.
        for item in manifest["cases"]:
            check(); pack=load_case(item); case=item["case"]
            report["active_case"]={"case_id":case["case_id"],"expert":case["expert"],"step":case["step"],"phase":"ORIGINAL_U_NEUTRAL"}
            neutral={"case_id":case["case_id"],"expected_head_sha256":item["tensor_sha256"]["head_rows"],"status":"STARTED"}
            report["neutral"].append(neutral);persist()
            u,target,rows = [upload(torch,pack[k],"cuda",ledger) for k in ("u_rows","head_rows","rows")]
            scale,shift=conditioning(pack,item)
            actual,head_sha=fullM_readout(torch,u,rows,heads[case["expert"]],scale,shift,ledger)
            neutral["head_sha256"]=head_sha
            with ledger("neutral_head_byte_checks"):
                require(same_bits(torch,actual,target), "CAL24 fullM selected-row neutral failed; stop before all A0 calls")
            with ledger("neutral_cases"):
                neutral.update(status="COMPLETE",bitwise_equal=True)
            persist()
        neutral_gate(report["neutral"])
        students,diagnostics={},{}
        for cell in cells:
            key=cell["expert"],cell["initialization"]
            with ledger("A0_export_pack_loads"):
                pack=load_tensors(labels.bound(hf,cell["export"]),A0_KEYS)
            for name,value in pack.items():
                with ledger("A0_deploy_tensor_checks"):
                    recorded=cell["export"]["tensors"][name]
                    require(str(value.dtype)==recorded["dtype"] and list(value.shape)==recorded["shape"]
                        and tensor_sha(torch,value)==recorded["sha256"], "Actual terminal A0 export differs")
            require(pack["groups"].tolist()==cell["groups"] and pack["channels"].tolist()==[g*64+j for g in cell["groups"] for j in range(64)], "A0 group/channel64 differs")
            students[key]={k:upload(torch,pack[k],"cuda",ledger) for k in ("W1","W2","b1","b2","smooth2")}
            result=read_bound(hf,cell["result"])
            diagnostics[key]={(x["index"],x["step"]):x["output_tensor_sha256"] for x in result["diagnostics"]["final_folded_BF16"]}
        predictions={}
        for item in plan["BH_predictions"]:
            names=tuple(x["case"]["case_id"]+".head_BF16" for x in manifest["cases"] if x["case"]["expert"]==item["expert"])
            with ledger("BH_prediction_pack_loads"):
                predictions.update(load_tensors(labels.bound(hf,item["binding"]),names))
        for item in manifest["cases"]:
            check();pack=load_case(item);case=item["case"];expert=case["expert"]
            report["active_case"]={"case_id":case["case_id"],"expert":expert,"step":case["step"],"phase":"SAVED_BH_AND_A0"}
            with ledger("V1_only_pack_loads"):
                v1=load_tensors(labels.bound(hf,{"path":case["operand_path"],"bytes":case["operand_bytes"],"sha256":case["operand_sha256"]}),("V1",))["V1"]
            with ledger("V1_tensor_checks"):
                require(v1.dtype==torch.bfloat16 and list(v1.shape)==[256,5120]
                    and bool(torch.isfinite(v1).all()), "Actual saved postQ1 V1 differs")
                v1_sha=tensor_sha(torch,v1)
            residual,gate,rows,v1=[upload(torch,v,"cuda",ledger) for v in (pack["r_rows"],pack["gate"],pack["rows"],v1)]
            scale,shift=conditioning(pack,item)
            target=upload(torch,pack["head_rows"],"cuda",ledger)
            bh=predictions[case["case_id"]+".head_BF16"]
            with ledger("saved_BH_head_checks"):
                require(bh.dtype==torch.bfloat16 and bh.shape==pack["head_rows"].shape, "Saved BH head geometry differs")
                bh_sha=tensor_sha(torch,bh)
            bh=upload(torch,bh,"cuda",ledger)
            row={k:case[k] for k in ("case_id","source_index","source_id","role","step","expert","seed")}
            row["V1_raw_sha256"]=v1_sha
            row["rows_sha256"]=item["tensor_sha256"]["rows"];row["arms"]={"BH_saved":{**head_metrics(torch,bh,target,ledger),"saved_head_sha256":bh_sha}}
            report["cases"].append(row)
            for initialization in MODES[1:]:
                report["active_case"]["initialization"]=initialization
                arm=row["arms"][initialization]={"status":"STARTED"};persist()
                y,u=a0_postwrite(torch,v1,residual,gate,students[expert,initialization],quantizer,ledger)
                check_a0_output(torch,y,diagnostics[expert,initialization][case["source_index"],case["step"]],arm,ledger)
                selected,head_sha=fullM_readout(torch,u,rows,heads[expert],scale,shift,ledger)
                arm.update(**head_metrics(torch,selected,target,ledger),head_sha256=head_sha,status="COMPLETE")
            persist()
        ledger.finish();report["source_summary"]=summarize(report["cases"])
        report["status"]="COMPLETE_MATCHED_CAL24_HEAD_ERRORS_NATIVE_NEUTRAL96_A0_192_BH_SAVED96"
        report["active_case"]=None
        check()
    except BaseException as error:
        report.update(status="FAILED_NO_RETRY",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
    finally:
        original_error=sys.exc_info()[1]
        failure=None
        try:
            if monitor is not None:
                monitor.close();report["assigned_GPU_resources"]=monitor.report()
                require(not report["assigned_GPU_resources"].get("violations"),"Assigned GPU monitor violation")
        except BaseException as error:
            report["status"]="FAILED_GPU_MONITOR";report["monitor_error"]={"type":type(error).__name__,"message":str(error)};failure=error
        report["wall_seconds"]=time.monotonic()-started
        if torch is not None and torch.cuda.is_initialized():report["peak_GPU_reserved_bytes"]=torch.cuda.max_memory_reserved()
        journal.close()
        report["operation_journal"]={"path":"OPERATIONS.jsonl","sha256":labels.sha(out/"OPERATIONS.jsonl")}
        persist();signal.alarm(0)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if failure is not None and original_error is None:raise failure
        if failure is not None and original_error is not None:original_error.add_note("Monitor cleanup failure retained in matched-head RUN")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("inspect","preflight","execute"));parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--hf-root",type=Path);parser.add_argument("--out",type=Path);args=parser.parse_args()
    if args.command=="inspect":
        plan=json.loads(args.protocol.read_text());validate_plan(plan);print(json.dumps({"id":ID,"binding_status":plan["binding_status"],"limits":exact_limits(),"native_calls_executed":0},indent=2))
    elif args.command=="preflight":
        require(args.hf_root is not None and args.out is not None and not args.out.exists(),"Fresh independent CPU report required")
        report=preflight(args.protocol,args.hf_root.resolve())
        with args.out.open("x") as stream:json.dump(report,stream,indent=2);stream.write("\n")
    else:
        require(args.hf_root is not None,"HF root required");run(args)


if __name__=="__main__":main()
