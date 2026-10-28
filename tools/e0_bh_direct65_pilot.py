#!/usr/bin/env python3
"""Registered saved-CAL24 B-H h128 direct65 predictability pilot.

No pretrained model/FFN/video invocation. One idle GPU trains the two terminal
expert students sequentially. Real fitting is disabled until immutable code,
audited labels, exact budget, independent CPU preflight and ACTIVE are bound.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

import bh_direct65_student as bh
import b_terminal_paired_discrimination as prior

ROOT = Path(__file__).resolve().parents[1]
ID = "E0_BH_RESIDUAL_ANCHORED_DIRECT65_H128_20261028"
RESOURCES = {"assigned_GPUs": 1, "session_max_simultaneously_assigned_idle_GPUs": 2,
             "repository_global_GPU_policy": "DYNAMIC_MAX_ACTUALLY_CALLABLE",
             "CPU_threads": 8, "max_GPU_peak_allocated_bytes": 4 * (1 << 30),
             "max_output_bytes": 64 * (1 << 20), "automatic_retry": False,
             "native_Wan_calls": 0, "video_generations": 0}


def exact_budget():
    # 96 prep anchors + 192 initial/final evaluation anchors = 288.
    # 192 teacher-statistic reconstructions + 192 evaluations = 384 readouts.
    # Thus range and quadratic GEMMs each 288+384=672, always paid even rank64.
    D, h, n = 5120, 64, 256
    case_H2D = n * (4 * D + 2 * h + 4 * (h + 1) + 4 * h + 4 + 4 * h)
    geometry_H2D = 4 * 4 * (h * D + 2 * h * h + h)
    master_H2D = 2 * 4 * bh.parameter_count()
    return {"CPU_head_pack_loads": 2, "CPU_step_conditioning_pack_loads": 4, "CPU_BF16_gain_adds": 4,
            "geometry_FP64_Gram_GEMM": 4, "geometry_symmetric_eigh": 4,
            "geometry_FP64_inverse_and_projector_GEMM": 8, "geometry_FP64_shift_GEMM": 4,
            "geometry_FP64_projector_validation_GEMM": 4,
            "case_pack_reads": 96, "saved_rows_payload_validations": 96, "prepared_source_state_cases": 96,
            "offline_FP64_teacher_head_projection_GEMM": 96, "offline_FP64_teacher_quadratic_GEMM": 96,
            "offline_FP64_teacher_readout": 96, "device_case_transfers": 96,
            "teacher_label_device_transfers": 96, "CUDA_H2D_tensor_copies": 698,
            "CUDA_H2D_payload_bytes": 96 * case_H2D + geometry_H2D + master_H2D,
            "train_normalization_mean_moment_cases": 64, "train_normalization_centered_moment_cases": 64,
            "normalizer_compiles": 2, "student_random_weight_draws": 4,
            "online_FP32_residual_projection_GEMM": 288, "online_FP32_range_projection_GEMM": 672,
            "online_FP32_quadratic_GEMM": 672, "online_residual_variance_reductions": 288,
            "online_anchor_summary": 288, "teacher_statistic_reconstruction_readouts": 192,
            "student_direct_readout": 384, "student_predict65_calls": 704,
            "student_input_normalizations": 704, "student_BF16_forward_GEMM": 2112,
            "student_BF16_tanh_GELU": 704, "training_source_batches": 512,
            "training_loss_evaluations": 512, "backward_calls": 512,
            "student_BF16_backward_GEMM": 2048, "finite_gradient_tensor_checks": 2560,
            "Adam_steps": 512, "Adam_parameter_tensor_updates": 2560,
            "finite_updated_parameter_tensor_checks": 2560, "diagnostic_source_state_forwards": 192,
            "initial_zero_output_checks": 96, "prediction_pack_writes": 4,
            "student_export_writes": 2, "export_reload_byte_checks": 2,
            "native_Wan_FFN_calls": 0, "native_LN_head_calls": 0, "student_Q2_calls": 0,
            "full_M_student_forwards": 0, "video_generations": 0}


class Ledger:
    def __init__(self, report, persist=lambda: None):
        self.report, self.persist, self.limits = report, persist, exact_budget()
        for key in ("predebited", "invoked", "completed"):
            report[key] = dict.fromkeys(self.limits, 0)

    def __call__(self, key, count=1):
        @contextmanager
        def operation():
            bh.check(key in self.limits and type(count) is int and count > 0 and self.report["predebited"][key] + count <= self.limits[key], "Exceeded fixed B-H work: " + key)
            self.report["predebited"][key] += count
            self.persist()
            self.report["invoked"][key] += count
            self.report["last_invoked"] = key
            self.persist()
            yield
            self.report["completed"][key] += count
            self.persist()
        return operation()

    def finish(self):
        bh.check(all(self.report[k] == self.limits for k in ("predebited", "invoked", "completed")), "B-H fixed budget did not close")


def validate_plan(plan, frozen=False):
    bh.check(plan.get("id") == ID and plan.get("parameters") == bh.PARAMETERS and plan.get("resources") == RESOURCES and plan.get("limits") == exact_budget(), "Changed B-H graph/optimizer/resource/budget")
    bh.check(plan.get("source_split") == {"training": list(bh.TRAIN), "diagnostic_heldout": list(bh.HELD)}, "Changed CAL24 split")
    bh.check(plan.get("parent_protocol", {}).get("sha256") == prior.PARENT_SHA, "Only actual completed CAL24 _v1 labels accepted")
    if frozen:
        bh.check(plan.get("assets_finalized") is True and plan.get("binding_status") == "BOUND_READY_FOR_REGISTRATION", "B-H assets/code not frozen")
        names = {x["path"] for x in plan["repo_sources"]}
        bh.check({"tools/bh_direct65_student.py", "tools/e0_bh_direct65_pilot.py", "tools/b_terminal_paired_discrimination.py", "tools/b_cal24_label_contract.py", "tests/test_bh_direct65_student.py"} <= names, "Missing immutable B-H dependencies")


def conditioning_identities(manifest):
    """Metadata-only gate: no source-dependent conditioning is silently removed."""
    signatures = {}
    for step in range(4):
        cases = [x for x in manifest["cases"] if x["case"]["step"] == step]
        bh.check(len(cases) == 24, "Missing source conditioning")
        observed = {tuple(item["tensor_sha256"][k] for k in ("gate", "head_scale", "head_shift")) for item in cases}
        bh.check(len(observed) == 1, "Source-dependent gate/head modulation requires a different graph; no fallback")
        signatures[step] = dict(zip(("gate", "head_scale", "head_shift"), next(iter(observed))))
    return signatures


def preflight(protocol, hf):
    bh.check("torch" not in sys.modules and "numpy" not in sys.modules and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Independent byte-only CPU preflight requires hidden GPUs and no numerical imports")
    reject_precision_overrides()
    plan = json.loads(protocol.read_text())
    validate_plan(plan, frozen=True)
    for item in plan["repo_sources"]:
        prior.bound(ROOT, item)
    versions = {k: importlib.metadata.version(k) for k in ("numpy", "torch", "safetensors")}
    bh.check(versions == plan["runtime_versions"], "B-H numerical versions differ")
    manifest = json.loads(prior.bound(hf, plan["label_manifest"]).read_text())
    prior.verify_manifest(manifest, plan, hf)
    identities = conditioning_identities(manifest)
    report = {"id": ID, "status": "PASS_BH_DIRECT65_CPU_BINDINGS_NO_NUMPY_NO_TORCH_NO_GPU",
              "protocol_sha256": prior.sha(protocol), "source_bindings_sha256": prior.state_sha(plan["repo_sources"]),
              "label_manifest_sha256": plan["label_manifest"]["sha256"], "conditioning_identities": identities,
              "cases": 96, "source_split": plan["source_split"], "runtime_versions": versions,
              "NumPy_imported": "numpy" in sys.modules, "Torch_or_CUDA_imported": "torch" in sys.modules or "torch.cuda" in sys.modules,
              "all_CUDA_devices_hidden": True, "model_calls": 0, "GPU_calls": 0, "training_updates": 0, "decompositions": 0}
    return plan, manifest, report


def reject_precision_overrides():
    bh.check(all(os.environ.get(key) != "1" for key in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")), "Environment forces TF32 despite the fixed FP32 projection contract")


def authorize(plan, digest):
    active = json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text())
    user = active.get("user_continuation_authorization_20261027", {})
    session = active.get("current_session_resource_constraint_20261028", {})
    policy = active.get("resource_scheduling_policy_20261028", {})
    bh.check(user.get("status") == "ACTIVE_CURRENT_USER_DIRECTED_SESSION" and user.get("new_scope_registration_allowed") is True, "Offline research session is inactive")
    bh.check(session.get("max_simultaneously_assigned_idle_GPUs") == 2 and policy.get("mode") == "DYNAMIC_MAX_ACTUALLY_CALLABLE" and policy.get("global_gpu_card_cap") is None, "Current session constraint/general dynamic resource policy differs")
    selected = [s for s in active["experiment_authority"]["current_user_authorized_scopes"] if s.get("id") == ID and s.get("protocol_sha256") == digest and s.get("execution_status") in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")]
    bh.check(len(selected) == 1 and selected[0].get("max_gpu_cards") == 1 and selected[0].get("session_max_simultaneously_assigned_idle_GPUs") == 2 and selected[0].get("limits") == exact_budget(), "Unregistered exact B-H student scope")
    return selected[0]


def prior_cpu_receipt(entry, plan, digest, hf):
    cpu = entry.get("cpu_preflight")
    bh.check(isinstance(cpu, dict), "Independent CPU preflight NOT_BOUND in ACTIVE")
    report = json.loads(prior.bound(hf, cpu["binding"]).read_text())
    receipt = json.loads(prior.bound(hf, cpu["launch_exit"]).read_text())
    bh.check(report.get("id") == ID and report.get("status") == "PASS_BH_DIRECT65_CPU_BINDINGS_NO_NUMPY_NO_TORCH_NO_GPU" and report.get("protocol_sha256") == digest and report.get("source_bindings_sha256") == prior.state_sha(plan["repo_sources"]) and report.get("label_manifest_sha256") == plan["label_manifest"]["sha256"] and report.get("NumPy_imported") is False and report.get("Torch_or_CUDA_imported") is False and report.get("all_CUDA_devices_hidden") is True and report.get("cases") == 96, "Independent CPU preflight differs")
    bh.check(receipt.get("id") == ID and receipt.get("protocol_sha256") == digest and receipt.get("launcher_returncode") == 0 and receipt.get("process_exited") is True and receipt.get("RUN_sha256") == cpu["binding"]["sha256"], "CPU preflight actual exit differs")
    return cpu


def tensors_cpu(path, names):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {k: handle.get_tensor(k).contiguous() for k in names}


def tensor_sha(torch, value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def same_bits(torch, a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def load_geometries(torch, manifest, hf, p, work):
    heads, tables, records, conditioning = {}, {}, {}, {}
    for item in manifest["head_assets"]:
        with work("CPU_head_pack_loads", 1):
            pack = tensors_cpu(prior.bound(hf, item["binding"]), ("head_weight_runtime_KN", "head_bias"))
        heads[item["expert"]] = pack
    for step in range(4):
        item = next(c for c in manifest["cases"] if c["case"]["source_index"] == 0 and c["case"]["step"] == step)
        with work("CPU_step_conditioning_pack_loads", 1):
            pack = tensors_cpu(prior.bound(hf, item["label"]), ("gate", "head_scale", "head_shift"))
        with work("CPU_BF16_gain_adds", 1):
            gain = 1 + pack["head_scale"].reshape(-1)
        head = heads[item["case"]["expert"]]
        tables[step] = bh.compile_geometry(head["head_weight_runtime_KN"].double().numpy(), gain.double().numpy(), pack["head_shift"].reshape(-1).double().numpy(), head["head_bias"].double().numpy(), p, work)
        records[str(step)] = tables[step]["record"]
        conditioning[step] = {k: tensor_sha(torch, v) for k, v in pack.items()}
    return tables, records, conditioning


def prepare_cases(torch, manifest, hf, expert, tables, device_tables, conditioning, p, work):
    import numpy as np
    cached, provenance = {}, []
    selected = sorted([x for x in manifest["cases"] if x["case"]["expert"] == expert], key=lambda x: (x["case"]["source_index"], x["case"]["step"]))
    bh.check(len(selected) == 48, "Expert needs all 24 sources x2 steps")
    for item in selected:
        case = item["case"]
        with work("case_pack_reads", 1):
            pack = tensors_cpu(prior.bound(hf, item["label"]), ("r_rows", "u_rows", "head_rows", "gate", "head_scale", "head_shift", "rows"))
            operand = prior.bound(hf, {"path": case["operand_path"], "bytes": case["operand_bytes"], "sha256": case["operand_sha256"]})
            # This consumer never asks the old pack for Y or V2.
            v1 = tensors_cpu(operand, ("V1",))["V1"]
        with work("saved_rows_payload_validations", 1):
            bh.check(pack["rows"].dtype == torch.int64 and pack["rows"].tolist() == case["rows"] and len(case["rows"]) == p["rows_per_case"], "Actual ordered CAL24 row payload differs")
        bh.check(v1.dtype == pack["r_rows"].dtype == pack["u_rows"].dtype == torch.bfloat16 and v1.shape == pack["r_rows"].shape == pack["u_rows"].shape == (p["rows_per_case"], p["D"]), "Actual original operand geometry differs")
        bh.check({k: tensor_sha(torch, pack[k]) for k in conditioning[case["step"]]} == conditioning[case["step"]], "Actual conditioning bytes changed across source")
        with work("device_case_transfers", 1):
            value = {"V1": bh.copy_to_device(torch, v1, "cuda", work), "r": bh.copy_to_device(torch, pack["r_rows"], "cuda", work),
                     "native_head": bh.copy_to_device(torch, pack["head_rows"], "cuda", work)}
        with torch.no_grad():
            anchor = bh.anchor_features(torch, value["r"], device_tables[case["step"]], p, work)
        reference_anchor = {k: anchor[k].detach().cpu().numpy() for k in ("qr", "slack")}
        truth = bh.truth_labels(pack["u_rows"].double().numpy(), reference_anchor, tables[case["step"]], p, work)
        with work("teacher_label_device_transfers", 1):
            truth_device = {key: bh.copy_to_device(torch, torch.from_numpy(np.asarray(truth[source], dtype=np.float32)), "cuda", work)
                for key, source in (("target65", "target65"), ("true_q", "q"), ("true_variance", "variance"), ("real_head", "real_head"))}
        value.update(source_index=case["source_index"], step=case["step"], role=case["role"], case=case, anchor=anchor, **truth_device)
        bh.check(all(bool(torch.isfinite(value[k]).all()) for k in ("V1", "r", "native_head", "target65")), "Nonfinite prepared data")
        cached[(case["source_index"], case["step"])] = value
        with work("prepared_source_state_cases", 1):
            provenance.append({"case": case, "label": item["label"], "V1_raw_sha256": tensor_sha(torch, v1),
                "r_raw_sha256": item["tensor_sha256"]["r_rows"], "u_raw_sha256": item["tensor_sha256"]["u_rows"],
                "head_raw_sha256": item["tensor_sha256"]["head_rows"], "anchor_diagnostics": {k: anchor[k] for k in ("qmin_diagnostic", "slack_diagnostic", "floor_rows")},
                "truth_diagnostics": truth["diagnostic"], "teacher_Y_or_V2_numerically_loaded": False})
    return cached, provenance


def metrics(torch, decoded, case):
    target = case["native_head"].float()
    prediction = decoded["head_BF16"].float()
    difference = prediction - target
    denominator = target.square().sum()
    bh.check(float(denominator) > 0, "Zero original native head norm")
    return {"head_relative_SSE": float(difference.square().sum() / denominator),
            "head_mean_row_SSE": float(difference.square().sum(dim=1).mean()),
            "q_relative_SSE": float((decoded["q"] - case["true_q"]).square().sum() / case["true_q"].square().sum().clamp_min(1e-30)),
            "variance_relative_SSE": float((decoded["variance"] - case["true_variance"]).square().sum() / case["true_variance"].square().sum().clamp_min(1e-30)),
            "qmin_diagnostic": decoded["qmin_diagnostic"],
            "physical_variance_slack_min": float(decoded["slack"].min()),
            "continuous_teacher_head_vs_native_mean_row_SSE": float((case["real_head"] - target).square().sum(dim=1).mean())}


def source_summary(rows, expert):
    summary = {}
    for name, indices in (("train", bh.TRAIN), ("heldout_exposed", bh.HELD)):
        source_values = {}
        for index in indices:
            matching = [r for r in rows if r["source_index"] == index]
            bh.check(len(matching) == 2 and {r["step"] for r in matching} == set(bh.steps_for(expert)), "Missing two nested expert steps")
            source_values[str(index)] = {k: sum(r[k] for r in matching) / 2 for k in ("head_relative_SSE", "head_mean_row_SSE", "q_relative_SSE", "variance_relative_SSE")}
        summary[name] = {"independent_source_count": len(indices), "per_source": source_values,
                         "equal_source_mean": {k: sum(v[k] for v in source_values.values()) / len(indices) for k in next(iter(source_values.values()))},
                         "identity": "historically exposed source clusters; rows and two steps nested"}
    return summary


def teacher_reconstructions(torch, cached, normalizer, geometries, p, work):
    rows = []
    with torch.no_grad():
        for case in cached.values():
            for dtype in (torch.float32, torch.bfloat16):
                raw = (case["target65"] / normalizer["target_rms"]).to(dtype)
                with work("teacher_statistic_reconstruction_readouts", 1):
                    decoded = bh.decode_statistics(torch, raw, case["anchor"], normalizer, geometries[case["step"]], p, work)
                rows.append({"case_id": case["case"]["case_id"], "source_index": case["source_index"], "step": case["step"],
                             "raw65_precision": str(dtype), **metrics(torch, decoded, case),
                             "identity": "teacher-statistic reconstruction reference, not an optimal native error lower bound"})
    return rows


def evaluate(torch, model, cached, normalizer, geometries, expert, phase, p, work):
    rows, predictions = [], {}
    with torch.no_grad():
        for case in cached.values():
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            begin.record()
            with work("diagnostic_source_state_forwards", 1):
                anchor = bh.anchor_features(torch, case["r"], geometries[case["step"]], p, work)
                raw = model.predict65(case["V1"], anchor["eta"], case["step"] - bh.steps_for(expert)[0], normalizer)
                decoded = bh.decode_statistics(torch, raw, anchor, normalizer, geometries[case["step"]], p, work)
            end.record()
            torch.cuda.synchronize()
            if phase == "initial":
                with work("initial_zero_output_checks", 1):
                    bh.check(bool((raw == 0).all()), "Zero-initialized output did not preserve residual-anchor baseline")
            rows.append({"case_id": case["case"]["case_id"], "source_index": case["source_index"], "step": case["step"], "role": case["role"],
                         "M": p["rows_per_case"], "phase": phase, "CUDA_event_ms_including_inline_accounting": float(begin.elapsed_time(end)),
                         "wall_seconds_including_inline_accounting": time.monotonic() - started,
                         **metrics(torch, decoded, case), "prediction_BF16_raw_sha256": tensor_sha(torch, decoded["head_BF16"]),
                         "full_M_runtime_claim": "NOT_MEASURED; M256 saved-state function evaluation only"})
            prefix = case["case"]["case_id"]
            predictions[prefix + ".head_BF16"] = decoded["head_BF16"].cpu().contiguous()
            predictions[prefix + ".q_FP32"] = decoded["q"].cpu().contiguous()
            predictions[prefix + ".variance_FP32"] = decoded["variance"].cpu().contiguous()
    return rows, predictions


def save_export(torch, path, model, normalizer, tables, expert, work):
    from safetensors.torch import load_file, save_file
    with work("student_export_writes", 1):
        pack = {}
        for name, value in model.state_dict().items():
            pack["master." + name] = value.detach().cpu().contiguous()
            pack["BF16." + name] = value.detach().bfloat16().cpu().contiguous()
        pack.update({"normalizer." + k: v.detach().cpu().contiguous() for k, v in normalizer.items()})
        for step in bh.steps_for(expert):
            pack.update({f"step{step}." + k: tables[step][k].detach().cpu().contiguous() for k in ("K", "Gplus", "projector", "c0")})
        # Each serialized name owns its bytes, even if two step tables happen
        # to share identical storage in an upstream cache.
        pack = {k: v.clone() for k, v in pack.items()}
        save_file(pack, str(path), metadata={"id": ID, "expert": expert, "scope": "M256 predictability pilot; postQ1 V1+r input; no hidden Q2", "teacher_Y_online": "false"})
    with work("export_reload_byte_checks", 1):
        reloaded = load_file(str(path), device="cpu")
        bh.check(set(reloaded) == set(pack) and all(same_bits(torch, pack[k], reloaded[k]) for k in pack), "Exported student/master/static bytes differ")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": prior.sha(path),
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "raw_sha256": tensor_sha(torch, v)} for k, v in pack.items()}}


def run(args):
    protocol, hf = args.protocol.resolve(), args.hf_root.resolve()
    reject_precision_overrides()
    # Numeric execution repeats the byte checks with CUDA hidden temporarily;
    # it still requires a *prior independently exited* preflight bound in ACTIVE.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    bh.check(visible is not None and visible.strip() and len(visible.split(",")) == 1, "One explicitly assigned GPU required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        plan, manifest, verification = preflight(protocol, hf)
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    digest = prior.sha(protocol)
    entry = authorize(plan, digest)
    receipt = prior_cpu_receipt(entry, plan, digest, hf)
    out = (hf / plan["output_HF_relative"]).resolve()
    bh.check(out != hf and out.is_relative_to(hf) and not out.exists(), "Fresh HF result directory required; no retries")
    out.mkdir(parents=True, exist_ok=False)
    report = {"id": ID, "protocol_sha256": digest, "status": "RUNNING_FIXED_SAVED_STATE_STUDENT_PILOT", "binding_preflight": verification,
              "prior_independent_CPU_preflight": receipt, "resources": RESOURCES, "logical_cost": bh.logical_cost(),
              "parameters_per_expert": bh.parameter_count(), "A0_parameters_per_expert": 712704,
              "parameter_difference_to_A0": bh.parameter_count() - 712704, "experts": [],
              "quality_identity": "CAL24 teacher-state M256 function predictability, all sources historically exposed; no video/rollout/system-speed conclusion"}
    def persist():
        prior.write(out / "RUN.json", report)
    ledger = Ledger(report, persist)
    persist()
    started = time.monotonic()
    try:
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[key] = str(RESOURCES["CPU_threads"])
        import numpy as np
        import torch
        from safetensors.torch import save_file
        torch.set_num_threads(RESOURCES["CPU_threads"])
        torch.set_num_interop_threads(1)
        bh.check(torch.cuda.device_count() == 1, "CUDA assignment is not one card")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.cuda.reset_peak_memory_stats()
        report["numerical_runtime"] = {"numpy": np.__version__, "torch": torch.__version__, "GPU_name": torch.cuda.get_device_name(0),
            "FP32_TF32": False, "BF16_reduced_precision_reduction": False, "training_gradient": "FP32 masters; BF16 casts/forward/backward matrix products; autograd cast-gradient surrogate"}
        p = plan["parameters"]
        cpu_tables, geometry_records, conditioning = load_geometries(torch, manifest, hf, p, ledger)
        report["head_geometry"] = geometry_records
        tables = {s: bh.device_geometry(v, torch, "cuda", ledger) for s, v in cpu_tables.items()}
        for expert in bh.EXPERTS:
            cached, provenance = prepare_cases(torch, manifest, hf, expert, cpu_tables, tables, conditioning, p, ledger)
            normalization = bh.normalizers(torch, [v for v in cached.values() if v["source_index"] in bh.TRAIN], p, ledger)
            model = bh.Direct65Student(p["D"], p["head_outputs"], p["hidden"], p["seed"], device="cuda", work=ledger)
            bh.check(sum(v.numel() for v in model.parameters()) == bh.parameter_count(), "Student parameter count differs")
            folder = out / expert
            folder.mkdir()
            provenance_path = folder / "PROVENANCE.json"
            prior.write(provenance_path, provenance)
            row = {"expert": expert, "parameters": bh.parameter_count(), "initialization": "Wx/Weta fixed Gaussian, zero Wout and both step biases; exact zero65 residual anchor with declared slack floor",
                   "provenance": {"path": str(provenance_path.relative_to(out)), "bytes": provenance_path.stat().st_size, "sha256": prior.sha(provenance_path)},
                   "teacher_Y_or_V2_numerically_loaded": False}
            reconstruction = teacher_reconstructions(torch, cached, normalization, tables, p, ledger)
            prior.write(folder / "TEACHER_STATISTIC_RECONSTRUCTION.json", reconstruction)
            for phase in ("initial", "final"):
                if phase == "final":
                    row["epoch_losses"] = bh.training_loop(torch, model, cached, normalization, expert, p, ledger,
                        on_epoch=lambda e: print(expert, "epoch", e["epoch"], "loss", e["mean_source_step_standardized_loss"], flush=True))
                metrics_rows, prediction_pack = evaluate(torch, model, cached, normalization, tables, expert, phase, p, ledger)
                prior.write(folder / (phase.upper() + "_METRICS.json"), metrics_rows)
                row[phase + "_source_summary"] = source_summary(metrics_rows, expert)
                prediction_path = folder / (phase + "_predictions.safetensors")
                with ledger("prediction_pack_writes", 1):
                    save_file(prediction_pack, str(prediction_path), metadata={"id": ID, "expert": expert, "phase": phase, "M": "256"})
                row[phase + "_prediction_pack"] = {"path": str(prediction_path.relative_to(out)), "bytes": prediction_path.stat().st_size, "sha256": prior.sha(prediction_path)}
                del prediction_pack
            row["export"] = save_export(torch, folder / "student.safetensors", model, normalization, tables, expert, ledger)
            report["experts"].append(row)
            persist()
            del cached, normalization, model
            torch.cuda.empty_cache()
            bh.check(torch.cuda.max_memory_allocated() <= RESOURCES["max_GPU_peak_allocated_bytes"], "Student pilot peak allocated memory exceeded")
        ledger.finish()
        report["status"] = "COMPLETE_BH_DIRECT65_H128_FIXED256_UPDATES_PER_EXPERT_M256"
    except BaseException as error:
        report.update(status="FAILED_FIXED_BH_SCOPE_NO_AUTOMATIC_RETRY", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        if "torch" in locals():
            report["GPU_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["GPU_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        persist()
        size = sum(x.stat().st_size for x in out.rglob("*") if x.is_file())
        if size > RESOURCES["max_output_bytes"]:
            report["status"] = "FAILED_FIXED_BH_OUTPUT_BUDGET_EXCEEDED"
            report["actual_output_bytes"] = size
            persist()
            raise RuntimeError("B-H output budget exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "preflight", "execute"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        plan = json.loads(args.protocol.read_text())
        validate_plan(plan)
        print(json.dumps({"id": ID, "binding_status": plan["binding_status"], "logical_cost": bh.logical_cost(), "limits": exact_budget(), "real_training_updates_executed": 0}, indent=2))
    elif args.command == "preflight":
        bh.check(args.hf_root is not None and args.out is not None and not args.out.exists(), "Independent preflight requires HF root and fresh report")
        _, _, report = preflight(args.protocol, args.hf_root.resolve())
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        bh.check(args.hf_root is not None, "Execute requires HF root")
        run(args)


if __name__ == "__main__":
    main()
