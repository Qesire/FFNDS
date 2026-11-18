"""Repair only the genuinely nonconverged hidden fits of a completed CAL shard."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from . import mode_local
from .full80_cal import check_modes, inventory, load_strict_site, numerical_acceptance, ref, same_ref
from .local_data import SITES, STEPS, bound_json, checked, fill_probes, load_pair, native_runtime, require, sha_file
from .local_worker import allocation, bind_fold_receipts, event, write_json


PCG_CAP = 80
TOLERANCE = 1e-5
MODE_FAMILY = "native_modes"
SOURCE_METHOD = "OBSERVED_SECANT_FULL80_CAL"
RECOVERY_METHOD = "OBSERVED_SECANT_FULL80_MODE_RECOVERY"


def direct_hidden_response(dz, w2, residual, gate):
    """Solve the same FP64 source equation in symmetrically scaled coordinates."""
    require(all(value.dtype == torch.float64 for value in (dz, w2, residual, gate)),
            "Direct source solve requires the same FP64 operands as PCG")
    n = len(dz)
    gate2 = gate.square()
    diag = dz.square().mean(dim=0) * (gate2 @ w2.square())
    active = diag > 0
    rhs = (dz * ((residual * gate2) @ w2)).sum(dim=0) / n
    rhs = torch.where(active, rhs, 0)
    result = torch.zeros_like(rhs)
    rhs_norm = torch.linalg.vector_norm(rhs).item()
    require(math.isfinite(rhs_norm) and bool(torch.isfinite(diag).all()), "Nonfinite source normal operands")
    metadata = {"method": "FP64_HADAMARD_GRAM_SCALED_CHOLESKY", "active_channels": int(active.sum()),
                "tolerance": TOLERANCE, "verified_by": "ORIGINAL_MATRIX_FREE_NORMAL_ACTION"}
    if rhs_norm == 0:
        return result, {**metadata, "status": "CONVERGED", "relative_residual": 0.0}

    active_diag = diag[active]
    regularizer = 0.001 * active_diag.clamp_min(1e-12 * active_diag.mean())
    inverse_sqrt = (active_diag + regularizer).rsqrt()
    z = dz if metadata["active_channels"] == len(active) else dz[:, active]
    weighted_w = (w2 if metadata["active_channels"] == len(active) else w2[:, active]) * gate[:, None]
    normal = (z.T @ z).div_(n)
    normal.mul_(weighted_w.T @ weighted_w)
    normal.diagonal().add_(regularizer)
    normal.mul_(inverse_sqrt[:, None]).mul_(inverse_sqrt[None, :])
    lower, info = torch.linalg.cholesky_ex(normal)
    del normal
    require(info.item() == 0, "Scaled direct source matrix did not factor as SPD")
    scaled_rhs = (rhs[active] * inverse_sqrt).unsqueeze(-1)
    scaled_solution = torch.cholesky_solve(scaled_rhs, lower).squeeze(-1)
    result[active] = scaled_solution * inverse_sqrt

    # Verify against the original matrix-free operation, including inactive rows.
    response = (dz * result) @ w2.T
    normal_times_result = (dz * ((response * gate2) @ w2)).sum(dim=0) / n
    normal_times_result[active] += regularizer * result[active]
    relative = torch.linalg.vector_norm(rhs - normal_times_result).item() / rhs_norm
    ok = relative <= TOLERANCE and bool(torch.isfinite(result).all())
    return result, {**metadata, "status": "CONVERGED" if ok else "FIT_NOT_CONVERGED",
                    "relative_residual": relative if math.isfinite(relative) else None}


def source_solver(archived):
    """Retain the original PCG algorithm for every source and recover only archived failures."""
    prior = {row["source"]: row for row in archived}

    def solve(source, dz, w2, residual, gate):
        original = prior[source]
        coefficients, replay = mode_local.fit_hidden_response(
            dz, w2, residual, gate, max_iterations=PCG_CAP, tolerance=TOLERANCE)
        require(replay["status"] == original["status"]
                and replay["active_channels"] == original["active_channels"],
                f"{source}: archived source PCG outcome did not reproduce")
        record = {"original_pcg": original, "replayed_pcg": replay, "direct": None}
        if original["status"] == "CONVERGED":
            return coefficients, {**replay, **record}
        start = perf_counter()
        coefficients, direct = direct_hidden_response(dz, w2, residual, gate)
        direct["elapsed_seconds"] = perf_counter() - start
        return coefficients, {**replay, **record, "status": direct["status"], "direct": direct,
                              "relative_residual": direct["relative_residual"]}

    return solve


def parent_shard(scope):
    parent_ref = scope["parent_cal_scope"]
    parent = bound_json(parent_ref)
    manifest_ref = scope["parent_cal_manifest"]
    manifest = bound_json(manifest_ref)
    exit_ref = scope["parent_cal_exit"]
    exited = bound_json(exit_ref)
    stage = Path(parent["stage"])
    require(stage.is_absolute() and Path(parent_ref["path"]) == stage / "scope.json"
            and Path(exit_ref["path"]) == stage / "EXIT.json"
            and Path(manifest_ref["path"]) == stage / "output/CAL_MANIFEST.json",
            "Parent CAL references escaped the completed shard")
    run = json.loads((stage / "RUN.json").read_bytes())
    attempt = json.loads((stage / "ATTEMPT_EXIT.json").read_bytes())
    expert = scope["expert"]
    require(expert in STEPS and parent["expert"] == manifest["expert"] == expert
            and parent["method"] == SOURCE_METHOD and parent["max_nfe"] == parent["max_d4"] == 0
            and run["scope_id"] == exited["scope_id"] == parent["scope_id"]
            and run["scope_sha256"] == attempt["scope_sha256"] == parent_ref["sha256"]
            and run["slurm_job_id"] == exited["slurm_job_id"] == attempt["slurm_job_id"]
            and exited["exit_code"] == attempt["exit_code"] == 1
            and manifest["schema"] == "FFNDS_OBSERVED_SECANT_FULL80_CAL_V1"
            and manifest["status"] == "EXECUTION_NOT_READY"
            and manifest["site_count"] == 40 and manifest["new_site_count"] == 37
            and manifest["nfe"] == manifest["d4"] == 0 and manifest["local_quality_gate"] is False,
            "Parent did not complete the original zero-trajectory CAL shard with reported unready sites")
    for name in ("bindings", "q1_cal_pairs", "strict_q1", "experiment_spec", "local6_exit",
                 "local6_parameters", "local6_acceptance"):
        require(scope[name]["sha256"] == parent[name]["sha256"], f"Parent {name} changed")
    for name in ("native_sources", "torch_version", "torch_cuda", "numpy_version", "python_executable_sha256"):
        require(scope[name] == parent[name], f"Parent runtime {name} changed")
    binding = bound_json(manifest["cal_bindings"])
    acceptance = bound_json(manifest["acceptance"])
    require(binding["status"] == "PASS" and binding["scope_sha256"] == parent_ref["sha256"]
            and binding["verified_expert_pairs"] == len(binding["pairs"]) == 440
            and binding["cal_sources"] == manifest["sources"] and binding["expert"] == expert
            and acceptance["status"] == "EXECUTION_NOT_READY" and acceptance["nfe"] == acceptance["d4"] == 0
            and str(acceptance["runtime"]["job_id"]) == str(exited["slurm_job_id"])
            and same_ref(binding["parent_bindings"], parent["bindings"])
            and set(manifest["sites"]) == set(acceptance["sites"]) == {f"{expert}{i:02d}" for i in range(40)},
            "Incomplete parent CAL evidence")
    return manifest, binding, acceptance["sites"], acceptance["fold_identities"]


def failed_sites(scope, manifest, acceptance):
    expert = scope["expert"]
    names = scope["recovery_sites"]
    require(isinstance(names, list) and len(names) == len(set(names)) > 0,
            "Review scope must name a finite nonempty list of distinct failed sites")
    result = {}
    for name in names:
        require(name in acceptance and (expert, int(name[-2:])) not in SITES
                and name in manifest["site_receipts"], f"{name}: not a newly fitted parent site")
        param_ref, receipt_ref = (manifest[key][name] for key in ("sites", "site_receipts"))
        doc, receipt = bound_json(param_ref), bound_json(receipt_ref)
        checks = acceptance[name]
        require(doc["site"] == receipt["site"] == name and doc["identity"] == receipt["identity"]
                and receipt["acceptance"] == checks and receipt["valid"] is False
                and checks["provenance"] == "FIT_CAL11" and checks["observed_readout"]["valid"] is True
                and doc["observed_readout"] and doc[MODE_FAMILY] is None
                and checks[MODE_FAMILY]["status"] == "FIT_NOT_CONVERGED"
                and checks[MODE_FAMILY]["valid"] is False,
                f"{name}: recovery requires an accepted readout and a genuine mode PCG failure")
        tuning = receipt["tuning"][MODE_FAMILY]
        pcg = tuning["pcg"]
        require(tuning["status"] == "FIT_NOT_CONVERGED" and [row["source"] for row in pcg] == manifest["sources"]
                and any(row["status"] == "FIT_NOT_CONVERGED" for row in pcg)
                and all(0 <= row["iterations"] <= PCG_CAP and (
                    (row["status"] == "CONVERGED" and not row["breakdown"]
                     and 0 <= row["relative_residual"] <= TOLERANCE)
                    or (row["status"] == "FIT_NOT_CONVERGED" and (
                        row["breakdown"] or (row["relative_residual"] is not None
                                             and row["relative_residual"] > TOLERANCE))))
                        for row in pcg), f"{name}: archived PCG receipt lacks source-level evidence")
        result[name] = (doc, param_ref, receipt_ref, pcg)
    return result


@torch.inference_mode()
def run(scope_path):
    scope = json.loads(scope_path.read_bytes())
    require(scope["method"] == RECOVERY_METHOD and scope["max_nfe"] == scope["max_d4"] == scope["retries"] == 0
            and Path(scope["stage"]) == Path.cwd(), "Unexpected mode recovery scope")
    manifest, parent_binding, acceptance, parent_folds = parent_shard(scope)
    selected = failed_sites(scope, manifest, acceptance)
    spec = bound_json(scope["experiment_spec"])
    require(spec["execution_eligibility"]["local_metric_video_gate"] is False
            and spec["execution_eligibility"]["local_oracle_video_gate"] is False
            and spec["assets"]["new_dense_capture_in_budget"] is False, "Expected amended CAL-only policy")
    device, allocation_record = allocation(scope)
    archive, bank, _, _, operators = inventory(scope, spec)
    expert = scope["expert"]
    checkpoint = checked(archive["checkpoints"][expert])
    require(archive["cal_sources"] == manifest["sources"]
            and parent_binding["checkpoints"] == archive["checkpoints"], "Parent archive identity changed")
    pairs = {(row["source"], row["expert"], row["layer"]): row for row in parent_binding["pairs"]}
    require(set(pairs) == {(s, expert, layer) for s in archive["cal_sources"] for layer in range(40)},
            "Parent did not bind every expected expert/source CAL pair")
    runtime = native_runtime(scope, archive)
    require(torch.get_float32_matmul_precision() == "highest" and not torch.backends.cuda.matmul.allow_tf32
            and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, "Native import changed precision")

    output = Path("output")
    output.mkdir()
    (output / "site_parameters").mkdir()
    (output / "site_receipts").mkdir()
    ids, replacements, receipts = {}, {}, {}
    for name, (original, original_ref, old_receipt, pcg) in selected.items():
        layer = int(name[-2:])
        event(output, "RECOVER_SITE", site=name)
        site = load_strict_site(expert, layer, operators[expert, layer], checkpoint, device, runtime)
        require(site["identity"] == original["identity"], f"{name}: original CAL tensor identity changed")
        panels = []
        for source in archive["cal_sources"]:
            pair = load_pair(pairs[source, expert, layer], archive, bank, ids, device)
            fill_probes(site, pair)
            panels.append(pair)
        require(bind_fold_receipts(ids) == parent_folds, f"{name}: frozen fold IDs differ from parent CAL")
        fitted = mode_local.process_site(site, panels, [], structure=False, hidden_fit=source_solver(pcg))
        params = fitted["parameters"]
        folds = check_modes(site, panels, params) if fitted["status"] == "OK" else []
        valid = bool(params) and numerical_acceptance(folds)
        receipt = output / "site_receipts" / f"{name}.json"
        write_json(receipt, {
            "site": name, "identity": site["identity"], "valid": valid,
            "original_parameters": original_ref, "original_receipt": old_receipt,
            "acceptance": {**acceptance[name], "provenance": "RECOVER_MODE_CAL11",
                           MODE_FAMILY: {"status": fitted["status"], "valid": valid, "folds": folds}},
            "mode_fit": {key: fitted.get(key) for key in ("status", "pcg", "spectrum", "tuning",
                                                        "tuning_mean_mse", "matrix_compile_cost", "acceptance")},
            "nfe": 0, "d4": 0,
        })
        receipts[name] = ref(receipt)
        if valid:
            target = output / "site_parameters" / f"{name}.json"
            write_json(target, {**original, MODE_FAMILY: params})
            replacements[name] = ref(target)
        torch.cuda.synchronize()
        event(output, "RECOVERY_SITE_DONE" if valid else "RECOVERY_SITE_EXECUTION_NOT_READY", site=name,
              peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device))
        del panels, pair, site
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    status = "PASS" if replacements.keys() == selected.keys() else "EXECUTION_NOT_READY"
    write_json(output / "RECOVERY_MANIFEST.json", {
        "schema": "FFNDS_OBSERVED_FULL80_MODE_RECOVERY_V1", "status": status,
        "scope_sha256": sha_file(scope_path), "expert": expert, "sources": archive["cal_sources"],
        "selected_sites": list(selected), "sites": replacements, "site_receipts": receipts,
        "parent_cal_manifest": scope["parent_cal_manifest"], "parent_cal_exit": scope["parent_cal_exit"],
        "parent_cal_scope": scope["parent_cal_scope"], "runtime": allocation_record,
        "pcg_cap": PCG_CAP, "verified_relative_residual_tolerance": TOLERANCE, "nfe": 0, "d4": 0,
    })
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", type=Path)
    raise SystemExit(run(parser.parse_args().scope))
