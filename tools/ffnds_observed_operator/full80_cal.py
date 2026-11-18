"""Fit the remaining sites of one expert with the frozen CAL11 archive."""

import argparse
import json
import math
from pathlib import Path
from pickle import UnpicklingError

from safetensors import safe_open
import torch
from torch.nn.functional import gelu

from tools.ffnds_state_operator.p0_bindings import INPUT_HASHES, validate_panel

from . import cal_readout, mode_local
from .local_data import (
    ORIGINAL_BINDINGS, SHAPES, SITES, STEPS, SUFFIXES, bound_json, checked, equal_bits,
    fill_probes, load_pair, native_linear, native_runtime, require, sha_file, tensor_contract, tensor_sha,
)
from .local_worker import allocation, bind_fold_receipts, event, write_json


ALL_SITES = {(expert, layer) for expert in STEPS for layer in range(40)}
REF_KEYS = ("path", "sha256", "bytes")
WITNESSES = ("disabled_matches_static_bitwise", "static_reference_bitwise", "fixed_reference_bitwise",
             "zero_delta_correction_is_zero", "gate_witness_passed", "outputs_finite")


def ref(path):
    return {"path": str(path.absolute()), "sha256": sha_file(path), "bytes": path.stat().st_size}


def same_ref(left, right):
    return all(left[k] == right[k] for k in ("path", "sha256"))


def local6(scope, archive):
    exit_record = bound_json(scope["local6_exit"])
    root = Path(scope["local6_exit"]["path"]).parent
    require(root.is_absolute() and exit_record["scope_id"] == "FFNDS_OBSERVED_SECANT_LOCAL6_20261118_V2"
            and exit_record["slurm_job_id"] == "173607" and exit_record["exit_code"] == 0,
            "LOCAL6 is not the completed frozen run")
    hashes = exit_record["output_sha256"]
    for name, key in (("CAL_PARAMETERS", "local6_parameters"), ("BF16_ACCEPTANCE", "local6_acceptance")):
        path = f"output/{name}.json"
        require(Path(scope[key]["path"]) == root / path and scope[key]["sha256"] == hashes[path],
                f"LOCAL6 {name} differs from its EXIT record")
    params = bound_json(scope["local6_parameters"])
    acceptance = bound_json(scope["local6_acceptance"])["sites"]
    require(params["schema"] == "FFNDS_OBSERVED_SECANT_LOCAL_CAL_V1"
            and params["base_matrix_policy"] == "L64_AT_ALL_SIX_BOUND_SITES"
            and params["sources"] == archive["cal_sources"]
            and set(params["sites"]) == set(acceptance) == set(archive["sites"]), "LOCAL6 identity changed")
    reused = {}
    for key, entry in params["sites"].items():
        path = f"output/site_parameters/{key}.json"
        require(entry["path"] == path and entry["sha256"] == hashes[path], f"{key}: LOCAL6 parameter locator")
        locator = dict(entry, path=str(root / path))
        row = bound_json(locator)
        identity, original = row["identity"], archive["sites"][key]
        require(row["site"] == key and identity["old_params"] == original["old_params"]
                and same_ref(identity["strict_q1"], original["strict_q1"])
                and identity["ffn_tensor_sha256"] == original["ffn_tensor_sha256"]
                and all(acceptance[key][family]["valid"] and row[family]
                        for family in ("observed_readout", "native_modes")),
                f"{key}: LOCAL6 parameters or BF16 acceptance differ")
        json.dumps(row, allow_nan=False)
        reused[key] = locator
    return reused, acceptance


def inventory(scope, spec):
    require(scope["bindings"]["sha256"] == ORIGINAL_BINDINGS, "Wrong P0 generation")
    archive = bound_json(scope["bindings"])
    require(archive["schema"] == "FFNDS_STATE_OPERATOR_BINDINGS_V1" and archive["status"] == "PASS"
            and archive["l64_policy"] == "REUSE_ALL_SIX", "P0 did not bind existing L64")
    sources = archive["cal_sources"]
    require(sources == spec["cohorts"]["CAL"] and len(set(sources)) == len(sources) == 11, "CAL11 changed")
    require([row["timestep"] for row in archive["schedule"]] == [1000., 937.5, 833.3333129882812, 625.],
            "Frozen timestep schedule changed")
    refs = archive["input_manifests"]
    for key, scope_key in (("cal_pairs", "q1_cal_pairs"), ("strict_q1_manifest", "strict_q1")):
        require(same_ref(scope[scope_key], refs[key]) and refs[key]["sha256"] == INPUT_HASHES[key],
                f"Frozen {key} differs")
    asset = spec["assets"]
    root = Path(archive["bank_root"])
    require(root.is_absolute() and root == Path(asset["school_work_root"]) / asset["bank_relative"]
            and archive["bank_manifest"] == refs["bank_manifest"], "Frozen bank locator differs")
    checked(refs["capture_scope"])
    bank = bound_json(archive["bank_manifest"])
    require(all(bank[s]["cycle_role"] == "CAL" for s in sources)
            and all(len({bank[s][k] for s in sources}) == 11 for k in ("image_sha256", "source_cluster_id")),
            "CAL role or source-cluster identity differs")
    raw = bound_json(refs["cal_pairs"])
    pairs = {(r["source"], r["expert"], r["layer"]): r for r in raw["input_pairs"]}
    require(len(pairs) == len(raw["input_pairs"]) == asset["registered_full80_cal_pairs"] == 880
            and pairs.keys() == {(s, *site) for s in sources for site in ALL_SITES}
            and set(raw["sources"]) == set(sources)
            and raw["v2_bank_manifest_sha256"] == refs["bank_manifest"]["sha256"]
            and raw["bindings_sha256"] == refs["old_bindings"]["sha256"], "Raw CAL pair identity differs")
    strict = bound_json(refs["strict_q1_manifest"])
    registered = {(r["expert"], r["layer"]): r for r in archive["strict_q1_full80"]}
    actual = {(r["expert"], r["layer"]): r for r in strict["sites"]}
    require(strict["status"] in {"ALL_80_CONVERGED", "ALL_80_FINITE_FIXED_ITERATES"}
            and len(registered) == len(archive["strict_q1_full80"]) == len(actual) == len(strict["sites"])
            == strict["num_sites"] == 80 and actual.keys() == registered.keys() == ALL_SITES, "Strict Q1 coverage differs")
    operators = {}
    for (expert, layer), row in actual.items():
        artifact, prior = row["artifact"], registered[expert, layer]
        identity = artifact["identity"]
        require(Path(artifact["path"]).is_absolute() and same_ref(artifact, prior)
                and row["status"] == prior["status"]
                and (identity["expert"], identity["layer"], identity["checkpoint_sha256"]) ==
                (expert, layer, archive["checkpoints"][expert]["sha256"])
                and identity["calibration_sources"] == sources
                and identity["calibration_manifest_sha256"] == refs["cal_pairs"]["sha256"],
                f"{expert}{layer:02d}: strict Q1 identity differs")
        operators[expert, layer] = dict(prior, identity=identity)
    return archive, bank, root, pairs, operators


def bind_expert(expert, archive, bank, root, pairs, operators):
    frozen_six = {(r["source"], r["expert"], r["layer"]): r for r in archive["pairs"] if r["split"] == "CAL"}
    sources = archive["cal_sources"]
    require(frozen_six.keys() == {(s, *site) for s in sources for site in SITES}, "P0 CAL6 coverage differs")
    bound, issues, seen = {}, [], set()
    for layer in range(40):
        site = (expert, layer)
        for source in sources:
            key = (source, *site)
            pair = {"source": source, "expert": expert, "layer": layer, "split": "CAL"}
            for role, step in zip(("anchor", "hit"), STEPS[expert], strict=True):
                raw = pairs[key][role]
                try:
                    require(raw["kind"] == "v2_panel"
                            and (raw["source"], raw["seed"], raw["step"], raw["layer"]) == (source, 42, step, layer),
                            f"Unexpected CAL producer/role: {raw['kind']}")
                    frozen = archive["checkpoints"][expert]
                    require(all(raw["input_identity"][k] == bank[source][k]
                                for k in ("source_cluster_id", "image_sha256", "prompt_sha256"))
                            and raw["input_identity"]["checkpoint_sha256"] == frozen["sha256"], "Raw endpoint identity")
                    path = Path(raw["panel"]["path"])
                    expected = root / source / f"{source}__seed42__s{step:02d}__b{layer:02d}.pt"
                    require(path == expected and path not in seen, f"Unexpected or aliased locator: {path}")
                    for locator in (path, path.with_suffix(".pt.json")):
                        if not locator.is_file():
                            raise FileNotFoundError(locator)
                        require(locator.resolve().is_relative_to(root.resolve()), f"Endpoint escaped archived bank: {locator}")
                    seen.add(path)
                    endpoint = validate_panel(path, source, site, step, bank[source], frozen,
                                              archive["input_manifests"]["capture_scope"], raw)
                    endpoint["archived_reference_kind"] = raw["kind"]
                    require(endpoint["timestep"] == archive["schedule"][step]["timestep"], "Timestep differs")
                    if site in SITES:
                        require(endpoint == frozen_six[key][role], "CAL6 endpoint changed")
                    baseline = frozen_six[source, expert, 4]["anchor"]["token_sha256"]
                    require(endpoint["token_sha256"] == baseline, "Source token IDs or order differ")
                    pair[role] = endpoint
                except (OSError, EOFError, KeyError, TypeError, ValueError, RuntimeError, UnpicklingError) as exc:
                    status = "ASSET_NOT_AVAILABLE" if isinstance(exc, FileNotFoundError) else "INVALID_ASSET"
                    issues.append({**key_fields(key), "role": role, "step": step, "status": status, "detail": str(exc)})
            if "anchor" in pair and "hit" in pair:
                bound[key] = pair
        assets = {"strict_q1": operators[site]}
        if site in SITES:
            assets["bound_l64"] = archive["sites"][f"{expert}{layer:02d}"]["old_params"]
        for role, asset in assets.items():
            path = Path(asset["path"])
            try:
                require(path.is_absolute(), f"{role} locator is not absolute")
                checked(asset)
            except (OSError, ValueError) as exc:
                status = "ASSET_NOT_AVAILABLE" if not path.is_file() else "INVALID_ASSET"
                issues.append({"expert": expert, "layer": layer, "role": role, "path": str(path),
                               "status": status, "detail": str(exc)})
    return bound, issues


def key_fields(key):
    source, expert, layer = key
    return {"source": source, "expert": expert, "layer": layer}


def load_strict_site(expert, layer, operator, checkpoint, device, runtime):
    key = f"{expert}{layer:02d}"
    strict = torch.load(checked(operator), mmap=True, map_location="cpu", weights_only=True)
    identity = strict["identity"]
    require(dict(identity, calibration_sources=list(identity["calibration_sources"])) == operator["identity"],
            f"{key}: strict Q1 tensor identity differs")
    a32 = strict["A_q1_f32"]
    require(tensor_contract(a32, (5120, 5120), torch.float32) and bool(torch.isfinite(a32).all()), f"{key}: invalid Q1")
    a = a32.bfloat16()
    require(bool(torch.isfinite(a).all()), f"{key}: Q1 BF16 conversion overflow")
    prefix = f"blocks.{layer}.ffn"
    weights, hashes = [], {}
    with safe_open(checkpoint, framework="pt", device="cpu") as container:
        for suffix, shape in zip(SUFFIXES, SHAPES, strict=True):
            name = f"{prefix}.{suffix}"
            value = container.get_tensor(name)
            require(tensor_contract(value, shape, torch.bfloat16) and bool(torch.isfinite(value).all()),
                    f"{key}: invalid checkpoint tensor {name}")
            hashes[name] = tensor_sha(value)
            weights.append(value.to(device))
    w1, b1, w2, b2 = weights
    factory, post, host = runtime
    first = native_linear(w1, b1, f"{prefix}.0", factory)
    second = native_linear(w2, b2, f"{prefix}.2", factory)

    def observe(value):
        require(value.shape == (64, 5120) and value.dtype == torch.bfloat16 and value.device == device,
                f"{key}: observer input contract")
        out = second(gelu(first(value), approximate="tanh"))
        require(out.shape == value.shape and out.dtype == torch.bfloat16 and bool(torch.isfinite(out).all()),
                f"{key}: observer output contract")
        return out

    def gate_witness(value, pair):
        require(value.shape == pair["u1"].shape and value.dtype == torch.bfloat16, f"{key}: pre-gate output contract")
        residual, gate = pair["u1"], pair["gate_bf16"]
        return equal_bits(post(host, residual.clone(), value, gate), torch.add(residual, value * gate.squeeze()))

    return {"key": key, "A": a.to(device), "w1": w1, "b1": b1, "w2": w2, "b2": b2, "observe": observe,
            "gate_witness": gate_witness, "identity": {
                "base_role": "STRICT_Q1", "strict_q1": {k: operator[k] for k in REF_KEYS},
                "strict_bf16_tensor_sha256": tensor_sha(a), "ffn_tensor_sha256": hashes}}


def check_modes(site, cal, params):
    device = site["A"].device
    modes = torch.tensor(params["modes"], device=device, dtype=torch.float32)
    theta = torch.tensor(params["theta_static"], device=device, dtype=torch.float64)
    metric = torch.tensor(params["lambda_cal"], device=device, dtype=torch.float64)
    a, w1, w2 = (site[k] for k in ("A", "w1", "w2"))
    folds = []
    with mode_local.full_fp32():
        current = mode_local.compile_matrix(a, w1, w2, modes, theta)
        for panel in cal:
            u0, u1, y0, gate = (panel[k] for k in ("u0", "u1", "y0", "gate"))
            db = (u1.float() - u0.float()).bfloat16()
            static = mode_local.mode_output(db, y0, current)
            disabled, _ = mode_local._deployed_hit(site, db, y0, modes, theta, adaptation=False, frozen_matrix=current)
            zero = torch.zeros_like(db) @ current
            static_ref = mode_local._reference_output(site, u0, u1, y0, modes, theta)
            for fold, observed in zip(panel["folds"], panel["probe_y1"], strict=True):
                p = fold["probe"]
                pg, pr = mode_local._probe_moments(db[p], y0[p], observed, gate, site, modes)
                online, coefficients = mode_local._deployed_hit(
                    site, db, y0, modes, theta, adaptation=True, fit=(pg, pr, metric, params["ridge"]))
                online_ref = mode_local._reference_output(site, u0, u1, y0, modes, coefficients)
                folds.append({
                    "source": panel["source"], "fold": fold["fold"], "probe_ids": fold["probe_ids"],
                    "solve_condition": float(torch.linalg.cond(pg + params["ridge"] * metric)),
                    "disabled_matches_static_bitwise": equal_bits(static, disabled),
                    "static_reference_bitwise": equal_bits(static, static_ref),
                    "fixed_reference_bitwise": equal_bits(online, online_ref),
                    "zero_delta_correction_is_zero": not bool(zero.count_nonzero()),
                    "gate_witness_passed": site["gate_witness"](static, panel) and site["gate_witness"](online, panel),
                    "outputs_finite": bool(torch.isfinite(static).all() and torch.isfinite(online).all()),
                })
    return folds


def numerical_acceptance(folds):
    return len(folds) == 22 and all(
        all(row[k] is True for k in WITNESSES)
        and (row["solve_condition"] is None or math.isfinite(row["solve_condition"]))
        for row in folds)


def fit_site(site, panels):
    acceptance, tuning, parameters = {}, {}, {}
    for family in ("observed_readout", "native_modes"):
        try:
            if family == "observed_readout":
                fit = cal_readout.fit_cal(site, panels)
                folds = fit.get("folds", [])
                details = {}
            else:
                fit = mode_local.process_site(site, panels, [], structure=False)
                details = {key: fit.get(key) for key in
                           ("pcg", "spectrum", "tuning_mean_mse", "matrix_compile_cost")}
                details["cal_action_support"] = fit["acceptance"].get("cal_action_support")
                folds = check_modes(site, panels, fit["parameters"]) if fit["status"] == "OK" else []
            params = fit["parameters"]
            tuning[family] = {"status": fit["status"], "tuning": fit["tuning"], **details}
            acceptance[family] = {"status": fit["status"], "valid": bool(params) and numerical_acceptance(folds),
                                  "folds": folds}
            # The strict JSON encoder also checks finite tuning, parameters and diagnostics.
            json.dumps([params, tuning[family], acceptance[family]], allow_nan=False)
            parameters[family] = params
        except (ValueError, RuntimeError) as exc:
            acceptance[family] = {"status": "INVALID_NUMERICAL_EXECUTION", "valid": False, "detail": str(exc)}
            tuning[family] = {"status": "INVALID_NUMERICAL_EXECUTION"}
            parameters[family] = None
    return parameters, acceptance, tuning


@torch.inference_mode()
def run(scope_path):
    scope = json.loads(scope_path.read_bytes())
    expert = scope["expert"]
    require(scope["method"] == "OBSERVED_SECANT_FULL80_CAL" and expert in STEPS
            and scope.get("max_nfe", 0) == scope.get("max_d4", 0) == 0
            and Path(scope["stage"]) == Path.cwd(), "Unexpected CAL worker scope or working directory")
    spec = bound_json(scope["experiment_spec"])
    require(spec["execution_eligibility"]["local_metric_video_gate"] is False
            and spec["execution_eligibility"]["local_oracle_video_gate"] is False
            and spec["assets"]["new_dense_capture_in_budget"] is False,
            "Expected the amended CAL-only scope")
    device, allocation_record = allocation(scope)
    output = Path("output")
    output.mkdir()
    sites_dir = output / "site_parameters"
    sites_dir.mkdir()
    receipts_dir = output / "site_receipts"
    receipts_dir.mkdir()
    event(output, "BIND_CAL_ARCHIVE", expert=expert)
    archive, bank, root, raw_pairs, operators = inventory(scope, spec)
    reused, reused_acceptance = local6(scope, archive)
    bound, issues = bind_expert(expert, archive, bank, root, raw_pairs, operators)
    try:
        require(Path(archive["checkpoints"][expert]["path"]).is_absolute(), "Expert checkpoint locator is not absolute")
        checkpoint = checked(archive["checkpoints"][expert])
    except (OSError, ValueError) as exc:
        status = "ASSET_NOT_AVAILABLE" if not Path(archive["checkpoints"][expert]["path"]).is_file() else "INVALID_ASSET"
        issues.append({"expert": expert, "role": "checkpoint", "status": status, "detail": str(exc)})
    status = ("INVALID_ASSET" if any(row["status"] == "INVALID_ASSET" for row in issues)
              else "ASSET_NOT_AVAILABLE" if issues else "PASS")
    bindings = {
        "schema": "FFNDS_OBSERVED_SECANT_FULL80_CAL_BINDINGS_V1", "status": status, "expert": expert,
        "parent_bindings": scope["bindings"], "experiment_spec": scope["experiment_spec"],
        "scope_sha256": sha_file(scope_path), "cal_sources": archive["cal_sources"], "schedule": archive["schedule"],
        "input_manifests": archive["input_manifests"], "bank_root": str(root), "native_sources": scope["native_sources"],
        "checkpoints": archive["checkpoints"], "registered_full80_pairs": len(raw_pairs),
        "verified_expert_pairs": len(bound), "pairs": list(bound.values()), "issues": issues,
        "strict_q1": {f"{e}{layer:02d}": row for (e, layer), row in operators.items() if e == expert},
        "local6": {key: scope[f"local6_{key}"] for key in ("exit", "parameters", "acceptance")},
        "source_clusters": {s: bank[s]["source_cluster_id"] for s in archive["cal_sources"]},
        "nfe": 0, "d4": 0,
    }
    write_json(output / "BINDINGS.json", bindings)
    if status != "PASS":
        event(output, status, expert=expert, asset_issue_count=len(issues))
        return 1
    require(len(bound) == 440, "Incomplete expert CAL archive")
    runtime = native_runtime(scope, archive)
    require(torch.get_float32_matmul_precision() == "highest" and not torch.backends.cuda.matmul.allow_tf32
            and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, "Native import changed precision")
    parameter_refs = {key: row for key, row in reused.items() if key.startswith(expert)}
    acceptance = {key: dict(reused_acceptance[key], provenance="REUSE_LOCAL6") for key in parameter_refs}
    ids_by_source, tuning, receipt_refs = {}, {}, {}
    for layer in range(40):
        site_id = expert, layer
        if site_id in SITES:
            continue
        key = f"{expert}{layer:02d}"
        event(output, "FIT_SITE", site=key)
        site = load_strict_site(*site_id, operators[site_id], checkpoint, device, runtime)
        panels = []
        for source in archive["cal_sources"]:
            pair = load_pair(bound[source, *site_id], archive, bank, ids_by_source, device)
            fill_probes(site, pair)
            panels.append(pair)
        parameters, checks, tuning[key] = fit_site(site, panels)
        acceptance[key] = {**checks, "provenance": "FIT_CAL11"}
        valid = all(row["valid"] for row in checks.values())
        receipt = receipts_dir / f"{key}.json"
        write_json(receipt, {"site": key, "identity": site["identity"], "acceptance": acceptance[key],
                             "tuning": tuning[key], "valid": valid})
        receipt_refs[key] = ref(receipt)
        # Preserve successfully fitted parameters and execution receipts even if another fit failed.
        path = sites_dir / f"{key}.json"
        write_json(path, {"site": key, "identity": site["identity"], **parameters})
        parameter_refs[key] = ref(path)
        torch.cuda.synchronize()
        event(output, "SITE_DONE" if valid else "SITE_EXECUTION_NOT_READY", site=key,
              peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
              family_status={family: row["status"] for family, row in checks.items()})
        del panels, site, pair
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    valid = all(row[family]["valid"] for row in acceptance.values() for family in ("observed_readout", "native_modes"))
    status = "PASS" if valid else "EXECUTION_NOT_READY"
    write_json(output / "CAL_TUNING.json", tuning)
    write_json(output / "BF16_ACCEPTANCE.json",
               {"status": status, "runtime": allocation_record, "sites": acceptance,
                "fold_identities": bind_fold_receipts(ids_by_source), "nfe": 0, "d4": 0})
    write_json(output / "CAL_MANIFEST.json",
               {"schema": "FFNDS_OBSERVED_SECANT_FULL80_CAL_V1", "status": status, "expert": expert,
                "sites": parameter_refs, "site_receipts": receipt_refs, "sources": archive["cal_sources"],
                "site_count": len(parameter_refs), "new_site_count": len(tuning),
                "bindings": scope["bindings"], "cal_bindings": ref(output / "BINDINGS.json"),
                "q1_cal_pairs": scope["q1_cal_pairs"], "strict_q1": scope["strict_q1"],
                "experiment_spec": scope["experiment_spec"], "acceptance": ref(output / "BF16_ACCEPTANCE.json"),
                "tuning": ref(output / "CAL_TUNING.json"), "local_quality_gate": False, "nfe": 0, "d4": 0})
    event(output, "CAL_COMPLETE" if valid else status, expert=expert, nfe=0, d4=0)
    return 0 if valid else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", type=Path)
    raise SystemExit(run(parser.parse_args().scope))
