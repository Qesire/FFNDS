"""Evaluate pre-FFN0 low-rank proxies, choosing on Fit before held-out reporting.

PR2's original implementation and completed result are preserved at 918bad2.
PR3 requires separate own-donor training scores and opposite-donor test scores.
"""
import argparse
import csv
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from ar2_core import donor_key, fit_nonnegative_gain, input_norm_gain, support_mask, top_groups
from proxy_energy import corrected_scores, fit_energy_residual
from run_ar2_restart import ROOT, key_name, load_scopes


def verified_store(path, status):
    meta = json.loads((path / "run_meta.json").read_text())
    identity = json.loads((path / "identity.json").read_text())
    store = CheckpointStore(path, identity["identity"])
    if meta["status"] != status or meta["fingerprint"] != store.fingerprint:
        raise RuntimeError(f"Incomplete or inconsistent parent {path}")
    return store, meta


def gate(rows):
    passed = [row for row in rows if row["pass"]]
    return {"passed_scopes": len(passed), "gate_pass": len(passed) >= 3 and
            {r["seed"] for r in passed} == {20261001, 20261002} and {r["step"] for r in passed} == {0, 2}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrices", type=Path, nargs="+", required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--family", choices=["pr2", "pr3"], default="pr2")
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "JOB.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = ROOT / "config" / {"pr2": "proxy_pr2_w1_svd.json", "pr3": "proxy_pr3_input_pca.json"}[args.family]
    protocol = json.loads(protocol_path.read_text())
    parent, parent_meta = verified_store(args.parent, "COMPLETED_LOCAL_DIAGNOSTIC")
    if digest(args.parent / "scope_policy_summary.csv") != parent_meta["final_csv_sha256"]:
        raise RuntimeError("AR2 baseline CSV changed")
    generators, generation_meta = {}, {}
    for noise in ["high", "low"]:
        expert = noise + "_noise_model"
        generators[expert], generation_meta[expert] = verified_store(args.scores / noise, f"COMPLETED_{args.family.upper()}_SCORE_GENERATION")
        if generation_meta[expert]["protocol_sha256"] != digest(protocol_path):
            raise RuntimeError("Score protocol differs")
    scopes, inputs = load_scopes(args.matrices)
    for store in generators.values():
        if json.loads((store.root / "identity.json").read_text())["identity"]["inputs"] != inputs:
            raise RuntimeError("Generated scores belong to different inputs")
    store = CheckpointStore(args.out, {"protocol": protocol, "protocol_sha256": digest(protocol_path),
        "parent": parent.fingerprint, "generators": {k: v.fingerprint for k, v in generators.items()},
        "inputs": inputs, "sources": {str(p): digest(p) for p in [Path(__file__), ROOT / "tools/proxy_energy.py", ROOT / "tools/ar2_core.py"]},
        "torch": torch.__version__, "cpu_threads": 8})
    meta = {"status": "RUNNING", "experiment": protocol["experiment"], "fingerprint": store.fingerprint,
            "quality_or_speed_acceptance": False, "selection_uses_validation": False}
    atomic_json(args.out / "run_meta.json", meta)
    started = time.perf_counter()
    candidates = [(f"rank{r}_{m}", f"rank{r}", m) for r in protocol["ranks"] for m in protocol["modes"]]
    variants = candidates + [(c, c, "raw") for c in protocol["controls"]]
    baseline = list(csv.DictReader((args.parent / "scope_policy_summary.csv").open()))
    raw, training_raw, energies, labels, models, gains = {}, {}, {}, {}, {}, {}
    summary, decisions = [], []
    try:
        for key in scopes:
            for _, variant, _ in variants:
                part = generators[key.expert].load("scores", f"{key_name(key)}_{variant}")
                if part is None:
                    raise RuntimeError("Missing generated score checkpoint")
                raw[key, variant] = part["arrays"]["C"]
                if key.task == 0:
                    if args.family == "pr3" and variant.startswith("rank"):
                        if ("training_C" not in part["arrays"] or not part["metadata"]["training_uses_own_donor_basis"]
                                or part["metadata"]["donor"] != asdict(donor_key(key))):
                            raise RuntimeError("PR3 own-training/opposite-test PCA binding missing")
                    training_raw[key, variant] = part["arrays"].get("training_C", part["arrays"]["C"])
                energies[key] = part["arrays"]["input_energy"]
            if key.task == 0:
                labels[key] = np.concatenate([parent.load("scores", f"{key_name(key)}_row{start:04d}")["arrays"]["C"] for start in range(0, 512, 128)])
        for key, data in scopes.items():
            if key.task != 0:
                continue
            dense = data["projection"].fp32_signal(data["a"], data["gate"], np.ones(tuple(data["a"].shape), np.float32)).numpy()
            for name, source, mode in variants:
                model_key = f"{key_name(key)}_{name}"
                part = store.load("parameters", model_key)
                if part is None:
                    model = fit_energy_residual(training_raw[key, source], labels[key], energies[key], data["weights"]) if mode == "energy_residual" else {"beta": np.zeros(216), "mean_energy": 1.0}
                    scores = corrected_scores(training_raw[key, source], energies[key], model)
                    ids = top_groups(scores, 108)
                    mask = support_mask(ids, rows=512, groups=216, group_size=64)
                    signal = data["projection"].fp32_signal(data["a"], data["gate"], mask).numpy()
                    fit = fit_nonnegative_gain(signal, dense, data["weights"])
                    store.save("parameters", model_key, {"beta": model["beta"]},
                               {"mean_energy": model["mean_energy"], "gain": asdict(fit), "donor": asdict(key)})
                    part = store.load("parameters", model_key)
                models[key, name] = {"beta": part["arrays"]["beta"], "mean_energy": part["metadata"]["mean_energy"]}
                gains[key, name] = part["metadata"]["gain"]["alpha"]
            print(f"FITTED {key_name(key)}", flush=True)
        for key, data in scopes.items():
            donor, scope = donor_key(key), key_name(key)
            controls = {r["policy"]: float(r["native_normalized_SSE"]) for r in baseline if r["scope"] == scope and r["gain"] == "policy_crossfit"}
            den = float(next(r for r in baseline if r["scope"] == scope)["denominator_T"])
            for name, source, mode in variants:
                support_key = f"{scope}_{name}"
                saved = store.load("supports", support_key)
                if saved is None:
                    ids = top_groups(corrected_scores(raw[key, source], energies[key], models[donor, name]), 108)
                    store.save("supports", support_key, {"group_ids": ids}, {"donor": asdict(donor), "variant": name})
                    saved = store.load("supports", support_key)
                ids = saved["arrays"]["group_ids"]
                for gain_name in protocol["gains"]:
                    valid = gain_name != "policy_crossfit" or gains[donor, name] is not None
                    total_native = total_fp32 = 0.0
                    if valid:
                        for start in range(0, 512, 128):
                            risk_key = f"{support_key}_{gain_name}_row{start:04d}"
                            part = store.load("risks", risk_key)
                            if part is None:
                                aa = data["a"][start:start + 128]
                                mask = support_mask(ids[start:start + 128], rows=128, groups=216, group_size=64)
                                alpha = input_norm_gain(aa, mask) if gain_name == "input_norm" else gains[donor, name] if gain_name == "policy_crossfit" else 1.0
                                target = data["target"][start:start + 128].double()
                                native = data["projection"].native(aa, data["gate"], mask, alpha)
                                fp32 = data["projection"].fp32_candidate(aa, data["gate"], mask, alpha)
                                arrays = {"native_SSE": (native.double() - target).square().sum(1).numpy(),
                                          "fp32_SSE": (fp32.double() - target).square().sum(1).numpy(), "weights": data["weights"][start:start + 128]}
                                if any(not np.isfinite(value).all() for value in arrays.values()):
                                    raise FloatingPointError("Nonfinite proxy risk")
                                store.save("risks", risk_key, arrays, {"donor": asdict(donor), "gain": gain_name, "variant": name})
                                part = store.load("risks", risk_key)
                            values = part["arrays"]
                            total_native += float(values["weights"] @ values["native_SSE"])
                            total_fp32 += float(values["weights"] @ values["fp32_SSE"])
                    risk = total_native / den if valid else None
                    summary.append({"scope": scope, **asdict(key), "variant": name, "gain": gain_name,
                        "native_risk": risk, "fp32_risk": total_fp32 / den if valid else None,
                        "C_dynamic": controls["C_dynamic"], "C_static": controls["C_static"], "R_mean": controls["R_mean"]})
                    if gain_name == "policy_crossfit":
                        decisions.append({"scope": scope, **asdict(key), "variant": name,
                            "pass": bool(valid and risk <= .8 * controls["R_mean"] and risk <= .9 * controls["C_static"] and risk <= 1.1 * controls["C_dynamic"]),
                            "risk": risk, "vs_random": risk / controls["R_mean"] if valid else None,
                            "vs_C_static": risk / controls["C_static"] if valid else None,
                            "vs_true_C": risk / controls["C_dynamic"] if valid else None})
                print(f"EVALUATED {scope} {name}", flush=True)
        gates = {name: {str(task): gate([r for r in decisions if r["variant"] == name and r["task"] == task]) for task in [0, 6]} for name, _, _ in variants}
        chosen = next((name for name, _, _ in candidates if gates[name]["0"]["gate_pass"]), None)
        selection = {"chosen_variant_on_Fit": chosen, "Fit_pass": chosen is not None,
                     "Validation_pass": gates[chosen]["6"]["gate_pass"] if chosen else None,
                     "route_pass": bool(chosen and gates[chosen]["6"]["gate_pass"]),
                     "other_Validation_results": "predeclared sensitivity only, not used for selection", "all_gates": gates}
        atomic_json(args.out / "decisions.json", decisions)
        atomic_json(args.out / "selection.json", selection)
        with (args.out / "scope_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        if torch.cuda.is_initialized():
            raise RuntimeError("Projection evaluator unexpectedly initialized CUDA")
        meta.update(status=f"COMPLETED_{args.family.upper()}_DIAGNOSTIC", elapsed_seconds=time.perf_counter() - started,
                    selection=selection, final_csv_sha256=digest(args.out / "scope_summary.csv"))
    except BaseException as error:
        meta.update(status="FAILED_RESUMABLE", error=str(error))
        raise
    finally:
        atomic_json(args.out / "run_meta.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
