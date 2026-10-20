"""Probe C-support prediction from V1, with no test post-FFN operands exposed.

Kernel hyperparameters use donor training folds only. Test V2/Delta are held by
the separate projection scorer. This is a row-wise CPU diagnostic, not a sparse
CUDA deployment or tile-sharing result.
"""
import argparse
from dataclasses import asdict
import csv
import fcntl
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from ar2_core import donor_key, fit_nonnegative_gain, input_norm_gain, support_mask, top_groups
from proxy_kernel import predict, select_training_only
from run_ar2_restart import ROOT, key_name, load_scopes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrices", type=Path, nargs="+", required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "JOB.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent_meta = json.loads((args.parent / "run_meta.json").read_text())
    if parent_meta["status"] != "COMPLETED_LOCAL_DIAGNOSTIC" or digest(args.parent / "scope_policy_summary.csv") != parent_meta["final_csv_sha256"]:
        raise RuntimeError("Verified complete parent AR2 result required")
    parent_identity = json.loads((args.parent / "identity.json").read_text())
    parent = CheckpointStore(args.parent, parent_identity["identity"])
    if parent.fingerprint != parent_meta["fingerprint"]:
        raise RuntimeError("Parent checkpoint identity mismatch")
    protocol_path = ROOT / "config/proxy_pr1_v1_kernel.json"
    protocol = json.loads(protocol_path.read_text())
    scopes, inputs = load_scopes(args.matrices)
    store = CheckpointStore(args.out, {"protocol": protocol, "protocol_sha256": digest(protocol_path),
        "parent_fingerprint": parent.fingerprint, "inputs": inputs,
        "source_sha256": {str(p): digest(p) for p in [Path(__file__), ROOT / "tools/proxy_kernel.py", ROOT / "tools/ar2_core.py"]},
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "numpy"]}, "cpu_threads": 8})
    report = args.out / "run_meta.json"
    meta = {"status": "RUNNING", "experiment": protocol["experiment"], "fingerprint": store.fingerprint,
            "independent_sources": 2, "selector_input": "V1 only, before FFN0", "quality_or_speed_acceptance": False}
    atomic_json(report, meta)
    started = time.perf_counter()
    try:
        features, labels, models, gains = {}, {}, {}, {}
        for key, data in scopes.items():
            with safe_open(data["scope"]["tensor_file"], framework="pt", device="cpu") as handle:
                v1 = handle.get_tensor("V1")
            if v1.dtype != torch.bfloat16 or v1.shape != (512, 5120) or not bool(torch.isfinite(v1).all()):
                raise RuntimeError("Actual finite BF16 V1 features required")
            features[key] = v1.float().numpy()
            if key.task == 0:
                labels[key] = np.concatenate([parent.load("scores", f"{key_name(key)}_row{start:04d}")["arrays"]["C"] for start in range(0, 512, 128)])
        for key, data in scopes.items():
            if key.task != 0:
                continue
            name = key_name(key)
            saved = store.load("models", name)
            if saved is None:
                times = np.asarray([coord[0] for coord in data["scope"]["latent_t_h_w"]])
                model, cv = select_training_only(features[key], labels[key], times, protocol["regularization_grid"])
                arrays = {name: value for name, value in model.items() if isinstance(value, np.ndarray)}
                metadata = {name: value for name, value in model.items() if not isinstance(value, np.ndarray)}
                metadata.update(donor=asdict(key), cv=cv)
                store.save("models", name, arrays, metadata)
                saved = store.load("models", name)
            model = {**saved["arrays"], **saved["metadata"]}
            models[key] = model
            train_ids = top_groups(predict(model, features[key]), 108)
            channel_mask = support_mask(train_ids, rows=512, groups=216, group_size=64)
            signal = data["projection"].fp32_signal(data["a"], data["gate"], channel_mask).numpy()
            dense = data["projection"].fp32_signal(data["a"], data["gate"], np.ones(tuple(data["a"].shape), np.float32)).numpy()
            fit = fit_nonnegative_gain(signal, dense, data["weights"])
            gains[key] = fit
            print(f"FITTED {name}: {model['kind']} lambda={model['regularization']} alpha={fit.alpha}", flush=True)
        baseline = list(csv.DictReader((args.parent / "scope_policy_summary.csv").open()))
        summary, decisions = [], []
        for key, data in scopes.items():
            donor = donor_key(key)
            selection_key = key_name(key)
            part = store.load("supports", selection_key)
            if part is None:
                ids = top_groups(predict(models[donor], features[key]), 108)
                store.save("supports", selection_key, {"group_ids": ids}, {"donor": asdict(donor), "model_kind": models[donor]["kind"]})
                part = store.load("supports", selection_key)
            ids = part["arrays"]["group_ids"]
            controls = {r["policy"]: float(r["native_normalized_SSE"]) for r in baseline if r["scope"] == selection_key and r["gain"] == "policy_crossfit"}
            reference = next(r for r in baseline if r["scope"] == selection_key)
            den = float(reference["denominator_T"])
            for gain_name in protocol["gains"]:
                valid = not (gain_name == "policy_crossfit" and gains[donor].alpha is None)
                total_native = total_fp32 = 0.0
                if valid:
                    for start in range(0, 512, 128):
                        name = f"{selection_key}_{gain_name}_row{start:04d}"
                        part = store.load("risks", name)
                        if part is None:
                            aa, gate = data["a"][start:start + 128], data["gate"]
                            mask = support_mask(ids[start:start + 128], rows=128, groups=216, group_size=64)
                            gain = input_norm_gain(aa, mask) if gain_name == "input_norm" else gains[donor].alpha if gain_name == "policy_crossfit" else 1.0
                            target = data["target"][start:start + 128].double()
                            native = data["projection"].native(aa, gate, mask, gain)
                            fp32 = data["projection"].fp32_candidate(aa, gate, mask, gain)
                            arrays = {"native_SSE": (native.double() - target).square().sum(1).numpy(),
                                      "fp32_SSE": (fp32.double() - target).square().sum(1).numpy(),
                                      "weights": data["weights"][start:start + 128]}
                            if any(not np.isfinite(value).all() for value in arrays.values()):
                                raise FloatingPointError("Non-finite proxy risk")
                            store.save("risks", name, arrays, {"donor": asdict(donor), "gain": gain_name})
                            part = store.load("risks", name)
                        values = part["arrays"]
                        total_native += float(values["weights"] @ values["native_SSE"])
                        total_fp32 += float(values["weights"] @ values["fp32_SSE"])
                risk = total_native / den if valid else None
                summary.append({"scope": selection_key, **asdict(key), "gain": gain_name,
                    "model_kind": models[donor]["kind"], "model_lambda": models[donor]["regularization"],
                    "native_risk": risk, "fp32_risk": total_fp32 / den if valid else None,
                    "C_dynamic": controls["C_dynamic"], "C_static": controls["C_static"], "R_mean": controls["R_mean"]})
                if gain_name == "policy_crossfit":
                    decisions.append({"scope": selection_key, **asdict(key), "pass": bool(valid and risk <= .8 * controls["R_mean"] and risk <= .9 * controls["C_static"] and risk <= 1.1 * controls["C_dynamic"]),
                        "risk": risk, "vs_random": risk / controls["R_mean"] if valid else None,
                        "vs_C_static": risk / controls["C_static"] if valid else None,
                        "vs_true_C": risk / controls["C_dynamic"] if valid else None})
            print(f"EVALUATED {selection_key}", flush=True)
        for task in [0, 6]:
            passed = [row for row in decisions if row["task"] == task and row["pass"]]
            decisions.append({"task": task, "decision": "proxy_gate", "passed_scopes": len(passed),
                "gate_pass": len(passed) >= 3 and {row["seed"] for row in passed} == {20261001, 20261002} and {row["step"] for row in passed} == {0, 2}})
        atomic_json(args.out / "decisions.json", decisions)
        with (args.out / "scope_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU proxy diagnostic unexpectedly initialized CUDA")
        meta.update(status="COMPLETED_PROXY_DIAGNOSTIC", elapsed_seconds=time.perf_counter() - started,
                    source_gates={str(r["task"]): r["gate_pass"] for r in decisions if r.get("decision") == "proxy_gate"})
    except BaseException as error:
        meta.update(status="FAILED_RESUMABLE", error=str(error))
        raise
    finally:
        atomic_json(report, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
