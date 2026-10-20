"""Checkpointed CPU AR2 diagnostic on the fresh, no-op-verified D-Q matrix.

Training is restricted to task0 opposite-seed donors. Selectors never receive a
test target. This measures local post-Q2 projection risk, not CUDA sparsity,
closed-loop video quality, or reproduction of the missing old R1 operands.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from ar2_core import (NativeProjection, ScopeKey, design_weights, donor_key,
                      fit_nonnegative_gain, input_norm_gain, score_families,
                      static_groups, support_mask, top_groups, weighted_quantile)

ROOT = Path(__file__).resolve().parents[1]
POLICIES = [f"{family}_{mode}" for family in "AEC" for mode in ["static", "dynamic"]] + ["R"]
STRATEGIES = POLICIES[:-1] + [f"R_mask{i}" for i in range(4)]
GAINS = ["unit", "input_norm", "policy_crossfit"]


def key_name(key):
    return f"task{key.task}_seed{key.seed}_step{key.step}_block{key.block}"


def tensor_sha(tensor):
    return hashlib.sha256(memoryview(tensor.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def frozen_masks():
    obj = json.loads((ROOT / "config/ar2_frozen_random_masks.json").read_text())["manifest"]
    result = {}
    for index, salt in enumerate(["20261020", "20261021"]):
        order = sorted(range(216), key=lambda gid: (hashlib.sha256(f"AR1_POSTQ2_R1|{salt}|{gid}".encode()).hexdigest(), gid))
        for half in range(2):
            ids = sorted(order[half * 108:(half + 1) * 108])
            saved = obj["masks"][index * 2 + half]
            bits_sha = hashlib.sha256(bytes(int(i in ids) for i in range(216))).hexdigest()
            if saved["kept_group_ids"] != ids or saved["mask_bits_sha256"] != bits_sha:
                raise RuntimeError("Frozen AR1 random mask differs from its original SHA256 construction")
            result[saved["mask_id"]] = np.asarray(ids, dtype=np.int64)
    return result


def load_scopes(matrix_paths):
    inputs, scopes, identities, weight_cache = {}, {}, {}, {}
    for matrix_path in matrix_paths:
        matrix = json.loads(matrix_path.read_text())
        if matrix["status"] != "COMPLETE_DATA_CAPTURE_AND_NOOP" or matrix["arm"] != "D-Q":
            raise RuntimeError(f"Matrix is not complete: {matrix_path}")
        for record in matrix["records"]:
            pair = (record["task"], record["seed"])
            if pair in inputs or pair[0] not in [0, 6] or pair[1] not in [20261001, 20261002]:
                raise RuntimeError("Duplicate/unregistered matrix input")
            root = Path(record["observed"])
            meta = json.loads((root / "run_meta.json").read_text())
            if digest(record["input_manifest"]) != record["input_sha256"] or json.loads(Path(record["input_manifest"]).read_text()) != [meta["input_record"]]:
                raise RuntimeError("Frozen research input record changed")
            cap_path = root / "ffn/CAPTURE_MANIFEST.json"
            if digest(cap_path) != record["capture_manifest_sha256"] or digest(record["observer_noop"]) != record["observer_noop_sha256"]:
                raise RuntimeError("Matrix capture/no-op evidence changed")
            noop = json.loads(Path(record["observer_noop"]).read_text())
            cap = json.loads(cap_path.read_text())
            if (meta["status"] != "PASS" or meta["arm"] != "D-Q" or cap["status"] != "PASS_CAPTURE_STRUCTURE_ONLY"
                    or noop["status"] != "PASS_OBSERVER_NOOP_THIS_INPUT" or noop["arm"] != "D-Q"
                    or noop["observed"] != str(root.resolve()) or noop["seed"] != pair[1]
                    or noop["observer_source_sha256"] != meta["observer_source_sha256"]
                    or noop["hif4_observer_source_sha256"] != meta["hif4_observer_source_sha256"]
                    or not all(noop["identity_checks"].values())
                    or [row["step"] for row in noop["checks"]] != [0, 1, 2, 3]
                    or not all(row["bitwise_equal"] and row["max_abs_error"] == 0 for row in noop["checks"])):
                raise RuntimeError("Every input must pass independent native observer no-op")
            source = "validation_000_multihuman_5" if pair[0] == 0 else "validation_003_singleobj_29"
            role = "fit" if pair[0] == 0 else "validation"
            if meta["input_record"]["task_id"] != pair[0] or meta["source_id"] != source or meta["input_record"]["split"] != role:
                raise RuntimeError("Task source/split differs from frozen comparison")
            if noop["source_id"] != source:
                raise RuntimeError("No-op evidence belongs to another source")
            identity = {name: meta[name] for name in ["quantized_cache_identity", "checkpoint_sha256", "versions",
                                                     "observer_source_sha256", "hif4_observer_source_sha256", "contract", "resolved_contract"]}
            if identities and identity != next(iter(identities.values())):
                raise RuntimeError("Matrix inputs use different model/cache/observer environments")
            identities[pair] = identity
            inputs[pair] = {"matrix": str(matrix_path), "matrix_sha256": digest(matrix_path),
                            "run_meta_sha256": digest(root / "run_meta.json"), **record}
            cache = json.loads(Path(meta["quantized_cache_identity"]["manifest_path"]).read_text())
            if digest(meta["quantized_cache_identity"]["manifest_path"]) != meta["quantized_cache_identity"]["manifest_sha256"] or cache["sha256"] != meta["quantized_cache_identity"]["sha256"]:
                raise RuntimeError("Quantized cache manifest changed")
            selected = [s for s in cap["scopes"] if s["block"] == 20 and s["step"] in [0, 2]]
            if len(selected) != 2:
                raise RuntimeError("Expected block20 steps0/2 per input")
            for scope in selected:
                key = ScopeKey(pair[0], pair[1], scope["step"], scope["expert"], 20)
                donor_key(key)  # Validate exact step/expert/block and seed binding.
                if key in scopes or scope["row_count"] != 512 or scope["coordinate_provenance"] != "OBSERVED_NEW_WORKER_RUNTIME_GRID_AND_SELECTED_ROWS":
                    raise RuntimeError("Expected exactly 512 newly observed uniform rows per scope")
                if digest(scope["tensor_file"]) != scope["tensor_sha256"] or not all(scope["checks"].values()):
                    raise RuntimeError("Capture tensor/boundary check changed")
                if (len(set(scope["selected_rows"])) != 512 or len(scope["latent_t_h_w"]) != 512
                        or len(scope["inclusion_probabilities"]) != 512 or len(scope["grid"]) != 3
                        or any(size <= 0 for size in scope["grid"]) or scope["post_process_branch"] != "infer_add_"):
                    raise RuntimeError("Invalid captured row/coordinate map")
                count = int(np.prod(scope["grid"]))
                for row, coord, probability in zip(scope["selected_rows"], scope["latent_t_h_w"], scope["inclusion_probabilities"]):
                    if not 0 <= row < count or coord != [row // (scope["grid"][1] * scope["grid"][2]), row // scope["grid"][2] % scope["grid"][1], row % scope["grid"][2]] or probability != 512 / count:
                        raise RuntimeError("Coordinate/inclusion probability mismatch")
                weights = design_weights(scope["design_weights"], 512)
                if not np.array_equal(weights, np.full(512, count / 512, dtype=np.float64)):
                    raise RuntimeError("Design weights differ from inverse inclusion probabilities")
                with safe_open(scope["tensor_file"], framework="pt", device="cpu") as handle:
                    tensors = {name: handle.get_tensor(name) for name in ["V2", "gate_broadcast", "Delta_writeback_operand", "b2"]}
                aa, gate, target, bias = [tensors[name] for name in ["V2", "gate_broadcast", "Delta_writeback_operand", "b2"]]
                if aa.shape != (512, 13824) or gate.numel() != 5120 or target.shape != (512, 5120):
                    raise RuntimeError("Unexpected native operands")
                if any(t.dtype != torch.bfloat16 or not bool(torch.isfinite(t).all()) for t in tensors.values()):
                    raise RuntimeError("Native V2/gate/Delta/bias must be finite BF16")
                bias = None if bias.numel() == 0 else bias.reshape(5120)
                weight_key = (key.expert, 20)
                if weight_key not in weight_cache:
                    shard = next(row for row in cache["shards"] if (row["expert"], row["block"]) == weight_key)
                    if digest(shard["path"]) != shard["sha256"]:
                        raise RuntimeError("Effective weight shard changed")
                    payload = torch.load(shard["path"], map_location="cpu", weights_only=True, mmap=True)
                    w = payload[key.expert]["blocks.20.ffn.2"]["weight_qdq"]
                    if w.dtype != torch.bfloat16 or w.shape != (5120, 13824):
                        raise RuntimeError("Unexpected actual effective W2 layout")
                    weight_cache[weight_key] = (NativeProjection(w.T, bias), tensor_sha(w))
                projection, wsha = weight_cache[weight_key]
                if wsha != scope["effective_W2"]["runtime_operand_sha256"] or (bias is None) != (projection.bias is None) or (bias is not None and not torch.equal(bias, projection.bias)):
                    raise RuntimeError("Actual captured effective W2/bias differ across inputs")
                scopes[key] = {"key": key, "a": aa, "gate": gate.reshape(1, 5120), "target": target,
                               "weights": weights, "projection": projection, "scope": scope}
    expected = {ScopeKey(task, seed, step, "high_noise_model" if step == 0 else "low_noise_model")
                for task in [0, 6] for seed in [20261001, 20261002] for step in [0, 2]}
    if set(scopes) != expected:
        raise RuntimeError("Complete fresh eight-scope matrix is required")
    return scopes, list(inputs.values())


def average_ranks(values):
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.repeat((starts + ends - 1) / 2, ends - starts)
    return ranks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrices", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, default=ROOT / "config/ar2_restart_protocol.json")
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if [protocol[name] for name in ["group_size", "groups", "keep_groups", "row_chunk_size", "cpu_threads"]] != [64, 216, 108, 128, 8]:
        raise RuntimeError("Unsupported frozen diagnostic geometry/resources")
    if [protocol[name] for name in ["dynamic_vs_random_max_ratio", "dynamic_vs_static_max_ratio", "C_vs_E_max_ratio", "E_vs_A_max_ratio"]] != [.8, .9, .9, .9]:
        raise RuntimeError("Frozen scientific gates changed")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "JOB.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    scopes, inputs = load_scopes(args.matrices)
    identity = {"protocol": protocol, "protocol_sha256": digest(args.protocol), "inputs": inputs,
                "source_sha256": {str(p): digest(p) for p in [Path(__file__), ROOT / "tools/ar2_core.py", ROOT / "tools/ar2_checkpoint.py"]},
                "random_masks_sha256": digest(ROOT / "config/ar2_frozen_random_masks.json"),
                "versions": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "safetensors"]},
                "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor()),
                "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                "machine": platform.machine(), "host": platform.node(), "threads": 8}
    store = CheckpointStore(args.out, identity)
    random = frozen_masks()
    started = time.perf_counter()
    scores, norms, parameters = {}, {}, {}
    status = {"status": "RUNNING", "experiment": protocol["experiment"], "fingerprint": store.fingerprint,
              "scope_count": 8, "rows_per_scope": 512, "independent_source_count": 2, "gpu_count": 0}
    report = args.out / "run_meta.json"
    atomic_json(report, status)

    def mask(ids, rows):
        return support_mask(ids, rows=rows, groups=216, group_size=64)

    try:
        for key, data in scopes.items():
            parts = []
            for start in range(0, 512, 128):
                part_key = f"{key_name(key)}_row{start:04d}"
                part = store.load("scores", part_key)
                if part is None:
                    t0 = time.perf_counter()
                    aa, gate, projection = data["a"][start:start + 128], data["gate"], data["projection"]
                    values = score_families(aa, projection.weight_fp32, gate)
                    dense = projection.native(aa, gate, np.ones(tuple(aa.shape), np.float32), 1.0)
                    no_bias = F.linear(aa, projection.weight_t, None) * gate
                    values["T"] = no_bias.double().square().sum(1).numpy()
                    values["dense_cpu_gpu_SSE"] = (dense.double() - data["target"][start:start + 128].double()).square().sum(1).numpy()
                    store.save("scores", part_key, values, {"start": start, "rows": 128, "compute_seconds": time.perf_counter() - t0})
                    part = store.load("scores", part_key)
                parts.append(part["arrays"])
            scores[key] = {family: np.concatenate([p[family] for p in parts]) for family in "AEC"}
            trows = np.concatenate([p["T"] for p in parts])
            den = float(data["weights"] @ trows)
            if not np.isfinite(den) or den <= 0:
                raise RuntimeError("Nonpositive native dense normalization")
            drift = float(data["weights"] @ np.concatenate([p["dense_cpu_gpu_SSE"] for p in parts]) / den)
            norms[key] = {"denominator_T": den, "dense_cpu_gpu_normalized_SSE": drift}
            print(f"SCORED {key_name(key)}; CPU/GPU dense drift={drift:.6g}", flush=True)

        def policy_ids(key, policy, training=False):
            if policy.startswith("R_"):
                return random[policy[2:]]
            family, mode = policy.split("_")
            if mode == "dynamic":
                return top_groups(scores[key][family], 108)
            donor = key if training else donor_key(key)
            return static_groups(scores[donor][family], scopes[donor]["weights"], 108)

        # Fit each distinct task0 training donor/policy once; no test target is passed here.
        for key, data in scopes.items():
            if key.task != 0:
                continue
            full_signal = data["projection"].fp32_signal(data["a"], data["gate"], np.ones(tuple(data["a"].shape), np.float32)).numpy()
            for policy in POLICIES:
                name = f"{key_name(key)}_{policy}"
                part = store.load("gains", name)
                if part is None:
                    strategies = [f"R_{mid}" for mid in random] if policy == "R" else [policy]
                    numerator = denominator = 0.0
                    for strategy in strategies:
                        signal = data["projection"].fp32_signal(data["a"], data["gate"], mask(policy_ids(key, strategy, True), 512)).numpy()
                        fit = fit_nonnegative_gain(signal, full_signal, data["weights"])
                        numerator += fit.numerator
                        denominator += fit.denominator
                    alpha = max(0.0, numerator / denominator) if denominator > 5120 * 1e-24 else None
                    store.save("gains", name, {}, {"donor": asdict(key), "policy": policy, "numerator": numerator,
                        "denominator": denominator, "alpha": alpha, "status": "OK" if alpha is not None else "NA_DENOM_FLOOR"})
                    part = store.load("gains", name)
                parameters[(key, policy)] = part["metadata"]
        summaries, geometry = [], []
        for key, data in scopes.items():
            for strategy in STRATEGIES:
                support_key = f"{key_name(key)}_{strategy}"
                support = store.load("supports", support_key)
                if support is None:
                    store.save("supports", support_key, {"group_ids": policy_ids(key, strategy)}, {"donor": asdict(donor_key(key)), "strategy": strategy})
                    support = store.load("supports", support_key)
                ids = support["arrays"]["group_ids"]
                for gain_name in GAINS:
                    parameter = parameters[(donor_key(key), "R" if strategy.startswith("R_") else strategy)]
                    total_native = total_fp32 = 0.0
                    valid = not (gain_name == "policy_crossfit" and parameter["alpha"] is None)
                    if valid:
                        for start in range(0, 512, 128):
                            part_key = f"{support_key}_{gain_name}_row{start:04d}"
                            part = store.load("risks", part_key)
                            if part is None:
                                aa, gate = data["a"][start:start + 128], data["gate"]
                                selected = ids if ids.ndim == 1 else ids[start:start + 128]
                                channel_mask = mask(selected, 128)
                                gain = (input_norm_gain(aa, channel_mask) if gain_name == "input_norm" else
                                        parameter["alpha"] if gain_name == "policy_crossfit" else 1.0)
                                native = data["projection"].native(aa, gate, channel_mask, gain)
                                fp32 = data["projection"].fp32_candidate(aa, gate, channel_mask, gain)
                                target = data["target"][start:start + 128].double()
                                arrays = {"SSE_native": (native.double() - target).square().sum(1).numpy(),
                                    "SSE_fp32": (fp32.double() - target).square().sum(1).numpy(),
                                    "design_weights": data["weights"][start:start + 128],
                                    "gain_fp32": np.broadcast_to(np.asarray(gain, np.float32), (128,)).copy(),
                                    "token_rows": np.asarray(data["scope"]["selected_rows"][start:start + 128], np.int64)}
                                if any(not np.isfinite(array).all() for array in arrays.values()):
                                    raise FloatingPointError("Non-finite native/control risk cannot pass a gate")
                                store.save("risks", part_key, arrays, {"support_key": support_key, "gain": gain_name,
                                           "gain_parameter_donor": asdict(donor_key(key)), "start": start, "rows": 128})
                                part = store.load("risks", part_key)
                            values = part["arrays"]
                            total_native += float(values["design_weights"] @ values["SSE_native"])
                            total_fp32 += float(values["design_weights"] @ values["SSE_fp32"])
                    summaries.append({"scope": key_name(key), **asdict(key), "policy": strategy, "gain": gain_name,
                        "native_normalized_SSE": total_native / norms[key]["denominator_T"] if valid else None,
                        "fp32_normalized_SSE": total_fp32 / norms[key]["denominator_T"] if valid else None,
                        "valid": valid, **norms[key]})
                print(f"RISK {key_name(key)} {strategy}: all controls checkpointed", flush=True)
            for gain in GAINS:
                rows = [r for r in summaries if r["scope"] == key_name(key) and r["gain"] == gain and r["policy"].startswith("R_")]
                valid = len(rows) == 4 and all(r["valid"] for r in rows)
                summaries.append({"scope": key_name(key), **asdict(key), "policy": "R_mean", "gain": gain,
                    "native_normalized_SSE": float(np.mean([r["native_normalized_SSE"] for r in rows])) if valid else None,
                    "fp32_normalized_SSE": float(np.mean([r["fp32_normalized_SSE"] for r in rows])) if valid else None,
                    "valid": valid, **norms[key]})
            rho = []
            cc, ee = top_groups(scores[key]["C"], 108), top_groups(scores[key]["E"], 108)
            overlap = np.array([len(np.intersect1d(c, e)) / 108 for c, e in zip(cc, ee)])
            for c, e in zip(scores[key]["C"], scores[key]["E"]):
                c, e = average_ranks(c), average_ranks(e)
                c, e = c - c.mean(), e - e.mean()
                den = np.linalg.norm(c) * np.linalg.norm(e)
                rho.append(float(c @ e / den) if den > 0 else np.nan)
            rho = np.asarray(rho)
            good = np.isfinite(rho)
            weights = data["weights"]
            geometry.append({"scope": key_name(key), "rho_defined_design_mass": float(weights[good].sum() / weights.sum()),
                "weighted_spearman_C_E": float(weights[good] @ rho[good] / weights[good].sum()) if good.any() else None,
                "rho_q10_median_inverse_CDF": weighted_quantile(rho[good], weights[good], [.1, .5]).tolist() if good.any() else None,
                "weighted_top108_overlap": float(weights @ overlap / weights.sum()),
                "overlap_q10_median_inverse_CDF": weighted_quantile(overlap, weights, [.1, .5]).tolist()})
        atomic_json(args.out / "score_geometry.json", geometry)
        with (args.out / "scope_policy_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        primary = {(r["scope"], r["policy"]): r["native_normalized_SSE"] for r in summaries if r["gain"] == "policy_crossfit"}
        decisions = []
        for task in [0, 6]:
            keys = [key for key in scopes if key.task == task]

            def enough(passed):
                return len(passed) >= 3 and {k.seed for k in passed} == {20261001, 20261002} and {k.step for k in passed} == {0, 2}

            for family in "AEC":
                passed = []
                for key in keys:
                    sk = key_name(key)
                    dynamic, static, random_risk = [primary[(sk, policy)] for policy in [f"{family}_dynamic", f"{family}_static", "R_mean"]]
                    valid = all(value is not None for value in [dynamic, static, random_risk])
                    ok = valid and dynamic <= .8 * random_risk and dynamic <= .9 * static
                    if ok:
                        passed.append(key)
                    decisions.append({"task": task, "family": family, "scope": sk, "dynamic": dynamic,
                                      "static": static, "random_mean": random_risk, "joint_pass": bool(ok)})
                decisions.append({"task": task, "family": family, "decision": "dynamic_support_value_gate",
                                  "passed_scopes": len(passed), "gate_pass": enough(passed)})
            for candidate, reference, label in [("C", "E", "C_vs_E_additional_gate"), ("E", "A", "E_vs_A_auxiliary")]:
                passed = []
                for key in keys:
                    x, y = primary[(key_name(key), f"{candidate}_dynamic")], primary[(key_name(key), f"{reference}_dynamic")]
                    ok = x is not None and y is not None and x <= .9 * y
                    if ok:
                        passed.append(key)
                    decisions.append({"task": task, "scope": key_name(key), "decision": label, "candidate": x,
                                      "reference": y, "pass": bool(ok)})
                decisions.append({"task": task, "decision": label, "passed_scopes": len(passed), "coverage_pass": enough(passed)})
        atomic_json(args.out / "decisions.json", decisions)
        both_sources = {}
        for family in "AEC":
            gates = [row for row in decisions if row.get("family") == family and row.get("decision") == "dynamic_support_value_gate"]
            both_sources[family] = len(gates) == 2 and all(row["gate_pass"] for row in gates)
        c_gates = [row for row in decisions if row.get("decision") == "C_vs_E_additional_gate" and "coverage_pass" in row]
        both_sources["C"] = both_sources["C"] and len(c_gates) == 2 and all(row["coverage_pass"] for row in c_gates)
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU diagnostic unexpectedly initialized CUDA")
        status.update(status="COMPLETED_LOCAL_DIAGNOSTIC", elapsed_seconds_this_resume=time.perf_counter() - started,
                      completed_utc=datetime.now(timezone.utc).isoformat(), precision="native CPU BF16 primary; matched FP32 controls",
                      local_family_gate_both_sources=both_sources,
                      quality_or_speed_acceptance=False, final_csv_sha256=digest(args.out / "scope_policy_summary.csv"))
    except BaseException as error:
        status.update(status="FAILED_RESUMABLE", error=str(error))
        raise
    finally:
        atomic_json(report, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
