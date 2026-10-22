"""CPU-only R12 E0 audit of existing operands; never runs a model or quantizer.

Historical full restorations are analyzed separately from the unobserved
equal-budget R12 actions. Public locators contain no workstation/worker names.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
INDICES = (0, 5, 10, 15)
FLOW_SHAPE = [16, 21, 58, 104]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locator(path):
    """Keep original private paths only in their already existing manifests."""
    p = Path(path)
    if "ffnds-derived" in p.parts:
        return "HF_DERIVED/" + str(Path(*p.parts[p.parts.index("ffnds-derived") + 1:]))
    return str(p.resolve().relative_to(ROOT))


def verified_safetensors(path, expected_sha, expected_tensors=None):
    """Read original bytes and validate header geometry without a GPU library."""
    p = Path(path)
    if sha(p) != expected_sha:
        raise ValueError("Saved operand SHA mismatch: " + locator(p))
    with p.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        if size > min(p.stat().st_size - 8, 1 << 24):
            raise ValueError("Invalid safetensors header")
        header = json.loads(f.read(size))
        payload_start = 8 + size
        result = {}
        for key, item in header.items():
            if key == "__metadata__":
                continue
            start, end = item["data_offsets"]
            width = {"BF16": 2, "F32": 4, "I64": 8}.get(item["dtype"])
            if (width is None or start < 0 or end < start
                    or end - start != math.prod(item["shape"]) * width
                    or payload_start + end > p.stat().st_size):
                raise ValueError("Invalid tensor byte geometry")
            f.seek(payload_start + start)
            h, remaining = hashlib.sha256(), end - start
            while remaining:
                chunk = f.read(min(8 << 20, remaining))
                if not chunk:
                    raise ValueError("Truncated saved operand")
                h.update(chunk)
                remaining -= len(chunk)
            result[key] = {"shape": item["shape"], "dtype": item["dtype"], "sha256": h.hexdigest()}
    if expected_tensors is not None:
        dtype_names = {"torch.bfloat16": "BF16", "torch.float32": "F32", "torch.int64": "I64"}
        expected_tensors = {k: {**v, "dtype": dtype_names.get(v["dtype"], v["dtype"])}
                            for k, v in expected_tensors.items()}
        if result != expected_tensors:
            raise ValueError("Decoded saved operand differs from its original record")
    return {"locator": locator(p), "sha256": expected_sha, "bytes": p.stat().st_size,
            "tensors": result}, header, payload_start


def historical_headroom(rows):
    """Descriptive R7 restore-only arithmetic, NOT values for R12 actions."""
    totals = {a: sum(r["flow_SSE"][a] for r in rows) for a in ("A", "B")}
    fixed = min(totals, key=totals.get)
    oracle = sum(min(r["flow_SSE"][a] for a in totals) for r in rows)
    loo = []
    for row in rows:
        chosen = min(totals, key=lambda a: totals[a] - row["flow_SSE"][a])
        loo.append({"index": row["index"], "chosen_on_other_three": chosen,
                    "loss": row["flow_SSE"][chosen]})
    return {
        "scope": "R7 full-restoration arms only; NOT R12 equal-budget action values or a bound on them",
        "best_fixed_restore_arm": fixed, "total_loss_by_restore_arm": totals,
        "restore_oracle_total_loss": oracle,
        "restore_oracle_gain_over_best_fixed_fraction": 1 - oracle / totals[fixed],
        "leave_one_source_out_static_selection": loo,
        "leave_one_out_total_loss": sum(r["loss"] for r in loo),
        "pair_main_groups": {"none": 324, "A": 378, "B": 378, "AB": 432},
        "global_main_MAC_fractions": {"none": .5, "A": .5 + 54 / (160 * 216),
                                      "B": .5 + 54 / (160 * 216), "AB": .5 + 108 / (160 * 216)},
        "matched_R12_actions_observed": 0,
        "compensation_caveat": "R7 computes UB1 plus an offline full teacher and replaces Y; it does not execute matched packed R12 actions or their charged corrections.",
    }


def audit_tf2():
    observations, cases = [], {}
    for arm in ("DQ", "UB1"):
        for index in INDICES:
            report = ROOT / f"reports/worker_20261020/TF2_{arm}_input{index:02d}.json"
            case = read(report)
            cases[arm, index] = case
            for step, row in enumerate(case["steps"]):
                expected = {k: {"shape": FLOW_SHAPE, "dtype": "F32", "sha256": row[k + "_sha256"]}
                            for k in ("input", "flow", "post")}
                item, _, _ = verified_safetensors(row["path"], row["sha256"], expected)
                item.update(family="TF2", arm=arm, index=index, step=step,
                            source_record=str(report.relative_to(ROOT)), source_record_sha256=sha(report))
                observations.append(item)
    for index in INDICES:
        for step in range(4):
            if cases["DQ", index]["steps"][step]["input_sha256"] != cases["UB1", index]["steps"][step]["input_sha256"]:
                raise ValueError("TF2 full-step inputs are not identical")
    return observations, cases


def audit_op2():
    observations, sources, gate_hashes, row_pairs = [], [], {30: set(), 31: set()}, []
    for shard in (0, 1):
        report = ROOT / f"reports/worker_20261020/OP2_UB1_calibration24_shard{shard}_RUN.json"
        for entry in read(report)["completed"]:
            p = Path(entry["input_report"])
            if sha(p) != entry["input_report_sha256"]:
                raise ValueError("Original OP2 input receipt changed")
            case = read(p)
            pair = {s["block"]: set(s["selected_rows"]) for s in case["scopes"]
                    if s["step"] == 3 and s["block"] in (30, 31)}
            row_pairs.append({"index": entry["index"], "common_native_row_ids": len(pair[30] & pair[31])})
            sources.append({"index": entry["index"], "image_sha256": case["input"]["input_provenance"]["image_sha256"],
                            "seed": case["input"]["seed"]})
            for row in case["scopes"]:
                if row["step"] != 3 or row["block"] not in (30, 31):
                    continue
                item, _, _ = verified_safetensors(row["path"], row["sha256"])
                if set(item["tensors"]) != {"V1", "BaseY", "StudentY", "TeacherY", "b2", "gate"}:
                    raise ValueError("Unexpected OP2 sampled operand set")
                for key in ("V1", "BaseY", "StudentY", "TeacherY"):
                    if item["tensors"][key]["shape"] != [256, 5120] or item["tensors"][key]["dtype"] != "BF16":
                        raise ValueError("OP2 saved rows are not the original 256 rows")
                if (row["full_rows"] != 31668 or row["rows"] != 256
                        or item["tensors"]["gate"]["sha256"] != row["gate_sha256"]):
                    raise ValueError("OP2 full/saved row or gate identity changed")
                gate_hashes[row["block"]].add(row["gate_sha256"])
                item.update(family="OP2", index=entry["index"], step=3, block=row["block"],
                            full_rows_seen_by_producer=31668, saved_rows=256,
                            selected_rows_sha256=hashlib.sha256(json.dumps(row["selected_rows"], separators=(",", ":")).encode()).hexdigest(),
                            input_receipt_sha256=entry["input_report_sha256"],
                            trajectory="free UB1; different sources/seeds/state construction from TF2")
                observations.append(item)
    if len(sources) != 24 or len(observations) != 48:
        raise ValueError("Incomplete scoped OP2 coverage")
    return observations, sources, {str(k): sorted(v) for k, v in gate_hashes.items()}, row_pairs


def audit_r7(hf_root, tf2):
    import numpy as np

    path = ROOT / "research/20261021/R7D1_results.json"
    saved = read(path)
    observations, rows = [], []
    for row in saved["rows"]:
        flows = {}
        for arm, binding in row["payloads"].items():
            p = hf_root / binding["HF_derived_relative"]
            expected = {k: {"shape": FLOW_SHAPE, "dtype": "F32", "sha256": binding[k + "_sha256"]}
                        for k in ("flow", "post")}
            item, header, start = verified_safetensors(p, binding["sha256"], expected)
            item.update(family="R7D1", arm=arm, index=row["index"], step=3)
            observations.append(item)
            flows[arm] = np.memmap(p, dtype="<f4", mode="r", offset=start + header["flow"]["data_offsets"][0],
                                  shape=tuple(FLOW_SHAPE)).astype(np.float64)
        if any(not np.isfinite(v).all() for v in flows.values()):
            raise ValueError("Nonfinite historical flow")
        e = flows["none"] - flows["DQ"]
        errors = {a: float(np.square(v - flows["DQ"]).sum()) for a, v in flows.items()}
        for a, sse in errors.items():
            recorded = 0. if a == "DQ" else row["flow_SSE"][a]
            if not math.isclose(sse, recorded, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("Historical R7 flow arithmetic changed")
        if not np.array_equal(flows["none"], flows["none_repeat"]):
            raise ValueError("Repeated native control is not identical")
        if row["payloads"]["none"]["flow_sha256"] != tf2["UB1", row["index"]]["steps"][3]["flow_sha256"]:
            raise ValueError("Historical R7 baseline differs from TF2")
        directions = {}
        for arm in ("A", "B"):
            delta = flows[arm] - flows["none"]
            norm2 = float(np.square(delta).sum())
            alignment_gain = float(-2 * (e * delta).sum())
            gain = errors["none"] - errors[arm]
            if not math.isclose(gain, alignment_gain - norm2, rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("Native signed gain identity failed")
            directions[arm] = {"alignment_term": alignment_gain, "change_norm2": norm2, "net_gain": gain,
                               "cosine_to_baseline_error": -alignment_gain / (2 * math.sqrt(errors["none"] * norm2))}
        scopes = {a: {s["block"]: s for s in ss} for a, ss in row["scopes"].items()}
        rows.append({"index": row["index"], "flow_SSE": errors, "signed_terms": directions,
                     "block30_input_same_across_restore_arms": len({scopes[a][30]["norm_input_sha256"] for a in ("none", "A", "B", "AB")}) == 1,
                     "block31_input_changes_after_A": scopes["none"][31]["norm_input_sha256"] != scopes["A"][31]["norm_input_sha256"],
                     "gates": {str(b): scopes["none"][b]["gate_sha256"] for b in (30, 31)},
                     "local_error_winner": row["local_larger_error"],
                     "downstream_restore_winner": row["downstream_larger_benefit"]})
    gates = {str(b): sorted({r["gates"][str(b)] for r in rows}) for b in (30, 31)}
    return observations, rows, {"source_record": str(path.relative_to(ROOT)), "source_record_sha256": sha(path),
                                **historical_headroom(rows), "gate_hashes_by_block": gates}


def audit_r11(run):
    saved = read(ROOT / "research/20261021/R11T1_matched_objective_results.json")
    wanted = {r["index"]: r for r in saved["captured_payloads"]}
    observations = []
    for shard in (0, 1):
        receipt = run / f"collect{shard}/RUN.json"
        if sha(receipt) != saved["raw_job_receipts_sha256"][f"collect{shard}"]:
            raise ValueError("R11 original collection receipt differs")
        for row in read(receipt)["completed"]:
            item, _, _ = verified_safetensors(row["path"], wanted[row["index"]]["sha256"], wanted[row["index"]]["tensors"])
            item.update(family="R11T1", index=row["index"], step=3, block=39,
                        source_receipt_locator=locator(receipt), source_receipt_sha256=sha(receipt),
                        pair_30_31_reusable=False)
            observations.append(item)
    return observations


def audit_weights():
    import numpy as np
    import torch

    manifest = read(ROOT / ".runtime/hif4_rebuilt_cache.json")
    policy = read(ROOT / "reports/worker_20261020/TF2_UB1_input00.json")["runtime"]["packed_reference"]["policy"]
    parent_binding = policy["parent_policy"]
    if sha(parent_binding["path"]) != parent_binding["sha256"]:
        raise ValueError("Original parent policy binding changed")
    parent = read(parent_binding["path"])

    def support(binding, block, field, count):
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Original support fit binding changed")
        fit = read(binding["path"])
        root = Path(fit["checkpoint_root"])
        sidecar = root / f"positions/step3_block{block:02d}.json"
        payload = sidecar.with_suffix(".npz")
        metadata = read(sidecar)
        if (metadata["status"] != "COMPLETE" or metadata["fingerprint"] != fit["fingerprint"]
                or sha(payload) != metadata["payload_sha256"]):
            raise ValueError("Original support checkpoint changed")
        with np.load(payload, allow_pickle=False) as arrays:
            groups = arrays[field].astype(np.int64)
        groups.sort()
        if (groups.shape != (count,) or len(np.unique(groups)) != count
                or int(groups.min()) < 0 or int(groups.max()) >= 216):
            raise ValueError("Invalid original whole64 support")
        return torch.from_numpy(groups), {"fit_report_sha256": binding["sha256"],
                                          "checkpoint_payload_sha256": metadata["payload_sha256"],
                                          "checkpoint_sidecar_sha256": sha(sidecar), "field": field}

    def tensor_hash(tensor):
        return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()

    observed = []
    for block in (30, 31):
        shard = next(s for s in manifest["shards"] if s["expert"] == "low_noise_model" and s["block"] == block)
        if sha(shard["path"]) != shard["sha256"]:
            raise ValueError("Original quantized weight shard changed")
        payload = torch.load(shard["path"], map_location="cpu", weights_only=True, mmap=True)["low_noise_model"]
        geometry = {}
        for slot, shape in ((0, [13824, 5120]), (2, [5120, 13824])):
            record = payload[f"blocks.{block}.ffn.{slot}"]
            if list(record["weight_qdq"].shape) != shape or record["weight_qdq"].dtype != torch.bfloat16:
                raise ValueError("Original effective weight geometry differs")
            geometry[str(slot)] = {"weight_qdq": shape, "smooth_scale": list(record["smooth_scale"].shape)}
        s108, b108 = support(parent["fit_reports"]["low_noise_model"], block, parent["support"] + "_groups", 108)
        s162, b162 = support(policy["changed_fits"]["3"], block, "groups", 162)
        variants = {}
        for count, groups in ((108, s108), (162, s162), (216, torch.arange(216))):
            channels = (groups[:, None] * 64 + torch.arange(64)).reshape(-1)
            packed = {}
            for slot in (0, 2):
                record = payload[f"blocks.{block}.ffn.{slot}"]
                weight = record["weight_qdq"].index_select(0 if slot == 0 else 1, channels).contiguous()
                scale = record["smooth_scale"] if slot == 0 else record["smooth_scale"].index_select(0, channels).contiguous()
                if count == 216 and not torch.equal(weight, record["weight_qdq"]):
                    raise ValueError("Full support packing changes original weights")
                packed[str(slot)] = {"weight_shape": list(weight.shape), "weight_sha256": tensor_hash(weight),
                                     "smooth_scale_shape": list(scale.shape), "smooth_scale_sha256": tensor_hash(scale)}
                del weight, scale
            variants[str(count)] = {"groups": groups.tolist(), "packed_effective_records": packed,
                                    "support_binding": b108 if count == 108 else b162 if count == 162 else "original full support"}
        observed.append({"family": "ORIGINAL_DQ_WEIGHTS", "block": block,
                         "locator": f"private restored-cache manifest: low_noise_model/block{block}",
                         "sha256": shard["sha256"], "tensors": geometry,
                         "CPU_pack_variants": variants, "S108_is_subset_of_S162": set(s108.tolist()).issubset(s162.tolist()),
                         "pack_status": "CPU_EFFECTIVE_WEIGHT_SLICES_ONLY_NO_ACTIVATION_QDQ_OR_NATIVE_FORWARD"})
        del payload
    if torch.cuda.is_initialized():
        raise ValueError("CPU audit must not initialize CUDA")
    return observed


def source_locators():
    paths = ["tools/step_budget_runtime.py", "tools/trained_packed_runtime.py", "tools/cross_layer_transport.py",
             "tools/run_cross_layer_transport.py", "tools/native_teacher_capture.py", "tools/native_head_replay.py",
             ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py",
             ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py",
             ".runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py",
             ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py"]
    return {p: sha(ROOT / p) for p in paths}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r11-run", type=Path, default=ROOT / "runs/r11t1_matched_attempt2")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from worker_identity import require_worker
    require_worker()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run through worker_cpu_exec.sh with no visible GPUs")
    if args.out.exists():
        raise ValueError("Refusing to overwrite a completed audit")
    started = time.monotonic()
    tf2_obs, tf2 = audit_tf2()
    op2_obs, calibration, op2_gates, row_pairs = audit_op2()
    tf_path = Path(tf2["DQ", 0]["steps"][3]["path"])
    hf = Path(*tf_path.parts[:tf_path.parts.index("ffnds-derived") + 1])
    r7_obs, signed, r7 = audit_r7(hf, tf2)
    r11_obs = audit_r11(args.r11_run)
    weights = audit_weights()
    overlap = sorted({s["image_sha256"] for s in calibration} &
                     {tf2["DQ", i]["input"]["input_provenance"]["image_sha256"] for i in INDICES})
    result = {"id": "R12_E0_SOURCE_OPERAND_AUDIT", "status": "PASS_CPU_SAVED_OPERAND_AUDIT_NO_MODEL_RUN",
              "audit_source_sha256": sha(Path(__file__)), "source_sha256": source_locators(),
              "evidence_level": "STRUCTURAL_AND_HISTORICAL_SAVED_FLOW_REANALYSIS_ONLY",
              "new_native_generations": 0, "CUDA_initialized": False,
              "eligible_model_method_ids": [], "observations": tf2_obs + op2_obs + r7_obs + r11_obs + weights,
              "counts": {"TF2_full_step_payloads": len(tf2_obs), "OP2_sampled_pair_payloads": len(op2_obs),
                         "R7_full_flow_payloads": len(r7_obs), "R11_wrong_position_full_payloads": len(r11_obs),
                         "DQ_effective_pair_weight_shards": len(weights), "R12_counterfactual_action_flows": 0},
              "historical_restore_analysis": r7, "historical_signed_flow_rows": signed,
              "pre_W1_information_audit": {"OP2_gate_unique_hashes_by_block": op2_gates,
                                          "R7_gate_unique_hashes_by_block": r7["gate_hashes_by_block"],
                                          "OP2_V1_distinct_hashes_by_block": {str(b): len({o["tensors"]["V1"]["sha256"] for o in op2_obs if o["block"] == b}) for b in (30, 31)},
                                          "OP2_cross_layer_sampled_row_overlap": row_pairs,
                                          "constant_gate_and_step_only_policy_is_fixed_on_this_scope": all(len(x) == 1 for x in op2_gates.values()) and all(len(x) == 1 for x in r7["gate_hashes_by_block"].values()),
                                          "block31_input_is_not_available_at_block30_decision": all(r["block31_input_changes_after_A"] for r in signed)},
              "source_identity_audit": {"OP2_calibration": calibration, "TF2_indices": list(INDICES),
                                        "image_sha_overlap_OP2_TF2": overlap,
                                        "all_sources_previously_exposed": True,
                                        "same_image_would_still_not_establish_same_seed_or_trajectory": True},
              "missing": ["Full modulated/post-Q1 block30 input on TF2 sources (R7 kept hashes only)",
                          "Action-specific propagated block31 inputs and 108-group outputs",
                          "Matched correction fits and native numeric validation of the CPU-verified packs",
                          "Equal-budget native A_216_108 and B_108_216 complete flows",
                          "Fresh source predictor/generalization, full-FFN latency and free-student quality"],
              "resources": {"node_cache_free_bytes_after": shutil.disk_usage(hf).free,
                            "raw_tensor_bytes_written": 0, "elapsed_seconds": time.monotonic() - started}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "historical_restore_oracle_gain_fraction": r7["restore_oracle_gain_over_best_fixed_fraction"],
                      "constant_gate_only_policy": result["pre_W1_information_audit"]["constant_gate_and_step_only_policy_is_fixed_on_this_scope"],
                      "elapsed_seconds": result["resources"]["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
