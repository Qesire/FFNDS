"""Fit all-position static C supports and centered gain/bias from calibration12."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from ar2_core import top_groups
from centered_signal_fit import fit_signal_moments
from generate_proxy_pr2 import contribution_scores
from run_ar2_restart import ROOT, tensor_sha


def load_training_reports(paths):
    records, bindings = {}, []
    common = None
    for path in paths:
        run = json.loads(path.read_text())
        if run["status"] != "PASS_SIX_INPUT_FULL160_OPERANDS_AND_NOOP":
            raise RuntimeError("All calibration collection shards must complete")
        identity = {k: v for k, v in run["identity"].items() if k != "shard"}
        if common is not None and common != identity:
            raise RuntimeError("Calibration shards use different identities")
        common = identity
        bindings.append({"run": str(path.resolve()), "sha256": digest(path), "inputs": run["completed"]})
        for row in run["completed"]:
            if row["index"] in records or digest(row["input_report"]) != row["input_report_sha256"]:
                raise RuntimeError("Duplicate/changed calibration input")
            record = json.loads(Path(row["input_report"]).read_text())
            if (record["status"] != "PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP"
                    or record["index"] != row["index"] or record["fingerprint"] != run["fingerprint"]
                    or record["trajectories"]["plain"] != record["trajectories"]["observed"]
                    or [r["step"] for r in record["trajectories"]["plain"]] != list(range(4))
                    or record["videos"]["plain"]["sha256"] != record["videos"]["observed"]["sha256"]):
                raise RuntimeError("Input lacks full no-op validation")
            if len(record["scopes"]) != 160 or {(s["step"],s["block"]) for s in record["scopes"]} != {(s,b) for s in range(4) for b in range(40)}:
                raise RuntimeError("Incomplete/duplicate calibration scopes")
            records[row["index"]] = record
    if set(records) != set(range(12)):
        raise RuntimeError("Exactly the original calibration12 required")
    return records, bindings, common


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collections", type=Path, nargs=2, required=True)
    ap.add_argument("--expert", choices=["high_noise_model", "low_noise_model"], required=True)
    ap.add_argument("--out", type=Path, required=True, help="Learned parameter checkpoints inside HF derived cache")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "JOB.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    records, bindings, capture_identity = load_training_reports(args.collections)
    calibration_path = ROOT / '.runtime/inputs/calibration12/canonical.json'
    if (capture_identity['calibration_sha256'] != digest(calibration_path)
            or capture_identity['protocol_sha256'] != digest(ROOT/'config/calibration_ffn_operands.json')):
        raise RuntimeError('Calibration inputs or strict-seed capture protocol differ')
    canonical = json.loads(calibration_path.read_text())
    if any(records[i]['input'] != canonical[i] for i in range(12)):
        raise RuntimeError('Training input metadata differs from canonical calibration12')
    protocol_path = ROOT / "config/cstatic_centered_fit.json"
    protocol = json.loads(protocol_path.read_text())
    cache_path = ROOT / ".runtime/hif4_rebuilt_cache.json"
    cache = json.loads(cache_path.read_text())
    if digest(cache_path) != capture_identity["assets"]["cache_manifest_sha256"]:
        raise RuntimeError("Training weight cache differs from captured operands")
    random_path = ROOT / "config/ar2_frozen_random_masks.json"
    random_groups = json.loads(random_path.read_text())["manifest"]["masks"][0]["kept_group_ids"]
    store = CheckpointStore(args.out, {"protocol": protocol, "protocol_sha256": digest(protocol_path),
        "training_collections": bindings, "expert": args.expert, "cache_manifest_sha256": digest(cache_path),
        "sources": {str(p): digest(p) for p in [Path(__file__), ROOT / "tools/centered_signal_fit.py", ROOT / "tools/generate_proxy_pr2.py"]},
        "random_mask_manifest_sha256": digest(random_path), "torch": torch.__version__, "TF32": False})
    meta = {"status": "RUNNING", "fingerprint": store.fingerprint, "expert": args.expert,
            "checkpoint_root": str(args.out.resolve()), "positions": [], "quality_or_speed_acceptance": False}
    started = time.perf_counter()
    atomic_json(args.report, meta)
    # Position fitting is below; keeping one record per position bounds memory.
    try:
        fit_positions(args, records, cache, protocol, random_groups, store, meta)
        meta.update(status="COMPLETED_CALIBRATION_PARAMETER_FIT", elapsed_seconds=time.perf_counter() - started)
    except BaseException as error:
        meta.update(status="FAILED_RESUMABLE", error=str(error))
        raise
    finally:
        atomic_json(args.report, meta)
        atomic_json(args.out / "run_meta.json", meta)
    print(json.dumps(meta, indent=2))


def fit_positions(args, records, cache, protocol, random_groups, store, meta):
    steps = [0, 1] if args.expert == "high_noise_model" else [2, 3]
    for block in range(40):
        shard = next(s for s in cache["shards"] if s["expert"] == args.expert and s["block"] == block)
        if digest(shard["path"]) != shard["sha256"]:
            raise RuntimeError("Effective weight shard changed")
        payload = torch.load(shard["path"], map_location="cpu", weights_only=True, mmap=True)
        w = payload[args.expert][f"blocks.{block}.ffn.2"]["weight_qdq"].cuda()
        wf = w.float()
        for step in steps:
            name = f"step{step}_block{block:02d}"
            previous = store.load("positions", name)
            if previous is not None:
                meta["positions"].append({"step": step, "block": block, **previous["metadata"]})
                continue
            entries, cmean, common_bias, common_gate = [], np.zeros(216), None, None
            den = 0.0
            for index in range(12):
                scope = next(s for s in records[index]["scopes"] if s["step"] == step and s["block"] == block)
                if scope["expert"] != args.expert or digest(scope["path"]) != scope["sha256"]:
                    raise RuntimeError("Training scope/weight provenance changed")
                if scope["rows"] != 512 or len(set(scope["selected_rows"])) != 512:
                    raise RuntimeError("Invalid actual uniform training rows")
                with safe_open(scope["path"], framework="pt", device="cpu") as handle:
                    a, y_native, b2, gate = [handle.get_tensor(n).cuda() for n in ["V2", "Y", "b2", "gate"]]
                b2, gate = b2.reshape(-1), gate.reshape(1, 5120)
                if a.dtype != torch.bfloat16 or a.shape != (512, 13824) or y_native.shape != (512, 5120):
                    raise RuntimeError("Training native operand geometry changed")
                if common_bias is not None and (tensor_sha(b2) != tensor_sha(common_bias) or tensor_sha(gate) != tensor_sha(common_gate)):
                    raise RuntimeError("Gate or original b2 varies across training inputs")
                common_bias, common_gate = b2, gate
                cmean += contribution_scores(a, w, gate).mean(0) / 12
                dense = F.linear(a.float(), wf, None)
                native_no_bias = F.linear(a, w, None) * gate
                den += float(native_no_bias.double().square().sum())
                entries.append({"a": a, "dense": dense, "target": y_native * gate})
            arrays = {"C_mean": cmean, "gate": common_gate.float().cpu().numpy(),
                      "original_b2": common_bias.float().cpu().numpy()}
            info = {"weight_sha256": tensor_sha(w), "original_b2_sha256": tensor_sha(common_bias),
                    "gate_sha256": tensor_sha(common_gate), "denominator_T": den, "policies": {}}
            for policy, groups in [("C", top_groups(cmean, 108)), ("R", np.asarray(random_groups))]:
                groups = np.sort(groups)
                channels = torch.tensor((groups[:, None] * 64 + np.arange(64)).reshape(-1), device="cuda")
                packed_w, packed_wf = w.index_select(1, channels).contiguous(), wf.index_select(1, channels).contiguous()
                sum_s, sum_y = torch.zeros(5120, dtype=torch.float64, device="cuda"), torch.zeros(5120, dtype=torch.float64, device="cuda")
                ss = sy = 0.0
                for entry in entries:
                    aa = entry["a"].index_select(1, channels)
                    signal = F.linear(aa.float(), packed_wf, None)
                    entry["sparse"] = aa
                    sum_s += signal.double().sum(0)
                    sum_y += entry["dense"].double().sum(0)
                    sg = (signal * common_gate.float()).double()
                    yg = (entry["dense"] * common_gate.float()).double()
                    ss += float(sg.square().sum())
                    sy += float((sg * yg).sum())
                fit = fit_signal_moments(12 * 512, sum_s.cpu().numpy(), sum_y.cpu().numpy(), ss, sy, common_gate.float().cpu().numpy())
                if fit["centered_alpha"] is None or fit["uncentered_alpha"] is None:
                    raise RuntimeError("Degenerate fixed calibration policy; no silent fallback")
                arrays[f"{policy}_groups"] = groups
                arrays[f"{policy}_mean_sparse"], arrays[f"{policy}_mean_dense"] = fit["mean_signal"], fit["mean_dense"]
                policy_info = {k: v for k, v in fit.items() if not isinstance(v, np.ndarray)}
                policy_info["modes"] = {}
                for mode in protocol["modes"]:
                    fit_alpha = fit["centered_alpha"] if mode == "centered_gain_bias" else fit["uncentered_alpha"] if mode == "uncentered_gain" else 1.0
                    alpha = float(np.float32(fit_alpha))
                    correction = fit["mean_dense"] - alpha * fit["mean_signal"] if mode in ["unit_mean", "centered_gain_bias"] else np.zeros(5120)
                    bias = (common_bias.double() + torch.as_tensor(correction, device="cuda")).to(torch.bfloat16)
                    arrays[f"{policy}_{mode}_b2"] = bias.float().cpu().numpy()
                    native_sse = fp32_sse = 0.0
                    for entry in entries:
                        effective = entry["sparse"].float() * alpha
                        native = F.linear(effective.to(torch.bfloat16), packed_w, bias) * common_gate
                        control = F.linear(effective, packed_wf, bias.float()) * common_gate.float()
                        native_sse += float((native.double() - entry["target"].double()).square().sum())
                        fp32_sse += float((control.double() - entry["target"].double()).square().sum())
                    policy_info["modes"][mode] = {"alpha": alpha, "alpha_fit_FP64": fit_alpha,
                        "b2_sha256": tensor_sha(bias), "native_training_risk": native_sse / den, "FP32_training_risk": fp32_sse / den}
                info["policies"][policy] = policy_info
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise RuntimeError("Nonfinite learned parameters")
            store.save("positions", name, arrays, info)
            meta["positions"].append({"step": step, "block": block, **info})
            atomic_json(args.report, meta)
            print(f"FITTED {args.expert} {name}: C centered train risk={info['policies']['C']['modes']['centered_gain_bias']['native_training_risk']:.6g}", flush=True)
        del payload, w, wf


if __name__ == "__main__":
    main()
