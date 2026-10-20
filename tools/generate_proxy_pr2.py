"""Generate pre-FFN0 weight-SVD proxy scores, keeping test operands diagnostic-only."""
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
from run_ar2_restart import ROOT, key_name, load_scopes, tensor_sha


@torch.no_grad()
def contribution_scores(activation, weight, gate):
    """Same FP32 GEMM/FP64 reduction as AR2, with bounded batched outputs."""
    aa = activation.float().reshape(-1, 216, 64).permute(1, 0, 2).contiguous()
    ww = weight.float().T.reshape(216, 64, 5120).contiguous()
    result = []
    for start in range(0, 216, 8):
        signal = torch.bmm(aa[start:start + 8], ww[start:start + 8])
        result.append((signal * gate.float()).double().square().sum(-1).T.cpu().numpy())
    return np.concatenate(result, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrices", type=Path, nargs="+", required=True)
    ap.add_argument("--expert", choices=["high_noise_model", "low_noise_model"], required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "JOB.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = ROOT / "config/proxy_pr2_w1_svd.json"
    protocol = json.loads(protocol_path.read_text())
    scopes, inputs = load_scopes(args.matrices)
    cache_path = ROOT / ".runtime/hif4_rebuilt_cache.json"
    cache = json.loads(cache_path.read_text())
    shard = next(s for s in cache["shards"] if s["expert"] == args.expert and s["block"] == 20)
    if digest(shard["path"]) != shard["sha256"]:
        raise RuntimeError("W1/W2 shard checksum mismatch")
    payload = torch.load(shard["path"], map_location="cpu", weights_only=True, mmap=True)[args.expert]
    w1 = payload["blocks.20.ffn.0"]["weight_qdq"].cuda()
    w2 = payload["blocks.20.ffn.2"]["weight_qdq"].cuda()
    scale = payload["blocks.20.ffn.2"]["smooth_scale"].cuda()
    mirror = json.loads((ROOT / ".runtime/node_local_model_view.json").read_text())
    source = next(row for row in mirror["files"] if f"/{args.expert}/" in row["view_path"] and row["filename"].endswith(".safetensors"))
    if digest(source["view_path"]) != source["sha256"]:
        raise RuntimeError("Raw distilled checkpoint checksum mismatch")
    with safe_open(source["view_path"], framework="pt", device="cpu") as handle:
        bias = handle.get_tensor("blocks.20.ffn.0.bias").cuda()
    store = CheckpointStore(args.out, {"protocol": protocol, "protocol_sha256": digest(protocol_path),
        "source_sha256": digest(Path(__file__)), "inputs": inputs, "expert": args.expert,
        "W1_sha256": tensor_sha(w1), "W2_sha256": tensor_sha(w2), "b1_sha256": tensor_sha(bias),
        "s2_sha256": tensor_sha(scale), "raw_checkpoint_sha256": source["sha256"],
        "cache_manifest_sha256": digest(cache_path), "torch": torch.__version__,
        "qdq_source_sha256": digest(ROOT / ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py"),
        "device": torch.cuda.get_device_name(), "TF32": False})
    parent_id = json.loads((args.parent / "identity.json").read_text())
    parent = CheckpointStore(args.parent, parent_id["identity"])
    factor_root = Path(shard["path"]).parent / "pr2_svd_factors"
    factor_root.mkdir(exist_ok=True)
    factor_path = factor_root / f"{args.expert}_{store.fingerprint}.pt"
    meta = {"status": "RUNNING", "fingerprint": store.fingerprint, "expert": args.expert,
            "protocol_sha256": digest(protocol_path), "quality_or_speed_acceptance": False, "cases": []}
    started = time.perf_counter()
    atomic_json(args.out / "run_meta.json", meta)
    try:
        if factor_path.exists():
            factors = torch.load(factor_path, map_location="cuda", weights_only=True)
            if factors["fingerprint"] != store.fingerprint:
                raise RuntimeError("Factor identity differs")
            u, s, v = [factors[k] for k in ["u", "s", "v"]]
        else:
            torch.manual_seed(20261020)
            u, s, v = torch.svd_lowrank(w1.float(), q=544, niter=3)
            temp = Path(str(factor_path) + ".tmp")
            torch.save({"u": u.cpu(), "s": s.cpu(), "v": v.cpu(), "fingerprint": store.fingerprint}, temp)
            temp.replace(factor_path)
        meta["factor_path"] = str(factor_path)
        meta["factor_sha256"] = digest(factor_path)
        total_energy = w1.double().square().sum()
        meta["weight_energy_retained"] = {str(r): float(s[:r].double().square().sum() / total_energy) for r in protocol["ranks"]}
        meta["proxy_FFN0_GEMM_flop_fraction"] = {str(r): r * (5120 + 13824) / (5120 * 13824) for r in protocol["ranks"]}
        print(f"FACTORIZED {args.expert}: {meta['weight_energy_retained']}", flush=True)
        for key, data in scopes.items():
            if key.expert != args.expert:
                continue
            name = key_name(key)
            with safe_open(data["scope"]["tensor_file"], framework="pt", device="cpu") as handle:
                operands = {k: handle.get_tensor(k).cuda() for k in ["V1", "Z", "V2"]}
            x, actual_z, actual_a = [operands[k] for k in ["V1", "Z", "V2"]]
            gate = data["gate"].cuda()
            labels = np.concatenate([parent.load("scores", f"{name}_row{start:04d}")["arrays"]["C"] for start in range(0, 512, 128)])
            for variant in ["captured_V2", "dense_ffn0_oracle"] + [f"rank{r}" for r in protocol["ranks"]]:
                part_key = f"{name}_{variant}"
                if store.load("scores", part_key) is not None:
                    continue
                if variant == "captured_V2":
                    aa, z_error = actual_a, None
                else:
                    if variant == "dense_ffn0_oracle":
                        z = F.linear(x, w1, bias)
                    else:
                        rank = int(variant[4:])
                        z = ((x.float() @ v[:, :rank]) * s[:rank]) @ u[:, :rank].T + bias.float()
                        z = z.to(torch.bfloat16)
                    z_error = float((z.double() - actual_z.double()).norm() / actual_z.double().norm())
                    h = F.gelu(z, approximate="tanh")
                    aa = qdq((h.float() / scale).to(torch.bfloat16), force_fp32=True)
                score = contribution_scores(aa, w2, gate)
                relative = float(np.linalg.norm(score - labels) / np.linalg.norm(labels))
                info = {"variant": variant, "Z_relative_L2_diagnostic": z_error,
                        "C_relative_L2_diagnostic": relative, "pre_FFN0_predictor_inputs_only": variant.startswith("rank"),
                        "oracle_control_excluded_from_selection": not variant.startswith("rank")}
                if variant == "captured_V2" and relative > 2e-6:
                    raise RuntimeError(f"GPU/CPU C score mismatch {relative}")
                store.save("scores", part_key, {"C": score, "input_energy": x.float().double().square().mean(1).cpu().numpy()}, info)
                meta["cases"].append({"scope": name, **info})
                atomic_json(args.out / "run_meta.json", meta)
                print(f"SCORED {part_key}: z_rel={z_error}, C_rel={relative:.6g}", flush=True)
        meta.update(status="COMPLETED_PR2_SCORE_GENERATION", elapsed_seconds=time.perf_counter() - started)
    except BaseException as error:
        meta.update(status="FAILED_RESUMABLE", error=str(error))
        raise
    finally:
        atomic_json(args.out / "run_meta.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
