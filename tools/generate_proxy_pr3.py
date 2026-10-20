"""Predict hidden activity using input PCA from a task0 opposite-seed donor."""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from ar2_core import donor_key
from generate_proxy_pr2 import contribution_scores
from run_ar2_restart import ROOT, key_name, load_scopes, tensor_sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrices", type=Path, nargs="+", required=True)
    ap.add_argument("--expert", choices=["high_noise_model", "low_noise_model"], required=True)
    ap.add_argument("--pr2-scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as qdq
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "JOB.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = ROOT / "config/proxy_pr3_input_pca.json"
    protocol = json.loads(protocol_path.read_text())
    scopes, inputs = load_scopes(args.matrices)
    cache_path = ROOT / ".runtime/hif4_rebuilt_cache.json"
    cache = json.loads(cache_path.read_text())
    shard = next(s for s in cache["shards"] if s["expert"] == args.expert and s["block"] == 20)
    if digest(shard["path"]) != shard["sha256"]:
        raise RuntimeError("Cache checksum mismatch")
    payload = torch.load(shard["path"], map_location="cpu", weights_only=True, mmap=True)[args.expert]
    w1, w2 = [payload[f"blocks.20.ffn.{i}"]["weight_qdq"].cuda() for i in [0, 2]]
    scale = payload["blocks.20.ffn.2"]["smooth_scale"].cuda()
    pr2_identity = json.loads((args.pr2_scores / "identity.json").read_text())
    pr2 = CheckpointStore(args.pr2_scores, pr2_identity["identity"])
    pr2_meta = json.loads((args.pr2_scores / "run_meta.json").read_text())
    if (pr2_meta["status"] != "COMPLETED_PR2_SCORE_GENERATION" or pr2_meta["fingerprint"] != pr2.fingerprint
            or pr2_identity["identity"]["inputs"] != inputs or pr2_identity["identity"]["W1_sha256"] != tensor_sha(w1)
            or pr2_identity["identity"]["W2_sha256"] != tensor_sha(w2)):
        raise RuntimeError("PR2 operator controls differ")
    mirror = json.loads((ROOT / ".runtime/node_local_model_view.json").read_text())
    source = next(r for r in mirror["files"] if f"/{args.expert}/" in r["view_path"] and r["filename"].endswith(".safetensors"))
    with safe_open(source["view_path"], framework="pt", device="cpu") as handle:
        bias = handle.get_tensor("blocks.20.ffn.0.bias").cuda()
    if tensor_sha(bias) != pr2_identity["identity"]["b1_sha256"] or tensor_sha(scale) != pr2_identity["identity"]["s2_sha256"]:
        raise RuntimeError("Bias/scale differs from verified PR2")
    store = CheckpointStore(args.out, {"protocol": protocol, "protocol_sha256": digest(protocol_path),
        "sources": {str(p): digest(p) for p in [Path(__file__), ROOT / "tools/generate_proxy_pr2.py"]},
        "expert": args.expert, "inputs": inputs, "pr2_controls": pr2.fingerprint,
        "cache_manifest_sha256": digest(cache_path), "torch": torch.__version__, "TF32": False})
    meta = {"status": "RUNNING", "fingerprint": store.fingerprint, "expert": args.expert,
            "protocol_sha256": digest(protocol_path), "quality_or_speed_acceptance": False, "factors": [], "cases": []}
    factor_root = Path(shard["path"]).parent / "pr3_pca_factors"
    factor_root.mkdir(exist_ok=True)
    started = time.perf_counter()
    atomic_json(args.out / "run_meta.json", meta)
    try:
        operands, factors = {}, {}
        for key, data in scopes.items():
            if key.expert == args.expert:
                with safe_open(data["scope"]["tensor_file"], framework="pt", device="cpu") as handle:
                    operands[key] = {n: handle.get_tensor(n).cuda() for n in ["V1", "Z"]}
        for key, tensors in operands.items():
            if key.task != 0:
                continue
            x = tensors["V1"].float()
            mu = x.mean(0)
            _, s, vh = torch.linalg.svd(x - mu, full_matrices=False)
            p = vh.T.contiguous()
            b = w1.float() @ p
            center = F.linear(mu, w1.float(), bias.float())
            factors[key] = (mu, p, b, center)
            path = factor_root / f"{key_name(key)}_{store.fingerprint}.pt"
            if not path.exists():
                temp = Path(str(path) + ".tmp")
                torch.save({"mu": mu.cpu(), "P": p.cpu(), "B": b.cpu(), "center": center.cpu(),
                            "fingerprint": store.fingerprint, "donor": asdict(key)}, temp)
                temp.replace(path)
            retained = {str(r): float(s[:r].double().square().sum() / s.double().square().sum()) for r in protocol["ranks"]}
            meta["factors"].append({"donor": asdict(key), "path": str(path), "sha256": digest(path), "training_centered_input_energy_retained": retained})
            print(f"PCA {key_name(key)}: {retained}", flush=True)
        for key, tensors in operands.items():
            name, donor = key_name(key), donor_key(key)
            gate = scopes[key]["gate"].cuda()
            energy = tensors["V1"].float().double().square().mean(1).cpu().numpy()
            for rank in protocol["ranks"]:
                variant = f"rank{rank}"
                part_key = f"{name}_{variant}"
                if store.load("scores", part_key) is not None:
                    continue
                arrays, errors = {"input_energy": energy}, {}
                bindings = [("C", donor)] + ([("training_C", key)] if key.task == 0 else [])
                for label, fit_key in bindings:
                    mu, p, b, center = factors[fit_key]
                    z = ((tensors["V1"].float() - mu) @ p[:, :rank]) @ b[:, :rank].T + center
                    z = z.to(torch.bfloat16)
                    h = F.gelu(z, approximate="tanh")
                    a = qdq((h.float() / scale).to(torch.bfloat16), force_fp32=True)
                    arrays[label] = contribution_scores(a, w2, gate)
                    errors[label] = float((z.double() - tensors["Z"].double()).norm() / tensors["Z"].double().norm())
                info = {"variant": variant, "donor": asdict(donor), "training_uses_own_donor_basis": key.task == 0,
                        "Z_relative_L2_diagnostic": errors, "pre_FFN0_predictor_inputs_only": True}
                store.save("scores", part_key, arrays, info)
                meta["cases"].append({"scope": name, **info})
                atomic_json(args.out / "run_meta.json", meta)
                print(f"SCORED {part_key}: {errors}", flush=True)
            for control in protocol["controls"]:
                part_key = f"{name}_{control}"
                if store.load("scores", part_key) is None:
                    old = pr2.load("scores", part_key)
                    store.save("scores", part_key, old["arrays"], {**old["metadata"], "reused_verified_PR2_control": pr2.fingerprint})
        meta.update(status="COMPLETED_PR3_SCORE_GENERATION", elapsed_seconds=time.perf_counter() - started)
    except BaseException as error:
        meta.update(status="FAILED_RESUMABLE", error=str(error))
        raise
    finally:
        atomic_json(args.out / "run_meta.json", meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
