"""Build a fresh Smooth(alpha=.5)+native HiF4 RTN W4A4 cache in HF cache.

Quantize one expert/block at a time; complete shards carry exact file hashes and
immutable identities. Smoothing and BF16 casting follow the archived collector,
then the original native_hif4_weight_qdq is called. RTN does not use a Hessian.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from collect_smooth_stats import validate_stats
from run_dense_bf16 import ROOT, atomic_json

SERIES = "WORKER_HIF4_SMOOTH_RTN_REBUILD_20261020"
EXPERTS = ["high_noise_model", "low_noise_model"]
DEFAULT_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "ffnds-derived" / SERIES


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paths(root, expert, block):
    path = root / "shards" / f"{expert}_block_{block:02d}.pt"
    return path, path.with_suffix(".json")


def save_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pt.tmp")
    torch.save(value, temp)
    temp.replace(path)


def load_verified_inputs(args):
    view = json.loads(args.model_view.read_text())
    calibration = json.loads((args.calibration / "RUN.json").read_text())
    if view["status"] != "PASS" or calibration["status"] != "PASS_FULL_NATIVE_CALIBRATION12":
        raise RuntimeError("Exact model verification and complete native calibration12 are required")
    act_path = Path(calibration["act_file"])
    if digest(act_path) != calibration["act_sha256"]:
        raise RuntimeError("Calibration statistics hash changed")
    stats = torch.load(act_path, map_location="cpu", weights_only=True)
    validate_stats(stats, 12)
    checkpoint_shas = {r["filename"]: r["sha256"] for r in view["files"] if r["filename"].endswith(".safetensors")}
    source_paths = [ROOT / p for p in ["tools/rebuild_hif4_cache.py",
        ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py",
        ".runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py",
        ".runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QFuncs/hifx.py",
        ".runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QTensor.py",
        ".runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QFuncs/quant_basic.py",
        ".runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QType.py"]]
    source_paths += sorted((ROOT / ".runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/cusrc").glob("*.so"))
    if sum(p.suffix == ".so" for p in source_paths) != 2:
        raise RuntimeError("Both audited native CUDA extensions must exist")
    identity = {"series": SERIES, "calibration_run_sha256": digest(args.calibration / "RUN.json"),
                "act_sha256": calibration["act_sha256"], "model_view_sha256": digest(args.model_view),
                "checkpoint_sha256": checkpoint_shas,
                "source_sha256": {str(p): digest(p) for p in source_paths},
                "versions": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "safetensors", "triton"]},
                "cuda_runtime": torch.version.cuda,
                "alpha": 0.5, "eps": 1e-8, "smooth_weight_dtype": "bfloat16",
                "weight_quantizer": "native_hif4_rtn", "force_fp32": True, "activation_qdq": True,
                "curvature": "NOT_USED_RTN", "old_DQ_cache_byte_identity": "NOT_CLAIMED_NEW_CACHE"}
    return view, stats, identity


def verify_shard(root, expert, block, identity):
    path, marker = paths(root, expert, block)
    meta = json.loads(marker.read_text())
    if meta["identity"] != identity or meta["expert"] != expert or meta["block"] != block or meta["status"] != "PASS":
        raise RuntimeError(f"Cache shard identity mismatch: {marker}")
    if path.stat().st_size != meta["bytes"] or digest(path) != meta["sha256"]:
        raise RuntimeError(f"Cache shard hash mismatch: {path}")
    return meta


def quantize(args, view, stats, identity):
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_weight_qdq
    from quant.hif4_backend import hif4_backend_metadata
    base = Path(view["view"])
    for noise in args.experts:
        expert = f"{noise}_noise_model"
        ckpt = base / expert / f"wan2.2_i2v_A14b_{noise}_noise_lightx2v_4step.safetensors"
        # Keep the exact original [out,in] checkpoint layout for F.linear.
        with safe_open(str(ckpt), framework="pt", device="cpu") as handle:
            for block in args.blocks:
                path, marker = paths(args.cache_root, expert, block)
                if marker.exists():
                    verify_shard(args.cache_root, expert, block, identity)
                    print(f"VERIFIED existing {expert} block {block}", flush=True)
                    continue
                if path.exists():
                    raise RuntimeError(f"Orphan shard requires inspection before replacement: {path}")
                records, audits = {}, []
                names = sorted(name for name in stats[expert] if name.startswith(f"blocks.{block}."))
                if len(names) != 10:
                    raise RuntimeError(f"Expected 10 modules for {expert} block {block}")
                for name in names:
                    key = name + ".weight"
                    weight = handle.get_tensor(key).to(torch.float32)
                    act_s = stats[expert][name]["s"].to(torch.float32).clamp_min(1e-8)
                    w_s = weight.abs().amax(dim=0).clamp_min(1e-8)
                    scale = (act_s.pow(0.5) / w_s.pow(0.5)).clamp_min(1e-8)
                    smoothed = (weight * scale.view(1, -1)).to(torch.bfloat16).contiguous()
                    qdq = native_hif4_weight_qdq(smoothed.cuda(), force_fp32=True).to(device="cpu", dtype=torch.bfloat16).contiguous()
                    if qdq.shape != smoothed.shape or not bool(torch.isfinite(qdq).all()):
                        raise RuntimeError(f"Invalid native QDQ output for {expert}.{name}")
                    records[name] = {"in_features": int(qdq.shape[1]), "out_features": int(qdq.shape[0]),
                        "weight_qdq": qdq, "smooth_scale": scale.cpu().contiguous(),
                        "activation_qdq": True, "force_fp32": True, "algorithm": "native_hif4_rtn",
                        "transforms": ["smoothquant_wan_wgt"], "source_name": name,
                        "metadata": {"format": "hifx4", "weight_quantizer": "rtn", "source_preprocess": "smooth",
                                     "calibration_act_sha256": identity["act_sha256"], "curvature": "NOT_USED_RTN"},
                        "report": {"native_QDQ_changed_elements": int(torch.count_nonzero(qdq != smoothed)),
                                   "elements": qdq.numel()}}
                    audits.append({"name": name, "checkpoint_key": key, "weight_shape": list(qdq.shape),
                                   "scale_min": float(scale.min()), "scale_max": float(scale.max()),
                                   **records[name]["report"]})
                    del weight, act_s, w_s, scale, smoothed, qdq
                backend = hif4_backend_metadata()
                meta = {"identity": identity, "expert": expert, "block": block, "variant": "hif4_smooth_rtn_w4a4",
                        "format": "hifx4", "weight_quantizer": "rtn", "activation_qdq": True,
                        "experiment_design": "smooth_rtn_without_curvature_w4a4_runtime", "backend": backend}
                payload = {"__meta__": meta, "high_noise_model": {}, "low_noise_model": {}}
                payload[expert] = records
                save_atomic(path, payload)
                atomic_json(marker, {**meta, "status": "PASS", "sha256": digest(path), "bytes": path.stat().st_size,
                                     "modules": audits, "completed_utc": datetime.now(timezone.utc).isoformat()})
                print(f"COMPLETED {expert} block {block}/39: {path.stat().st_size} bytes", flush=True)
                del payload, records


def merge(args, view, identity):
    output = args.cache_root / "hif4_smooth_rtn_w4a4.pt"
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise RuntimeError("Final cache already exists; do not silently replace it")
    payload = {"__meta__": {"identity": identity, "variant": "hif4_smooth_rtn_w4a4", "format": "hifx4",
                "weight_quantizer": "rtn", "activation_qdq": True, "curvature": "NOT_USED_RTN",
                "model_cls": "wan2.2_moe_distill", "scheduler": "Wan22StepDistillScheduler",
                "denoising_step_list": [1000, 750, 500, 250], "boundary_step_index": 2,
                "sample_shift": 5.0, "enable_cfg": False, "base_model_dir": view["view"],
                "target_frames": 81, "height": 480, "width": 832, "fps": 16,
                "module_counts": {expert: 400 for expert in EXPERTS}, "partial": False,
                "experiment_design": "smooth_rtn_without_curvature_w4a4_runtime"},
               "high_noise_model": {}, "low_noise_model": {}}
    shards = []
    for expert in EXPERTS:
        for block in range(40):
            info = verify_shard(args.cache_root, expert, block, identity)
            path, marker = paths(args.cache_root, expert, block)
            shard = torch.load(path, map_location="cpu", weights_only=True)
            expected_names = {name for row in info["modules"] for name in [row["name"]]}
            if len(expected_names) != 10 or set(shard[expert]) != expected_names:
                raise RuntimeError(f"Incorrect module coverage in shard: {path}")
            if set(payload[expert]) & set(shard[expert]):
                raise RuntimeError("Duplicate quantized modules")
            payload[expert].update(shard[expert])
            shards.append({"expert": expert, "block": block, "path": str(path), "sha256": info["sha256"]})
            print(f"MERGED verified {expert} block {block}/39", flush=True)
    if any(len(payload[expert]) != 400 for expert in EXPERTS):
        raise RuntimeError("Expected exactly 800 quantized linear modules")
    print("Writing complete cache; full file SHA256 follows", flush=True)
    save_atomic(output, payload)
    sha = digest(output)
    atomic_json(manifest, {"status": "PASS_FULL_NATIVE_HIF4_RTN_CACHE", "identity": identity,
                          "path": str(output.resolve()), "sha256": sha, "bytes": output.stat().st_size,
                          "shards": shards, "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "PASS_FULL_NATIVE_HIF4_RTN_CACHE", "path": str(output), "sha256": sha}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["quantize", "merge"])
    ap.add_argument("--model-view", type=Path, default=ROOT / ".runtime/model_view.json")
    ap.add_argument("--calibration", type=Path, default=ROOT / "runs/worker_restart_20261020/calibration12_smooth")
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--experts", choices=["high", "low"], nargs="+", default=["high", "low"])
    ap.add_argument("--blocks", type=int, choices=range(40), nargs="+", default=list(range(40)))
    args = ap.parse_args()
    # Derived model assets stay in the configured HF cache, alongside cached model inputs.
    hf_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).resolve()
    if not args.cache_root.resolve().is_relative_to(hf_root):
        ap.error("Quantized weights must stay inside HF_HOME")
    if len(args.blocks) != len(set(args.blocks)) or len(args.experts) != len(set(args.experts)):
        ap.error("Expert/block lists must not contain duplicates")
    view, stats, identity = load_verified_inputs(args)
    if args.stage == "quantize":
        quantize(args, view, stats, identity)
    else:
        merge(args, view, identity)


if __name__ == "__main__":
    main()
