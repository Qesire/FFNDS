"""Shared strict asset checks for new persistent dense-pipeline experiments."""
import json
from pathlib import Path
from worker_identity import matches_worker

from ar2_checkpoint import digest

ROOT = Path(__file__).resolve().parents[1]


def verified_assets():
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    logical_path = ROOT / ".runtime/model_view.json"
    local_path = ROOT / ".runtime/node_local_model_view.json"
    cache_path = ROOT / ".runtime/hif4_rebuilt_cache.json"
    logical, local, cache = [json.loads(p.read_text()) for p in [logical_path, local_path, cache_path]]
    if logical["status"] != "PASS" or local["status"] != "PASS" or not matches_worker(local["host"]):
        raise RuntimeError("Need verified model views on their declared node")
    if local["logical_model_view_sha256"] != digest(logical_path):
        raise RuntimeError("Node model mirror belongs to another logical view")
    def key(row):
        return tuple(row[n] for n in ["repo", "revision", "filename", "bytes", "sha256"])
    if len(local["files"]) != 12 or {key(r) for r in local["files"]} != {key(r) for r in logical["files"]}:
        raise RuntimeError("Local and logical model identities differ")
    for row in local["files"]:
        p = Path(row["view_path"])
        if not row["copy_sha256_verified"] or p.stat().st_size != row["bytes"] or not p.samefile(row["blob"]):
            raise RuntimeError("Node-local HF view changed")
    if cache["status"] != "PASS_FULL_NATIVE_HIF4_RTN_CACHE" or cache["identity"]["model_view_sha256"] != digest(logical_path):
        raise RuntimeError("Quantized cache/model mismatch")
    if cache["identity"]["checkpoint_sha256"] != {r["filename"]: r["sha256"] for r in logical["files"] if r["filename"].endswith(".safetensors")}:
        raise RuntimeError("Distilled checkpoint identities differ")
    for p, expected in cache["identity"]["source_sha256"].items():
        if digest(p) != expected:
            raise RuntimeError("Quantization source changed")
    if Path(cache["path"]).stat().st_size != cache["bytes"] or digest(cache["path"]) != cache["sha256"]:
        raise RuntimeError("Quantized cache full checksum mismatch")
    base = Path(local["view"])
    kwargs = {"base_model_dir": base, **{f"{noise}_noise_ckpt": base / f"{noise}_noise_model/wan2.2_i2v_A14b_{noise}_noise_lightx2v_4step.safetensors" for noise in ["high", "low"]}}
    identity = {"model_view_sha256": digest(logical_path), "local_view_sha256": digest(local_path),
                "cache_manifest_sha256": digest(cache_path), "cache_sha256": cache["sha256"],
                "contract": adapter.DISTILL_RUNTIME_CONTRACT, "checkpoint_sha256": cache["identity"]["checkpoint_sha256"]}
    return kwargs, cache, identity
