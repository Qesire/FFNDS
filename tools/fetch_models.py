"""Fetch the frozen D checkpoints and base assets into the standard HF cache.

No local_dir is used, and no original-O diffusion weights are downloaded.
The completion manifest is updated after each file, so reruns can reuse verified
blobs. SHA256 checks read the complete downloaded checkpoint, not just its header.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DISTILL = "lightx2v/Wan2.2-Distill-Models"
DISTILL_REV = "52699089e1dac804cbedd2400d0d2aea969577de"
BASE = "Wan-AI/Wan2.2-I2V-A14B"
BASE_REV = "206a9ee1b7bfaaf8f7e4d81335650533490646a3"
FILES = [
    (DISTILL, DISTILL_REV, f"wan2.2_i2v_A14b_{noise}_noise_lightx2v_4step.safetensors")
    for noise in ("high", "low")
] + [(BASE, BASE_REV, name) for name in [
    "configuration.json", "high_noise_model/config.json", "low_noise_model/config.json",
    "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "examples/i2v_input.JPG",
    "google/umt5-xxl/special_tokens_map.json", "google/umt5-xxl/spiece.model",
    "google/umt5-xxl/tokenizer.json", "google/umt5-xxl/tokenizer_config.json",
]]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / ".runtime/models.json")
    args = ap.parse_args()
    # Honor the user's existing cache configuration; the library otherwise uses
    # ~/.cache/huggingface/hub. Never put weight copies in the repository.
    metadata = {}
    api = HfApi()
    for repo, rev in [(DISTILL, DISTILL_REV), (BASE, BASE_REV)]:
        info = api.model_info(repo, revision=rev, files_metadata=True)
        if info.sha != rev:
            raise RuntimeError(f"Revision mismatch for {repo}")
        metadata[repo] = {item.rfilename: item for item in info.siblings}
    contract = json.loads((ROOT / "upstream_audit/source_deltas/HiF4_Sp/files/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json").read_text())
    result = {"status": "IN_PROGRESS", "files": [], "hf_home": os.environ.get("HF_HOME"),
              "hf_hub_cache": os.environ.get("HF_HUB_CACHE"), "started_unix": time.time()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repo, rev, name in FILES:
        meta = metadata[repo][name]
        expected = meta.lfs.sha256 if meta.lfs else None
        if repo == DISTILL:
            noise = "high" if "high_noise" in name else "low"
            if expected != contract[f"{noise}_noise_sha256"]:
                raise RuntimeError(f"Public checkpoint does not match handoff: {name}")
        print(f"Fetching {repo}@{rev}: {name}", flush=True)
        path = Path(hf_hub_download(repo, name, revision=rev))
        if path.stat().st_size != meta.size:
            raise RuntimeError(f"Size mismatch: {path}")
        actual = sha256(path)
        if expected and actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}")
        result["files"].append({"repo": repo, "revision": rev, "filename": name,
                                "path": str(path), "blob": str(path.resolve()),
                                "bytes": path.stat().st_size, "sha256": actual,
                                "lfs_sha256": expected})
        temp = args.output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.output)
        print(f"Verified {name}: {actual}", flush=True)
    result.update(status="PASS", completed_unix=time.time())
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
