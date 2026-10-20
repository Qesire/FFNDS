"""Stage the three-metric VBench weights inside the Hugging Face cache."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from huggingface_hub import hf_hub_download

from ar2_checkpoint import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "ffnds-derived/vbench-models"
SOURCES = [
    {"name": "CLIP_ViT_L14", "relative": "clip_model/ViT-L-14.pt",
     "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
     "sha_prefix": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"},
    {"name": "LAION_aesthetic_linear", "relative": "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
     "url": "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/6d122adad522ab246644d9dc1c6d7a3810ee255f/sa_0_4_vit_l_14_linear.pth",
     "source_revision": "6d122adad522ab246644d9dc1c6d7a3810ee255f"},
    {"name": "MUSIQ_SPAQ", "relative": "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
     "url": "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth",
     "sha_prefix": "358bb6af"},
    {"name": "AMT_S", "relative": "amt_model/amt-s.pth", "repo": "lalala125/AMT", "file": "amt-s.pth",
     "revision": "31d0420ef7788f6dc22aad3445148705007141bb",
     "sha_prefix": "07e7e03405c213fe4405678db9b42d05671e48c56faf6a441b99bb0047d3cf77"},
]


def stage(source):
    path = CACHE / source["relative"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if "repo" in source:
            original = Path(hf_hub_download(source["repo"], source["file"], revision=source["revision"]))
            path.hardlink_to(original.resolve())
        else:
            temp = Path(str(path) + ".incomplete")
            offset = temp.stat().st_size if temp.exists() else 0
            request = urllib.request.Request(source["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with temp.open(mode) as handle:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)
            actual = digest(temp)
            if not actual.startswith(source.get("sha_prefix", "")):
                raise RuntimeError(f"Published model checksum mismatch: {source['name']}")
            temp.replace(path)
    actual = digest(path)
    if not actual.startswith(source.get("sha_prefix", "")):
        raise RuntimeError(f"Cached model checksum mismatch: {path}")
    result = {**source, "path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    print(json.dumps({"model": source["name"], "status": "VERIFIED", "bytes": result["bytes"]}), flush=True)
    return result


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(stage, SOURCES))
    report = {"status": "PASS_STAGED_VBENCH3_MODELS", "cache_dir": str(CACHE),
              "models": records, "source_sha256": digest(Path(__file__)), "metric_execution_acceptance": False}
    atomic_json(ROOT / ".runtime/vbench_models.json", report)


if __name__ == "__main__":
    main()
