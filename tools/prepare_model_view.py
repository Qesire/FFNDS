"""Create the historical file layout using hardlinks entirely inside the HF cache.

The archived adapter resolves symlinks before checking expert directory names.
HF snapshot symlinks therefore fail its filename contract. Hardlinks preserve
the exact cached bytes and required names without copying 57 GB of weights or
changing the numerical adapter. Existing mismatching paths are never replaced.
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

from fetch_models import BASE, BASE_REV, DISTILL, DISTILL_REV, FILES

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / ".runtime/models.json")
    ap.add_argument("--output", type=Path, default=ROOT / ".runtime/model_view.json")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["status"] != "PASS":
        raise RuntimeError("Wait for download and complete SHA256 validation")
    files = {(f["repo"], f["revision"], f["filename"]): f for f in manifest["files"]}
    if set(files) != set(FILES):
        raise RuntimeError("Model manifest does not contain exactly the pinned files")
    view = Path(HF_HUB_CACHE) / ".ffnds-model-views" / f"distill4-{DISTILL_REV[:12]}-base-{BASE_REV[:12]}"
    links = []
    for key, meta in files.items():
        repo, revision, filename = key
        source = Path(meta["path"]).resolve(strict=True)
        if source.stat().st_size != meta["bytes"]:
            raise RuntimeError(f"Cached file size changed: {source}")
        if repo == DISTILL:
            noise = "high" if "high_noise" in filename else "low"
            target = view / f"{noise}_noise_model" / filename
        else:
            target = view / filename
        if not target.resolve().is_relative_to(view.resolve()):
            raise RuntimeError(f"Model view path escapes its root: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise RuntimeError(f"Expected a hardlink, found symlink: {target}")
        if target.exists():
            if not os.path.samefile(source, target):
                raise RuntimeError(f"Existing model view differs from cached blob: {target}")
        else:
            os.link(source, target)
        links.append({**meta, "view_path": str(target), "same_inode_as_hf_blob": os.path.samefile(source, target)})
    result = {"status": "PASS", "view": str(view), "layout": "hardlinks inside HF cache; no weight byte copies",
              "manifest": str(args.manifest.resolve()), "files": links}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
