#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.generate_opens2v_eval import load_eval_samples, select_samples


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(
    eval_jsons: list[Path], image_root: Path, *, expected: int, start: int = 0,
    hash_images: bool = False,
) -> dict[str, Any]:
    if expected <= 0:
        raise ValueError("expected must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    for path in eval_jsons:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name == "Single-Domain_Eval.json":
            raise RuntimeError("fallback calibration manifest is forbidden: Single-Domain_Eval.json")
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)

    samples = load_eval_samples([p.resolve() for p in eval_jsons], image_root.resolve())
    selected = select_samples(samples, start, expected)
    if len(selected) != expected:
        raise RuntimeError(f"expected exactly {expected} selected samples, got {len(selected)}")

    ids = [str(s.videoid) for s in selected]
    prompts = [str(s.prompt).strip() for s in selected]
    paths = [Path(s.used_image).resolve() for s in selected]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"duplicate sample/video IDs: {ids}")
    if any(not p for p in prompts):
        bad = [ids[i] for i, p in enumerate(prompts) if not p]
        raise RuntimeError(f"empty prompt for sample IDs: {bad}")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"resolved calibration images missing: {missing}")
    path_strings = [str(p) for p in paths]
    if len(set(path_strings)) != len(path_strings):
        raise RuntimeError(f"duplicate resolved image paths: {path_strings}")

    image_sha256 = [sha256_file(p) for p in paths] if hash_images else []
    if hash_images and len(set(image_sha256)) != len(image_sha256):
        raise RuntimeError("duplicate calibration image content detected by SHA-256")

    manifest_entries = [
        {"path": str(p.resolve()), "sha256": sha256_file(p)} for p in eval_jsons
    ]
    manifest_sha256 = sha256_json(manifest_entries)
    ordered_sample_sha256 = sha256_json(ids)
    resolved_image_paths_sha256 = sha256_json(path_strings)
    report: dict[str, Any] = {
        "artifact": "opens2v_manifest_validation_v48",
        "manifest_paths": [str(p.resolve()) for p in eval_jsons],
        "manifest_files": manifest_entries,
        "manifest_sha256": manifest_sha256,
        "image_root": str(image_root.resolve()),
        "num_samples": expected,
        "start": start,
        "ordered_sample_ids": ids,
        "ordered_sample_ids_sha256": ordered_sample_sha256,
        "resolved_image_paths": path_strings,
        "resolved_image_paths_sha256": resolved_image_paths_sha256,
        "prompts_nonempty": True,
        "sample_ids_unique": True,
        "resolved_image_paths_unique": True,
        "image_content_sha256": image_sha256,
        "image_content_unique": (len(set(image_sha256)) == len(image_sha256)) if hash_images else None,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict OpenS2V calibration-manifest gate for v4.8")
    ap.add_argument("--eval-json", nargs="+", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--expected", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--hash-images", action="store_true")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    report = validate_manifest(
        [Path(p) for p in args.eval_json], Path(args.image_root),
        expected=args.expected, start=args.start, hash_images=args.hash_images,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("OPENS2V_MANIFEST_V48_OK")


if __name__ == "__main__":
    main()
