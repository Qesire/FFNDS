#!/usr/bin/env python3
"""Register school-owned video/latent and native RUN for fishscale review.

This creates only a local school registry; never copies or generates video.
A successful registration is not a substitute for strict Slurm preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from fishscale_school_preflight import checked_file, sha256, parse_safetensors, video_info, check_geometry, under

CFG = "research/diagnostics/wan_temporal_fishscale/SCHOOL_CLUSTER_CONFIG.json"

def digest_entry(path: Path, root: Path) -> dict:
    file = checked_file(root, str(path))
    if not file.is_file():
        raise FileNotFoundError(file)
    return {"path": str(file), "sha256": sha256(file), "bytes": file.stat().st_size}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(CFG))
    parser.add_argument("--out", type=Path, required=True, help="School-local SCHOOL_PAIR_REGISTRY.json")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--latent", type=Path)
    parser.add_argument("--generation-run", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["school_project_root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError("School project root must be actually present: " + str(root))
    destination = args.out.expanduser().resolve()
    if not under(root, destination):
        raise ValueError("School registry output must stay under school project root")

    video = digest_entry(args.video, root)
    vi = video_info(Path(video["path"]))
    if not check_geometry(vi, cfg["model_recipe"]["reference_geometry"]):
        raise ValueError("Full video has different geometry: " + str(vi))
    video.update({"frames": vi["frames"], "width": vi["width"], "height": vi["height"], "fps": vi["fps"]})

    source: dict = {"id": args.pair_id, "seed": args.seed, "video": video,
                    "source_role": "SCHOOL_REUSED_UNVERIFIED_LINEAGE",
                    "prompt_sha256": None, "status": "REGISTERED_FOR_REVIEW_ONLY"}
    if args.latent:
        info = parse_safetensors(checked_file(root, str(args.latent)))
        if info["dtype"] != "BF16" or info["shape"] != [16,21,58,104]:
            raise ValueError("Native final latent must be BF16 [16,21,58,104]; old .float() capture is rejected")
        source["final_latent"] = dict(digest_entry(args.latent, root), tensor_name=info["tensor_name"],
                                      dtype=info["dtype"], shape=info["shape"])
    if args.image:
        source["image"] = digest_entry(args.image, root)
    if args.prompt_file:
        prompt = checked_file(root, str(args.prompt_file)).read_bytes()
        if not prompt:
            raise ValueError("Empty prompt file")
        source["prompt_sha256"] = hashlib.sha256(prompt).hexdigest()
    if args.generation_run:
        source["parent_run"] = digest_entry(args.generation_run, root)

    required = ["final_latent", "image", "parent_run"]
    if all(k in source for k in required) and source["prompt_sha256"] and args.seed is not None:
        raw = Path(source["parent_run"]["path"]).read_text(encoding="utf-8")
        if source["video"]["sha256"] not in raw or source["final_latent"]["sha256"] not in raw:
            raise ValueError("Native RUN must bind both exact video SHA and final latent SHA")
        source["status"] = "REGISTERED_FOR_STRICT_DECODE_PREFLIGHT"
        source["source_role"] = "SCHOOL_CURRENT_OR_NEW_CLUSTER_BASELINE"

    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("schema") != "FFNDS_FISHSCALE_SCHOOL_ASSET_REGISTRY_V2":
            raise ValueError("Refuse unknown school registry schema")
    else:
        existing = {"schema": "FFNDS_FISHSCALE_SCHOOL_ASSET_REGISTRY_V2",
                    "status": "SCHOOL_LOCAL_REGISTRY_NOT_YET_SLURM_VERIFIED",
                    "school_source_pairs": []}
    if any(p.get("id") == args.pair_id for p in existing["school_source_pairs"]):
        raise ValueError("Duplicate pair ID; never overwrite an existing record")
    existing["school_source_pairs"].append(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(destination)
    print(json.dumps({"status": source["status"], "pair_id": args.pair_id,
                      "registry": str(destination)}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
