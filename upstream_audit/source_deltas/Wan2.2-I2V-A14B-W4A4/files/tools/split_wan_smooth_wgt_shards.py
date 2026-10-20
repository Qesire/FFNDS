#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

EXPERTS = ("low_noise_model", "high_noise_model")
BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Split Wan Smooth wgt.pt into expert/block shards.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--expected-blocks", type=int, default=40)
    ap.add_argument("--expected-modules-per-expert", type=int, default=400)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    source = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve()
    obj = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict in {source}")

    index: dict[str, Any] = {
        "__meta__": {
            "artifact": "wan_smooth_prepared_shard_index",
            "source": str(source),
            "source_sha256": sha256(source),
            "expected_blocks": args.expected_blocks,
        },
        "experts": {},
    }
    total = 0
    for expert in EXPERTS:
        records = obj.get(expert)
        if not isinstance(records, dict):
            raise KeyError(f"{source} missing {expert}")
        if args.expected_modules_per_expert > 0 and len(records) != args.expected_modules_per_expert:
            raise RuntimeError(
                f"{expert}: expected {args.expected_modules_per_expert} modules, got {len(records)}"
            )
        by_block: dict[int, dict[str, Any]] = {i: {} for i in range(args.expected_blocks)}
        for name, rec in records.items():
            match = BLOCK_RE.match(str(name))
            if match is None:
                raise ValueError(f"unexpected module outside blocks: {expert}.{name}")
            block = int(match.group(1))
            if block not in by_block:
                raise ValueError(f"block out of range: {expert}.{name}")
            if not isinstance(rec, dict) or "weight" not in rec or "scale" not in rec:
                raise ValueError(f"invalid smooth record: {expert}.{name}")
            by_block[block][str(name)] = rec
        expert_index: dict[str, Any] = {}
        for block, block_records in by_block.items():
            if not block_records:
                raise RuntimeError(f"empty smooth block shard: {expert} block={block}")
            path = out_dir / f"{expert}_block{block:02d}.pt"
            if path.exists() and not args.overwrite:
                existing = torch.load(path, map_location="cpu", weights_only=False)
                if not isinstance(existing, dict) or len(existing.get(expert, {})) != len(block_records):
                    raise RuntimeError(f"existing shard is inconsistent: {path}")
            else:
                payload = {
                    "__meta__": {
                        "artifact": "wan_smooth_prepared_weight_shard",
                        "source": str(source),
                        "source_sha256": index["__meta__"]["source_sha256"],
                        "expert": expert,
                        "block": block,
                        "module_count": len(block_records),
                        "preprocess": "smooth",
                        "weight_domain": "smooth_weight_only",
                    },
                    "low_noise_model": {},
                    "high_noise_model": {},
                }
                payload[expert] = block_records
                save_atomic(payload, path)
            expert_index[str(block)] = {
                "path": str(path),
                "module_count": len(block_records),
                "bytes": path.stat().st_size,
            }
            total += len(block_records)
        index["experts"][expert] = expert_index

    index["__meta__"]["total_modules"] = total
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "COMPLETE").write_text("WAN_SMOOTH_SHARDS_COMPLETE\n", encoding="utf-8")
    print(json.dumps(index["__meta__"], indent=2, ensure_ascii=False))
    print("WAN_SMOOTH_WGT_SPLIT_OK")


if __name__ == "__main__":
    main()
