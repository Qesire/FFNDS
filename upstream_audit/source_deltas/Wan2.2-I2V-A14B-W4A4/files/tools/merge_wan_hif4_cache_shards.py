#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

EXPERTS = ("low_noise_model", "high_noise_model")


def save_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge expert/block Wan HiF4 cache shards.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--expected-blocks", type=int, default=40)
    ap.add_argument("--expected-modules-per-expert", type=int, default=400)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    root = Path(args.input_dir).resolve()
    files = sorted(root.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no cache shards under {root}")

    merged: dict[str, Any] = {
        "__meta__": {
            "artifact": "wan_hif4_preprocessed_linear_cache",
            "variant": args.variant,
            "format": "hifx4",
            "source_shards": [],
        },
        "low_noise_model": {},
        "high_noise_model": {},
    }
    blocks = {expert: set() for expert in EXPERTS}
    algorithms = set()
    quantizers = set()
    for path in files:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            raise TypeError(f"invalid cache shard: {path}")
        meta = obj.get("__meta__", {})
        if str(meta.get("variant", "")) != args.variant:
            continue
        expert = str(meta.get("expert", ""))
        block = int(meta.get("block", -1))
        if expert not in blocks or block < 0:
            raise ValueError(f"invalid shard metadata: {path}: {meta}")
        blocks[expert].add(block)
        quantizers.add(str(meta.get("weight_quantizer", "")))
        records = obj.get(expert, {})
        if not isinstance(records, dict) or not records:
            raise ValueError(f"empty cache shard: {path}")
        for name, rec in records.items():
            if name in merged[expert]:
                raise KeyError(f"duplicate cache module {expert}.{name}")
            merged[expert][name] = rec
            algorithms.add(str(rec.get("algorithm", "")))
        merged["__meta__"]["source_shards"].append(str(path))

    missing_blocks = {
        expert: sorted(set(range(args.expected_blocks)) - blocks[expert])
        for expert in EXPERTS
    }
    module_counts = {expert: len(merged[expert]) for expert in EXPERTS}
    if not args.allow_partial:
        if any(missing_blocks.values()):
            raise RuntimeError(f"missing blocks: {missing_blocks}")
        for expert, count in module_counts.items():
            if count != args.expected_modules_per_expert:
                raise RuntimeError(
                    f"{expert}: expected {args.expected_modules_per_expert} modules, got {count}"
                )

    merged["__meta__"].update({
        "blocks": {k: sorted(v) for k, v in blocks.items()},
        "missing_blocks": missing_blocks,
        "partial": any(missing_blocks.values()),
        "module_counts": module_counts,
        "algorithms": sorted(algorithms),
        "weight_quantizers": sorted(quantizers),
    })
    output = Path(args.output).resolve()
    save_atomic(merged, output)
    summary = {
        **merged["__meta__"],
        "output": str(output),
        "bytes": output.stat().st_size,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output.with_suffix(output.suffix + ".COMPLETE").write_text(
        "WAN_HIF4_CACHE_COMPLETE\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("MERGE_WAN_HIF4_CACHE_SHARDS_OK")


if __name__ == "__main__":
    main()
