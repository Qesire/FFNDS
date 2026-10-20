#!/usr/bin/env python3
"""Run or plan Wan I2V VBench manifest shards.

This runner intentionally keeps the first version conservative:
- dry_run is fully supported and does not load any model code
- resume skips samples that already have a recorded success and an existing
  video hash
- non-dry-run emits explicit not_implemented/error metrics instead of
  pretending generation succeeded

The CLI is shaped for the shard wrapper and future GPU execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sharded Wan I2V VBench manifest")
    parser.add_argument("--manifest", type=str, required=True, help="JSONL manifest path")
    parser.add_argument("--mode", type=str, required=True, choices=["bf16", "svdquant", "w4a4_mxfp4"], help="Inference mode")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory")
    parser.add_argument("--shard_index", type=int, required=True, help="Shard index, 0-based")
    parser.add_argument("--shard_count", type=int, required=True, help="Total number of shards")
    parser.add_argument("--ckpt_dir", type=str, default="", help="Wan checkpoint directory")
    parser.add_argument("--ptq_dir", type=str, default="", help="SVDQuant PTQ artifact path")
    parser.add_argument("--load_mxfp4", type=str, default="", help="W4A4 MXFP4 artifact path")
    parser.add_argument("--freeze_condition_latent", type=_str2bool, default=True, help="Enable freeze_condition_latent generation variant")
    parser.add_argument("--resume", action="store_true", default=False, help="Skip samples that already have successful outputs")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Plan only; do not load or run models")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned or "sample"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(f"{path}:{line_no}: {exc.msg}", exc.doc, exc.pos) from exc
            if not isinstance(item, dict):
                raise TypeError(f"{path}:{line_no}: expected dict, got {type(item).__name__}")
            rows.append(item)
    return rows


def _load_previous_successes(metrics_path: Path) -> Dict[str, Dict[str, Any]]:
    if not metrics_path.is_file():
        return {}
    latest: Dict[str, Dict[str, Any]] = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for raw in f:
            text = raw.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            sample_id = str(record.get("sample_id", "")).strip()
            if not sample_id:
                continue
            if record.get("status") == "success":
                latest[sample_id] = record
    return latest


def _planned_video_path(run_dir: Path, sample_id: str, manifest_line: int) -> Path:
    return run_dir / "videos" / f"{_safe_filename(sample_id)}__line{manifest_line:06d}.mp4"


def _populate_common_record(record: Dict[str, Any], row: Dict[str, Any], *, shard_index: int, shard_count: int) -> None:
    record["manifest_line"] = int(row.get("_manifest_line", 0))
    record["shard_index"] = int(shard_index)
    record["shard_count"] = int(shard_count)
    record["dimension"] = row.get("dimension")
    record["prompt"] = row.get("prompt")
    record["image_path"] = row.get("image_path")
    record["seed"] = row.get("seed")
    record["resolution"] = row.get("resolution")
    record["frame_num"] = row.get("frame_num")
    record["sample_steps"] = row.get("sample_steps")
    record["sample_solver"] = row.get("sample_solver")
    record["shift"] = row.get("shift")
    record["samples_per_prompt"] = row.get("samples_per_prompt")


def _base_metric(sample_id: str, mode: str, video_path: Path, freeze_condition_latent: bool) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "mode": mode,
        "status": "planned",
        "video_path": str(video_path),
        "video_sha256": "",
        "latency_sec": None,
        "peak_gpu_mem_mb": None,
        "freeze_condition_latent": bool(freeze_condition_latent),
        "error": "",
    }


def _emit(metrics_path: Path, record: Dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _non_dry_run_not_implemented(sample: Dict[str, Any], *, mode: str, video_path: Path, freeze_condition_latent: bool) -> Dict[str, Any]:
    record = _base_metric(str(sample.get("sample_id", "")), mode, video_path, freeze_condition_latent)
    record["status"] = "not_implemented"
    record["error"] = (
        "non-dry-run generation is not implemented in this shard runner yet; "
        "emit metrics only. Use the existing generation scripts directly for GPU runs."
    )
    return record


def main() -> int:
    args = _parse_args()

    if args.shard_count <= 0:
        raise ValueError(f"--shard_count must be > 0, got {args.shard_count}")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError(f"--shard_index must be in [0, {args.shard_count}), got {args.shard_index}")
    if not args.ckpt_dir and not args.dry_run:
        raise ValueError("--ckpt_dir is required unless --dry_run is set")

    manifest_path = Path(args.manifest).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / f"metrics_shard{args.shard_index}.jsonl"
    previous_successes = _load_previous_successes(metrics_path) if args.resume else {}

    rows = _read_jsonl(manifest_path)
    shard_rows = []
    for row_index, row in enumerate(rows):
        row["_manifest_line"] = row_index
        if row_index % args.shard_count == args.shard_index:
            shard_rows.append(row)

    for row in shard_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            sample_id = f"manifest_line_{row.get('_manifest_line', 0):06d}"

        planned_video = _planned_video_path(run_dir, sample_id, int(row.get("_manifest_line", 0)))
        record = _base_metric(sample_id, args.mode, planned_video, args.freeze_condition_latent)
        _populate_common_record(record, row, shard_index=args.shard_index, shard_count=args.shard_count)

        prior = previous_successes.get(sample_id)
        if args.resume and prior is not None:
            prior_path = Path(str(prior.get("video_path", ""))).expanduser()
            prior_hash = _sha256_file(prior_path)
            if prior_hash != "missing":
                record["status"] = "skipped"
                record["video_path"] = str(prior_path)
                record["video_sha256"] = prior_hash
                record["latency_sec"] = prior.get("latency_sec")
                record["peak_gpu_mem_mb"] = prior.get("peak_gpu_mem_mb")
                record["error"] = ""
                _emit(metrics_path, record)
                continue

        if args.dry_run:
            record["status"] = "planned"
            _emit(metrics_path, record)
            continue

        record = _non_dry_run_not_implemented(row, mode=args.mode, video_path=planned_video, freeze_condition_latent=args.freeze_condition_latent)
        _populate_common_record(record, row, shard_index=args.shard_index, shard_count=args.shard_count)
        _emit(metrics_path, record)

    return 0 if args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
