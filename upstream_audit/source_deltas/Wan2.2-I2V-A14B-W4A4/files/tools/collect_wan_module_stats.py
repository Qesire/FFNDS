#!/usr/bin/env python3
"""Collect per-module probe statistics from Wan I2V DiT.

Modes (--mode):
  bf16                 — BF16 reference forward, collect module static/activation stats
  svdquant             — SVDQuant model forward (+ optional PTQ stats)
  w4a4_mxfp4           — W4A4 MXFP4 quantized model forward
  svdquant_freeze_off  — SVDQuant, freeze_condition_latent=False
  w4a4_freeze_off      — W4A4 MXFP4, freeze_condition_latent=False

Output files (all in --output_dir):
  module_static.csv         — Per-module metadata (param count, flops, shape, …)
  activation_stats.jsonl    — Activation distribution statistics per module
  quant_error_stats.jsonl   — Quantization error metrics (BF16 reference needed)
  layer_latency.csv         — Per-module / per-step latency
  summary_by_block.csv      — Per-block aggregated stats
  summary_by_module_type.csv — Per-type aggregated stats

IMPORTANT: Full forward-hook collection is NOT yet wired.
  - propagated error → null with version2_planned marker
  - raw activations are never saved to disk
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

LOGGER = logging.getLogger("collect_wan_module_stats")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_FILES: List[str] = [
    "module_static.csv",
    "activation_stats.jsonl",
    "quant_error_stats.jsonl",
    "layer_latency.csv",
    "summary_by_block.csv",
    "summary_by_module_type.csv",
]

DRY_RUN_SCHEMA: Dict[str, Any] = {
    "tool": "collect_wan_module_stats",
    "version": "0.1.0",
    "dry_run": True,
    "created_at": None,
    "output_files": OUTPUT_FILES,
    "fields": {
        "module_static.csv": [
            "expert", "block_id", "module_name", "module_type",
            "in_features", "out_features", "num_params", "flops_per_step",
            "dtype", "device",
        ],
        "activation_stats.jsonl": [
            "expert", "block_id", "module_name", "module_type",
            "step_idx", "calls", "mean", "std", "min", "max",
            "absmean", "absmax", "sparsity", "outlier_ratio",
            "status", "raw_activation_saved",
        ],
        "quant_error_stats.jsonl": [
            "expert", "block_id", "module_name", "module_type",
            "step_idx", "calls",
            "mae", "mse", "rmse", "rel_l2", "max_abs_error",
            "cosine", "cosine_error",
            "propagated_mae", "propagated_mse",
            "propagated_error_status",
            "version2_planned",  # Always True — not yet wired
        ],
        "layer_latency.csv": [
            "expert", "block_id", "module_name", "module_type",
            "step_idx", "latency_ms",
        ],
        "summary_by_block.csv": [
            "expert", "block_id", "num_modules", "flops_total",
            "mean_mae", "mean_rmse", "mean_rel_l2", "mean_cosine",
            "mean_latency_ms",
        ],
        "summary_by_module_type.csv": [
            "expert", "module_type", "count", "flops_total",
            "mean_mae", "mean_rmse", "mean_rel_l2", "mean_cosine",
            "mean_latency_ms",
        ],
    },
    "notes": {
        "version2_planned": "propagated error fields (propagated_mae, propagated_mse) are null/0 in v1; "
                             "full propagated error tracing is planned for v2.",
        "raw_activations": "Raw activation tensors are never saved to disk. "
                           "Only scalar reductions are persisted.",
        "target_modules": "Hooks are registered on: Linear, self_attn, cross_attn, ffn, modulation. "
                          "All DiT blocks (high/low experts) are instrumented.",
    },
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("collect_wan_module_stats")
    p.add_argument("--manifest", required=True, help="JSONL manifest with probe samples")
    p.add_argument(
        "--mode",
        required=True,
        choices=["bf16", "svdquant", "w4a4_mxfp4", "svdquant_freeze_off", "w4a4_freeze_off"],
    )
    p.add_argument("--output_dir", required=True, help="Output directory for stats files")
    p.add_argument("--ckpt_dir", default="", help="BF16 checkpoint directory")
    p.add_argument("--ptq_dir", default="", help="Path to ptq_stats.pt")
    p.add_argument("--load_mxfp4", default="", help="Path to MXFP4 state .pt")
    p.add_argument("--freeze_condition_latent", type=str, default="True", choices=["True", "False"])
    p.add_argument("--max_samples", type=int, default=1)
    p.add_argument("--sample_steps", type=int, default=0)
    p.add_argument("--size", type=str, default="832*480")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true", default=False, help="Skip model load, only emit schema")
    p.add_argument("--device_id", type=int, default=0)
    return p.parse_args()


def _make_output_dir(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_dry_run_schema(out_dir: Path, args: argparse.Namespace) -> None:
    schema = dict(DRY_RUN_SCHEMA)
    schema["created_at"] = datetime.now(timezone.utc).isoformat()
    schema["cli_args"] = {
        "manifest": args.manifest,
        "mode": args.mode,
        "ckpt_dir": args.ckpt_dir or "(empty)",
        "ptq_dir": args.ptq_dir or "(empty)",
        "load_mxfp4": args.load_mxfp4 or "(empty)",
        "freeze_condition_latent": args.freeze_condition_latent,
        "max_samples": args.max_samples,
        "sample_steps": args.sample_steps,
        "size": args.size,
        "seed": args.seed,
        "dry_run": True,
    }
    (out_dir / "dry_run_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "module_static.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(DRY_RUN_SCHEMA["fields"]["module_static.csv"])
        w.writerow([
            "dit", "0", "blocks.0.attn.q", "Linear", "", "", "", "",
            "schema_only", "dry_run",
        ])

    with (out_dir / "activation_stats.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "expert": "dit",
            "block_id": 0,
            "module_name": "blocks.0.attn.q",
            "module_type": "Linear",
            "step_idx": None,
            "calls": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "absmean": None,
            "absmax": None,
            "sparsity": None,
            "outlier_ratio": None,
            "status": "dry_run_schema",
            "raw_activation_saved": False,
        }, ensure_ascii=False) + "\n")

    with (out_dir / "quant_error_stats.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "expert": "dit",
            "block_id": 0,
            "module_name": "blocks.0.attn.q",
            "module_type": "Linear",
            "step_idx": None,
            "calls": 0,
            "mae": None,
            "mse": None,
            "rmse": None,
            "rel_l2": None,
            "max_abs_error": None,
            "cosine": None,
            "cosine_error": None,
            "propagated_mae": None,
            "propagated_mse": None,
            "propagated_error_status": "version2_planned",
            "version2_planned": True,
        }, ensure_ascii=False) + "\n")

    with (out_dir / "layer_latency.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(DRY_RUN_SCHEMA["fields"]["layer_latency.csv"])
    with (out_dir / "summary_by_block.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(DRY_RUN_SCHEMA["fields"]["summary_by_block.csv"])
    with (out_dir / "summary_by_module_type.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(DRY_RUN_SCHEMA["fields"]["summary_by_module_type.csv"])
    LOGGER.info("Dry-run schema written to %s", out_dir)


def _load_manifest(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples > 0 and len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _validate_manifest(samples: Sequence[Dict[str, Any]]) -> None:
    required_keys = {"sample_id", "image_path", "prompt"}
    for i, s in enumerate(samples):
        missing = required_keys - set(s.keys())
        if missing:
            raise ValueError(f"Manifest sample {i} missing keys: {missing}")


def _validate_non_dry_run_args(args: argparse.Namespace) -> None:
    errors: List[str] = []
    if not args.ckpt_dir and args.mode == "bf16":
        errors.append("--ckpt_dir is required for mode=bf16")
    if not args.ptq_dir and args.mode in ("svdquant", "svdquant_freeze_off"):
        errors.append("--ptq_dir is required for svdquant modes")
    if not args.load_mxfp4 and args.mode in ("w4a4_mxfp4", "w4a4_freeze_off"):
        errors.append("--load_mxfp4 is required for w4a4_mxfp4 modes")
    if not args.ckpt_dir and args.mode in ("svdquant", "w4a4_mxfp4", "svdquant_freeze_off", "w4a4_freeze_off"):
        errors.append("--ckpt_dir is required for skeleton loading")

    if errors:
        for e in errors:
            LOGGER.error("Validation error: %s", e)
        sys.exit(1)


def _write_placeholder_outputs(out_dir: Path, manifest: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Write structured placeholder outputs with status=not_implemented.

    This is a placeholder until real forward hooks are wired.
    """
    metadata: Dict[str, Any] = {
        "tool": "collect_wan_module_stats",
        "version": "0.1.0",
        "status": "not_implemented",
        "error": "Real forward hooks are not yet wired. "
                 "This is a structured placeholder. "
                 "Set version2_planned=True for propagated error fields.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cli_args": {
            "manifest": args.manifest,
            "mode": args.mode,
            "ckpt_dir": args.ckpt_dir,
            "ptq_dir": args.ptq_dir,
            "load_mxfp4": args.load_mxfp4,
            "freeze_condition_latent": args.freeze_condition_latent,
            "max_samples": args.max_samples,
            "sample_steps": args.sample_steps,
            "size": args.size,
            "seed": args.seed,
            "dry_run": False,
        },
        "manifest_samples": [
            {"sample_id": s.get("sample_id", ""), "image_path": s.get("image_path", ""),
             "prompt": s.get("prompt", "")[:60]}
            for s in manifest
        ],
    }

    (out_dir / "collection_status.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (out_dir / "module_static.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(DRY_RUN_SCHEMA["fields"]["module_static.csv"])
        w.writerow([
            "not_implemented", "", "", "", "", "", "", "",
            "not_implemented", "hooks_not_wired",
        ])

    with (out_dir / "activation_stats.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "expert": "not_implemented",
            "block_id": None,
            "module_name": "",
            "module_type": "",
            "step_idx": None,
            "calls": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "absmean": None,
            "absmax": None,
            "sparsity": None,
            "outlier_ratio": None,
            "status": "not_implemented",
            "raw_activation_saved": False,
        }, ensure_ascii=False) + "\n")

    with (out_dir / "quant_error_stats.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "expert": "not_implemented",
            "block_id": None,
            "module_name": "",
            "module_type": "",
            "step_idx": None,
            "calls": 0,
            "mae": None,
            "mse": None,
            "rmse": None,
            "rel_l2": None,
            "max_abs_error": None,
            "cosine": None,
            "cosine_error": None,
            "propagated_mae": None,
            "propagated_mse": None,
            "propagated_error_status": "version2_planned",
            "version2_planned": True,
        }, ensure_ascii=False) + "\n")

    with (out_dir / "layer_latency.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(DRY_RUN_SCHEMA["fields"]["layer_latency.csv"])
    with (out_dir / "summary_by_block.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(DRY_RUN_SCHEMA["fields"]["summary_by_block.csv"])
    with (out_dir / "summary_by_module_type.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(DRY_RUN_SCHEMA["fields"]["summary_by_module_type.csv"])

    LOGGER.error("Collection is not implemented. "
                 "Real forward hooks must be wired. "
                 "See %s/collection_status.json for details.", out_dir)
    sys.exit(1)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    out_dir = _make_output_dir(Path(args.output_dir))

    if args.dry_run:
        _write_dry_run_schema(out_dir, args)
        return

    samples = _load_manifest(Path(args.manifest), int(args.max_samples))
    _validate_manifest(samples)

    _validate_non_dry_run_args(args)

    _write_placeholder_outputs(out_dir, samples, args)


if __name__ == "__main__":
    main()
