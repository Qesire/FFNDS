#!/usr/bin/env python3
"""Build fixed VBench subset/probe manifests for Wan I2V evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_DIMS = ["i2v_subject", "i2v_background", "camera_motion"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed VBench subset/probe manifests")
    parser.add_argument("--full_info_json", type=str, required=True, help="Path to vbench2_i2v_full_info.json")
    parser.add_argument("--vbench_root", type=str, required=True, help="Root directory of vbench2_beta_i2v")
    parser.add_argument("--ratio", type=str, default="16-9", help="VBench crop ratio folder")
    parser.add_argument(
        "--dims",
        nargs="+",
        default=DEFAULT_DIMS,
        help="Target VBench dimensions to select, e.g. i2v_subject i2v_background camera_motion",
    )
    parser.add_argument("--per_dim", type=int, default=10, help="Number of samples to keep per dimension")
    parser.add_argument("--probe_per_dim", type=int, default=2, help="Number of probe samples to keep per dimension")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--size", type=str, default="480*832", help="Resolution string")
    parser.add_argument("--frame_num", type=int, default=61, help="Frame count")
    parser.add_argument("--sample_steps", type=int, default=40, help="Sampling steps")
    parser.add_argument("--sample_solver", type=str, default="unipc", help="Sampling solver")
    parser.add_argument("--shift", type=float, default=5.0, help="Sampling shift")
    parser.add_argument("--subset_out", type=str, required=True, help="Subset manifest jsonl output")
    parser.add_argument("--probe_out", type=str, required=True, help="Probe manifest jsonl output")
    return parser.parse_args()


def _load_full_info(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {path}, got {type(data).__name__}")
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            logging.warning("Skipping non-dict item at index %d", idx)
            continue
        out.append(item)
    return out


def _as_dim_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    if value is None:
        return []
    return [str(value)]


def _item_prompt(item: Dict[str, Any]) -> str:
    for key in ("prompt_en", "prompt", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _item_image_name(item: Dict[str, Any]) -> str:
    value = item.get("image_name")
    return str(value).strip() if value is not None else ""


def _build_records(
    full_info: Sequence[Dict[str, Any]],
    dims: Sequence[str],
    per_dim: int,
    *,
    probe_per_dim: int,
    base_seed: int,
    vbench_root: Path,
    ratio: str,
    size: str,
    frame_num: int,
    sample_steps: int,
    sample_solver: str,
    shift: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    subset_rows: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []

    # Prepare per-dimension ordered buckets while preserving the source order.
    per_dim_items: Dict[str, List[Dict[str, Any]]] = {dim: [] for dim in dims}
    seen: set[Tuple[str, str, str]] = set()
    for item in full_info:
        item_dims = set(_as_dim_list(item.get("dimension")))
        if not item_dims:
            continue
        prompt = _item_prompt(item)
        image_name = _item_image_name(item)
        if not prompt or not image_name:
            continue
        for dim in dims:
            if dim not in item_dims:
                continue
            key = (dim, prompt, image_name)
            if key in seen:
                continue
            seen.add(key)
            per_dim_items[dim].append(item)

    for dim in dims:
        ordered_items = per_dim_items.get(dim, [])
        for i, item in enumerate(ordered_items[:per_dim]):
            prompt = _item_prompt(item)
            image_name = _item_image_name(item)
            image_path = (vbench_root / "data" / "crop" / ratio / image_name).resolve()
            row = {
                "sample_id": f"{dim}_{i:03d}",
                "dimension": dim,
                "prompt": prompt,
                "image_path": str(image_path),
                "seed": base_seed + i,
                "resolution": size,
                "frame_num": frame_num,
                "sample_steps": sample_steps,
                "sample_solver": sample_solver,
                "shift": shift,
                "samples_per_prompt": 1,
            }
            subset_rows.append(row)
        for i, item in enumerate(ordered_items[:probe_per_dim]):
            prompt = _item_prompt(item)
            image_name = _item_image_name(item)
            image_path = (vbench_root / "data" / "crop" / ratio / image_name).resolve()
            row = {
                "sample_id": f"{dim}_{i:03d}",
                "dimension": dim,
                "prompt": prompt,
                "image_path": str(image_path),
                "seed": base_seed + i,
                "resolution": size,
                "frame_num": frame_num,
                "sample_steps": sample_steps,
                "sample_solver": sample_solver,
                "shift": shift,
                "samples_per_prompt": 1,
            }
            probe_rows.append(row)

    return subset_rows, probe_rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    dims = [str(dim).strip() for dim in args.dims if str(dim).strip()]
    if not dims:
        raise ValueError("At least one dimension must be provided")

    full_info_path = Path(args.full_info_json).expanduser().resolve()
    vbench_root = Path(args.vbench_root).expanduser().resolve()
    full_info = _load_full_info(full_info_path)

    subset_rows, probe_rows = _build_records(
        full_info,
        dims,
        int(args.per_dim),
        probe_per_dim=int(args.probe_per_dim),
        base_seed=int(args.seed),
        vbench_root=vbench_root,
        ratio=str(args.ratio),
        size=str(args.size),
        frame_num=int(args.frame_num),
        sample_steps=int(args.sample_steps),
        sample_solver=str(args.sample_solver),
        shift=float(args.shift),
    )

    _write_jsonl(Path(args.subset_out).expanduser().resolve(), subset_rows)
    _write_jsonl(Path(args.probe_out).expanduser().resolve(), probe_rows)
    logging.info("Wrote subset=%d probe=%d", len(subset_rows), len(probe_rows))


if __name__ == "__main__":
    main()
