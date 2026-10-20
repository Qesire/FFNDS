#!/usr/bin/env python3
"""Offline audit for Wan SVDQuant PTQ artifacts.

This script does not instantiate the Wan model and does not require weights.
It inspects compact artifacts such as `ptq_stats.pt`, `act.pt`, and `wgt.pt`
and reports smooth scales, calibration activation ranges, low-rank branch
metadata, and packed weight quantization metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from quant.mxfp4 import dequantize_linear_weight_groupwise
from quant.stream_stats import compare_tensors, summarize_tensor
from quant.trace_context import write_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Audit Wan SVDQuant PTQ artifacts offline")
    p.add_argument("--ptq", default="", help="Path to combined ptq_stats.pt or PTQ directory")
    p.add_argument("--act", default="", help="Optional act.pt path")
    p.add_argument("--wgt", default="", help="Optional smoothed weights wgt.pt path")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--spectrum_topk", type=int, default=64)
    p.add_argument("--compute_svd_for_wgt", action="store_true", default=False, help="Compute SVD of smoothed weights in wgt.pt; expensive for large matrices")
    p.add_argument("--max_svd_modules", type=int, default=8)
    p.add_argument("--dequant_main_weight", action="store_true", default=False, help="Dequantize main packed weights for error/norm stats; can be memory-heavy")
    return p.parse_args()


def _torch_load(path: str | Path) -> Dict[str, Any]:
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict artifact at {path}, got {type(obj).__name__}")
    return obj


def _load_ptq(path: str) -> Tuple[Dict[str, Any], str]:
    if not path:
        return {}, ""
    root = Path(path)
    combined = root if root.is_file() else root / "ptq_stats.pt"
    if combined.exists():
        payload = _torch_load(combined)
        if "low_noise_model" in payload and "high_noise_model" in payload:
            return payload, str(combined)
        # Allow passing one split ptq_state.pt directly.
        return {"single_model": payload}, str(combined)
    low_path = root / "low_noise_model" / "ptq_state.pt"
    high_path = root / "high_noise_model" / "ptq_state.pt"
    payload: Dict[str, Any] = {}
    if low_path.exists():
        payload["low_noise_model"] = _torch_load(low_path)
    if high_path.exists():
        payload["high_noise_model"] = _torch_load(high_path)
    if not payload:
        raise FileNotFoundError(f"PTQ artifact not found: {path}")
    return payload, str(root)


def _iter_model_modules(payload: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for model_name, model_state in payload.items():
        if not isinstance(model_state, dict):
            continue
        for module_name, state in model_state.items():
            if module_name == "__meta__" or not isinstance(state, dict):
                continue
            yield str(model_name), str(module_name), state


def _summarize_1d(prefix: str, x: Optional[torch.Tensor]) -> Dict[str, Any]:
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return {}
    stats = summarize_tensor(x, include_channel_stats=False, include_token_norms=False)
    keep = ["numel", "mean", "std", "min", "max", "absmax", "abs_p50", "abs_p90", "abs_p99", "abs_p999", "p99_over_p50", "absmax_over_p99"]
    return {f"{prefix}_{k}": stats[k] for k in keep if k in stats}


def _branch_weight(branch: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
    if not isinstance(branch, dict):
        return None
    a = branch.get("a.weight")
    b = branch.get("b.weight")
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return None
    return b.to(torch.float32) @ a.to(torch.float32)


def _ptq_module_row(model_name: str, module_name: str, state: Dict[str, Any], dequant_main: bool) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    qstate = state.get("quant_linear", {}) if isinstance(state.get("quant_linear"), dict) else {}
    branch = state.get("branch")
    row: Dict[str, Any] = {
        "artifact": "ptq",
        "model": model_name,
        "module": module_name,
        "has_quant_linear": bool(qstate),
        "has_branch": bool(state.get("has_branch", branch is not None)),
        "branch_rank": "",
        "in_features": qstate.get("in_features", ""),
        "out_features": qstate.get("out_features", ""),
        "scheme": qstate.get("scheme", ""),
        "quant_method": qstate.get("quant_method", ""),
        "weight_group_size": qstate.get("w_group_size", qstate.get("weight_group_size", "")),
        "act_group_size": qstate.get("act_group_size", ""),
        "act_clip_ratio": qstate.get("act_clip_ratio", ""),
        "act_scale_method": qstate.get("act_scale_method", ""),
        "act_exp_offset": qstate.get("act_exp_offset", ""),
        "quantize_input": qstate.get("quantize_input", ""),
        "smooth_enabled": qstate.get("smooth_enabled", ""),
        "input_channels_dim": qstate.get("input_channels_dim", ""),
    }
    row.update(_summarize_1d("smooth_scale", qstate.get("smooth_scale")))
    row.update(_summarize_1d("bias", qstate.get("bias")))

    arrays: Dict[str, np.ndarray] = {}
    if isinstance(branch, dict):
        row["branch_rank"] = int(branch.get("rank", -1))
        a = branch.get("a.weight")
        b = branch.get("b.weight")
        row["branch_a_shape"] = list(a.shape) if torch.is_tensor(a) else ""
        row["branch_b_shape"] = list(b.shape) if torch.is_tensor(b) else ""
        w_branch = _branch_weight(branch)
        if w_branch is not None:
            row["branch_fro_norm"] = float(torch.linalg.matrix_norm(w_branch).item())
            # SVD of the low-rank branch is cheap because rank is small.
            try:
                s = torch.linalg.svdvals(w_branch)
                row["branch_top_singular"] = float(s[0].item()) if s.numel() else 0.0
                row["branch_effective_rank_99"] = _energy_rank(s, 0.99)
                arrays[_safe_key(model_name, module_name, "branch_singular_values")] = s[: min(64, int(s.numel()))].cpu().numpy()
            except RuntimeError:
                row["branch_svd_error"] = "svd_failed"

    scales = qstate.get("w_scales")
    if torch.is_tensor(scales) and scales.numel() > 0:
        scale_codes = scales.reshape(-1).to(torch.int64)
        hist = torch.bincount(scale_codes, minlength=255)
        row["weight_scale_code_min"] = int(scale_codes.min().item())
        row["weight_scale_code_max"] = int(scale_codes.max().item())
        row["weight_scale_code_unique"] = int(torch.unique(scale_codes).numel())
        arrays[_safe_key(model_name, module_name, "weight_scale_code_hist")] = hist.cpu().numpy().astype(np.int64)
    qweight = qstate.get("qweight_packed")
    if torch.is_tensor(qweight):
        row["qweight_packed_numel"] = int(qweight.numel())
        row["qweight_packed_bytes"] = int(qweight.numel() * qweight.element_size())

    if dequant_main and qstate:
        try:
            w = dequantize_linear_weight_groupwise(
                qweight_packed=qstate["qweight_packed"],
                scales=qstate["w_scales"].to(torch.uint8),
                in_features=int(qstate["w_in_features"]),
                group_size=int(qstate["w_group_size"]),
                out_dtype=torch.float32,
                device=torch.device("cpu"),
                scheme=str(qstate.get("scheme", "mxfp4")),
            )
            row["main_dequant_fro_norm"] = float(torch.linalg.matrix_norm(w.to(torch.float32)).item())
            row["main_dequant_absmax"] = float(w.abs().amax().item())
            if isinstance(branch, dict):
                w_branch = _branch_weight(branch)
                if w_branch is not None and w_branch.shape == w.shape:
                    row["branch_to_main_fro_ratio"] = float(torch.linalg.matrix_norm(w_branch).item() / max(torch.linalg.matrix_norm(w.to(torch.float32)).item(), 1e-30))
        except Exception as exc:
            row["main_dequant_error"] = type(exc).__name__
    return row, arrays


def _act_rows(act: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name in ("high_noise_model", "low_noise_model", "single_model"):
        model_stats = act.get(model_name)
        if not isinstance(model_stats, dict):
            continue
        for module_name, rec in model_stats.items():
            if module_name == "__meta__" or not isinstance(rec, dict):
                continue
            row = {"artifact": "act", "model": model_name, "module": module_name, "num_calls": rec.get("num_calls", "")}
            row.update(_summarize_1d("act_s", rec.get("s")))
            rows.append(row)
    return rows


def _wgt_rows(wgt: Dict[str, Any], *, compute_svd: bool, topk: int, max_svd_modules: int) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    svd_done = 0
    for model_name in ("high_noise_model", "low_noise_model", "single_model"):
        model_payload = wgt.get(model_name)
        if not isinstance(model_payload, dict):
            continue
        for module_name, rec in model_payload.items():
            if module_name == "__meta__" or not isinstance(rec, dict):
                continue
            row = {"artifact": "wgt", "model": model_name, "module": module_name}
            weight = rec.get("weight")
            scale = rec.get("scale")
            act_s = rec.get("act_s")
            w_s = rec.get("w_s")
            if torch.is_tensor(weight):
                row["weight_shape"] = list(weight.shape)
                row["weight_numel"] = int(weight.numel())
                w32 = weight.to(torch.float32)
                row["weight_absmax"] = float(w32.abs().amax().item()) if w32.numel() else 0.0
                row["weight_fro_norm"] = float(torch.linalg.matrix_norm(w32).item()) if w32.ndim == 2 else float(torch.linalg.vector_norm(w32).item())
                if compute_svd and w32.ndim == 2 and (max_svd_modules <= 0 or svd_done < max_svd_modules):
                    s = torch.linalg.svdvals(w32)
                    arrays[_safe_key(model_name, module_name, "smoothed_weight_singular_values")] = s[: min(topk, int(s.numel()))].cpu().numpy()
                    row["weight_top_singular"] = float(s[0].item()) if s.numel() else 0.0
                    for rank in (1, 2, 4, 8, 16, 32, 64):
                        row[f"weight_energy_rank_{rank}"] = _energy_at_rank(s, rank)
                    row["weight_effective_rank_99"] = _energy_rank(s, 0.99)
                    svd_done += 1
            row.update(_summarize_1d("smooth_scale", scale))
            row.update(_summarize_1d("act_s", act_s))
            row.update(_summarize_1d("w_s", w_s))
            rows.append(row)
    return rows, arrays


def _energy_at_rank(s: torch.Tensor, rank: int) -> float:
    if s.numel() == 0:
        return 0.0
    e = s.to(torch.float64).pow(2)
    return float(e[: min(int(rank), int(e.numel()))].sum().item() / max(e.sum().item(), 1e-30))


def _energy_rank(s: torch.Tensor, target: float) -> int:
    if s.numel() == 0:
        return 0
    e = s.to(torch.float64).pow(2)
    c = torch.cumsum(e, dim=0) / e.sum().clamp_min(1e-30)
    idx = torch.searchsorted(c, torch.tensor(float(target), dtype=c.dtype))
    return int(min(int(idx.item()) + 1, int(s.numel())))


def _safe_key(model_name: str, module_name: str, suffix: str) -> str:
    raw = f"{model_name}__{module_name}__{suffix}"
    return raw.replace(".", "_").replace("/", "_").replace(":", "_")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Any] = {
        "ptq": args.ptq,
        "act": args.act,
        "wgt": args.wgt,
        "compute_svd_for_wgt": bool(args.compute_svd_for_wgt),
        "dequant_main_weight": bool(args.dequant_main_weight),
    }

    if args.ptq:
        ptq, resolved = _load_ptq(args.ptq)
        metadata["ptq_resolved"] = resolved
        ptq_rows: List[Dict[str, Any]] = []
        for model_name, module_name, state in _iter_model_modules(ptq):
            row, arr = _ptq_module_row(model_name, module_name, state, dequant_main=bool(args.dequant_main_weight))
            ptq_rows.append(row)
            arrays.update(arr)
        _write_csv(out_dir / "ptq_module_summary.csv", ptq_rows)
        all_rows.extend(ptq_rows)

    if args.act:
        act_rows = _act_rows(_torch_load(args.act))
        _write_csv(out_dir / "act_summary.csv", act_rows)
        all_rows.extend(act_rows)

    if args.wgt:
        wgt_rows, wgt_arrays = _wgt_rows(
            _torch_load(args.wgt),
            compute_svd=bool(args.compute_svd_for_wgt),
            topk=int(args.spectrum_topk),
            max_svd_modules=int(args.max_svd_modules),
        )
        _write_csv(out_dir / "wgt_summary.csv", wgt_rows)
        all_rows.extend(wgt_rows)
        arrays.update(wgt_arrays)

    _write_csv(out_dir / "svdquant_artifact_summary.csv", all_rows)
    _write_jsonl(out_dir / "svdquant_artifact_summary.jsonl", all_rows)
    if arrays:
        np.savez_compressed(out_dir / "artifact_arrays.npz", **arrays)

    metadata["num_rows"] = len(all_rows)
    metadata["array_keys"] = sorted(arrays.keys())
    metadata["outputs"] = {
        "summary_csv": str(out_dir / "svdquant_artifact_summary.csv"),
        "summary_jsonl": str(out_dir / "svdquant_artifact_summary.jsonl"),
        "arrays_npz": str(out_dir / "artifact_arrays.npz") if arrays else "",
    }
    write_metadata(out_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
