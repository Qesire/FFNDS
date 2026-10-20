#!/usr/bin/env python3
"""Summarize SCOW benchmark runs and artifacts.

This script is intentionally defensive: it tolerates missing manifests,
missing run directories, malformed JSONL lines, and absent probe/artifact
files so that local smoke tests can still generate reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MODE_ORDER = ["bf16", "svdquant", "w4a4_mxfp4"]
STATUS_PRIORITY = {
    "success": 4,
    "skipped": 3,
    "not_implemented": 2,
    "planned": 2,
    "failed": 1,
    "missing": 0,
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "missing"


def csv_write(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(headers: List[str], rows: Iterable[Iterable[Any]]) -> str:
    rows = list(rows)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body]) if rows else "\n".join([header_line, sep_line])


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_manifest(manifest_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    sample_index: Dict[str, Dict[str, Any]] = {}
    samples_by_dim: Dict[str, List[str]] = defaultdict(list)
    for row in read_jsonl(manifest_path):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if sample_id not in sample_index:
            sample_index[sample_id] = row
            dimension = str(row.get("dimension", "unknown")) or "unknown"
            samples_by_dim[dimension].append(sample_id)
    return sample_index, samples_by_dim


def load_metrics(experiment_root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return the latest observed record for each (sample_id, mode)."""
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    runs_dir = experiment_root / "runs"
    if not runs_dir.is_dir():
        return latest

    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    for run_dir in run_dirs:
        metrics_files = [run_dir / "metrics.jsonl"]
        metrics_files.extend(sorted(run_dir.glob("metrics_shard*.jsonl"), key=lambda p: p.name))
        for metrics_file in metrics_files:
            for record in read_jsonl(metrics_file):
                sample_id = str(record.get("sample_id", "")).strip()
                mode = str(record.get("mode", "")).strip()
                if not sample_id or not mode:
                    continue
                status = str(record.get("status", "")).strip().lower()
                existing = latest.get((sample_id, mode))
                existing_status = ""
                if existing:
                    existing_status = str(existing["record"].get("status", "")).strip().lower()
                if existing and STATUS_PRIORITY.get(status, 0) < STATUS_PRIORITY.get(existing_status, 0):
                    continue
                latest[(sample_id, mode)] = {
                    "record": record,
                    "run_dir": run_dir.name,
                    "metrics_file": str(metrics_file),
                }
    return latest


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    if value != value:
        return ""
    return f"{value:.3f}"


def write_reports(experiment_root: Path, manifest_path: Path, out_dir: Path) -> None:
    sample_index, samples_by_dim = load_manifest(manifest_path)
    metrics = load_metrics(experiment_root)

    dimensions = list(samples_by_dim.keys())
    if not dimensions and sample_index:
        dimensions = ["unknown"]

    paired_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    # Determine all dimensions present in manifest; keep a stable order.
    ordered_dims = sorted(dimensions)
    if not ordered_dims and sample_index:
        ordered_dims = ["unknown"]

    for dimension in ordered_dims:
        sample_ids = samples_by_dim.get(dimension, [])
        for mode in MODE_ORDER:
            total = len(sample_ids)
            success = 0
            failed = 0
            missing = 0
            latency_values: List[float] = []
            mem_values: List[float] = []

            for sample_id in sample_ids:
                paired_row = next((r for r in paired_rows if r["sample_id"] == sample_id and r["dimension"] == dimension), None)
                if paired_row is None:
                    paired_row = {
                        "sample_id": sample_id,
                        "dimension": dimension,
                        "bf16_video_path": "",
                        "svdquant_video_path": "",
                        "w4a4_mxfp4_video_path": "",
                        "bf16_sha256": "",
                        "svdquant_sha256": "",
                        "w4a4_mxfp4_sha256": "",
                    }
                    paired_rows.append(paired_row)

                entry = metrics.get((sample_id, mode))
                if not entry:
                    missing += 1
                    missing_rows.append(
                        {
                            "sample_id": sample_id,
                            "dimension": dimension,
                            "mode": mode,
                            "status": "missing",
                            "run_dir": "",
                            "video_path": "",
                            "error": "metrics record missing",
                        }
                    )
                    continue

                record = entry["record"]
                status = str(record.get("status", "")).strip().lower()
                video_path = str(record.get("video_path", "") or "")
                video_sha256 = str(record.get("video_sha256", "") or "")
                latency_sec = safe_float(record.get("latency_sec"))
                peak_gpu_mem_mb = safe_float(record.get("peak_gpu_mem_mb"))
                error = record.get("error")
                error_text = "" if error in (None, "null") else str(error)

                if status in {"success", "skipped"}:
                    success += 1
                    if latency_sec is not None:
                        latency_values.append(latency_sec)
                    if peak_gpu_mem_mb is not None:
                        mem_values.append(peak_gpu_mem_mb)
                elif status:
                    failed += 1
                    missing_rows.append(
                        {
                            "sample_id": sample_id,
                            "dimension": dimension,
                            "mode": mode,
                            "status": status or "failed",
                            "run_dir": entry.get("run_dir", ""),
                            "video_path": video_path,
                            "error": error_text or f"status={status or 'unknown'}",
                        }
                    )

                if status in {"success", "skipped"}:
                    paired_row[f"{mode}_video_path"] = video_path
                    paired_row[f"{mode}_sha256"] = video_sha256

            summary_rows.append(
                {
                    "mode": mode,
                    "dimension": dimension,
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "missing": missing,
                    "avg_latency_sec": format_float(sum(latency_values) / len(latency_values) if latency_values else None),
                    "avg_peak_gpu_mem_mb": format_float(sum(mem_values) / len(mem_values) if mem_values else None),
                }
            )

    # Ensure paired rows include all manifest sample IDs that had any success.
    paired_rows.sort(key=lambda r: (r["dimension"], r["sample_id"]))

    # Artifact index
    artifact_rows: List[Dict[str, Any]] = []
    artifacts_dir = experiment_root / "artifacts"
    allowed_suffixes = {".pt", ".json", ".jsonl", ".yaml", ".yml", ".csv"}
    if artifacts_dir.is_dir():
        for file_path in sorted((p for p in artifacts_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_suffixes), key=lambda p: str(p)):
            rel = file_path.relative_to(experiment_root)
            try:
                size_bytes = file_path.stat().st_size
            except OSError:
                size_bytes = "missing"
            artifact_rows.append(
                {
                    "path": str(file_path),
                    "relative_path": str(rel),
                    "size_bytes": size_bytes,
                    "sha256": sha256_file(file_path),
                }
            )

    # Probe summary
    probe_dir = experiment_root / "probe"
    probe_lines: List[str]
    if probe_dir.is_dir():
        probe_files = sorted([p for p in probe_dir.rglob("*") if p.is_file()], key=lambda p: str(p))
        if probe_files:
            probe_lines = ["# Probe Summary", "", markdown_table(["path", "relative_path", "size_bytes"], ((str(p), str(p.relative_to(experiment_root)), p.stat().st_size) for p in probe_files))]
        else:
            probe_lines = ["# Probe Summary", "", "probe directory exists but contains no files."]
    else:
        probe_lines = ["# Probe Summary", "", "probe directory not_available."]

    # Missing/failed report
    if missing_rows:
        missing_md = ["# Missing or Failed Runs", "", markdown_table(["sample_id", "dimension", "mode", "status", "run_dir", "video_path", "error"], ((r["sample_id"], r["dimension"], r["mode"], r["status"], r["run_dir"], r["video_path"], r["error"]) for r in missing_rows))]
    else:
        missing_md = ["# Missing or Failed Runs", "", "All manifest entries found successful runs for the scanned modes, or the manifest is empty."]

    benchmark_csv = out_dir / "benchmark_summary.csv"
    benchmark_md = out_dir / "benchmark_summary.md"
    paired_csv = out_dir / "paired_outputs.csv"
    artifact_csv = out_dir / "artifact_index.csv"
    probe_md = out_dir / "probe_summary.md"
    missing_md_path = out_dir / "missing_or_failed_runs.md"

    csv_write(
        benchmark_csv,
        ["mode", "dimension", "total", "success", "failed", "missing", "avg_latency_sec", "avg_peak_gpu_mem_mb"],
        summary_rows,
    )
    csv_write(
        paired_csv,
        [
            "sample_id",
            "dimension",
            "bf16_video_path",
            "svdquant_video_path",
            "w4a4_mxfp4_video_path",
            "bf16_sha256",
            "svdquant_sha256",
            "w4a4_mxfp4_sha256",
        ],
        paired_rows,
    )
    csv_write(artifact_csv, ["path", "relative_path", "size_bytes", "sha256"], artifact_rows)

    benchmark_md_text = ["# Benchmark Summary", "", markdown_table(["mode", "dimension", "total", "success", "failed", "missing", "avg_latency_sec", "avg_peak_gpu_mem_mb"], ((r["mode"], r["dimension"], r["total"], r["success"], r["failed"], r["missing"], r["avg_latency_sec"], r["avg_peak_gpu_mem_mb"]) for r in summary_rows))]

    benchmark_md.write_text("\n".join(benchmark_md_text) + "\n", encoding="utf-8")
    probe_md.write_text("\n".join(probe_lines) + "\n", encoding="utf-8")
    missing_md_path.write_text("\n".join(missing_md) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SCOW benchmark outputs.")
    parser.add_argument("--experiment_root", required=True, help="Experiment root directory")
    parser.add_argument("--benchmark_manifest", required=True, help="Benchmark manifest JSONL")
    parser.add_argument("--out_dir", required=True, help="Output report directory")
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    manifest_path = Path(args.benchmark_manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_reports(experiment_root, manifest_path, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
