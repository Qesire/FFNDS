#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from HiFloat4.ptq.state import CurvatureStats
from HiFloat4.ptq.transforms.magr import MagRConfig, apply_magr

EXPERTS = ("low_noise_model", "high_noise_model")


def load_dict(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict: {path}")
    return obj


def save_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def curvature_from_record(name: str, rec: dict[str, Any], expected_domain: str) -> CurvatureStats:
    matrix = rec.get("hessian", rec.get("matrix"))
    if not torch.is_tensor(matrix):
        raise TypeError(f"missing Hessian for {name}")
    out = CurvatureStats(
        matrix=matrix.detach().float().cpu().contiguous(),
        nsamples=int(rec["nsamples"]),
        in_features=int(rec.get("in_features", matrix.shape[0])),
        normalization=str(rec.get("normalization", "mean_xtx")),
        domain=str(rec.get("domain", "")),
        module_name=name,
    )
    out.validate()
    if out.domain != expected_domain:
        raise ValueError(f"{name}: expected curvature domain={expected_domain}, got {out.domain}")
    return out


def optional_float(value: str) -> float | None:
    return None if value.strip().lower() in {"", "none", "auto"} else float(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build one reusable Wan MagR prepared-weight shard (v4.8).")
    ap.add_argument("--smooth-shard", required=True)
    ap.add_argument("--hessian-shard", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["group64", "leaf4", "hif4_tree"], default="group64")
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--spectral", choices=["power", "exact"], default="exact")
    ap.add_argument("--power-iters", type=int, default=32)
    ap.add_argument("--dykstra-iters", type=int, default=32)
    ap.add_argument("--dykstra-tol", type=float, default=1e-6)
    ap.add_argument("--compute-device", default="cuda:0")
    ap.add_argument("--save-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    ap.add_argument("--expected-curvature-domain", choices=["smooth_a4_qdq", "smooth_bf16_noqdq", "smooth_a4", "smooth_a16"], required=True)
    ap.add_argument("--normalize-tree-group-count", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--normalize-tree-lambdas", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--lambda64", default="auto")
    ap.add_argument("--lambda8", default="auto")
    ap.add_argument("--lambda4", default="auto")
    args = ap.parse_args()

    smooth_path = Path(args.smooth_shard).resolve()
    hessian_path = Path(args.hessian_shard).resolve()
    smooth = load_dict(smooth_path)
    hessian = load_dict(hessian_path)
    smeta = smooth.get("__meta__", {})
    hmeta = hessian.get("__meta__", {})
    expert = str(smeta.get("expert", ""))
    block = int(smeta.get("block", -1))
    if expert not in EXPERTS or block < 0:
        raise ValueError(f"invalid smooth shard metadata: {smeta}")
    if str(hmeta.get("expert", "")) != expert or int(hmeta.get("block", -1)) != block:
        raise ValueError("smooth/curvature shard expert-block mismatch")
    if str(hmeta.get("domain", "")) != args.expected_curvature_domain:
        raise ValueError(
            f"shard domain mismatch: expected {args.expected_curvature_domain}, got {hmeta.get('domain')}"
        )

    smooth_records = smooth.get(expert, {})
    hessian_records = hessian.get(expert, {})
    if set(smooth_records) != set(hessian_records):
        missing_h = sorted(set(smooth_records) - set(hessian_records))
        extra_h = sorted(set(hessian_records) - set(smooth_records))
        raise RuntimeError(f"module mismatch missing_h={missing_h} extra_h={extra_h}")

    cfg = MagRConfig(
        mode=args.mode,
        alpha=args.alpha,
        outer_iters=args.iters,
        spectral_method=args.spectral,
        power_iters=args.power_iters,
        dykstra_iters=args.dykstra_iters,
        dykstra_tol=args.dykstra_tol,
        compute_device=args.compute_device,
        normalize_tree_group_count=bool(args.normalize_tree_group_count),
        normalize_tree_lambdas=bool(args.normalize_tree_lambdas),
        lambda64=optional_float(args.lambda64),
        lambda8=optional_float(args.lambda8),
        lambda4=optional_float(args.lambda4),
    )
    cfg.validate()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.save_dtype]
    resolved_lambdas = cfg.resolved_level_weights()

    output: dict[str, Any] = {
        "__meta__": {
            "artifact": "wan_magr_prepared_weight_shard_v48",
            "schema_version": 48,
            "expert": expert,
            "block": block,
            "module_count": len(smooth_records),
            "source_smooth_shard": str(smooth_path),
            "source_hessian_shard": str(hessian_path),
            "preprocess": f"smooth_magr_{args.mode}",
            "magr": {
                "mode": args.mode,
                "alpha": args.alpha,
                "iters": args.iters,
                "spectral": args.spectral,
                "power_iters": args.power_iters,
                "dykstra_iters": args.dykstra_iters,
                "dykstra_tol": args.dykstra_tol,
                "compute_device": args.compute_device,
                "normalize_tree_group_count": bool(args.normalize_tree_group_count),
                "normalize_tree_lambdas": bool(args.normalize_tree_lambdas),
                "resolved_lambdas": list(resolved_lambdas),
            },
            "curvature_domain": args.expected_curvature_domain,
            "manifest_sha256": hmeta.get("manifest_sha256"),
            "ordered_sample_ids_sha256": hmeta.get("ordered_sample_ids_sha256"),
            "resolved_image_paths_sha256": hmeta.get("resolved_image_paths_sha256"),
            "save_dtype": args.save_dtype,
        },
        "low_noise_model": {}, "high_noise_model": {},
    }

    summaries = []
    for idx, name in enumerate(sorted(smooth_records), start=1):
        rec = smooth_records[name]
        weight = rec["weight"].detach()
        scale = rec["scale"].detach().float().cpu().contiguous()
        curvature = curvature_from_record(name, hessian_records[name], args.expected_curvature_domain)
        result = apply_magr(weight, curvature, cfg)
        output[expert][name] = {
            "weight": result.weight.detach().to(device="cpu", dtype=dtype).contiguous(),
            "scale": scale,
            "input_channels_dim": int(rec.get("input_channels_dim", -1)),
            "shape": list(result.weight.shape),
            "source_preprocess": "smooth",
            "transforms": ["smoothquant_wan_wgt", f"magr_{args.mode}"],
            "magr_metadata": result.metadata,
            "magr_report": result.report,
        }
        summaries.append({"name": name, **result.report})
        print(f"[{idx}/{len(smooth_records)}] MagR {expert}.{name}")

    out = Path(args.output).resolve()
    save_atomic(output, out)
    summary = {**output["__meta__"], "output": str(out), "bytes": out.stat().st_size, "modules": summaries}
    out.with_suffix(out.suffix + ".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("WAN_MAGR_PREPARED_SHARD_V48_OK")


if __name__ == "__main__":
    main()
