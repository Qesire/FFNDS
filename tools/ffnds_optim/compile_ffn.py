"""Offline initializer for explicit FFN factors from existing frozen CAL packs.

No model run, calibration collection or online SVD. Every exported factor
records its source tensors and is labelled weight-Frobenius initialized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import torch

from .ffn import compile_factor_asset, factor_cycle_budget


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _site(value):
    if not re.fullmatch(r"(?:high|low)(?:0[0-9]|[12][0-9]|3[0-9])", value):
        raise ValueError("site must be high00..high39 or low00..low39")
    return value


def _gauge(path):
    if path is None:
        return None
    if path.suffix == ".json":
        value = json.loads(path.read_text())
    else:
        value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        value = value["gauge"]
    return torch.as_tensor(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cal-manifest", type=Path)
    source.add_argument("--pack", type=Path)
    parser.add_argument("--site", action="append", help="Explicit site(s); single pack requires exactly one")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New, nonexisting directory")
    gauges = parser.add_mutually_exclusive_group()
    gauges.add_argument("--gauge", type=Path, help="One fixed gauge vector for the selected sites")
    gauges.add_argument("--gauge-dir", type=Path, help="Explicit high00.pt / low00.pt gauge files")
    parser.add_argument("--dtype", choices=("bfloat16", "float32", "float64"), default="bfloat16")
    parser.add_argument("--rows", type=int, default=31668, help="Resource-accounting geometry, no forward run")
    parser.add_argument("--layers", type=int, default=40, help="Resource-accounting resident layer count")
    args = parser.parse_args(argv)
    if args.rank < 1 or args.rows < 1 or args.layers < 1:
        parser.error("rank, rows and layers must be positive")
    selected = None if args.site is None else [_site(s) for s in args.site]
    if selected is not None and len(set(selected)) != len(selected):
        parser.error("duplicate --site")
    if args.pack:
        if selected is None or len(selected) != 1:
            parser.error("--pack requires exactly one --site")
        entries = {selected[0]: {"path": str(args.pack.resolve())}}
        source_identity = {"kind": "single_pack", "path": str(args.pack.resolve())}
    else:
        manifest = json.loads(args.cal_manifest.read_text())
        if manifest.get("status") != "PASS" or not isinstance(manifest.get("sites"), dict):
            raise ValueError("expected an existing PASS frozen CAL manifest with sites")
        names = selected if selected is not None else sorted(manifest["sites"])
        entries = {}
        for name in names:
            _site(name)
            ref = dict(manifest["sites"][name])
            path = Path(ref["path"])
            if not path.is_absolute():
                path = args.cal_manifest.resolve().parent / path
            if not ref.get("sha256") or file_digest(path) != ref["sha256"]:
                raise ValueError("frozen source pack fingerprint differs: " + name)
            if "bytes" in ref and path.stat().st_size != ref["bytes"]:
                raise ValueError("frozen source pack size differs: " + name)
            entries[name] = dict(ref, path=str(path.resolve()))
        source_identity = {"kind": "cal_manifest", "path": str(args.cal_manifest.resolve()),
                           "sha256": file_digest(args.cal_manifest)}
    # Refuse preexisting outputs; partial earlier attempts remain reviewable.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = args.output_dir / "COMPILE_RECEIPT.json"
    receipt = dict(status="RUNNING", method="weight_frobenius_svd_initializer",
                    task_quality="unmeasured", recovery_aware_fitting_performed=False,
                    new_model_forwards=0, calibration_collection=False,
                    source=source_identity, rank=args.rank, factor_dtype=args.dtype,
                    resource_rows=args.rows, resource_layers=args.layers, sites={})
    def save_receipt():
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    save_receipt()
    try:
        for name, ref in entries.items():
            path = Path(ref["path"])
            pack = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            gauge_path = args.gauge
            if args.gauge_dir is not None:
                gauge_path = args.gauge_dir / (name + ".pt")
            asset = compile_factor_asset(pack, args.rank, gauge=_gauge(gauge_path),
                                           dtype=getattr(torch, args.dtype))
            asset.update(site=name, source_pack_sha256=file_digest(path))
            target = args.output_dir / (name + ".pt")
            temporary = target.with_suffix(".pt.tmp")
            torch.save(asset, temporary)
            temporary.replace(target)
            row = dict(path=target.name, sha256=file_digest(target), bytes=target.stat().st_size,
                       source_path=str(path), source_pack_sha256=asset["source_pack_sha256"],
                       source_a0_sha256=asset["source_a0_sha256"],
                       source_jmap_sha256=asset["source_jmap_sha256"], source_v_sha256=asset["source_v_sha256"],
                       gauge_path=None if gauge_path is None else str(gauge_path.resolve()),
                       gauge_sha256=None if gauge_path is None else file_digest(gauge_path),
                       discarded_singular_energy=asset["discarded_singular_energy"],
                       total_singular_energy=asset["total_singular_energy"],
                       budget=factor_cycle_budget(args.rows, pack["A0"].shape[0],len(pack["sel"]),
                           args.rank, pack["v"].shape[0], args.layers))
            receipt["sites"][name] = row
            save_receipt()
            print(json.dumps(dict(site=name, output=str(target), rank=args.rank,
                                   method=receipt["method"])), flush=True)
        receipt["status"] = "COMPLETE"
        save_receipt()
        return receipt
    except Exception as exc:
        receipt.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        save_receipt()
        raise


if __name__ == "__main__":
    main()
