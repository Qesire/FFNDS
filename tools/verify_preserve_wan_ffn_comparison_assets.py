#!/usr/bin/env python3
"""Read-only verification and optional non-overwriting backup of Wan FFN evidence.

The manifest contains historical HF-relative paths and recorded SHA-256 identities.
No original file is changed. Running this on a machine without the original HF
cache cannot establish physical preservation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / (
    "research/20261030/WAN_FFN_CROSS_METHOD_ASSET_PRESERVATION_MANIFEST_20261030.json"
)
BLOCK = 8 * 1024 * 1024


def safe_relative(raw: str) -> Path:
    rel = PurePosixPath(raw)
    if (rel.is_absolute() or not rel.parts or
            any(x in ("", ".", "..") for x in rel.parts) or
            rel.parts[0] != "ffnds-derived"):
        raise ValueError(f"unsafe or non-HF relative_path: {raw}")
    return Path(*rel.parts)


def within_root(root: Path, rel: Path) -> Path:
    base = root.resolve(strict=True)
    full = (base / rel).resolve(strict=False)
    if not full.is_relative_to(base):
        raise ValueError(f"symlink escapes root: {rel}")
    return full


def check_sha(path: Path, expected_bytes: int | None, expected: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, "MISSING"
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        return False, f"SIZE_MISMATCH:{actual_bytes}"
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(BLOCK):
            digest.update(chunk)
    got = digest.hexdigest()
    return (got == expected, "SHA256_OK" if got == expected else f"SHA256_MISMATCH:{got}")


def copy_if_absent(source: Path, destination: Path, expected: str, byte_count: int | None) -> str:
    if destination.exists():
        valid, status = check_sha(destination, byte_count, expected)
        if not valid:
            raise RuntimeError(f"archive path exists with different contents: {destination}: {status}")
        return "ARCHIVE_ALREADY_VERIFIED"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A temporary file under the same destination directory permits atomic,
    # non-overwriting hard-link publication. No source file is changed.
    fd, temporary = tempfile.mkstemp(prefix=".wanffn-partial-", dir=destination.parent)
    temp_path = Path(temporary)
    try:
        h = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            while chunk := inp.read(BLOCK):
                out.write(chunk)
                h.update(chunk)
            out.flush()
            os.fsync(out.fileno())
        if byte_count is not None and temp_path.stat().st_size != byte_count:
            raise RuntimeError("source changed size during copy")
        if h.hexdigest() != expected:
            raise RuntimeError("source changed hash during copy")
        try:
            os.link(temp_path, destination)  # EEXIST will not overwrite a prior archive
        except FileExistsError:
            valid, status = check_sha(destination, byte_count, expected)
            if not valid:
                raise RuntimeError(f"conflicting concurrently archived file: {status}")
            return "ARCHIVE_ALREADY_VERIFIED"
        return "ARCHIVE_COPIED_SHA256_OK"
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--hf-root", type=Path, help="Exact existing HF cache root; not guessed")
    parser.add_argument("--archive-root", type=Path, help="Independent safe archive root for --mode copy")
    parser.add_argument("--mode", choices=("inventory", "verify", "copy"), default="inventory")
    parser.add_argument("--family", action="append", help="Repeat to select exact manifest family")
    parser.add_argument("--role", action="append", help="Repeat, e.g. --role video or --role flow_step3")
    parser.add_argument("--limit", type=int, default=0, help="0 means all matching files")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional new report JSON; must not exist")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.mode in ("verify", "copy") and (args.hf_root is None or not args.hf_root.is_dir()):
        parser.error("--hf-root must identify the actual accessible HF cache for verify/copy")
    if args.mode == "copy" and (args.archive_root is None or
                                 not args.archive_root.is_dir()):
        parser.error("--archive-root must already exist for --mode copy")
    if args.mode == "copy" and not args.family:
        parser.error("--mode copy requires at least one explicit --family to bound the archive operation")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = [
        x for x in manifest["assets"]
        if (not args.family or x["family"] in args.family)
        and (not args.role or x["role"] in args.role)
    ]
    if args.limit:
        entries = entries[:args.limit]
    counters = {"selected": len(entries), "ok": 0, "missing_or_corrupt": 0, "copied": 0, "existing_archive": 0}
    details = []
    for entry in entries:
        relative = safe_relative(entry["relative_path"])
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise ValueError(f"invalid hash in {relative}")
        status = "INVENTORY_METADATA_ONLY"
        try:
            if args.mode in ("verify", "copy"):
                original = within_root(args.hf_root, relative)
                ok, status = check_sha(original, entry.get("bytes"), entry["sha256"])
                if not ok:
                    counters["missing_or_corrupt"] += 1
                else:
                    counters["ok"] += 1
                    if args.mode == "copy":
                        archive_path = within_root(args.archive_root, relative)
                        status = copy_if_absent(original, archive_path, entry["sha256"], entry.get("bytes"))
                        if status.startswith("ARCHIVE_COPIED"):
                            counters["copied"] += 1
                        else:
                            counters["existing_archive"] += 1
            details.append({"family": entry["family"], "source": entry["source"],
                            "arm": entry["arm"], "role": entry["role"],
                            "path": entry["relative_path"], "status": status})
        except (OSError, ValueError, RuntimeError) as exc:
            counters["missing_or_corrupt"] += 1
            details.append({"family": entry["family"], "source": entry["source"],
                            "arm": entry["arm"], "role": entry["role"],
                            "path": entry["relative_path"], "status": f"ERROR:{exc}"})
        if not args.quiet:
            print(details[-1]["status"], details[-1]["path"], sep="\t")
    result = {"mode": args.mode, "manifest_source_commit": manifest["source_commit"],
              "manifest": str(args.manifest), "counts": counters,
              "limitations": "No model experiments. SHA verification or copying happens only when invoked on accessible original HF storage.",
              "items": details}
    if args.report:
        with args.report.open("x", encoding="utf-8") as out:
            json.dump(result, out, ensure_ascii=False, indent=2)
            out.write("\n")
    print(json.dumps({"mode": args.mode, "counts": counters}, ensure_ascii=False))
    return 1 if counters["missing_or_corrupt"] else 0


if __name__ == "__main__":
    sys.exit(main())
