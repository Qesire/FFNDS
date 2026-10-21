"""Copy verified completed HiF4 shards to worker's node-local HF cache.

Use only after stopping the previous writer. Reads each source once, checks its
full SHA256 while copying, and publishes the destination only after validation.
Incomplete shards are preserved at the original location and never adopted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from worker_identity import require_worker

ROOT = Path(__file__).resolve().parents[1]
SERIES = "WORKER_HIF4_SMOOTH_RTN_REBUILD_20261020"


def checksum(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path.home() / ".cache/huggingface/ffnds-derived" / SERIES)
    ap.add_argument("--hf-home", type=Path, default=Path("/var/tmp/ffnds-xiang/huggingface"))
    ap.add_argument("--report", type=Path, default=ROOT / ".runtime/local_hif4_cache.json")
    args = ap.parse_args()
    require_worker()
    os.umask(0o077)
    target = args.hf_home / "ffnds-derived" / SERIES
    markers = sorted((args.source / "shards").glob("*.json"))
    if not markers:
        raise RuntimeError("No completed cache shards to stage")
    target.mkdir(parents=True, exist_ok=True)
    if target.stat().st_dev == args.source.stat().st_dev:
        raise RuntimeError("Destination must use a separate node-local filesystem")
    # Reserve room for both complete shards and the merged cache, plus 80 GiB.
    if shutil.disk_usage(target).free < 2 * 80 * (703 << 20) + (80 << 30):
        raise RuntimeError("Insufficient disk space for complete cache and reserve")
    copied, identity = [], None
    for marker in markers:
        meta = json.loads(marker.read_text())
        if meta["status"] != "PASS":
            raise RuntimeError(f"Incomplete cache marker: {marker}")
        if identity is None:
            identity = meta["identity"]
        if meta["identity"] != identity:
            raise RuntimeError("Source cache shards have different identities")
        source = marker.with_suffix(".pt")
        dest = target / "shards" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            raise RuntimeError(f"Unexpected destination symlink: {dest}")
        if dest.exists():
            if dest.stat().st_size != meta["bytes"] or checksum(dest) != meta["sha256"]:
                raise RuntimeError(f"Existing destination differs: {dest}")
        else:
            temp = dest.with_suffix(".pt.copying")
            sha = hashlib.sha256()
            with source.open("rb") as reader, temp.open("wb") as writer:
                for chunk in iter(lambda: reader.read(16 << 20), b""):
                    sha.update(chunk)
                    writer.write(chunk)
            if temp.stat().st_size != meta["bytes"] or sha.hexdigest() != meta["sha256"]:
                raise RuntimeError(f"Source copy checksum differs: {source}")
            temp.replace(dest)
        dest_marker = dest.with_suffix(".json")
        if dest_marker.exists() and dest_marker.read_bytes() != marker.read_bytes():
            raise RuntimeError(f"Existing marker differs: {dest_marker}")
        shutil.copyfile(marker, dest_marker)
        copied.append({"source": str(source), "destination": str(dest), "bytes": meta["bytes"], "sha256": meta["sha256"]})
        print(f"STAGED verified {dest.name}", flush=True)
    report = {"status": "PASS_COMPLETED_SHARDS_STAGED", "host": "worker", "hf_home": str(args.hf_home),
              "cache_root": str(target), "source": str(args.source), "files": copied,
              "bytes": sum(row["bytes"] for row in copied), "identity": identity,
              "disk_free_after_bytes": shutil.disk_usage(target).free}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temp = args.report.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(args.report)
    print(json.dumps({"status": report["status"], "completed_shards": len(copied), "cache_root": str(target)}))


if __name__ == "__main__":
    main()
