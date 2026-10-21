"""Mirror exact verified model bytes into worker's local HF hub cache.

The original logical model identity is preserved. This creates a separate I/O
view with identical repo/revision/filename/size/SHA256 tuples and local hardlinks,
so repeated fresh-process observer runs do not reread checkpoints through NFS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from worker_identity import require_worker

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-view", type=Path, default=ROOT / ".runtime/model_view.json")
    ap.add_argument("--hf-home", type=Path, default=Path("/var/tmp/ffnds-xiang/huggingface"))
    ap.add_argument("--output", type=Path, default=ROOT / ".runtime/node_local_model_view.json")
    args = ap.parse_args()
    require_worker()
    os.umask(0o077)
    original = json.loads(args.model_view.read_text())
    if original["status"] != "PASS" or len(original["files"]) != 12:
        raise RuntimeError("Expected the exact completed 12-file model view")
    hub = args.hf_home / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    if hub.stat().st_dev == args.model_view.stat().st_dev:
        raise RuntimeError("Destination must use the node-local filesystem")
    required = sum(row["bytes"] for row in original["files"])
    if shutil.disk_usage(hub).free < required + (80 << 30):
        raise RuntimeError("Insufficient local disk space with 80 GiB reserve")
    view = hub / ".ffnds-model-views" / Path(original["view"]).name
    files = []
    for row in original["files"]:
        source = Path(row["blob"])
        if source.stat().st_size != row["bytes"] or not source.samefile(row["view_path"]):
            raise RuntimeError("Verified original model view changed")
        repo = hub / ("models--" + row["repo"].replace("/", "--"))
        blob = repo / "blobs" / source.name
        blob.parent.mkdir(parents=True, exist_ok=True)
        if blob.is_symlink():
            raise RuntimeError("Unexpected local blob symlink")
        if blob.exists():
            if blob.stat().st_size != row["bytes"] or sha(blob) != row["sha256"]:
                raise RuntimeError(f"Local HF blob differs: {blob}")
        else:
            temp = blob.with_suffix(".copying")
            actual = hashlib.sha256()
            with source.open("rb") as reader, temp.open("wb") as writer:
                for chunk in iter(lambda: reader.read(16 << 20), b""):
                    actual.update(chunk)
                    writer.write(chunk)
            if temp.stat().st_size != row["bytes"] or actual.hexdigest() != row["sha256"]:
                raise RuntimeError(f"Copied model checksum differs: {source}")
            temp.replace(blob)
        snapshot = repo / "snapshots" / row["revision"] / row["filename"]
        relative = Path(row["view_path"]).relative_to(original["view"])
        target = view / relative
        for link in [snapshot, target]:
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or (link.exists() and not link.samefile(blob)):
                raise RuntimeError(f"Conflicting local HF cache entry: {link}")
            if not link.exists():
                os.link(blob, link)
        files.append({**row, "original_blob": str(source), "original_view_path": row["view_path"],
                      "path": str(snapshot), "blob": str(blob), "view_path": str(target),
                      "copy_sha256_verified": True})
        print(f"STAGED exact model bytes: {row['filename']}", flush=True)
    result = {"status": "PASS", "host": "worker", "view": str(view), "files": files,
              "logical_model_view": str(args.model_view.resolve()), "logical_model_view_sha256": sha(args.model_view),
              "layout": "node-local HF blobs/snapshots/view hardlinks; exact byte mirror only",
              "disk_free_after_bytes": shutil.disk_usage(hub).free}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(args.output)
    print(json.dumps({"status": "PASS_LOCAL_IO_VIEW_EXACT_BYTES", "files": len(files), "view": str(view)}))


if __name__ == "__main__":
    main()
