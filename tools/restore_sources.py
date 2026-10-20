"""Restore audited upstream ZIPs into a new runtime directory, then apply deltas.

Archives are downloaded into ~/.cache/ffnds/upstream. Existing source trees are
never extracted over; restore_delta.py checks their contents before any write.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("LightX2V", "ModelTC/LightX2V", "ed146db64d0b6130c669285fc1aa0b280a6a4823",
     "389e469f866da4fc3e8b4a388ac591c079a08a549bb6067c851430ea7ac89dce"),
    ("HiF4_Sp", "alvinshao0313/HiF4_Sp", "9aa2d3be6de265910731aa7c0d053e9c773cf556",
     "13b424f086ac1389bf21ce02ce10f9688e7d43e30202dadd7b89bbcd900414eb"),
    ("Wan2.2-I2V-A14B-W4A4", "CGCL-codes/Wan2.2-I2V-A14B-W4A4", "a49a935677c7f760ef368cd334801416fff5aaa3",
     "0642cc1ddba3a6cfe6180416a31d9c5d87bffb7cd7fc4ac04ad9cf5b47dc94ce"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-root", type=Path, default=ROOT / ".runtime/code")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache/ffnds/upstream")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.repos_root.mkdir(parents=True, exist_ok=True)
    for name, repo, revision, expected in SOURCES:
        archive = args.cache / f"{name}-{revision}.zip"
        if not archive.exists():
            partial = archive.with_suffix(".zip.partial")
            with urllib.request.urlopen(f"https://codeload.github.com/{repo}/zip/{revision}", timeout=120) as src:
                with partial.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            partial.replace(archive)
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Archive SHA256 mismatch: {archive}: {actual}")
        target = args.repos_root / name
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix=".restore-", dir=args.repos_root) as temp:
                with zipfile.ZipFile(archive) as z:
                    prefix = f"{repo.split('/')[-1]}-{revision}/"
                    for info in z.infolist():
                        if not info.filename.startswith(prefix):
                            raise RuntimeError(f"Unexpected ZIP member: {info.filename}")
                        rel = Path(info.filename[len(prefix):])
                        if rel.is_absolute() or ".." in rel.parts:
                            raise RuntimeError(f"Unsafe ZIP path: {info.filename}")
                        # GitHub source archives used here contain no symlinks.
                        if (info.external_attr >> 16) & 0o170000 == 0o120000:
                            raise RuntimeError(f"Unexpected ZIP symlink: {info.filename}")
                    z.extractall(temp)
                (Path(temp) / prefix.rstrip("/")).rename(target)
        print(json.dumps({"repo": repo, "revision": revision, "zip_sha256": actual, "directory": str(target)}), flush=True)
    for mode in ["--apply", "--verify-only"]:
        subprocess.run([sys.executable, str(ROOT / "upstream_audit/restore_delta.py"),
                        "--repos-root", str(args.repos_root), mode], check=True)


if __name__ == "__main__":
    main()
