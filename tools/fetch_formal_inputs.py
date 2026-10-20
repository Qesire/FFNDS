"""Recover the frozen 20 formal inputs, retaining historical identity limits.

Only the named 16:9 crops are read from the pinned HF mirror archive. The
historical handoff records no image SHA; prompt/path agreement is not evidence
of old image byte identity. A new full input identity is frozen for paired runs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

from huggingface_hub import hf_hub_download
from PIL import Image

from ar2_checkpoint import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "lxl-158/vbench-i2v"
REVISION = "577ba45c1653ede26e29996737f42841f2ec9eea"
ARCHIVE_FILE = "vbench2_beta_i2v/data.tar.gz"
ARCHIVE_SHA = "aa6af89775242f35dbcd1f4c53421a6547d0700b1a99206e98b87a28ea393865"
REFERENCES = {"i2v_subject": "4434d2f545e8115f56027e50da8b6d68aff19a666c80070f681143beecdb5064",
              "i2v_background": "9883ea70b2ecf375ff8b5d0bd21301f66f27afbb9a5a722f8352b7020b4c2302"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff-root", type=Path, default=Path.home() / ".cache/ffnds/handoff-20261020/Wan22_FFN_HANDOFF_20261020")
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / ".runtime/inputs/formal20")
    args = ap.parse_args()
    records, source_records = [], []
    for dimension, expected in REFERENCES.items():
        source = args.handoff_root / f"evidence/context_20261011/metadata/formal_manifests/{dimension}_start10_n10.json"
        if digest(source) != expected:
            raise RuntimeError("Historical formal manifest changed")
        rows = json.loads(source.read_text())
        if len(rows) != 10 or [r["filtered_index"] for r in rows] != list(range(10, 20)):
            raise RuntimeError("Expected original filtered slice10:20")
        for row in rows:
            if row["dimension"] != dimension or row["seed"] != 42 or row["negative_prompt"] != "" or hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
                raise RuntimeError("Formal dimension/seed/prompt changed")
            if not row["image_path"].endswith("/crop/16-9/" + row["image_name"]):
                raise RuntimeError("Historical input was not the expected 16:9 crop")
            records.append(row)
        source_records.append({"path": str(source), "sha256": expected})
    names = {r["image_name"] for r in records}
    if len(names) != 20 or len({r["source_group"] for r in records}) != 20:
        raise RuntimeError("Need 20 distinct formal sources and images")
    archive = args.archive or Path(hf_hub_download(REPOSITORY, ARCHIVE_FILE, revision=REVISION))
    if archive.stat().st_size != 2607670436 or digest(archive) != ARCHIVE_SHA:
        raise RuntimeError("Pinned image archive full checksum mismatch")
    hf_home = Path.home() / ".cache/huggingface"
    images = hf_home / "ffnds-derived/vbench-i2v" / REVISION / "crop/16-9"
    images.mkdir(parents=True, exist_ok=True)
    found = {}
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 3 or parts[-3:-1] != ("crop", "16-9") or parts[-1] not in names:
                continue
            name = parts[-1]
            if name in found or not member.isfile() or not 0 < member.size < 64 << 20:
                raise RuntimeError("Unexpected duplicate/type/size in archive")
            content = handle.extractfile(member).read((64 << 20) + 1)
            if len(content) != member.size:
                raise RuntimeError("Truncated image archive member")
            with Image.open(io.BytesIO(content)) as image:
                size = list(image.size)
                image.verify()
            if abs(size[0] / size[1] - 16 / 9) > .02:
                raise RuntimeError("Recovered image is not the registered crop ratio")
            path = images / name
            sha = hashlib.sha256(content).hexdigest()
            if path.exists():
                if digest(path) != sha:
                    raise RuntimeError("Existing derived cached crop differs")
            else:
                temp = Path(str(path) + ".tmp")
                temp.write_bytes(content)
                temp.replace(path)
            found[name] = {"path": str(path), "sha256": sha, "bytes": len(content), "size": size, "archive_member": member.name}
    if set(found) != names:
        raise RuntimeError(f"Formal images absent from pinned mirror: {sorted(names - set(found))}")
    if len({v["sha256"] for v in found.values()}) != 20:
        raise RuntimeError("Distinct formal names have duplicate image bytes")
    calibration_path = ROOT / ".runtime/inputs/calibration12/recovery_report.json"
    calibration = json.loads(calibration_path.read_text())
    if calibration["status"] != "PASS_EXACT_CALIBRATION_INPUT_RECOVERY":
        raise RuntimeError("Verified calibration input identity required")
    overlap = {v["sha256"] for v in found.values()} & {r["image_sha256"] for r in calibration["records"]}
    if overlap:
        raise RuntimeError("Formal input image overlaps the quantization calibration set")
    args.out.mkdir(parents=True, exist_ok=True)
    frozen, index = [], []
    for number, row in enumerate(records):
        image = found[row["image_name"]]
        provenance = {"repository": REPOSITORY, "revision": REVISION, "archive_sha256": ARCHIVE_SHA,
                      "image_sha256": image["sha256"], "archive_member": image["archive_member"],
                      "historical_image_sha256": None,
                      "identity_limit": "historical image SHA absent; original prompt/name/crop16:9 agree, old byte identity unverified"}
        record = {**row, "task_id": 10000 + number, "role": "formal_evaluation", "split": "formal_evaluation",
                  "image_path": image["path"], "input_provenance": provenance}
        path = args.out / f"input_{number:02d}_seed42.json"
        atomic_json(path, [record])
        frozen.append(record)
        index.append({"input_manifest": str(path.resolve()), "sha256": digest(path), "source_group": row["source_group"], **image})
    atomic_json(args.out / "canonical.json", frozen)
    atomic_json(args.out / "recovery_report.json", {"status": "PASS_NEW_FORMAL20_INPUT_RECOVERY", "references": source_records,
        "archive": str(archive), "archive_sha256": ARCHIVE_SHA, "canonical_sha256": digest(args.out / "canonical.json"),
        "unique_inputs": 20, "seed": 42, "camera_excluded": True, "records": index,
        "calibration_report_sha256": digest(calibration_path), "calibration_image_sha_overlap": [],
        "official_image_suite_guidance": "https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v",
        "mirror_is_third_party": True, "historical_byte_identity": "UNVERIFIED", "quality_acceptance": False})
    print(json.dumps({"status": "PASS_NEW_FORMAL20_INPUT_RECOVERY", "unique_inputs": 20}))


if __name__ == "__main__":
    main()
