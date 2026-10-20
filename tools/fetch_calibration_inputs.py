"""Recover the original 12 HiF4 calibration inputs, including image hashes.

This rebuilds a new cache with the old frozen input list. It does not recover
the old quantized bytes. The original list includes the task10 development
image; that source must never be counted as independent validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from huggingface_hub import hf_hub_download

from fetch_research_inputs import DATASET, REVISION, ROOT

SOURCE = ROOT / ".runtime/archive/server/artifacts/qap3b_diagnostic_return/QAP3B_G92_DIAGNOSTIC_20260826T013901Z/payload/06_data_candidates/small_manifests/home__scc__pb23061276__projects__wan-quant__data__OpenS2V-Eval__calib_12_manifest.jsonl"
REFERENCE = ROOT / ".runtime/archive/server/experiments/hifx4_shared_assets_lightx2v_distill_v4_9_2/manifests/calib_12_canonical.report.json"
REFERENCE_SHA256 = "9774c04f4b18bbba6dcaf55b9938032e338f1d34d5e6a178bb0cf523fdf51992"
SOURCE_SHA256 = "fa2514bf667bcad42f8ffdefd73c55ecb859f93014b6abb2366045f4b3a06008"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_reference_files(archive):
    """Read only two named regular files and verify content before any write."""
    required = [(SOURCE, SOURCE_SHA256), (REFERENCE, REFERENCE_SHA256)]
    if all(path.is_file() for path, expected in required):
        return
    with tarfile.open(archive, "r:gz") as handle:
        for path, expected in required:
            if path.exists():
                continue
            member = handle.getmember(str(path.relative_to(ROOT / ".runtime/archive")))
            if not member.isfile() or member.size > 4 << 20:
                raise RuntimeError("Unexpected archived calibration reference type/size")
            with handle.extractfile(member) as source:
                content = source.read((4 << 20) + 1)
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError("Archived calibration reference checksum differs")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--server-payload", type=Path, default=Path.home() / ".cache/ffnds/handoff-20261020/Wan22_FFN_HANDOFF_20261020/server_payload.tar.gz")
    ap.add_argument("--out", type=Path, default=ROOT / ".runtime/inputs/calibration12")
    args = ap.parse_args()
    if args.source == SOURCE and args.reference == REFERENCE:
        restore_reference_files(args.server_payload)
    if sha(args.source) != SOURCE_SHA256 or sha(args.reference) != REFERENCE_SHA256:
        raise RuntimeError("Expected the pinned archived calibration list and report")
    ref = json.loads(args.reference.read_text())
    if sha(args.source) != ref["source_sha256"]:
        raise RuntimeError("Archived calibration list differs from its canonicalization report")
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    if len(rows) != ref["canonical_record_count"] or len(rows) != 12:
        raise RuntimeError("Expected exactly the original 12 calibration inputs")
    pool_path = Path(hf_hub_download(DATASET, "Open-Domain_Eval.json", revision=REVISION, repo_type="dataset"))
    pool = json.loads(pool_path.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    canonical, records = [], []
    for index, row in enumerate(rows):
        name = row["meta"]["image_rel"]
        public = pool[row["meta"]["sample_id"]]
        if public["prompt"] != row["prompt"] or public["img_paths"][0] != name:
            raise RuntimeError(f"Public calibration identity differs at input {index}")
        if not ref["resolved_image_paths"][index].endswith("/" + name):
            raise RuntimeError(f"Archived calibration ordering differs at input {index}")
        path = Path(hf_hub_download(DATASET, name, revision=REVISION, repo_type="dataset"))
        actual = sha(path)
        if actual != ref["image_sha256"][index]:
            raise RuntimeError(f"Historical calibration image hash differs at input {index}")
        item = {"videoid": ref["ordered_videoids"][index], "prompt": row["prompt"], "img_paths": [str(path)],
                "source_index": index, "calibration_seed": 42 + index,
                "input_provenance": {"dataset": DATASET, "revision": REVISION, "image_name": name,
                    "image_sha256": actual, "historical_image_sha256_matches": True,
                    "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    "source_sample_id": row["meta"]["sample_id"]}}
        canonical.append(item)
        records.append({"index": index, "image_name": name, "image_sha256": actual,
                        "source_sample_id": row["meta"]["sample_id"], "seed": 42 + index})
        print(f"Verified calibration {index + 1}/12: {name}", flush=True)
    research = json.loads((ROOT / ".runtime/inputs/recovery_manifest.json").read_text())
    hashes = {r["image_sha256"] for r in records}
    overlaps = [{"task_id": r["task_id"], "seed": r["seed"], "image_name": r["image_name"]}
                for r in research["records"] if r["image_sha256"] in hashes]
    if any(row["task_id"] in [0, 6] for row in overlaps):
        raise RuntimeError("Calibration overlaps the frozen task0/task6 research comparison")
    target = args.out / "canonical.json"
    text = json.dumps(canonical, indent=2, ensure_ascii=False) + "\n"
    if target.exists() and target.read_text() != text:
        raise RuntimeError("Refusing to overwrite a changed calibration manifest")
    target.write_text(text)
    report = {"status": "PASS_EXACT_CALIBRATION_INPUT_RECOVERY", "num_samples": 12,
              "series": "WORKER_HIF4_SMOOTH_RTN_REBUILD_20261020", "canonical": str(target.resolve()),
              "canonical_sha256": sha(target), "source_sha256": sha(args.source),
              "reference_sha256": sha(args.reference), "dataset": DATASET, "revision": REVISION,
              "research_overlaps": overlaps, "task10_is_independent_validation": False,
              "old_DQ_cache_byte_identity": "NOT_CLAIMED_NEW_CALIBRATION_RUN_AND_CACHE", "records": records}
    (args.out / "recovery_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "samples": 12, "research_overlaps": overlaps}))


if __name__ == "__main__":
    main()
