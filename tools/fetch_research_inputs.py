"""Recover the three existing V5 sources from a pinned public dataset snapshot.

Historical image hashes were not saved in the handoff manifests. Matching path,
prompt and byte count is recorded honestly; it does not prove historical image
byte identity. Images and the source pool are cached using huggingface_hub.
"""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DATASET = "BestWishYsh/OpenS2V-Eval"
REVISION = "b1e792b1853e5cbffdb7e7cc0d77a2bea7008e9a"
SOURCES = [(0, "fit", "task_000_seed1_source.json"),
           (6, "validation", "task_006_seed_20261001.json"),
           (10, "development", "task_010_seed_20261001.json")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff-root", type=Path,
                    default=Path.home() / ".cache/ffnds/handoff-20261020/Wan22_FFN_HANDOFF_20261020")
    ap.add_argument("--out", type=Path, default=ROOT / ".runtime/inputs")
    args = ap.parse_args()
    refs = args.handoff_root / "evidence/recent_outputs/V5_20261013/asset_collection/R1/manifests"
    info = HfApi().dataset_info(DATASET, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    metadata = {item.rfilename: item for item in info.siblings}
    pool_path = Path(hf_hub_download(DATASET, "Open-Domain_Eval.json", revision=REVISION, repo_type="dataset"))
    pool = json.loads(pool_path.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    recovered = []
    for task, split, filename in SOURCES:
        reference = refs / filename
        rows = json.loads(reference.read_text())
        assert len(rows) == 1
        row = rows[0]
        public = pool[row["source_record_key"]]
        assert public["prompt"] == row["prompt"], "Public prompt differs from handoff"
        image_name = public["img_paths"][0]
        assert row["image_path"].endswith("/" + image_name)
        image = Path(hf_hub_download(DATASET, image_name, revision=REVISION, repo_type="dataset"))
        assert image.stat().st_size == row["image_bytes"] == metadata[image_name].size
        sha = hashlib.sha256(image.read_bytes()).hexdigest()
        if metadata[image_name].lfs:
            assert sha == metadata[image_name].lfs.sha256
        provenance = {"dataset": DATASET, "revision": REVISION, "image_name": image_name,
                      "image_sha256": sha, "historical_manifest": str(reference),
                      "prompt_matches_handoff": True, "image_bytes_match_handoff": True,
                      "historical_image_sha256": None,
                      "identity_limit": "historical image hash absent; path/prompt/size match only"}
        for seed in [20261001, 20261002]:
            record = {"task_id": task, "split": split, "role": split,
                      "source_group": row["source_group"], "videoid": row["videoid"],
                      "prompt": row["prompt"], "image_path": str(image), "seed": seed,
                      "input_provenance": provenance}
            target = args.out / f"task_{task:03d}_seed_{seed}.json"
            target.write_text(json.dumps([record], indent=2, ensure_ascii=False) + "\n")
            recovered.append({"path": str(target), "task_id": task, "seed": seed, **provenance})
    (args.out / "recovery_manifest.json").write_text(json.dumps({"status": "PASS_PUBLIC_RECOVERY",
        "independent_source_count": 3, "records": recovered}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS_PUBLIC_RECOVERY", "sources": 3, "records": len(recovered)}))


if __name__ == "__main__":
    main()
