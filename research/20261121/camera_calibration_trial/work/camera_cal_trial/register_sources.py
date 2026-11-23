"""Register teacher prompts and source-disjoint validation before any model output."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "anchored_scale"))
import cohort_build as archive


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    provenance = json.loads((HERE.parent / "anchored_scale/cohort_provenance.json").read_bytes())
    expanded = json.loads((HERE.parent / "anchored_scale/cohort.json").read_bytes())
    digests = provenance["sha256_inputs"]
    blob = subprocess.check_output(archive.SSH + ["cd " + archive.ARCH + " && tar cf - " + " ".join(digests)], timeout=120)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        records = {name: tar.extractfile(name).read() for name in digests}
    assert {k: sha(data) for k, data in records.items()} == digests
    meta = json.loads(records["vbench_metadata.json"])
    clusters = json.loads(records["g0a_clusters.json"])["clusters"]
    bound = json.loads(records["output/VB_BOUND_INPUTS.json"])["records"]
    excluded = {r["payload_binding"]["source_cluster_id"] for r in bound} | {r["source_cluster_id"] for r in expanded}
    files = {PurePosixPath(m["path"]).name: (c["cluster_id"], m) for c in clusters for m in c["members"]
             if str(PurePosixPath(m["path"]).parent) == archive.ROOT and m["width"] > m["height"]}
    moving_words = {"running", "jumping", "dancing", "riding", "climbing", "swimming", "walking"}
    train = {"vbi_0032": "camera zooms in", "vbi_0304": "camera pans left", "vbi_0456": "camera pans right"}
    sources, candidates = [], []
    for i, item in enumerate(meta):
        if "camera_motion" in item["dimension"] or item["image_name"] not in files:
            continue
        cluster, entry = files[item["image_name"]]
        source = dict(record_id=f"vbi_{i:04d}", source_cluster_id=cluster, category=item["image_type"],
                      image_path=entry["path"], image_sha256=entry["sha256"], prompt=item["prompt_en"])
        if source["record_id"] in train:
            sources.append(dict(source, split="calibration", clause=train[source["record_id"]]))
        elif (cluster not in excluded and source["category"] in {"animal", "single-human"}
              and set(source["prompt"].lower().split()) & moving_words):
            candidates.append(source)
    assert len(sources) == 3
    for category in ("animal", "single-human"):
        subset = [r for r in candidates if r["category"] == category and r["source_cluster_id"] not in excluded]
        row = min(subset, key=lambda r: sha(("ffnds-camera-cal-v1|" + r["source_cluster_id"] + "|" + r["record_id"]).encode()))
        excluded.add(row["source_cluster_id"])
        sources.append(dict(row, split="validation", clause="camera pans left"))
    for key in ("image_sha256", "source_cluster_id", "record_id", "prompt"):
        assert len({r[key] for r in sources}) == 5
    assert not any(r["record_id"] == "vbi_0951" for r in sources)
    trajectories = []
    for r in sources:
        grid = ((42,"base",["A","B"]),(43,"base",["A"]),(43,"camera",["B"])) if r["split"] == "calibration" else ((44,"base",[]),(44,"camera",[]))
        for seed, condition, arms in grid:
            prompt = r["prompt"] if condition == "base" else r["prompt"] + ", " + r["clause"]
            trajectories.append(dict(trajectory_id=f"{r['record_id']}_{condition}_s{seed}", source_id=r["record_id"],
                                     split=r["split"], condition=condition, seed=seed, arms=arms, prompt=prompt,
                                     prompt_sha256=sha(prompt.encode()), image_path=r["image_path"], image_sha256=r["image_sha256"]))
    assert len(trajectories) == 13
    protocol = dict(trial="FFNDS-camera-cal-v1", layers=[0,13,26,39], expert="high", sources=sources,
                    trajectories=trajectories, historical_input_sha256=digests, excluded_cluster_count=len({r["payload_binding"]["source_cluster_id"] for r in bound}|{r["source_cluster_id"] for r in expanded}),
                    validation_selection="Before rendering, one animal and one single-human source with an action word, ordered by SHA-256 over frozen salt, cluster and record. Both validation sources pair their base prompt with camera pans left. Screening used input metadata only.",
                    shrinkage="Hold out all prompts and seeds for a source together. Use equal weight per calibration source and per trajectory within source.",
                    comparison="A and B each train on six trajectories. Keep frozen features, source weights, point/spatial loss and probe budget fixed. Independently report temporal signed projection, orthogonal error, source and condition. Two validation images form a pilot.")
    target = HERE / "protocol.json"
    data = (json.dumps(protocol, ensure_ascii=False, indent=2) + "\n").encode()
    if target.exists() and target.read_bytes() != data:
        raise RuntimeError("A different protocol is already registered.")
    target.write_bytes(data)
    print(sha(data), [(r["record_id"],r["split"],r["clause"]) for r in sources])


if __name__ == "__main__":
    main()
