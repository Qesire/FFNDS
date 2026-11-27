"""Bind the frozen school VBench metadata to its actual 16:9 crop catalog."""
import argparse
import hashlib
import json
from pathlib import Path


INPUTS = {
    "g0a_clusters.json": "b882383262a6b795c58370a5d0962d289539e6f20de352f1a512bb595692465e",
    "vbench_metadata.json": "8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8",
}


def load(archive, name):
    data = (archive / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != INPUTS[name]:
        raise ValueError(f"Unrecognized frozen input {name}")
    return json.loads(data)


def materialize(archive, crops, output):
    official = load(archive, "vbench_metadata.json")
    clusters = load(archive, "g0a_clusters.json")["clusters"]
    by_image = {}
    for cluster in clusters:
        for item in cluster["members"]:
            path = Path(item["path"])
            if path.parent != crops or item["variant"] != "vbench_crop169" or not path.is_file():
                continue
            info = (cluster["cluster_id"], item["sha256"], str(path))
            if by_image.setdefault(path.name, info) != info:
                raise ValueError(f"Ambiguous frozen catalog entry {path.name}")
    candidates = []
    for i, row in enumerate(official):
        name = row["image_name"]
        if name not in by_image:
            continue
        cluster, digest, path = by_image[name]
        prompt = row["prompt_en"]
        candidates.append(dict(source_id=f"vbi_{i:04d}", source_cluster_id=cluster, category=row["image_type"],
            image_name=name, image_path=path, image_sha256=digest, prompt=prompt,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(), original_dimension_tags=row["dimension"]))
    if not candidates:
        raise ValueError("Frozen metadata did not bind to any existing 16:9 crop")
    output.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("CATALOG", len(candidates), len({r["source_cluster_id"] for r in candidates}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "crops", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    materialize(args.archive, args.crops, args.out)
