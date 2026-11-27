#!/usr/bin/env python3
"""Freeze the 50-source CUSTOM_PAIRED cohort using metadata, never model outputs.

Candidates: a JSON list or {"sources": [...]} with source_id,
source_cluster_id, image_path, image_sha256, prompt, prompt_sha256, category,
and original_dimension_tags. image_name defaults to the image basename.

Registry: {"as_of": "...", "calibration_cluster_ids": [...],
"development_cluster_ids": [...], "exposed_cluster_ids": [...]}.
For each prefix, *_source_ids and *_image_sha256 lists are also accepted.
hard_excluded_* is an additional explicit hard-exclusion prefix.
Mandatory calibration/development lists must be supplied, even when empty.

Usage:
  python prepare_cohort.py --candidates PRIVATE_CANDIDATES.json \
    --exclusions CURRENT_REGISTRY.json --method-binding FROZEN_METHOD_BINDING.json \
    --output-dir cohort

The output cohort.json contains private locators and prompt text. Publish only
cohort_public.json. A completed output requires actual selected image bytes.
"""

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL = "FFNDS-centered-medium-20261127"
STRATA = {
    "structured": {"categories": {"indoor", "architecture", "abstract"}, "count": 17},
    "natural": {"categories": {"scenery", "plant"}, "count": 16},
    "subject": {"categories": {"single-human", "multiple-human", "animal", "transportation", "food", "other"}, "count": 17},
}
DIMS = [
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "motion_smoothness", "imaging_quality", "i2v_subject", "i2v_background",
]


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def key(purpose, identity):
    return sha_bytes(f"{PROTOCOL}-{purpose}-v1|{identity}".encode("utf-8"))


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(candidates_path, registry_path, method_path, output_dir):
    raw_candidates = candidates_path.read_bytes()
    raw_registry = registry_path.read_bytes()
    raw_method = method_path.read_bytes()
    payload = json.loads(raw_candidates)
    rows = payload if isinstance(payload, list) else payload["sources"]
    registry = json.loads(raw_registry)
    method = json.loads(raw_method)
    for field in ("as_of", "calibration_cluster_ids", "development_cluster_ids"):
        if field not in registry:
            raise ValueError(f"Current exclusion registry must explicitly supply {field}")
    if not method.get("candidate_id"):
        raise ValueError("Method binding requires candidate_id")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Candidates must contain nonempty sources")
    seen_source_ids = set()
    normalized = []
    for original in rows:
        row = dict(original)
        for field in ("source_id", "source_cluster_id", "image_path", "image_sha256",
                      "prompt", "prompt_sha256", "category", "original_dimension_tags"):
            if field not in row:
                raise ValueError(f"Candidate {row.get('source_id')} is missing {field}")
        for field in ("source_id", "source_cluster_id"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"{field} must be a nonempty string; preserve zero padding")
        if row["source_id"] in seen_source_ids:
            raise ValueError(f"Duplicate source_id: {row['source_id']}")
        seen_source_ids.add(row["source_id"])
        if not isinstance(row["prompt"], str):
            raise ValueError("Prompt must be the unmodified string")
        if sha_bytes(row["prompt"].encode("utf-8")) != row["prompt_sha256"]:
            raise ValueError(f"Prompt hash mismatch: {row['source_id']}")
        image_hash = row["image_sha256"]
        if (not isinstance(image_hash, str) or len(image_hash) != 64
                or any(c not in "0123456789abcdef" for c in image_hash)):
            raise ValueError(f"Expected lowercase image SHA256: {row['source_id']}")
        if not isinstance(row["original_dimension_tags"], list):
            raise ValueError("original_dimension_tags must be a list (empty is allowed)")
        normalized.append(row)

    # Union both declared clusters and byte-identical images. A cluster can
    # contain several crops; another declared ID for an identical crop is an alias.
    parent = list(range(len(normalized)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for field in ("source_cluster_id", "image_sha256"):
        seen = {}
        for i, row in enumerate(normalized):
            value = row[field]
            if value in seen:
                union(i, seen[value])
            else:
                seen[value] = i
    groups = {}
    for i, row in enumerate(normalized):
        groups.setdefault(find(i), []).append(row)

    def registered(members, prefixes):
        for prefix in prefixes:
            for suffix, field in (("cluster_ids", "source_cluster_id"),
                                  ("source_ids", "source_id"),
                                  ("image_sha256", "image_sha256")):
                values = set(registry.get(f"{prefix}_{suffix}", []))
                if any(m[field] in values for m in members):
                    return True
        return False

    pools = {name: [] for name in STRATA}
    hard_excluded = 0
    category_excluded = 0
    for members in groups.values():
        if registered(members, ("calibration", "development", "hard_excluded")):
            hard_excluded += 1
            continue
        representative = dict(min(members, key=lambda r: r["source_id"]))
        stratum = next((name for name, spec in STRATA.items()
                        if representative["category"] in spec["categories"]), None)
        if stratum is None:
            category_excluded += 1
            continue
        aliases = sorted({r["source_cluster_id"] for r in members})
        representative["source_cluster_id"] = aliases[0]
        representative["declared_source_cluster_ids"] = aliases
        representative["stratum"] = stratum
        representative["prior_exposure"] = registered(members, ("exposed",))
        representative["selection_sha256"] = key("cohort", aliases[0])
        pools[stratum].append(representative)

    chosen = []
    pool_summary = {}
    for name, spec in STRATA.items():
        pool = sorted(pools[name], key=lambda r: (
            r["prior_exposure"], r["selection_sha256"], r["source_cluster_id"]))
        if len(pool) < spec["count"]:
            raise ValueError(f"Insufficient {name} clusters: {len(pool)} < {spec['count']}; "
                             "do not change quotas after seeing outcomes")
        chosen.extend(pool[:spec["count"]])
        pool_summary[name] = {
            "eligible_clusters": len(pool),
            "unexposed_clusters": sum(not r["prior_exposure"] for r in pool),
            "selected_clusters": spec["count"],
            "selected_exposed_clusters": sum(r["prior_exposure"] for r in pool[:spec["count"]]),
        }
    chosen.sort(key=lambda r: (r["selection_sha256"], r["source_cluster_id"]))

    verified = {}
    for row in chosen:
        image = Path(row["image_path"]).expanduser()
        if not image.is_absolute():
            image = candidates_path.parent / image
        image = image.resolve(strict=True)
        actual = verified.setdefault(str(image), None)
        if actual is None:
            actual = verified[str(image)] = digest_file(image)
        if actual != row["image_sha256"]:
            raise ValueError(f"Image bytes mismatch: {row['source_id']}")
        row["image_path"] = str(image)
        row["image_name"] = row.get("image_name") or image.name
        row["seed"] = 42
        row["custom_dimensions"] = list(DIMS)
        row["image_bytes_verified"] = True

    def subset(purpose, count_per_stratum):
        selected = set()
        for name in STRATA:
            eligible = [r for r in chosen if r["stratum"] == name]
            eligible.sort(key=lambda r: (key(purpose, r["source_cluster_id"]), r["source_id"]))
            selected.update(r["source_id"] for r in eligible[:count_per_stratum])
        # Preserve the main source order for deterministic source-index balancing.
        return [r["source_id"] for r in chosen if r["source_id"] in selected]

    timing_ids = subset("timing", 4)
    review_ids = subset("review", 8)
    for row in chosen:
        row["timing_selected"] = row["source_id"] in timing_ids
        row["blind_review_selected"] = row["source_id"] in review_ids
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": "INPUTS_VERIFIED_COHORT_FROZEN_NO_MODEL_RUN",
        "mode": "CUSTOM_PAIRED",
        "candidate_id": method["candidate_id"],
        "method_binding_sha256": sha_bytes(raw_method),
        "research_commit": method.get("research_commit"),
        "seed": 42,
        "arms": ["D", "W", "M"],
        "geometry": {"width": 832, "height": 464, "frames": 81, "fps": 16},
        "input_sha256": {"candidates": sha_bytes(raw_candidates), "exposure_registry": sha_bytes(raw_registry)},
        "exposure_registry_as_of": registry["as_of"],
        "selection": {
            "candidate_rows": len(normalized),
            "independent_clusters_after_alias_merge": len(groups),
            "hard_excluded_clusters": hard_excluded,
            "outside_three_strata_clusters": category_excluded,
            "canonical_representative": "lexicographically_smallest_source_id",
            "stratum_assignment": "canonical_representative_category",
            "ordering": "unexposed_first_then_sha256(salt|canonical_cluster_id)",
            "salt": f"{PROTOCOL}-cohort-v1",
            "strata": pool_summary,
            "selected_n_by_category": {category: sum(r["category"] == category for r in chosen)
                                       for category in sorted(set().union(*(s["categories"] for s in STRATA.values())))},
        },
        "timing_source_ids": timing_ids,
        "blind_review_source_ids": review_ids,
        "sources": chosen,
    }
    # Explicit allow-list prevents private fields inherited from candidates from
    # leaking into the public manifest.
    public_fields = (
        "source_id", "source_cluster_id", "declared_source_cluster_ids", "image_name",
        "image_sha256", "prompt_sha256", "category", "original_dimension_tags", "stratum",
        "prior_exposure", "selection_sha256", "seed", "custom_dimensions",
        "image_bytes_verified", "timing_selected", "blind_review_selected",
    )
    public = dict(manifest)
    public["sources"] = [{k: row[k] for k in public_fields} for row in chosen]
    public["redacted_fields"] = ["image_path", "prompt", "other_private_candidate_metadata"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("cohort.json", manifest), ("cohort_public.json", public)):
        target = output_dir / name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite frozen cohort: {target}")
    for name, value in (("cohort.json", manifest), ("cohort_public.json", public)):
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return {"sources": len(chosen), "timing_sources": len(timing_ids),
            "blind_review_sources": len(review_ids), "strata": pool_summary,
            "status": manifest["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--method-binding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.candidates, args.exclusions, args.method_binding, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
