#!/usr/bin/env python3
"""Structural preflight for FFNDS method-ranking evidence, not video acceptance.
Must be followed by source-bank, native receipt, full-frame metric and cost audits.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

SITES = {f"{step}:{block}" for step in range(4) for block in range(40)}
AXES = ("imaging_quality", "aesthetic_quality", "motion_smoothness",
        "continuous_raft", "subject_consistency", "background_consistency",
        "semantic_action", "camera_motion", "temporal_behavior")
ALWAYS = {"imaging_quality", "aesthetic_quality", "motion_smoothness",
          "continuous_raft", "temporal_behavior"}
STRATA = {"subject_motion", "background", "static", "camera_motion", "mixed"}
LINKS = ("dense_run_locator", "candidate_run_locator", "dense_video_locator",
         "candidate_video_locator", "video_metric_report_locator", "cost_record_locator")
COST = ("dense_ffn_seconds", "candidate_ffn_seconds", "dense_generate_seconds",
        "candidate_generate_seconds")

def good_id(v):
    return isinstance(v, str) and bool(v.strip())

def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def validate(m):
    errs = []
    phase = m.get("phase")
    if m.get("schema") != "ffnds.video_first_candidate_bundle.v1":
        errs.append("invalid schema")
    if phase not in ("development", "confirmation"):
        errs.append("invalid phase")
    if not good_id(m.get("numeric_recipe")) or not good_id(m.get("method_version")):
        errs.append("missing method_version or numeric_recipe")
    rows = m.get("sources")
    if not isinstance(rows, list):
        return ["sources must be array", *errs], {}
    needed = 20 if phase == "development" else 48
    if phase == "confirmation" and "confirmation_source_target" in m:
        target = m.get("confirmation_source_target")
        if (not isinstance(target, int) or isinstance(target, bool) or target < 24
                or not good_id(m.get("source_target_change_reason"))
                or m.get("protocol_frozen_before_outcomes") is not True):
            errs.append("alternative confirmation target needs pre-frozen target >=24 and reason")
        else:
            needed = target
    if len(rows) < needed:
        errs.append(f"need >= {needed} independent image clusters, have {len(rows)}")
    ids = [r.get("image_cluster_id") if isinstance(r, dict) else None for r in rows]
    if any(not good_id(i) for i in ids) or len(set(i for i in ids if good_id(i))) != len(ids):
        errs.append("duplicate or missing image_cluster_id")
    strata = Counter()
    count_samples = 0
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            errs.append(f"source[{i}] not object")
            continue
        stratum = row.get("scene_stratum")
        if stratum not in STRATA:
            errs.append(f"source[{i}] invalid scene_stratum")
        else:
            strata[stratum] += 1
        if phase == "confirmation" and row.get("exposure_history") != "UNEXPOSED":
            errs.append(f"source[{i}] was exposed")
        samples = row.get("paired_samples")
        if not isinstance(samples, list) or not samples:
            errs.append(f"source[{i}] has no genuine paired videos")
            continue
        if phase == "confirmation" and len(samples) < 5:
            errs.append(f"source[{i}] requires >=5 independent generated paired seeds")
        seen_seeds = set()
        for j, v in enumerate(samples):
            count_samples += 1
            mark = f"source[{i}]/seed[{j}]"
            if not isinstance(v, dict):
                errs.append(mark+" missing sample object")
                continue
            seed = v.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed in seen_seeds:
                errs.append(mark+" invalid/repeated seed")
            else:
                seen_seeds.add(seed)
            if v.get("frames") != 81 or set(v.get("candidate_site_ids", [])) != SITES or len(v.get("candidate_site_ids", [])) != 160:
                errs.append(mark+" incomplete 81-frame / 4step x 40block = 160-site coverage")
            if v.get("matched_input_and_noise") is not True or v.get("same_numeric_recipe") is not True:
                errs.append(mark+" must match Dense image/prompt/noise and recipe")
            for key in LINKS:
                if not good_id(v.get(key)):
                    errs.append(mark+" missing "+key)
            metrics = v.get("metrics")
            if not isinstance(metrics, dict):
                errs.append(mark+" missing metric vector")
                continue
            for axis in AXES:
                item = metrics.get(axis)
                if isinstance(item, dict) and all(number(item.get(k)) for k in ("dense", "candidate")):
                    continue
                applicable = axis in ALWAYS or (axis == "semantic_action" and stratum in ("subject_motion", "mixed")) or (axis == "camera_motion" and stratum in ("camera_motion", "mixed"))
                if not applicable and isinstance(item, dict) and item.get("status") == "NOT_APPLICABLE" and good_id(item.get("reason")):
                    continue
                errs.append(mark+" missing required source-relevant axis: "+axis)
            cost = v.get("measured_cost")
            if not isinstance(cost, dict) or any(not number(cost.get(k)) or cost[k] < 0 for k in COST):
                errs.append(mark+" missing full-model FFN/wall costs")
    if not (strata["subject_motion"] or strata["mixed"]):
        errs.append("no moving-subject scenario")
    if not (strata["background"] or strata["static"] or strata["mixed"]):
        errs.append("no background/static scenario")
    if not (strata["camera_motion"] or strata["mixed"]):
        errs.append("no camera-motion scenario")
    if phase == "confirmation" and m.get("protocol_frozen_before_outcomes") is not True:
        errs.append("confirmation protocol not pre-frozen")
    return errs, {
        "source_clusters": len(set(i for i in ids if good_id(i))),
        "strata": dict(strata), "paired_samples": count_samples,
        "quality_verified": False, "runtime_verified": False,
        "meaning": "structural preflight only; no scientific success verdict"
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", type=Path)
    opts = p.parse_args()
    obj = json.loads(opts.manifest.read_text(encoding="utf-8"))
    errors, summary = validate(obj)
    result = {"status": "STRUCTURE_ONLY_PASS" if not errors else "STRUCTURE_FAIL",
              "errors": errors, "summary": summary}
    serialized = json.dumps(result, ensure_ascii=False, indent=2)+"\n"
    if opts.out:
        opts.out.parent.mkdir(parents=True, exist_ok=True)
        opts.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 2 if errors else 0

if __name__ == "__main__":
    sys.exit(main())
