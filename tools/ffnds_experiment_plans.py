"""Prepare traceable FFNDS source cohorts, calibrated row draws and real I2V run jobs.

This module is deliberately CPU-only. Outputs are *plans*, not completed evidence.
Legacy original run receipts remain unchanged. Standard-library dependencies only.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from ffnds_cohort_preflight import audit

ROLES = {"DISCOVERY", "CALIBRATION", "DEVELOPMENT", "FROZEN_CONFIRMATION", "OFFICIAL_BENCHMARK"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sources_from_legacy(specs):
    """Each spec is ROLE=canonical.json; a human must assign its actual exposure role."""
    result, prior_by_image = [], defaultdict(set)
    for spec in specs:
        role, sep, file = spec.partition("=")
        if not sep or role not in ROLES:
            raise ValueError("Legacy input must be ROLE=path with a registered data role")
        for i, row in enumerate(load(file)):
            provenance = row.get("input_provenance", {})
            image_name = row.get("image_name", provenance.get("image_name"))
            if not image_name:
                raise ValueError("Missing frozen image name; supply an explicit normalized source bank")
            origin = str(provenance.get("repository", "legacy_ffnds")) + "@" + str(provenance.get("revision", "unrecorded"))
            group = str(row.get("source_group", row.get("videoid", "")))
            if not group:
                raise ValueError("Missing source group in legacy canonical record")
            sid = origin + ":" + group
            if any(s["source_id"] == sid for s in result):
                raise ValueError("Repeated canonical source ID: " + sid)
            image_id = origin + ":image:" + image_name
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("Missing source prompt in " + file)
            image_path = row.get("image_path") or (row.get("img_paths") or [None])[0]
            if not image_path:
                raise ValueError("Missing image path: " + sid)
            fingerprint = provenance.get("image_sha256")
            prior_by_image[image_id].add(role)
            result.append({
                "source_id": sid, "image_id": image_id,
                "prompt_id": origin + ":prompt:" + prompt,
                "conditioning_id": sid + ":native_crop_condition",
                "role": role, "dataset_revision": origin,
                "scene_stratum": str(row.get("dimension", "UNCLASSIFIED")),
                "image_type": str(row.get("image_type", "UNCLASSIFIED")),
                "image_name": image_name, "image_path": str(image_path),
                "native_source_ordinal": row.get("source_index"),
                "prompt": prompt, "source_group": group,
                "legacy_image_fingerprint": fingerprint,
                "exposure_history": {
                    "fitting": role == "CALIBRATION",
                    "method_selection": role == "DEVELOPMENT",
                    "prior_evaluation": role == "DEVELOPMENT",
                },
            })
    by_fp = defaultdict(set)
    for s in result:
        if s.get("legacy_image_fingerprint"):
            by_fp[s["legacy_image_fingerprint"]].add(s["role"])
    if any(len(roles) > 1 for roles in prior_by_image.values()) or any(len(roles) > 1 for roles in by_fp.values()):
        raise ValueError("Images overlap data roles in legacy canonical inputs; resolve before planning")
    return {"schema": "ffnds.source.bank.v1", "sources": result,
            "scope": "original source records with declared historical exposure"}


def source_bank(path):
    d = load(path)
    if d.get("schema") != "ffnds.source.bank.v1" or not isinstance(d.get("sources"), list):
        raise ValueError("Expected ffnds.source.bank.v1")
    check = audit({"schema": "ffnds.cohort.v1", "sources": d["sources"],
                   "samples": [], "calibration": [], "mode": "CUSTOM_PAIRED"})
    if check["errors"]:
        raise ValueError("; ".join(check["errors"]))
    return d["sources"]


def choose(sources, role, strata, seed, *, allow_same_image_with_distinct_prompts=False):
    grouped = defaultdict(list)
    for s in sources:
        if s["role"] == role:
            grouped[s["scene_stratum"]].append(s)
    chosen = []
    for stratum_index, (stratum, n) in enumerate(sorted(strata.items())):
        candidates = sorted(grouped[stratum], key=lambda s: s["source_id"])
        rng = random.Random(seed + 1000003 * (stratum_index + 1))
        rng.shuffle(candidates)
        if n > len(candidates):
            raise ValueError(f"{stratum}: need {n} independent sources, have {len(candidates)}")
        chosen.extend(candidates[:n])
    ids = [s["image_id"] for s in chosen]
    if len(set(ids)) != len(ids) and not allow_same_image_with_distinct_prompts:
        raise ValueError("Chosen calibration sources contain image-cluster duplicates")
    return sorted(chosen, key=lambda s: s["source_id"])


def parse_strata(spec):
    out = {}
    for part in spec.split(","):
        kind, mark, count = part.partition(":")
        if not mark or not kind or kind in out or not count.isdecimal() or int(count) < 1:
            raise ValueError("strata must look like i2v_subject:6,i2v_background:6")
        out[kind] = int(count)
    return out


def row_draw(population, rows, *, base_seed, source_index, step, block, state_pair):
    if not 0 < rows <= population:
        raise ValueError("Calibration row count exceeds the actual native population")
    if step not in range(4) or block not in range(40):
        raise ValueError("Wan four-step / forty-block coordinate outside model")
    # Draws are nested for row-count ablations; paired arms share the same coordinates.
    seed = base_seed + (source_index + 1) * 1000003 + 1009 * step + 37 * block
    if not state_pair:
        seed += 700000001
    # Draw a fixed maximum prefix to preserve nested selections across 128/256/512 rows.
    # The same policy remains valid for larger designs; all must freeze row_count before sampling.
    take = min(population, max(512, rows))
    return sorted(random.Random(seed).sample(range(population), take)[:rows]), seed


def plan_calibration(bank, geometry, *, strata, rows, seed, output):
    all_sources = source_bank(bank)
    selected = choose(all_sources, "CALIBRATION", parse_strata(strata), seed)
    native_ordinals = {s["source_id"]: s.get("native_source_ordinal") for s in selected}
    if any(not isinstance(x, int) or x < 0 for x in native_ordinals.values()):
        raise ValueError("Calibration requires stable original native_source_ordinal per source")
    if len(set(native_ordinals.values())) != len(native_ordinals):
        raise ValueError("Original calibration source ordinals must be independent and unique")
    by_geometry = {}
    for x in load(geometry):
        key = (x["source_id"], x["step"], x["block"])
        if key in by_geometry:
            raise ValueError("Duplicate source-step-block geometry")
        n = x["population_rows"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("Invalid actual native population")
        by_geometry[key] = n
    result = []
    for source_index, s in enumerate(selected):
        for step in range(4):
            for block in range(40):
                key = (s["source_id"], step, block)
                if key not in by_geometry:
                    raise ValueError("Missing observed native geometry " + repr(key))
                ids, rs = row_draw(by_geometry[key], rows, base_seed=seed,
                                   source_index=native_ordinals[s["source_id"]], step=step, block=block,
                                   state_pair=True)
                result.append({"source_id": s["source_id"], "step": step, "block": block,
                               "expert": "high_noise_model" if step < 2 else "low_noise_model",
                               "source_state": "teacher_input",
                               "n_rows": rows, "population_rows": by_geometry[key],
                               "selected_row_ids": ids, "sampling_seed": rs,
                               "inclusion_probability": rows / by_geometry[key],
                               "row_draw_id": f"draw:{seed}:{source_index}:{step}:{block}"})
    value = {"schema": "ffnds.cohort.v1", "mode": "CUSTOM_PAIRED",
             "sources": selected, "samples": [], "calibration": result,
             "phase": "REGISTERED_CALIBRATION_ROW_PLAN",
             "training_rows_per_position": len(selected) * rows,
             "source_count": len(selected), "rows_per_source_per_position": rows,
             "native_scope_count": len(result), "actual_expert_steps": ["high", "high", "low", "low"]}
    checked = audit(value)
    if checked["errors"]:
        raise ValueError("; ".join(checked["errors"]))
    save_new(output, value)
    return value


def plan_video(bank, *, role, strata, seed, samples, variants, experiment,
               frames, fps, width, height, scheduler, output):
    if role not in ("DEVELOPMENT", "FROZEN_CONFIRMATION", "OFFICIAL_BENCHMARK"):
        raise ValueError("Video cohort needs declared evaluation role")
    selected = choose(source_bank(bank), role, parse_strata(strata), seed,
                      allow_same_image_with_distinct_prompts=(role == "OFFICIAL_BENCHMARK"))
    if samples not in (1, 5):
        raise ValueError("Use one real diagnostic sample or five real official samples")
    if len(variants) < 2 or "dense" not in variants or len(set(variants)) != len(variants):
        raise ValueError("Require dense and at least one unique candidate variant")
    if not experiment.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Experiment ID must use letters, digits, underscore or hyphen")
    jobs = []
    for s in selected:
        for i in range(samples):
            actual_seed = seed + i
            for variant in variants:
                job_id = f"{experiment}:{s['source_id']}:{i}:{variant}"
                jobs.append({
                    "source_id": s["source_id"], "variant": variant,
                    "seed": actual_seed, "sample_index": i,
                    "run_id": job_id, "asset_id": "video:" + job_id,
                    "conditioning_id": s["conditioning_id"],
                    "initial_noise_id": f"{experiment}:{s['source_id']}:noise:{i}",
                    "scheduler_id": scheduler,
                    "frames": frames, "fps": fps, "width": width, "height": height,
                    "status": "PLANNED", "run_receipt_path": None, "video_path": None,
                })
    value = {"schema": "ffnds.cohort.v1",
             "mode": "OFFICIAL_I2V_5SAMPLE" if samples == 5 else "CUSTOM_PAIRED",
             "phase": "REGISTERED_REAL_GENERATION_PLAN",
             "experiment": experiment, "sources": selected, "samples": jobs, "calibration": [],
             "seed_policy": "seed = base_seed + sample_index; scheduler.generator=None per request",
             "required_variants": variants}
    checked = audit(value)
    if checked["errors"]:
        raise ValueError("; ".join(checked["errors"]))
    save_new(output, value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("source-bank")
    a.add_argument("--legacy", action="append", required=True, help="ROLE=path/to/canonical.json")
    a.add_argument("--out", required=True)
    a = sub.add_parser("calibration-plan")
    a.add_argument("--bank", required=True)
    a.add_argument("--geometry", required=True, help="JSON list of actually observed source/step/block populations")
    a.add_argument("--strata", required=True)
    a.add_argument("--rows", type=int, required=True)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--out", required=True)
    a = sub.add_parser("video-plan")
    a.add_argument("--bank", required=True)
    a.add_argument("--role", required=True)
    a.add_argument("--strata", required=True)
    a.add_argument("--samples-per-source", type=int, choices=(1, 5), default=1)
    a.add_argument("--variants", nargs="+", default=["dense", "UB1"])
    a.add_argument("--experiment", required=True)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--frames", type=int, default=81)
    a.add_argument("--fps", type=float, default=16)
    a.add_argument("--width", type=int, default=832)
    a.add_argument("--height", type=int, default=464)
    a.add_argument("--scheduler", default="Wan2.2-DQ-4step")
    a.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "source-bank":
        result = sources_from_legacy(args.legacy)
        save_new(args.out, result)
        print(f"Source bank: {len(result['sources'])} records (exposure recorded)")
    elif args.command == "calibration-plan":
        v = plan_calibration(args.bank, args.geometry, strata=args.strata, rows=args.rows,
                             seed=args.seed, output=args.out)
        print(f"Calibrated row plan: {v['source_count']} sources × {args.rows} rows × 160 native scopes")
    else:
        v = plan_video(args.bank, role=args.role, strata=args.strata, seed=args.seed,
                       samples=args.samples_per_source, variants=args.variants,
                       experiment=args.experiment, frames=args.frames, fps=args.fps,
                       width=args.width, height=args.height, scheduler=args.scheduler, output=args.out)
        print(f"Real video jobs planned: {len(v['samples'])}; generated/validated: 0")


if __name__ == "__main__":
    main()
