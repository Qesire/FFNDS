"""Static FFNDS cohort preflight: independent sources, calibration rows, paired generations.

JSON input is a *new* experiment manifest, not a parser for legacy RUN records.
This check requires no GPU or non-stdlib dependencies and never launches experiments.
"""
import argparse
from collections import defaultdict
import filecmp
import json
from pathlib import Path

SCHEMA = "ffnds.cohort.v1"
ROLES = {"DISCOVERY", "CALIBRATION", "DEVELOPMENT", "FROZEN_CONFIRMATION", "OFFICIAL_BENCHMARK"}
MODES = {"CUSTOM_PAIRED", "OFFICIAL_I2V_5SAMPLE"}
PAIR_FIELDS = ("conditioning_id", "initial_noise_id", "scheduler_id", "frames", "fps", "width", "height")
REQUIRED_SAMPLE = ("source_id", "variant", "seed", "sample_index", "run_id", "asset_id", *PAIR_FIELDS)


def audit(data, verify_files=False):
    errors, warnings = [], []
    if data.get("schema") != SCHEMA:
        errors.append("schema must be ffnds.cohort.v1")
    mode = data.get("mode", "CUSTOM_PAIRED")
    if mode not in MODES:
        errors.append("mode must be CUSTOM_PAIRED or OFFICIAL_I2V_5SAMPLE")
    sources = data.get("sources", [])
    samples = data.get("samples", [])
    calibration = data.get("calibration", [])
    if not isinstance(sources, list) or not isinstance(samples, list) or not isinstance(calibration, list):
        return {"status": "INVALID", "errors": ["sources, samples, calibration must be lists"], "warnings": []}
    by_source, image_roles = {}, defaultdict(set)
    for i, s in enumerate(sources):
        if not isinstance(s, dict):
            errors.append(f"sources[{i}] must be an object")
            continue
        identity = (s.get("source_id"), s.get("image_id"), s.get("prompt_id"))
        if any(not isinstance(v, str) or not v.strip() for v in identity):
            errors.append(f"sources[{i}] requires nonempty source_id/image_id/prompt_id")
            continue
        sid, img, _ = identity
        role = s.get("role")
        if sid in by_source:
            errors.append(f"duplicate source_id {sid}")
        by_source[sid] = s
        if role not in ROLES:
            errors.append(f"unknown role for {sid}: {role}")
        image_roles[img].add(role)
        exposure = s.get("exposure_history", {})
        if role in ("FROZEN_CONFIRMATION", "OFFICIAL_BENCHMARK") and any(
            exposure.get(field) for field in ("fitting", "method_selection", "prior_evaluation")
        ):
            errors.append(f"previously exposed source {sid} cannot be a fresh confirmation")
    for img, roles in image_roles.items():
        if len(roles) > 1:
            errors.append(f"image_id {img} is present in multiple data roles: {sorted(roles)}")
    observed, by_key, run_scopes, asset_scopes = defaultdict(list), {}, {}, {}
    for i, x in enumerate(samples):
        if not isinstance(x, dict):
            errors.append(f"samples[{i}] must be an object")
            continue
        absent = [f for f in REQUIRED_SAMPLE if x.get(f) is None or x.get(f) == ""]
        if absent:
            errors.append(f"samples[{i}] missing {','.join(absent)}")
            continue
        sid, arm = x["source_id"], x["variant"]
        if sid not in by_source:
            errors.append(f"unknown sample source {sid}")
            continue
        if not isinstance(x["sample_index"], int) or x["sample_index"] < 0:
            errors.append(f"invalid sample_index for {sid}/{arm}")
            continue
        key = (sid, arm, x["sample_index"])
        if key in by_key:
            errors.append(f"duplicate sample index {key}")
        by_key[key] = x
        observed[sid, arm].append(x)
        for name, seen in (("run_id", run_scopes), ("asset_id", asset_scopes)):
            identity = x[name]
            scope = (sid, arm, x["sample_index"])
            if identity in seen and seen[identity] != scope:
                errors.append(f"reused {name} {identity}: file names are not independent generations")
            seen[identity] = scope
        if by_source[sid].get("role") == "CALIBRATION":
            warnings.append(f"source {sid} has generated samples in calibration role; label their use precisely")
    for (sid, arm), items in observed.items():
        seeds = [str(x["seed"]) for x in items]
        if len(set(seeds)) != len(seeds):
            errors.append(f"same-seed sample repeats for {sid}/{arm}")
        indices = sorted(x["sample_index"] for x in items)
        if mode == "OFFICIAL_I2V_5SAMPLE" and indices != list(range(5)):
            errors.append(f"official I2V requires five distinct generated samples 0..4 for {sid}/{arm}")
        if verify_files:
            for i, left in enumerate(items):
                p = Path(left.get("video_path", ""))
                if not p.is_file():
                    errors.append(f"missing video file for {sid}/{arm}/{i}: {p}")
                    continue
                for right in items[:i]:
                    q = Path(right.get("video_path", ""))
                    if q.is_file() and filecmp.cmp(p, q, shallow=False):
                        errors.append(f"byte-identical video copies within {sid}/{arm}")
                        break
    variants = sorted({x.get("variant") for x in samples if isinstance(x, dict) and x.get("variant")})
    if "dense" in variants and len(variants) > 1:
        for (sid, arm), items in observed.items():
            if arm == "dense":
                continue
            for x in items:
                ref = by_key.get((sid, "dense", x["sample_index"]))
                if ref is None:
                    errors.append(f"missing paired dense sample {sid}/{x['sample_index']} for {arm}")
                    continue
                for field in (*PAIR_FIELDS, "seed"):
                    if x[field] != ref[field]:
                        errors.append(f"pair mismatch in {field} for {sid}/{arm}/{x['sample_index']}")
    positions = defaultdict(set)
    for i, row in enumerate(calibration):
        if not isinstance(row, dict):
            errors.append(f"calibration[{i}] must be an object")
            continue
        sid, step, block = row.get("source_id"), row.get("step"), row.get("block")
        if sid not in by_source:
            errors.append(f"calibration[{i}] references unknown source")
            continue
        if by_source[sid].get("role") != "CALIBRATION":
            errors.append(f"calibration source {sid} has role {by_source[sid].get('role')}")
        if not isinstance(step, int) or step not in range(4) or not isinstance(block, int) or block not in range(40):
            errors.append(f"invalid Wan step/block for calibration[{i}]")
            continue
        key = (sid, step, block, row.get("source_state"))
        if key in positions[sid]:
            errors.append(f"duplicate calibration source/step/block/state {key}")
        positions[sid].add(key)
        ids = row.get("selected_row_ids")
        n, N = row.get("n_rows"), row.get("population_rows")
        if (not isinstance(ids, list) or not isinstance(n, int) or not isinstance(N, int)
                or n < 1 or N < n or len(ids) != n or len(set(map(str, ids))) != n
                or any(not isinstance(j, int) or j < 0 or j >= N for j in ids)):
            errors.append(f"invalid within-source sampled row coordinates for calibration[{i}]")
        if row.get("source_state") not in ("teacher_input", "student_input"):
            errors.append(f"undefined calibration input-state semantics for calibration[{i}]")
    summary = {
        "source_count": len(by_source),
        "source_count_by_role": {role: sum(s.get("role") == role for s in by_source.values()) for role in sorted(ROLES)},
        "independent_generation_runs": len(run_scopes),
        "sample_records": len(samples),
        "calibration_records": len(calibration),
        "independent_unit": "source_id; seeds are within-source repetitions; FFN positions and rows are nested",
        "sampling_mode": mode,
    }
    return {"status": "PASS" if not errors else "INVALID", "errors": errors, "warnings": warnings, "summary": summary}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--verify-file-content", action="store_true")
    args = p.parse_args()
    report = audit(json.loads(args.manifest.read_text(encoding="utf-8")), args.verify_file_content)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
