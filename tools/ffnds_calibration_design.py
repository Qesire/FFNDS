"""FFNDS calibration v2: validate registered factorial design, profile real source-level support.

Only standard-library dependencies. A profile is a summary from ACTUAL native operands;
a registration, synthetic test or historical video score is never a measurement.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean, pstdev

SCHEMA = "ffnds.calibration.design.v2"
PROFILE = "ffnds.calibration.source_statistics.v2"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, object_):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(object_, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def validate_design(design):
    errors = []
    if design.get("schema") != SCHEMA:
        errors.append("invalid calibration design schema")
    model = design.get("model", {})
    for key, value in (("steps", 4), ("blocks", 40), ("input_width", 5120),
                       ("hidden_width", 13824), ("channel_group", 64), ("groups", 216)):
        if model.get(key) != value:
            errors.append(f"native model coordinate {key} differs")
    if model.get("experts") != ["high", "high", "low", "low"]:
        errors.append("native four-step expert order differs")
    arms = design.get("calibration_arms", [])
    if not arms:
        errors.append("at least one registered calibration arm required")
    ids = set()
    for arm in arms:
        aid, sources, rows = arm.get("id"), arm.get("source_count"), arm.get("rows_per_source")
        if not aid or aid in ids:
            errors.append("calibration arm ID is missing or duplicated")
        ids.add(aid)
        if (type(sources) is not int or type(rows) is not int or sources < 1 or rows < 1):
            errors.append(f"invalid source count / row density for {aid}")
        allocation = arm.get("allocation", {})
        if (not isinstance(allocation, dict) or not allocation
                or any(k not in design.get("strata", []) or type(v) is not int or v < 0
                       for k, v in allocation.items())):
            errors.append(f"arm {aid} has invalid stratum allocation")
        elif type(sources) is int and sum(allocation.values()) != sources:
            errors.append(f"arm {aid} lacks exact source strata allocation")
        if arm.get("status") not in {"PLANNED_NOT_COLLECTED",
                                   "REFERENCE_HISTORICAL_BUDGET_NOT_V2_FIT"}:
            errors.append(f"arm {aid} claims execution without native evidence")
    if not design.get("split_policy", {}).get("unit") == "image_cluster":
        errors.append("image cluster must define independent split unit")
    if not design.get("randomization", {}).get("cohort") == "nested_per_stratum_frozen_permutation":
        errors.append("required shared nested cohort sampling")
    return errors


def validate_bank(sources):
    errors = []
    source_ids, roles, image_roles = set(), defaultdict(int), defaultdict(set)
    image_strata = defaultdict(set)
    for s in sources:
        sid, image_id, role = s.get("source_id"), s.get("image_id"), s.get("role")
        if not sid or sid in source_ids:
            errors.append("missing or repeated source_id")
        source_ids.add(sid)
        if not image_id or not s.get("scene_stratum"):
            errors.append("source image cluster or scene stratum missing")
        if role not in ("DISCOVERY", "CALIBRATION", "DEVELOPMENT", "FROZEN_CONFIRMATION", "OFFICIAL_BENCHMARK"):
            errors.append("unregistered data role")
        roles[role] += 1
        image_roles[image_id].add(role)
        image_strata[image_id].add(s.get("scene_stratum"))
        if role == "FROZEN_CONFIRMATION" and any(s.get("exposure_history", {}).values()):
            errors.append("exposed source cannot be new confirmation")
    for image, members in image_roles.items():
        if len(members) > 1:
            errors.append(f"same image cluster {image} crosses data purposes")
        if len(image_strata[image]) > 1:
            errors.append(f"same image cluster {image} occupies incompatible scene strata")
    return errors


def coverage(design, bank):
    errors = validate_design(design)
    if bank.get("schema") != "ffnds.source.bank.v1":
        errors.append("source bank schema differs")
    sources = bank.get("sources", [])
    errors += validate_bank(sources)
    if errors:
        return {"status": "INVALID", "errors": errors, "arms": []}
    available = defaultdict(set)
    by_role = defaultdict(set)
    for row in sources:
        if row["role"] == "CALIBRATION":
            available[row["scene_stratum"]].add(row["image_id"])
            by_role["CALIBRATION"].add(row["image_id"])
    records = []
    for arm in design["calibration_arms"]:
        need = arm["source_count"]
        observed = len(by_role["CALIBRATION"])
        exact = all(len(available[stratum]) >= n for stratum, n in arm["allocation"].items())
        records.append({
            "arm_id": arm["id"],
            "source_target": need,
            "rows_per_source_per_position": arm["rows_per_source"],
            "token_row_budget_per_position": need * arm["rows_per_source"],
            "available_unique_calibration_images": observed,
            "required_strata_counts": arm["allocation"],
            "available_strata_counts": {k: len(v) for k, v in sorted(available.items())},
            "status": "SOURCE_CANDIDATES_PRESENT" if observed >= need and exact else "AWAITING_NEW_STRATIFIED_SOURCES",
            "native_operands": "NOT_VERIFIED",
            "native_fit": "NOT_RUN",
            "independent_video": "NOT_RUN",
        })
    return {"status": "STATIC_BANK_COVERAGE_ONLY", "errors": [], "arms": records,
            "independent_unit": "image_cluster; rows nested within source"}


def topk(values, n):
    # Deterministic tie rule: stable low group index. No other precision transformation.
    return set(sorted(range(len(values)), key=lambda i: (-values[i], i))[:n])


def signed_removed_group_energy(records, removed):
    """Cross-group complement from actual gated signed-output Gram, if completely observed.

    G[g,h] = E <e5*W2_g*a_g, e5*W2_h*a_h>; use equal image-cluster
    weights, average prompt repeats within each image, then image clusters.
    A missing Gram is a missing measurement, not zero cross interaction.
    """
    observed = [r.get("signed_group_gram") is not None for r in records]
    if not any(observed):
        return {"status": "NOT_MEASURED", "diagonal_removed_energy": None,
                "cross_term": None, "total_signed_removed_energy": None}
    if not all(observed):
        return {"status": "PARTIAL_MEASUREMENT", "diagonal_removed_energy": None,
                "cross_term": None, "total_signed_removed_energy": None}
    by_image = defaultdict(list)
    for row in records:
        gram = row["signed_group_gram"]
        energy = row["group_energy"]
        if not isinstance(gram, list) or len(gram) != 216:
            raise ValueError("Signed group Gram requires 216 actual native rows")
        diagonal = 0.
        interactions = 0.
        for i in range(216):
            line = gram[i]
            if not isinstance(line, list) or len(line) != 216:
                raise ValueError("Signed group Gram must be 216x216")
            if not math.isfinite(line[i]) or not math.isclose(
                    line[i], energy[i], abs_tol=1e-4, rel_tol=1e-3):
                raise ValueError("Signed Gram diagonal contradicts group-energy values")
        for i in removed:
            diagonal += energy[i]
            for j in removed:
                val = gram[i][j]
                if type(val) not in (int, float) or not math.isfinite(val):
                    raise ValueError("Nonfinite signed Gram interaction")
                if not math.isclose(val, gram[j][i], abs_tol=1e-4, rel_tol=1e-3):
                    raise ValueError("Native signed Gram is not symmetric")
                interactions += val
        if interactions < -1e-3 * max(1.0, diagonal):
            raise ValueError("Signed Gram produces materially negative squared removed output")
        by_image[row["image_id"]].append((diagonal, interactions - diagonal, interactions))
    averaged = [tuple(mean(item[k] for item in values) for k in range(3))
                for values in by_image.values()]
    d, c, total = [mean(row[i] for row in averaged) for i in range(3)]
    return {"status": "ACTUAL_SIGNED_GRAM_DIAGNOSTIC",
            "independent_image_clusters": len(averaged),
            "diagonal_removed_energy": d, "cross_term": c,
            "total_signed_removed_energy": total}


def source_support_diagnostic(records, *, keep_groups, bootstrap=0, seed=42):
    """Compute source-cluster support influence for one step/block/state.

    records: [{source_id, image_id, group_energy: [216 actual nonnegative floats]}].
    All statistics give equal weight to independent image clusters, not raw tokens.
    """
    if not 1 <= keep_groups <= 216:
        raise ValueError("group support count must be 1..216")
    by_image = defaultdict(list)
    for r in records:
        image, values = r.get("image_id"), r.get("group_energy")
        if not image or not r.get("source_id"):
            raise ValueError("record lacks source and image identity")
        if (not isinstance(values, list) or len(values) != 216
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError("expected 216 finite nonnegative actual group energy values")
        by_image[image].append(values)
    if not by_image:
        raise ValueError("No actual group energy records")
    # A profile contains one statistic per source/conditioning at one native site.
    # Repeated seeds should first be reduced by a declared source-level producer.
    source_ids = set()
    for row in records:
        sid = row["source_id"]
        if sid in source_ids:
            raise ValueError("Repeated source_id within native diagnostic scope: " + str(sid))
        source_ids.add(sid)
    cluster_vectors = []
    for image in sorted(by_image):
        x = by_image[image]
        cluster_vectors.append([mean(r[i] for r in x) for i in range(216)])
    n = len(cluster_vectors)
    avg = [mean(r[i] for r in cluster_vectors) for i in range(216)]
    primary = topk(avg, keep_groups)
    sorted_energy = sorted(avg, reverse=True)
    margin = sorted_energy[keep_groups - 1] - sorted_energy[keep_groups] if keep_groups < 216 else None
    removed = sorted(set(range(216)) - primary)
    signed = signed_removed_group_energy(records, removed)
    if n < 2:
        return {"status": "SOURCE_COUNT_INSUFFICIENT_FOR_STABILITY", "independent_image_clusters": n,
                "training_top_groups": sorted(primary), "boundary_energy_margin": margin,
                "signed_removed_group_energy": signed,
                "leave_one_cluster_out_jaccard": None, "bootstrap_jaccard_interval": None,
                "source_level_diagnostics": "INCONCLUSIVE"}
    leave = []
    for omit in range(n):
        retained = [cluster_vectors[j] for j in range(n) if j != omit]
        vector = [mean(v[i] for v in retained) for i in range(216)]
        support = topk(vector, keep_groups)
        leave.append(len(primary & support) / len(primary | support))
    lo = None
    if bootstrap > 0:
        rng = random.Random(seed)
        samples = []
        for _ in range(bootstrap):
            draws = rng.choices(cluster_vectors, k=n)
            candidate = topk([mean(r[i] for r in draws) for i in range(216)], keep_groups)
            samples.append(len(primary & candidate) / len(primary | candidate))
        samples.sort()
        lo = [samples[int((len(samples) - 1) * p)] for p in (0.025, 0.975)]
    dispersion = [pstdev(r[i] for r in cluster_vectors) for i in range(216)]
    return {"status": "SOURCE_LEVEL_DIAGNOSTICS_ONLY",
            "independent_image_clusters": n,
            "source_prompt_records": len(records),
            "training_top_groups": sorted(primary),
            "boundary_energy_margin": margin,
            "signed_removed_group_energy": signed,
            "leave_one_cluster_out_jaccard": {"mean": mean(leave), "min": min(leave), "max": max(leave)},
            "bootstrap_jaccard_interval": lo,
            "top_group_between_source_std_mean": mean(dispersion[i] for i in primary),
            "calibration_sufficiency": "NOT_ESTABLISHED_BY_LOCAL_DIAGNOSTICS",
            "student_state_and_motion": "NOT_MEASURED"}


def diagnose(profile, *, keep_groups_by_step, bootstrap=0, seed=42):
    if (not isinstance(keep_groups_by_step, (tuple, list)) or len(keep_groups_by_step) != 4
            or any(type(n) is not int or not 1 <= n <= 216 for n in keep_groups_by_step)):
        raise ValueError("Specify four real step-specific support counts (e.g. 54,108,108,162)")
    if profile.get("schema") != PROFILE or profile.get("evidence_kind") != "ACTUAL_NATIVE_OBSERVATION":
        raise ValueError("Require source statistics explicitly labeled actual native observations")
    grouped = defaultdict(list)
    for r in profile.get("records", []):
        step, block, state = r.get("step"), r.get("block"), r.get("source_state")
        if type(step) is not int or step not in range(4) or type(block) is not int or block not in range(40):
            raise ValueError("Invalid native step/block coordinate")
        if state not in ("teacher_input", "student_input"):
            raise ValueError("Undefined native input-state semantics")
        if r.get("statistic_semantics") != "native_gate_times_W2_group_contribution_l2_squared":
            raise ValueError("Group statistic must be actual native gated W2 contribution energy")
        grouped[(step, block, state)].append(r)
    if not grouped:
        raise ValueError("No actual source statistics recorded")
    out = {}
    for key in sorted(grouped):
        out[f"step{key[0]}_block{key[1]:02d}_{key[2]}"] = source_support_diagnostic(
            grouped[key], keep_groups=keep_groups_by_step[key[0]], bootstrap=bootstrap, seed=seed)
    return {"schema": "ffnds.calibration.diagnostic.v2", "status": "SOURCE_LEVEL_DIAGNOSTICS_ONLY",
            "actual_native_record_count": len(profile["records"]), "positions": out,
            "fit": "NOT_RUN", "new_video_scores": "NOT_RUN",
            "evidence_unit": "source image cluster"}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cmd = cli.add_subparsers(dest="command", required=True)
    a = cmd.add_parser("coverage")
    a.add_argument("--design", type=Path, required=True)
    a.add_argument("--bank", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a = cmd.add_parser("diagnose")
    a.add_argument("--statistics", type=Path, required=True)
    a.add_argument("--keep-groups-by-step", type=str, required=True,
                   help="Four comma-separated UB1 (or candidate) group widths per denoising step")
    a.add_argument("--bootstrap", type=int, default=200)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "coverage":
        result = coverage(read(args.design), read(args.bank))
    else:
        if args.bootstrap < 0:
            raise ValueError("bootstrap must be nonnegative")
        support = [int(v) for v in args.keep_groups_by_step.split(",")]
        result = diagnose(read(args.statistics), keep_groups_by_step=support,
                          bootstrap=args.bootstrap, seed=args.seed)
    write_new(args.out, result)
    print(json.dumps({"status": result["status"], "out": str(args.out)}, ensure_ascii=False))
    if result["status"] == "INVALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
