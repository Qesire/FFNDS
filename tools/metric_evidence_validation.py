"""Typed measurement and claim provenance checks; no model or network execution."""
import json
import math
from pathlib import Path
import re

OFFICIAL_I2V = {
    "vbench_i2v_subject", "vbench_i2v_background", "vbench_camera_motion",
    "vbench_i2v_subject_consistency", "vbench_i2v_background_consistency",
    "vbench_i2v_motion_smoothness", "vbench_i2v_dynamic_degree",
    "vbench_i2v_aesthetic_quality", "vbench_i2v_imaging_quality",
}
VIDEO_CLAIMS = {"video_noncollapse_smoke", "independent_video_quality", "video_speed_quality_pareto"}
INDEPENDENT_CLAIMS = {"independent_video_quality", "video_speed_quality_pareto"}
TIMINGS = {"complete_ffn_latency_ms", "generation_gpu_time_s", "generation_wall_time_s"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")

def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)

def existing_json(path, root):
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise ValueError("expected repository-relative JSON file")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()): raise ValueError("file leaves repository")
    return json.loads(target.read_text(encoding="utf-8"))

def observation_errors(m, ref):
    errors = []
    mid = m["metric_id"]
    v = m.get("value")
    kind = ref.get("value_kind", "number")
    if kind == "number":
        if not finite_number(v): errors.append(mid + " value must be a finite number")
        elif ref["unit"] == "cosine":
            if not -1 <= v <= 1: errors.append(mid + " cosine outside [-1,1]")
        else:
            signed_estimate = mid == "jedi_dataset" and m.get("estimator") == "unbiased_u_statistic"
            if v < 0 and not signed_estimate: errors.append(mid + " value must be nonnegative")
            if (mid in TIMINGS or mid == "synthetic_backend_speedup") and v <= 0:
                errors.append(mid + " value must be positive")
            if mid.endswith("equivalence") and int(v) != v:
                errors.append(mid + " mismatch count must be integral")
    elif kind == "boolean":
        if type(v) is not bool: errors.append(mid + " value must be boolean")
    elif kind == "review":
        if not isinstance(v, dict) or v.get("verdict") not in {
            "NO_CATASTROPHIC_COLLAPSE", "COLLAPSE", "INCONCLUSIVE"
        }: errors.append(mid + " value must contain a review verdict")
    elif kind == "summary":
        if not isinstance(v, dict) or not v or any(not finite_number(x) or x < 0 for x in v.values()):
            errors.append(mid + " summary must contain named finite nonnegative numbers")
    if ref.get("allowed_execution_modes") and m.get("execution_mode") not in ref["allowed_execution_modes"]:
        errors.append(mid + " execution_mode incompatible with metric")
    if not SHA256.fullmatch(str(m.get("implementation_sha", ""))):
        errors.append(mid + " implementation_sha must be SHA256")
    if mid in {"post_writeback_nmse", "ffn_only_nmse"}:
        if not finite_number(m.get("denominator")) or m["denominator"] <= 0:
            errors.append(mid + " requires positive denominator")
        if not m.get("denominator_definition"): errors.append(mid + " requires denominator definition")
    if mid == "jedi_dataset" and m.get("estimator") not in {"unbiased_u_statistic", "biased_v_statistic"}:
        errors.append(mid + " requires explicit MMD estimator")
    return errors

def claim_errors(r, entries, root):
    errors = []
    claim = r.get("claim")
    by_id = {}
    for m in entries:
        if isinstance(m, dict): by_id.setdefault(m.get("metric_id"), []).append(m)
    if claim in VIDEO_CLAIMS:
        if r.get("full_student_free_running") is not True or r.get("paired_baseline_full_video") is not True:
            errors.append("video claim lacks true full free-running paired student evidence")
        for m in by_id.get("free_rollout_completed", []):
            if m.get("value") is not True: errors.append("video claim contains incomplete rollout")
        for m in by_id.get("visual_collapse_review", []):
            v = m.get("value")
            if not isinstance(v, dict) or v.get("verdict") != "NO_CATASTROPHIC_COLLAPSE":
                errors.append("video claim contains failed or inconclusive review")
        try:
            from check_candidate_rollout_gate import validate_record, PROMOTED
            rollout = existing_json(r.get("candidate_rollout_record"), root)
            if not isinstance(rollout, dict): raise ValueError("rollout record must be object")
            errors += ["candidate rollout: " + x for x in validate_record(rollout, root)]
            if rollout.get("status") not in PROMOTED: errors.append("candidate rollout is not promoted")
            if rollout.get("candidate_id") != r.get("method_id"):
                errors.append("candidate rollout method identity mismatch")
        except (ValueError, OSError) as exc:
            errors.append("candidate_rollout_record: " + str(exc))
    if claim in INDEPENDENT_CLAIMS:
        if not OFFICIAL_I2V.issubset(by_id): errors.append("full independent claim requires all nine official I2V dimensions")
        if r.get("source_overlap_audit_passed") is not True: errors.append("independent claim requires source overlap audit")
        if not re.fullmatch(r"[0-9a-f]{40}", str(r.get("official_suite_revision", ""))):
            errors.append("independent claim requires pinned official suite revision")
        for mid in OFFICIAL_I2V:
            for m in by_id.get(mid, []):
                if m.get("source_exposure") != "fresh_final": errors.append(mid + " requires fresh_final sources")
                if not SHA256.fullmatch(str(m.get("source_manifest_sha256", ""))):
                    errors.append(mid + " requires dimension source manifest SHA256")
                if not finite_number(m.get("baseline_value")): errors.append(mid + " requires paired baseline score")
    if claim in INDEPENDENT_CLAIMS | {"native_speedup"}:
        context = r.get("benchmark_context", {})
        if not isinstance(context, dict): context = {}
        for key in ("deployment_id", "checkpoint_identity", "precision", "scheduler", "actual_timesteps", "frames", "resolution"):
            if not context.get(key): errors.append("comparison missing " + key)
        if claim in {"native_speedup", "video_speed_quality_pareto"} and not context.get("hardware"):
            errors.append("speed comparison missing hardware")
        for m in entries:
            if not isinstance(m, dict): continue
            if not context.get("deployment_id") or m.get("deployment_id") != context["deployment_id"]:
                errors.append(str(m.get("metric_id")) + " deployment identity mismatch")
    for mid in TIMINGS:
        for m in by_id.get(mid, []):
            if not finite_number(m.get("baseline_value")) or m["baseline_value"] <= 0:
                errors.append(mid + " requires positive baseline latency")
            if type(m.get("repetitions")) is not int or m["repetitions"] <= 0:
                errors.append(mid + " requires positive repetition count")
            if not isinstance(m.get("timing_boundary"), str) or not m["timing_boundary"].strip():
                errors.append(mid + " requires timing boundary")
            if claim == "native_speedup" and mid == "complete_ffn_latency_ms":
                if finite_number(m.get("value")) and finite_number(m.get("baseline_value")) and m["value"] >= m["baseline_value"]:
                    errors.append("native_speedup requires faster measured complete FFN")
    if claim == "video_speed_quality_pareto":
        if r.get("matched_prior_art_or_strong_fixed") is not True: errors.append("competitive claim lacks matched strong comparison")
        # Contextual motion and uncertainty need an explicit scientific review.
        # This checks the linked review's structure, not its scientific truth.
        try:
            review = existing_json(r.get("pareto_review_record"), root)
            if not isinstance(review, dict): raise ValueError("Pareto review must be object")
            for key in ("comparison_ids", "quality_tradeoff_definition", "uncertainty_analysis", "per_source_results", "decision"):
                if not review.get(key): errors.append("Pareto review missing " + key)
            if review.get("method_id") != r.get("method_id"): errors.append("Pareto review method mismatch")
            if review.get("decision") != "SUPPORTED": errors.append("Pareto review does not support claim")
        except (ValueError, OSError) as exc:
            errors.append("pareto_review_record: " + str(exc))
    return errors
