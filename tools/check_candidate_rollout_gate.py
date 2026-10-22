#!/usr/bin/env python3
"""Static metadata gate for selected FFNDS candidates.

No model/GPU/network access. This validates evidence structure, not the video itself.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research/candidate_rollout/registry"
PROMOTED = {"ROLLOUT_PASSED_SMOKE", "READY_FOR_PARETO_BENCHMARK"}
SELECTED_PENDING = {"LOCALLY_SUPPORTED", "WAITING_FOR_ROLLOUT_AUTHORIZATION", "ROLLOUT_RUNNING"}
OTHER = {"EXPLORATORY", "ROLLOUT_FAILED_QUALITY", "CLOSED"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MATCHED = (
    "same_model_and_checkpoint", "same_source_image", "same_prompt", "same_seed",
    "same_initial_noise", "same_scheduler_and_timesteps", "same_frames_and_resolution",
    "same_precision_regime",
)
VISUAL = (
    "full_duration_reviewed", "subject_identity_and_scene_integrity",
    "motion_coherence", "static_background_and_small_objects_stable",
    "stable_lighting_without_unexpected_flicker", "no_catastrophic_artifacts",
)

def validate_record(record, root=ROOT):
    errors = []
    if not isinstance(record, dict):
        return ["record must be a JSON object"]
    cid = record.get("candidate_id")
    if not isinstance(cid, str) or not cid.strip():
        errors.append("candidate_id must be nonempty")
    status = record.get("status")
    if status not in PROMOTED | SELECTED_PENDING | OTHER:
        return errors + ["unknown status"]
    if status in SELECTED_PENDING:
        if record.get("next_required") != "COMPLETE_PAIRED_FREE_RUNNING_VIDEO_SMOKE":
            errors.append("selected candidate must list full free-running video smoke as next required milestone")
        if record.get("rollout") and record["rollout"].get("outcome") == "PASS":
            errors.append("pending candidate claims pass; use reviewed promotion status")
    if status in PROMOTED:
        r = record.get("rollout")
        if not isinstance(r, dict):
            return errors + ["promoted candidate lacks full rollout evidence"]
        if r.get("outcome") != "PASS":
            errors.append("promoted rollout outcome must be PASS")
        if r.get("candidate_video_count", 0) < 1:
            errors.append("must have >=1 actually completed candidate video")
        if r.get("completed_student_steps") != 4:
            errors.append("must complete native four-step student denoising")
        for k in ("candidate_executed_at_intended_positions", "student_freely_evolved_across_steps",
                  "video_decoded", "paired_dense_DQ_video_exists"):
            if r.get(k) is not True:
                errors.append(k + " must be true")
        if r.get("intermediate_latent_reset_to_DQ") is not False:
            errors.append("stepwise D-Q restoration does not count as candidate rollout")
        matched = r.get("matched", {})
        for k in MATCHED:
            if not isinstance(matched, dict) or matched.get(k) is not True:
                errors.append("matched." + k + " must be true")
        for name in ("candidate_video", "baseline_video"):
            video = r.get(name, {})
            if not isinstance(video, dict) or not SHA256.fullmatch(str(video.get("sha256", ""))):
                errors.append(name + " must have SHA256")
            if not isinstance(video, dict) or not isinstance(video.get("locator"), str) or not video["locator"].strip():
                errors.append(name + " must have a nonempty verified artifact locator")
        review = r.get("visual_review", {})
        for k in VISUAL:
            if not isinstance(review, dict) or review.get(k) is not True:
                errors.append("visual_review." + k + " must be true")
        if not isinstance(review, dict) or review.get("verdict") != "NO_CATASTROPHIC_COLLAPSE":
            errors.append("full-video visual verdict required")
        if not isinstance(review, dict) or len(str(review.get("notes", "")).strip()) < 20:
            errors.append("time-aware human visual notes are required")
        report = r.get("review_report_path", "")
        if not isinstance(report, str) or not report or Path(report).is_absolute():
            errors.append("review_report_path must be a repository-relative file")
        else:
            target = (root / report).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                errors.append("review_report_path must resolve to an existing repository file")
        if not isinstance(r.get("authorization_reference"), str) or not r["authorization_reference"].strip():
            errors.append("separate model authorization reference required")
        if status == "READY_FOR_PARETO_BENCHMARK" and not str(record.get("next_validation_plan", "")).strip():
            errors.append("benchmark candidate needs fresh-source paired video and end-to-end latency plan")
    return errors

def main():
    if not REGISTRY.is_dir():
        raise SystemExit("Candidate registry directory missing")
    paths = sorted(REGISTRY.glob("*.json"))
    bad = 0
    for p in paths:
        try:
            errors = validate_record(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, OSError) as exc:
            errors = [str(exc)]
        if errors:
            bad += 1
            for msg in errors:
                print(f"{p.relative_to(ROOT)}: {msg}", file=sys.stderr)
    print(f"candidate full-rollout metadata gate: {len(paths)} records, {bad} invalid")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
