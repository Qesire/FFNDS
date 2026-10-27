"""Verify truly generated I2V clips, construct dimension-specific VBench inputs, and optionally evaluate.

The stage uses original generation receipts and actual initial-noise tensor files. For
historical single-clip cohorts prefer CUSTOM_PAIRED. Standard benchmark identity needs
the complete official Image Suite, five real clips per image-prompt and all dimensions.
"""
import argparse
from collections import defaultdict
import filecmp
import json
from pathlib import Path
import shutil
import os
import subprocess
import sys

from ffnds_cohort_preflight import audit
from ffnds_experiment_plans import load, save_new

DIMS = ("i2v_subject", "i2v_background", "camera_motion", "subject_consistency",
        "background_consistency", "motion_smoothness", "dynamic_degree",
        "aesthetic_quality", "imaging_quality")


def valid_name(text):
    if (not text or "/" in text or "\\" in text or "\x00" in text
            or text in (".", "..") or len(text.encode("utf-8")) > 210):
        raise ValueError("Unsafe or unrepresentable official prompt-based filename")
    return text


def collect(plan, root):
    if plan.get("phase") != "REGISTERED_REAL_GENERATION_PLAN":
        raise ValueError("Require a native registered generation plan")
    errors = audit(plan)["errors"]
    if errors:
        raise ValueError("Input plan invalid: " + "; ".join(errors))
    rows, by_source = [], {s["source_id"]: s for s in plan["sources"]}
    by_key, videos_by_arm = {}, defaultdict(list)
    for index, job in enumerate(plan["samples"]):
        path = Path(root) / f"job{index:06d}" / "RUN.json"
        if not path.is_file():
            raise FileNotFoundError(f"Real native job is incomplete at index {index}: {path}")
        row = load(path)
        if (row.get("status") != "COMPLETE_NATIVE_GENERATION_ONLY"
                or row.get("schema") != "ffnds.registered.native.video.v1"
                or row.get("job") != job or row.get("native_ffn_sites") != 160
                or row.get("source") != by_source[job["source_id"]]
                or row.get("scheduler_generator_reset") is not True):
            raise ValueError(f"Missing native FFN coverage or mismatched frozen job {index}")
        video = Path(row["video_path"])
        noise = Path(row["actual_initial_noise_file"])
        if not video.is_file() or not noise.is_file():
            raise FileNotFoundError(f"Missing actual generated video / initial latent {index}")
        if row["decoded_frames"] != job["frames"] or row["decoded_size"] != [job["width"], job["height"]]:
            raise ValueError(f"Decoded video identity differs at job {index}")
        if row.get("native_ffn_counters", {}).get("linear_forward_calls") != 1600:
            raise ValueError(f"Actual native full FFN invocation count differs at job {index}")
        key = (job["source_id"], job["sample_index"], job["variant"])
        if key in by_key:
            raise ValueError("Duplicate native source/sample/variant")
        by_key[key] = (job, row)
        videos_by_arm[(job["source_id"], job["variant"])].append(video)
        rows.append({**job, "status": "COMPLETED", "video_path": str(video),
                     "run_receipt_path": str(path),
                     "actual_initial_noise_file": str(noise)})
    for key, paths in videos_by_arm.items():
        for i, left in enumerate(paths):
            for right in paths[:i]:
                if left.samefile(right) or filecmp.cmp(left, right, shallow=False):
                    raise ValueError(f"Repeated video bytes in different sample indices for {key}")
    for (sid, sample, arm), (job, row) in by_key.items():
        if arm == "dense":
            continue
        reference = by_key.get((sid, sample, "dense"))
        if reference is None:
            raise ValueError("Missing real dense paired reference")
        other_job, other_row = reference
        left = Path(row["actual_initial_noise_file"])
        right = Path(other_row["actual_initial_noise_file"])
        # Exact saved tensors, not merely equal seeds or a claimed digest.
        if not filecmp.cmp(left, right, shallow=False):
            raise ValueError(f"Actual Dense/candidate initial-noise tensors differ at {(sid, sample)}")
        for field in ("conditioning_id", "scheduler_id", "frames", "fps", "width", "height", "seed"):
            if job[field] != other_job[field]:
                raise ValueError(f"Paired native condition mismatch in {field}")
    out = {**plan, "phase": "COMPLETE_REGISTERED_GENERATION_VERIFIED", "samples": rows}
    check = audit(out, verify_files=True)
    if check["errors"]:
        raise ValueError("Postrun cohort audit failed: " + "; ".join(check["errors"]))
    return out


def stage(complete, original_full_info, output):
    source_meta = {(row["prompt_en"], row["image_name"]): row for row in original_full_info}
    expected = {(s["prompt"], s["image_name"]) for s in complete["sources"]}
    missing = expected - set(source_meta)
    if missing:
        raise ValueError("Source cohort differs from pinned original official full-info: " + str(list(missing)[:3]))
    official_full = expected == set(source_meta) and complete["mode"] == "OFFICIAL_I2V_5SAMPLE"
    root = Path(output)
    if root.exists():
        raise FileExistsError("A new evaluation staging directory is required: " + str(root))
    root.mkdir(parents=True)
    names = set()
    count = defaultdict(int)
    sources = {s["source_id"]: s for s in complete["sources"]}
    for job in complete["samples"]:
        source = sources[job["source_id"]]
        original = source_meta[(source["prompt"], source["image_name"])]
        prompt = valid_name(source["prompt"])
        for dim in original["dimension"]:
            if dim not in DIMS:
                continue  # Extra official dimensions are intentionally not silently promoted.
            leaf = root / job["variant"] / dim
            videos = leaf / "videos"
            images = leaf / "images"
            videos.mkdir(parents=True, exist_ok=True)
            images.mkdir(parents=True, exist_ok=True)
            basename = valid_name(f"{prompt}-{job['sample_index']}")
            key = (job["variant"], dim, basename)
            if key in names:
                raise ValueError(f"Official filename collision: {key}")
            names.add(key)
            clip = videos / (basename + ".mp4")
            image = images / (basename + Path(source["image_name"]).suffix)
            try:
                os.link(job["video_path"], clip)
            except OSError:
                shutil.copyfile(job["video_path"], clip)
            try:
                os.link(source["image_path"], image)
            except OSError:
                shutil.copyfile(source["image_path"], image)
            count[(job["variant"], dim)] += 1
    for (variant, dim), n in count.items():
        save_new(root / variant / dim / "RUN.json",
                 {"status": "STAGED_REAL_VIDEOS_FOR_ORIGINAL_EVALUATOR",
                  "dimension": dim, "variant": variant,
                  "real_generated_video_count": n,
                  "input_mode": "OFFICIAL_I2V_5SAMPLE" if official_full else "CUSTOM_INPUT_SUBSET",
                  "official_leaderboard_comparable": False,
                  "evaluation": "NOT_RUN"})
    save_new(root / "INDEX.json", {"status": "STAGED_PENDING_ORIGINAL_VBENCH_EVALUATION",
              "input_mode": "OFFICIAL_I2V_5SAMPLE" if official_full else "CUSTOM_INPUT_SUBSET",
              "full_image_suite_matched": official_full,
              "dimensions": [{"variant": a, "dimension": b, "clips": c}
                             for (a, b), c in sorted(count.items())],
              "official_scores": "NOT_RUN",
              "source_clusters": len({s["image_id"] for s in complete["sources"]})})
    return root / "INDEX.json"


def evaluate(stage_root, original_checkout, ratio, dims, *, execute):
    root = Path(stage_root)
    index = load(root / "INDEX.json")
    vbench = Path(original_checkout)
    evaluator = vbench / "evaluate_i2v.py"
    if not evaluator.is_file():
        raise FileNotFoundError("Need installed pinned VBench evaluate_i2v.py")
    requested = set(dims)
    if not requested or requested - set(DIMS):
        raise ValueError("Unrecognized original VBench-I2V metric dimensions")
    commands = []
    for item in index["dimensions"]:
        if item["dimension"] not in requested:
            continue
        leaf = root / item["variant"] / item["dimension"]
        target = leaf / "result"
        if target.exists():
            raise FileExistsError("A dimension result already exists; use a fresh evaluation root")
        cmd = [sys.executable, str(evaluator),
               "--videos_path", str((leaf / "videos").resolve()),
               "--dimension", item["dimension"],
               "--ratio", ratio, "--output_path", str(target.resolve())]
        if index["full_image_suite_matched"]:
            cmd += ["--mode", "vbench_standard"]
        else:
            cmd += ["--mode", "custom_input", "--custom_image_folder", str((leaf / "images").resolve())]
        commands.append((leaf, cmd))
    if not commands:
        raise ValueError("Selected official dimensions have no actual sampled source videos")
    if not execute:
        return {"status": "PREFLIGHT_ONLY", "commands": [c for _, c in commands]}
    for leaf, command in commands:
        with (leaf / "evaluation_stdout.log").open("x") as stdout, (leaf / "evaluation_stderr.log").open("x") as stderr:
            subprocess.run(command, cwd=vbench, stdout=stdout, stderr=stderr, check=True)
        save_new(leaf / "EVALUATION.json", {
            "status": "ORIGINAL_VBENCH_RUN_RETURNED_SUCCESS",
            "command": command, "original_checkout": str(vbench),
            "quality_accepted": False,
            "source_metric_aggregation": "Inspect evaluator original output; this receipt does not invent score"})
    return {"status": "ORIGINAL_EVALUATOR_INVOKED", "count": len(commands)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("stage")
    a.add_argument("--plan", required=True)
    a.add_argument("--run-root", required=True)
    a.add_argument("--official-full-info", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("evaluate")
    a.add_argument("--stage-root", required=True)
    a.add_argument("--vbench-checkout", required=True)
    a.add_argument("--ratio", default="16-9")
    a.add_argument("--dimensions", nargs="+", required=True)
    a.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.command == "stage":
        complete = collect(load(args.plan), args.run_root)
        staged = stage(complete, load(args.official_full_info), args.out)
        save_new(Path(args.out) / "VERIFIED_GENERATIONS.json", complete)
        print(json.dumps({"status": "STAGED_FROM_REAL_NATIVE_GENERATIONS", "index": str(staged)}))
    else:
        print(json.dumps(evaluate(args.stage_root, args.vbench_checkout, args.ratio,
                                  args.dimensions, execute=args.execute), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
