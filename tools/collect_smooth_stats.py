"""Collect original SmoothQuant activation absmax with per-input checkpoints.

Uses the unchanged archived native observer on all 400 modules per expert. Each
completed four-step input is committed atomically. Interrupted inputs rerun from
their original seed; completed inputs are verified and never silently replaced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

import torch

from run_dense_bf16 import ROOT, atomic_json, digest


def validate_stats(stats, samples):
    expected_names = {f"blocks.{block}.{name}" for block in range(40) for name in [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2"]}
    for expert, steps in [("high_noise_model", [0, 1]), ("low_noise_model", [2, 3])]:
        if set(stats[expert]) != expected_names:
            raise RuntimeError(f"Native calibration module coverage mismatch for {expert}")
        for name, record in stats[expert].items():
            value = record["s"]
            channels = 13824 if name.endswith("ffn.2") else 5120
            if value.dtype != torch.float32 or list(value.shape) != [channels]:
                raise RuntimeError(f"Unexpected calibration dtype/shape: {expert}.{name}")
            if not bool(torch.isfinite(value).all()) or not bool((value >= 0).all()):
                raise RuntimeError(f"Invalid absmax: {expert}.{name}")
            if record["num_calls"] != 2 * samples or record["step_calls"] != {str(s): samples for s in steps}:
                raise RuntimeError(f"Incomplete native step coverage: {expert}.{name}: {record['step_calls']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-view", type=Path, default=ROOT / ".runtime/model_view.json")
    ap.add_argument("--calibration", type=Path, default=ROOT / ".runtime/inputs/calibration12/canonical.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    view_info = json.loads(args.model_view.read_text())
    if view_info["status"] != "PASS":
        raise RuntimeError("Complete model verification before calibration")
    for row in view_info["files"]:
        path = Path(row["view_path"])
        if path.stat().st_size != row["bytes"] or not path.samefile(row["blob"]):
            raise RuntimeError(f"Model cache view changed: {path}")
    source_rows = json.loads(args.calibration.read_text())
    if len(source_rows) != 12:
        raise RuntimeError("Frozen calibration requires 12 samples")
    for index, sample in enumerate(source_rows):
        if sample["calibration_seed"] != 42 + index:
            raise RuntimeError("Changed original calibration seed/order")
        if digest(sample["img_paths"][0]) != sample["input_provenance"]["image_sha256"]:
            raise RuntimeError(f"Calibration image changed: {index}")
    identity = {"series": "WORKER_HIF4_SMOOTH_RTN_REBUILD_20261020",
                "model_view_sha256": digest(args.model_view), "calibration_sha256": digest(args.calibration),
                "script_sha256": digest(__file__), "adapter_sha256": digest(adapter.__file__),
                "native_source_sha256": {str(p): digest(p) for p in [
                    ROOT / ".runtime/code/LightX2V/lightx2v/common/ops/mm/mm_weight.py",
                    ROOT / ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py"]},
                "contract": adapter.DISTILL_RUNTIME_CONTRACT,
                "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "numpy", "triton"]},
                "hardware": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda,
                "statistics": "unchanged ActivationAbsmaxObserver; per-channel maximum over all native rows and four steps"}
    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out / "RUN.json"
    completed = []
    if report.exists():
        old = json.loads(report.read_text())
        if old["identity"] != identity:
            raise RuntimeError("Refusing calibration resume with a changed run identity")
        completed = old["completed"]
    done = set()
    for row in completed:
        if row["index"] in done or row["index"] not in range(12):
            raise RuntimeError("Invalid/duplicate calibration checkpoint")
        path = args.out / row["stats_file"]
        if digest(path) != row["stats_sha256"]:
            raise RuntimeError(f"Calibration checkpoint hash changed: {path}")
        validate_stats(torch.load(path, map_location="cpu", weights_only=True), 1)
        done.add(row["index"])
    meta = {"status": "RUNNING", "identity": identity, "completed": completed,
            "started_or_resumed_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(report, meta)
    try:
        pending = [i for i in range(12) if i not in done]
        if pending:
            base = Path(view_info["view"])
            weights = {noise: base / f"{noise}_noise_model" / f"wan2.2_i2v_A14b_{noise}_noise_lightx2v_4step.safetensors" for noise in ["high", "low"]}
            print(f"Initializing unchanged pipeline for {len(pending)} pending calibration inputs", flush=True)
            pipe = adapter.build_lightx2v_distill_pipeline(base_model_dir=base,
                       high_noise_ckpt=weights["high"], low_noise_ckpt=weights["low"])
            adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
            scheduler = pipe.runner.scheduler
            for index in pending:
                sample = source_rows[index]
                trajectory = []
                original_post = scheduler.step_post

                def observed_post():
                    result = original_post()
                    step = int(scheduler.step_index)
                    if not bool(torch.isfinite(scheduler.latents).all()):
                        raise RuntimeError(f"Non-finite calibration latents at input={index}, step={step}")
                    trajectory.append({"step": step, "denoising_step": int(scheduler.denoising_step_list[step]),
                                       "scheduler_timestep": float(scheduler.timesteps[step]),
                                       "sigma": float(scheduler.sigmas[step]), "latent_shape": list(scheduler.latents.shape)})
                    return result

                observer = adapter.ActivationAbsmaxObserver()
                adapter.activate_lightx2v_hooks(observer=observer)
                scheduler.step_post = observed_post
                video = args.out / f"calibration_{index:02d}.mp4"
                start = time.perf_counter()
                try:
                    with torch.inference_mode():
                        pipe.generate(seed=42 + index, image_path=sample["img_paths"][0], prompt=sample["prompt"],
                                      negative_prompt="", save_result_path=str(video.resolve()))
                finally:
                    scheduler.step_post = original_post
                    adapter.deactivate_lightx2v_hooks()
                if [t["step"] for t in trajectory] != [0, 1, 2, 3]:
                    raise RuntimeError("Calibration did not execute exactly four native steps")
                stats = observer.export(save_dtype=torch.float32)
                validate_stats(stats, 1)
                target = args.out / f"sample_{index:02d}_act.pt"
                if target.exists():
                    raise RuntimeError(f"Uncommitted calibration artifact exists; inspect before replacing: {target}")
                temp = target.with_suffix(".pt.tmp")
                torch.save(stats, temp)
                temp.replace(target)
                row = {"index": index, "videoid": sample["videoid"], "seed": 42 + index,
                       "stats_file": target.name, "stats_sha256": digest(target), "trajectory": trajectory,
                       "video_sha256": digest(video), "generation_seconds_including_observation": time.perf_counter() - start}
                completed.append(row)
                atomic_json(report, meta)
                print(f"COMPLETED calibration {index + 1}/12; atomic stats SHA256={row['stats_sha256']}", flush=True)
        if {row["index"] for row in completed} != set(range(12)):
            raise RuntimeError("Calibration is incomplete")
        merged = None
        for row in sorted(completed, key=lambda row: row["index"]):
            path = args.out / row["stats_file"]
            if digest(path) != row["stats_sha256"]:
                raise RuntimeError(f"Calibration checkpoint changed before merge: {path}")
            stats = torch.load(path, map_location="cpu", weights_only=True)
            validate_stats(stats, 1)
            if merged is None:
                merged = stats
                continue
            for expert in ["high_noise_model", "low_noise_model"]:
                for name, rec in stats[expert].items():
                    out = merged[expert][name]
                    out["s"] = torch.maximum(out["s"], rec["s"])
                    out["num_calls"] += rec["num_calls"]
                    for step, calls in rec["step_calls"].items():
                        out["step_calls"][step] += calls
        validate_stats(merged, 12)
        merged["__meta__"].update(num_calib_samples=12, calib_seed=42, run_identity=identity,
                                 step_calls={expert: {str(s): 4800 for s in steps} for expert, steps in
                                             [("high_noise_model", [0, 1]), ("low_noise_model", [2, 3])]})
        target = args.out / "act.pt"
        temp = target.with_suffix(".pt.tmp")
        torch.save(merged, temp)
        temp.replace(target)
        meta.update(status="PASS_FULL_NATIVE_CALIBRATION12", act_file=str(target.resolve()), act_sha256=digest(target),
                    completed_utc=datetime.now(timezone.utc).isoformat())
        print(json.dumps({"status": meta["status"], "act_sha256": meta["act_sha256"]}), flush=True)
    except BaseException as error:
        meta.update(status="FAILED_RESUMABLE", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(report, meta)


if __name__ == "__main__":
    main()
