"""Execute one registered FFNDS Dense/UB1 real video job with an original-noise receipt.

Requires the pinned LightX2V/HiF4 GPU environment already used by the repository.
Each invocation performs at most one generation. No automatic retries or implicit tuning.
"""
import argparse
import json
from pathlib import Path
import time

from ffnds_experiment_plans import load, save_new
from ffnds_cohort_preflight import audit


def job_preflight(plan, index, ub1_policy):
    if plan.get("phase") != "REGISTERED_REAL_GENERATION_PLAN":
        raise ValueError("Require a frozen real generation plan")
    check = audit(plan)
    if check["errors"]:
        raise ValueError("Invalid generation cohort: " + "; ".join(check["errors"]))
    if index not in range(len(plan["samples"])):
        raise ValueError("Registered job index out of range")
    job = plan["samples"][index]
    if job["status"] != "PLANNED":
        raise ValueError("Job must be planned, never already completed")
    if job["variant"] not in ("dense", "UB1"):
        raise ValueError("This native runner supports frozen dense and original UB1 only")
    if job["variant"] == "UB1" and (ub1_policy is None or not ub1_policy.is_file()):
        raise ValueError("UB1 needs its exact frozen step-budget policy")
    source = next(s for s in plan["sources"] if s["source_id"] == job["source_id"])
    if not Path(source["image_path"]).is_file():
        raise FileNotFoundError("Unavailable native I2V condition image: " + source["image_path"])
    if job["frames"] != 81 or job["fps"] != 16:
        raise ValueError("Current four-step LightX2V runner is registered for 81 frames / 16 fps")
    return job, source


def registration_check(plan, job, registration):
    if registration is None or not registration.is_file():
        raise ValueError("Real GPU execution requires a registered bounded experiment scope")
    scope = load(registration)
    if (scope.get("schema") != "ffnds.native.scope.v1"
            or scope.get("status") != "REGISTERED_BOUNDED_GPU_EXECUTION"
            or scope.get("experiment") != plan["experiment"]
            or job["run_id"] not in scope.get("allowed_run_ids", [])
            or scope.get("max_visible_GPUs") != 1
            or scope.get("user_authorization") != "USER_RESUME_OFFLINE_IFC_20261027"):
        raise ValueError("Frozen job is outside its registered model execution authority")
    from run_ar2_restart import ROOT
    active = load(ROOT / "research/ACTIVE_RESEARCH.json")
    latest = active.get("user_continuation_authorization_20261027", {})
    if latest.get("status") != "ACTIVE_CURRENT_USER_DIRECTED_SESSION" or latest.get("id") != scope["user_authorization"]:
        raise ValueError("Active research authority differs from registered scope")
    return scope


def run(plan, index, out_root, ub1_policy, registration):
    job, source = job_preflight(plan, index, ub1_policy)
    registration_check(plan, job, registration)
    # Explicit node ownership: the user's scheduler should expose exactly one GPU.
    import torch
    from safetensors.torch import save_file
    import imageio.v2 as imageio
    from distill_assets import verified_assets
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    from worker_identity import require_worker

    require_worker()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Run one registered job on one isolated visible worker GPU")
    torch.set_num_threads(8)
    kwargs, cache, assets = verified_assets()
    folder = Path(out_root) / f"job{index:06d}"
    folder.mkdir(parents=True, exist_ok=False)
    identity = {"job_index": index, "run_id": job["run_id"], "source_id": source["source_id"],
                "model_assets": assets, "variant": job["variant"],
                "method_policy_path": str(ub1_policy) if job["variant"] == "UB1" else None,
                "status": "NATIVE_RUN_STARTED"}
    save_new(folder / "STARTED.json", identity)
    started = time.perf_counter()
    pipe = None
    try:
        from run_ar2_restart import ROOT  # Preserve the repository's existing import path.
        _ = ROOT
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        if job["variant"] == "UB1":
            from step_budget_runtime import create_runtime
            runtime = create_runtime(cache["path"], ub1_policy)
        else:
            runtime = adapter.HIF4CacheRuntime(cache["path"])
        adapter.activate_lightx2v_hooks(runtime=runtime)
        scheduler = pipe.runner.scheduler
        original_post = scheduler.step_post
        scheduler.generator = None
        trajectory, initial = [], []
        before = dict(runtime.counters)
        if hasattr(runtime, "begin_request"):
            runtime.begin_request()

        def post():
            step = int(scheduler.step_index)
            if step == 0:
                if initial:
                    raise RuntimeError("Initial latent recorded more than once")
                tensor = scheduler.latents.detach().cpu().contiguous()
                save_file({"latents": tensor}, str(folder / "initial_noise.safetensors"))
                initial.append(list(tensor.shape))
            result = original_post()
            tensor = scheduler.latents.detach()
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError("Nonfinite native denoising state")
            trajectory.append({"step": step, "shape": list(tensor.shape)})
            return result

        scheduler.step_post = post
        video = folder / "video.mp4"
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                pipe.generate(seed=job["seed"], image_path=source["image_path"],
                              prompt=source["prompt"], negative_prompt="",
                              save_result_path=str(video))
            torch.cuda.synchronize()
        finally:
            scheduler.step_post = original_post
            adapter.deactivate_lightx2v_hooks()
        if len(initial) != 1 or [p["step"] for p in trajectory] != list(range(4)):
            raise RuntimeError("Incomplete original native four-step latent trajectory")
        counters = {k: runtime.counters[k] - before[k] for k in before}
        if (counters.get("linear_forward_calls") != 1600
                or counters.get("activation_qdq_calls") != 1600
                or counters.get("cache_misses") != 0):
            raise RuntimeError("Native all-block FFN/QDQ coverage differs from registered Wan4step")
        if job["variant"] == "UB1" and len(runtime.coverage) != 320:
            raise RuntimeError("UB1 full160 native FFN sites were not all visited")
        with imageio.get_reader(video) as reader:
            frames, info = reader.count_frames(), reader.get_meta_data()
        size = info["size"]
        if (frames != job["frames"] or abs(float(info["fps"]) - job["fps"]) > .01
                or size != (job["width"], job["height"])):
            raise RuntimeError("Actual decoded video geometry/fps differs from frozen job")
        receipt = {
            "schema": "ffnds.registered.native.video.v1",
            "status": "COMPLETE_NATIVE_GENERATION_ONLY",
            "job": job, "source": source,
            "actual_initial_noise_file": str(folder / "initial_noise.safetensors"),
            "initial_noise_shape": initial[0], "video_path": str(video),
            "decoded_frames": frames, "decoded_fps": float(info["fps"]), "decoded_size": list(size),
            "native_step_trace": trajectory, "native_ffn_counters": counters,
            "native_ffn_sites": 160, "model_assets": assets,
            "generation_wall_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scheduler_generator_reset": True,
            "official_quality": "NOT_RUN", "quality_accepted": False}
        save_new(folder / "RUN.json", receipt)
        return receipt
    except BaseException as exc:
        save_new(folder / "FAILED.json", {"status": "FAILED_PRESERVED", "run_id": job["run_id"],
                                          "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--job-index", type=int, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--ub1-policy", type=Path)
    p.add_argument("--registration", type=Path,
                   help="Explicit frozen, bounded model-execution registration; required for GPU runs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    plan = load(args.plan)
    job, source = job_preflight(plan, args.job_index, args.ub1_policy)
    if args.dry_run:
        print(json.dumps({"status": "NATIVE_INPUT_PREFLIGHT_ONLY", "job": job,
                          "source": source["source_id"]}, ensure_ascii=False))
        return
    record = run(plan, args.job_index, args.out_root, args.ub1_policy, args.registration)
    print(json.dumps({"status": record["status"], "run_id": job["run_id"],
                      "video_path": record["video_path"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
