"""Run a dense D-BF16 or original dense HiF4 D-Q four-step baseline.

The HF model view must already be complete. This is a baseline/smoke run, not
the formal20 evaluation. Post-step latents are observed and saved for audit;
wall time includes that observation and must not be reported as a benchmark.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from worker_identity import matches_worker
import sys
import time
import traceback

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    temp.replace(path)


def main(arm="D-BF16"):
    if arm not in ["D-BF16", "D-Q"]:
        raise ValueError(arm)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-view", type=Path, default=ROOT / ".runtime/model_view.json")
    ap.add_argument("--io-model-view", type=Path, help="Exact node-local HF byte mirror; auto-detect the compatible staged view")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--input-manifest", type=Path, help="A single recovered V5 input record")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--prompt", default="Two cats playfully box while the camera remains steady.")
    ap.add_argument("--source-id", default="wan_official_example_migration_smoke")
    ap.add_argument("--capture-ffn", action="store_true", help="Observe sampled native boundaries; compare against a separate plain run")
    ap.add_argument("--capture-blocks", type=int, nargs="+", default=[0, 10, 20, 38])
    ap.add_argument("--capture-rows", type=int, default=512)
    if arm == "D-Q":
        ap.add_argument("--cache-manifest", type=Path, required=True, help="Verified complete rebuilt HiF4 cache manifest")
        ap.add_argument("--packed-reference-policy", type=Path, help="Explicit static packed operator intervention; separate arm, no acceptance claim")
    args = ap.parse_args()
    packed_policy = getattr(args, "packed_reference_policy", None)
    if packed_policy and args.capture_ffn:
        ap.error("Dense boundary observer requires full hidden width; use a plain packed reference run")
    args.out = args.out.resolve()
    record = None
    if args.input_manifest:
        records = json.loads(args.input_manifest.read_text())
        if not isinstance(records, list) or len(records) != 1:
            ap.error("Expected exactly one input record")
        record = records[0]
        args.image = Path(record["image_path"])
        args.prompt = record["prompt"]
        args.seed = int(record["seed"])
        args.source_id = record["source_group"]
    args.out.mkdir(parents=True, exist_ok=False)
    meta = {"status": "RUNNING", "arm": arm, "source_id": args.source_id,
            "purpose": "migration smoke baseline; not formal20 or a performance benchmark",
            "started_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
            "prompt": args.prompt, "prompt_sha256": hashlib.sha256(args.prompt.encode()).hexdigest(),
            "python": sys.version, "trajectory": []}
    if record is not None:
        meta["input_record"] = record
    if packed_policy:
        meta.update(arm="D-Q-PACKED-REFERENCE", purpose="closed-loop packed operator reference, no speed/formal20 acceptance",
                    packed_policy_path=str(packed_policy.resolve()), packed_policy_sha256=digest(packed_policy))
    report = args.out / "run_meta.json"
    atomic_json(report, meta)
    try:
        view_info = json.loads(args.model_view.read_text())
        if view_info["status"] != "PASS":
            raise RuntimeError("Model view is not verified")
        for row in view_info["files"]:
            path = Path(row["view_path"])
            if path.stat().st_size != row["bytes"] or not path.samefile(row["blob"]):
                raise RuntimeError(f"Cached model view changed: {path}")
        io_path = args.io_model_view or ROOT / ".runtime/node_local_model_view.json"
        io_info = None
        if io_path.exists():
            candidate = json.loads(io_path.read_text())
            compatible = (candidate.get("status") == "PASS"
                and candidate.get("logical_model_view_sha256") == digest(args.model_view)
                and matches_worker(candidate.get("host")))
            if compatible:
                def asset_id(row):
                    return tuple(row[name] for name in ["repo", "revision", "filename", "bytes", "sha256"])
                if len(candidate["files"]) != 12 or {asset_id(row) for row in candidate["files"]} != {asset_id(row) for row in view_info["files"]}:
                    raise RuntimeError("Local I/O mirror differs from logical model asset identities")
                for row in candidate["files"]:
                    path = Path(row["view_path"])
                    if not row["copy_sha256_verified"] or path.stat().st_size != row["bytes"] or not path.samefile(row["blob"]):
                        raise RuntimeError(f"Local I/O byte mirror changed: {path}")
                io_info = candidate
                meta.update(io_model_view=str(io_path.resolve()), io_model_view_sha256=digest(io_path),
                            io_view_selection="VERIFIED_NODE_LOCAL_EXACT_BYTES")
            elif args.io_model_view:
                raise RuntimeError("Explicit local I/O view is incompatible with this model identity/host")
        elif args.io_model_view:
            raise FileNotFoundError(io_path)
        base = Path((io_info or view_info)["view"])
        weights = {noise: base / f"{noise}_noise_model" / f"wan2.2_i2v_A14b_{noise}_noise_lightx2v_4step.safetensors"
                   for noise in ["high", "low"]}
        if io_info is None:
            meta["io_view_selection"] = "LOGICAL_HF_MODEL_VIEW"
        image = args.image or base / "examples/i2v_input.JPG"
        if not image.is_file():
            raise FileNotFoundError(image)
        meta.update(image_path=str(image), image_sha256=digest(image),
                    model_identity_manifest=str(args.model_view.resolve()),
                    checkpoint_sha256={r["filename"]: r["sha256"] for r in view_info["files"] if r["filename"].endswith(".safetensors")},
                    versions={name: importlib.metadata.version(name) for name in
                              ["torch", "torchvision", "transformers", "diffusers", "safetensors", "numpy", "triton"]})
        if record and record.get("input_provenance", {}).get("image_sha256", meta["image_sha256"]) != meta["image_sha256"]:
            raise RuntimeError("Input image differs from its frozen provenance")
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        if getattr(adapter, "_ACTIVE_RUNTIME", None) is not None or getattr(adapter, "_PATCHED", False):
            raise RuntimeError("Baseline entry requires a fresh process with no previous HiF4 runtime/hooks")
        contract = dict(adapter.DISTILL_RUNTIME_CONTRACT)
        source_files = [Path(__file__), Path(sys.argv[0]).resolve(), Path(adapter.__file__), Path(adapter.__file__).with_name("lightx2v_wan_distill_contract.json")]
        lightx = ROOT / ".runtime/code/LightX2V/lightx2v"
        source_files += [lightx / p for p in ["pipeline.py", "common/ops/mm/mm_weight.py",
            "models/networks/wan/infer/transformer_infer.py", "models/schedulers/wan/step_distill/scheduler.py"]]
        meta.update(contract=contract, source_sha256={str(p): digest(p) for p in source_files},
                    device=torch.cuda.get_device_name(), cuda_runtime=torch.version.cuda)
        atomic_json(report, meta)
        print(f"Building unmodified distill pipeline for {arm}", flush=True)
        start = time.perf_counter()
        pipe = adapter.build_lightx2v_distill_pipeline(base_model_dir=base,
                  high_noise_ckpt=weights["high"], low_noise_ckpt=weights["low"])
        meta["pipeline_initialization_seconds"] = time.perf_counter() - start
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        meta["resolved_contract"] = {k: pipe.runner.config.get(k) for k in contract if k in pipe.runner.config}
        scheduler = pipe.runner.scheduler
        meta["scheduler_class"] = type(scheduler).__name__
        runtime = None
        cache_identity = None
        if arm == "D-Q":
            from rebuild_hif4_cache import digest as streamed_digest
            cache = json.loads(args.cache_manifest.read_text())
            if cache["status"] != "PASS_FULL_NATIVE_HIF4_RTN_CACHE":
                raise RuntimeError("D-Q requires the complete verified rebuilt HiF4 cache")
            if cache["identity"]["checkpoint_sha256"] != meta["checkpoint_sha256"] or cache["identity"]["model_view_sha256"] != digest(args.model_view):
                raise RuntimeError("Quantized cache and runtime distilled checkpoint identities differ")
            for path, expected in cache["identity"]["source_sha256"].items():
                if digest(path) != expected:
                    raise RuntimeError(f"Quantized cache source/backend changed: {path}")
            cache_path = Path(cache["path"])
            if cache_path.stat().st_size != cache["bytes"] or streamed_digest(cache_path) != cache["sha256"]:
                raise RuntimeError("HiF4 cache file changed")
            cache_identity = {"manifest_path": str(args.cache_manifest.resolve()), "manifest_sha256": digest(args.cache_manifest),
                              "path": str(cache_path), "sha256": cache["sha256"], "identity": cache["identity"]}
            if packed_policy:
                from packed_reference_runtime import create_runtime
                runtime = create_runtime(cache_path, packed_policy)
                meta["packed_runtime_source_sha256"] = digest(ROOT / "tools/packed_reference_runtime.py")
            else:
                runtime = adapter.HIF4CacheRuntime(cache_path)
            expected_names = {f"blocks.{b}.{name}" for b in range(40) for name in [
                "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k",
                "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2"]}
            if any(set(runtime.records[expert]) != expected_names for expert in ["high_noise_model", "low_noise_model"]):
                raise RuntimeError("Expected all 800 HiF4 linear records")
            meta["quantized_cache_identity"] = cache_identity
            adapter.activate_lightx2v_hooks(runtime=runtime)
        observer = None
        if args.capture_ffn:
            from bf16_observer import BF16Observer
            observer_kwargs = {"source_id": args.source_id, "seed": args.seed,
                               "blocks": args.capture_blocks, "rows": args.capture_rows}
            if arm == "D-Q":
                from hif4_observer import HiF4Observer
                observer = HiF4Observer(args.out / "ffn", runtime=runtime, cache_manifest=cache_identity, **observer_kwargs)
                meta["hif4_observer_source_sha256"] = digest(ROOT / "tools/hif4_observer.py")
            else:
                observer = BF16Observer(args.out / "ffn", **observer_kwargs)
            meta["observer_source_sha256"] = digest(ROOT / "tools/bf16_observer.py")
            meta["observer_manifest"] = str(args.out / "ffn/CAPTURE_MANIFEST.json")
        atomic_json(report, meta)
        original_step_post = scheduler.step_post

        def observed_step_post(*pos, **kw):
            result = original_step_post(*pos, **kw)
            step = int(scheduler.step_index)
            latent = scheduler.latents.detach().float().cpu().contiguous()
            if not torch.isfinite(latent).all():
                raise RuntimeError(f"Non-finite scheduler latents at step {step}")
            path = args.out / f"latent_step{step}.safetensors"
            save_file({"latents": latent}, str(path))
            meta["trajectory"].append({"step": step, "denoising_step": int(scheduler.denoising_step_list[step]),
                "scheduler_timestep": float(scheduler.timesteps[step]),
                "sigma": float(scheduler.sigmas[step]), "shape": list(latent.shape),
                "dtype": str(scheduler.latents.dtype), "finite": True, "path": str(path),
                "sha256": digest(path), "expert": "high_noise_model" if step < 2 else "low_noise_model"})
            atomic_json(report, meta)
            return result

        scheduler.step_post = observed_step_post
        video = args.out / "output.mp4"
        try:
            start = time.perf_counter()
            with torch.inference_mode(), observer if observer is not None else nullcontext():
                pipe.generate(seed=args.seed, image_path=str(image), prompt=args.prompt,
                              negative_prompt="", save_result_path=str(video))
            torch.cuda.synchronize()
            meta["generation_wall_seconds_including_capture"] = time.perf_counter() - start
        finally:
            scheduler.step_post = original_step_post
            if runtime is not None:
                meta["hif4_runtime"] = runtime.report()
                adapter.deactivate_lightx2v_hooks()
        assert [row["step"] for row in meta["trajectory"]] == [0, 1, 2, 3]
        assert [row["denoising_step"] for row in meta["trajectory"]] == [1000, 750, 500, 250]
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError("Pipeline did not write the requested video")
        import imageio.v2 as imageio
        with imageio.get_reader(video) as reader:
            info = reader.get_meta_data()
            frames = reader.count_frames()
        assert frames == 81, frames
        assert abs(float(info["fps"]) - 16) < 0.01, info
        if runtime is not None:
            counters = runtime.report()
            if counters["linear_forward_calls"] != 1600 or counters["activation_qdq_calls"] != 1600 or counters["cache_misses"]:
                raise RuntimeError(f"Incomplete HiF4 runtime coverage: {runtime.counters}")
            if packed_policy and counters["packed_reference"]["actual_FFN_linear_calls"] != 320:
                raise RuntimeError("Packed reference failed full 160-position FFN coverage")
        meta.update(status="PASS", video={"path": str(video), "sha256": digest(video), "frames": frames,
                    "fps": info["fps"], "size": info["size"]}, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                    hif4_runtime_active=getattr(adapter, "_ACTIVE_RUNTIME", None) is not None,
                    completed_utc=datetime.now(timezone.utc).isoformat())
        print(json.dumps({k: v for k, v in meta.items() if k in ["status", "video", "generation_wall_seconds_including_capture"]}, indent=2))
    except BaseException as error:
        meta.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(report, meta)


if __name__ == "__main__":
    main()
