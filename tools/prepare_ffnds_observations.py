"""Original A100 discovery preparation and explicit single-source native collection."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

from ffnds_observation_common import audit_headers, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "config/observation_20261025_01a16d3d.json"


def validate_protocol(p):
    for name, upper in [("steps", 4), ("blocks", 40), ("weight_panel_groups", 216)]:
        values = p[name]
        if not values or any(type(v) is not int or not 0 <= v < upper for v in values) or len(set(values)) != len(values):
            raise ValueError("invalid protocol axis: " + name)
    if p["steps"] != [0, 1, 2, 3] or p["native_generations_per_source"] != 2 or p["max_sources_per_invocation"] != 1:
        raise ValueError("driver supports one source and two complete four-step phases")
    if type(p["uniform_rows"]) is not int or not 1 <= p["uniform_rows"] <= 256:
        raise ValueError("invalid row count")


def preflight(root, protocol, inputs=None):
    files = {}
    for name, expected in protocol["native_source_sha256"].items():
        path = root / name
        actual = sha256(path) if path.is_file() else None
        files[name] = {"exists": path.is_file(), "sha256": actual, "match": actual == expected}
    required = [".runtime/model_view.json", ".runtime/node_local_model_view.json",
                ".runtime/hif4_rebuilt_cache.json"]
    manifests = {name: (root / name).is_file() for name in required}
    input_path = inputs or root / protocol["input_default"]
    manifests[str(input_path)] = input_path.is_file()
    deps = {name: importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "safetensors", "imageio")}
    return {"status": "PREPARED_STATIC_PREFLIGHT" if all(r["match"] for r in files.values()) and all(manifests.values()) and all(deps.values()) else "ENVIRONMENT_INCOMPLETE",
            "native_sources": files, "manifests_present": manifests, "dependencies_present": deps,
            "model_loaded": False, "gpu_initialized": False,
            "native_observer_validated": False}


def collect(args, protocol):
    if not args.execute_native:
        raise ValueError("collect is a native GPU operation; specify --execute-native only when running the prepared study")
    result = preflight(ROOT, protocol, args.inputs)
    if result["status"] != "PREPARED_STATIC_PREFLIGHT":
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    # Existing repository validation remains authoritative; no SSH, scheduler or retry.
    for name in ("HiF4_Sp", "LightX2V", "Wan2.2-I2V-A14B-W4A4"):
        sys.path.insert(0, str(ROOT / ".runtime/code" / name))
    sys.path.insert(0, str(ROOT / ".runtime/code/HiF4_Sp/HiFloat4"))
    sys.path.insert(0, str(ROOT / ".runtime/code/HiF4_Sp/HiFloat4/hif4_gpu"))
    import torch
    from contextlib import nullcontext
    from historical_source_snapshot import HistoricalSources
    from restored_assets import verified_restored_assets
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    from ffnds_observation_capture import DiscoveryCapture

    device_name = torch.cuda.get_device_name(0)
    if "A100" not in device_name:
        raise RuntimeError("Prepared target is original A100, got " + device_name)

    inputs = args.inputs or ROOT / protocol["input_default"]
    samples = json.loads(inputs.read_text())
    if not 0 <= args.index < len(samples):
        raise ValueError("source index out of range")
    sample = samples[args.index]
    image = Path(sample.get("image_path") or sample["img_paths"][0])
    seed = sample.get("seed", sample.get("calibration_seed"))
    expected_image = sample["input_provenance"]["image_sha256"]
    if type(seed) is not int or sha256(image) != expected_image:
        raise ValueError("input image/seed identity mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.out).free < protocol["disk_reserve_gib"] * (1 << 30):
        raise RuntimeError("Preserve configured disk reserve")
    history = HistoricalSources()
    kwargs, cache, assets = verified_restored_assets(history)
    record = {"status": "STARTED_NOT_ACCEPTED", "protocol_sha256": sha256(args.protocol),
              "source_id": expected_image, "source_index": args.index, "seed": seed,
              "inputs_sha256": sha256(inputs), "assets": assets, "preflight": result,
              "collector_sha256": {p.name: sha256(p) for p in [
                  Path(__file__), ROOT / "tools/ffnds_observation_capture.py",
                  ROOT / "tools/ffnds_observation_common.py"]},
              "phases": {}, "quality_or_speed_acceptance": False,
              "native_environment": {"device": device_name, "torch": torch.__version__,
                                     "cuda_runtime": torch.version.cuda}}
    write_json(args.out / "RUN.json", record)
    torch.set_num_threads(8)
    pipe = None
    try:
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        runtime = adapter.HIF4CacheRuntime(cache["path"])
        adapter.activate_lightx2v_hooks(runtime=runtime)
        observer = DiscoveryCapture(runtime, args.out / "observed", expected_image, protocol, assets)
        scheduler = pipe.runner.scheduler
        for phase in ("plain", "observed"):
            if shutil.disk_usage(args.out).free < protocol["disk_reserve_gib"] * (1 << 30):
                raise RuntimeError("Disk reserve reached")
            old_post, trajectory = scheduler.step_post, []
            def post():
                value = old_post()
                latent = scheduler.latents.detach().float().cpu().contiguous()
                if not bool(torch.isfinite(latent).all()):
                    raise RuntimeError("Non-finite latent")
                import hashlib
                digest = hashlib.sha256(latent.view(torch.uint8).numpy().tobytes()).hexdigest()
                trajectory.append({"step": int(scheduler.step_index), "sha256": digest,
                                   "shape": list(latent.shape)})
                return value
            scheduler.step_post = post
            scheduler.generator = None  # same seed must restart the original generator
            video = args.out / (phase + ".mp4")
            before, started = dict(runtime.counters), time.perf_counter()
            try:
                with torch.inference_mode(), (observer if phase == "observed" else nullcontext()):
                    pipe.generate(seed=seed, image_path=str(image), prompt=sample["prompt"],
                                  negative_prompt="", save_result_path=str(video))
                torch.cuda.synchronize()
            finally:
                scheduler.step_post = old_post
            counters = {k: runtime.counters[k] - before[k] for k in before}
            if counters["linear_forward_calls"] != 1600 or counters["activation_qdq_calls"] != 1600 or counters["cache_misses"]:
                raise RuntimeError("Original dense native call counts changed")
            if [r["step"] for r in trajectory] != [0, 1, 2, 3]:
                raise RuntimeError("Incomplete four-step trajectory")
            record["phases"][phase] = {"trajectory": trajectory, "video_sha256": sha256(video),
                                       "native_counts": counters,
                                       "elapsed_including_capture": time.perf_counter() - started}
            write_json(args.out / "RUN.json", record)
        a, b = record["phases"]["plain"], record["phases"]["observed"]
        if a["trajectory"] != b["trajectory"] or a["video_sha256"] != b["video_sha256"]:
            raise RuntimeError("Observer changed native trajectory or video bytes")
        expected = {(s, b) for s in protocol["steps"] for b in protocol["blocks"]}
        got = {(r["step"], r["block"]) for r in observer.scopes}
        if got != expected or len(observer.scopes) != len(expected):
            raise RuntimeError("Incomplete scope coverage")
        record.update(status="PASS_PASSIVE_CAPTURE_NOOP", scopes=observer.scopes,
                      no_op_scope="same initialized pipeline, freshly seeded requests; not independent processes",
                      extra_teacher_ffn_calls=0)
    except BaseException as error:
        record.update(status="FAILED_NOT_ACCEPTED", error=str(error))
        raise
    finally:
        adapter.deactivate_lightx2v_hooks()
        write_json(args.out / "RUN.json", record)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["preflight", "headers", "collect"])
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--files", nargs="*", type=Path, default=[])
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--execute-native", action="store_true")
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    if args.command == "preflight":
        write_json(args.out, preflight(args.root, protocol, args.inputs))
    elif args.command == "headers":
        write_json(args.out, audit_headers(args.files))
    else:
        if args.root.resolve() != ROOT.resolve():
            raise ValueError("collect must run inside the target checkout")
        collect(args, protocol)


if __name__ == "__main__":
    main()

