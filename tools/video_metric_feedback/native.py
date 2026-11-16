"""Roll out one candidate inside an existing, registered native producer.

``generate_batch`` takes an already-built LightX2V pipeline.  Its caller owns
checkpoint/LoRA loading, the registered scope, allocation and runtime identity
checks.  This module does not launch a job, approve a scope or change ACTIVE.
It connects the complete matrix manifest to the archived q1_r2 native FFN
binding, creates fresh anchor caches for every sample, and returns videos for
subsequent metric evaluation.  It is not an alternative command-line runner.

The real adapter is deliberately limited to the existing four-step HHLL,
40-layer, width-5120 path and 832x464 / 81-frame / 16-fps video decoder contract.
The default FFN input shape is (31668, 5120).  A recipe can state that shape as
``ffn_input_shape`` (or ``expected_shape``).  Small dimensions are available
only to an explicitly synthetic injected runtime for lifecycle unit tests.

Baseline execution retains native prefetch and the original binding's gate
placement.  Its timing includes instrumentation and is not a speed benchmark.
PyTorch and LightX2V imports are lazy; importing this module runs no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable

from .matrices import preflight_manifest, sha256_file


_LOCK = threading.Lock()
_SHA = re.compile(r"[0-9a-f]{64}")
_SCHEDULE = (0, 0, 1, 1)
_DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "reports/20261107/g19/source/q1_r2"


@dataclass
class NativeRuntime:
    """Dependencies injected by tests, or loaded from the existing native code.

    ``synthetic=True`` marks fixture outputs and permits reduced dimensions.
    The production loader supplies the archived classes and helper functions;
    it does not replace their FFN arithmetic with a local implementation.
    """

    torch: Any
    binding_cls: Any
    matrices_cls: Any
    transformer_cls: Any
    dispatcher_cls: Any
    reset_scheduler_generator: Callable
    finite_tensor_sha: Callable
    video_digest: Callable
    device: Any = "cuda:0"
    synthetic: bool = False
    verify_phase: Callable | None = None
    source_identity: dict = field(default_factory=dict)


def load_native_runtime(source_dir: str | Path | None = None) -> NativeRuntime:
    """Load the actual q1_r2 implementation without invoking its runner.

    The archived files use absolute sibling imports.  Temporary aliases keep
    those imports local and restore any caller modules with the same names.
    Call this before concurrent model use; ``generate_batch`` serializes its
    own runtime loading and process-wide class patches.
    """
    source = Path(source_dir or _DEFAULT_SOURCE).resolve()
    names = ("runtime_guard", "direct", "binding", "resident", "legacy_helpers")
    paths = {name: source / f"{name}.py" for name in names}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Native source missing from full repository checkout: {path}")
    import torch
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
    from lightx2v.common.ops.mm.mm_weight import MMWeight

    if not torch.cuda.is_available():
        raise RuntimeError("Native rollout requires the caller's CUDA model allocation")
    file_hashes = {name: sha256_file(path) for name, path in paths.items()}
    source_key = json.dumps({"directory": str(source), "files": file_hashes}, sort_keys=True)
    namespace = "_ffnds_video_feedback_native_" + hashlib.sha256(source_key.encode()).hexdigest()[:16]
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in names}
    modules = {}
    created = []
    try:
        for name, path in paths.items():
            private_name = f"{namespace}_{name}"
            if private_name in sys.modules:
                module = sys.modules[private_name]
                sys.modules[name] = module
                modules[name] = module
                continue
            spec = importlib.util.spec_from_file_location(private_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load native source {path}")
            module = importlib.util.module_from_spec(spec)
            # exec_module does not register the spec's name.  Dataclasses with
            # postponed annotations, inspect and later class lookup need it.
            sys.modules[private_name] = module
            created.append(private_name)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    except BaseException:
        for private_name in created:
            sys.modules.pop(private_name, None)
        raise
    finally:
        for name, original in previous.items():
            if original is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def verify_phase(phase):
        # The native runner checks these physical weights at anchor sites.
        for weight, shape in ((phase.ffn_0, (5120, 13824)),
                              (phase.ffn_2, (13824, 5120))):
            if (type(weight) is not MMWeight or weight.bias is None
                    or getattr(weight, "has_lora_branch", True)
                    or hasattr(weight, "weight_diff") or hasattr(weight, "bias_diff")):
                raise RuntimeError("Unsupported native FFN weight")
            if tuple(weight.bias.shape) != (shape[1],) or weight.bias.device.type != "cuda":
                raise RuntimeError("Native FFN bias mismatch")
            actual = weight._get_actual_weight()
            buffer = getattr(weight, "weight_cuda_buffer", None)
            if (buffer is None or actual.device.type != "cuda"
                    or tuple(actual.shape) != shape or actual.data_ptr() != buffer.data_ptr()):
                raise RuntimeError("Native FFN buffer mismatch")

    helper = modules["legacy_helpers"]
    return NativeRuntime(
        torch=torch, binding_cls=modules["binding"].NativeBinding,
        matrices_cls=modules["resident"].ResidentMatrices,
        transformer_cls=WanTransformerInfer, dispatcher_cls=MultiDistillModelStruct,
        reset_scheduler_generator=helper.reset_scheduler_generator,
        finite_tensor_sha=helper.finite_tensor_sha, video_digest=helper.video_digest,
        device=torch.device("cuda:0"), verify_phase=verify_phase,
        source_identity={"source_directory": str(source), "torch_version": torch.__version__,
                         "files_sha256": file_hashes, "module_namespace": namespace},
    )


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _recipe_contract(recipe: dict) -> dict:
    if not isinstance(recipe, dict):
        raise ValueError("Native oracle_identity.recipe must be an object")
    shape = recipe.get("ffn_input_shape", recipe.get("expected_shape", [31668, 5120]))
    if ("ffn_input_shape" in recipe and "expected_shape" in recipe
            and recipe["ffn_input_shape"] != recipe["expected_shape"]):
        raise ValueError("Conflicting FFN input shapes in recipe")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError("ffn_input_shape must contain token count and width")
    shape = [_positive_integer(x, "FFN shape dimension") for x in shape]
    layers = _positive_integer(recipe.get("num_layers", 40), "num_layers")
    if recipe.get("infer_steps", 4) != 4:
        raise ValueError("Native adapter supports exactly four inference steps")
    if recipe.get("expert_schedule", ["high", "high", "low", "low"]) != ["high", "high", "low", "low"]:
        raise ValueError("Native adapter requires the HHLL expert schedule")
    if recipe.get("execution", "baseline") != "baseline":
        raise ValueError("Native adapter uses baseline execution")
    for key, expected in {"video_width": 832, "video_height": 464,
                          "video_frames": 81, "video_fps": 16}.items():
        if recipe.get(key, expected) != expected:
            raise ValueError(f"Native decoder requires {key}={expected}")
    return {"ffn_input_shape": shape, "num_layers": layers, "infer_steps": 4,
            "expert_schedule": ["high", "high", "low", "low"], "execution": "baseline",
            "video_width": 832, "video_height": 464, "video_frames": 81, "video_fps": 16,
            "default_ffn_input_shape_used": "ffn_input_shape" not in recipe and "expected_shape" not in recipe}


def _validate_request(request: dict) -> tuple[dict, list[dict]]:
    if request.get("schema_version") != 1 or request.get("mode") != "video":
        raise ValueError("Native adapter requires a schema_version=1 video request")
    if not isinstance(request.get("candidate_id"), str) or not request["candidate_id"]:
        raise ValueError("Missing candidate_id")
    for key in ("execution_fingerprint", "evaluation_key"):
        if not isinstance(request.get(key), str) or not _SHA.fullmatch(request[key]):
            raise ValueError(f"Missing or malformed {key}")
    identity = request.get("oracle_identity")
    if not isinstance(identity, dict):
        raise ValueError("Missing oracle_identity")
    contract = _recipe_contract(identity.get("recipe", {}))
    cohort = request.get("cohort")
    if not isinstance(cohort, list) or not cohort:
        raise ValueError("Native cohort must be nonempty")
    seen = set()
    rows = []
    for row in cohort:
        if not isinstance(row, dict):
            raise ValueError("Each cohort sample must be an object")
        for key in ("source_id", "image_id", "image_path", "prompt"):
            if not isinstance(row.get(key), str) or (key != "prompt" and not row[key]):
                raise ValueError(f"Invalid cohort {key}")
        if not isinstance(row.get("negative_prompt", ""), str):
            raise ValueError("negative_prompt must be a string")
        seed, sample_index = row.get("seed"), row.get("sample_index", 0)
        if (isinstance(seed, bool) or not isinstance(seed, int)
                or isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0):
            raise ValueError("Invalid cohort seed/sample_index")
        key = (row["source_id"], row["image_id"], seed, sample_index)
        if key in seen:
            raise ValueError("Duplicate cohort sample")
        seen.add(key)
        path = Path(row["image_path"]).expanduser().resolve()
        image_sha = sha256_file(path)
        prompt_sha = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        if row.get("image_sha256", image_sha) != image_sha:
            raise ValueError("Cohort source image SHA256 mismatch")
        if row.get("prompt_sha256", prompt_sha) != prompt_sha:
            raise ValueError("Cohort prompt SHA256 mismatch")
        noise_sha = row.get("initial_noise_sha256")
        if noise_sha is not None and (not isinstance(noise_sha, str) or not _SHA.fullmatch(noise_sha)):
            raise ValueError("Malformed expected initial_noise_sha256")
        rows.append({**row, "image_path": str(path), "sample_index": sample_index,
                     "image_sha256": image_sha, "prompt_sha256": prompt_sha})
    return contract, rows


def _run_sample(pipe, row: dict, video: Path, matrices, runtime: NativeRuntime,
                contract: dict) -> dict:
    torch = runtime.torch
    owner, dispatcher = runtime.transformer_cls, runtime.dispatcher_cls
    scheduler = pipe.runner.scheduler
    models = pipe.runner.model.model
    if len(models) != 2:
        raise RuntimeError("Expected exactly two native experts")
    for model in models:
        model.transformer_infer.offload_manager.need_init_first_buffer = True
    layers = contract["num_layers"]

    def on_site(model, phase, step, expert, layer):
        if runtime.verify_phase is not None and step in (0, 2):
            runtime.verify_phase(phase)

    # A new binding owns a new DirectFFN slab for this entire sample.
    binding = runtime.binding_cls("g19", matrices, layers=layers,
                                  expected_shape=tuple(contract["ffn_input_shape"]),
                                  execution="baseline", on_site=on_site)
    original_dispatch = dispatcher.get_current_model_index
    original_block, original_ffn = owner.infer_block, owner.infer_ffn
    dispatch, sites, initial = [], [], []
    expected_sites = [(step, layer) for step in range(4) for layer in range(layers)]

    def dispatch_call(model):
        value = original_dispatch(model)
        pair = (int(model.scheduler.step_index), int(model.cur_model_index))
        index = len(dispatch)
        if index >= 4 or pair != (index, _SCHEDULE[index]):
            raise RuntimeError("Actual dispatch differs from four-step HHLL")
        dispatch.append(pair)
        if pair[0] == 0:
            noise = runtime.finite_tensor_sha(model.scheduler.latents.detach().clone(), torch)
            if row.get("initial_noise_sha256", noise["sha256"]) != noise["sha256"]:
                raise RuntimeError("Initial noise mismatch before native forward")
            initial.append(noise)
        binding.dispatch = pair[1]
        return value

    def block_call(model, *args, **kwargs):
        pair = (int(model.scheduler.step_index), int(model.block_idx))
        index = len(sites)
        if index >= len(expected_sites) or pair != expected_sites[index]:
            raise RuntimeError("Unexpected native block site order")
        sites.append(pair)
        return original_block(model, *args, **kwargs)

    try:
        dispatcher.get_current_model_index = dispatch_call
        owner.infer_block = block_call
        binding.install(owner)
        runtime.reset_scheduler_generator(scheduler, pipe.runner.model)
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.no_grad():
            pipe.generate(seed=row["seed"], image_path=row["image_path"], prompt=row["prompt"],
                          negative_prompt=row.get("negative_prompt", ""), save_result_path=str(video))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        counts = binding.finish()
        final_latent = runtime.finite_tensor_sha(scheduler.latents, torch)
    finally:
        try:
            binding.uninstall()
        finally:
            # Restore all three methods even if install/uninstall raises.
            owner.infer_ffn, owner.infer_block = original_ffn, original_block
            dispatcher.get_current_model_index = original_dispatch
    if len(dispatch) != 4 or sites != expected_sites or len(initial) != 1:
        raise RuntimeError("Incomplete native dispatch/site/initial-noise coverage")
    expected_counts = {"dense": 0, "anchor": 2 * layers, "hit": 2 * layers,
                       "native_ffn": 2 * layers, "predictions": 4 * layers}
    if counts != expected_counts:
        raise RuntimeError("Native FFN action counts differ from HHLL contract")
    video_info = runtime.video_digest(video)
    if any(video_info.get(key) != contract[recipe_key] for key, recipe_key in
           (("frames", "video_frames"), ("width", "video_width"), ("height", "video_height"))):
        raise RuntimeError("Decoded video geometry differs from recipe")
    if abs(float(video_info.get("fps", 0)) - contract["video_fps"]) > 0.01:
        raise RuntimeError("Decoded video FPS differs from recipe")
    return {"source_id": row["source_id"], "image_id": row["image_id"], "seed": row["seed"],
            "sample_index": row["sample_index"], "image_path": row["image_path"],
            "image_sha256": row["image_sha256"], "prompt_sha256": row["prompt_sha256"],
            "video_path": str(video), "initial_noise_sha256": initial[0]["sha256"],
            "initial_noise": initial[0], "final_latent": final_latent, "counts": counts,
            "dispatch": [list(pair) for pair in dispatch], "site_count": len(sites),
            "video": video_info, "generation_wall_seconds": elapsed}


def generate_batch(pipe, request: dict, output_dir: str | Path, *,
                   runtime: NativeRuntime | None = None) -> dict:
    """Generate the complete fixed cohort with one candidate's exact matrices.

    ``request`` is the schema emitted by ``workflow.ExternalOracle``.  The
    caller adds metrics to the returned records in its producer; this function
    writes ``videos_manifest.json`` and never claims measured video quality.
    Existing video/manifest outputs are not overwritten.  A failed rollout
    leaves its partial files for diagnosis and closes matrices/restores patches.
    Each new invocation must use a fresh output directory for its video files.
    """
    request = json.loads(json.dumps(request, allow_nan=False))
    contract, rows = _validate_request(request)
    exported = request.get("matrix_export", {})
    manifest = Path(request["matrix_manifest_path"]).expanduser().resolve()
    manifest_sha = sha256_file(manifest)
    if exported.get("matrix_manifest_sha256") != manifest_sha:
        raise ValueError("Exported matrix manifest SHA256 differs from request")
    if exported.get("fingerprint") != request["execution_fingerprint"]:
        raise ValueError("Matrix execution fingerprint differs from request")
    preflight = preflight_manifest(manifest, expected_width=contract["ffn_input_shape"][1],
                                   expected_layers=contract["num_layers"], verify_hashes=True)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in entries.values():
        path = Path(entry["path"]).expanduser()
        entry["path"] = str((path if path.is_absolute() else manifest.parent / path).resolve())
    output = Path(output_dir).expanduser().resolve()
    manifest_output = output / "videos_manifest.json"
    video_paths = [output / "videos" / f"sample_{index:04d}.mp4" for index in range(len(rows))]
    required_config = {"infer_steps": 4, "num_layers": contract["num_layers"],
                       "boundary_step_index": 2, "enable_cfg": False}
    recipe_config = request["oracle_identity"].get("recipe", {}).get("required_runtime_config", {})
    if not isinstance(recipe_config, dict):
        raise ValueError("recipe.required_runtime_config must be an object")
    if any(key in required_config and value != required_config[key] for key, value in recipe_config.items()):
        raise ValueError("Recipe runtime config conflicts with the native HHLL contract")
    required_config.update(recipe_config)
    for key, value in required_config.items():
        if pipe.runner.config.get(key) != value:
            raise RuntimeError(f"Pipeline runtime config differs: {key}")
    records = []
    with _LOCK:
        # Recheck after waiting for other calls: their videos may have appeared
        # while this request was validating its input and matrix manifests.
        if manifest_output.exists() or any(path.exists() for path in video_paths):
            raise FileExistsError("Native output already contains candidate videos or manifest")
        runtime = runtime or load_native_runtime()
        if not runtime.synthetic and (contract["num_layers"] != 40 or contract["ffn_input_shape"][1] != 5120):
            raise ValueError("Production ResidentMatrices requires 40 layers and width 5120")
        (output / "videos").mkdir(parents=True, exist_ok=True)
        matrices = runtime.matrices_cls(entries, sha256_file, runtime.device)
        try:
            for row, video in zip(rows, video_paths):
                records.append(_run_sample(pipe, row, video, matrices, runtime, contract))
        finally:
            matrices.close()
    result = {"schema_version": 1, "schema": "ffnds.video_metric_feedback.videos.v1",
              "candidate_id": request["candidate_id"], "evaluation_key": request["evaluation_key"],
              "execution_fingerprint": request["execution_fingerprint"],
              "parameters": request.get("parameters", {}), "oracle_identity": request["oracle_identity"],
              "synthetic": runtime.synthetic, "status": "VIDEOS_GENERATED_METRICS_PENDING",
              "matrix_manifest_path": str(manifest), "matrix_manifest_sha256": manifest_sha,
              "matrix_preflight": preflight, "runtime_source_identity": runtime.source_identity,
              "effective_contract": contract, "checked_runtime_config": required_config,
              "cache_lifecycle": "new ResidentMatrices per candidate; new NativeBinding and anchor slab per sample",
              "transfer": "NATIVE_FULL_PREFETCH",
              "timing_note": "Includes noise hashing and other instrumentation; not a speed benchmark",
              "records": records, "manifest_path": str(manifest_output)}
    with manifest_output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result
