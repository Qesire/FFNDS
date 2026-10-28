"""Four Dense final-block captures; default is a pure CPU provenance check.

Two exposed development sources, seed42, steps0/3. Native input encoders and
prepare run once per source. Step3 imports the exact historical Dense post2.
No decode/video, training, decomposition, head replay or sensitivity probe.
"""
import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import time
import traceback

from b_final_block_capture import (D, GRID, M, OUT, SHAPE, STEPS, FinalBlockTap,
    cpu_copy, import_post_state, patch_method, raw_sha, read_state_bytes, scheduler_state,
    state_sha, tensor_record, tree_state)
from ub1_consumer_runtime import confined, sha256, verify_binding
from check_e0_full_depth_interface import REQUIRED_CONFIG


ROOT = Path(__file__).resolve().parents[1]
ID = "E0_B_FINAL_BLOCK_NATIVE_CAPTURE_HEAD_ASSET_20261028"
INDICES = (1, 4)
AUTHORIZATION = "USER_RESUME_OFFLINE_IFC_20261027"
LIMITS = {"input_encoder_requests": 2, "T5_positive_prompt_encodes": 2, "VAE_encodes": 2,
          "CLIP_encodes": 0, "scheduler_prepare": 2, "FP32_initial_noise_draws": 2,
          "historical_post2_BF16_imports": 2, "scheduler_step_pre": 4, "scheduler_step_post": 4,
          "complete_Dense_flows": 4, "native_blocks": 160, "native_FFN_calls": 160,
          "native_FFN_linears": 320, "native_FFN_activation_QDQ": 320,
          "native_non_FFN_linears": 1280, "native_non_FFN_activation_QDQ": 1280,
          "pre_infer_calls": 4, "pre_Default_MM_calls": 20, "patch_Conv3D_calls": 4,
          "final_LayerNorm_calls": 4, "head_Default_MM_calls": 4, "post_unpatchify_calls": 4,
          "setup_decoded_weight_GPU_transfers": 800, "setup_decoded_weight_bytes": 56203673600,
          "full_u_files": 4, "capture_packs": 4, "condition_packs": 2, "head_asset_packs": 2,
          "prepared_FP32_noise_files": 2,
          "CPU_BF16_writeback_mul": 4, "CPU_BF16_writeback_inplace_add": 4,
          "CPU_BF16_writeback_mul_elements": 4 * 256 * D,
          "CPU_BF16_writeback_add_elements": 4 * 256 * D,
          "head_replays": 0, "sensitivity_probes": 0, "shadow_FFN_calls": 0,
          "VAE_decodes": 0, "video_generations": 0, "training_updates": 0,
          "decompositions": 0, "weight_quantizations": 0, "official_quality_calls": 0}
RESOURCES = {"GPU": 1, "global_max_idle_GPU": 4, "CPU_threads": 2,
             "compute_wall_seconds": 1200, "external_wall_seconds": 1500,
             "new_HF_bytes": 2 << 30, "HF_free_min_bytes": 20 << 30,
             "host_available_min_bytes": 128 << 30, "resident_weight_max_bytes": 64 << 30,
             "reference_original_peak_physical_MiB": 21390, "memory_safety_bytes": 2 << 30}
LIMITS.update(CPU_original_head_view_file_loads=2, CPU_original_head_tensor_validations=6,
              CPU_head_weight_transpose_copies=2, CPU_head_weight_transpose_elements=655360)
SOURCE_EXPOSURE = "PREVIOUSLY_EXPOSED_DEVELOPMENT_COHORT_N2_IMAGES_SHARED_SEED42"
REQUIRED_SOURCES = {"tools/b_final_block_capture.py", "tools/e0_b_final_block_capture.py",
    "tests/test_b_final_block_capture.py", "tools/ffnds_experiment_plans.py", "tools/restored_assets.py",
    "tools/b_head_asset_export.py", "tools/e0_b_final_block_capture_head_asset.py",
    "tools/e0_b_final_block_capture_head_asset_local.py", "tests/test_b_head_asset_export.py",
    "tools/historical_source_snapshot.py", "tools/ffnds_all_weight_residency.py", "tools/e0_whole_flow_counterfactual.py",
    ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill.py",
    ".runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/lightx2v_wan_distill_contract.json",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/pre_infer.py",
    ".runtime/code/LightX2V/lightx2v/models/networks/wan/infer/post_infer.py",
    ".runtime/code/LightX2V/lightx2v/models/runners/default_runner.py",
    ".runtime/code/LightX2V/lightx2v/models/runners/wan/wan_distill_runner.py",
    ".runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py",
    ".runtime/code/LightX2V/lightx2v/models/schedulers/wan/scheduler.py",
    ".runtime/code/LightX2V/lightx2v/models/schedulers/scheduler.py",
    ".runtime/code/LightX2V/lightx2v/common/ops/mm/mm_weight.py",
    ".runtime/code/LightX2V/lightx2v/common/ops/norm/layer_norm_weight.py"}


def run_ids():
    return [f"b_final_capture_source{i:02d}_seed42_step{s}" for i in INDICES for s in STEPS]


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verify_actual_head_asset(capture, original_head):
    actual = capture["actual_head_operator"]
    if (actual["weight"]["raw_tensor_sha256"] != original_head["head_weight_runtime_KN"]["raw_tensor_sha256"]
            or actual["bias"]["raw_tensor_sha256"] != original_head["head_bias"]["raw_tensor_sha256"]
            or capture["capture_file"]["tensors"]["head_modulation"]["raw_tensor_sha256"]
            != original_head["head_modulation"]["raw_tensor_sha256"]):
        raise ValueError("Actual native head weight/bias/modulation differs from bound CPU view")


def bound(binding, roots):
    return verify_binding(roots[binding.get("root", "repo")], binding)


def validate_plan(plan, frozen=False):
    if (plan.get("id") != ID or plan.get("limits") != LIMITS or plan.get("resource_limits") != RESOURCES
            or plan.get("allowed_run_ids") != run_ids() or plan.get("steps") != list(STEPS)
            or plan.get("block") != 39 or plan.get("rows_per_case") != 256
            or plan.get("automatic_retry") is not False or plan.get("cohort_exposure") != SOURCE_EXPOSURE
            or plan.get("full_u_shape") != [M, D] or plan.get("head_shape") != [M, OUT]
            or plan.get("original_use_image_encoder") is not False):
        raise ValueError("Exact four-flow capture scope differs")
    if frozen and plan.get("assets_finalized") is not True:
        raise ValueError("Capture assets/code have not been frozen")
    from ffnds_experiment_plans import row_draw
    if [c["index"] for c in plan["cases"]] != list(INDICES):
        raise ValueError("Only source01/source04 are selected")
    for case in plan["cases"]:
        if case["seed"] != 42 or case["image"]["sha256"] != case["image_sha256"]:
            raise ValueError("Frozen source/noise identity differs")
        if "part0_recovery_20261027_v1/part0/" not in case["generation"]["path"]:
            raise ValueError("Fresh recovery Dense source required")
        for step in STEPS:
            rows, seed = row_draw(M, 256, base_seed=42, source_index=case["native_source_ordinal"],
                                  step=step, block=39, state_pair=True)
            item = case["row_draws"][str(step)]
            if item != {"rows": rows, "seed": seed}:
                raise ValueError("Actual source-ordinal row draw differs")
        if set(case["post"]) != {"0", "2", "3"}:
            raise ValueError("Historical post0/post2/post3 bindings required")
    if set(plan["head_assets"]) != {"high_noise_model", "low_noise_model"}:
        raise ValueError("Both original heads must be bound")
    sources = [item["path"] for item in plan["repo_sources"]]
    if len(sources) != len(set(sources)) or not REQUIRED_SOURCES <= set(sources):
        raise ValueError("Required exact native/capture source bindings incomplete")
    libraries = [p for p in sources if p.endswith(".so") and any(s in p for s in ("hifxg_quant", "lowbit_quant"))]
    if {Path(p).name.split(".")[0] for p in libraries} != {"hifxg_quant", "lowbit_quant"}:
        raise ValueError("Both original compiled quantizer libraries required")


def authorize(plan, digest, active, run_id=None):
    validate_plan(plan, frozen=True)
    authority = active.get("user_continuation_authorization_20261027", {})
    scopes = [s for s in active.get("experiment_authority", {}).get("current_user_authorized_scopes", [])
              if s.get("id") == ID]
    if (authority.get("id") != AUTHORIZATION
            or authority.get("status") != "ACTIVE_CURRENT_USER_DIRECTED_SESSION"
            or authority.get("max_simultaneously_assigned_idle_GPUs") != 4 or len(scopes) != 1):
        raise ValueError("Active continuation and unique capture registration required")
    scope = scopes[0]
    if (scope.get("protocol_sha256") != digest or scope.get("allowed_run_ids") != run_ids()
            or scope.get("execution_status") not in ("AUTHORIZED_NOT_STARTED", "RUNNING_FIXED_SCOPE")
            or scope.get("limits") != LIMITS or scope.get("max_simultaneously_assigned_idle_GPUs") != 1
            or run_id is not None and run_id not in run_ids()):
        raise ValueError("Exact source/step/code/budget registration differs")


def preflight(protocol, hf_root, root=ROOT):
    started = time.monotonic()
    torch_before = "torch" in sys.modules
    plan = json.loads(Path(protocol).read_text())
    validate_plan(plan)
    roots = {"repo": Path(root).resolve(), "hf": Path(hf_root).resolve()}
    for item in plan["repo_sources"]:
        bound(item, roots)
    head_view = json.loads(bound(plan["head_view_manifest"], roots).read_text())
    head_file = bound(plan["head_view_file"], roots)
    head_run = json.loads(bound(plan["head_view_RUN"], roots).read_text())
    head_exit = json.loads(bound(plan["head_view_exit"], roots).read_text())
    if (head_view.get("status") != "COMPLETE" or head_view.get("schema") != "ffnds.b_terminal_head_asset_view.v1"
            or head_run.get("status") != "COMPLETE" or head_exit.get("launcher_returncode") != 0
            or head_exit.get("process_exited") is not True or head_exit.get("RUN_sha256") != plan["head_view_RUN"]["sha256"]
            or head_run.get("manifest_sha256") != plan["head_view_manifest"]["sha256"]):
        raise ValueError("Original two-head CPU view producer differs")
    with head_file.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        raw = stream.read()
    import hashlib
    if len(head_view["tensors"]) != 6 or set(header) - {"__metadata__"} != {t["view_key"] for t in head_view["tensors"]}:
        raise ValueError("Exact six original head tensors required")
    for item in head_view["tensors"]:
        h = header[item["view_key"]]
        start, end = h["data_offsets"]
        if (h["dtype"] != item["dtype"] or h["shape"] != item["shape"] or end - start != item["bytes"]
                or hashlib.sha256(raw[start:end]).hexdigest() != item["tensor_sha256"]):
            raise ValueError("Original head view raw tensor binding differs")
        field = {"head.head.weight": "weight_raw_sha256", "head.head.bias": "bias_raw_sha256",
                 "head.modulation": "modulation_raw_sha256"}[item["key"]]
        if plan["head_assets"][item["expert"]][field] != item["tensor_sha256"]:
            raise ValueError("Frozen original head identity differs")
    producer = json.loads(bound(plan["historical_producer_RUN"], roots).read_text())
    launch = json.loads(bound(plan["historical_producer_exit"], roots).read_text())
    prior_protocol_path = bound(plan["historical_producer_protocol"], roots)
    if (producer["status"] != "PASS_FIXED_DISJOINT20_PART_FULL_DEPTH_CORE_METRICS_QUALITY_PENDING"
            or producer["protocol_sha256"] != sha256(prior_protocol_path)
            or launch.get("launcher_returncode") != 0 or launch.get("process_exited") is not True
            or launch["protocol_sha256"] != producer["protocol_sha256"]):
        raise ValueError("Completed fresh recovery Dense producer/exit differs")
    samples = json.loads(bound(plan["canonical_inputs"], roots).read_text())
    original = json.loads(bound(plan["original_cohort_protocol"], roots).read_text())
    logical = json.loads(bound(plan["native_logical_model_view"], roots).read_text())
    cache_manifest_path = Path(root) / ".runtime/hif4_rebuilt_cache.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    if ({k: cache_manifest[k] for k in ("sha256", "bytes")} != plan["native_cache_identity"]
            or cache_manifest["identity"]["model_view_sha256"] != plan["native_logical_model_view"]["sha256"]):
        raise ValueError("Worker pointer does not reference the exact original D-Q asset")
    for parent in head_view["parents"]:
        checkpoint = [r for r in logical["files"] if r["filename"] == parent["filename"]]
        if len(checkpoint) != 1 or checkpoint[0]["sha256"] != parent["existing_verified_file_sha256"]:
            raise ValueError("Original native model and head view parents differ")
    cases = []
    for case in plan["cases"]:
        sample, source = samples[case["index"]], original["cohort_plan"]["sources"][case["index"]]
        image = bound(case["image"], roots)
        generation = json.loads(bound(case["generation"], roots).read_text())
        if (sample["seed"] != case["seed"] or source["prompt"] != case["prompt"] or sample["prompt"] != case["prompt"]
                or sample["input_provenance"]["image_sha256"] != case["image_sha256"] or sample["negative_prompt"] != ""
                or case["source_id"] != source["source_id"] or case["conditioning_id"] != source["conditioning_id"]
                or source["native_source_ordinal"] != case["native_source_ordinal"]
                or sha256(image) != case["image_sha256"]
                or __import__("hashlib").sha256(case["prompt"].encode()).hexdigest() != case["prompt_sha256"]
                or generation["source_index"] != case["index"] or generation["seed"] != 42
                or generation["source_id"] != case["image_sha256"] or generation["mode"] != "dense"
                or generation["semantic_source_id"] != case["source_id"] or generation["conditioning_id"] != case["conditioning_id"]
                or generation["prompt_sha256"] != case["prompt_sha256"]
                or generation["protocol_sha256"] != producer["protocol_sha256"]
                or generation["status"] != "PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE"
                or generation["completed_native_flow_steps"] != 4
                or generation["native_counter_delta"] != {"linear_forward_calls": 1600,
                    "activation_qdq_calls": 1600, "weight_transfers": 0, "cache_misses": 0}):
            raise ValueError("Image/prompt/source/seed/fresh Dense evidence differs")
        initial_path = bound(case["initial"], roots)
        initial = read_state_bytes(initial_path, "BF16")
        if initial["raw_tensor_sha256"] != generation["initial_noise"]["raw_tensor_sha256"]:
            raise ValueError("Original Dense initial state differs")
        posts = {}
        for step in (0, 2, 3):
            path = bound(case["post"][str(step)], roots)
            item = read_state_bytes(path, "F32")
            if item["raw_tensor_sha256"] != generation["trajectory"][step]["raw_tensor_sha256"]:
                raise ValueError("Original post-state raw bytes differ")
            posts[str(step)] = item
        cases.append({"index": case["index"], "initial": initial, "post": posts,
                      "original_producer": "FRESH_PART0_RECOVERY_DENSE", "image_sha256": case["image_sha256"]})
    versions = {name: importlib.metadata.version(name) for name in plan["runtime_versions"]}
    if versions != plan["runtime_versions"]:
        raise ValueError("Native environment package versions differ")
    return plan, roots, {"status": "PASS_CPU_BYTE_BINDINGS_NATIVE_CAPTURE_NOT_EXECUTED",
        "protocol_sha256": sha256(protocol), "cases": cases, "versions": versions,
        "native_cache_manifest_pointer_sha256": sha256(cache_manifest_path),
        "Torch_or_CUDA_imported_by_preflight": not torch_before and "torch" in sys.modules,
        "native_calls": 0, "seconds": time.monotonic() - started}


class WorkLedger:
    """Debit fixed backbone calls before execution; preserve failed reservations."""
    def __init__(self, record, persist=lambda: None):
        self.record, self.persist = record, persist
        record.setdefault("predebited", {})
        record.setdefault("observed", {})

    def reserve(self, costs, run_id):
        if run_id not in run_ids() and run_id not in ("setup", "source01", "source04"):
            raise ValueError("Unregistered operation identity")
        used = self.record["predebited"]
        if any(k not in LIMITS or type(v) is not int or v < 0 or used.get(k, 0) + v > LIMITS[k]
               for k, v in costs.items()):
            raise ValueError("Capture operation budget exceeded")
        for key, value in costs.items():
            used[key] = used.get(key, 0) + value
        self.record["last_reserved_operation"] = run_id
        self.persist()

    def observe(self, costs):
        observed, used = self.record["observed"], self.record["predebited"]
        if any(observed.get(k, 0) + v > used.get(k, 0) for k, v in costs.items()):
            raise ValueError("Observed work was not reserved")
        for key, value in costs.items():
            observed[key] = observed.get(key, 0) + value
        self.persist()


SOURCE_COST = {k: 1 for k in ("input_encoder_requests", "T5_positive_prompt_encodes", "VAE_encodes",
                              "scheduler_prepare", "FP32_initial_noise_draws")}
FLOW_COST = {"complete_Dense_flows": 1, "native_blocks": 40, "native_FFN_calls": 40,
             "native_FFN_linears": 80, "native_FFN_activation_QDQ": 80,
             "native_non_FFN_linears": 320, "native_non_FFN_activation_QDQ": 320,
             "scheduler_step_pre": 1, "scheduler_step_post": 1, "pre_infer_calls": 1,
             "pre_Default_MM_calls": 5, "patch_Conv3D_calls": 1, "final_LayerNorm_calls": 1,
             "head_Default_MM_calls": 1, "post_unpatchify_calls": 1,
             "CPU_BF16_writeback_mul": 1, "CPU_BF16_writeback_inplace_add": 1,
             "CPU_BF16_writeback_mul_elements": 256 * D, "CPU_BF16_writeback_add_elements": 256 * D}


class NativeTwoStepSegment:
    """Original native step_pre -> wrapper.infer -> step_post, steps0/3 only."""
    def __init__(self, runner, tap, case, historical, paths, ledger, guard, save_pack, check,
                 save_case=lambda record: None, head_assets=()):
        self.runner, self.tap, self.case, self.historical, self.paths = runner, tap, case, historical, paths
        self.ledger, self.guard, self.save_pack, self.check = ledger, guard, save_pack, check
        self.completed = []
        self.save_case = save_case
        self.head_assets = head_assets

    def run_segment(self, segment_idx=0):
        import torch
        if segment_idx != 0 or type(self.runner.scheduler).__name__ != "Wan22StepDistillScheduler":
            raise ValueError("One ordinary distill segment required")
        scheduler = self.runner.scheduler
        if scheduler._uses_conditioned_latent_prefix() or scheduler.keep_latents_dtype_in_scheduler:
            raise ValueError("Unexpected stateful/prefix scheduler branch")
        if not torch.equal(scheduler.sigmas, torch.tensor([1., .9375, .8333333134651184, .625])):
            raise ValueError("Original four native sigmas differ")
        self.prepare_noise = self.save_pack(f"source{self.case['index']:02d}", "prepared_noise",
            {"raw_prepared_FP32_noise": cpu_copy(scheduler.latents)})
        self.condition_state = tree_state(self.runner.inputs)
        condition_tensors = {}
        def collect(value, prefix):
            if torch.is_tensor(value):
                condition_tensors[prefix] = cpu_copy(value)
            elif isinstance(value, dict):
                for key, item in value.items():
                    collect(item, prefix + "." + key if prefix else key)
            elif value is not None:
                raise ValueError("Unexpected actual condition payload")
        collect(self.runner.inputs, "")
        self.condition_pack = self.save_pack(f"source{self.case['index']:02d}", "conditions", condition_tensors)
        for step in STEPS:
            run_id = f"b_final_capture_source{self.case['index']:02d}_seed42_step{step}"
            self.guard(run_id)
            self.check()
            cost = dict(FLOW_COST)
            if step == 3:
                cost["historical_post2_BF16_imports"] = 1
            self.ledger.reserve(cost, run_id)
            if step == 3:
                scheduler.latents = import_post_state(self.paths["post2"],
                    self.historical["post"]["2"]["raw_tensor_sha256"], scheduler.latents.device)
            entry = scheduler_state(scheduler)
            scheduler.step_pre(step_index=step)
            scheduler.infer_condition = True
            pre = cpu_copy(scheduler.latents)
            expected = self.historical["initial"] if step == 0 else self.historical["post"]["2"]
            if tuple(pre.shape) != SHAPE or pre.dtype != torch.bfloat16 or raw_sha(pre) != expected["exact_native_BF16_sha256"]:
                raise ValueError("Actual original step_pre state does not reproduce historical Dense bits")
            after_pre = scheduler_state(scheduler)
            counters_before = dict(self.tap.runtime.counters)
            self.tap.begin(run_id, step)
            started = time.perf_counter()
            self.runner.model.infer(self.runner.inputs)
            if self.runner.model.cur_model_index != (0 if step == 0 else 1):
                raise ValueError("Original wrapper selected wrong expert")
            flow = cpu_copy(scheduler.noise_pred)
            if tuple(flow.shape) != SHAPE or flow.dtype != torch.float32:
                raise ValueError("Original native full flow shape/dtype differs")
            before_post = scheduler_state(scheduler)
            scheduler.step_post()
            post = cpu_copy(scheduler.latents)
            after_post = scheduler_state(scheduler)
            if raw_sha(post.float()) != self.historical["post"][str(step)]["raw_tensor_sha256"]:
                raise ValueError("Historical fresh Dense native post-state did not reproduce")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            captured = self.tap.finish(flow, pre, post)
            delta = {k: self.tap.runtime.counters[k] - v for k, v in counters_before.items()}
            if delta != {"linear_forward_calls": 400, "activation_qdq_calls": 400, "weight_transfers": 0, "cache_misses": 0}:
                raise ValueError("Actual native Dense linear/QDQ count differs")
            captured.update(wall_seconds_with_taps_and_hashes=elapsed, native_counter_delta=delta,
                schema="ffnds.b_terminal.capture_case.v1", source_index=self.case["index"], seed=42,
                source_id=self.case["source_id"], image_sha256=self.case["image_sha256"],
                prompt_sha256=self.case["prompt_sha256"], conditioning_id=self.case["conditioning_id"],
                native_source_ordinal=self.case["native_source_ordinal"], grid=list(GRID),
                row_draw=self.case["row_draws"][str(step)],
                original_head_file=self.head_assets[0 if step == 0 else 1] if self.head_assets else None,
                condition_file=self.condition_pack,
                scheduler_before_step_pre=entry, scheduler_after_step_pre=after_pre,
                scheduler_before_step_post=before_post, scheduler_after_step_post=after_post,
                condition_state_sha256=state_sha(self.condition_state), historical_post_raw_FP32_SHA_reproduced=True,
                actual_flow=tensor_record(flow), actual_native_pre=tensor_record(pre), actual_native_post=tensor_record(post))
            self.completed.append(captured)
            self.save_case(captured)
            self.ledger.observe(cost)
            self.check()
        return scheduler.latents


def verify_native_quantizer(backend, plan):
    """Resolve the original compiled API as registered setup, without a QDQ."""
    function, _, api_name = backend.get_hif4_api()
    module = sys.modules[function.__module__]
    if api_name != "HiFloat4.main" or not getattr(module, "CUDA_KERNELS", ()):
        raise ValueError("Original compiled HiFloat4.main is required")
    libraries = {}
    for name, imported in list(sys.modules.items()):
        filename = getattr(imported, "__file__", None)
        if filename and filename.endswith(".so") and any(k in name for k in ("hifxg_quant", "lowbit_quant")):
            path = Path(filename).resolve()
            bindings = [b for b in plan["repo_sources"] if (ROOT / b["path"]).resolve() == path]
            if len(bindings) != 1 or sha256(path) != bindings[0]["sha256"]:
                raise ValueError("Actually imported native CUDA extension differs")
            libraries[bindings[0]["path"]] = bindings[0]["sha256"]
    if len(libraries) != 2:
        raise ValueError("Both native quantizer libraries must be bound")
    return {"API": api_name, "libraries": libraries, "QDQ_calls_in_this_resolution": 0}


def execute(args, prepared):
    plan, roots, readiness = prepared
    digest = sha256(args.protocol)
    def guard(run_id=None):
        authorize(plan, digest, json.loads((ROOT / "research/ACTIVE_RESEARCH.json").read_text()), run_id)
    guard()
    from worker_identity import private_config_path, require_worker
    require_worker()
    if sha256(private_config_path()) not in plan["private_worker_config_sha256"]:
        raise ValueError("Worker profile was not frozen")
    out = confined(roots["hf"], plan["output_HF_relative"])
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out / "ATTEMPT.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"id": ID, "protocol_sha256": digest, "automatic_retry": False}, stream)
    record = {"id": ID, "schema": "ffnds.b_terminal.capture_run.v1", "protocol_sha256": digest,
              "status": "RUNNING_FIXED_CAPTURE_NOT_ACCEPTED",
              "automatic_retry": False, "CPU_preflight": readiness, "cases": [], "capture_only": True,
              "quality_or_speed_acceptance": False, "scope_limits": LIMITS}
    persist = lambda: write(out / "RUN.json", record)
    ledger, monitor, adapter, runtime = WorkLedger(record, persist), None, None, None
    persist()
    started = time.monotonic()
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError("Fixed B capture wall budget reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(RESOURCES["compute_wall_seconds"])
    def check(projected=0):
        if monitor:
            monitor.check()
        used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if used + projected > RESOURCES["new_HF_bytes"] or shutil.disk_usage(out).free - projected < RESOURCES["HF_free_min_bytes"]:
            raise RuntimeError("Fixed output/free-space budget reached")
        available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024
        if available < RESOURCES["host_available_min_bytes"]:
            raise RuntimeError("Host-memory reserve reached")
    try:
        check()
        snap = out / "EXECUTED_SOURCES"
        snap.mkdir(exist_ok=False)
        for item in plan["repo_sources"]:
            source = bound(item, roots)
            target = snap / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            check(source.stat().st_size)
            shutil.copyfile(source, target)
        shutil.copyfile(args.protocol, snap / "protocol.json")
        for name in ("HiF4_Sp", "LightX2V", "Wan2.2-I2V-A14B-W4A4"):
            sys.path.insert(0, str(ROOT / ".runtime/code" / name))
        for name in ("HiFloat4", "HiFloat4/hif4_gpu"):
            sys.path.insert(0, str(ROOT / ".runtime/code/HiF4_Sp" / name))
        import torch
        from safetensors.torch import save_file
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, reservation_receipt = reserve_before_assets(torch)
        record["CUDA_reservation"] = reservation_receipt
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0) or os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
            raise ValueError("One idle A100 with original arithmetic required")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_grad_enabled(False)
        from ffnds_all_weight_residency import AllWeightResidency
        from e0_whole_flow_counterfactual import LatentOnlyMain
        from historical_source_snapshot import HistoricalSources
        from restored_assets import verified_restored_assets
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        history = HistoricalSources()
        kwargs, cache, assets = verified_restored_assets(history)
        if cache["sha256"] != plan["native_cache_identity"]["sha256"] or cache["bytes"] != plan["native_cache_identity"]["bytes"]:
            raise ValueError("Original full native D-Q store differs")
        record.update(assets=assets, historical_source_verifications=history.receipts,
                      BF16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        for key, expected in REQUIRED_CONFIG.items():
            if key not in ("scheduler", "attention") and pipe.runner.config.get(key) != expected:
                raise ValueError("Actual frozen native branch differs: " + key)
        for key in ("lazy_load", "unload_modules", "use_compile", "compile", "use_stream_vae", "lora_configs", "lora_dynamic_apply"):
            if pipe.runner.config.get(key):
                raise ValueError("Unsupported native capture branch: " + key)
        for model in pipe.runner.model.model:
            infer = model.transformer_infer
            if (infer.has_post_adapter or infer.seq_p_group is not None or infer.infer_dtype != torch.bfloat16
                    or infer.sensitive_layer_dtype != torch.bfloat16 or len(model.transformer_weights.blocks) != 40):
                raise ValueError("Plain BF16 forty-block native model required")
        runtime = adapter.HIF4CacheRuntime(cache["path"])
        import importlib
        record["native_quantizer_setup"] = verify_native_quantizer(importlib.import_module("quant.hif4_backend"), plan)
        if any(runtime.counters.values()):
            raise ValueError("Native quantizer setup performed model work")
        adapter.activate_lightx2v_hooks(runtime=runtime)

        def save_pack(run_id, name, tensors):
            folder = out / run_id
            folder.mkdir(exist_ok=True)
            path = folder / (name + ".safetensors")
            if path.exists():
                raise FileExistsError("Capture pack already exists")
            field = {"u": "full_u_files", "capture": "capture_packs", "conditions": "condition_packs",
                     "original_head": "head_asset_packs", "prepared_noise": "prepared_FP32_noise_files"}[name]
            ledger.reserve({field: 1}, run_id if run_id in run_ids() else "setup" if run_id.endswith("_model") else run_id)
            check(sum(v.numel() * v.element_size() for v in tensors.values()) + (1 << 20))
            temporary = path.with_suffix(".tmp")
            save_file({k: cpu_copy(v) for k, v in tensors.items()}, str(temporary))
            temporary.replace(path)
            ledger.observe({field: 1})
            return {"path": str(path.relative_to(out)), "sha256": sha256(path), "bytes": path.stat().st_size,
                    "tensors": {k: tensor_record(v) for k, v in tensors.items()}}

        with torch.inference_mode(), ExitStack() as stack:
            policy = stack.enter_context(AllWeightResidency(runtime, "all_resident_no_flush", RESOURCES["resident_weight_max_bytes"]))
            expected_bytes = sum(r["weight_qdq"].numel() * r["weight_qdq"].element_size()
                                 for rows in runtime.records.values() for r in rows.values())
            estimated = expected_bytes + RESOURCES["reference_original_peak_physical_MiB"] * (1 << 20) + RESOURCES["memory_safety_bytes"]
            if expected_bytes != LIMITS["setup_decoded_weight_bytes"] or estimated > torch.cuda.get_device_properties(0).total_memory:
                raise RuntimeError("Native all-resident weights exceed original A100 headroom")
            ledger.reserve({"setup_decoded_weight_GPU_transfers": 800,
                            "setup_decoded_weight_bytes": expected_bytes}, "setup")
            for expert, rows in runtime.records.items():
                for module_name, item in rows.items():
                    block = int(module_name.split(".")[1])
                    runtime._weight(expert=expert, block_index=block, module_name=module_name, record=item,
                                    device=torch.device("cuda", 0), dtype=torch.bfloat16)
            if runtime.counters["weight_transfers"] != 800 or runtime.counters["linear_forward_calls"] != 0 or runtime.counters["activation_qdq_calls"] != 0:
                raise ValueError("Setup performed unexpected native work")
            ledger.observe({"setup_decoded_weight_GPU_transfers": 800, "setup_decoded_weight_bytes": expected_bytes})
            record["head_assets"] = []
            for index, expert in enumerate(("high_noise_model", "low_noise_model")):
                weights = pipe.runner.model.model[index].transformer_weights
                head = weights.head
                if head.has_lora_branch or getattr(head, "has_diff", False) or weights.norm.eps != 1e-6:
                    raise ValueError("Original unadapted native head required")
                # Original offload weights are intentionally absent until native infer.
                # Export the bound CPU checkpoint view; actual head identity is still
                # checked by FinalBlockTap during the original native head forward.
                from b_head_asset_export import export as export_cpu_head
                cpu_cost = {"CPU_original_head_view_file_loads": 1,
                            "CPU_original_head_tensor_validations": 3,
                            "CPU_head_weight_transpose_copies": 1,
                            "CPU_head_weight_transpose_elements": D * OUT}
                ledger.reserve(cpu_cost, "setup")
                tensors = export_cpu_head(bound(plan["head_view_file"], roots),
                    plan["head_view_file"]["sha256"], expert, plan["head_assets"][expert])
                ledger.observe(cpu_cost)
                if tuple(tensors["head_weight_runtime_KN"].shape) != (D, OUT) or tuple(tensors["head_bias"].shape) != (OUT,):
                    raise ValueError("Original native head layout differs")
                expected = plan["head_assets"][expert]
                if (raw_sha(tensors["head_weight_runtime_KN"].t().contiguous()) != expected["weight_raw_sha256"]
                        or raw_sha(tensors["head_bias"]) != expected["bias_raw_sha256"]
                        or raw_sha(tensors["head_modulation"]) != expected["modulation_raw_sha256"]):
                    raise ValueError("Actual loaded head/bias/modulation bytes differ")
                saved = save_pack(expert, "original_head", tensors)
                saved["setup_original_layout"] = {
                    "identity": "BOUND_CPU_CHECKPOINT_VIEW_NOT_A_PREFETCHED_NATIVE_WEIGHT",
                    "weight": tensor_record(tensors["head_weight_runtime_KN"]),
                    "bias": tensor_record(tensors["head_bias"]),
                    "modulation": tensor_record(tensors["head_modulation"]),
                    "actual_native_layout": "Recorded independently in each CASE.actual_head_operator"}
                record["head_assets"].append(saved)
            persist()
            for case, old in zip(plan["cases"], readiness["cases"]):
                scheduler = pipe.runner.scheduler
                scheduler.generator = None
                record_case = {"index": case["index"], "source_id": case["source_id"], "step_captures": []}
                record["cases"].append(record_case)
                rows = {s: case["row_draws"][str(s)]["rows"] for s in STEPS}
                ledger.reserve(SOURCE_COST, f"source{case['index']:02d}")
                source_actual = dict.fromkeys(SOURCE_COST, 0)
                with ExitStack() as source_stack:
                    def counted(owner, name, label):
                        original = getattr(owner, name)
                        def call(*a, **kw):
                            if source_actual[label] != 0:
                                raise RuntimeError("Duplicate native input preparation")
                            source_actual[label] += 1
                            return original(*a, **kw)
                        source_stack.enter_context(patch_method(owner, name, call))
                    counted(pipe.runner, "run_input_encoder", "input_encoder_requests")
                    counted(pipe.runner, "run_text_encoder", "T5_positive_prompt_encodes")
                    counted(pipe.runner, "run_vae_encoder", "VAE_encodes")
                    counted(scheduler, "prepare", "scheduler_prepare")
                    original_prepare_latents = scheduler.prepare_latents
                    def actual_noise_draw(seed, latent_shape, dtype=torch.float32):
                        if (seed != 42 or tuple(latent_shape) != SHAPE or dtype != torch.float32
                                or source_actual["FP32_initial_noise_draws"] != 0 or scheduler.generator is not None):
                            raise ValueError("One fresh original FP32 scheduler noise draw required")
                        source_actual["FP32_initial_noise_draws"] += 1
                        result = original_prepare_latents(seed, latent_shape, dtype=dtype)
                        if scheduler.latents.dtype != torch.float32 or tuple(scheduler.latents.shape) != SHAPE:
                            raise ValueError("Actual prepared native noise shape/dtype differs")
                        return result
                    source_stack.enter_context(patch_method(scheduler, "prepare_latents", actual_noise_draw))
                    def forbid_clip(*a, **kw):
                        raise RuntimeError("CLIP is disabled in this frozen native contract")
                    source_stack.enter_context(patch_method(pipe.runner, "run_image_encoder", forbid_clip))
                    def forbid_decode(*a, **kw):
                        raise RuntimeError("VAE decode is outside this capture scope")
                    source_stack.enter_context(patch_method(pipe.runner, "run_vae_decoder", forbid_decode))
                    with FinalBlockTap(runtime, adapter, pipe.runner.model.model, rows, save_pack, check) as tap:
                        segment = NativeTwoStepSegment(pipe.runner, tap, case, old,
                            {"post2": bound(case["post"]["2"], roots)}, ledger, guard, save_pack, check,
                            save_case=lambda item: write(out / item["run_id"] / "CASE.json", item),
                            head_assets=record["head_assets"])
                        record_case["step_captures"] = segment.completed
                        source_stack.enter_context(patch_method(pipe.runner, "run_segment", segment.run_segment))
                        source_stack.enter_context(LatentOnlyMain(pipe.runner))
                        pipe.generate(seed=42, image_path=str(bound(case["image"], roots)), prompt=case["prompt"],
                                      negative_prompt="", save_result_path=str(out / "NO_VIDEO_WILL_BE_WRITTEN.mp4"))
                        if source_actual != SOURCE_COST or [v["step"] for v in segment.completed] != list(STEPS):
                            raise ValueError("Actual input encoder/prepare/flow ledger differs")
                        record_case.update(step_captures=segment.completed, prepared_noise=segment.prepare_noise,
                            condition_pack=segment.condition_pack, condition_state=segment.condition_state,
                            condition_state_sha256=state_sha(segment.condition_state))
                        for capture in segment.completed:
                            original_head = record["head_assets"][0 if capture["step"] == 0 else 1]["tensors"]
                            verify_actual_head_asset(capture, original_head)
                ledger.observe(source_actual)
                record_case["actual_preparation_calls"] = source_actual
                persist()
                check()
            expected_native = {"linear_forward_calls": 1600, "activation_qdq_calls": 1600,
                               "weight_transfers": 800, "cache_misses": 0}
            if runtime.counters != expected_native:
                raise ValueError("Whole-scope native counters differ")
            if any(record["predebited"].get(k, 0) != v or record["observed"].get(k, 0) != v for k, v in LIMITS.items()):
                raise ValueError("Whole-scope predebited/observed work differs from exact scope")
            record.update(native_counters=runtime.counters, residency_policy=policy.snapshot(),
                          status="COMPLETE_B_FINAL_BLOCK_CAPTURE_HISTORICAL_POSTSTATE_BYTES_MATCH")
    except BaseException as error:
        record.update(status="FAILED_FIXED_CAPTURE_NO_AUTOMATIC_RETRY", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        original_error = sys.exc_info()[1]
        cleanup_errors = []
        if runtime is not None:
            record["actual_native_counters_including_failed_work"] = dict(runtime.counters)
        if adapter is not None:
            try:
                adapter.deactivate_lightx2v_hooks()
            except BaseException as error:
                cleanup_errors.append(error)
        if monitor is not None:
            try:
                monitor.close()
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                record["GPU_resources"] = monitor.report()
                if record["GPU_resources"].get("violations"):
                    cleanup_errors.append(RuntimeError("Assigned GPU monitor recorded violations"))
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            record["status"] = "FAILED_FIXED_CAPTURE_CLEANUP_OR_GPU_MONITOR_PRESERVED"
            record["cleanup_errors"] = [{"type": type(e).__name__, "message": str(e)} for e in cleanup_errors]
        try:
            record["wall_seconds_including_setup"] = time.monotonic() - started
            record["HF_output_bytes_before_final_RUN"] = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            persist()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_alarm)
        if cleanup_errors:
            if original_error is not None:
                original_error.add_note("Capture cleanup failures preserved in RUN.json")
            else:
                raise RuntimeError("Capture cleanup/GPU monitor failed; receipt preserved") from cleanup_errors[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, default=Path.home() / ".cache/huggingface")
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()
    prepared = preflight(args.protocol, args.hf_root)
    if args.execute_native:
        execute(args, prepared)
    else:
        print(json.dumps(prepared[2], indent=2))


if __name__ == "__main__":
    main()
