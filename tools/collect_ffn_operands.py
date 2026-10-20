"""Capture all-position calibration FFN operands with per-input native no-op."""
import argparse
import contextvars
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from safetensors.torch import save_file

from ar2_checkpoint import atomic_json, canonical_json, digest
from distill_assets import ROOT, verified_assets


def tensor_sha(value):
    return hashlib.sha256(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


class OperandCapture:
    def __init__(self, runtime, out, index, *, rows=512, sampling_namespace="FFNDS_CALIB12_UNIFORM_V1"):
        self.runtime, self.out, self.index = runtime, out, index
        self.rows,self.sampling_namespace=rows,sampling_namespace
        self.context = contextvars.ContextVar("ffnds_calibration_ffn", default=None)
        self.samples, self.scopes = {}, []

    def __enter__(self):
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        self.old_apply, self.old_qdq = self.runtime.apply, adapter.native_hif4_activation_qdq
        self.old_block, self.old_ffn = WanTransformerInfer.infer_block, WanTransformerInfer.infer_ffn
        observer = self

        def block(infer, block_weight, x, pre):
            token = observer.context.set({"grid": [int(v) for v in pre.grid_sizes.tuple]})
            try:
                return observer.old_block(infer, block_weight, x, pre)
            finally:
                observer.context.reset(token)

        def ffn(infer, phase, x, attn, shift, scale, gate=None):
            if gate is None or gate.numel() != 5120:
                raise RuntimeError("Need actual row-constant gate")
            state = {**observer.context.get(), "gate": gate.reshape(1, 5120).detach().cpu().contiguous()}
            token = observer.context.set(state)
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            try:
                result = observer.old_ffn(infer, phase, x, attn, shift, scale, gate)
                if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                    raise RuntimeError("Operand observation altered RNG")
                return result
            finally:
                observer.context.reset(token)

        def qdq(activation, *, force_fp32=True):
            result = observer.old_qdq(activation, force_fp32=force_fp32)
            state = observer.context.get()
            slot = state.get("slot") if state else None
            if slot in [0, 2]:
                name = "V1" if slot == 0 else "V2"
                observer.samples[name] = result.reshape(-1, result.shape[-1]).index_select(0, state["indices"]).detach().cpu().contiguous()
            return result

        def apply(mm, value, *, expert, step_index, block_index, module_name):
            if not module_name.endswith((".ffn.0", ".ffn.2")):
                return observer.old_apply(mm, value, expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
            state = observer.context.get()
            slot = 0 if module_name.endswith(".ffn.0") else 2
            if state is None:
                raise RuntimeError("FFN operand outside native block context")
            count = value.numel() // value.shape[-1]
            if count != int(np.prod(state["grid"])):
                raise RuntimeError("Observed rows/grid differ")
            if slot == 0:
                if observer.samples:
                    raise RuntimeError("Previous FFN capture unfinished")
                salt = f"{observer.sampling_namespace}|{observer.index}|{step_index}|{block_index}|{state['grid']}"
                seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "little")
                selected = np.sort(np.random.default_rng(seed).choice(count, observer.rows, replace=False)).tolist()
                state.update(selected=selected, indices=torch.tensor(selected, device=value.device), seed=seed)
            state["slot"] = slot
            output = observer.old_apply(mm, value, expert=expert, step_index=step_index, block_index=block_index, module_name=module_name)
            state.pop("slot")
            if slot == 2:
                observer.samples["Y"] = output.reshape(-1, 5120).index_select(0, state["indices"]).detach().cpu().contiguous()
                bias = mm._get_actual_bias()
                observer.samples["b2"] = bias.detach().cpu().contiguous() if bias is not None else torch.empty(0, dtype=torch.bfloat16)
                observer.samples["gate"] = state["gate"]
                if set(observer.samples) != {"V1", "V2", "Y", "b2", "gate"} or any(t.dtype != torch.bfloat16 or not bool(torch.isfinite(t).all()) for t in observer.samples.values()):
                    raise RuntimeError("Incomplete/nonfinite native BF16 operands")
                path = observer.out / f"step{step_index}_block{block_index:02d}.safetensors"
                if path.exists():
                    raise RuntimeError("Refusing to overwrite a completed scope")
                temp = Path(str(path) + ".tmp")
                save_file(observer.samples, str(temp))
                temp.replace(path)
                observer.scopes.append({"step": step_index, "block": block_index, "expert": expert,
                    "rows": observer.rows, "grid": state["grid"], "selected_rows": state["selected"],
                    "sampling_seed": state["seed"], "inclusion_probability": observer.rows / count, "inverse_inclusion_weight": count / observer.rows,
                    "path": str(path), "sha256": digest(path), "bytes": path.stat().st_size})
                observer.samples.clear()
            return output

        self.runtime.apply, adapter.native_hif4_activation_qdq = apply, qdq
        WanTransformerInfer.infer_block, WanTransformerInfer.infer_ffn = block, ffn
        return self

    def __exit__(self, *error):
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        self.runtime.apply, adapter.native_hif4_activation_qdq = self.old_apply, self.old_qdq
        WanTransformerInfer.infer_block, WanTransformerInfer.infer_ffn = self.old_block, self.old_ffn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, choices=[0, 1], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs",type=Path,default=ROOT/".runtime/inputs/calibration12/canonical.json")
    ap.add_argument("--protocol",type=Path,default=ROOT/"config/calibration_ffn_operands.json")
    ap.add_argument("--raw-root", type=Path, default=Path("/var/tmp/ffnds-xiang/huggingface/ffnds-derived/calibration12_ffn_operands_strictseed_20261021"))
    args = ap.parse_args()
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "JOB.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = args.protocol
    protocol=json.loads(protocol_path.read_text())
    rows=protocol.get('rows_per_scope',protocol.get('rows',512))
    count=protocol.get('input_count',12)
    namespace=protocol.get('sampling_namespace','FFNDS_CALIB12_UNIFORM_V1')
    if (count,rows) not in [(12,512),(24,256)]:
        raise RuntimeError('Only the two registered calibration geometries are supported')
    calibration_path = args.inputs
    samples = json.loads(calibration_path.read_text())
    if len(samples) != count or any(s["calibration_seed"] != 42 + i or digest(s["img_paths"][0]) != s["input_provenance"]["image_sha256"] for i, s in enumerate(samples)):
        raise RuntimeError("Original calibration inputs changed")
    if count==24:
        if not args.raw_root.expanduser().resolve().is_relative_to((Path.home()/'.cache/huggingface/ffnds-derived').resolve()):
            raise RuntimeError('Calibration24 operands must use the shared home HF derived cache, preserving node root space')
        recovery_path=calibration_path.parent/'recovery_report.json'
        recovery=json.loads(recovery_path.read_text())
        if (recovery['status']!='PASS_SOURCE_DISJOINT_CALIBRATION24_INPUT_RECOVERY'
                or recovery['canonical_sha256']!=digest(calibration_path) or recovery['protocol_sha256']!=digest(protocol_path)):
            raise RuntimeError('Disjoint24 input recovery identity changed')
    kwargs, cache, assets = verified_assets()
    identity = {"protocol_sha256": digest(protocol_path), "calibration_sha256": digest(calibration_path),
                "source_sha256": {str(p): digest(p) for p in [Path(__file__), ROOT / "tools/distill_assets.py"]},
                "assets": assets, "shard": args.shard, "torch": torch.__version__, "host": "worker",
                'input_count':count,'rows_per_scope':rows,
                'versions':{n:importlib.metadata.version(n) for n in ['torch','numpy','safetensors','transformers','diffusers','triton']},
                'native_sources':{str(p):digest(p) for p in [Path(adapter.__file__),
                    ROOT/'.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py',
                    ROOT/'.runtime/code/LightX2V/lightx2v/common/ops/mm/mm_weight.py',
                    ROOT/'.runtime/code/LightX2V/lightx2v/models/schedulers/wan/scheduler.py']}}
    fingerprint = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    report = args.out / "RUN.json"
    completed = []
    if report.exists():
        previous = json.loads(report.read_text())
        if previous["identity"] != identity:
            raise RuntimeError("Collection identity changed; select fresh output")
        completed = previous["completed"]
    for row in completed:
        if digest(row["input_report"]) != row["input_report_sha256"]:
            raise RuntimeError("Completed input report changed")
        old = json.loads(Path(row["input_report"]).read_text())
        if old["status"] != "PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP" or any(digest(s["path"]) != s["sha256"] for s in old["scopes"]):
            raise RuntimeError("Completed operands changed")
    pending = [i for i in range(count) if i % 2 == args.shard and i not in {r["index"] for r in completed}]
    meta = {"status": "RUNNING", "identity": identity, "fingerprint": fingerprint, "completed": completed}
    atomic_json(report, meta)
    try:
        args.raw_root.mkdir(parents=True, exist_ok=True)
        if pending and shutil.disk_usage(args.raw_root).free < (80 + 4.5 * rows / 512 * len(pending)) * 2**30:
            raise RuntimeError("Insufficient disk for this six-input shard plus80GiB reserve")
        if pending:
            pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
            adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
            runtime = adapter.HIF4CacheRuntime(cache["path"])
            adapter.activate_lightx2v_hooks(runtime=runtime)
            try:
                for index in pending:
                    if shutil.disk_usage(args.raw_root).free < 85 * 2**30:
                        raise RuntimeError("80GiB reserve threatened")
                    sample = samples[index]
                    parent = args.raw_root / fingerprint / f"input{index:02d}"
                    parent.mkdir(parents=True, exist_ok=True)
                    attempt = 1
                    while (parent / f"attempt{attempt}").exists():
                        attempt += 1
                    out = parent / f"attempt{attempt}"
                    out.mkdir()
                    observed = OperandCapture(runtime, out, index,rows=rows,sampling_namespace=namespace)
                    trajectories, videos = {}, {}
                    for phase in ["plain", "observed"]:
                        trajectory = []
                        original_post = pipe.runner.scheduler.step_post
                        def post():
                            result = original_post()
                            scheduler = pipe.runner.scheduler
                            latent = scheduler.latents.detach().float().cpu().contiguous()
                            if not bool(torch.isfinite(latent).all()):
                                raise RuntimeError("Nonfinite calibration trajectory")
                            trajectory.append({"step": int(scheduler.step_index), "sha256": tensor_sha(latent), "shape": list(latent.shape)})
                            return result
                        pipe.runner.scheduler.step_post = post
                        video = out / f"{phase}.mp4"
                        started = time.perf_counter()
                        try:
                            from contextlib import nullcontext
                            # Native prepare_latents otherwise continues the previous
                            # Generator stream, ignoring this request's seed argument.
                            pipe.runner.scheduler.generator = None
                            with torch.inference_mode(), observed if phase == "observed" else nullcontext():
                                pipe.generate(seed=42 + index, image_path=sample["img_paths"][0], prompt=sample["prompt"], negative_prompt="", save_result_path=str(video))
                            torch.cuda.synchronize()
                        finally:
                            pipe.runner.scheduler.step_post = original_post
                        if [r["step"] for r in trajectory] != [0, 1, 2, 3]:
                            raise RuntimeError("Incomplete four-step generation")
                        trajectories[phase] = trajectory
                        videos[phase] = {"path": str(video), "sha256": digest(video), "elapsed_including_observation": time.perf_counter() - started}
                        atomic_json(out / f"PHASE_{phase}.json", {"phase": phase, "seed": 42 + index,
                            "scheduler_generator_reset": True, "trajectory": trajectory, "video": videos[phase]})
                        print(f"DONE input{index:02d} {phase}", flush=True)
                    if trajectories["plain"] != trajectories["observed"] or videos["plain"]["sha256"] != videos["observed"]["sha256"]:
                        atomic_json(out / "FAILED_NOOP.json", {"status": "FAILED_NOOP", "trajectories": trajectories,
                            "videos": videos, "scopes": observed.scopes, "raw_operands_accepted": False})
                        raise RuntimeError("Observer no-op failed; raw operands remain unaccepted")
                    if {(s["step"], s["block"]) for s in observed.scopes} != {(s,b) for s in range(4) for b in range(40)} or len(observed.scopes) != 160:
                        raise RuntimeError("Missing/duplicate native FFN scopes")
                    result_path = out / "INPUT.json"
                    atomic_json(result_path, {"status": "PASS_NATIVE_OPERANDS_AND_SAME_PIPELINE_NOOP", "fingerprint": fingerprint,
                        "index": index, "input": sample, "trajectories": trajectories, "videos": videos, "scopes": observed.scopes,
                        "no_op_limit": "same initialized pipeline with fresh seeded invocations; not fresh processes"})
                    completed.append({"index": index, "input_report": str(result_path), "input_report_sha256": digest(result_path)})
                    atomic_json(report, meta)
                    print(f"COMMITTED input{index:02d}, 160 scopes, four latent hashes and video no-op PASS", flush=True)
            finally:
                adapter.deactivate_lightx2v_hooks()
        if len(completed) != count//2 or {r["index"] for r in completed} != {i for i in range(count) if i % 2 == args.shard}:
            raise RuntimeError("Incomplete/duplicate calibration input shard")
        meta["status"] = "PASS_SIX_INPUT_FULL160_OPERANDS_AND_NOOP" if count==12 else "PASS_CALIBRATION24_SHARD_FULL160_OPERANDS_AND_NOOP"
    except BaseException as error:
        meta.update(status="FAILED_RESUMABLE", error=str(error))
        raise
    finally:
        atomic_json(report, meta)


if __name__ == "__main__":
    main()
