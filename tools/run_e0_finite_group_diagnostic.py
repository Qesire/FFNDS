"""Fixed E0 four-arm native diagnostic; no sparse kernel or speed claim.

One post-Q2 mask at step0/block0, original full dense GEMMs, unmodified gates,
native residual/LN, and four freely running steps. Dense repeat closes controls.
"""
import argparse
import contextvars
import json
from pathlib import Path
import shutil
import sys
import time

from ffnds_observation_common import row_plan, sha256, write_json
from prepare_ffnds_observations import preflight

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"dense": (), "A": (0,), "B": (53,), "AB": (0, 53), "dense_repeat": ()}


def tensor_sha(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def row_losses(value, reference):
    import torch
    if value.shape != reference.shape or value.ndim != 2:
        raise ValueError("Paired full matrices required")
    parts = []
    for begin in range(0, len(value), 1024):
        d = value[begin:begin+1024].double()-reference[begin:begin+1024].double()
        if not bool(torch.isfinite(d).all()):
            raise ValueError("Nonfinite native response")
        parts.append(d.square().sum(-1))
    return torch.cat(parts)


def interaction(dense, a, b, ab):
    """Exact four-arm SSE identity on actual saved output tensors."""
    x = [v.reshape(-1, v.shape[-1]) for v in [dense, a, b, ab]]
    acc = {k: 0. for k in ["A_SSE", "B_SSE", "AB_SSE", "linear_cross_term",
                            "nonlinear_term", "curvature_norm2", "additive_response_norm2"]}
    for begin in range(0, len(x[0]), 256):
        d, aa, bb, cc = [v[begin:begin+256].double() for v in x]
        aa, bb, abv = aa-d, bb-d, cc-d
        curve = abv-aa-bb
        acc["A_SSE"] += float(aa.square().sum())
        acc["B_SSE"] += float(bb.square().sum())
        acc["AB_SSE"] += float(abv.square().sum())
        acc["linear_cross_term"] += float(2*(aa*bb).sum())
        acc["nonlinear_term"] += float(2*((aa+bb)*curve).sum()+curve.square().sum())
        acc["curvature_norm2"] += float(curve.square().sum())
        acc["additive_response_norm2"] += float((aa+bb).square().sum())
    acc["interaction_SSE"] = acc["AB_SSE"]-acc["A_SSE"]-acc["B_SSE"]
    acc["identity_residual"] = acc["interaction_SSE"]-acc["linear_cross_term"]-acc["nonlinear_term"]
    if abs(acc["identity_residual"]) > 1e-10*max(1., acc["AB_SSE"], acc["A_SSE"], acc["B_SSE"]):
        raise ValueError("Native interaction identity failed")
    return acc


class FiniteGroupObserver:
    def __init__(self, runtime, adapter, arm, source_id):
        self.runtime, self.adapter, self.arm, self.source_id = runtime, adapter, arm, source_id
        self.selected = contextvars.ContextVar("e0_selected_second_linear", default=False)
        self.state, self.calls = {}, {"selected_q2": 0, "selected_post": 0, "selected_nextnorm": 0}

    def __enter__(self):
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        self.cls = WanTransformerInfer
        self.old_apply, self.old_qdq = self.runtime.apply, self.adapter.native_hif4_activation_qdq
        self.old_post, self.old_block = self.cls.post_process, self.cls.infer_block
        owner = self

        def apply(mm, value, *, expert, step_index, block_index, module_name):
            selected = (expert, step_index, block_index, module_name) == (
                "high_noise_model", 0, 0, "blocks.0.ffn.2")
            token = owner.selected.set(selected)
            try:
                return owner.old_apply(mm, value, expert=expert, step_index=step_index,
                                       block_index=block_index, module_name=module_name)
            finally:
                owner.selected.reset(token)

        def qdq(value, *, force_fp32=True):
            result = owner.old_qdq(value, force_fp32=force_fp32)
            if owner.selected.get():
                if tuple(value.shape) != (31668, 13824) or not force_fp32:
                    raise ValueError("Changed selected full-M Q2 semantics")
                owner.calls["selected_q2"] += 1
                owner.state["unmodified_q2_sha256"] = tensor_sha(result)
                owner.state["pre_q2_sha256"] = tensor_sha(value)
                if ARMS[owner.arm]:
                    result = result.clone()
                    for group in ARMS[owner.arm]:
                        result[:, group*64:(group+1)*64] = 0
                owner.state["returned_q2_sha256"] = tensor_sha(result)
            return result

        def post(infer, x, y, gate, pre_infer_out=None):
            selected = int(infer.scheduler.step_index)==0 and int(infer.block_idx)==0
            if selected:
                if tuple(x.shape) != (31668, 5120):
                    raise ValueError("Changed selected residual geometry")
                owner.calls["selected_post"] += 1
                owner.state["pre_FFN_writeback_sha256"] = tensor_sha(x)
                owner.state["gate_sha256"] = tensor_sha(gate)
                owner.state["ffn_output"] = y.detach().cpu().contiguous().clone()
                owner.state["gate"] = gate.detach().reshape(1, 5120).cpu().contiguous().clone()
            result = owner.old_post(infer, x, y, gate, pre_infer_out)
            if selected:
                owner.state["post"] = result.detach().cpu().contiguous().clone()
            return result

        def block(infer, weights, x, pre):
            if int(infer.scheduler.step_index)!=0 or int(infer.block_idx)!=1:
                return owner.old_block(infer, weights, x, pre)
            norm = weights.compute_phases[0].norm1
            old_norm = norm.apply
            def captured(value):
                result = old_norm(value)
                owner.calls["selected_nextnorm"] += 1
                owner.state["nextnorm"] = result.detach().cpu().contiguous().clone()
                return result
            norm.apply = captured
            try:
                return owner.old_block(infer, weights, x, pre)
            finally:
                norm.apply = old_norm

        self.runtime.apply, self.adapter.native_hif4_activation_qdq = apply, qdq
        self.cls.post_process, self.cls.infer_block = post, block
        return self

    def __exit__(self, *error):
        self.runtime.apply, self.adapter.native_hif4_activation_qdq = self.old_apply, self.old_qdq
        self.cls.post_process, self.cls.infer_block = self.old_post, self.old_block
        if not any(error) and set(self.calls.values()) != {1}:
            raise RuntimeError("Incomplete or duplicate native observation boundaries")


def execute(args, protocol):
    if args.index not in protocol["source_indices"] or protocol["arms"] != list(ARMS):
        raise ValueError("Out of frozen scope")
    if protocol["intervention"]["groups_A"] != list(ARMS["A"]) or protocol["intervention"]["groups_B"] != list(ARMS["B"]):
        raise ValueError("Changed frozen group definitions")
    if sha256(ROOT/protocol["accepted_observations"]) != protocol["accepted_observations_sha256"]:
        raise ValueError("Accepted observation manifest changed")
    for name, expected in protocol["driver_sources_sha256"].items():
        if sha256(ROOT/name) != expected:
            raise ValueError("Frozen driver source changed")
    static = preflight(ROOT, protocol)
    if static["status"] != "PREPARED_STATIC_PREFLIGHT":
        raise RuntimeError("Native source, environment or model manifest preflight failed")
    for name in ["HiF4_Sp", "LightX2V", "Wan2.2-I2V-A14B-W4A4"]:
        sys.path.insert(0, str(ROOT/".runtime/code"/name))
    sys.path.insert(0, str(ROOT/".runtime/code/HiF4_Sp/HiFloat4"))
    sys.path.insert(0, str(ROOT/".runtime/code/HiF4_Sp/HiFloat4/hif4_gpu"))
    import torch
    import numpy as np
    from safetensors.torch import save_file
    from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
    from historical_source_snapshot import HistoricalSources
    from restored_assets import verified_restored_assets
    from startup_gpu_reservation import reserve_before_assets
    from worker_identity import require_worker
    require_worker()
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Frozen diagnostic requires A100")
    accepted = json.loads((ROOT/protocol["accepted_observations"]).read_text())
    original = next(s for s in accepted["sources"] if s["index"]==args.index)
    inputs = ROOT/protocol["input_default"]
    sample = json.loads(inputs.read_text())[args.index]
    image = Path(sample.get("image_path") or sample["img_paths"][0])
    seed = sample.get("seed", sample.get("calibration_seed"))
    if sha256(image) != original["source_id"] or seed != original["seed"]:
        raise ValueError("Frozen image/seed changed")
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out/"EXECUTED_SOURCES"
    snapshot.mkdir()
    for name in protocol["driver_sources_sha256"]:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    shutil.copyfile(args.protocol, snapshot/args.protocol.name)
    record = {"status": "STARTED_NOT_ACCEPTED", "protocol_sha256": sha256(args.protocol),
              "source_index": args.index, "source_id": original["source_id"], "seed": seed,
              "inputs_sha256": sha256(inputs), "phases": {}, "generations_started": 0,
              "quality_or_speed_acceptance": False, "automatic_retry": False}
    write_json(args.out/"RUN.json", record)
    monitor = None
    torch.set_num_threads(8)
    states, trajectories = {}, {}
    try:
        reservation, monitor, receipt = reserve_before_assets(torch)
        record["early_CUDA_reservation"] = receipt
        write_json(args.out/"RUN.json", record)
        kwargs, cache, assets = verified_restored_assets(HistoricalSources())
        record["assets"] = assets
        monitor.check()
        pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        adapter.validate_effective_pipeline_config_against_contract(pipe.runner.config)
        runtime = adapter.HIF4CacheRuntime(cache["path"])
        adapter.activate_lightx2v_hooks(runtime=runtime)
        scheduler = pipe.runner.scheduler
        for arm in ARMS:
            monitor.check()
            if shutil.disk_usage(args.out).free < protocol["disk_reserve_gib"]*(1<<30):
                raise RuntimeError("Disk reserve reached")
            if sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) > protocol["max_output_gib_per_source"]*(1<<30):
                raise RuntimeError("Output limit reached")
            before, started = dict(runtime.counters), time.perf_counter()
            old_pre, old_post = scheduler.step_pre, scheduler.step_post
            trajectory, current = [], {}
            def pre(step_index):
                value = old_pre(step_index)
                current["input"] = scheduler.latents.detach().float().cpu().contiguous().clone()
                return value
            def post():
                step = int(scheduler.step_index)
                flow = scheduler.noise_pred.detach().float().cpu().contiguous().clone()
                value = old_post()
                latent = scheduler.latents.detach().float().cpu().contiguous().clone()
                tensors = {"input": current["input"], "flow": flow, "post": latent}
                if not all(bool(torch.isfinite(v).all()) for v in tensors.values()):
                    raise RuntimeError("Nonfinite native trajectory")
                path = args.out/f"{arm}_step{step}.safetensors"
                save_file(tensors, str(path))
                trajectory.append({"step": step, "path": path.name, "sha256": sha256(path),
                                   **{k+"_sha256": tensor_sha(v) for k, v in tensors.items()}})
                return value
            scheduler.step_pre, scheduler.step_post, scheduler.generator = pre, post, None
            observer = FiniteGroupObserver(runtime, adapter, arm, original["source_id"])
            record["generations_started"] += 1
            if record["generations_started"] > protocol["max_generations_per_source"]:
                raise RuntimeError("Generation budget exceeded")
            record["current_phase"] = arm
            write_json(args.out/"RUN.json", record)
            try:
                with torch.inference_mode(), observer:
                    pipe.generate(seed=seed, image_path=str(image), prompt=sample["prompt"],
                                  negative_prompt="", save_result_path=str(args.out/f"{arm}.mp4"))
                torch.cuda.synchronize()
                monitor.check()
            finally:
                scheduler.step_pre, scheduler.step_post = old_pre, old_post
            counts = {k: runtime.counters[k]-before[k] for k in before}
            if counts != original["native_counts_each_phase"] or [s["step"] for s in trajectory] != [0,1,2,3]:
                raise RuntimeError("Original native counts or four-step coverage changed")
            phase = {"trajectory": trajectory, "video_sha256": sha256(args.out/f"{arm}.mp4"),
                     "native_counts": counts, "observer_calls": observer.calls,
                     "elapsed_including_diagnostics": time.perf_counter()-started,
                     "state_sha256": {k: tensor_sha(observer.state[k]) for k in ["post", "nextnorm", "ffn_output"]},
                     "prefix": {k: v for k,v in observer.state.items() if isinstance(v,str)}}
            if arm in ("dense", "dense_repeat"):
                if phase["video_sha256"] != original["video_sha256"] or any(
                    s["post_sha256"] != old["sha256"] for s,old in zip(trajectory, original["trajectory"])):
                    raise RuntimeError("Dense native history identity failed")
            if arm != "dense" and any(phase["prefix"][k] != record["phases"]["dense"]["prefix"][k]
                                      for k in ["pre_q2_sha256", "unmodified_q2_sha256", "pre_FFN_writeback_sha256", "gate_sha256"]):
                raise RuntimeError("Intervention did not share the exact dense prefix")
            plan = row_plan([21,29,52], original["source_id"], 32)
            ids = torch.tensor(plan["uniform_rows"])
            save_file({k: observer.state[k].index_select(0,ids) for k in ["post", "nextnorm", "ffn_output"]},
                      str(args.out/f"{arm}_samples.safetensors"))
            states[arm] = observer.state
            trajectories[arm] = trajectory
            if arm != "dense":
                losses = {k: row_losses(observer.state[k], states["dense"][k]).numpy() for k in ["post", "nextnorm", "ffn_output"]}
                np.savez_compressed(args.out/f"{arm}_row_losses.npz", **losses)
                phase["native_full_rows_SSE"] = {k: float(v.sum()) for k,v in losses.items()}
            record["phases"][arm] = phase
            write_json(args.out/"RUN.json", record)
            print(f"E0 finite source{args.index:02d} {arm} complete", flush=True)
        record["native_interaction"] = {k: interaction(*(states[a][k] for a in ["dense","A","B","AB"]))
                                         for k in ["post", "nextnorm", "ffn_output"]}
        if any(record["phases"]["dense"]["state_sha256"][k] != record["phases"]["dense_repeat"]["state_sha256"][k]
               for k in ["post", "nextnorm", "ffn_output"]):
            raise RuntimeError("Repeated dense local control failed")
        record.update(status="PASS_FIXED_E0_FINITE_NATIVE_DIAGNOSTIC", extra_native_FFN_calls=0,
                      full_free_running_steps=True, output_bytes=sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()))
    except BaseException as error:
        record.update(status="FAILED_PRESERVED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        adapter.deactivate_lightx2v_hooks()
        if monitor is not None:
            monitor.close()
            record["assigned_GPU_resources"] = monitor.report()
        write_json(args.out/"RUN.json", record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute-native", action="store_true")
    args = p.parse_args()
    if not args.execute_native:
        raise ValueError("Explicit native execution flag required")
    execute(args, json.loads(args.protocol.read_text()))


if __name__ == "__main__":
    main()
