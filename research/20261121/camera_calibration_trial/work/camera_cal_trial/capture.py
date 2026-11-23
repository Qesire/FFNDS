"""Capture shared high-noise native FFN panels on the frozen paired prompts."""
import ast
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import sys
import textwrap
from time import monotonic

import numpy as np
import torch

from generate import digest, load, save
from icc_native_job import build_pipeline


def make_panel(probes):
    grid = (21, 29, 52)
    rng = np.random.default_rng(80512867)
    occupied = set(probes)
    spatial, temporal = [], []
    for origin in rng.permutation(np.prod(grid)):
        t, y, x = np.unravel_index(origin, grid)
        if t == 20 or x == 51 or y == 28:
            continue
        a = int(origin)
        b = a + (1 if len(spatial) % 4 == 0 else 52)
        c, d = a + 1508, b + 1508
        if {a, b, c, d} & occupied:
            continue
        occupied.update((a, b, c, d))
        spatial.extend(((a, b), (c, d)))
        temporal.extend(((a, c), (b, d)))
        if len(occupied) == 512:
            break
    assert len(occupied) == 512 and len(spatial) == len(temporal) == 224
    ids = np.asarray(sorted(occupied))
    return dict(token_indices=torch.from_numpy(ids), token_thw=torch.from_numpy(np.asarray(np.unravel_index(ids, grid)).T.copy()),
                spatial_edges=torch.from_numpy(np.searchsorted(ids, spatial)), temporal_edges=torch.from_numpy(np.searchsorted(ids, temporal)),
                probe_rows=torch.from_numpy(np.searchsorted(ids, probes)))


@contextmanager
def observe(owner, expected_sha, layers, tap, tap_w1):
    native = owner.infer_ffn
    file = inspect.getfile(native)
    assert digest(file) == expected_sha
    tree = ast.parse(textwrap.dedent(inspect.getsource(native)))
    function = tree.body[0]
    assert not function.decorator_list and ast.unparse(function.body[-1]) == "return y"
    assert ast.unparse(function.body[-2]) == "y = phase.ffn_2.apply(y)"
    w1 = [i for i, node in enumerate(function.body) if ast.unparse(node) == "y = phase.ffn_0.apply(norm2_out)"]
    assert len(w1) == 1
    function.body[w1[0]+1:w1[0]+1] = ast.parse("capture_w1(self, y)").body
    function.body[-1:-1] = ast.parse("capture_endpoint(self, phase, norm2_out, y, c_gate_msa)").body
    namespace = dict(native.__globals__, capture_endpoint=tap, capture_w1=tap_w1)
    exec(compile(ast.fix_missing_locations(tree), file, "exec"), namespace)
    instrumented = namespace[function.name]

    def wrapper(model, *args, **kwargs):
        if int(model.scheduler.step_index) in (0, 1) and int(model.block_idx) in layers:
            assert not model.clean_cuda_cache and not model._use_mxfp8_quant_fuse()
            return instrumented(model, *args, **kwargs)
        return native(model, *args, **kwargs)

    owner.infer_ffn = wrapper
    try:
        yield
    finally:
        owner.infer_ffn = native


def run():
    base = load("base.json")
    protocol = load("protocol.json")
    rows = protocol["trajectories"]
    layers = protocol["layers"]
    assert len(rows) == len({r["trajectory_id"] for r in rows}) == 13
    assert layers == [0, 13, 26, 39] and os.environ["SLURM_JOB_PARTITION"] == "P107-A100"
    for row in rows:
        assert digest(row["image_path"]) == row["image_sha256"]
        assert hashlib.sha256(row["prompt"].encode()).hexdigest() == row["prompt_sha256"]
    ref = Path(base["v2_reference_dir"])
    assert all(digest(ref/name) == sha for name, sha in base["reference_sha256"].items())
    sys.path.insert(0, str(ref))
    import runner_v2 as v2
    out = Path("output")
    out.mkdir(exist_ok=True)
    (out/"dense").mkdir(exist_ok=True)
    runtime = Path("runtime_"+os.environ["SLURM_JOB_ID"])
    runtime.mkdir()
    pipe, config, _ = build_pipeline(runtime, base, ref)
    assert digest(config/"EFFECTIVE_CONFIG.json") == base["strict_effective_config"]["sha256"]
    shutil.copyfile(config/"EFFECTIVE_CONFIG.json", out/"EFFECTIVE_CONFIG.json")
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
    sched = pipe.runner.scheduler
    panel = make_panel(base["observer_ids"])
    ids = panel["token_indices"].cuda()
    probes = torch.tensor(base["observer_ids"], device="cuda")
    assert digest(base["cal_manifest"]["path"]) == base["cal_manifest"]["sha256"]
    sites = load(base["cal_manifest"]["path"])["sites"]
    selected = {}
    for layer in layers:
        pack = sites[f"high{layer:02d}"]
        assert digest(pack["path"]) == pack["sha256"]
        selected[layer] = torch.load(pack["path"], weights_only=True, map_location="cpu")["sel"].cuda()
    tensor_hash = lambda t: v2.finite_tensor_sha(t, torch)["sha256"]
    state = dict(status="running", protocol_sha256=digest("protocol.json"), manifest_sha256=digest("MANIFEST.sha256"),
                 job=os.environ["SLURM_JOB_ID"], sites=[], trajectories=[])
    save(out/"CAPTURE_INDEX.json", state)
    for row in rows:
        trajectory = row["trajectory_id"]
        data = {l:dict(panel) for l in layers}
        witnesses, flows, latents = [], [], []
        def tap_w1(model, z):
            if int(model.scheduler.step_index) == 0:
                layer = int(model.block_idx)
                data[layer]["anchor_z0k"] = z.index_select(0, ids).index_select(1, selected[layer]).cpu()

        def tap(model, phase, u, y, gate):
            step, layer = int(model.scheduler.step_index), int(model.block_idx)
            assert u.dtype == y.dtype == torch.bfloat16 and u.shape == y.shape == (31668, 5120)
            assert f"U{step}" not in data[layer]
            data[layer].update({f"U{step}":u.index_select(0, ids).cpu(), f"Y{step}":y.index_select(0, ids).cpu()})
            if step == 1:
                gate = gate.squeeze()
                assert gate.shape == (5120,)
                data[layer]["gate1"] = gate.detach().cpu()
                probe_z = phase.ffn_0.apply(u.index_select(0, probes))
                data[layer]["probe_Y1"] = phase.ffn_2.apply(torch.nn.functional.gelu(probe_z, approximate="tanh")).cpu()

        dispatch = MultiDistillModelStruct.get_current_model_index
        native_post = sched.step_post
        def get_model(model):
            if int(sched.step_index) == 0:
                witnesses.append(sched.latents.detach().clone())
            value = dispatch(model)
            assert int(model.cur_model_index) == int(sched.step_index)//2
            return value
        def post():
            step = int(sched.step_index)
            assert step == len(flows) and base["schedule"][step] == dict(timestep=float(sched.timesteps[step]), sigma=float(sched.sigmas[step]))
            flows.append(sched.noise_pred.detach().clone())
            result = native_post()
            latents.append(sched.latents.detach().clone())
            return result

        file = out/"dense"/(trajectory+".mp4")
        start = monotonic()
        print("CAPTURE", trajectory, flush=True)
        try:
            MultiDistillModelStruct.get_current_model_index = get_model
            sched.step_post = post
            v2.reset_scheduler_generator(sched, pipe.runner.model)
            with observe(WanTransformerInfer, base["native_transformer_sha256"], layers, tap, tap_w1):
                pipe.generate(seed=row["seed"], image_path=row["image_path"], prompt=row["prompt"], negative_prompt="", save_result_path=str(file))
        finally:
            MultiDistillModelStruct.get_current_model_index = dispatch
            sched.step_post = native_post
        seconds = monotonic() - start
        assert len(witnesses) == 1 and len(flows) == len(latents) == 4
        video = v2.video_digest(file)
        assert (video["frames"], video["width"], video["height"], video["fps"]) == (81, 832, 464, 16)
        receipt = dict(trajectory_id=trajectory, source_id=row["source_id"], seed=row["seed"], condition=row["condition"],
                       image_sha256=row["image_sha256"], prompt_sha256=row["prompt_sha256"], initial=tensor_hash(witnesses[0]),
                       flows=list(map(tensor_hash, flows)), latents=list(map(tensor_hash, latents)), video=video, seconds=seconds,
                       path=str(file))
        save(file.with_suffix(".json"), receipt)
        for layer, payload in data.items():
            assert all(payload[k].isfinite().all().item() for k in ("U0","U1","Y0","Y1","gate1","anchor_z0k","probe_Y1"))
            path = out/"panels"/trajectory/f"high{layer:02d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path.with_suffix(".pending"))
            path.with_suffix(".pending").replace(path)
            state["sites"].append(dict(trajectory_id=trajectory, source_id=row["source_id"], layer=layer, path=str(path), sha256=digest(path)))
        state["trajectories"].append(receipt)
        save(out/"CAPTURE_INDEX.json", state)
        print("SAVED", trajectory, "total",len(state["trajectories"]), flush=True)
    for source in protocol["sources"]:
        paired = [r for r in state["trajectories"] if r["source_id"] == source["record_id"] and r["seed"] != 42]
        assert len(paired) == 2 and paired[0]["initial"] == paired[1]["initial"] and paired[0]["flows"][0] != paired[1]["flows"][0]
    assert len(state["sites"]) == 52
    state.update(status="complete", nfe=52)
    save(out/"CAPTURE_INDEX.json", state)


if __name__ == "__main__":
    with torch.inference_mode():
        run()
