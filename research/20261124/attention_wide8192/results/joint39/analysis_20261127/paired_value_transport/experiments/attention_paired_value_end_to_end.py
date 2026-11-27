"""Free trajectories of the low-expert paired-value and diagonal recoveries."""
from contextlib import nullcontext
from hashlib import sha256
import math
from pathlib import Path
import sys

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from attention_mass_centered_low import encode, moments
from attention_online_structured import BACKEND, OnlineAttention, SCALE, head_first, save_trajectory, schur_fit
from attention_paired_value_ops import paired_map
from generate import Experiment, digest, load, save


PLAN = "9aed8ad210683fc88284b1b243a7e512e1a33574"
DEV = ("vbi_0392", "vbi_0800", "vbi_1013")
INDEPENDENT = "vbi_0128"
ARMS = ("diagonal", "paired")
IDENTITY = ("seed", "prompt_sha256", "image_sha256", "source_cluster_id")


class EndToEnd(OnlineAttention):
    def __init__(self, experiment, fit, positions):
        super().__init__(experiment, fit)
        self.positions = torch.tensor(positions, dtype=torch.long, device="cuda")
        self.mode = None

    def low_cache(self, q, k, v, route, native, lse, arm):
        assert arm == "OUT-D"
        selected, selected_lse, _ = self.selected(q, k, v, route, False)
        basis = self.fits["low"][0]["out_delta"]["basis"]
        projected = ((selected.float() @ self.pv) @ basis).bfloat16()
        rho, recomputed = self.anchor_mass(q, k, route, selected_lse, lse)
        rho = rho[..., None]
        mu, _, inverse = moments(v)
        z = encode(native, selected, rho, mu, inverse)
        u = None
        if self.mode == "paired":
            rows = v.index_select(0, self.positions).transpose(0, 1).float()
            u = ((rows-mu)*inverse).bfloat16().contiguous()
        return (route, rho, z, projected, u), recomputed

    def approximate(self, q, k, v, cache, arm):
        if arm == "OUT-ABS":
            return super().approximate(q, k, v, cache, arm)
        assert arm == "OUT-D"
        route, rho, z, projected, u = cache
        mats, ridge, scale_r, _ = self.fits["low"]
        raw, _, _ = self.selected(q, k, v, route, False)
        selected = raw.float()
        now = ((selected @ self.pv) @ mats["out_delta"]["basis"]).bfloat16().float()
        diff = now-projected.float()
        x = torch.cat((torch.ones_like(diff[..., :1]), diff), dim=-1)
        mu, scale, _ = moments(v)
        base = (1-rho)*selected+rho*mu
        if self.mode == "paired":
            y = v.index_select(0, self.positions).transpose(0, 1).float()-mu
            m, _ = paired_map(u.float(), y, scale, 0.001)
            base.add_(z.float() @ m)
        else:
            assert self.mode == "diagonal" and u is None
            base.add_(z.float()*scale)
        probe = q.index_select(0, self.probes)
        target = F.scaled_dot_product_attention(*map(head_first, (probe, k, v)),
                    dropout_p=0.0, is_causal=False, scale=SCALE)[0]
        xp = x.index_select(1, self.probes)
        label = (target.float()-base.index_select(1, self.probes))/scale_r
        coefficient = schur_fit(xp, label, None, None, len(self.probes)*ridge[arm], mu=0)[0]
        out = (base + (x @ coefficient)*scale_r).transpose(0, 1).to(q.dtype).contiguous()
        out.index_copy_(0, self.probes, target.transpose(0, 1))
        return out, target, selected


def tensor_error(prediction, reference):
    assert prediction.shape == reference.shape
    difference = prediction.double() - reference.double()
    error = float(difference.square().sum())
    norm = float(reference.double().square().sum())
    assert math.isfinite(error) and norm > 0
    return dict(relative_l2=math.sqrt(error/norm), rmse=math.sqrt(error/reference.numel()))


def trajectory_errors(arm, ref):
    assert torch.equal(load_file(str(arm/"initial.safetensors"))["noise"], load_file(str(ref/"initial.safetensors"))["noise"])
    out = []
    for step in range(4):
        a = load_file(str(arm/f"step{step}.safetensors"))
        b = load_file(str(ref/f"step{step}.safetensors"))
        out.append({name:tensor_error(a[name], b[name]) for name in ("flow", "latent")})
    return out


def archive(experiment, source, file):
    data = load(file)
    assert data["status"] == "complete" and data["source"]["record_id"] == source
    assert data["native_sha256"] == experiment.base["native_transformer_sha256"] and data["backend_sha256"] == BACKEND
    assert all(data["source"][k] == experiment.rows[source][k] for k in IDENTITY)
    controls = {}
    for arm in ("Dense", "wide8192"):
        result = data["backgrounds"][arm]
        root = file.parent/arm/"trajectory"
        for name in ("initial", *(f"step{i}" for i in range(4))):
            assert digest(root/f"{name}.safetensors") == result["assets"][f"trajectory/{name}"]["sha256"]
        controls[arm] = (root, result["record"])
    return controls


@torch.inference_mode()
def run(fit, config):
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = load(config)
    assert spec["plan_commit"] == PLAN and spec["sources"] == [*DEV,INDEPENDENT]
    local = Path(spec["local_replay"])
    assert digest(local) == spec["local_replay_sha256"]
    positions = load(local)["value_fit_rows"]
    assert len(positions) == len(set(positions)) == 256
    method = load("CONFIRM_METHOD.json")
    assert digest("attention_online_structured.py") == method["attention_source_sha256"]
    assert digest(fit/"FIT.sha256") == method["frozen_fit_manifest_sha256"]
    exp = Experiment()
    assert digest("CONFIRM_SOURCES.json") == spec["independent_manifest_sha256"]
    row, = [r for r in load("CONFIRM_SOURCES.json")["sources"] if r["source_id"] == INDEPENDENT]
    assert INDEPENDENT not in exp.rows and digest(row["image_path"]) == row["image_sha256"]
    assert sha256(row["prompt"].encode()).hexdigest() == row["prompt_sha256"]
    exp.rows[INDEPENDENT] = dict(row, record_id=INDEPENDENT, seed=42)
    active = EndToEnd(exp, fit, positions)
    assert active.calibration == method["calibration"]
    wide = exp.build("wide8192")
    result_file = exp.output/"PAIRED_VALUE_E2E.json"
    result = dict(status="running", job=exp.job, gpu=exp.state["gpu"], plan_commit=PLAN,
        source_sha256=digest(__file__), local_replay_sha256=spec["local_replay_sha256"], probe_queries_per_candidate=5120,
        position_count=len(positions), steps=4, layer_count=40, seed=42,
        timing_scope="trajectory-instrumented, no formal timing comparison", records={})

    def render(source, arm):
        active.mode = arm
        context = active.bind("OUT-D") if arm in ARMS else nullcontext(({}, [], []))
        root = exp.output/arm/(source+"_trajectory")
        with context as (counts, _, _), save_trajectory(exp, root) as tensor_receipt:
            rec = exp.render(source, arm, None if arm == "Dense" else wide)
        assert tuple(rec["video"][k] for k in ("width", "height", "frames", "fps")) == (832,464,81,16)
        if arm in ARMS:
            assert tuple(counts[k] for k in ("attention_sites", "anchors", "hits", "native_probe_queries")) == (160,80,80,5120)
            assert rec["counters"]["tile_rows"] == 8192
        return root, dict(record=rec, attention=dict(counts), states=tensor_receipt)

    for index, source in enumerate(spec["sources"]):
        order = ARMS if index%2 == 0 else ARMS[::-1]
        candidates = {arm:render(source,arm) for arm in order}
        a,b = (candidates[k][1]["record"] for k in ARMS)
        assert a["initial"] == b["initial"] and all(a[k][:3] == b[k][:3] for k in ("flows","latents"))
        if source in DEV:
            controls = archive(exp,source,Path(spec["controls"][source]))
        else:
            controls = {arm:(root, info["record"]) for arm in ("Dense", "wide8192") for root, info in [render(source, arm)]}
        result["records"][source] = dict(fold="previously_held_out" if source == INDEPENDENT else "DEV",order=order,
            candidates={arm:item[1] for arm,item in candidates.items()}, controls={name:rec for name,(_,rec) in controls.items()},
            errors={arm:{name:trajectory_errors(root, ref) for name,(ref,_) in controls.items()} for arm,(root,_) in candidates.items()},
            paired_vs_diagonal=trajectory_errors(candidates["paired"][0],candidates["diagonal"][0]), identical_prefix_steps=3)
        save(result_file,result)
        print("E2E_SOURCE_VERIFIED",source,flush=True)
    result["status"] = "complete"
    save(result_file,result)


if __name__ == "__main__":
    run(*(Path(p).resolve() for p in sys.argv[1:]))
