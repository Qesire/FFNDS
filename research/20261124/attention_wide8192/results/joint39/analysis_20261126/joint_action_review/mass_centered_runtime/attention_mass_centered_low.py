"""Mass-center the frozen OUT-D history in place, with matched Wan replay and timing."""
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from attention_joint_action import JointTail, gram
from attention_online_structured import AUDIT_LAYERS, OnlineAttention, SCALE, head_first, schur_fit
from generate import Experiment, digest, equal_trace, load, save


PLAN = "ff8fd9dcbed3812b1da7f589f3d2e817f48c26c9"
SOURCES = ("vbi_0392", "vbi_1013")
FLOOR = 1e-12


def moments(value):
    variance, mean = torch.var_mean(value.float(), dim=0, correction=0)
    scale = variance.clamp_min(0).sqrt()
    scale = torch.where(scale > FLOOR, scale, 0)
    inverse = torch.where(scale > 0, scale.reciprocal(), 0)
    return mean[:, None], scale[:, None], inverse[:, None]


def encode(native, selected, rho, mean, inverse):
    history = native.float() - (1 - rho) * selected.float()
    return ((history - rho * mean) * inverse).to(native.dtype)


def decode(selected, rho, centered, mean, scale):
    return ((1 - rho) * selected.float() + rho * mean) + centered.float() * scale


def energy(value):
    return float(value.double().square().sum())


class MassCenteredLow(OnlineAttention):
    def __init__(self, experiment, calibration):
        super().__init__(experiment, calibration)
        self.get_moments = torch.compile(moments, fullgraph=True, dynamic=False)
        self.encode_cache = torch.compile(encode, fullgraph=True, dynamic=False)
        self.decode_base = torch.compile(decode, fullgraph=True, dynamic=False)
        self.mode = "raw"
        self.layer = None
        self.shadow = None
        self.anchor_shadow = {}
        self.diagnostics = {}

    def low_cache(self, q, k, v, route, native, lse, arm):
        measured = self.layer in AUDIT_LAYERS
        if arm != "OUT-D" or (self.mode == "raw" and not measured):
            return super().low_cache(q, k, v, route, native, lse, arm)
        selected, lse_s, _ = self.selected(q, k, v, route, False)
        matrices, _, _, _ = self.fits["low"]
        projected = ((selected.float() @ self.pv) @ matrices["out_delta"]["basis"]).bfloat16()
        rho, recomputed = self.anchor_mass(q, k, route, lse_s, lse)
        rho = rho.unsqueeze(-1)
        mean, scale, inverse = self.get_moments(v)
        centered = self.encode_cache(native, selected, rho, mean, inverse)
        if measured:
            assert self.mode == "raw"
            raw = (native.float() - (1-rho)*selected.float()).bfloat16()
            self.anchor_shadow[self.layer] = centered.index_select(1, self.panel_ids)
            native_p = native.index_select(1, self.panel_ids).float()
            selected_p = selected.index_select(1, self.panel_ids)
            rho_p = rho.index_select(1, self.panel_ids)
            reconstructed = decode(selected_p, rho_p, self.anchor_shadow[self.layer], mean, scale)
            raw_reconstructed = (1-rho_p)*selected_p.float() + raw.index_select(1, self.panel_ids).float()
            target_norm = max(energy(native_p), 1e-30)
            self.diagnostics[self.layer] = dict(anchor_relative_l2=(energy(reconstructed-native_p)/target_norm)**0.5,
                anchor_raw_relative_l2=(energy(raw_reconstructed-native_p)/target_norm)**0.5,
                inactive_anchor_channels=int((scale==0).sum()), negative_rho_rows=int((rho<0).sum()), rho_min=float(rho.min()))
            centered = raw
        return (route, rho, centered, projected, None), recomputed

    def approximate(self, q, k, v, cache, arm):
        measured = self.layer in AUDIT_LAYERS
        if arm != "OUT-D" or (self.mode == "raw" and not measured):
            return super().approximate(q, k, v, cache, arm)
        route, rho, history, anchor_design, unused = cache
        assert unused is None
        matrices, lambdas, scale_r, _ = self.fits["low"]
        selected, _, _ = self.selected(q, k, v, route, False)
        selected = selected.float()
        diff = ((selected @ self.pv) @ matrices["out_delta"]["basis"]).bfloat16().float() - anchor_design.float()
        x = torch.cat((torch.ones_like(diff[..., :1]), diff), dim=-1)
        x_p = x.index_select(1, self.probes)
        truth = self.teacher(q.index_select(0, self.probes), k, v)
        alpha = len(self.probes)*lambdas[arm]
        mean, scale, _ = self.get_moments(v)
        if measured:
            base = (1-rho)*selected + history.float()
        else:
            base = self.decode_base(selected, rho, history, mean, scale)
        coefficient = schur_fit(x_p, (truth.float()-base.index_select(1, self.probes))/scale_r,
                                None, None, alpha, mu=0)[0]
        estimate = base + (x @ coefficient)*scale_r
        predicted = estimate.transpose(0, 1).to(q.dtype).contiguous()
        predicted.index_copy_(0, self.probes, truth.transpose(0, 1))
        if measured:
            self.compare_shadow(q, k, v, rho, selected, x, x_p, base, estimate, truth, mean, scale, alpha, scale_r)
        return predicted, truth, selected

    @staticmethod
    def teacher(q, k, v):
        return F.scaled_dot_product_attention(*map(head_first, (q,k,v)), dropout_p=0.0,
                                              is_causal=False, scale=SCALE)[0]

    def compare_shadow(self, q, k, v, rho, selected, x, x_p, base, estimate, truth, mean, scale, alpha, scale_r):
        panel, p = self.panel_ids, self.panel_rows
        mask = torch.ones(len(panel), device=panel.device, dtype=torch.bool)
        mask[p] = False
        h = mask.nonzero().flatten()
        current = decode(selected.index_select(1, panel), rho.index_select(1, panel),
                         self.anchor_shadow.pop(self.layer), mean, scale)
        current_p = current.index_select(1, p)
        coef = schur_fit(x_p, (truth.float()-current_p)/scale_r, None, None, alpha, mu=0)[0]
        x_panel = x.index_select(1, panel)
        shadow = current + (x_panel @ coef)*scale_r
        raw = estimate.index_select(1, panel)
        base_panel = base.index_select(1, panel)
        delta = current - base_panel
        delta_coef = schur_fit(x_p, -delta.index_select(1, p)/scale_r, None, None, alpha, mu=0)[0]
        expected = delta + (x_panel @ delta_coef)*scale_r
        measured = shadow - raw
        h_truth = self.teacher(q.index_select(0, panel.index_select(0, h)), k, v)
        native = torch.empty_like(raw, dtype=q.dtype)
        native.index_copy_(1, h, h_truth)
        native.index_copy_(1, p, truth)
        raw, shadow = raw.bfloat16(), shadow.bfloat16()
        raw.index_copy_(1, p, truth)
        shadow.index_copy_(1, p, truth)
        p_equal = all(torch.equal(x.index_select(1,p), truth) for x in (raw, shadow))
        assert p_equal
        self.diagnostics[self.layer].update(inactive_current_channels=int((scale==0).sum()),
            identical_final_attention_probes=p_equal,
            difference_identity_relative_l2=(energy((measured-expected).index_select(1,h)) /
                                              max(energy(measured.index_select(1,h)), 1e-30))**0.5,
            held_base_error_gram=gram(*(y.index_select(1,h).float()-h_truth.float() for y in (base_panel,current))))
        self.shadow = dict(native=native, raw=raw, transport=shadow)


class LowTail(JointTail):
    live_arm = "raw"

    @contextmanager
    def bind(self, online):
        owner = online.exp.owner
        original = owner.infer_block

        @wraps(original)
        def block(model, weights, x, pre):
            step, layer = int(model.scheduler.step_index), int(model.block_idx)
            audit = step in (2, 3) and layer in AUDIT_LAYERS
            if not audit:
                return original(model, weights, x, pre)
            current = x.index_select(0, self.ids) if step == 3 else None
            online.layer = layer
            try:
                with self.capture_native_probe(weights) if step == 3 else nullcontext():
                    live = original(model, weights, x, pre)
            finally:
                online.layer = None
            if step == 2:
                self.anchor(model)
            else:
                self.branches, online.shadow = online.shadow, None
                online.diagnostics[layer]["tail"] = self.replay(model, weights, pre, current, live)
            return live

        owner.infer_block = block
        try:
            yield
        finally:
            owner.infer_block = original


@torch.inference_mode()
def main(fit_dir, config_path):
    conf = load(config_path)
    assert conf["plan_commit"] == PLAN and digest(conf["quality_receipt"]) == conf["quality_sha256"]
    recipe = load("CONFIRM_METHOD.json")
    assert digest("attention_online_structured.py") == recipe["attention_source_sha256"]
    assert digest(fit_dir / "FIT.sha256") == recipe["frozen_fit_manifest_sha256"]
    historical = load(conf["quality_receipt"])
    expected = {r["source"]:r["record"] for r in historical["renders"] if r["arm"] == "C"}
    torch.backends.cuda.matmul.allow_tf32 = False
    exp = Experiment()
    online = MassCenteredLow(exp, fit_dir)
    assert online.calibration == recipe["calibration"]
    wide = exp.build("wide8192")
    path = exp.output / "MASS_CENTERED_RESULT.json"
    result = dict(status="running", job=exp.job, plan_commit=PLAN, source_sha256=digest(__file__),
                  gpu=exp.state["gpu"], fixed={}, warmup={}, pairs=[],
                  scope=dict(anchor_step=2, hit_step=3, audit_layers=AUDIT_LAYERS, frozen_high=True,
                             sources=SOURCES, timing_repeats=2, threshold=FLOOR, probes=64))

    def render(source, mode, group, tail=None):
        online.mode = mode
        with (tail.bind(online) if tail else nullcontext()), online.bind("OUT-D") as (count,_,_):
            row = exp.render(source, group, wide)
        assert tuple(count[k] for k in ("attention_sites", "anchors", "hits", "native_probe_queries")) == (160,80,80,5120)
        assert tuple(row["counters"][k] for k in ("tile_rows", "anchors", "hits")) == (8192,80,80)
        assert row["video"]["frames"] == 81
        return dict(record=row, attention=dict(count))

    for source in SOURCES:
        tail = LowTail(exp, wide, online)
        row = render(source, "raw", "fixed_raw", tail)
        assert equal_trace(row["record"], expected[source])
        assert not online.anchor_shadow and not tail.cached and set(online.diagnostics) == set(AUDIT_LAYERS)
        result["fixed"][source] = dict(track=row, sites=online.diagnostics)
        online.diagnostics = {}
        save(path, result)
        print("MASS_CENTERED_FIXED_VERIFIED", source, flush=True)

    for mode in ("raw", "centered"):
        row = render(SOURCES[0], mode, "warm_"+mode)
        result["warmup"][mode] = row
        save(path, result)

    for repeat in (0,1):
        for i, source in enumerate(SOURCES):
            order = ("raw", "centered") if (i+repeat)%2 == 0 else ("centered", "raw")
            records = {mode:render(source, mode, f"pair_{repeat}_{mode}") for mode in order}
            raw, centered = (records[k]["record"] for k in ("raw", "centered"))
            assert equal_trace(raw, expected[source])
            assert raw["initial"] == centered["initial"]
            assert all(raw[k][:3] == centered[k][:3] for k in ("flows", "latents"))
            result["pairs"].append(dict(repeat=repeat, source=source, order=order, arms=records, prefix_exact=True))
            save(path, result)
    result["status"] = "complete"
    save(path, result)


if __name__ == "__main__":
    main(*(Path(p).resolve() for p in sys.argv[1:]))
