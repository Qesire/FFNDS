"""Shared-probe joint replay with stage Grams and an explicit routing-action witness."""
from contextlib import contextmanager
from pathlib import Path
import math
import sys

import torch

from attention_defect_transport import TransportAttention, channel_scale, fit, group_catalog
from attention_online_structured import AUDIT_LAYERS, OnlineAttention
from generate import Experiment, digest, equal_trace, load, save
from run_attention_defect_transport import TailReplay, bind, ffn_fit, ffn_operands, ffn_predict, native_ffn, rms_rel


PLAN = "51b6e235964df2f1fb1370604387cb9c998f6591"
SOURCES = ("vbi_0392", "vbi_1013")
METHOD_SHA = "52fd025d3b82eb7be12002e38b98ddffa7e5417932dbc3ee93cb86d209c8c752"


def gram(*fields):
    rows = torch.stack([x.reshape(-1).double() for x in fields])
    return (rows @ rows.T).cpu().tolist()


def standardized(v):
    scale, inverse = channel_scale(v)
    v = v.float()
    return (v - v.mean(1, keepdim=True)) * inverse[:, None], scale


def route_action(query, k, v, labels, selected, rows, direction, block, chunk=32):
    """Apply R=S-S0 to supplied value columns using all actual keys, on panel rows."""
    _, counts, means_k, means_v = group_catalog(k, v, direction, block, labels)
    parent_count = counts.shape[1]
    assert selected.shape[-1] <= parent_count and len(rows) == query.shape[1]
    mask = torch.zeros((*selected.shape[:2], parent_count), dtype=torch.bool, device=k.device)
    mask.scatter_(2, selected.long(), True)
    log_count = counts.flatten(1).log()
    kt = k.float().transpose(1, 2).contiguous()
    mkt = means_k.flatten(1, 2).transpose(1, 2).contiguous()
    mv = means_v.flatten(1, 2).contiguous()
    scale = 1 / math.sqrt(k.shape[-1])
    results = []
    for queries, ids in zip(query.split(chunk, 1), rows.split(chunk), strict=True):
        q = queries.float()
        logits = (q @ kt).mul_(scale)
        native = logits.softmax(-1) @ v
        keep = mask.index_select(1, ids // block)
        missing = ~keep.repeat_interleave(block, dim=-1)[..., :k.shape[1]]
        logits.masked_fill_(missing, -torch.inf)
        coarse = (q @ mkt).mul_(scale).add_(log_count[:, None])
        coarse.masked_fill_(keep.repeat_interleave(2, dim=-1), -torch.inf)
        shift = torch.maximum(logits.amax(-1), coarse.amax(-1))
        weights = logits.sub_(shift[..., None]).exp_()
        coarse.sub_(shift[..., None]).exp_()
        grouped = (weights @ v + coarse @ mv) / (weights.sum(-1) + coarse.sum(-1))[..., None]
        results.append(native - grouped)
    return torch.cat(results, dim=1)


class MeasuredTransport(TransportAttention):
    def __init__(self, online):
        super().__init__(online, query_chunk=2048)
        self.operands = {}
        self.hit_values = None

    def anchor(self, q, k, v, native, audit_rows=None):
        state = super().anchor(q, k, v, native, audit_rows)
        assert audit_rows is not None
        self.operands[id(state)] = (q.index_select(0, audit_rows).transpose(0, 1).contiguous(),
                                    k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous())
        return state

    def recover(self, q, k, v, state, *args):
        self.hit_values = q, k, v
        try:
            return super().recover(q, k, v, state, *args)
        finally:
            self.hit_values = None

    def audit(self, base, state, truth, coefficients, xp, gp, teacher, short, panel, p, h):
        report, branches = super().audit(base, state, truth, coefficients, xp, gp, teacher, short, panel, p, h)
        qa, ka, va = self.operands.pop(id(state))
        qs, ks, vs = self.hit_values
        qa_value, sa = standardized(va)
        qs_value, ss = standardized(vs.transpose(0, 1))
        anchor_a, anchor_s = route_action(qa, ka, torch.cat((qa_value, qs_value), -1), state.labels,
                    state.selected, panel, self.pk[:, 0], self.block).split(qa_value.shape[-1], dim=-1)
        current = route_action(qs.index_select(0, panel).transpose(0, 1), ks.transpose(0, 1), qs_value,
                               state.labels, state.selected, panel, self.pk[:, 0], self.block)
        routed = current - anchor_s
        shaped = anchor_s - anchor_a
        innovation = current - anchor_a
        assert torch.allclose(routed + shaped, innovation, atol=2e-6, rtol=2e-5)
        query, selected, grouped, mass = base.evaluate(panel)
        psi = base.features(query, selected, grouped, mass, self.pk, self.pv)
        held = lambda value: value.index_select(1, h)
        previous_phys = anchor_a * sa[:, None]
        current_phys = current * ss[:, None]
        expected_a = state.normalized_panel
        expected_s = (branches["native"].float() - grouped) * base.inverse[:, None]
        witness = {"anchor_relative_l2_physical": rms_rel(held(previous_phys), held(state.raw_panel)),
                   "hit_relative_l2_physical": rms_rel(held(current_phys), held(branches["native"].float() - grouped)),
                   "anchor_relative_l2_standardized": rms_rel(held(anchor_a), held(expected_a)),
                   "hit_relative_l2_standardized": rms_rel(held(current), held(expected_s))}
        active = (ss > 0) & (sa > 0)
        scale_ratio = ss[active] / sa[active]
        raw_remaining = current_phys - previous_phys * (ss > 0)[:, None]
        scale_change = anchor_a * (ss - sa)[:, None] * (ss > 0)[:, None]
        raw_route, raw_shape = routed * ss[:, None], shaped * ss[:, None]
        assert torch.allclose(raw_route + raw_shape + scale_change, raw_remaining, atol=2e-6, rtol=2e-5)

        def remaining(value):
            return held(value - psi @ fit(xp, value.index_select(1, p)))

        report["innovation_decomposition"] = dict(witness=witness,
            route_shape_gram_standardized=gram(held(routed), held(shaped)),
            route_shape_scale_gram_physical=gram(held(raw_route), held(raw_shape), held(scale_change)),
            route_shape_scale_post_fit_gram_physical=gram(*map(remaining, (raw_route, raw_shape, scale_change))),
            normalized_value_shape_relative_l2=rms_rel(qs_value, qa_value),
            current_to_anchor_scale_quantiles=torch.quantile(scale_ratio, torch.tensor([.1, .5, .9], device=ss.device)).tolist(),
            independent_queries=len(panel), keys=ks.shape[0])
        return report, branches


class JointTail(TailReplay):
    layers = AUDIT_LAYERS
    live_arm = "native"
    arms = ("raw", "transport")
    edges = None

    @contextmanager
    def capture_native_probe(self, block):
        phase = block.compute_phases[2]
        first, last = phase.ffn_0, phase.ffn_2
        original_first, original_last = first.apply, last.apply
        captured = {}

        def observe_input(u):
            assert u.shape[0] == len(self.p) and "input" not in captured
            captured["input"] = u.clone()
            return original_first(u)

        def observe_label(value):
            result = original_last(value)
            assert result.shape[0] == len(self.p) and "label" not in captured
            captured["label"] = result.clone()
            return result

        first.apply, last.apply = observe_input, observe_label
        try:
            yield
        finally:
            first.apply, last.apply = original_first, original_last
        assert set(captured) == {"input", "label"}
        self.native_probe = captured

    def anchor(self, model):
        expert = "high" if int(model.scheduler.step_index) == 0 else "low"
        layer = int(model.block_idx)
        engine = self.wide.inner
        assert engine.slots.owner[layer] == expert
        fields = {key: getattr(engine.slots, key)[layer].index_select(0, self.ids).contiguous()
                  for key in ("u0", "y0", "z0k")}
        self.cached[expert, layer] = (fields, engine.weights[expert, layer], engine.inv_metric)

    def replay(self, model, block, pre, x, live):
        expert = "high" if int(model.scheduler.step_index) == 1 else "low"
        layer = int(model.block_idx)
        cached, weights, inverse = self.cached.pop((expert, layer))
        attention, cross, ffn = block.compute_phases[:3]
        _, _, gate_a, shift_f, scale_f, gate_f = model.pre_process(attention.modulation, pre.embed0)
        branches, self.branches = self.branches, None
        assert branches is not None
        graphs = {}
        for name in ("native", *self.arms):
            head = branches[name].transpose(0, 1).reshape(len(self.ids), -1).contiguous()
            projected = attention.self_attn_o.apply(head)
            residual, cross_out = self.cross(model, cross, x.clone(), pre.context, projected.clone(), gate_a)
            merged = residual.clone().add_(cross_out)
            u = model.modulate_func(ffn.norm2.apply(merged), scale=scale_f, shift=shift_f).squeeze().contiguous()
            graphs[name] = dict(attention_head=head, output_projection=projected,
                attention_gate_response=projected * gate_a.reshape(-1), self_residual=residual,
                cross_response=cross_out, cross_residual=merged, ffn_input=u)
        native = graphs["native"]
        probe_u = native["ffn_input"].index_select(0, self.p)
        assert all(torch.equal(row["ffn_input"].index_select(0, self.p), probe_u) for row in graphs.values())
        actual = self.native_probe
        self.native_probe = None
        probe_drift = rms_rel(probe_u, actual["input"])
        cache_probe = {key: values.index_select(0, self.p).contiguous() for key, values in cached.items()}
        state = ffn_fit(ffn_operands(actual["input"], cache_probe, weights), actual["label"], weights, inverse)
        for row in graphs.values():
            value = ffn_predict(ffn_operands(row["ffn_input"], cached, weights), state, weights["zmap"].shape[0])
            row.update(ffn_response=value, ffn_gate_response=value * gate_f.reshape(-1),
                       block_exit=self.post(model, row["cross_residual"].clone(), value, gate_f, pre))
        held = lambda z: z.index_select(0, self.h)
        live_relative = rms_rel(held(graphs[self.live_arm]["block_exit"]), held(live.index_select(0, self.ids)))
        assert live_relative < 0.025, (expert, layer, live_relative)
        # Preserve the original fitting teacher and obtain the missing Dense reference on H only.
        dense_ffn = native_ffn(held(native["ffn_input"]), ffn)
        dense_exit = self.post(model, held(native["cross_residual"]).clone(), dense_ffn, gate_f, pre)
        native_grams, dense_grams = {}, {}
        dense = {"ffn_response": dense_ffn, "ffn_gate_response": dense_ffn * gate_f.reshape(-1), "block_exit": dense_exit}
        for stage, ref in native.items():
            ref_h = held(ref).float()
            differences = [held(graphs[name][stage]).float() - ref_h for name in self.arms]
            native_grams[stage] = gram(*differences)
            if stage in dense:
                background = ref_h - dense[stage].float()
                dense_grams[stage] = gram(*differences, background)
        result = dict(shared_probe_ffn_input_exact=True, live_relative_l2=live_relative,
                    shared_probe_vs_live_relative_l2=probe_drift, reused_live_ffn_probe_teacher=True,
                    axes=[len(self.h), native["ffn_input"].shape[-1]],
                    stage_gram_raw_transport=native_grams, stage_gram_raw_transport_background=dense_grams,
                    dense_native_ffn_extra_held_rows=len(self.h))
        if self.edges is not None:
            # Edge row IDs refer to the full panel. Map H-only Dense output into it.
            dense_panel = torch.zeros_like(native["block_exit"])
            dense_panel.index_copy_(0, self.h, dense_exit)
            result["block_edges"] = {}
            for kind, (left, right) in self.edges.items():
                delta = lambda y: y.index_select(0,left).float()-y.index_select(0,right).float()
                wide_edge, dense_edge = delta(native["block_exit"]), delta(dense_panel)
                errors = [delta(graphs[arm]["block_exit"])-wide_edge for arm in self.arms]
                result["block_edges"][kind] = gram(*errors, wide_edge-dense_edge)
        return result


@torch.inference_mode()
def main(fit_dir, config_path):
    config = load(config_path)
    assert config["plan_commit"] == PLAN and digest(config["quality_receipt"]) == config["quality_sha256"]
    assert digest(Path(__file__).with_name("attention_defect_transport.py")) == METHOD_SHA
    historical = load(config["quality_receipt"])
    torch.backends.cuda.matmul.allow_tf32 = False
    exp = Experiment()
    online = OnlineAttention(exp, fit_dir)
    assert len(online.probes) == 64 and len(online.panel_ids) == 512
    method = MeasuredTransport(online)
    wide = exp.build("wide8192")
    file = exp.output / "JOINT_ACTION_RESULT.json"
    result = dict(status="running", job=exp.job, plan_commit=PLAN, source_sha256=digest(__file__),
                  method_sha256=METHOD_SHA, gpu=exp.state["gpu"], layers=AUDIT_LAYERS, sources={})
    for source in SOURCES:
        tail = JointTail(exp, wide, online)
        with bind(exp, method, tail) as (counts, sites):
            row = exp.render(source, "joint_replay", wide)
        assert equal_trace(row, historical["control"][source]["wide8192"])
        assert not method.operands and not tail.cached and len(sites) == 8
        result["sources"][source] = dict(trace_exact=True, track=row, counters=dict(counts), sites=sites)
        save(file, result)
        print("JOINT_ACTION_REPLAY_VERIFIED", source, flush=True)
    result["status"] = "complete"
    save(file, result)


if __name__ == "__main__":
    main(*(Path(path).resolve() for path in sys.argv[1:]))
