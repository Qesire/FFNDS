"""Real Wan wide8192 replay followed by opt-in full defect-transport renders."""
from collections import Counter
from contextlib import contextmanager, nullcontext
from functools import wraps
import inspect
from pathlib import Path
import sys

import torch

from attention_defect_transport import TransportAttention, energy
from attention_joint39 import ffn_fit, ffn_operands, ffn_predict, native_ffn, rms_rel
from attention_online_structured import AUDIT_LAYERS, BACKEND, HEADS, HEAD_DIM, N, OnlineAttention
from generate import Experiment, digest, equal_trace, load, save


PLAN = "0496ef979a4add0c50a50351a37ced61fd0faf79"
SOURCES = ("vbi_0392", "vbi_1013")


class TailReplay:
    """Read both histories through actual last-block projection and fixed wide8192 weights."""
    layers = (39,)

    def capture_native_probe(self, block):
        return nullcontext()
    def __init__(self, experiment, wide, online):
        self.wide = wide
        self.ids, self.p = online.panel_ids, online.panel_rows
        mask = torch.ones(len(self.ids), device=self.ids.device, dtype=torch.bool)
        mask[self.p] = False
        self.h = mask.nonzero().flatten()
        self.cross = experiment.owner.infer_cross_attn
        self.post = experiment.owner.post_process
        self.cached = {}
        self.branches = None

    def anchor(self, model):
        expert = "high" if int(model.scheduler.step_index) == 0 else "low"
        engine = self.wide.inner
        assert engine.slots.owner[39] == expert
        fields = {key: getattr(engine.slots, key)[39].index_select(0, self.ids).contiguous()
                  for key in ("u0", "y0", "z0k")}
        self.cached[expert] = (fields, engine.weights[expert, 39], engine.inv_metric)

    def replay(self, model, block, pre, x, live):
        expert = "high" if int(model.scheduler.step_index) == 1 else "low"
        cached, weights, inverse = self.cached.pop(expert)
        phases = block.compute_phases[:3]
        attn, cross, ffn = phases
        _, _, gate_a, shift_f, scale_f, gate_f = model.pre_process(attn.modulation, pre.embed0)
        branches = self.branches
        self.branches = None
        assert branches is not None
        inputs = {}
        for name, row in branches.items():
            a = attn.self_attn_o.apply(row.transpose(0, 1).reshape(len(self.ids), -1).contiguous())
            residual, cross_out = self.cross(model, cross, x.clone(), pre.context, a, gate_a)
            y = residual.clone().add_(cross_out)
            u = model.modulate_func(ffn.norm2.apply(y), scale=scale_f, shift=shift_f).squeeze().contiguous()
            inputs[name] = (y, u)
        native_probe = inputs["native"][1].index_select(0, self.p)
        probe_equal = {key: bool(torch.equal(u.index_select(0, self.p), native_probe))
                       for key, (_, u) in inputs.items()}
        assert all(probe_equal.values()), probe_equal
        cache_probe = {name: value.index_select(0, self.p).contiguous() for name, value in cached.items()}
        state = ffn_fit(ffn_operands(native_probe, cache_probe, weights), native_ffn(native_probe, ffn), weights, inverse)
        exits = {}
        for key, (y, u) in inputs.items():
            value = ffn_predict(ffn_operands(u, cached, weights), state, weights["zmap"].shape[0])
            exits[key] = self.post(model, y.clone(), value, gate_f, pre)
        native = exits["native"].index_select(0, self.h)
        live_error = rms_rel(native, live.index_select(0, self.ids).index_select(0, self.h))
        assert live_error < 0.025, (expert, live_error)
        return dict(probe_ffn_input_exact=probe_equal, native_live_block_relative_l2=live_error,
                    held_block_energy={key: energy(value.index_select(0, self.h), native)
                                       for key, value in exits.items() if key != "native"})


@contextmanager
def bind(experiment, method, tail=None):
    from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight

    owner = experiment.owner
    original, block_impl, original_apply = owner.infer_self_attn, owner.infer_block, TorchSDPAWeight.apply
    assert digest(Path(inspect.getfile(original))) == experiment.base["native_transformer_sha256"]
    assert digest(Path(inspect.getfile(original_apply))) == BACKEND
    signature = inspect.signature(original_apply)
    slots, counts, reports = {}, Counter(), {}
    active = None

    @wraps(block_impl)
    def block(model, current, x, pre):
        layer = int(model.block_idx)
        observed = layer in tail.layers
        step = int(model.scheduler.step_index)
        panel = x.index_select(0, tail.ids) if observed and step % 2 else None
        with tail.capture_native_probe(current) if observed and step % 2 else nullcontext():
            output = block_impl(model, current, x, pre)
        if observed:
            if step % 2:
                reports[f"{step}:{layer}"]["wan_wide8192_tail"] = tail.replay(model, current, pre, panel, output)
            else:
                tail.anchor(model)
        return output

    @wraps(original)
    def self_attention(model, phase, *args, **kwargs):
        nonlocal active
        step, layer = int(model.scheduler.step_index), int(model.block_idx)
        assert active is None and step in (0, 1, 2, 3) and 0 <= layer < 40
        if tail is not None and layer not in AUDIT_LAYERS:
            return original(model, phase, *args, **kwargs)
        assert not model.config["seq_parallel"] and not model.config["use_image_encoder"]
        active = (step, layer, phase)
        try:
            return original(model, phase, *args, **kwargs)
        finally:
            active = None

    @wraps(original_apply)
    def apply(op, *args, **kwargs):
        if active is None:
            return original_apply(op, *args, **kwargs)
        step, layer, phase = active
        assert op is phase.self_attn_1
        call = signature.bind(op, *args, **kwargs)
        call.apply_defaults()
        p = call.arguments
        q, k, v = (p[key] for key in ("q", "k", "v"))
        assert q.shape == k.shape == v.shape == (N, HEADS, HEAD_DIM)
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        assert p["attn_mask"] is None and not p["causal"] and not p["drop_rate"]
        assert p["max_seqlen_q"] == p["max_seqlen_kv"] == N
        counts["sites"] += 1
        if step % 2 == 0:
            assert layer not in slots
            value = original_apply(op, *args, **kwargs)
            slots[layer] = method.anchor(q, k, v, value, None if tail is None else tail.ids)
            counts["anchors"] += 1
            counts["peak_slots"] = max(counts["peak_slots"], len(slots))
            return value
        state = slots.pop(layer)
        native = None if tail is None else original_apply(op, *args, **kwargs)
        audit_args = () if tail is None else (native, tail.ids, tail.p, tail.h)
        predicted, detail, native_probes = method.recover(q, k, v, state, *audit_args)
        if tail is not None:
            assert detail is not None
            report, branches = detail
            assert report["short_teacher_relative_l2"] < 0.006
            reports[f"{step}:{layer}"] = report
            if layer in tail.layers:
                tail.branches = branches
        counts["hits"] += 1
        counts["native_probe_queries"] += native_probes
        return predicted

    owner.infer_self_attn, TorchSDPAWeight.apply = self_attention, apply
    if tail is not None:
        owner.infer_block = block
    try:
        yield counts, reports
        layers = 40 if tail is None else len(AUDIT_LAYERS)
        assert not slots and counts["peak_slots"] == layers
        assert (counts["sites"], counts["anchors"], counts["hits"], counts["native_probe_queries"]) == (
            4 * layers, 2 * layers, 2 * layers, 128 * layers)
    finally:
        slots.clear()
        owner.infer_self_attn, owner.infer_block = original, block_impl
        TorchSDPAWeight.apply = original_apply


@torch.inference_mode()
def main(fit, config_path):
    config = load(config_path)
    assert config["plan_commit"] == PLAN
    assert digest(config["quality_receipt"]) == config["quality_sha256"]
    historical = load(config["quality_receipt"])
    torch.backends.cuda.matmul.allow_tf32 = False
    exp = Experiment()
    online = OnlineAttention(exp, fit)
    assert len(online.probes) == 64 and len(online.panel_ids) == 512
    method = TransportAttention(online, query_chunk=2048)
    wide = exp.build("wide8192")
    tail = TailReplay(exp, wide, online)
    state = dict(status="running", job=exp.job, plan_commit=PLAN, gpu=exp.state["gpu"],
                 method_sha256=digest(Path(__file__).with_name("attention_defect_transport.py")),
                 source_sha256=digest(__file__), reference_job=historical["job"], candidates={})
    result = exp.output / "DEFECT_TRANSPORT_RESULT.json"
    with bind(exp, method, tail) as (counters, observations):
        audit = exp.render(SOURCES[0], "native_replay", wide)
    assert equal_trace(audit, historical["control"][SOURCES[0]]["wide8192"])
    assert len(observations) == 8 and all(f"{step}:39" in observations for step in (1, 3))
    state.update(replay=dict(counters=dict(counters), record=audit, observations=observations))
    save(result, state)
    print("DEFECT_TRANSPORT_REPLAY_VERIFIED", flush=True)
    for source in SOURCES:
        with bind(exp, method) as (counters, _):
            record = exp.render(source, "candidate_transport", wide)
        reference = historical["control"][source]["wide8192"]
        assert record["initial"] == reference["initial"]
        assert all(record[key][0] == reference[key][0] for key in ("flows", "latents"))
        state["candidates"][source] = dict(record=record, counters=dict(counters))
        save(result, state)
    state["status"] = "complete"
    save(result, state)


if __name__ == "__main__":
    main(*(Path(x).resolve() for x in sys.argv[1:]))
