"""Opt-in adapters around the archived OnlineAttention/centered call boundary."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import inspect

import torch
import torch.nn.functional as F

from .attention import (AttentionReadout, LowHistory, fold_high_output,
                        partition_attention, prepare_selected_layout,
                        selected_attention, value_moments)


@dataclass
class AttentionState:
    support: object
    route_indices: torch.Tensor
    history: LowHistory | None = None
    original_cache: object = None


def native_teacher(q, k, v):
    """Same SDPA teacher shape and scale as the frozen P64 call."""
    head = lambda x: x.transpose(0, 1).unsqueeze(0)
    return F.scaled_dot_product_attention(head(q), head(k), head(v),
        dropout_p=0., is_causal=False, scale=q.shape[-1] ** -.5)[0]


class HitAttention:
    """A consumed layer's attention state, usable by standalone or block tiles."""
    def __init__(self, session, state, step):
        self.session, self.state, self.step = session, state, step
        self.reader = session.readers["high" if step == 1 else "low"]
        self.mean = self.scale = None
        self.output_projection = self.folded = None

    def bind_output_projection(self, phase):
        if self.step != 1 or not self.session.config.high_output_fold:
            return
        mm = phase.self_attn_o
        if (getattr(mm, "has_lora_branch", False) or getattr(mm, "has_diff", False)
                or hasattr(mm, "weight_diff") or hasattr(mm, "bias_diff")):
            raise ValueError("High folding requires the native affine W_o without extra branches")
        weight = mm._get_actual_weight()
        if weight.is_cuda:
            from lightx2v.common.ops.mm.mm_weight import MMWeight
            from fourstep_anchor import SOURCES, _fingerprint
            if (type(mm) is not MMWeight or getattr(mm.apply, "__func__", None) is not MMWeight.apply
                    or _fingerprint(MMWeight) != SOURCES["mm_weight"]):
                raise ValueError("High folding requires the bound default affine MM implementation")
        self.output_projection = mm

    def prepare(self, q_p, k, v, probes):
        torch._assert_async((probes == self.reader.probes).all(), "Attention/FFN probe order differs")
        if self.state.history is not None:
            self.mean, self.scale, _ = value_moments(v)
        selected = self.session.selected_subset(q_p, k, v, self.state, self.reader.probes)
        truth = self.session.teacher(q_p, k, v)
        fitted = self.reader.fit(selected, truth, cache=self.state.history,
                                 mean=self.mean, scale=self.scale)
        if self.output_projection is not None:
            weight = self.output_projection._get_actual_weight()
            bias = self.output_projection._get_actual_bias()
            self.folded = fold_high_output(self.reader, fitted, weight, bias=bias,
                                            storage_dtype=q_p.dtype)
        return fitted, truth.transpose(0, 1).contiguous()

    def read(self, q, k, v, token_ids, fitted):
        """Consume a RowPlan nonprobe tile; P rows follow the separate teacher chain."""
        selected = self.session.selected_subset(q, k, v, self.state, token_ids)
        if self.folded is not None:
            from .wan_block import ProjectedAttentionRows
            return ProjectedAttentionRows(self.folded.apply(selected, token_ids, output_dtype=q.dtype,
                                                            overwrite_probes=False))
        result = self.reader.read(selected, token_ids, fitted, cache=self.state.history,
            mean=self.mean, scale=self.scale, output_dtype=q.dtype,
            joint_mode=self.session.config.joint_readout, overwrite_probes=False)
        return result.transpose(0, 1).contiguous()

    def full(self, q, k, v):
        cfg, probes = self.session.config, self.reader.probes
        if cfg.attention_streaming or cfg.block_streaming:
            fitted, truth = self.prepare(q.index_select(0, probes), k, v, probes)
            out = torch.empty_like(q)
            plan = self.session.row_plan(len(q), probes)
            for ids in plan.tiles:
                if len(ids):
                    out.index_copy_(0, ids, self.read(q.index_select(0, ids), k, v, ids, fitted))
            out.index_copy_(0, probes, truth)
            return out
        # Retain the frozen selected attention kernel and full query shape when
        # only changing history representation or a projection contraction.
        selected, _, _ = self.session.legacy.selected(q, k, v, self.state.support, False)
        truth = self.session.teacher(q.index_select(0, probes), k, v)
        if self.state.history is not None:
            self.mean, self.scale, _ = value_moments(v)
        fitted = self.reader.fit(selected.index_select(1, probes), truth,
            cache=self.state.history, mean=self.mean, scale=self.scale)
        ids = torch.arange(len(q), device=q.device)
        out = self.reader.read(selected, ids, fitted, cache=self.state.history,
            mean=self.mean, scale=self.scale, output_dtype=q.dtype,
            joint_mode=cfg.joint_readout)
        return out.transpose(0, 1).contiguous()


class AttentionSession:
    """Four-step route/history lifecycle, with no changes to frozen modules."""
    def __init__(self, legacy, config, *, basis_loader=None, layers=40,
                 teacher=native_teacher, native_flash=None):
        self.legacy, self.config, self.basis_loader = legacy, config, basis_loader
        self.layers, self.teacher = layers, teacher
        self.native_flash = native_flash
        self.probes = legacy.probes
        self.slots, self.counter, self.readers = {}, Counter(), {}
        self._probe_layouts = {}
        self._row_plans, self._tile_layouts = {}, {}
        self._bound = False
        for expert, arm, key in (("high", "OUT-ABS", "out_current"),
                                 ("low", "OUT-D", "out_delta")):
            matrices, lambdas, scale_r, _ = legacy.fits[expert]
            self.readers[expert] = AttentionReadout(legacy.pv, matrices[key]["basis"],
                self.probes, ridge_lambda=lambdas[arm], scale_r=scale_r,
                precompose=config.projection_precompose)

    def row_plan(self, total_rows, probes=None):
        """One fixed layout per geometry/probe tensor, reused by all layers."""
        from .block import make_row_plan
        probes = self.probes if probes is None else probes
        key = total_rows, probes.device, probes.data_ptr(), self.config.tile_rows
        if key not in self._row_plans:
            torch._assert_async((probes == self.probes).all(), "Attention/FFN probe orders differ")
            plan = make_row_plan(total_rows, probes, self.config.tile_rows)
            self._row_plans[key] = plan
            if self.config.selected_backend == "triton":
                for ids in plan.tiles:
                    self._tile_layouts[(total_rows, ids.data_ptr())] = prepare_selected_layout(ids, total_rows)
        return self._row_plans[key]

    def selected_subset(self, q, k, v, state, ids):
        layout = None
        if self.config.selected_backend == "triton":
            if ids is self.probes:
                key = (len(k), q.device)
                if key not in self._probe_layouts:
                    self._probe_layouts[key] = prepare_selected_layout(ids, len(k))
                layout = self._probe_layouts[key]
            else:
                key = len(k), ids.data_ptr()
                if key not in self._tile_layouts:
                    self._tile_layouts[key] = prepare_selected_layout(ids, len(k))
                layout = self._tile_layouts[key]
        return selected_attention(q, k, v, state.route_indices, ids,
            backend=self.config.selected_backend, layout=layout, validate=False)[0]

    def anchor(self, step, layer, q, k, v, native_call):
        if step not in (0, 2) or layer in self.slots:
            raise RuntimeError("Unexpected attention anchor or unconsumed state")
        support, route = self.legacy.route(q, k)
        if step == 0:
            result = native_call()
            state = AttentionState(support, route)
        elif self.config.anchor_partition == "native":
            flash = self.native_flash
            if flash is None:
                from attention_online_structured import native_flash as flash
            native, lse = flash(q, k, v)
            original, reconciled = self.legacy.low_cache(q, k, v, support, native[0], lse[0], "OUT-D")
            _, rho, c, a, unused = original
            if unused is not None:
                raise ValueError("Expected centered OUT-D history")
            history = LowHistory.from_fields(c, a, rho, self.probes)
            state = AttentionState(support, route, history, original)
            result = native[0].transpose(0, 1).reshape(len(q), -1).contiguous()
            self.counter["low_anchor_query_heads_with_direct_omitted_mass"] += reconciled
        else:
            mean, _, inverse = value_moments(v)
            parts = partition_attention(q, k, v, route, backend=self.config.anchor_partition,
                                        mean=mean, inverse=inverse, validate=False)
            projected = self.readers["low"].features(parts.selected).to(q.dtype)
            history = LowHistory.from_fields(parts.centered, projected, parts.rho, self.probes)
            original = (support, parts.rho, parts.centered, projected, None)
            state = AttentionState(support, route, history, original)
            result = parts.full.transpose(0, 1).reshape(len(q), -1).contiguous()
            self.counter["partition_anchors"] += 1
        if step == 2 and self.basis_loader is not None:
            asset = self.basis_loader("low", layer)
            basis = asset["basis"] if isinstance(asset, dict) else asset
            basis = basis.to(device=q.device)
            state.history = state.history.encode_joint(basis,
                tile_rows=self.config.tile_rows, code_dtype=getattr(torch, self.config.attention_code_dtype))
            state.original_cache = None  # release full C/A; no hidden old payload
            self.counter["joint_encoded_anchors"] += 1
        self.slots[layer] = state
        self.counter["anchors"] += 1
        self.counter["attention_sites"] += 1
        live_bytes = sum(s.history.storage_bytes() for s in self.slots.values() if s.history is not None)
        self.counter["history_payload_peak_bytes"] = max(self.counter["history_payload_peak_bytes"], live_bytes)
        return result

    def take_hit(self, step, layer):
        if step not in (1, 3) or layer not in self.slots:
            raise RuntimeError("Attention hit has no corresponding anchor")
        state = self.slots.pop(layer)
        if (step == 1) != (state.history is None):
            raise RuntimeError("Attention state belongs to another expert")
        self.counter["hits"] += 1
        self.counter["attention_sites"] += 1
        self.counter["native_probe_queries"] += len(self.probes)
        return HitAttention(self, state, step)

    @contextmanager
    def bind(self, *, owner=None, op_type=None):
        if self._bound or self.slots:
            raise RuntimeError("Attention session is already active")
        if owner is None:
            owner = self.legacy.exp.owner
        if op_type is None:
            from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight
            op_type = TorchSDPAWeight
        original_self, original_apply = owner.infer_self_attn, op_type.apply
        signature = inspect.signature(original_apply)
        active = None
        self.counter = Counter()

        @wraps(original_self)
        def infer_self(model, phase, *args, **kwargs):
            nonlocal active
            if active is not None:
                raise RuntimeError("Reentrant self-attention call")
            active = int(model.scheduler.step_index), int(model.block_idx), phase
            try:
                return original_self(model, phase, *args, **kwargs)
            finally:
                active = None

        @wraps(original_apply)
        def apply(op, *args, **kwargs):
            if active is None:
                return original_apply(op, *args, **kwargs)
            step, layer, phase = active
            if op is not phase.self_attn_1:
                raise ValueError("Unexpected attention operator at self boundary")
            bound = signature.bind(op, *args, **kwargs)
            bound.apply_defaults()
            p = bound.arguments
            q, k, v = (p[key] for key in ("q", "k", "v"))
            if p.get("attn_mask") is not None or p.get("causal", False) or p.get("drop_rate", 0):
                raise ValueError("Expected the frozen unmasked, noncausal, no-dropout graph")
            if step % 2 == 0:
                return self.anchor(step, layer, q, k, v, lambda: original_apply(op, *args, **kwargs))
            hit = self.take_hit(step, layer)
            if not self.config.uses_new_attention_readout:
                cache = hit.state.support if step == 1 else hit.state.original_cache
                predicted = self.legacy.approximate(q, k, v, cache, "OUT-ABS" if step == 1 else "OUT-D")[0]
            else:
                predicted = hit.full(q, k, v)
            return predicted.reshape(len(q), -1)

        owner.infer_self_attn, op_type.apply, self._bound = infer_self, apply, True
        try:
            yield self.counter
            if self.slots or tuple(self.counter[k] for k in ("attention_sites", "anchors", "hits", "native_probe_queries")) != (
                    4*self.layers, 2*self.layers, 2*self.layers, 2*self.layers*len(self.probes)):
                raise RuntimeError("Incomplete optimized attention HHLL trajectory")
        finally:
            owner.infer_self_attn, op_type.apply = original_self, original_apply
            self.slots.clear()
            self._bound = False
