"""Row-streamed hit-block adapter for the inspected native Wan BF16 graph.

No LightX2V import is required here: actual phase/model operations are supplied
by the existing runtime. Anchors are dispatched to their existing path by the
outer binding. No frozen model weights, CAL choices, or P coordinates change.
"""
from dataclasses import dataclass

import torch

from .block import stream_hit_block


def select_modulation(value, token_ids, total_rows, width):
    """Select original rows for token-wise modulation; retain shared channels."""
    v = value.squeeze()
    if v.ndim == 1 and v.numel() == width:
        return v
    if v.ndim == 2 and v.shape == (total_rows, width):
        return v.index_select(0, token_ids)
    raise ValueError(f"Unsupported modulation layout {tuple(value.shape)} for N={total_rows}, d={width}")


def native_rope_rows(model, tensor, token_ids):
    """Gather the global cache rows, then reuse the actual selected RoPE backend.

    Wan pre_infer stores token axis 0 for both flashinfer real cos/sin and torch
    complex caches. The second input is a private clone because some backends
    mutate both arguments. This computes an unused second RoPE result rather
    than an unused Q projection; callers may inject a verified single-input
    backend to remove that extra vector work.
    """
    cache = model.cos_sin
    if not isinstance(cache, torch.Tensor) or cache.ndim < 2:
        raise ValueError("Expected native Wan cos_sin with original token axis 0")
    cache_rows = cache.index_select(0, token_ids.to(cache.device))
    # Keep the caller's unrotated tensor intact, including backends using in-place updates.
    result, _ = model.apply_rope_func(tensor.clone(), tensor.clone(), cache_rows)
    return result


@dataclass
class WanGlobalState:
    k: torch.Tensor
    v: torch.Tensor
    text_k: torch.Tensor
    text_v: torch.Tensor


@dataclass(frozen=True)
class ProjectedAttentionRows:
    """Explicit post-W_o values [rows, model_width], before gate/residual.

    Only the high folded reader emits this type. Native P64 attention remains
    an ordinary [P,H,D] tensor and therefore uses the original W_o operation.
    """
    value: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.value, torch.Tensor) or self.value.ndim != 2:
            raise ValueError("projected attention must have shape [rows, model_width]")


class WanBlockOperators:
    """Connect the generic schedule to actual native Wan phase interfaces.

    attention.prepare(qP,k,v,probes) -> (fit, native_P [P,H,D])
    attention.read(q,k,v,ids,fit) -> restored [Q,H,D] or ProjectedAttentionRows
    optional attention.bind_output_projection(self_phase) binds native W_o.
    ffn.fit(uP) -> object with observed [P,model_width]
    ffn.apply_rows(ids,u,fit,overwrite_probes=False) -> [Q,model_width]

    The adapter retains full self K/V and computes text K/V once. Norm1 is
    recomputed for the later Q pass to avoid retaining full norm1 or Q. This
    extra normalization and fallback paired RoPE work belong in GPU timing.
    """

    def __init__(self, model, block, pre_infer_out, attention, ffn, *,
                 tile_rows=8192, rope_rows=None):
        self.model, self.block, self.pre = model, block, pre_infer_out
        self.attention, self.ffn = attention, ffn
        self.tile_rows, self.rope_rows = int(tile_rows), rope_rows or native_rope_rows
        if self.tile_rows <= 0:
            raise ValueError("tile_rows must be positive")
        self._validate_supported_graph()
        self.self_phase, self.cross_phase, self.ffn_phase = block.compute_phases[:3]
        bind_projection = getattr(self.attention, "bind_output_projection", None)
        if bind_projection is not None:
            bind_projection(self.self_phase)
        self.modulation = model.pre_process(self.self_phase.modulation, pre_infer_out.embed0)
        if len(self.modulation) != 6:
            raise ValueError("native pre_process must produce six modulation fields")
        self._cu = {}
        self.stats = {"self_kv_passes": 0, "text_kv_passes": 0, "query_rows": 0,
                      "max_query_rows": 0, "all_row_query_tensor": False}
        self.n = self.width = None

    def _validate_supported_graph(self):
        model, block = self.model, self.block
        if int(model.scheduler.step_index) not in (1, 3):
            raise ValueError("whole-block streaming is a hit-step 1/3 path")
        if model.config.get("seq_parallel", False) or model.config.get("use_image_encoder", True):
            raise ValueError("streamed binding requires the frozen nonparallel, no-image-encoder graph")
        if getattr(model, "has_post_adapter", False):
            raise ValueError("post-adapter graph has not been bound to row streaming")
        if getattr(model, "_use_mxfp8_quant_fuse", lambda: False)():
            raise ValueError("MXFP8 fused graph requires a separate streaming binding")
        if len(block.compute_phases) != 3:
            raise ValueError("expected the frozen three-phase Wan block")
        first, last = block.compute_phases[0], block.compute_phases[2]
        before = getattr(first, "before_proj", None)
        if before is not None and getattr(before, "weight", None) is not None:
            raise ValueError("before_proj adapter is outside the frozen graph")
        if hasattr(last, "after_proj"):
            raise ValueError("after_proj adapter is outside the frozen graph")

    def _field(self, index, ids):
        return select_modulation(self.modulation[index], ids, self.n, self.width)

    def _norm_modulate(self, phase, norm_name, rows, ids, scale_index, shift_index):
        model = self.model
        value = getattr(phase, norm_name).apply(rows)
        if model.sensitive_layer_dtype != model.infer_dtype:
            value = value.to(model.sensitive_layer_dtype)
        scale, shift = self._field(scale_index, ids), self._field(shift_index, ids)
        smooth_name = "smooth_" + norm_name + "_weight"
        if hasattr(phase, smooth_name):
            weight = (1 + scale) * getattr(phase, smooth_name).tensor
            bias = shift * getattr(phase, "smooth_" + norm_name + "_bias").tensor
            value.mul_(weight).add_(bias)
        else:
            # The native fused op expects [B,1,C] or [B,L,C] modulation.
            # Gates and smooth-norm weights still use the trailing-width layout.
            scale = scale.reshape(1, -1, self.width)
            shift = shift.reshape(1, -1, self.width)
            value = model.modulate_func(value, scale=scale, shift=shift).reshape(len(ids), self.width)
        if model.sensitive_layer_dtype != model.infer_dtype:
            value = value.to(model.infer_dtype)
        return value

    def _shape_heads(self, tensor):
        return tensor.view(-1, self.model.num_heads, self.model.head_dim)

    def _query(self, x, ids):
        phase = self.self_phase
        u = self._norm_modulate(phase, "norm1", x.index_select(0, ids), ids, 1, 0)
        q = self._shape_heads(phase.self_attn_norm_q.apply(phase.self_attn_q.apply(u)))
        q = self.rope_rows(self.model, q, ids)
        self.stats["query_rows"] += len(ids)
        self.stats["max_query_rows"] = max(self.stats["max_query_rows"], len(ids))
        self.stats["all_row_query_tensor"] |= len(ids) == self.n
        return q

    def prepare_global(self, x):
        self.n, self.width = x.shape
        model, phase = self.model, self.self_phase
        n, heads, dim = self.n, model.num_heads, model.head_dim
        if model.cos_sin.shape[0] != n:
            raise ValueError("global RoPE token extent differs from the current block")
        if heads * dim != self.width:
            raise ValueError("model head shape does not match hidden width")
        k = torch.empty((n, heads, dim), dtype=model.infer_dtype, device=x.device)
        v = torch.empty_like(k)
        for start in range(0, n, self.tile_rows):
            ids = torch.arange(start, min(start + self.tile_rows, n), device=x.device)
            u = self._norm_modulate(phase, "norm1", x[start:start + len(ids)], ids, 1, 0)
            keys = self._shape_heads(phase.self_attn_norm_k.apply(phase.self_attn_k.apply(u)))
            values = self._shape_heads(phase.self_attn_v.apply(u))
            k[start:start + len(ids)].copy_(self.rope_rows(model, keys, ids))
            v[start:start + len(ids)].copy_(values)
        self.stats["self_kv_passes"] += 1
        phase, context = self.cross_phase, self.pre.context
        if model.sensitive_layer_dtype != model.infer_dtype:
            context = context.to(model.infer_dtype)
        text_k = self._shape_heads(phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)))
        text_v = self._shape_heads(phase.cross_attn_v.apply(context))
        self.stats["text_kv_passes"] += 1
        return WanGlobalState(k, v, text_k, text_v)

    def prepare_attention(self, x, ids, global_state):
        q = self._query(x, ids)
        return self.attention.prepare(q, global_state.k, global_state.v, ids)

    def attention_rows(self, x, ids, global_state, fitted):
        q = self._query(x, ids)
        return self.attention.read(q, global_state.k, global_state.v, ids, fitted)

    def _lengths(self, length, template):
        key = (int(length), template.device)
        if key not in self._cu:
            self._cu[key] = torch.tensor([0, length], dtype=torch.int32, device=template.device)
        return self._cu[key]

    def tail_to_ffn(self, x, ids, attention, global_state):
        model, phase = self.model, self.cross_phase
        if isinstance(attention, ProjectedAttentionRows):
            y = attention.value
            if y.shape != (len(ids), self.width) or y.device != x.device:
                raise ValueError("projected attention rows differ from current tile shape/device")
            if y.dtype != model.infer_dtype:
                raise ValueError("projected attention must use the native inference output dtype")
        else:
            y = self.self_phase.self_attn_o.apply(attention.reshape(len(ids), self.width))
        residual = x.index_select(0, ids)  # private: native cross/FFN tails mutate their residual
        gate = self._field(2, ids)
        if model.sensitive_layer_dtype != model.infer_dtype:
            residual = residual.to(model.sensitive_layer_dtype) + y.to(model.sensitive_layer_dtype) * gate
        else:
            residual.add_(y * gate)
        norm3 = phase.norm3.apply(residual)
        q = self._shape_heads(phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3)))
        attention = phase.cross_attn_1.apply(
            q=q, k=global_state.text_k, v=global_state.text_v,
            cu_seqlens_q=self._lengths(len(ids), model.cross_attn_cu_seqlens_q),
            cu_seqlens_kv=self._lengths(len(global_state.text_k), model.cross_attn_cu_seqlens_kv),
            max_seqlen_q=len(ids), max_seqlen_kv=len(global_state.text_k))
        residual.add_(phase.cross_attn_o.apply(attention.reshape(len(ids), self.width)))
        u = self._norm_modulate(self.ffn_phase, "norm2", residual, ids, 4, 3)
        return residual, u

    def fit_ffn(self, u_p):
        return self.ffn.fit(u_p)

    def ffn_rows(self, ids, u, fitted):
        return self.ffn.apply_rows(ids, u, fitted, overwrite_probes=False)

    def finish(self, ids, residual, y):
        gate = self._field(5, ids)
        if self.model.sensitive_layer_dtype != self.model.infer_dtype:
            return residual.to(self.model.sensitive_layer_dtype) + y.to(self.model.sensitive_layer_dtype) * gate
        residual.add_(y * gate)
        return residual


def streamed_wan_hit_block(model, block, x, pre_infer_out, attention, ffn,
                          probes, *, tile_rows=8192, rope_rows=None, stats=None,
                          row_plan=None):
    """Convenience entry point called by the outer anchor/hit dispatcher."""
    operators = WanBlockOperators(model, block, pre_infer_out, attention, ffn,
                                  tile_rows=tile_rows, rope_rows=rope_rows)
    return stream_hit_block(x, probes, operators, tile_rows=tile_rows, stats=stats,
                            row_plan=row_plan)
