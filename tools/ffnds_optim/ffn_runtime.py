"""Native Wan binding with separately allocated full/projected anchor slots.

This port never constructs the old ResidentJoint. A factor asset therefore
removes full u0 residency and full GPU A0 preparation, not merely their use.
"""
from contextlib import contextmanager
import inspect
import math

import numpy as np
import torch

from .ffn import FFNCache, FFNStaged, _project, prepare_runtime_tensors


class ProjectedAnchorSlots:
    """Reusable, expert-shared storage with a checked anchor->take lifecycle."""
    def __init__(self, layers, rows, width, selected, probes, device, *, rank=None,
                 modes=16, dtype=torch.bfloat16, projection_dtype=torch.bfloat16,
                 preserve_probe_u=True, tile_rows=8192, anchor_direct=True):
        self.layers, self.rows, self.width, self.selected = layers, rows, width, selected
        self.rank, self.tile_rows, self.anchor_direct = rank, tile_rows, anchor_direct
        self.probes = probes
        self.owner = [None] * layers
        self.fields = {"y0": torch.empty((layers, rows, width), dtype=dtype, device=device),
                       "z0k": torch.empty((layers, rows, selected), dtype=dtype, device=device)}
        if rank is None:
            self.fields["u0"] = torch.empty_like(self.fields["y0"])
        else:
            self.fields["a0"] = torch.empty((layers, rows, rank), dtype=projection_dtype, device=device)
            self.fields["j0"] = torch.empty((layers, rows, modes), dtype=torch.float32, device=device)
            if preserve_probe_u:
                self.fields["u0p"] = torch.empty((layers, len(probes), width), dtype=dtype, device=device)
        self.bytes = sum(t.numel() * t.element_size() for t in self.fields.values())
        self.stats = {"anchor_direct": 0, "anchor_fallback": 0, "copy_bytes_elided": 0,
                      "anchor_fallback_reasons": {}}

    def _direct_operands(self, mm, z, dst):
        """Use the same approved Default MMWeight source, never a generic bypass."""
        if not self.anchor_direct:
            return None, "disabled"
        if z.device.type != "cuda" or z.dtype != torch.bfloat16:
            return None, "native_bf16_cuda_required"
        try:
            from fourstep_anchor import SOURCES, _fingerprint
            from lightx2v.common.ops.mm.mm_weight import MMWeight, MMWeightTemplate
        except ImportError:
            return None, "native_modules_unavailable"
        if (type(mm) is not MMWeight or getattr(mm.apply, "__func__", None) is not MMWeight.apply
                or _fingerprint(MMWeight) != SOURCES["mm_weight"]):
            return None, "default_mm_source_or_apply_changed"
        if (getattr(mm, "has_lora_branch", False) or getattr(mm, "has_diff", False)
                or hasattr(mm, "weight_diff") or hasattr(mm, "bias_diff")):
            return None, "lora_or_diff"
        if (getattr(mm._get_actual_weight, "__func__", None) is not MMWeightTemplate._get_actual_weight
                or getattr(mm._get_actual_bias, "__func__", None) is not MMWeightTemplate._get_actual_bias):
            return None, "modified_weight_accessors"
        weight, bias = mm._get_actual_weight(), mm._get_actual_bias()
        if (type(weight) is not torch.Tensor or type(bias) is not torch.Tensor
                or weight is not mm.weight or bias is not mm.bias
                or weight.shape != (z.shape[1], dst.shape[1]) or bias.shape != (dst.shape[1],)
                or any(t.device != dst.device or t.dtype != dst.dtype for t in (weight, bias, z))
                or not z.is_contiguous() or not dst.is_contiguous() or not bias.is_contiguous()
                or weight.stride() not in ((1, weight.shape[0]), (weight.shape[1], 1))):
            return None, "unsupported_native_operands"
        if getattr(mm, "create_cuda_buffer", False):
            wb, bb = getattr(mm, "weight_cuda_buffer", None), getattr(mm, "bias_cuda_buffer", None)
            if (not isinstance(wb, torch.Tensor) or not isinstance(bb, torch.Tensor)
                    or weight.data_ptr() != wb.data_ptr() or bias.data_ptr() != bb.data_ptr()):
                return None, "prefetch_buffer_ownership"
        if any(dst.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in (z, weight, bias)):
            return None, "output_alias"
        return (weight, bias), None

    @torch.inference_mode()
    def anchor(self, expert, layer, u, phase, selector, prepared):
        if self.owner[layer] is not None:
            raise RuntimeError("unconsumed anchor slot")
        if u.shape != (self.rows, self.width) or u.dtype != self.fields["y0"].dtype:
            raise ValueError("anchor input axes/dtype differ")
        factored = "factor_left" in prepared
        if factored != (self.rank is not None) or (factored and prepared["factor_left"].shape[1] != self.rank):
            raise ValueError("all resident sites must share the selected cache layout")
        z = phase.ffn_0.apply(u)
        torch.index_select(z, 1, selector, out=self.fields["z0k"][layer])
        torch.ops.aten.gelu.out(z, approximate="tanh", out=z)
        dst = self.fields["y0"][layer]
        operands, reason = self._direct_operands(phase.ffn_2, z, dst)
        if operands is not None:
            weight, bias = operands
            torch.addmm(bias, z, weight, out=dst)
            y = dst
            self.stats["anchor_direct"] += 1
            self.stats["copy_bytes_elided"] += dst.numel() * dst.element_size()
        else:
            y = phase.ffn_2.apply(z)
            dst.copy_(y)
            self.stats["anchor_fallback"] += 1
            reasons = self.stats["anchor_fallback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
        del z
        if not factored:
            self.fields["u0"][layer].copy_(u)
        else:
            if "u0p" in self.fields:
                torch.index_select(u, 0, self.probes, out=self.fields["u0p"][layer])
            for start in range(0, self.rows, self.tile_rows):
                sl = slice(start, min(start + self.tile_rows, self.rows))
                dst_a = self.fields["a0"][layer, sl]
                dst_a.copy_(_project(u[sl], prepared["factor_left"], dst_a.dtype))
                self.fields["j0"][layer, sl].copy_(u[sl].float() @ prepared["jmap"])
        self.owner[layer] = expert
        return y

    def take(self, expert, layer):
        if self.owner[layer] != expert:
            raise RuntimeError("hit has no matching expert anchor")
        self.owner[layer] = None
        return FFNCache({key: value[layer] for key, value in self.fields.items()},
                         self.probes, self.rows, "full" if self.rank is None else "projected")


class OptimizationFFNPort:
    """Serialized HHLL port, also exposing a staged hit to the block scheduler."""
    variant = "wide8192_optim"

    def __init__(self, site_loader, observer_ids, *, device="cuda:0", grid=(21, 29, 52),
                 width=5120, selected=2560, layers=40, tile_rows=8192,
                 backend="native", factor_loader=None, gauge_loader=None,
                 projection_dtype=torch.bfloat16, preserve_probe_u=True,
                 inv_metric=None, anchor_direct=True):
        self.site_loader, self.factor_loader, self.gauge_loader = site_loader, factor_loader, gauge_loader
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.rows, self.width, self.selected, self.layers = math.prod(grid), width, selected, layers
        self.tile_rows, self.backend = tile_rows, backend
        self.projection_dtype, self.preserve_probe_u, self.anchor_direct = projection_dtype, preserve_probe_u, anchor_direct
        self.probes = torch.as_tensor(observer_ids, dtype=torch.long, device=self.device)
        if (self.probes.ndim != 1 or len(self.probes) != len(torch.unique(self.probes))
                or bool(((self.probes < 0) | (self.probes >= self.rows)).any())):
            raise ValueError("invalid unique original probe rows")
        if inv_metric is None:
            from native_joint import probe_metric_inverse
            ids = self.probes.cpu().numpy()
            coords = np.column_stack(np.unravel_index(ids, grid))
            inv_metric, _ = probe_metric_inverse(ids, coords, self.device)
        self.inv_metric = inv_metric.to(self.device)
        self.weights, self.selectors = {}, {}
        self.slots = None
        self.inner = self
        self.busy, self.bound, self.completion = False, False, None
        self.execution_stream = None
        self.counters = {}

    def _position(self, model):
        if not self.busy:
            raise RuntimeError("open trajectory before FFN execution")
        step, layer = int(model.scheduler.step_index), int(model.block_idx)
        if (step, layer) != divmod(self.counters["sites"], self.layers) or step >= 4:
            raise RuntimeError("out-of-order FFN site")
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            if self.execution_stream is not None and stream != self.execution_stream:
                stream.wait_stream(self.execution_stream)
            self.execution_stream = stream
        return step, layer, "high" if step < 2 else "low"

    def _prepare(self, expert, layer, phase):
        key = expert, layer
        if key not in self.weights:
            pack = self.site_loader(expert, layer)
            factor = None if self.factor_loader is None else self.factor_loader(expert, layer)
            gauge = None if self.gauge_loader is None else self.gauge_loader(expert, layer)
            if self.factor_loader is not None and factor is None:
                raise ValueError("factor loader must provide every requested site")
            t = prepare_runtime_tensors(phase, pack, self.device, factor_asset=factor,
                                         gauge=gauge, backend=self.backend)
            self.weights[key] = t
            self.selectors[key] = pack["sel"].to(self.device)
            self.counters["new_packed_sites"] += 1
            if self.slots is None:
                rank = t["factor_left"].shape[1] if "factor_left" in t else None
                self.slots = ProjectedAnchorSlots(self.layers, self.rows, self.width, self.selected,
                    self.probes, self.device, rank=rank, modes=t["v"].shape[0],
                    projection_dtype=self.projection_dtype, preserve_probe_u=self.preserve_probe_u,
                    tile_rows=self.tile_rows, anchor_direct=self.anchor_direct)
            self.counters["anchor_gpu_capacity_bytes"] = self.slots.bytes
        self.counters["ffn_factor_rank"] = self.slots.rank
        self.counters["ffn_projection_dtype"] = str(self.projection_dtype) if self.slots.rank is not None else None
        self.counters["ffn_exact_probe_sidecar"] = "u0p" in self.slots.fields
        self.counters["ffn_factor_methods"] = sorted({
            value["factor_method"] for value in self.weights.values() if "factor_method" in value})
        self.counters["ffn_paired_gauge_active"] = any("gauge" in value for value in self.weights.values())
        return self.weights[key]

    def take_hit(self, model, phase):
        step, layer, expert = self._position(model)
        if step not in (1, 3) or self.slots is None:
            raise RuntimeError("staged hit requires a prior anchor")
        key = expert, layer
        cached = self.slots.take(expert, layer)
        staged = FFNStaged(phase, self.weights[key], cached, self.probes,
                           self.inv_metric, backend=self.backend, validate_probes=False)
        self.counters["sites"] += 1
        self.counters["hits"] += 1
        self.counters["staged_hits"] += 1
        return staged

    @torch.inference_mode()
    def __call__(self, model, phase, u):
        step, layer, expert = self._position(model)
        if u.shape != (self.rows, self.width) or u.dtype != torch.bfloat16 or u.device != self.device:
            raise ValueError("native FFN input differs from configured geometry/device/BF16")
        if step % 2:
            return self.take_hit(model, phase).full(u, tile_rows=self.tile_rows)[0]
        t = self._prepare(expert, layer, phase)
        y = self.slots.anchor(expert, layer, u, phase, self.selectors[expert, layer], t)
        self.counters["sites"] += 1
        self.counters["anchors"] += 1
        self.counters.update(self.slots.stats)
        return y

    @contextmanager
    def trajectory(self):
        if self.busy:
            raise RuntimeError("concurrent optimized FFN trajectory")
        if self.device.type == "cuda":
            if torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError("frozen FP32 contractions require TF32 disabled")
            self.execution_stream = torch.cuda.current_stream(self.device)
            if self.completion is not None:
                self.execution_stream.wait_event(self.completion)
        self.busy = True
        self.counters = dict(variant=self.variant, tile_rows=self.tile_rows, sites=0, anchors=0,
                             hits=0, staged_hits=0, new_packed_sites=0, gate_native=0, gate_fused=0,
                             anchor_d2h=0, anchor_h2d=0, factor_cache=self.factor_loader is not None)
        if self.slots is not None:
            self.slots.stats.update(anchor_direct=0, anchor_fallback=0, copy_bytes_elided=0,
                                    anchor_fallback_reasons={})
            self.counters["anchor_gpu_capacity_bytes"] = self.slots.bytes
        try:
            yield self
            if self.counters["sites"] != 4 * self.layers or any(x is not None for x in self.slots.owner):
                raise RuntimeError("incomplete optimized HHLL trajectory")
        finally:
            if self.slots is not None:
                self.slots.owner[:] = [None] * self.layers
            if self.device.type == "cuda":
                self.completion = torch.cuda.Event()
                self.completion.record(self.execution_stream)
            self.execution_stream = None
            self.busy = False

    @contextmanager
    def bind(self, owner, transformer_sha):
        from wan_ffn_port import bind_native_ffn, source_digest
        original = owner.post_process
        if self.bound or source_digest(inspect.getfile(original)) != transformer_sha:
            raise RuntimeError("transformer source or binding changed")

        def post(model, x, y, gate, pre_infer_out=None):
            # Preserve the existing fused BF16 post under its supported contract.
            if (self.backend == "native" and y is not None and gate is not None
                    and model.sensitive_layer_dtype == model.infer_dtype == torch.bfloat16
                    and not model.clean_cuda_cache and self.width == 5120
                    and x.dtype == y.dtype == gate.dtype == torch.bfloat16
                    and x.device == y.device == gate.device == self.device
                    and x.shape == y.shape and gate.squeeze().shape == (self.width,)
                    and x.is_contiguous() and y.is_contiguous() and gate.is_contiguous()):
                from q1_fused_post import run
                run(x, y, None, None, gate, action="post")
                self.counters["gate_fused"] += 1
                return x
            self.counters["gate_native"] += 1
            return original(model, x, y, gate, pre_infer_out)

        self.bound = True
        try:
            with bind_native_ffn(owner, self, exact_source_sha=transformer_sha):
                owner.post_process = post
                try:
                    yield self
                finally:
                    owner.post_process = original
        finally:
            self.bound = False
