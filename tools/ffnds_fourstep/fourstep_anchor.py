"""Write native anchor W2 directly into the existing resident y0 slot.

Usage, around a complete native render:
    stats = {}
    with bind_anchor(port, stats):
        experiment.render(source, group, port)

The reviewed Default MMWeight already calls addmm(..., out=empty_BF16).
This changes only that output destination to y0[layer] and omits y0.copy_(y).
The selected z0 capture, in-place GELU, u0 copy and cache ownership are unchanged.
Native and reviewed fused post functions only read y and modify residual x.

Optional diagnostic flags in stats:
  capture_only=True: validate/log layouts but run the original anchor.
  verify_bits=True: also compute native W2 and compare every BF16 bit; expensive,
                    synchronizing diagnostic, never enabled by default.
Stats contain only JSON-serializable values. No attention or GLS changes.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

import torch


SOURCES = {
    "mm_weight": "5987645f1883e92c08a7e416fbe25be3c0fadec2355ae4732fa1182d9094b64c",
    "resident": "bce0d31d3f3c1c6981e55fbfed7846e993dd74e22176cac43378521efc7b218f",
    "exact_port": "dde8a2b4eff670c4e59c68f9b45ac5ad2139a1218e5221053e4abaeb383e8c43",
    "q1_fused_post": "ababf1c3510faaf528bb74201c258ce27b136de28f86b98b4673f778119ec99f",
}


def _fingerprint(obj):
    return hashlib.sha256(Path(inspect.getfile(obj)).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _source_guard(port, slots, original):
    try:
        import exact_port
        import q1_fused_post
        import resident
        from lightx2v.common.ops.mm.mm_weight import MMWeight, MMWeightTemplate
    except ImportError:
        return "native_modules_unavailable", None, None
    if type(port) is not exact_port.ResidentPort or getattr(port, "mode", None) != "fast512":
        return "unsupported_port_or_mode", None, None
    if type(slots) is not resident.AnchorSlots or getattr(original, "__func__", None) is not resident.AnchorSlots.anchor:
        return "unsupported_anchor_implementation", None, None
    if getattr(port.bind, "__func__", None) is not exact_port.ResidentPort.bind:
        return "modified_post_binding", None, None
    targets = {"mm_weight": MMWeight, "resident": resident.AnchorSlots,
               "exact_port": exact_port.ResidentPort, "q1_fused_post": q1_fused_post.run}
    for name, obj in targets.items():
        try:
            if _fingerprint(obj) != SOURCES[name]:
                return "source_changed_"+name, None, None
        except (OSError, TypeError):
            return "source_unavailable_"+name, None, None
    return None, MMWeight, MMWeightTemplate


def _storage_alias(a, b):
    # Compare allocation identities, not just first-element pointers of views.
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _layout_guard(port, slots, expert, layer, u, phase, selector, mm_type, template):
    if expert not in ("high", "low") or not 0 <= layer < 40:
        return "unsupported_site", None
    if not getattr(port, "busy", False) or not getattr(port, "bound", False):
        return "outside_native_trajectory", None
    if torch.is_grad_enabled() or torch.is_autocast_enabled():
        return "grad_or_autocast_enabled", None
    if (getattr(port, "rows", None), getattr(port, "width", None), getattr(port, "layers", None)) != (31668, 5120, 40):
        return "unsupported_geometry", None
    if tuple(slots.y0.shape) != (40, 31668, 5120) or tuple(slots.u0.shape) != tuple(slots.y0.shape):
        return "unsupported_slot_shape", None
    dst = slots.y0[layer]
    if any(type(t) is not torch.Tensor for t in (u, dst, slots.u0, slots.z0k, selector)):
        return "tensor_subclass", None
    if (u.shape != dst.shape or u.dtype != torch.bfloat16 or dst.dtype != u.dtype
            or u.device.type != "cuda" or dst.device != u.device or slots.u0.device != u.device
            or slots.z0k.device != u.device or not u.is_contiguous() or not dst.is_contiguous()
            or tuple(dst.stride()) != (5120, 1) or dst.data_ptr() % 256):
        return "unsupported_output_layout", None
    if (tuple(selector.shape) != (2560,) or selector.dtype != torch.long
            or selector.device != u.device or tuple(slots.z0k.shape) != (40,31668,2560)
            or slots.u0.dtype != u.dtype or slots.z0k.dtype != u.dtype):
        return "unsupported_selected_cache", None
    mm = phase.ffn_2
    if type(mm) is not mm_type or getattr(mm.apply, "__func__", None) is not mm_type.apply:
        return "nondefault_or_wrapped_w2", None
    if (getattr(mm, "has_lora_branch", False) or getattr(mm, "has_diff", False)
            or hasattr(mm, "weight_diff") or hasattr(mm, "bias_diff")):
        return "lora_or_diff", None
    if (getattr(mm._get_actual_weight, "__func__", None) is not template._get_actual_weight
            or getattr(mm._get_actual_bias, "__func__", None) is not template._get_actual_bias):
        return "modified_weight_accessors", None
    # Resolve current prefetched operands at EVERY anchor; never retain a ring
    # buffer pointer or a phase-object identity across layers/experts.
    weight, bias = mm._get_actual_weight(), mm._get_actual_bias()
    if type(weight) is not torch.Tensor or type(bias) is not torch.Tensor:
        return "missing_native_weight_or_bias", None
    if weight is not mm.weight or bias is not mm.bias:
        return "transformed_weight_or_bias", None
    if (tuple(weight.shape) != (13824, 5120) or tuple(bias.shape) != (5120,)
            or weight.dtype != u.dtype or bias.dtype != u.dtype
            or weight.device != u.device or bias.device != u.device
            or tuple(weight.stride()) not in ((1,13824), (5120,1))
            or not bias.is_contiguous()):
        return "unsupported_w2_layout", None
    if getattr(mm, "create_cuda_buffer", False):
        wb, bb = getattr(mm, "weight_cuda_buffer", None), getattr(mm, "bias_cuda_buffer", None)
        if (not isinstance(wb, torch.Tensor) or not isinstance(bb, torch.Tensor)
                or weight.data_ptr() != wb.data_ptr() or bias.data_ptr() != bb.data_ptr()):
            return "prefetch_buffer_ownership", None
    if any(_storage_alias(dst, t) for t in (u, weight, bias, slots.u0, slots.z0k)):
        return "output_aliases_an_operand", None
    return None, (dst, weight, bias)


@contextmanager
def bind_anchor(port, stats):
    """Patch only this port's slot instance; restore it on normal/exception exit.

Unsupported source/backend/layout executes the original anchor. Default path
does no tensor comparison, device synchronization, or additional CUDA kernel.
"""
    if not isinstance(stats, dict):
        raise TypeError("stats must be a dictionary")
    capture_only = bool(stats.get("capture_only", False))
    verify_bits = bool(stats.get("verify_bits", False))
    for name in ("seen", "eligible", "applied", "fallback", "captured", "verified",
                 "bitwise_mismatches", "copy_bytes_elided"):
        stats.setdefault(name, 0)
    stats.setdefault("fallback_reasons", {})
    stats.setdefault("layout_samples", [])
    stats["capture_only"], stats["verify_bits"] = capture_only, verify_bits
    slots = port.slots
    if "anchor" in slots.__dict__:
        raise RuntimeError("An anchor binding already exists on this slot instance")
    original = slots.anchor
    source_reason, mm_type, template = _source_guard(port, slots, original)
    stats["source_guard_ok"] = source_reason is None
    stats["source_guard_reason"] = source_reason
    disabled_reason = None

    def fallback(reason, expert, layer, u, phase, selector):
        stats["fallback"] += 1
        reasons = stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        return original(expert, layer, u, phase, selector)

    def anchor(expert, layer, u, phase, selector):
        nonlocal disabled_reason
        stats["seen"] += 1
        reason = source_reason or disabled_reason
        if reason is not None:
            return fallback(reason, expert, layer, u, phase, selector)
        if slots.owner[layer] is not None:
            # Preserve the original state-machine error before any work.
            raise RuntimeError("Anchor slot has an unconsumed value")
        reason, operands = _layout_guard(port, slots, expert, layer, u, phase, selector, mm_type, template)
        if reason is not None:
            return fallback(reason, expert, layer, u, phase, selector)
        dst, weight, bias = operands
        stats["eligible"] += 1
        if len(stats["layout_samples"]) < 2:
            stats["layout_samples"].append(dict(expert=expert, layer=layer,
                output_shape=list(dst.shape), output_stride=list(dst.stride()),
                weight_shape=list(weight.shape), weight_stride=list(weight.stride()),
                dtype=str(dst.dtype), device=str(dst.device), output_alignment_mod256=dst.data_ptr()%256))
        if capture_only:
            stats["captured"] += 1
            return original(expert, layer, u, phase, selector)
        # Identical order to AnchorSlots.anchor through GELU; the selected z0
        # columns must be saved before overwriting z with tanh-GELU.
        z = phase.ffn_0.apply(u)
        torch.index_select(z, 1, selector, out=slots.z0k[layer])
        torch.ops.aten.gelu.out(z, approximate="tanh", out=z)
        if (type(z) is not torch.Tensor or z.dtype != u.dtype or z.device != u.device
                or tuple(z.shape) != (31668,13824) or not z.is_contiguous()
                or _storage_alias(dst, z)):
            # W1 already ran: finish the unchanged native path without running
            # it twice. No output destination has been written yet.
            y = phase.ffn_2.apply(z)
            slots.u0[layer].copy_(u)
            dst.copy_(y)
            slots.owner[layer] = expert
            stats["fallback"] += 1
            reasons = stats["fallback_reasons"]
            reasons["unsupported_w1_output"] = reasons.get("unsupported_w1_output", 0)+1
            return y
        native = phase.ffn_2.apply(z) if verify_bits else None
        # This is the exact Default MMWeight.apply arithmetic, using its actual
        # already-transposed W2, same input, bias, dtype, shape and out layout.
        torch.addmm(bias, z, weight, out=dst)
        if verify_bits:
            stats["verified"] += 1
            if not torch.equal(dst.view(torch.int16), native.view(torch.int16)):
                stats["bitwise_mismatches"] += 1
                disabled_reason = "bitwise_mismatch"
                dst.copy_(native)
                slots.u0[layer].copy_(u)
                slots.owner[layer] = expert
                stats["fallback"] += 1
                reasons = stats["fallback_reasons"]
                reasons[disabled_reason] = reasons.get(disabled_reason, 0)+1
                return native
        slots.u0[layer].copy_(u)
        slots.owner[layer] = expert
        stats["applied"] += 1
        stats["copy_bytes_elided"] += dst.numel()*dst.element_size()
        return dst

    slots.anchor = anchor
    stats["binding_active"] = True
    try:
        yield port
    finally:
        del slots.anchor
        stats["binding_active"] = False
