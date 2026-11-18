"""BF16 execution arithmetic for the state-conditioned endpoint operators."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Endpoint:
    matrix: torch.Tensor
    projection: torch.Tensor | None = None
    offset: torch.Tensor | None = None
    readout: torch.Tensor | None = None
    q_transpose: torch.Tensor | None = None
    alpha: torch.Tensor | None = None

    def __call__(self, u):
        if u.ndim != 2 or u.dtype != torch.bfloat16 or not u.is_contiguous():
            raise RuntimeError("expected contiguous BF16 pre-W1 rows")
        base = u @ self.matrix
        if self.projection is None:
            return base
        if self.alpha is None:
            s = (u @ self.projection).float() - self.offset
            h = (0.5 * s.square()).to(u.dtype)
        else:
            z = torch.addmm(self.offset, u, self.projection)
            h = (F.gelu(z, approximate="tanh").float() - self.alpha * z.float()).to(u.dtype)
        narrow = h @ self.readout
        for start in range(0, len(u), 1024):
            rows = slice(start, start + 1024)
            branch = narrow[rows] @ self.q_transpose
            base[rows] = (base[rows].float() + branch.float()).to(u.dtype)
        return base


def compile_endpoint(arm, p, a_q1_math, device, *, w1=None, b1=None):
    """Fold the fit's linear terms in FP64, then cast each deployed operand."""
    if arm not in ("L0", "L+", "Q32", "N32"):
        raise ValueError(f"unknown arm {arm}")
    q = p["Q32"].double()
    p_l = p["P64"].double() / p["gamma_L"].double()
    delta = (p_l @ p["C_L"].double()) @ q.T
    matrix = a_q1_math.double() + delta
    if arm == "L+":
        read_extra = p["C_Lplus"].double() / p["gamma_Lplus"].double()[:, None]
        matrix.add_((p["P_extra64"].double() @ read_extra) @ q.T)
    projection = offset = readout = alpha = None
    if arm in ("Q32", "N32"):
        read = p["C_" + arm].double() / p["gamma_" + arm].double()[:, None]
        standardized = p["P64"][:, :32].double() / p["sigma"].double()
        matrix.add_((standardized @ read[:32]) @ q.T)
        readout = read[32:]
        if arm == "Q32":
            projection = standardized
            offset = p["b"].double() / p["sigma"].double()
        else:
            if w1 is None or b1 is None:
                raise ValueError("N32 requires the native W1 and bias")
            indices = p["hidden_indices"].to(w1.device)
            projection = w1.index_select(0, indices).T
            offset = b1.index_select(0, indices)
            alpha = p["hidden_alpha"]

    def operand(value, dtype=torch.bfloat16):
        if value is None:
            return None
        if not torch.isfinite(value).all().item():
            raise RuntimeError("non-finite endpoint parameter")
        cast = value.to(device=device, dtype=dtype).contiguous()
        if not torch.isfinite(cast).all().item():
            raise RuntimeError("endpoint parameter overflows runtime dtype")
        return cast

    # The accepted L64 export casts folded FP64 matrices via FP32 before BF16.
    return Endpoint(operand(matrix.float()), operand(projection), operand(offset, torch.float32 if alpha is None else torch.bfloat16),
                    operand(readout), operand(q.T) if projection is not None else None,
                    operand(alpha, torch.float32))


def residual(anchor_native, endpoint_anchor):
    result = torch.empty_like(anchor_native)
    for start in range(0, len(result), 1024):
        rows = slice(start, start + 1024)
        result[rows] = (anchor_native[rows].float() - endpoint_anchor[rows].float()).to(result.dtype)
    return result


def replay_hit(endpoint_current, cache):
    result = torch.empty_like(endpoint_current)
    for start in range(0, len(result), 1024):
        rows = slice(start, start + 1024)
        result[rows] = (endpoint_current[rows].float() + cache[rows].float()).to(result.dtype)
    return result


def quantization_receipt(native_anchor, b_anchor, b_hit, cache, current_gate):
    """Measure hit and zero-delta rounding relative to the same deployed end points."""
    exact_cache = native_anchor.double() - b_anchor.double()
    zero = replay_hit(b_anchor, cache)
    hit = replay_hit(b_hit, cache)
    exact_hit = b_hit.double() + exact_cache
    cache_round = (cache.double() - exact_cache).abs()
    roundoff = 2 ** -8 + 2 ** -23
    tiny = 2 ** -133
    cache_bound = roundoff * exact_cache.abs() + tiny
    hit_input = b_hit.double() + cache.double()
    zero_input = b_anchor.double() + cache.double()
    hit_bound = cache_bound + roundoff * hit_input.abs() + tiny
    zero_bound = cache_bound + roundoff * zero_input.abs() + tiny
    signed_error = hit.double() - exact_hit
    error = signed_error.abs()
    zero_error = (zero.double() - native_anchor.double()).abs()
    if (not torch.isfinite(hit).all().item() or not (cache_round <= cache_bound).all().item()
            or not (error <= hit_bound).all().item() or not (zero_error <= zero_bound).all().item()):
        raise RuntimeError("BF16 endpoint replay exceeded its signed rounding envelope")
    return {
        "zero_delta_bitwise_equal": bool(torch.equal(zero.view(torch.int16), native_anchor.view(torch.int16))),
        "zero_delta_max_abs": zero_error.max().item(),
        "hit_rounding_max_abs": error.max().item(),
        "hit_rounding_signed_min": signed_error.min().item(),
        "hit_rounding_signed_max": signed_error.max().item(),
        "gated_hit_rounding_max_abs": (error * current_gate.double().abs()).max().item(),
    }

def endpoint_reference(endpoint, u):
    """Independent program using complete panel matrices and the deployed operation order."""
    base = torch.mm(u, endpoint.matrix)
    if endpoint.projection is None:
        return base
    if endpoint.alpha is None:
        z = torch.mm(u, endpoint.projection).float()
        phi = ((z - endpoint.offset).square() / 2).to(torch.bfloat16)
    else:
        z = torch.addmm(endpoint.offset, u, endpoint.projection)
        gelu = torch.nn.functional.gelu(z, approximate="tanh")
        phi = (gelu.float() - z.float() * endpoint.alpha).to(torch.bfloat16)
    correction = torch.mm(torch.mm(phi, endpoint.readout), endpoint.q_transpose)
    return (base.float() + correction.float()).to(torch.bfloat16)


def bitwise_equal(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    bits = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[left.dtype]
    return torch.equal(left.contiguous().view(bits), right.contiguous().view(bits))


def check_panel(endpoint, u0, u1, dense_anchor, phase_gate, *, q1, original_post, host, x=None):
    """Check 512 real native coordinates; the addend is synthetic unless supplied."""
    if (u0.shape != (512, 5120) or u1.shape != u0.shape or dense_anchor.shape != u0.shape
            or dense_anchor.dtype != torch.bfloat16 or phase_gate.squeeze().shape != (5120,)):
        raise RuntimeError("expected frozen P512 native operands")
    if not all(torch.isfinite(item).all().item() for item in (u0, u1, dense_anchor, phase_gate)):
        raise RuntimeError("non-finite P512 replay inputs")
    b0, b1 = endpoint(u0), endpoint(u1)
    ref0, ref1 = endpoint_reference(endpoint, u0), endpoint_reference(endpoint, u1)
    if not bitwise_equal(b0, ref0) or not bitwise_equal(b1, ref1):
        raise RuntimeError("production endpoint differs from independent same-operator reference")
    cache, ref_cache = residual(dense_anchor, b0), (dense_anchor.float() - ref0.float()).bfloat16()
    hit = replay_hit(b1, cache)
    ref_hit = (ref1.float() + ref_cache.float()).bfloat16()
    if not bitwise_equal(cache, ref_cache) or not bitwise_equal(hit, ref_hit):
        raise RuntimeError("production cache/hit differs from same-operator reference")
    results = quantization_receipt(dense_anchor, b0, b1, cache, phase_gate.squeeze())
    q0, qn = u0 @ q1, u1 @ q1
    parent = (qn.float() + (dense_anchor.float() - q0.float()).bfloat16().float()).bfloat16()
    candidate_parent = replay_hit(qn, residual(dense_anchor, q0))
    if not bitwise_equal(parent, candidate_parent):
        raise RuntimeError("shared cache arithmetic differs from strict Q1")
    current_x = torch.zeros_like(u1) if x is None else x
    if current_x.shape != u1.shape:
        raise RuntimeError("invalid residual writeback replay addend")
    dtype = host.sensitive_layer_dtype if host.sensitive_layer_dtype != host.infer_dtype else current_x.dtype
    expected = current_x.to(dtype) + hit.to(dtype) * phase_gate.squeeze()
    actual = original_post(host, current_x.clone(), hit, phase_gate)
    if not bitwise_equal(actual, expected):
        raise RuntimeError("native post_process differs from one current gate write")
    return {**results, "exact_deployed_order": True, "q1_shared_arithmetic_bitwise_equal": True,
            "native_gate_write_bitwise_equal": True, "native_residual_captured": x is not None}
