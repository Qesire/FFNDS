"""P-first block scheduling and deferred normalization consumers.

The scheduler changes row scheduling, not the algebraic target. GEMM shapes and
floating-point reduction paths change, so this is a real-equivalent graph
rewrite rather than a production bitwise certificate.
"""
from dataclasses import dataclass
from typing import Protocol

import torch


class BlockOperators(Protocol):
    """Raw attention outputs use token-first [rows, heads, head_dim].

    A concrete consumer may additionally accept an explicit already-projected
    type; the scheduler passes it through without assuming or rebuilding axes.

    Every ``ids`` argument contains ORIGINAL global row indices. The immutable
    input ``x`` must remain readable throughout the call. No method may retain
    an all-row Q or reconstructed all-row attention/FFN intermediate.
    """

    def prepare_global(self, x): ...
    def prepare_attention(self, x, ids, global_state): ...
    def attention_rows(self, x, ids, global_state, fitted): ...
    def tail_to_ffn(self, x, ids, attention, global_state): ...
    def fit_ffn(self, u_p): ...
    def ffn_rows(self, ids, u, fitted): ...
    def finish(self, ids, residual, y): ...


@dataclass
class BlockScheduleStats:
    total_rows: int = 0
    probe_rows: int = 0
    nonprobe_tiles: int = 0
    maximum_nonprobe_tile_rows: int = 0
    probe_attention_chains: int = 0
    probe_ffn_fits: int = 0
    materialized_full_q: bool = False
    precision_class: str = "real_equivalent_row_schedule"


def _valid_probes(probes, n, device):
    if probes.ndim != 1 or probes.dtype != torch.long or probes.device != device:
        raise ValueError("probes must be a one-dimensional int64 tensor on x.device")
    if not 0 < probes.numel() <= n:
        raise ValueError("probe count must be between one and the number of rows")
    ordered = probes.sort().values
    torch._assert_async(((ordered >= 0) & (ordered < n)).all(), "probe index out of bounds")
    torch._assert_async((ordered[1:] != ordered[:-1]).all(), "duplicate probe index")


def _tensor_version(value):
    # Tensors created under inference_mode deliberately have no version counter.
    try:
        return value._version
    except RuntimeError:
        return None


@dataclass(frozen=True)
class RowPlan:
    """A one-time-validated partition of original non-P IDs into stable tiles.

    Reuse binds the same probe tensor object. Versioned tensors are checked for
    in-place mutation using host metadata; inference-mode ID tensors are owned
    by this immutable plan and must remain read-only. The plan retains its tile
    objects so downstream SelectedLayout instances can use tensor identity.
    """
    total_rows: int
    probes: torch.Tensor
    tile_rows: int
    tiles: tuple

    def __post_init__(self):
        if type(self.total_rows) is not int or self.total_rows < 1:
            raise ValueError("row plan requires a positive total row count")
        if type(self.tile_rows) is not int or self.tile_rows < 1:
            raise ValueError("row plan requires a positive tile size")
        _valid_probes(self.probes, self.total_rows, self.probes.device)
        tiles = tuple(self.tiles)
        for ids in tiles:
            if ids.ndim != 1 or ids.dtype != torch.long or ids.device != self.probes.device:
                raise ValueError("row-plan tiles must be global int64 IDs on the probe device")
            if not 0 < len(ids) <= self.tile_rows:
                raise ValueError("row-plan tile extent is invalid")
        if sum(len(ids) for ids in tiles) != self.total_rows - len(self.probes):
            raise ValueError("row-plan nonprobe count differs from the geometry")
        joined = torch.cat((self.probes, *tiles))
        expected = torch.arange(self.total_rows, device=self.probes.device)
        torch._assert_async((joined.sort().values == expected).all(),
                            "row-plan IDs must cover each original row exactly once")
        object.__setattr__(self, "tiles", tiles)
        object.__setattr__(self, "_probe_version", _tensor_version(self.probes))
        object.__setattr__(self, "_tile_versions", tuple(_tensor_version(ids) for ids in tiles))

    def check_reuse(self, x, probes, tile_rows):
        """Shape/identity/version metadata only: no device-to-host scalar reads."""
        if len(x) != self.total_rows or tile_rows != self.tile_rows or x.device != self.probes.device:
            raise ValueError("row-plan geometry/device differs from the block")
        if probes is not self.probes:
            raise ValueError("row plan must reuse the original probe tensor object")
        if _tensor_version(probes) != self._probe_version:
            raise ValueError("row-plan probe tensor was modified in-place")
        if any(_tensor_version(ids) != version for ids, version in zip(self.tiles, self._tile_versions)):
            raise ValueError("row-plan tile IDs were modified in-place")


def make_row_plan(total_rows, probes, tile_rows=8192):
    """Prepare original non-P tile tensors once for repeated layers/requests."""
    if type(total_rows) is not int or total_rows < 1 or type(tile_rows) is not int or tile_rows < 1:
        raise ValueError("row count and tile size must be positive integers")
    _valid_probes(probes, total_rows, probes.device)
    keep = torch.ones(total_rows, dtype=torch.bool, device=probes.device)
    keep[probes] = False
    nonprobes = keep.nonzero(as_tuple=False).flatten()
    tiles = tuple(nonprobes.split(tile_rows)) if len(nonprobes) else ()
    return RowPlan(total_rows, probes, tile_rows, tiles)


@torch.no_grad()
def stream_hit_block(x, probes, operators: BlockOperators, *, tile_rows=8192,
                     out=None, stats=None, row_plan=None):
    """Evaluate exact P rows first, then complete each non-P row through the block.

    ``operators.prepare_attention`` returns ``(fit, native_attention_P)``.
    ``operators.fit_ffn`` returns a prepared reader with ``.observed`` holding
    the native FFN output in original probe order. Thus the probe chain needs
    neither the non-probe attention fit nor non-probe FFN intermediates.

    Full K/V and the layer output remain live. The layer output is returned
    only after all row tiles finish, preserving the next layer's K/V barrier.
    """
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must contain nonempty token rows [N,D]")
    if type(tile_rows) is not int or tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    if row_plan is None:
        row_plan = make_row_plan(len(x), probes, tile_rows)
    elif not isinstance(row_plan, RowPlan):
        raise TypeError("row_plan must be a RowPlan made for this geometry")
    row_plan.check_reuse(x, probes, tile_rows)
    if out is not None:
        if out.shape != x.shape or out.device != x.device:
            raise ValueError("out shape/device differs from x")
        # Reject any shared storage, including differently offset views.
        if out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
            raise ValueError("streamed output must not alias the immutable block input")
    stats = stats if stats is not None else BlockScheduleStats()
    stats.total_rows, stats.probe_rows = len(x), len(probes)
    stats.nonprobe_tiles = stats.maximum_nonprobe_tile_rows = 0
    stats.probe_attention_chains = stats.probe_ffn_fits = 0

    global_state = operators.prepare_global(x)
    attention_fit, attention_p = operators.prepare_attention(x, probes, global_state)
    stats.probe_attention_chains += 1
    residual_p, u_p = operators.tail_to_ffn(x, probes, attention_p, global_state)
    ffn_fit = operators.fit_ffn(u_p)
    stats.probe_ffn_fits += 1
    observed = ffn_fit.observed
    if observed.shape != residual_p.shape:
        raise ValueError("FFN fit.observed must match probe residual rows")
    probe_output = operators.finish(probes, residual_p, observed)
    if probe_output.shape != (len(probes), x.shape[1]):
        raise ValueError("probe block output has an unexpected shape")
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=probe_output.dtype)
    elif out.dtype != probe_output.dtype:
        raise ValueError("out dtype differs from the native residual/gate result")
    out.index_copy_(0, probes, probe_output)
    del residual_p, u_p, attention_p, probe_output

    for ids in row_plan.tiles:
        attention = operators.attention_rows(x, ids, global_state, attention_fit)
        residual, u = operators.tail_to_ffn(x, ids, attention, global_state)
        y = operators.ffn_rows(ids, u, ffn_fit)
        result = operators.finish(ids, residual, y)
        if result.shape != (len(ids), x.shape[1]) or result.dtype != out.dtype:
            raise ValueError("row tile output differs in shape or dtype")
        out.index_copy_(0, ids, result)
        stats.nonprobe_tiles += 1
        stats.maximum_nonprobe_tile_rows = max(stats.maximum_nonprobe_tile_rows, len(ids))
        del attention, residual, u, y, result
    stats.materialized_full_q = bool(getattr(operators, "stats", {}).get("all_row_query_tensor", False))
    return out


class DeferredLayerNormLinear:
    """Prepare LN(B + F U), optional affine, then a linear output consumer.

    Uses row moments and small contractions without materializing F U or
    B + F U. This utility is explicit opt-in: covariance/cross-moment work can
    exceed the saved writes at some shapes. It makes no latency claim.
    Matrices use row convention: U [r,d], W [d,out], F [n,r], B [n,d].
    """

    precision_class = "real_equivalent_moment_reduction"

    def __init__(self, basis, weight, *, gamma=None, beta=None, eps=1e-6,
                 accumulator_dtype=None, output_dtype=None):
        if basis.ndim != 2 or weight.ndim != 2 or basis.shape[1] != weight.shape[0]:
            raise ValueError("basis/weight must have shapes [r,d] and [d,out]")
        if basis.device != weight.device or eps <= 0:
            raise ValueError("basis/weight device mismatch or nonpositive eps")
        dtype = accumulator_dtype or (torch.float64 if basis.dtype == torch.float64
                                       or weight.dtype == torch.float64 else torch.float32)
        self.basis, self.weight = basis.to(dtype), weight.to(dtype)
        self.eps, self.output_dtype = float(eps), output_dtype
        self.d = basis.shape[1]
        self.gamma = torch.ones(self.d, dtype=dtype, device=basis.device) if gamma is None else gamma.to(dtype)
        self.beta = torch.zeros(self.d, dtype=dtype, device=basis.device) if beta is None else beta.to(dtype)
        if self.gamma.shape != (self.d,) or self.beta.shape != (self.d,):
            raise ValueError("gamma and beta must be one-dimensional feature parameters")
        if self.gamma.device != basis.device or self.beta.device != basis.device:
            raise ValueError("affine parameter device mismatch")
        self.basis_mean = self.basis.mean(-1)
        self.centered_basis = self.basis - self.basis_mean[:, None]
        self.gram = self.centered_basis @ self.centered_basis.T
        self.basis_readout = (self.basis * self.gamma) @ self.weight
        self.mean_readout = self.gamma @ self.weight
        self.bias_readout = self.beta @ self.weight

    def __call__(self, base, factors):
        if base.ndim != 2 or factors.ndim != 2 or base.shape != (len(factors), self.d):
            raise ValueError("base/factors must be row-aligned [n,d] and [n,r]")
        if factors.shape[1] != len(self.basis):
            raise ValueError("factor width differs from basis rank")
        if base.device != self.basis.device or factors.device != base.device:
            raise ValueError("consumer operands must share a device")
        b, f = base.to(self.basis.dtype), factors.to(self.basis.dtype)
        base_mean = b.mean(-1)
        centered = b - base_mean[:, None]
        mean = base_mean + f @ self.basis_mean
        cross = centered @ self.centered_basis.T
        variance = ((centered * centered).sum(-1) + 2 * (f * cross).sum(-1)
                    + ((f @ self.gram) * f).sum(-1)) / self.d
        inv = torch.rsqrt(variance.clamp_min(0) + self.eps)
        numerator = ((b * self.gamma) @ self.weight + f @ self.basis_readout
                     - mean[:, None] * self.mean_readout)
        result = numerator * inv[:, None] + self.bias_readout
        return result if self.output_dtype is None else result.to(self.output_dtype)


def deferred_layer_norm_linear(base, factors, basis, weight, **kwargs):
    """Convenience form; reuse DeferredLayerNormLinear for a fixed consumer."""
    return DeferredLayerNormLinear(basis, weight, **kwargs)(base, factors)
