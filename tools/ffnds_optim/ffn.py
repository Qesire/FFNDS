"""Staged FFNDS readout and explicitly approximate projected anchor state.

The native backend preserves the frozen difference/LUT/base kernels. ``torch``
is a CPU-capable numerical reference; it is not a replacement CUDA kernel.
Factor and gauge options change finite-precision semantics and are opt-in.
"""
from dataclasses import dataclass
import hashlib
from typing import Mapping

import torch
import torch.nn.functional as F


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _ids(ids, rows, device):
    if isinstance(ids, slice):
        start, stop, step = ids.indices(rows)
        return torch.arange(start, stop, step, device=device)
    return torch.as_tensor(ids, dtype=torch.long, device=device)


@dataclass
class FFNCache:
    fields: dict
    probes: torch.Tensor
    rows: int
    mode: str = "full"

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.fields.values())

    def __getitem__(self, key):
        return self.fields[key]


@dataclass
class FFNFit:
    xp: torch.Tensor
    readout: torch.Tensor
    observed: torch.Tensor
    probes: torch.Tensor
    gain: torch.Tensor
    coefficient: torch.Tensor


def compile_factor_asset(pack, rank, *, gauge=None, dtype=torch.bfloat16):
    """Offline Frobenius SVD initializer; no task-quality claim or hidden rank.

    The input is the actual frozen CPU site pack, before native GPU preparation.
    The native teacher weights are never modified. External recovery-aware
    factors can use the same left/right schema after their own fitting.
    """
    a = pack["A0"].detach().cpu().double()
    if type(rank) is not int or not 1 <= rank <= min(a.shape):
        raise ValueError("factor rank must be explicitly in [1, input width]")
    c = None if gauge is None else torch.as_tensor(gauge, dtype=pack["jmap"].dtype).cpu()
    if c is not None:
        if c.shape != (pack["v"].shape[0],):
            raise ValueError("gauge must have one scalar per existing mode")
        a = a + (pack["jmap"].cpu().double() * c.double()) @ pack["v"].cpu().double()
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    root = s[:rank].sqrt()
    left = (u[:, :rank] * root).to(dtype).contiguous()
    right = (root[:, None] * vh[:rank]).to(dtype).contiguous()
    return dict(left=left, right=right, gauge=c,
                source_a0_sha256=tensor_digest(pack["A0"]),
                source_jmap_sha256=tensor_digest(pack["jmap"]),
                source_v_sha256=tensor_digest(pack["v"]),
                rank=rank, method="weight_frobenius_svd_initializer",
                target="A0" if c is None else "A0+Jdiag(c)V",
                discarded_singular_energy=float(s[rank:].square().sum()),
                total_singular_energy=float(s.square().sum()),
                task_quality="unmeasured", format_version=1)


def apply_gauge(prepared, gauge):
    """Return paired A/B constants; caller retains original native teacher."""
    c = torch.as_tensor(gauge, dtype=prepared["jmap"].dtype,
                        device=prepared["jmap"].device)
    if c.shape != (prepared["v"].shape[0],):
        raise ValueError("invalid gauge dimension")
    result = dict(prepared)
    if "A0" in result:
        a = result["A0"]
        work = torch.float64 if a.dtype == torch.float64 else torch.float32
        result["A0"] = (a.to(work) + (prepared["jmap"].to(work) * c.to(work))
                         @ prepared["v"].to(work)).to(a.dtype).contiguous()
    b = result["w2k"]
    work = torch.float64 if b.dtype == torch.float64 else torch.float32
    result["w2k"] = (b.to(work) - (prepared["zmap"].to(work) * c.to(work))
                      @ prepared["v"].to(work)).to(b.dtype).contiguous()
    result["gauge"] = c
    return result


def prepare_runtime_tensors(phase, pack, device, *, factor_asset=None,
                            gauge=None, backend="native"):
    """Prepare a site without first allocating its full GPU A0 when factored."""
    device = torch.device(device)
    width, width2 = pack["A0"].shape
    sel = pack["sel"].to(device)
    k, modes = pack["zmap"].shape
    if width != width2 or sel.shape != (k,):
        raise ValueError("incompatible frozen A0/selected shape")
    names = ("zmap", "jmap", "v", "kernel_beta", "kernel_scale")
    t = {name: pack[name].to(device).contiguous() for name in names}
    if factor_asset is None:
        t["A0"] = pack["A0"].to(device).contiguous()
    else:
        for name in ("A0", "jmap", "v"):
            identity = factor_asset.get("source_" + name.lower() + "_sha256")
            if identity is None:
                raise ValueError("factor asset is missing source identity for " + name)
            if identity != tensor_digest(pack[name]):
                raise ValueError("factor asset belongs to another " + name)
        left, right = factor_asset["left"], factor_asset["right"]
        if left.ndim != 2 or left.shape[0] != width or right.shape != (left.shape[1], width):
            raise ValueError("invalid factor axes")
        if left.dtype != right.dtype:
            raise ValueError("factor dtypes must match")
        asset_gauge = factor_asset.get("gauge")
        if gauge is not None and (asset_gauge is None or not torch.equal(
                torch.as_tensor(gauge).cpu().to(asset_gauge.dtype), asset_gauge.cpu())):
            raise ValueError("factor gauge and requested paired gauge differ")
        gauge = asset_gauge
        t["factor_left"] = left.to(device).contiguous()
        t["factor_right"] = right.to(device).contiguous()
        t["factor_method"] = factor_asset.get("method", "external_explicit_factors")
    w1, w2 = (p._get_actual_weight() for p in (phase.ffn_0, phase.ffn_2))
    if w1.shape[0] != width or w2.shape != w1.T.shape:
        raise ValueError("native FFN weight axes differ")
    b1 = phase.ffn_0._get_actual_bias().reshape(-1)
    t["w1k"] = w1.T.index_select(0, sel.to(w1.device)).contiguous().to(device)
    t["b1k"] = b1.index_select(0, sel.to(b1.device)).to(device)
    t["w2k"] = w2.index_select(0, sel.to(w2.device)).contiguous().to(device)
    if gauge is not None:
        t = apply_gauge(t, gauge)
    t["vd"] = t["v"].double()
    t["vv"] = t["vd"] @ t["vd"].T
    if backend == "native":
        from exact_port import pointwise
        from lookup import prepare_lookup
        if device.type != "cuda":
            raise ValueError("native FFN kernels require CUDA")
        index = device.index if device.index is not None else torch.cuda.current_device()
        t["difference"], t["base"] = pointwise(index)
        t["lut"] = prepare_lookup(index)
    elif backend != "torch":
        raise ValueError("backend must be native or torch")
    return t


def _project(u, left, storage_dtype):
    """Projection output dtype is explicit, including FP32 coordinate option."""
    value = u.to(left.dtype)
    if storage_dtype == torch.float32 and left.dtype in (torch.float16, torch.bfloat16):
        if u.device.type == "cuda":
            return torch.mm(value, left, out_dtype=torch.float32)
        return value.float() @ left.float()
    return (value @ left).to(storage_dtype)


def encode_anchor(u, y, z_selected, prepared, probes, *, projection_dtype=torch.bfloat16,
                  preserve_probe_u=True, tile_rows=8192):
    """Owned cache tensors: a factored cache never retains a full u view."""
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    probes = _ids(probes, len(u), u.device)
    fields = {"y0": y.detach().clone(), "z0k": z_selected.detach().clone()}
    if "factor_left" not in prepared:
        fields["u0"] = u.detach().clone()
        return FFNCache(fields, probes, len(u), "full")
    left = prepared["factor_left"]
    fields["a0"] = torch.empty((len(u), left.shape[1]), dtype=projection_dtype, device=u.device)
    fields["j0"] = torch.empty((len(u), prepared["jmap"].shape[1]),
                                dtype=prepared["jmap"].dtype, device=u.device)
    if preserve_probe_u:
        fields["u0p"] = u.index_select(0, probes).detach().clone()
    for start in range(0, len(u), tile_rows):
        sl = slice(start, start + tile_rows)
        fields["a0"][sl].copy_(_project(u[sl], left, projection_dtype))
        fields["j0"][sl].copy_(u[sl].to(prepared["jmap"].dtype) @ prepared["jmap"])
    return FFNCache(fields, probes, len(u), "projected")


class FFNStaged:
    """Fit from only current P rows; read arbitrary original token subsets."""
    def __init__(self, phase, prepared, cache, probes, inv_metric, *, backend="native", validate_probes=True):
        self.phase, self.t, self.backend = phase, prepared, backend
        if backend not in ("native", "torch"):
            raise ValueError("backend must be native or torch")
        if isinstance(cache, Mapping):
            cache = FFNCache(dict(cache), probes, len(cache["y0"]),
                             "projected" if "a0" in cache else "full")
        self.cache = cache
        self.probes = _ids(probes, cache.rows, cache["y0"].device)
        if self.probes.ndim != 1 or (validate_probes and len(self.probes) != len(torch.unique(self.probes))):
            raise ValueError("probes must be unique original row IDs")
        same_probes = cache.probes is self.probes
        if not same_probes and not torch.equal(cache.probes.to(self.probes.device), self.probes):
            raise ValueError("probe order differs from anchor sidecar")
        self.inv_metric = inv_metric
        if inv_metric.shape != (len(self.probes), len(self.probes)):
            raise ValueError("probe metric axes differ")
        self.k = prepared["zmap"].shape[0]

    def _selected_mm(self, dh):
        w = self.t["w2k"]
        if dh.dtype == torch.float64:
            return dh @ w
        if self.backend == "native":
            return torch.mm(dh, w, out_dtype=torch.float32)
        return dh.float() @ w.float()

    def operands(self, token_ids, now, *, probe_fit=False):
        is_slice = isinstance(token_ids, slice)
        ids = None if is_slice else _ids(token_ids, self.cache.rows, now.device)
        count = len(range(*token_ids.indices(self.cache.rows))) if is_slice else len(ids)
        if now.ndim != 2 or len(now) != count:
            raise ValueError("current rows and original token IDs differ")
        t, cache = self.t, self.cache
        pick = lambda value: value[token_ids] if isinstance(token_ids, slice) else value.index_select(0, ids)
        real = now.dtype == torch.float64
        work = torch.float64 if real else torch.float32
        if cache.mode == "full":
            previous = pick(cache["u0"])
            if self.backend == "native":
                from exact_port import apply
                du, dub = torch.empty_like(now, dtype=torch.float32), torch.empty_like(now)
                apply(t["difference"], du.numel(), now, previous, dub, du, du.numel())
            else:
                du = now.to(work) - previous.to(work)
                dub = du.to(now.dtype)
            linear = dub @ t["A0"]
            ju = du @ t["jmap"]
        else:
            left, right = t["factor_left"], t["factor_right"]
            previous_a = pick(cache["a0"])
            current_a = _project(now, left, previous_a.dtype)
            delta_a = current_a.to(work) - previous_a.to(work)
            linear = delta_a.to(right.dtype) @ right
            if probe_fit and "u0p" in cache.fields:
                # This call is exactly the original P order; preserve the old Q_P path.
                if ids is not self.probes and (ids is None or not torch.equal(ids, self.probes)):
                    raise ValueError("probe fit rows must use original P order")
                ju = (now.to(work) - cache["u0p"].to(work)) @ t["jmap"]
            else:
                ju = now.to(t["jmap"].dtype) @ t["jmap"] - pick(cache["j0"])
        z = F.linear(now, t["w1k"], t["b1k"])
        z0 = pick(cache["z0k"])
        if self.backend == "native":
            from exact_port import lookup_secant
            dh, dh32, x = lookup_secant(z, z0, t["kernel_beta"], t["kernel_scale"], *t["lut"])
        else:
            # F.gelu is a CPU reference for the archived BF16 LUT values.
            dh = (F.gelu(z, approximate="tanh").to(work)
                  - F.gelu(z0, approximate="tanh").to(work)).to(now.dtype)
            dh32 = dh.to(work)
            x = (dh32 - (z.to(work) - z0.to(work)) * t["kernel_beta"]) / t["kernel_scale"]
        q = (dh32 @ t["zmap"]).sub_(ju)
        base = self._selected_mm(dh)
        old_y = pick(cache["y0"])
        if self.backend == "native":
            from exact_port import apply
            apply(t["base"], base.numel(), old_y, linear.to(now.dtype), base, base.numel())
        else:
            base = (old_y.to(work) + linear.to(now.dtype).to(work)) + base
        return base, q, x

    def fit(self, u_p, observed=None):
        base, qp, xp = self.operands(self.probes, u_p, probe_fit=True)
        if observed is None:
            observed = self.phase.ffn_2.apply(F.gelu(self.phase.ffn_0.apply(u_p), approximate="tanh"))
        target, qd, v = (observed.to(base.dtype) - base).double(), qp.double(), self.t["vd"]
        system = ((xp @ xp.T) / self.k).double() + self.inv_metric.double()
        chol, info = torch.linalg.cholesky_ex(system)
        torch._assert_async((info == 0).all(), "Nonpositive probe covariance")
        iq = torch.cholesky_solve(qd, chol)
        gain, info = torch.linalg.solve_ex((qd.T @ iq) * self.t["vv"],
                                           (iq * (target @ v.T)).sum(0))
        torch._assert_async((info == 0).all(), "Singular probe gains")
        coefficient = torch.cholesky_solve(target - (qd * gain) @ v, chol)
        torch._assert_async(torch.isfinite(coefficient).all() & torch.isfinite(gain).all(),
                            "Nonfinite FFN readout")
        out_type = torch.float64 if base.dtype == torch.float64 else torch.float32
        readout = torch.cat((gain.to(out_type)[:, None] * self.t["v"],
                             0.5 * coefficient.to(out_type)))
        return FFNFit(xp, readout, observed, self.probes, gain, coefficient)

    def apply_rows(self, token_ids, u_rows, fit, *, overwrite_probes=False, out=None):
        base, q, x = self.operands(token_ids, u_rows)
        design = torch.cat((q, (x @ fit.xp.T) / self.k), dim=1)
        torch.addmm(base, design, fit.readout, out=base)
        if out is None:
            out = base.to(u_rows.dtype)
        else:
            if out.shape != u_rows.shape or out.dtype != u_rows.dtype:
                raise ValueError("output buffer axes/dtype differ")
            out.copy_(base)
        if overwrite_probes:
            # Used for arbitrary standalone subsets; streamed non-P tiles skip it.
            ids = _ids(token_ids, self.cache.rows, u_rows.device)
            same = ids[:, None] == fit.probes[None, :]
            locations = same.nonzero()
            out.index_copy_(0, locations[:, 0], fit.observed.index_select(0, locations[:, 1]))
        return out

    def full(self, u, *, tile_rows=8192):
        if tile_rows < 1 or len(u) != self.cache.rows:
            raise ValueError("invalid complete FFN row count/tile")
        fit = self.fit(u.index_select(0, self.probes))
        out = torch.empty_like(u)
        for start in range(0, len(u), tile_rows):
            sl = slice(start, min(start + tile_rows, len(u)))
            self.apply_rows(sl, u[sl], fit, out=out[sl])
        out.index_copy_(0, self.probes, fit.observed)
        return out, None


def make_hit(tile_rows=8192, *, backend="native"):
    """Drop-in fourstep_hit signature; default uses original native kernels."""
    def hit(u, phase, cached, prepared, probes, inv_metric):
        return FFNStaged(phase, prepared, cached, probes, inv_metric,
                         backend=backend).full(u, tile_rows=tile_rows)
    return hit


def factor_cycle_budget(rows, width, selected, rank, modes=16, layers=40,
                        *, projection_bytes=2, preserve_probe_u=True, probes=64):
    per_slot = rows * (2 * (width + selected) + projection_bytes * rank + 4 * modes)
    if preserve_probe_u:
        per_slot += 2 * probes * width
    old = 2 * rows * width * width
    new = 6 * rows * width * rank + 2 * rows * width * modes
    return dict(anchor_hit_old_flops=old, anchor_hit_factor_flops=new,
                whole_video_net_saved_flops=2 * layers * (old - new),
                resident_history_bytes=layers * per_slot,
                compute_break_even_rank=(width - modes) / 3,
                scope="major matrix arithmetic; includes anchor projections; excludes pointwise")


def recovery_filter(error, q, x, v, probes, inv_metric):
    """Differentiable real-GLS action M_H (I-R_F P) error.

    All operands are supplied observations/features from the caller. The
    function neither creates model data nor changes the native teacher. The
    inverse/GLS solve uses FP64 as in the existing fit, while the large action
    follows ``error.dtype``. Finite-precision native readout casts are outside
    this real-operator fitting objective.
    """
    if error.ndim != 2 or q.shape[0] != len(error) or x.shape[0] != len(error):
        raise ValueError("recovery error/features must share original row axis")
    if v.shape != (q.shape[1], error.shape[1]):
        raise ValueError("recovery mode/output axes differ")
    ids = _ids(probes, len(error), error.device)
    qp, xp = q.index_select(0, ids), x.index_select(0, ids)
    target = error.index_select(0, ids)
    k = x.shape[1]
    system = ((xp @ xp.T) / k).double() + inv_metric.double()
    chol = torch.linalg.cholesky(system)
    iq = torch.cholesky_solve(qp.double(), chol)
    vd = v.double()
    gain = torch.linalg.solve((qp.double().T @ iq) * (vd @ vd.T),
                              (iq * (target.double() @ vd.T)).sum(0))
    coefficient = torch.cholesky_solve(target.double() - (qp.double() * gain) @ vd, chol)
    result = error - (q * gain.to(error.dtype)) @ v
    result = result - 0.5 * ((x @ xp.T) / k) @ coefficient.to(error.dtype)
    return result.index_fill(0, ids, 0)


def recovery_aware_factor_loss(left, right, target_a0, delta_u, q, x, v, probes,
                               inv_metric, *, gauge=None, jmap=None,
                               output_weight=None, reduction="mean"):
    """Loss for supplied CAL actions after their existing probe recovery.

    Computes delta-U L R without materializing L R. A paired fixed/learnable
    gauge targets A0+Jdiag(c)V, matching the paired B_c transform. Ordinary SVD
    does not optimize this loss; callers must explicitly choose/record fitting.
    """
    error = (delta_u @ left) @ right - delta_u @ target_a0
    if gauge is not None:
        if jmap is None:
            raise ValueError("a gauge target requires its original J map")
        error = error - ((delta_u @ jmap) * gauge) @ v
    result = recovery_filter(error, q, x, v, probes, inv_metric)
    if output_weight is not None:
        result = result @ output_weight
    if reduction == "mean":
        return result.square().mean()
    if reduction == "sum":
        return result.square().sum()
    if reduction == "none":
        return result
    raise ValueError("reduction must be mean, sum or none")
