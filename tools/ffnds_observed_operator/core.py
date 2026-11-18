"""Finite-secant FFN constructions in NumPy FP64.

Rows are tokens. U: [N,d], W1: [m,d], W2: [d,m], modes: [q,m].
All returned FFN predictions are pre-gate. This module neither emulates BF16
rounding nor binds a Wan model. Reference costs are not latency measurements.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _array(value, ndim, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return value


def _rows(value, n, *, count=None):
    original = np.asarray(value)
    if original.ndim != 1 or not np.issubdtype(original.dtype, np.integer):
        raise ValueError("probe rows must be integer row indices")
    rows = original.astype(np.int64, copy=False)
    if (not len(rows) or np.any(rows < 0) or np.any(rows >= n)
            or len(np.unique(rows)) != len(rows)
            or (count is not None and len(rows) != count)):
        raise ValueError("probe rows must be unique, in range, and of the requested count")
    return rows


def _positive_matrix(value, r, name):
    value = np.eye(r) if value is None else _array(value, 2, name)
    if value.shape != (r, r) or not np.allclose(value, value.T, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{name} must be symmetric with shape [{r},{r}]")
    try:
        factor = np.linalg.cholesky((value + value.T) * 0.5)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return value, factor


def _cholesky_solve(matrix, rhs):
    lower = np.linalg.cholesky((matrix + matrix.T) * 0.5)
    return np.linalg.solve(lower.T, np.linalg.solve(lower, rhs))


def gelu_tanh(z):
    """The real-arithmetic tanh GELU used by this reference."""
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))


def dense_ffn(u, w1, bias_in, w2, bias_out):
    """FP64 F(U); useful as a small synthetic observer in the algebra tests."""
    u, w1, w2 = _array(u, 2, "u"), _array(w1, 2, "w1"), _array(w2, 2, "w2")
    bias_in, bias_out = _array(bias_in, 1, "bias_in"), _array(bias_out, 1, "bias_out")
    d, m = u.shape[1], w1.shape[0]
    if w1.shape != (m, d) or w2.shape != (d, m) or bias_in.shape != (m,) or bias_out.shape != (d,):
        raise ValueError("native FFN geometry mismatch")
    return gelu_tanh(u @ w1.T + bias_in) @ w2.T + bias_out


def paired_features(u, w1, bias_in, indices, beta, cal_scale):
    """Phi(U)=[Z_K, GELU(Z_K)-beta*Z_K]/CAL_scale, shape [N,2k].

    Production proposal: k=16 and r=32. Smaller k is accepted for algebra tests.
    beta and strictly positive scales are frozen CAL assets, not fit here.
    cal_scale describes the endpoint differences; no per-hit rescaling is hidden.
    """
    u, w1 = _array(u, 2, "u"), _array(w1, 2, "w1")
    bias_in = _array(bias_in, 1, "bias_in")
    indices = _rows(indices, len(w1))
    beta, cal_scale = _array(beta, 1, "beta"), _array(cal_scale, 1, "cal_scale")
    k = len(indices)
    if (u.shape[1] != w1.shape[1] or bias_in.shape != (len(w1),)
            or beta.shape != (k,) or cal_scale.shape != (2 * k,)
            or np.any(cal_scale <= 0)):
        raise ValueError("paired feature geometry or CAL scale mismatch")
    z = u @ w1[indices].T + bias_in[indices]
    return np.concatenate((z, gelu_tanh(z) - z * beta), axis=1) / cal_scale


@dataclass(frozen=True)
class FeatureSupport:
    # basis maps retained CAL-whitened feature coefficients back to input columns.
    basis: np.ndarray             # [r,k], basis.T @ Gamma_CAL @ basis = I
    coordinates: np.ndarray       # [N,k], X @ basis
    whitened: np.ndarray          # [N,k], coordinates / current RMS singular values
    singular_values: np.ndarray   # [k], singular values of X Gamma_CAL^{-1/2}/sqrt(N)
    discarded_energy: float

    @property
    def rank(self):
        return len(self.singular_values)


def prepare_support(x, *, cal_gram=None, singular_rtol=1e-10, rms_atol=1e-12):
    """CAL-whiten, then retain current nonzero feature directions by an SVD.

    For a rebasing X'=XT, transform Gamma_CAL'=T.T Gamma_CAL T as well.
    The absolute threshold is in CAL-whitened endpoint-difference RMS units.
    Thresholding is numerical truncation, not an estimate of BF16 noise.
    """
    x = _array(x, 2, "x")
    n, r = x.shape
    if not n or singular_rtol < 0 or rms_atol < 0:
        raise ValueError("need nonempty token axis and nonnegative support thresholds")
    _, lower = _positive_matrix(cal_gram, r, "cal_gram")
    # X @ L^{-T}; use a triangular system rather than forming an inverse.
    cal_coordinates = np.linalg.solve(lower, x.T).T
    _, singular, vh = np.linalg.svd(cal_coordinates / np.sqrt(n), full_matrices=False)
    cutoff = max(rms_atol, singular_rtol * (singular[0] if len(singular) else 0.0))
    keep = singular > cutoff
    basis = np.linalg.solve(lower.T, vh[keep].T)
    coordinates = x @ basis
    kept = singular[keep]
    whitened = coordinates / kept if len(kept) else np.empty((n, 0))
    total = float(singular @ singular)
    dropped = float(singular[~keep] @ singular[~keep]) / total if total else 0.0
    return FeatureSupport(basis, coordinates, whitened, kept, dropped)


def select_probes(x, p=64, *, policy="fixed", token_ids=None, fixed_ids=None,
                  seed=42, support=None, cal_gram=None):
    """Return row indices, not global token IDs, without consulting real residuals.

    fixed: an explicitly supplied list of global IDs, or a frozen seed permutation
    of sorted global IDs. This default is NOT claimed to match legacy P64 IDs.
    geometry: greedy log-determinant gain on current whitened features, with
    information matrix initially I. Near ties (1e-12 relative) choose the lowest
    global token ID. Each step is a rank-one information update, O(p*N*r).
    """
    x = _array(x, 2, "x")
    n = len(x)
    if not isinstance(p, (int, np.integer)) or p < 1 or p > n:
        raise ValueError("p must be an integer in [1,N]")
    ids = np.arange(n, dtype=np.int64) if token_ids is None else np.asarray(token_ids)
    if (ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != n):
        raise ValueError("token_ids must contain one unique integer ID per row")
    order = np.argsort(ids, kind="stable")
    if policy == "fixed":
        if fixed_ids is None:
            return order[np.random.default_rng(seed).permutation(n)[:p]]
        fixed_ids = np.asarray(fixed_ids)
        if (fixed_ids.shape != (p,) or not np.issubdtype(fixed_ids.dtype, np.integer)
                or len(np.unique(fixed_ids)) != p):
            raise ValueError("fixed_ids must have exactly p distinct integer IDs")
        lookup = {int(item): row for row, item in enumerate(ids)}
        try:
            return np.array([lookup[int(item)] for item in fixed_ids], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("a fixed probe ID is absent from this token panel") from exc
    if policy != "geometry" or fixed_ids is not None:
        raise ValueError("geometry probes use current features; fixed_ids are for fixed policy")
    support = prepare_support(x, cal_gram=cal_gram) if support is None else support
    if support.whitened.shape[0] != n:
        raise ValueError("support belongs to a different token panel")
    w = support.whitened
    if not support.rank:
        return order[:p].copy()
    information_inverse = np.eye(support.rank)
    score = np.einsum("na,na->n", w, w)
    available = np.ones(n, dtype=bool)
    selected = []
    for _ in range(p):
        top = np.max(score[available])
        tied = available & np.isclose(score, top, rtol=1e-12, atol=1e-14)
        row = order[np.flatnonzero(tied[order])[0]]
        selected.append(row)
        available[row] = False
        direction = information_inverse @ w[row]
        denominator = 1.0 + w[row] @ direction
        projections = w @ direction
        score = np.maximum(0.0, score - projections**2 / denominator)
        information_inverse -= np.outer(direction, direction) / denominator
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class ReadoutFit:
    readout: np.ndarray           # [r,d]
    update: np.ndarray            # [r,d]
    feature_rank: int
    probe_rank: int
    probe_singular_values: np.ndarray
    discarded_feature_energy: float
    probe_residual_rms: float
    prediction_update_rms: float


def fit_observed_readout(x, residual_probe, probe_rows, *, ridge,
                         prior=None, cal_gram=None, cal_floor=0.01,
                         singular_rtol=1e-10, rms_atol=1e-12, support=None):
    """Fit a signed full-output readout on current observed token rows.

    Objective in retained feature support:
      ||X_P V - R_P||_F^2/p
      + ridge * tr((V-V0).T [X.T X/N + cal_floor^2 Gamma_CAL] (V-V0)).

    cal_floor=0 is the earlier algebraic limit. Positive cal_floor supplies an
    absolute CAL-scale noise floor as X tends to zero; this is separate from SVD
    truncation. No independent per-output gate weighting is silently introduced.
    """
    x, rp = _array(x, 2, "x"), _array(residual_probe, 2, "residual_probe")
    n, r = x.shape
    rows = _rows(probe_rows, n, count=len(rp))
    if not np.isfinite(ridge) or ridge < 0 or not np.isfinite(cal_floor) or cal_floor < 0:
        raise ValueError("ridge and cal_floor must be finite and nonnegative")
    v0 = np.zeros((r, rp.shape[1])) if prior is None else _array(prior, 2, "prior")
    if v0.shape != (r, rp.shape[1]):
        raise ValueError("prior must have shape [r,d]")
    support = prepare_support(x, cal_gram=cal_gram, singular_rtol=singular_rtol,
                              rms_atol=rms_atol) if support is None else support
    if support.basis.shape[0] != r or support.coordinates.shape[0] != n:
        raise ValueError("support belongs to different features")
    xp = x[rows]
    discrepancy = rp - xp @ v0
    delta = np.zeros_like(v0)
    k = support.rank
    probe_singular = np.linalg.svd(support.whitened[rows] / np.sqrt(len(rows)),
                                   compute_uv=False)
    threshold = singular_rtol * (probe_singular[0] if len(probe_singular) else 0.0)
    probe_rank = int(np.count_nonzero(probe_singular > max(threshold, rms_atol)))
    if k:
        zp = support.coordinates[rows]
        if ridge == 0:
            if probe_rank < k:
                raise ValueError("unregularized probes do not identify the retained feature support")
            # Use the same geometry for the rank check and actual solve. Solving
            # raw zp with this rcond would silently discard small but retained
            # CAL-scale directions a second time.
            white_coefficients = np.linalg.lstsq(support.whitened[rows], discrepancy,
                                                 rcond=singular_rtol)[0]
            coefficients = white_coefficients / support.singular_values[:, None]
        else:
            regularizer = support.singular_values**2 + cal_floor**2
            system = zp.T @ zp / len(rows) + ridge * np.diag(regularizer)
            coefficients = _cholesky_solve(system, zp.T @ discrepancy / len(rows))
        delta = support.basis @ coefficients
    v = v0 + delta
    return ReadoutFit(v, delta, k, probe_rank, probe_singular,
                      support.discarded_energy,
                      float(np.linalg.norm(rp - xp @ v) / np.sqrt(len(rows))),
                      float(np.linalg.norm(x @ delta) / np.sqrt(n)))


@dataclass(frozen=True)
class ObservedAnchor:
    matrix: np.ndarray            # [d,d], fixed at anchor and hit
    residual: np.ndarray          # [N,d], F(U0)-U0@A
    features: np.ndarray          # [N,r], Phi(U0)


def make_observed_anchor(u0, f0, matrix, phi0):
    u0, f0 = _array(u0, 2, "u0"), _array(f0, 2, "f0")
    matrix, phi0 = _array(matrix, 2, "matrix"), _array(phi0, 2, "phi0")
    if f0.shape != u0.shape or matrix.shape != (u0.shape[1],) * 2 or len(phi0) != len(u0):
        raise ValueError("anchor geometry mismatch")
    return ObservedAnchor(matrix.copy(), f0 - u0 @ matrix, phi0.copy())


def observed_hit(u1, phi1, anchor, probe_ffn: Callable, *, p=64, ridge,
                 cal_floor=0.01, cal_gram=None, prior=None, policy="fixed",
                 token_ids=None, fixed_ids=None, seed=42,
                 singular_rtol=1e-10, rms_atol=1e-12, adaptation=True):
    """One hit; probe_ffn receives exactly p current rows, never the full U1.

    Returns (pre_gate_prediction, diagnostics). Probe outputs are not spliced into
    prediction: all rows use the same fitted operator, including observed rows.
    adaptation=False returns the matched static prior using the same caches and
    performs zero observer calls. The caller owns expert/layer binding and
    discards the anchor after its window.
    """
    u1, phi1 = _array(u1, 2, "u1"), _array(phi1, 2, "phi1")
    if u1.shape != anchor.residual.shape or phi1.shape != anchor.features.shape:
        raise ValueError("hit does not match anchor token and feature geometry")
    base, x = u1 @ anchor.matrix + anchor.residual, phi1 - anchor.features
    if not adaptation:
        static = np.zeros((x.shape[1], u1.shape[1])) if prior is None else _array(prior, 2, "prior")
        if static.shape != (x.shape[1], u1.shape[1]):
            raise ValueError("prior must have shape [r,d]")
        return base + x @ static, {"probe_rows": np.empty(0, dtype=np.int64),
                                  "fit": None, "adaptation": False}
    support = prepare_support(x, cal_gram=cal_gram, singular_rtol=singular_rtol, rms_atol=rms_atol)
    rows = select_probes(x, p, policy=policy, token_ids=token_ids, fixed_ids=fixed_ids,
                         seed=seed, support=support)
    actual = _array(probe_ffn(u1[rows]), 2, "probe_ffn output")
    if actual.shape != (p, u1.shape[1]):
        raise ValueError("observer must return p pre-gate FFN rows")
    fit = fit_observed_readout(x, actual - base[rows], rows, ridge=ridge, prior=prior,
                               cal_gram=cal_gram, cal_floor=cal_floor,
                               singular_rtol=singular_rtol, rms_atol=rms_atol, support=support)
    return base + x @ fit.readout, {"probe_rows": rows, "fit": fit, "adaptation": True}


def native_mode_templates(delta_u_probe, w1, w2, modes):
    """Return T[j,p,c] without materializing B[j,i,c]. modes[j,h]."""
    du, w1 = _array(delta_u_probe, 2, "delta_u_probe"), _array(w1, 2, "w1")
    w2, modes = _array(w2, 2, "w2"), _array(modes, 2, "modes")
    if w1.shape[1] != du.shape[1] or w2.shape != (du.shape[1], len(w1)) or modes.shape[1] != len(w1):
        raise ValueError("native mode geometry mismatch")
    dz = du @ w1.T
    return (dz[None, :, :] * modes[:, None, :]) @ w2.T


def compile_native_modes(theta, w1, w2, modes):
    """Compile delta A from a combined hidden coefficient vector, with no diag."""
    theta, modes = _array(theta, 1, "theta"), _array(modes, 2, "modes")
    w1, w2 = _array(w1, 2, "w1"), _array(w2, 2, "w2")
    if theta.shape != (len(modes),) or modes.shape[1] != len(w1) or w2.shape != (w1.shape[1], len(w1)):
        raise ValueError("native mode compilation geometry mismatch")
    return (w1.T * (theta @ modes)) @ w2.T


@dataclass(frozen=True)
class ModeFit:
    theta: np.ndarray
    gram: np.ndarray
    template_rank: int
    gated_probe_residual_rms: float


def fit_mode_coefficients(templates, residual_probe, gate, *, ridge,
                          prior=None, cal_metric=None, row_weights=None,
                          rank_rtol=1e-10):
    """One shared theta across output channels; gate affects this optimum.

    Objective: sum_p w_p ||(sum_j theta_j*T_j[p]-R[p])*g||^2
               + ridge*(theta-prior).T CAL_metric (theta-prior).
    Default weights sum to 1. The result remains a pre-gate operator.
    """
    t, rp = _array(templates, 3, "templates"), _array(residual_probe, 2, "residual_probe")
    gate = _array(gate, 1, "gate")
    q, p, d = t.shape
    if not q or not p or rp.shape != (p, d) or gate.shape != (d,) or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("mode fit geometry or regularization mismatch")
    v0 = np.zeros(q) if prior is None else _array(prior, 1, "prior")
    metric, _ = _positive_matrix(cal_metric, q, "cal_metric")
    weights = np.full(p, 1.0 / p) if row_weights is None else _array(row_weights, 1, "row_weights")
    if v0.shape != (q,) or weights.shape != (p,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid mode prior or nonnegative row weights")
    weights = weights / weights.sum()
    multiplier = np.sqrt(weights)[:, None] * gate[None, :]
    design = (t * multiplier[None, :, :]).reshape(q, -1).T
    target = (rp * multiplier).reshape(-1)
    singular = np.linalg.svd(design, compute_uv=False)
    threshold = rank_rtol * (singular[0] if len(singular) else 0.0)
    rank = int(np.count_nonzero(singular > threshold))
    gram = design.T @ design
    if ridge == 0:
        if rank < q:
            raise ValueError("unregularized probes do not identify the shared mode coefficients")
        theta = np.linalg.lstsq(design, target, rcond=rank_rtol)[0]
    else:
        theta = _cholesky_solve(gram + ridge * metric,
                                design.T @ target + ridge * metric @ v0)
    error = np.einsum("j,jpc->pc", theta, t) - rp
    return ModeFit(theta, gram, rank, float(np.linalg.norm(error * multiplier)))


@dataclass(frozen=True)
class ModeAnchor:
    inputs: np.ndarray            # [N,d]
    outputs: np.ndarray           # [N,d], true F(U0)
    matrix: np.ndarray            # [d,d], fixed base A0


def make_mode_anchor(u0, f0, matrix):
    u0, f0, matrix = _array(u0, 2, "u0"), _array(f0, 2, "f0"), _array(matrix, 2, "matrix")
    if u0.shape != f0.shape or matrix.shape != (u0.shape[1],) * 2:
        raise ValueError("mode anchor geometry mismatch")
    return ModeAnchor(u0.copy(), f0.copy(), matrix.copy())


def mode_hit(u1, anchor, probe_ffn: Callable, w1, w2, modes, gate, probe_rows,
             *, ridge, prior=None, cal_metric=None, adaptation=True):
    """One full-rank-family hit with explicitly bound probe rows and current gate."""
    u1 = _array(u1, 2, "u1")
    if u1.shape != anchor.inputs.shape:
        raise ValueError("mode hit does not match anchor geometry")
    du = u1 - anchor.inputs
    if not adaptation:
        theta_static = np.zeros(len(modes)) if prior is None else _array(prior, 1, "prior")
        current = anchor.matrix + compile_native_modes(theta_static, w1, w2, modes)
        return anchor.outputs + du @ current, {
            "probe_rows": np.empty(0, dtype=np.int64), "fit": None, "adaptation": False}
    rows = _rows(probe_rows, len(u1))
    actual = _array(probe_ffn(u1[rows]), 2, "probe_ffn output")
    if actual.shape != (len(rows), u1.shape[1]):
        raise ValueError("observer must return p pre-gate FFN rows")
    rp = actual - anchor.outputs[rows] - du[rows] @ anchor.matrix
    templates = native_mode_templates(du[rows], w1, w2, modes)
    fit = fit_mode_coefficients(templates, rp, gate, ridge=ridge, prior=prior, cal_metric=cal_metric)
    current = anchor.matrix + compile_native_modes(fit.theta, w1, w2, modes)
    return anchor.outputs + du @ current, {"probe_rows": rows.copy(), "fit": fit, "adaptation": True}
