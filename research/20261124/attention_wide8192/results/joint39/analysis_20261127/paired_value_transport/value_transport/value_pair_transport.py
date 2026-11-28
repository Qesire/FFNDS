"""CPU reference for one head's paired-value transport.

This is a mechanism reference, not a production Wan/GPU integration.
Token rows have persistent identities across the anchor and current call.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ValuePairState:
    indices: np.ndarray
    normalized_anchor_rows: np.ndarray


def value_moments(value, floor=1e-12):
    value = np.asarray(value, dtype=np.float64)
    mean = value.mean(axis=0)
    scale = np.sqrt(np.maximum(value.var(axis=0), 0.0))
    inverse = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale > floor)
    return mean, scale, inverse


def capture_value_pairs(value_anchor, indices):
    mean, scale, inverse = value_moments(value_anchor)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("Pair indices must be distinct token rows")
    standardized = (np.asarray(value_anchor, dtype=np.float64)[indices] - mean) * inverse
    return ValuePairState(indices.copy(), standardized.copy())


def fit_value_map(state, value_current, *, ridge=1e-3):
    """Fit M around current diagonal scaling D.

Minimize mean_j ||X_j (M-D) - (Y_j-X_j D)||^2 + ridge ||M-D||^2.
No native-attention output is used. ridge=0 uses least squares; exact-map
identification then requires the sampled source rows to span the needed space.
"""
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    value_current = np.asarray(value_current, dtype=np.float64)
    mean, scale, _ = value_moments(value_current)
    x = state.normalized_anchor_rows
    if value_current.shape[1] != x.shape[1]:
        raise ValueError("Value channel count changed")
    y = value_current[state.indices] - mean
    diagonal = np.diag(scale)
    residual = y - x @ diagonal
    if ridge == 0:
        delta = np.linalg.lstsq(x, residual, rcond=None)[0]
    else:
        count, dim = x.shape
        gram = x.T @ x / count + ridge * np.eye(dim)
        cross = x.T @ residual / count
        delta = np.linalg.solve(gram, cross)
    return diagonal + delta, mean, scale


def encode_centered(native_anchor, selected_anchor, rho_anchor, value_anchor):
    mean, _, inverse = value_moments(value_anchor)
    rho = np.asarray(rho_anchor, dtype=np.float64).reshape(-1, 1)
    return (native_anchor - (1-rho)*selected_anchor - rho*mean) * inverse


def decode_centered(selected_current, rho_anchor, history, mean_current, value_map):
    rho = np.asarray(rho_anchor, dtype=np.float64).reshape(-1, 1)
    return (1-rho)*selected_current + rho*mean_current + history @ value_map


def ridge_repair(base, features, native_probe, probe_rows, *, alpha=6.4):
    """Fixed-feature reference correction with exact probe overwrite."""
    design = np.column_stack((np.ones(len(features)), features))
    sampled = design[probe_rows]
    target = native_probe - base[probe_rows]
    coefficient = np.linalg.solve(sampled.T @ sampled + alpha*np.eye(design.shape[1]),
                                  sampled.T @ target)
    output = base + design @ coefficient
    output[probe_rows] = native_probe
    return output


def sparsepr_v2_repair(sparse, native_probe, probe_rows, *, weights=None,
                       ridge=1.0, epsilon=1e-6):
    """Paper-defined v2 recovery on a supplied base and supplied probes.

This does not implement response-coupled partitioning or author probe selection.
Supplying weights=None means equal coverage, not inferred author group weights.
"""
    sparse = np.asarray(sparse, dtype=np.float64)
    probe_rows = np.asarray(probe_rows, dtype=np.int64)
    p = len(probe_rows)
    w = np.ones(p) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (p,) or np.any(w <= 0):
        raise ValueError("weights must be positive, one per probe")
    w = w / w.mean()
    mean_x = (w[:, None]*sparse[probe_rows]).mean(axis=0)
    centered_x = sparse[probe_rows] - mean_x
    scale_x = np.sqrt((w[:, None]*centered_x**2).mean(axis=0) + epsilon)
    x = (sparse - mean_x)/scale_x
    residual = native_probe - sparse[probe_rows]
    mean_r = (w[:, None]*residual).mean(axis=0)
    y = residual - mean_r
    xp = x[probe_rows]
    coefficient = np.linalg.solve(xp.T @ (w[:, None]*xp) + ridge*np.eye(x.shape[1]),
                                  xp.T @ (w[:, None]*y))
    output = sparse + mean_r + x @ coefficient
    output[probe_rows] = native_probe
    return output
