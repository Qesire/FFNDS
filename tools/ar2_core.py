"""CPU primitives for the frozen AR2 post-Q2 support-selection diagnostic.

Selectors see only a, effective W2, gate, and training scores. Bias is excluded
from the linear signal used to fit gains and added exactly once in native
projection. This module does not load an experiment, select a winning method,
or claim full-model quality/speed. Native projection keeps the saved BF16 order.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ScopeKey:
    task: int
    seed: int
    step: int
    expert: str
    block: int = 20


def donor_key(test: ScopeKey) -> ScopeKey:
    if test.task not in (0, 6) or test.seed not in (20261001, 20261002):
        raise ValueError("AR2 uses only the two frozen tasks and seeds")
    expected = {0: "high_noise_model", 2: "low_noise_model"}
    if test.step not in expected or test.expert != expected[test.step] or test.block != 20:
        raise ValueError("AR2 scope violates the frozen step/expert/block binding")
    opposite = 20261002 if test.seed == 20261001 else 20261001
    return ScopeKey(0, opposite, test.step, test.expert, test.block)


def design_weights(values, rows):
    weights = np.asarray(values, dtype=np.float64)
    if rows <= 0 or weights.shape != (rows,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Explicit design weights must match every actual row")
    return weights


def top_groups(scores, keep):
    scores = np.asarray(scores)
    if scores.ndim not in (1, 2) or not 0 < keep <= scores.shape[-1] or not np.isfinite(scores).all():
        raise ValueError("Invalid finite score matrix/keep count")
    # Columns are group IDs in ascending order, so stable sort resolves ties by ID.
    return np.argsort(-scores, axis=-1, kind="stable")[..., :keep]


def static_groups(training_scores, training_weights, keep):
    scores = np.asarray(training_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("Training scores must be rows x groups")
    weights = design_weights(training_weights, len(scores))
    return top_groups(np.sum(scores * weights[:, None], axis=0), keep)


def support_mask(group_ids, *, rows, groups, group_size):
    ids = np.asarray(group_ids)
    if ids.dtype.kind not in "iu" or ids.ndim not in (1, 2):
        raise ValueError("Group IDs must be integer vectors/matrices")
    if ids.ndim == 1:
        ids = np.broadcast_to(ids, (rows, len(ids)))
    if ids.shape[0] != rows or ids.shape[1] == 0 or group_size <= 0:
        raise ValueError("Support shape does not match actual rows")
    if np.any(ids < 0) or np.any(ids >= groups) or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError("Support contains duplicate or out-of-range groups")
    mask = np.zeros((rows, groups), dtype=np.float32)
    np.put_along_axis(mask, ids, 1, axis=1)
    return np.repeat(mask, group_size, axis=1)


@torch.no_grad()
def score_families(activation, weight, gate, *, group_size=64):
    """FP32 A/E/C operands and GEMMs; FP64 output-energy reductions per spec."""
    if activation.device.type != "cpu" or weight.device.type != "cpu" or gate.device.type != "cpu":
        raise ValueError("This is the CPU AR2 diagnostic")
    if activation.ndim != 2 or weight.ndim != 2 or activation.shape[1] != weight.shape[0]:
        raise ValueError("Expected a[n,F] and mathematical W2[F,D]")
    rows, hidden = activation.shape
    if group_size <= 0 or hidden % group_size:
        raise ValueError("Hidden width must be divisible by group size")
    if gate.shape[-1] != weight.shape[1]:
        raise ValueError("Gate output width does not match W2")
    a, w = activation.float(), weight.float()
    g = gate.reshape(-1, weight.shape[1]).float()
    if len(g) == 1:
        g = g.expand(rows, -1)
    if len(g) != rows:
        raise ValueError("Gate must be broadcast or have one value per row")
    groups = hidden // group_size
    scores = {name: np.empty((rows, groups), dtype=np.float64) for name in "AEC"}
    for index in range(groups):
        aa = a[:, index * group_size:(index + 1) * group_size]
        ww = w[index * group_size:(index + 1) * group_size]
        scores["A"][:, index] = aa.square().sum(1, dtype=torch.float64).numpy()
        scores["E"][:, index] = ((aa.square() @ ww.square()) * g.square()).sum(1, dtype=torch.float64).numpy()
        scores["C"][:, index] = ((aa @ ww) * g).double().square().sum(1).numpy()
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Non-finite selector scores")
    return scores


@dataclass
class GainFit:
    numerator: float
    denominator: float
    alpha: float | None


def fit_nonnegative_gain(signal, dense_signal, weights):
    """Fit from one training donor only; both inputs exclude the fixed bias."""
    signal = np.asarray(signal, dtype=np.float64)
    target = np.asarray(dense_signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape != target.shape:
        raise ValueError("Training signal/target shapes must agree")
    weights = design_weights(weights, len(signal))
    if not np.isfinite(signal).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite training signals")
    numerator = float(np.sum(weights[:, None] * signal * target))
    denominator = float(np.sum(weights[:, None] * signal * signal))
    alpha = max(0.0, numerator / denominator) if denominator > signal.shape[1] * 1e-24 else None
    return GainFit(numerator, denominator, alpha)


class NativeProjection:
    """Cache weight layouts once; avoid a 140 MB transpose/copy per token row."""
    def __init__(self, weight, bias=None):
        if weight.device.type != "cpu" or weight.dtype != torch.bfloat16 or weight.ndim != 2:
            raise ValueError("Expected the saved CPU BF16 W2[F,D] operand")
        if bias is not None and (bias.device.type != "cpu" or bias.dtype != weight.dtype or bias.shape != (weight.shape[1],)):
            raise ValueError("Bias must be one native BF16 output-width vector")
        self.weight_t = weight.T.contiguous()
        self.weight_fp32 = weight.float().contiguous()
        self.bias = bias

    @torch.no_grad()
    def native(self, activation, gate, mask, gain):
        if activation.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
            raise ValueError("Native a/gate operands must remain BF16")
        if activation.device.type != "cpu" or gate.device.type != "cpu" or activation.ndim != 2 or activation.shape[1] != self.weight_t.shape[1]:
            raise ValueError("Expected actual CPU activation rows and the matching W2 width")
        if gate.shape[-1] != self.weight_t.shape[0]:
            raise ValueError("Gate width does not match W2")
        gate = gate.reshape(-1, self.weight_t.shape[0])
        if len(gate) not in (1, len(activation)):
            raise ValueError("Gate rows do not match activation rows")
        mask = torch.as_tensor(mask, dtype=torch.float32)
        if mask.shape != activation.shape:
            raise ValueError("Mask must match the selected actual activation rows")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Native support masks must contain only zero/one")
        gain = torch.as_tensor(gain, dtype=torch.float32)
        if gain.ndim > 1 or (gain.ndim == 1 and len(gain) != len(activation)):
            raise ValueError("Gain must be a scalar or one value per actual row")
        if not torch.isfinite(gain).all() or torch.any(gain < 0):
            raise ValueError("Gain must be finite and nonnegative")
        if gain.ndim == 1:
            gain = gain[:, None]
        effective = ((activation.float() * mask) * gain).to(activation.dtype)
        return F.linear(effective, self.weight_t, self.bias) * gate

    @torch.no_grad()
    def fp32_signal(self, activation, gate, mask):
        a = activation.float() * torch.as_tensor(mask, dtype=torch.float32)
        return (a @ self.weight_fp32) * gate.reshape(-1, self.weight_t.shape[0]).float()


def weighted_quantile(values, weights, levels):
    values = np.asarray(values, dtype=np.float64)
    weights = design_weights(weights, len(values))
    levels = np.asarray(levels, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((levels < 0) | (levels > 1)):
        raise ValueError("Invalid quantile values/levels")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    ids = np.searchsorted(cumulative, levels * cumulative[-1], side="left")
    return values[order[np.minimum(ids, len(values) - 1)]]
