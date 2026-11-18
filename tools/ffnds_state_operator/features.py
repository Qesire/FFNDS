"""FP64 CAL features for the four frozen state-operator arms."""

import math

import torch
import torch.nn.functional as F

ARMS = ("L0", "L+", "Q32", "N32")


def source_rms(blocks):
    return torch.stack([x.square().mean(0) for x in blocks]).mean(0).sqrt().clamp_min(1e-6)


def extra_basis(cal, p64, rank=64):
    """Top CAL displacement directions in the orthogonal complement of the old span."""
    weighted = []
    for pair in cal:
        v = pair["X"]
        weighted.append((v - (v @ p64) @ p64.T) / math.sqrt(len(cal) * len(v)))
    _, singular, vh = torch.linalg.svd(torch.cat(weighted), full_matrices=False)
    if len(singular) < rank or singular[rank - 1] <= singular[0] * 1e-12:
        raise ValueError("CAL displacement does not resolve the required complement rank")
    candidate = vh[:rank].T
    basis = torch.linalg.qr(candidate - p64 @ (p64.T @ candidate)).Q
    rows = basis.abs().argmax(0)
    cols = torch.arange(rank, device=basis.device)
    return (basis * basis[rows, cols].sign()).contiguous()


def quadratic(pair, params):
    p = params["P64"][:, :params["Q32"].shape[1]] / params["sigma"]
    mean = (pair["u0"] + pair["u1"]) / 2
    displacement = pair["X"] @ p
    return torch.cat((displacement, displacement * (mean @ p - params["b"] / params["sigma"])), 1)


def native_chord(pair, w1, b1):
    a = pair["u0"] @ w1.T + b1
    u = pair["u1"] @ w1.T + b1
    return u - a, F.gelu(u, approximate="tanh") - F.gelu(a, approximate="tanh")


def native_secant(pair, params, w1, b1):
    idx = params["hidden_indices"]
    dz, dg = native_chord(pair, w1[idx], b1[idx])
    p = params["P64"][:, :params["Q32"].shape[1]] / params["sigma"]
    return torch.cat((pair["X"] @ p, dg - dz * params["hidden_alpha"]), 1)


def select_hidden(cal, params, w1, b1, rank=32):
    """Greedy conditional gain of a fixed projected CAL surrogate.

    Regress the projected gated score (E_L * g^2) Q / k on feature columns.
    Each source/row receives weight k / (n_sources * n_rows), where k is
    trace(Q.T diag(g^2) Q) / r. This scalarizes the gate Gram for selection
    only. The final 64x32 fit and CAL Schur statistic use the actual gate Gram.
    Remove the shared D features and all previously chosen secants before
    each selection. Exact score ties use the lowest channel index.
    """
    q = params["Q32"]
    p = params["P64"][:, :q.shape[1]] / params["sigma"]
    dz_blocks, dg_blocks = zip(*(native_chord(pair, w1, b1) for pair in cal))
    dz_norm = torch.stack([dz.square().mean(0) for dz in dz_blocks]).sum(0)
    dz_dot = torch.stack([(dz * dg).mean(0) for dz, dg in zip(dz_blocks, dg_blocks)]).sum(0)
    alpha = dz_dot / dz_norm.clamp_min(torch.finfo(dz_norm.dtype).tiny)

    features, linear, response = [], [], []
    for pair, dz, dg in zip(cal, dz_blocks, dg_blocks):
        gate2 = pair["g"].square()
        k = (q.square().sum(1) * gate2).sum() / q.shape[1]
        if k == 0:
            continue
        weight = torch.sqrt(k / (len(cal) * len(dz)))
        features.append((dg - dz * alpha) * weight)
        linear.append((pair["X"] @ p) * weight)
        response.append(((pair["R"] * gate2) @ q) * (weight / k))
    if not features:
        raise ValueError("No CAL gate signal for hidden selection")

    candidates = torch.cat(features)
    target = torch.cat(response)
    del features, dz_blocks, dg_blocks
    left, singular, _ = torch.linalg.svd(torch.cat(linear), full_matrices=False)
    base = left[:, singular > singular[0] * 1e-12]
    candidates -= base @ (base.T @ candidates)
    target -= base @ (base.T @ target)
    available = torch.ones(w1.shape[0], device=w1.device, dtype=torch.bool)
    indices = []
    gains = []
    for _ in range(rank):
        norm2 = torch.linalg.vector_norm(candidates, dim=0).square()
        valid = available & (norm2 > norm2.max() * 1e-24)
        scores = (candidates.T @ target).square().sum(1) / norm2.clamp_min(1e-300)
        scores.masked_fill_(~valid, -torch.inf)
        idx = int(scores.argmax())
        if not torch.isfinite(scores[idx]):
            raise ValueError("CAL native secants have insufficient conditional rank")
        direction = candidates[:, idx] / norm2[idx].sqrt()
        candidates.addr_(direction, direction @ candidates, alpha=-1)
        target.addr_(direction, direction @ target, alpha=-1)
        available[idx] = False
        indices.append(idx)
        gains.append(float(scores[idx]))
    idx = torch.tensor(indices, device=w1.device, dtype=torch.int64)
    return idx, alpha[idx], gains


def arm_features(arm, pair, params, w1, b1):
    if arm == "L+":
        return pair["X"] @ params["P_extra64"]
    if arm == "Q32":
        return quadratic(pair, params)
    if arm == "N32":
        return native_secant(pair, params, w1, b1)
    raise ValueError(f"No added features for {arm}")
