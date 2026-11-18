"""Real-arithmetic structure diagnostics and source-level risk."""

import math

import torch
import torch.nn.functional as F


def prepare_pair(pair, w1, b1, w2):
    a, u = pair["u0"], pair["u1"]
    za, zu = a @ w1.T + b1, u @ w1.T + b1
    pair["Y"] = (F.gelu(zu, approximate="tanh") - F.gelu(za, approximate="tanh")) @ w2.T
    z = ((a + u) / 2) @ w1.T + b1
    t = torch.tanh(math.sqrt(2 / math.pi) * (z + .044715 * z.pow(3)))
    derivative = (1 + t + z * (1 - t.square()) * math.sqrt(2 / math.pi)
                  * (1 + 3 * .044715 * z.square())) / 2
    pair["Jv"] = ((pair["X"] @ w1.T) * derivative) @ w2.T


def structure(pair, base, q):
    gate = pair["g"]
    state = (pair["Jv"] - base) * gate
    chord = (pair["Y"] - pair["Jv"]) * gate
    residual = (pair["Y"] - base) * gate
    gated_q = gate[:, None] * q
    left, singular, _ = torch.linalg.svd(gated_q, full_matrices=False)
    left = left[:, singular > singular[0] * max(gated_q.shape) * torch.finfo(q.dtype).eps]
    missed = residual - (residual @ left) @ left.T
    row = {
        "state_energy": float(state.square().mean()),
        "chord_energy": float(chord.square().mean()),
        "cross_energy": float(2 * (state * chord).mean()),
        "residual_energy": float(residual.square().mean()),
        "output_missed_energy": float(missed.square().mean()),
        "closure_max_abs": float((state + chord - residual).abs().max()),
    }
    scale = max(float((pair["Y"] * gate).square().mean()), float((base * gate).square().mean()))
    row["resolution_floor_mse"] = (64 * torch.finfo(q.dtype).eps) ** 2 * scale
    row["eta_out"] = (row["output_missed_energy"] / row["residual_energy"]
                      if row["residual_energy"] > row["resolution_floor_mse"] else None)
    return row


def metrics(pair, pred):
    y = pair["Y"] * pair["g"]
    yhat = pred * pair["g"]
    mse = float((yhat - y).square().mean())
    true = float(y.square().mean())
    norm = float(yhat.square().mean())
    cos = float((yhat * y).mean()) / math.sqrt(norm * true) if norm * true > 0 else None
    return {"gated_mse": mse, "relative_risk": mse / true if true > 0 else None,
            "cosine": cos, "norm_ratio": math.sqrt(norm / true) if true > 0 else None}


def verify_features(pair, params, w1, b1):
    from .features import native_secant, quadratic

    dim = params["Q32"].shape[1]
    p = params["P64"][:, :dim] / params["sigma"]
    a, u = pair["u0"], pair["u1"]
    s_a, s_u = a @ p - params["b"] / params["sigma"], u @ p - params["b"] / params["sigma"]
    torch.testing.assert_close(quadratic(pair, params)[:, dim:], (s_u.square() - s_a.square()) / 2,
                               rtol=1e-10, atol=1e-10)
    idx, alpha = params["hidden_indices"], params["hidden_alpha"]
    z_a, z_u = a @ w1[idx].T + b1[idx], u @ w1[idx].T + b1[idx]
    delta = (F.gelu(z_u, approximate="tanh") - z_u * alpha
             - (F.gelu(z_a, approximate="tanh") - z_a * alpha))
    torch.testing.assert_close(native_secant(pair, params, w1, b1)[:, dim:], delta,
                               rtol=1e-10, atol=1e-10)
    zero = dict(pair, u1=a, X=torch.zeros_like(pair["X"]))
    for feature in (quadratic(zero, params), native_secant(zero, params, w1, b1)):
        if torch.count_nonzero(feature):
            raise ValueError("Zero displacement must yield a zero feature difference")
