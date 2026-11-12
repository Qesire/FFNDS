"""G20 R1 same-input screening math. Compute only under the guarded runner.

Arms: Q1-r2 deployed BF16 matrix; FREE_RIDGE A_Q1+D (exact shared-Gram eigen
solve, offline diagnostic); ICC svd_r512 two-residual frozen factors. Chipmunk
author block-mean arm is NOT scored here: its real full-M proxy groups/masks are
absent for the CAL sources (recorded MISSING_ASSET, deferred to R2 capture).

Numeric contract: TF32 off; FP32 math lift; BF16 replay reproduces the deployed
cache/gate rounding boundaries; all error reductions accumulate in FP64.
"""
import torch
import torch.nn.functional as F

D = 5120
M = 13824
RANK_TOL = 1e-6          # pre-registered numerical-rank relative threshold
EPS = 1e-12


def _f64_sse(pred, target, g):
    d = (pred.double() - target.double()) * g.double()
    v = d.square().sum()
    if not torch.isfinite(v):
        raise ValueError('nonfinite gated SSE')
    return v


def native_ffn(u, w1, b1, w2, b2):
    z = torch.addmm(b1, u, w1.T)
    h = F.gelu(z, approximate='tanh')
    y = torch.addmm(b2, h, w2.T)
    return z, h, y


def bf16_replay(u0, u1, y0, y1, g, a):
    """Deployed anchor-residual hit path with BF16 cache/gate rounding.

    Returns pre-gate prediction and gated prediction/target so callers can score
    either the math lift or the native rounding proxy. Final residual-add into
    the stream is NOT represented (no archived full-M x)."""
    q0 = u0 @ a
    q1 = u1 @ a
    cache = (y0.float() - q0.float()).bfloat16()
    base = (q1.float() + cache.float()).bfloat16()
    gated = (base.float() * g).bfloat16()
    target = (y1.float() * g).bfloat16()
    return base, gated, target


def cheap_site_stats(p, a, w1, b1, w2, b2):
    """Per-source lightweight homomorphic statistics + Q1/reuse-Y readouts."""
    u0, u1, y0, y1, g = p['u0'], p['u1'], p['y0'], p['y1'], p['g']
    _, _, y0n = native_ffn(u0, w1, b1, w2, b2)
    _, _, y1n = native_ffn(u1, w1, b1, w2, b2)
    x = u1.float() - u0.float()
    dy = y1n.float() - y0n.float()
    n = u0.shape[0]
    # Q1 math lift (FP32 real matrix, current gate), target = native F(U_t).
    pred_inc = x @ a.float()
    pred_full = y0n.float() + pred_inc
    q1_math_sse = _f64_sse(pred_full, y1n.float(), g)
    # Q1 native BF16 replay (cache + gate rounding).
    _, q1_gated, q1_target = bf16_replay(u0, u1, y0n, y1n, g, a)
    q1_replay_sse = _f64_sse(q1_gated.float(), q1_target.float(), torch.ones_like(g))
    # reuse-Y baseline (predict zero increment): error is the gated increment.
    reuse_sse = _f64_sse(y0n.float(), y1n.float(), g)
    out = {
        'rows': n,
        'token_indices': p['ids'],
        'du_fro2': float(x.double().square().sum()),
        'dy_fro2': float(dy.double().square().sum()),
        'gated_dy_fro2': float((dy.double() * g.double()).square().sum()),
        'gated_y1_fro2': float((y1n.double() * g.double()).square().sum()),
        'gate_fro2': float(g.double().square().sum() * n),
        'q1_math_gated_sse': float(q1_math_sse),
        'q1_replay_gated_sse': float(q1_replay_sse),
        'reuse_y_gated_sse': float(reuse_sse),
        'archive_vs_recomputed_y1_rel_l2': float(
            (y1.float() - y1n.float()).norm() / y1n.float().norm().clamp_min(1e-20)),
        'archive_vs_recomputed_y1_bit_mismatch': float(
            (y1.view(torch.int16) != y1n.view(torch.int16)).float().mean()),
    }
    return out


def _normalize_parts(x_list, y_list, g, n_list):
    s = len(x_list)
    z = sum(float((y.double() * g.double()).square().sum()) / n
            for y, n in zip(y_list, n_list)) / s + EPS
    return s, z


def free_ridge_fit(x_list, y_list, r_list, g, n_list, a_q1, lam):
    """Exact ridge fit of D in A=A_Q1+D under source-equal-weight gated loss.

    Gate is per-channel and shared across rows/sources (asserted by caller), so
    every output column shares the Gram M=sum_c X_c^T X_c/(S n_c Z); column j
    solves (g_j^2 M + lam/ZA I) d_j = g_j^2 b_j. One symmetric eigendecomposition
    gives the exact unique solution for all columns. Offline diagnostic only."""
    s, z = _normalize_parts(x_list, y_list, g, n_list)
    d = x_list[0].shape[1]
    za = float(a_q1.double().square().sum()) + EPS
    m = torch.zeros((d, d), device=g.device, dtype=torch.float32)
    b = torch.zeros((d, d), device=g.device, dtype=torch.float32)
    for x, r, n in zip(x_list, r_list, n_list):
        m += (x.T @ x) / (s * n * z)
        b += (x.T @ r) / (s * n * z)
    m = (m + m.T) * 0.5
    mu, v = torch.linalg.eigh(m)
    mu = mu.clamp_min(0.0)
    g2 = g.double().square()                       # (D,)
    mu64 = mu.double()
    ridge = lam / za
    vb = v.T @ b                                   # (D,D)
    filt = g2[None, :] / (g2[None, :] * mu64[:, None] + ridge)   # (D,D)
    d64 = v.double() @ (filt * vb.double())
    d = d64.float()
    if not torch.isfinite(d).all():
        raise ValueError('nonfinite ridge solution')
    # True normal-equation residual in F64: column j is g_j^2 M d_j + ridge d_j - g_j^2 b_j.
    lhs = (m.double() @ d64) * g2[None, :] + ridge * d64
    rhs = b.double() * g2[None, :]
    res = float((lhs - rhs).norm() / rhs.norm().clamp_min(1e-30))
    svals = torch.linalg.svdvals(d)
    smax = float(svals.max())
    num_rank = int((svals > RANK_TOL * max(smax, 1e-30)).sum())
    gram_rank = int((mu > RANK_TOL * float(mu.max().clamp_min(1e-30))).sum())
    info = {
        'lambda': lam, 'Z': z, 'ZA': za, 'ridge_over_ZA': ridge,
        'normal_relative_residual': res,
        'D_numerical_rank': num_rank, 'D_max_singular': smax,
        'gram_numerical_rank': gram_rank,
        'gram_max_eigen': float(mu.max()), 'gram_min_eigen': float(mu.min()),
        'D_fro2': float(d.double().square().sum()),
    }
    return d, mu, v, info


def omega_source(x_test, mu, v):
    """||X_test (I-Pi_train)||^2 / ||X_test||^2; Pi_train spans CAL DeltaU rows."""
    active = mu > RANK_TOL * float(mu.max().clamp_min(1e-30))
    va = v[:, active].double()
    xt = x_test.double()
    proj = (xt @ va) @ va.T
    resid = xt - proj
    denom = float(xt.square().sum())
    return float(resid.square().sum()) / denom if denom > 0 else 0.0


def icc_two_residual(p, fc1_left, fc1_right, fc2_left, fc2_right, w1, b1, w2, b2):
    """Frozen ICC svd_r512 ffnds_two_residual hit on one paired source."""
    u0, u1, g = p['u0'], p['u1'], p['g']
    z0, h0, y0n = native_ffn(u0, w1, b1, w2, b2)
    _, _, y1n = native_ffn(u1, w1, b1, w2, b2)

    def first(x):
        return F.linear(F.linear(x, fc1_right), fc1_left)

    def second(x):
        return F.linear(F.linear(x, fc2_right), fc2_left)

    q1 = (z0 - first(u0)).detach()
    q2 = (y0n - second(h0)).detach()
    z = first(u1) + q1
    h = F.gelu(z, approximate='tanh')
    pred = second(h) + q2
    # zero-increment control: hit at the anchor input.
    zc = first(u0) + q1
    hc = F.gelu(zc, approximate='tanh')
    zero = second(hc) + q2
    if not torch.isfinite(pred).all():
        raise ValueError('nonfinite ICC prediction')
    math_sse = _f64_sse(pred.float(), y1n.float(), g)
    n = u0.shape[0]
    return {
        'math_gated_sse': float(math_sse),
        'math_gated_mse': float(math_sse) / (n * D),
        'zero_delta_max_abs': float((zero.float() - y0n.float()).abs().max()),
        'zero_delta_rel_l2': float((zero.float() - y0n.float()).norm()
                                   / y0n.float().norm().clamp_min(1e-20)),
    }