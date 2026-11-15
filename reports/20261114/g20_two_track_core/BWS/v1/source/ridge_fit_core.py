"""G20 R1b numerical recheck math. Compute only under the guarded runner.

Fixes over the frozen R1 fit_core.py (sha256 34db25d270a9f99ef14a4fa291a3e3f7cbee2ce02725c1abd184fa1a3e671d44):

1. stable_ridge_fit: F64 accumulation of the Gram/RHS from raw X/R, weighted
   thin-SVD sample-space solve (D lies in the CAL row space by construction,
   rank <= CAL rows), plus an F64 eigh cross-check on the F64-formed normal
   system. The R1 FP32 normal-equation path amplified FP32 null-space noise by
   g_j^2/ridge ~ 1e6-1e7 (ridge ~ 1.4e-7..2.9e-7), producing D numerical ranks
   4239-4732 far above the exact-solution bound rank(X_CAL) <= 517/523.
2. free_ridge_fit_fp32_original: the R1 solver retained VERBATIM so the original
   D can be recomputed and its BF16 export checked against the frozen R1 SHA.
3. icc_two_residual_decomposed: total error split into propagated stage-1 (A),
   stage-2 at the native hidden state (B) and the cross term, all reduced in
   F64 from the same BF16 operands; gated zero-increment SSE added.
4. cheap_site_stats gate_fro2: the gate is already (n, D); the extra factor n
   in R1 is removed. Diagnostic field only; no score consumed it.

Numeric contract: TF32 off; FP32 math lift; BF16 replay reproduces the deployed
cache/gate rounding boundaries; all error reductions accumulate in FP64.
"""
import math

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
    """Per-source lightweight homomorphic statistics + Q1/reuse-Y readouts.

    Carried from R1 with the gate_fro2 fix (g is already (n, D)); not invoked
    by the R1b runner, which reuses the published 80-site landscape."""
    u0, u1, y0, y1, g = p['u0'], p['u1'], p['y0'], p['y1'], p['g']
    _, _, y0n = native_ffn(u0, w1, b1, w2, b2)
    _, _, y1n = native_ffn(u1, w1, b1, w2, b2)
    x = u1.float() - u0.float()
    dy = y1n.float() - y0n.float()
    n = u0.shape[0]
    pred_inc = x @ a.float()
    pred_full = y0n.float() + pred_inc
    q1_math_sse = _f64_sse(pred_full, y1n.float(), g)
    _, q1_gated, q1_target = bf16_replay(u0, u1, y0n, y1n, g, a)
    q1_replay_sse = _f64_sse(q1_gated.float(), q1_target.float(), torch.ones_like(g))
    reuse_sse = _f64_sse(y0n.float(), y1n.float(), g)
    out = {
        'rows': n,
        'token_indices': p['ids'],
        'du_fro2': float(x.double().square().sum()),
        'dy_fro2': float(dy.double().square().sum()),
        'gated_dy_fro2': float((dy.double() * g.double()).square().sum()),
        'gated_y1_fro2': float((y1n.double() * g.double()).square().sum()),
        'gate_fro2': float(g.double().square().sum()),
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


def free_ridge_fit_fp32_original(x_list, y_list, r_list, g, n_list, a_q1, lam):
    """VERBATIM R1 solver (fit_core.py free_ridge_fit). Recomputation only.

    FP32 Gram/RHS, FP32 eigh, clamp_min(0), filter on all 5120 modes. Known to
    amplify FP32 null-space noise by g_j^2/ridge; retained so the original D and
    its BF16 export can be reproduced and compared against the frozen R1 SHA."""
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


def stable_ridge_fit(x_list, y_list, r_list, g, n_list, a_q1, lam):
    """Exact ridge solution of the SAME registered objective, stably computed.

    Primary: weighted thin SVD of the stacked CAL increments in F64. With
    Xw = W^{1/2} X = U S Vh and Rw = W^{1/2} R, column j solves
    (g_j^2 M + rho I) d_j = g_j^2 b_j inside span(Vh):
        c_ji = g_j^2 s_i (U^T Rw)_ji / (g_j^2 s_i^2 + rho),  d_j = Vh^T c_j.
    All min(N, d) modes are used (no truncation): the filter is bounded and
    small-singular-value contributions vanish linearly, so no null-space noise
    is amplified. D is in the CAL row space by construction.

    Cross-check: F64 eigh of the F64-formed Gram with the same column filter.
    Both solve the F64 normal system formed from raw X/R; agreement certifies
    the solve, and the F64 normal residual is measured against that system.

    Two row-space readouts: D_out_of_rowspace_rel_fro projects on the FULL
    sample space (all N right-singular vectors) — an implementation certificate
    (~1e-15). D_out_of_calgram_rel_fro projects on the CAL subspace at R1's
    EXACT Gram criterion mu > 1e-6*mu_max with mu = sv^2 (equivalently
    sv > 1e-3*smax) — the same basis definition as R1's omega_source and
    gram_numerical_rank, so omega is comparable by construction. With
    rho ~ 1e-7 even F64 noise-floor modes (s ~ 1e-16*smax) contribute ~s/rho
    ~ 1e-9 relative to D, so the CAL-subspace quantity is NOT certified against
    a tight threshold; the original FP32 D is measured against the same basis
    for the artifact comparison. Returns F64 D, both bases, and diagnostics."""
    s, z = _normalize_parts(x_list, y_list, g, n_list)
    dim = x_list[0].shape[1]
    za = float(a_q1.double().square().sum()) + EPS
    ridge = lam / za
    dev = g.device
    weights = [1.0 / (s * n * z) for n in n_list]
    xw = torch.cat([x.double() * math.sqrt(w) for x, w in zip(x_list, weights)], 0)
    rw = torch.cat([r.double() * math.sqrt(w) for r, w in zip(r_list, weights)], 0)
    u, sv, vh = torch.linalg.svd(xw, full_matrices=False)
    g2 = g.double().square()
    t = u.T @ rw                                            # (N, d)
    denom = g2[None, :] * sv[:, None].square() + ridge      # (N, d)
    c = (g2[None, :] * sv[:, None] * t) / denom
    d64 = vh.T @ c                                          # (d, d) F64
    if not torch.isfinite(d64).all():
        raise ValueError('nonfinite stable ridge solution')
    # F64 normal system from raw inputs (independent of the SVD path).
    m64 = torch.zeros((dim, dim), device=dev, dtype=torch.float64)
    b64 = torch.zeros((dim, dim), device=dev, dtype=torch.float64)
    for x, r, w in zip(x_list, r_list, weights):
        x64 = x.double()
        m64 += (x64.T @ x64) * w
        b64 += (x64.T @ r.double()) * w
    m64 = (m64 + m64.T) * 0.5
    mu, v = torch.linalg.eigh(m64)
    mu = mu.clamp_min(0.0)
    filt = g2[None, :] / (g2[None, :] * mu[:, None] + ridge)
    d64_eigh = v @ (filt * (v.T @ b64))
    cross_rel = float((d64 - d64_eigh).norm() / d64_eigh.norm().clamp_min(1e-30))
    lhs = (m64 @ d64) * g2[None, :] + ridge * d64
    rhs = b64 * g2[None, :]
    res = float((lhs - rhs).norm() / rhs.norm().clamp_min(1e-30))
    svals = torch.linalg.svdvals(d64)
    smax = float(svals.max())
    num_rank = int((svals > RANK_TOL * max(smax, 1e-30)).sum())
    mu_sv = sv.square()
    active = mu_sv > RANK_TOL * float(mu_sv.max().clamp_min(1e-30))
    gram_rank = int(active.sum())
    va_all = vh.T.contiguous()                          # (d, N) full sample space
    va_cal = vh[active, :].T.contiguous()               # (d, r) R1 Gram criterion
    proj = va_all @ (va_all.T @ d64)
    oos = float((d64 - proj).norm() / d64.norm().clamp_min(1e-30))
    proj_cal = va_cal @ (va_cal.T @ d64)
    oos_cal = float((d64 - proj_cal).norm() / d64.norm().clamp_min(1e-30))
    info = {
        'lambda': lam, 'Z': z, 'ZA': za, 'ridge_over_ZA': ridge,
        'solver': 'f64_weighted_thin_svd_sample_space',
        'normal_relative_residual_f64': res,
        'cross_check_f64_eigh_rel_l2': cross_rel,
        'D_numerical_rank': num_rank, 'D_max_singular': smax,
        'gram_numerical_rank': gram_rank,
        'gram_max_singular': float(sv.max()),
        'D_out_of_rowspace_rel_fro': oos,
        'D_out_of_calgram_rel_fro': oos_cal,
        'D_fro2': float(d64.square().sum()),
    }
    return d64, {'all': va_all, 'calgram': va_cal}, info


def out_of_space_rel(dmat, va):
    """||(I - Pi_train) D||_F / ||D||_F in F64; va spans the CAL row space."""
    d64 = dmat.double()
    proj = va @ (va.T @ d64)
    return float((d64 - proj).norm() / d64.norm().clamp_min(1e-30))


def omega_source(x_test, va):
    """||X_test (I-Pi_train)||^2 / ||X_test||^2; Pi_train spans CAL DeltaU rows."""
    xt = x_test.double()
    proj = (xt @ va) @ va.T
    resid = xt - proj
    denom = float(xt.square().sum())
    return float(resid.square().sum()) / denom if denom > 0 else 0.0


def icc_two_residual_decomposed(p, fc1_left, fc1_right, fc2_left, fc2_right,
                                w1, b1, w2, b2):
    """Frozen ICC svd_r512 two-residual hit with F64 two-stage attribution.

    pred is computed exactly as in R1 (BF16 ops). Attribution uses the same
    BF16 operands lifted to F64:
        A = second(h_pred) - second(h1n)   (stage-1 error propagated via GELU+second)
        B = second(h1n) + q2 - y1n         (stage-2 error at the native hidden state)
        total_f64 = ||(A+B) g||^2 = sA + sB + 2<Ag, Bg>  (F64 identity)
    math_gated_sse keeps the R1 BF16-pred definition; the relative difference
    between the two totals quantifies the BF16 rounding of pred itself."""
    u0, u1, g = p['u0'], p['u1'], p['g']
    z0, h0, y0n = native_ffn(u0, w1, b1, w2, b2)
    z1n, h1n, y1n = native_ffn(u1, w1, b1, w2, b2)

    def first(x):
        return F.linear(F.linear(x, fc1_right), fc1_left)

    def second(x):
        return F.linear(F.linear(x, fc2_right), fc2_left)

    q1 = (z0 - first(u0)).detach()
    q2 = (y0n - second(h0)).detach()
    z_pred = first(u1) + q1
    h_pred = F.gelu(z_pred, approximate='tanh')
    pred = second(h_pred) + q2
    if not torch.isfinite(pred).all():
        raise ValueError('nonfinite ICC prediction')
    math_sse = _f64_sse(pred.float(), y1n.float(), g)
    sec_pred = second(h_pred).double()
    sec_nat = second(h1n).double()
    gd = g.double()
    ad = (sec_pred - sec_nat) * gd
    bd = (sec_nat + q2.double() - y1n.double()) * gd
    s_a = float(ad.square().sum())
    s_b = float(bd.square().sum())
    cross = float(2.0 * (ad * bd).sum())
    total_f64 = float((ad + bd).square().sum())
    decomp_res = abs(total_f64 - (s_a + s_b + cross)) / max(total_f64, 1e-30)
    if decomp_res > 1e-10:
        raise ValueError('ICC decomposition identity broken')
    bf16_round_rel = abs(float(math_sse) - total_f64) / max(total_f64, 1e-30)
    zc = first(u0) + q1
    hc = F.gelu(zc, approximate='tanh')
    zero = second(hc) + q2
    n = u0.shape[0]
    return {
        'math_gated_sse': float(math_sse),
        'math_gated_mse': float(math_sse) / (n * D),
        'total_f64_gated_sse': total_f64,
        'stage1_propagated_gated_sse': s_a,
        'stage2_native_h_gated_sse': s_b,
        'cross_term_gated': cross,
        'decomposition_rel_residual': decomp_res,
        'bf16_pred_rounding_rel': bf16_round_rel,
        'stage1_pre_gelu_fro2': float((z_pred.double() - z1n.double()).square().sum()),
        'hidden_err_fro2': float((h_pred.double() - h1n.double()).square().sum()),
        'zero_delta_max_abs': float((zero.float() - y0n.float()).abs().max()),
        'zero_delta_rel_l2': float((zero.float() - y0n.float()).norm()
                                   / y0n.float().norm().clamp_min(1e-20)),
        'zero_delta_gated_sse': float(_f64_sse(zero.float(), y0n.float(), g)),
    }