"""Deterministic engineering checks; executed only inside the reviewed job."""
import torch
from fit_core import (free_ridge_fit, icc_two_residual, cheap_site_stats,
                      bf16_replay, native_ffn, EPS)


def _dense_ref(xs, ys, rs, g, ns, a_q1, lam):
    s = len(xs)
    z = sum(float((y.double() * g.double()).square().sum()) / n
            for y, n in zip(ys, ns)) / s + EPS
    za = float(a_q1.double().square().sum()) + EPS
    m = sum((x.T @ x) / (s * n * z) for x, n in zip(xs, ns))
    b = sum((x.T @ r) / (s * n * z) for x, r, n in zip(xs, rs, ns))
    g2 = g.double().square()
    d = xs[0].shape[1]
    ref = torch.empty(d, d, device=g.device, dtype=torch.float64)
    for j in range(d):
        lhs = g2[j] * m.double() + (lam / za) * torch.eye(d, device=g.device, dtype=torch.float64)
        rhs = g2[j] * b.double()[:, j]
        ref[:, j] = torch.linalg.solve(lhs, rhs)
    return ref


def run():
    torch.manual_seed(20261111)
    dev = 'cuda'
    d, n = 5, 7
    # --- free ridge exact eigen solve vs per-column dense F64 solve ---
    xs = [torch.randn(n, d, device=dev) for _ in range(2)]
    ys = [torch.randn(n, d, device=dev) for _ in range(2)]
    g = torch.rand(d, device=dev) + 0.5
    a_q1 = torch.randn(d, d, device=dev) * 0.1
    rs = [y - x @ a_q1 for x, y in zip(xs, ys)]
    ns = [n, n]
    lam = 1e-2
    d_fit, mu, v, info = free_ridge_fit(xs, ys, rs, g, ns, a_q1, lam)
    ref = _dense_ref(xs, ys, rs, g, ns, a_q1, lam)
    rel = float((d_fit.double() - ref).norm() / ref.norm().clamp_min(1e-30))
    if rel > 2e-4:
        raise RuntimeError('ridge eigen solve mismatch: %g' % rel)
    if info['normal_relative_residual'] > 1e-4:
        raise RuntimeError('ridge normal residual too large')

    # --- rank-deficient Gram (the actual underdetermined regime) ---
    xr = torch.randn(n, 2, device=dev) @ torch.randn(2, d, device=dev)
    xs2 = [xr, xr.clone()]
    ys2 = [torch.randn(n, d, device=dev) for _ in range(2)]
    rs2 = [y - xr @ a_q1 for y in ys2]
    d2, mu2, v2, info2 = free_ridge_fit(xs2, ys2, rs2, g, ns, a_q1, lam)
    ref2 = _dense_ref(xs2, ys2, rs2, g, ns, a_q1, lam)
    rel2 = float((d2.double() - ref2).norm() / ref2.norm().clamp_min(1e-30))
    if rel2 > 2e-4:
        raise RuntimeError('rank-deficient ridge eigen solve mismatch: %g' % rel2)
    if info2['normal_relative_residual'] > 1e-4:
        raise RuntimeError('rank-deficient ridge normal residual too large')
    if info2['gram_numerical_rank'] > 2:
        raise RuntimeError('rank-deficient Gram rank not detected')

    # --- ICC two-residual zero factors reproduce native exactly ---
    w1 = torch.randn(9, d, device=dev, dtype=torch.bfloat16)
    b1 = torch.randn(9, device=dev, dtype=torch.bfloat16)
    w2 = torch.randn(d, 9, device=dev, dtype=torch.bfloat16)
    b2 = torch.randn(d, device=dev, dtype=torch.bfloat16)
    u0 = torch.randn(n, d, device=dev, dtype=torch.bfloat16)
    u1 = torch.randn(n, d, device=dev, dtype=torch.bfloat16)
    gg = torch.ones(n, d, device=dev)
    p = {'u0': u0, 'u1': u1, 'g': gg, 'ids': list(range(n)),
         'y0': torch.zeros(n, d, device=dev, dtype=torch.bfloat16),
         'y1': torch.zeros(n, d, device=dev, dtype=torch.bfloat16)}
    zl = torch.zeros(9, 3, device=dev, dtype=torch.bfloat16)
    zr = torch.zeros(3, d, device=dev, dtype=torch.bfloat16)
    sl = torch.zeros(d, 3, device=dev, dtype=torch.bfloat16)
    sr = torch.zeros(3, 9, device=dev, dtype=torch.bfloat16)
    icc = icc_two_residual(p, zl, zr, sl, sr, w1, b1, w2, b2)
    if icc['zero_delta_max_abs'] != 0.0:
        raise RuntimeError('ICC zero-factor zero-delta not exact')

    # --- cheap stats: reuse-Y SSE equals gated increment energy ---
    a = torch.randn(d, d, device=dev, dtype=torch.bfloat16) * 0.1
    cs = cheap_site_stats(p, a, w1, b1, w2, b2)
    if abs(cs['reuse_y_gated_sse'] - cs['gated_dy_fro2']) > 1e-6 * max(cs['gated_dy_fro2'], 1e-12):
        raise RuntimeError('reuse-Y identity broken')

    # --- BF16 replay with zero matrix returns cached anchor output ---
    zero_a = torch.zeros(d, d, device=dev, dtype=torch.bfloat16)
    _, _, y0n = native_ffn(u0, w1, b1, w2, b2)
    _, _, y1n = native_ffn(u1, w1, b1, w2, b2)
    base, gated, target = bf16_replay(u0, u1, y0n, y1n, gg, zero_a)
    if not torch.equal(base, y0n):
        raise RuntimeError('zero-matrix replay must return anchor output')
    if not torch.isfinite(gated).all() or not torch.isfinite(target).all():
        raise RuntimeError('nonfinite replay')
    return {'ridge_vs_dense_rel_l2': rel,
            'ridge_rank_deficient_vs_dense_rel_l2': rel2,
            'ridge_rank_deficient_gram_rank': info2['gram_numerical_rank'],
            'icc_zero_factor_zero_delta_exact': True,
            'reuse_y_identity': True,
            'zero_matrix_replay_anchor_exact': True}