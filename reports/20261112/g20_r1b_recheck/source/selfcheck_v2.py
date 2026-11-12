"""Deterministic engineering checks for R1b; executed only inside the reviewed job.

Keeps the R1 checks (same rounded-system eigen-vs-dense agreement, rank-deficient
Gram, ICC zero factors, reuse-Y identity, zero-matrix replay) and adds the check
R1 lacked: a realistically ill-conditioned problem (rho ~ 1e-7, rank-deficient
wide X constructed in F64 so numerical rank == mathematical rank 6) whose
reference is built in F64 FROM RAW X/R, not from FP32 normal equations. The
stable solver must match that reference (condition-number-aware F64 tolerance
1e-7), stay in the full sample space (<= 1e-12) and obey rank(D) <= 6; the
original FP32 path (fed FP32-rounded copies, mirroring R1's data path) must
leak > 1e-4 of its energy outside the R1-criterion CAL subspace, proving the
test discriminates the R1 failure mode."""
import torch
from fit_core_v2 import (free_ridge_fit_fp32_original, stable_ridge_fit,
                         out_of_space_rel, icc_two_residual_decomposed,
                         cheap_site_stats, bf16_replay, native_ffn, EPS)


def _dense_ref(xs, ys, rs, g, ns, a_q1, lam):
    """R1 reference: FP32 normal system solved per-column in F64 (kept as-is)."""
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
    # --- R1 continuity: eigen solve vs per-column dense F64 solve (rounded system) ---
    xs = [torch.randn(n, d, device=dev) for _ in range(2)]
    ys = [torch.randn(n, d, device=dev) for _ in range(2)]
    g = torch.rand(d, device=dev) + 0.5
    a_q1 = torch.randn(d, d, device=dev) * 0.1
    rs = [y - x @ a_q1 for x, y in zip(xs, ys)]
    ns = [n, n]
    lam = 1e-2
    d_fit, mu, v, info = free_ridge_fit_fp32_original(xs, ys, rs, g, ns, a_q1, lam)
    ref = _dense_ref(xs, ys, rs, g, ns, a_q1, lam)
    rel = float((d_fit.double() - ref).norm() / ref.norm().clamp_min(1e-30))
    if rel > 2e-4:
        raise RuntimeError('ridge eigen solve mismatch: %g' % rel)
    if info['normal_relative_residual'] > 1e-4:
        raise RuntimeError('ridge normal residual too large')

    # --- R1 continuity: rank-deficient Gram ---
    xr = torch.randn(n, 2, device=dev) @ torch.randn(2, d, device=dev)
    xs2 = [xr, xr.clone()]
    ys2 = [torch.randn(n, d, device=dev) for _ in range(2)]
    rs2 = [y - xr @ a_q1 for y in ys2]
    d2, mu2, v2, info2 = free_ridge_fit_fp32_original(xs2, ys2, rs2, g, ns, a_q1, lam)
    ref2 = _dense_ref(xs2, ys2, rs2, g, ns, a_q1, lam)
    rel2 = float((d2.double() - ref2).norm() / ref2.norm().clamp_min(1e-30))
    if rel2 > 2e-4:
        raise RuntimeError('rank-deficient ridge eigen solve mismatch: %g' % rel2)
    if info2['normal_relative_residual'] > 1e-4:
        raise RuntimeError('rank-deficient ridge normal residual too large')
    if info2['gram_numerical_rank'] > 2:
        raise RuntimeError('rank-deficient Gram rank not detected')

    # --- NEW: realistic ill-conditioning, F64-from-raw reference ---
    # X is constructed in F64 so its NUMERICAL rank equals the mathematical
    # rank 6 (an FP32 rank-6 product has numerical rank 12: rounding injects
    # 6 modes at ~3e-8 which the all-modes resolvent legitimately amplifies
    # by s/rho, spuriously tripping the rank bound). The ORIGINAL FP32 path
    # receives FP32-rounded copies, mirroring R1's FP32 data path, so its
    # Gram-formation leak discrimination is preserved.
    dh, nh, rank = 96, 12, 6
    xh = torch.randn(nh, rank, device=dev, dtype=torch.float64) \
        @ torch.randn(rank, dh, device=dev, dtype=torch.float64)
    xsh = [xh, xh.clone()]
    ysh = [torch.randn(nh, dh, device=dev) for _ in range(2)]
    gh = torch.rand(dh, device=dev) + 0.5
    a_h = torch.randn(dh, dh, device=dev)          # ZA ~ dh^2 -> rho ~ 1e-7
    rsh = [y.double() - xh @ a_h.double() for y in ysh]
    nsh = [nh, nh]
    lam_h = 1e-3
    s_h = len(xsh)
    z_h = sum(float((y.double() * gh.double()).square().sum()) / nn
              for y, nn in zip(ysh, nsh)) / s_h + EPS
    za_h = float(a_h.double().square().sum()) + EPS
    rho_h = lam_h / za_h
    if not 1e-8 < rho_h < 1e-6:
        raise RuntimeError('ill-conditioned test rho outside real regime: %g' % rho_h)
    m64 = sum((x.double().T @ x.double()) / (s_h * nn * z_h)
              for x, nn in zip(xsh, nsh))
    b64 = sum((x.double().T @ r.double()) / (s_h * nn * z_h)
              for x, r, nn in zip(xsh, rsh, nsh))
    m64 = (m64 + m64.T) * 0.5
    g2h = gh.double().square()
    probes = [0, 1, 37, 95]
    d_st, bases_h, info_st = stable_ridge_fit(xsh, ysh, rsh, gh, nsh, a_h, lam_h)
    probe_rel = []
    for j in probes:
        lhs = g2h[j] * m64 + rho_h * torch.eye(dh, device=dev, dtype=torch.float64)
        rhs = g2h[j] * b64[:, j]
        col = torch.linalg.solve(lhs, rhs)
        probe_rel.append(float((d_st[:, j] - col).norm() / col.norm().clamp_min(1e-30)))
    max_probe_rel = max(probe_rel)
    # F64 tolerances are condition-number-aware: kappa ~ g^2 smax^2/rho can
    # reach ~1e7 here, so eps64*kappa ~ 1e-9..1e-8; 1e-7 keeps >=10x margin
    # while staying >=4 orders below the FP32 artifact scale.
    if max_probe_rel > 1e-7:
        raise RuntimeError('stable solve vs raw-F64 reference mismatch: %g' % max_probe_rel)
    if info_st['cross_check_f64_eigh_rel_l2'] > 1e-7:
        raise RuntimeError('stable SVD vs F64 eigh cross-check mismatch')
    if info_st['D_numerical_rank'] > rank:
        raise RuntimeError('stable D rank exceeds CAL row rank')
    if info_st['D_out_of_rowspace_rel_fro'] > 1e-12:
        raise RuntimeError('stable D leaves the full sample space')
    d_old, mu_o, v_o, info_o = free_ridge_fit_fp32_original(
        [x.float() for x in xsh], ysh, [r.float() for r in rsh], gh, nsh, a_h, lam_h)
    oos_old = out_of_space_rel(d_old, bases_h['calgram'])
    old_probe_rel = []
    for j in probes:
        lhs = g2h[j] * m64 + rho_h * torch.eye(dh, device=dev, dtype=torch.float64)
        rhs = g2h[j] * b64[:, j]
        col = torch.linalg.solve(lhs, rhs)
        old_probe_rel.append(float((d_old[:, j].double() - col).norm() / col.norm().clamp_min(1e-30)))
    if oos_old <= 1e-4:
        raise RuntimeError('ill-conditioned test no longer discriminates the FP32 leak: %g' % oos_old)

    # --- ICC decomposition identities ---
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
    icc0 = icc_two_residual_decomposed(p, zl, zr, sl, sr, w1, b1, w2, b2)
    if icc0['zero_delta_max_abs'] != 0.0:
        raise RuntimeError('ICC zero-factor zero-delta not exact')
    if icc0['stage1_propagated_gated_sse'] != 0.0:
        raise RuntimeError('ICC zero-factor stage-1 term not exactly zero')
    reuse = float(((native_ffn(u0, w1, b1, w2, b2)[2].double()
                    - native_ffn(u1, w1, b1, w2, b2)[2].double())).square().sum())
    if abs(icc0['total_f64_gated_sse'] - reuse) > 1e-12 * max(reuse, 1e-30):
        raise RuntimeError('ICC zero-factor total must equal reuse-Y energy')
    fl = torch.randn(9, 3, device=dev, dtype=torch.bfloat16) * 0.1
    fr = torch.randn(3, d, device=dev, dtype=torch.bfloat16) * 0.1
    gl = torch.randn(d, 3, device=dev, dtype=torch.bfloat16) * 0.1
    gr = torch.randn(3, 9, device=dev, dtype=torch.bfloat16) * 0.1
    icc1 = icc_two_residual_decomposed(p, fl, fr, gl, gr, w1, b1, w2, b2)
    if icc1['decomposition_rel_residual'] > 1e-10:
        raise RuntimeError('ICC decomposition identity broken (nonzero factors)')

    # --- cheap stats: reuse-Y identity + fixed gate_fro2 ---
    a = torch.randn(d, d, device=dev, dtype=torch.bfloat16) * 0.1
    cs = cheap_site_stats(p, a, w1, b1, w2, b2)
    if abs(cs['reuse_y_gated_sse'] - cs['gated_dy_fro2']) > 1e-6 * max(cs['gated_dy_fro2'], 1e-12):
        raise RuntimeError('reuse-Y identity broken')
    if abs(cs['gate_fro2'] - float(gg.double().square().sum())) > 0:
        raise RuntimeError('gate_fro2 must equal the (n,D) gate energy without extra n')

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
            'illcond_rho': rho_h,
            'stable_vs_raw_f64_probe_max_rel_l2': max_probe_rel,
            'stable_cross_check_rel_l2': info_st['cross_check_f64_eigh_rel_l2'],
            'stable_D_rank': info_st['D_numerical_rank'],
            'stable_out_of_rowspace_rel': info_st['D_out_of_rowspace_rel_fro'],
            'stable_out_of_calgram_rel': info_st['D_out_of_calgram_rel_fro'],
            'original_fp32_out_of_rowspace_rel': oos_old,
            'original_fp32_probe_max_rel_l2': max(old_probe_rel),
            'icc_zero_factor_zero_delta_exact': True,
            'icc_zero_factor_stage1_zero_exact': True,
            'icc_decomposition_identity_nonzero': True,
            'reuse_y_identity': True,
            'gate_fro2_fixed': True,
            'zero_matrix_replay_anchor_exact': True}