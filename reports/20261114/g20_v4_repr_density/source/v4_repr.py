"""V4 representation / density fitting math (NEW author code).

Reuses verbatim:
  q1_fit_core.Problem        (R2 matrix-prior PCG, sha b76a0efa) for CURRENT_Q1.
  ridge_fit_core._f64_sse / native_ffn (R1b, sha b22ac372) for scoring / Q0 hidden state.

Frozen V4 contract (see ROOT_SELF_AUDIT):
  * A0 = fresh per-config Q0 prior A(alpha_Q0), alpha_Q0 the diagonal hidden-match
    ratio E[dZ dH]/E[dZ^2] on the CAL training subset (user adjudicated 2026-11-14).
    Common base shared by every representation within a config; anti-leakage clean.
  * CURRENT_Q1: Problem PCG fit of alpha (maxiter 240, tol 1e-5, ridge 1e-3) toward
    A0, init alpha_Q0. Fresh fit -> NO R2 old/new per-layer selection (no old solution).
  * STABLE_FREE_LINEAR: free DxD correction D on the Q1 residual, ridge toward 0,
    rho = lam/||A_Q1||^2. Reuses ridge_fit_core.stable_ridge_fit VERBATIM (F64 weighted
    thin-SVD direct solve + F64 eigh cross-check). The bank gate c_gate_msa is timestep
    modulation, per-channel and SHARED across sources at a site (verified empirically;
    common_gate() asserts this, mirroring the R1b runner). A_free = A_Q1 + D.
  * L32 / DELTA_N32 / ENDPOINT_N32 / L64_Q32: B(u)=uA0+(uP)C_L Q^T+tanh(uP)C_N Q^T.
    P/Q FIXED by CAL data SVD (top-r right singular vectors); readout C fitted convexly
    (gated least squares, sum-of-Kronecker normal system, F64). bias b=0, no center
    offset -> odd symmetry: zero displacement->zero, endpoint swap->sign flip.
      L32        : Phi = X P32                       (linear only, k=32)
      L64_Q32    : Phi = X P64                       (linear only, k=64; readout 64x32
                                                      = 2048 matches N methods C_L+C_N)
      DELTA_N32  : Phi = [X P32, tanh(X P32)]        (k=64; X = u_t - u_s)
      ENDPOINT_N32: Phi = [u_t P32 - u_s P32, tanh(u_t P32) - tanh(u_s P32)]  (k=64)
  * Primary score = F64 gated pre-gate increment SSE/(rows*5120) (math lift).
    BF16 cache/gate replay is a SEPARATE V4D check, not the V4 primary.

Numeric contract: TF32 off; Q1 PCG in FP32 (R2 verbatim); SVD / readout / free-ridge /
error reductions in FP64. Inputs U/Y are bf16 lifted to FP32/FP64.
"""
import math

import torch
import torch.nn.functional as F

from q1_fit_core import Problem
from ridge_fit_core import _f64_sse, native_ffn, stable_ridge_fit

D = 5120
MODELS = ('CURRENT_Q1', 'STABLE_FREE_LINEAR', 'L32', 'DELTA_N32', 'ENDPOINT_N32', 'L64_Q32')
READOUT_EPS = 1e-6          # relative Tikhonov floor on the convex readout normal system


def source_weight(n_sources, n_rows):
    return 1.0 / (n_sources * n_rows)


def q0_prior(cal, w1, b1, w2):
    """Fresh per-config Q0 prior. cal: list of {u0,u1,X,Y,g,n}. Returns alpha_Q0 (fp32),
    A0 (fp64 dxd), Z (fp64 scalar). Hidden increment dH uses tanh-GELU of native z."""
    s = len(cal)
    m = w1.shape[0]
    num = torch.zeros(m, dtype=torch.float64, device=w1.device)
    den = torch.zeros(m, dtype=torch.float64, device=w1.device)
    z_norm = 0.0
    w1f, b1f = w1.float(), b1.float()
    for c in cal:
        w = source_weight(s, c['n'])
        u0, u1 = c['u0'].float(), c['u1'].float()
        z0 = torch.addmm(b1f, u0, w1f.T)
        z1 = torch.addmm(b1f, u1, w1f.T)
        dz = (z1 - z0).double()
        dh = (F.gelu(z1, approximate='tanh') - F.gelu(z0, approximate='tanh')).double()
        num += w * (dz * dh).sum(0)
        den += w * dz.square().sum(0)
        yg = (c['Y'].double() * c['g'].double())
        z_norm += w * float(yg.square().sum())
    alpha = (num / den.clamp_min(1e-30))
    a0 = (w1.double().T * alpha) @ w2.double().T
    return alpha.float(), a0, z_norm + 1e-12


def select_pq(cal, a0, r_in=64, r_out=32):
    """Fixed CAL-data SVD feature bases. P: top-r_in right singular vectors of weighted
    stacked X; Q: top-r_out of weighted stacked gated residual (Y - X A0) odot g.
    Returns P (d x r_in), Q (d x r_out), fp64. Nested: P32 = P[:, :32]."""
    s = len(cal)
    xw = torch.cat([c['X'].double() * math.sqrt(source_weight(s, c['n'])) for c in cal], 0)
    rw = torch.cat([((c['Y'].double() - c['X'].double() @ a0) * c['g'].double())
                    * math.sqrt(source_weight(s, c['n'])) for c in cal], 0)
    _, _, vh_x = torch.linalg.svd(xw, full_matrices=False)
    _, _, vh_r = torch.linalg.svd(rw, full_matrices=False)
    p = vh_x[:r_in].T.contiguous()
    q = vh_r[:r_out].T.contiguous()
    return p, q


def feature_map(model, c, p, q):
    """Phi (n x k) fp64 for a source. p is d x 64 (nested), q is d x 32."""
    x = c['X'].double()
    if model == 'L32':
        return x @ p[:, :32]
    if model == 'L64_Q32':
        return x @ p[:, :64]
    if model == 'DELTA_N32':
        h = x @ p[:, :32]
        return torch.cat([h, torch.tanh(h)], 1)
    if model == 'ENDPOINT_N32':
        ht = c['u1'].double() @ p[:, :32]
        hs = c['u0'].double() @ p[:, :32]
        return torch.cat([ht - hs, torch.tanh(ht) - torch.tanh(hs)], 1)
    raise ValueError('no readout feature map for ' + model)


def fit_readout(model, cal, a0, p, q, eps=READOUT_EPS):
    """Convex gated least squares for C (k x r_out). Normal system is the sum over
    sources of kron(GPhi_c, GQ_c) (per-source gates break single-Kronecker separability).
    Returns C fp64 and diagnostics."""
    s = len(cal)
    k = feature_map(model, cal[0], p, q).shape[1]
    r_out = q.shape[1]
    dim = k * r_out
    dev = a0.device
    kern = torch.zeros((dim, dim), dtype=torch.float64, device=dev)
    tmat = torch.zeros((k, r_out), dtype=torch.float64, device=dev)
    for c in cal:
        w = source_weight(s, c['n'])
        phi = feature_map(model, c, p, q)
        res = c['Y'].double() - c['X'].double() @ a0
        g2 = c['g'].double().square()
        # source weight w appears ONCE in the Kronecker product (on GPhi); putting it on
        # both factors would scale the normal matrix by w^2 and C by 1/w = S*n.
        gphi = w * (phi.T @ phi)
        gq = q.T @ (g2[:, None] * q)
        kern += torch.kron(gphi, gq)
        tmat += w * (phi.T @ ((res * g2) @ q))
    rho = eps * float(torch.diagonal(kern).sum()) / dim
    kern += rho * torch.eye(dim, dtype=torch.float64, device=dev)
    rhs = tmat.reshape(-1)
    cvec = torch.linalg.solve(kern, rhs)
    cout = cvec.reshape(k, r_out)
    resid = float((kern @ cvec - rhs).norm() / rhs.norm().clamp_min(1e-30))
    info = {'readout_ridge_rho': rho, 'readout_normal_relative_residual': resid,
            'readout_dim': dim, 'k': k, 'r_out': r_out}
    return cout, info


def predict_repr(model, c, a0, p, q, cout):
    phi = feature_map(model, c, p, q)
    return c['X'].double() @ a0 + phi @ cout @ q.T


def fit_q1(cal, w1, w2, a0, alpha_q0, lam, maxiter=240, tol=1e-5):
    """CURRENT_Q1 via R2 Problem PCG (verbatim). Fresh fit, init alpha_Q0, no old/new
    selection. Returns A_Q1 (fp64), alpha (fp32), info."""
    sources = [{'x': c['X'].float(), 'y': c['Y'].float(), 'g': c['g'].float()} for c in cal]
    prob = Problem(w1, w2, a0.float(), sources, ridge=lam)
    alpha, info = prob.fit(alpha_q0.clone(), maxiter, tol)
    a_q1 = prob.matrix(alpha).double()
    return a_q1, alpha, info


def common_gate(cal):
    """The bank gate c_gate_msa is timestep modulation: per-channel and SHARED across
    sources at a site (verified empirically on the V2 bank; mirrors the R1b runner
    assertion). Return the common gate and fail if any source differs."""
    g0 = cal[0]['g']
    for c in cal:
        if not torch.equal(c['g'], g0):
            raise RuntimeError('gate not per-channel shared across sources at site')
    return g0


def fit_stable_free_linear(cal, a_q1, lam):
    """Free DxD correction D on the Q1 residual, ridge toward 0 (rho=lam/||A_Q1||^2).
    Reuses ridge_fit_core.stable_ridge_fit VERBATIM (R1b sha b22ac372): F64 weighted
    thin-SVD sample-space DIRECT solve + F64 eigh cross-check, common per-channel gate.
    A_free = A_Q1 + D. Returns D (fp64), info with cross-check / row-space certificates."""
    g = common_gate(cal)
    x_list = [c['X'] for c in cal]
    y_list = [c['Y'] for c in cal]
    r_list = [c['Y'].double() - c['X'].double() @ a_q1 for c in cal]
    n_list = [c['n'] for c in cal]
    d64, bases, info = stable_ridge_fit(x_list, y_list, r_list, g, n_list, a_q1, lam)
    if info['cross_check_f64_eigh_rel_l2'] > 1e-7:
        raise RuntimeError('stable solver cross-check failed')
    if info['D_out_of_rowspace_rel_fro'] > 1e-10:
        raise RuntimeError('stable D outside full sample space')
    return d64, info


def gated_mse(pred, target, g, n_rows):
    return float(_f64_sse(pred, target, g)) / (n_rows * D)