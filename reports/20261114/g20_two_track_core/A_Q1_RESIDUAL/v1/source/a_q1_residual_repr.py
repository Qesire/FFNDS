"""A Q1-residual four-arm representation math (NEW author code, rev5 handoff A2-A4).

Four fixed arms around the exact published Q1 (no refit):
    CURRENT_Q1            pred = X A_q1
    Q1_PLUS_LINEAR64_Q32  pred = X A_q1 + (X P64 / gamma_L) C_L Q32^T
    Q1_PLUS_DELTA_N32     pred = X A_q1 + Phi_D C_D Q32^T,  Phi_D=[h, tanh(h/sigma)], h=X P32
    Q1_PLUS_ENDPOINT_N32  pred = X A_q1 + Phi_E C_E Q32^T,
                          Phi_E=[h, tanh((U1 P32-b)/sigma)-tanh((U0 P32-b)/sigma)]

Reuses VERBATIM:
  ridge_fit_core.bf16_replay / _f64_sse / native_ffn (R1b sha b22ac372).
  The V4 fit_readout sum-of-Kronecker normal-equation solver body (v4_repr sha 8ead06e5),
  with base A0 replaced by the rebuilt A_q1 and features replaced by the fixed RMS-normalized
  raw features below. rho = READOUT_EPS * trace(K) / 2048 (relative Tikhonov, unchanged).

Key rev5 coordinate choice (A2.2): Q32 is the top-32 right singular basis of the PRE-GATE
Q1 residual (Y - X A_q1), NOT the gated residual. The true gate weights stay in the
fit_readout normal equations. P64 is the top-64 right singular basis of weighted stacked X.

Numeric contract: TF32 off (caller); A_q1 rebuilt FP32-then-F64 (Problem.matrix order);
SVD / readout / error reductions in F64; BF16 replay is a separate local cache/operator/gate
path (A4), not a video/timing claim.
"""
import math

import torch
import torch.nn.functional as F

from ridge_fit_core import _f64_sse, bf16_replay, native_ffn

D = 5120
ARMS = ('CURRENT_Q1', 'Q1_PLUS_LINEAR64_Q32', 'Q1_PLUS_DELTA_N32', 'Q1_PLUS_ENDPOINT_N32')
RESIDUAL_ARMS = ('Q1_PLUS_LINEAR64_Q32', 'Q1_PLUS_DELTA_N32', 'Q1_PLUS_ENDPOINT_N32')
READOUT_EPS = 1e-6
R_IN = 64
R_OUT = 32


def source_weight(n_sources, n_rows):
    return 1.0 / (n_sources * n_rows)


def Qb(x):
    return x.float().bfloat16()


def rebuild_a_q1(w1, w2, alpha_q1_f32):
    """A2.1: FP32 formation (W1^T diag(alpha)) W2^T then F64 lift; Problem.matrix order.
    w1/w2 BF16; alpha_q1_f32 FP32 [13824]. Returns (A_q1 F64, A_q1_f32 FP32)."""
    alpha = alpha_q1_f32.float()
    a_f32 = (w1.float().T * alpha) @ w2.float().T
    return a_f32.double(), a_f32


def select_bases(cal, a_q1, r_in=R_IN, r_out=R_OUT):
    """P64: top-r_in right singular vectors of weighted stacked X (F64).
    Q32: top-r_out right singular vectors of weighted stacked PRE-GATE residual Y-X A_q1.
    P32 = P64[:, :32]. No DEV, no random projection, no extra basis candidates."""
    s = len(cal)
    xw = torch.cat([c['X'].double() * math.sqrt(source_weight(s, c['n'])) for c in cal], 0)
    rw = torch.cat([(c['Y'].double() - c['X'].double() @ a_q1) * math.sqrt(source_weight(s, c['n']))
                    for c in cal], 0)
    _, _, vh_x = torch.linalg.svd(xw, full_matrices=False)
    _, _, vh_r = torch.linalg.svd(rw, full_matrices=False)
    p64 = vh_x[:r_in].T.contiguous()
    q32 = vh_r[:r_out].T.contiguous()
    return p64, q32


def center_scale(cal, p32):
    """A2.2 fixed endpoint center/scale. b = pooled mean of U0@P32 and U1@P32 projections;
    sigma = pooled std around b, clamp_min(1e-6). Source-equal/row weights w_s=1/(S*n_s)."""
    s = len(cal)
    dim = p32.shape[1]
    dev = p32.device
    b = torch.zeros(dim, dtype=torch.float64, device=dev)
    for c in cal:
        w = source_weight(s, c['n'])
        h0 = c['u0'].double() @ p32
        h1 = c['u1'].double() @ p32
        b += w * (h0.sum(0) + h1.sum(0)) / 2
    var = torch.zeros(dim, dtype=torch.float64, device=dev)
    for c in cal:
        w = source_weight(s, c['n'])
        h0 = c['u0'].double() @ p32
        h1 = c['u1'].double() @ p32
        var += w * (((h0 - b).square()).sum(0) + ((h1 - b).square()).sum(0)) / 2
    sigma = var.sqrt().clamp_min(1e-6)
    return b, sigma


def feature_raw(arm, c, p64, p32, b, sigma):
    """A2.3 raw features (n x 64) F64. D and E share the identical first 32 columns X@P32.
    Delta tanh has no b offset (odd); Endpoint uses centered endpoint tanh difference."""
    x = c['X'].double()
    if arm == 'Q1_PLUS_LINEAR64_Q32':
        return x @ p64
    h = x @ p32
    if arm == 'Q1_PLUS_DELTA_N32':
        return torch.cat([h, torch.tanh(h / sigma)], 1)
    if arm == 'Q1_PLUS_ENDPOINT_N32':
        h0 = c['u0'].double() @ p32
        h1 = c['u1'].double() @ p32
        return torch.cat([h, torch.tanh((h1 - b) / sigma) - torch.tanh((h0 - b) / sigma)], 1)
    raise ValueError('no feature map for ' + arm)


def feature_gamma(arm, cal, p64, p32, b, sigma):
    """A2.3 per-arm deterministic CAL RMS rule: gamma_j = sqrt(sum_s w_s sum_rows Phi_raw^2),
    clamp_min(1e-6). No feature-mean subtraction (preserves zero-increment / antisymmetry)."""
    s = len(cal)
    dim = R_IN
    dev = p64.device
    acc = torch.zeros(dim, dtype=torch.float64, device=dev)
    for c in cal:
        w = source_weight(s, c['n'])
        phi = feature_raw(arm, c, p64, p32, b, sigma)
        acc += w * phi.square().sum(0)
    return acc.sqrt().clamp_min(1e-6)


def fit_readout(arm, cal, a_q1, p64, p32, q32, b, sigma, gamma, eps=READOUT_EPS):
    """A3: V4 fit_readout solver body verbatim, base A_q1, RMS-normalized features.
    K += kron(w Phi^T Phi, Q^T diag(g^2) Q); T += w Phi^T ((R g^2) Q), R=Y-X A_q1.
    rho = eps*trace(K)/2048; vec(C)=solve(K+rho I, vec(T)). Source weight appears ONCE.
    Returns C (64x32 F64) + diagnostics. trace(K)=0 & RHS=0 -> C=0 NO_FEATURE_SIGNAL."""
    s = len(cal)
    k = R_IN
    r_out = R_OUT
    dim = k * r_out
    dev = a_q1.device
    kern = torch.zeros((dim, dim), dtype=torch.float64, device=dev)
    tmat = torch.zeros((k, r_out), dtype=torch.float64, device=dev)
    for c in cal:
        w = source_weight(s, c['n'])
        phi = feature_raw(arm, c, p64, p32, b, sigma) / gamma
        res = c['Y'].double() - c['X'].double() @ a_q1
        g2 = c['g'].double().square()
        gphi = w * (phi.T @ phi)
        gq = q32.T @ (g2[:, None] * q32)
        kern += torch.kron(gphi, gq)
        tmat += w * (phi.T @ ((res * g2) @ q32))
    trace_k = float(torch.diagonal(kern).sum())
    rhs = tmat.reshape(-1)
    if trace_k == 0.0 and float(rhs.norm()) == 0.0:
        cout = torch.zeros((k, r_out), dtype=torch.float64, device=dev)
        return cout, {'readout_ridge_rho': 0.0, 'readout_normal_relative_residual': 0.0,
                      'readout_dim': dim, 'k': k, 'r_out': r_out, 'status': 'NO_FEATURE_SIGNAL'}
    rho = eps * trace_k / dim
    kern += rho * torch.eye(dim, dtype=torch.float64, device=dev)
    cvec = torch.linalg.solve(kern, rhs)
    cout = cvec.reshape(k, r_out)
    resid = float((kern @ cvec - rhs).norm() / rhs.norm().clamp_min(1e-30))
    info = {'readout_ridge_rho': rho, 'readout_normal_relative_residual': resid,
            'readout_dim': dim, 'k': k, 'r_out': r_out, 'status': 'SOLVED'}
    return cout, info


def predict_math(arm, c, a_q1, p64, p32, q32, b, sigma, gamma, cout):
    """A2.3 F64 math prediction. C=0 baseline is the identical Q1."""
    if arm == 'CURRENT_Q1':
        return c['X'].double() @ a_q1
    phi = feature_raw(arm, c, p64, p32, b, sigma) / gamma
    return c['X'].double() @ a_q1 + phi @ cout @ q32.T


def gated_mse(pred, target, g, n_rows):
    return float(_f64_sse(pred, target, g)) / (n_rows * D)


def cal_objective(cal, arm, a_q1, p64, p32, q32, b, sigma, gamma, cout, rho):
    """A5 check two: J(C)=sum_s w_s ||g*(X A_q1 + Phi C Q^T - Y)||_F^2 + rho||C||_F^2.
    Returns (J0=J(C=0), Jfit=J(C), unregularized CAL total risk at C). Aggregate only;
    per-CAL-source improvement is NOT required."""
    s = len(cal)
    j0 = 0.0
    jfit_data = 0.0
    for c in cal:
        w = source_weight(s, c['n'])
        g = c['g'].double()
        base = c['X'].double() @ a_q1
        res0 = (base - c['Y'].double()) * g
        j0 += w * float(res0.square().sum())
        phi = feature_raw(arm, c, p64, p32, b, sigma) / gamma
        pred = base + phi @ cout @ q32.T
        resf = (pred - c['Y'].double()) * g
        jfit_data += w * float(resf.square().sum())
    reg = rho * float(cout.square().sum())
    return j0, jfit_data + reg, jfit_data


# ---- A4 BF16 local replay paths (cache/operator/gate; no residual-stream writeback) ----

def bf16_q1(c, a_q1_f32):
    """Q1: original bf16_replay with Qb(A_q1_f32). Returns (base, gated, target)."""
    a_bf16 = Qb(a_q1_f32)
    return bf16_replay(c['u0'].bfloat16(), c['u1'].bfloat16(),
                       c['y0'].bfloat16(), c['y1'].bfloat16(), c['g'].float(), a_bf16)


def bf16_l64(c, a_q1, p64, q32, gamma_l, c_l):
    """L64: fold full linear residual into A_q1, one Qb, original residual replay."""
    a_l_math = a_q1 + (p64 / gamma_l[None, :]) @ c_l @ q32.T
    a_l_bf16 = Qb(a_l_math)
    return bf16_replay(c['u0'].bfloat16(), c['u1'].bfloat16(),
                       c['y0'].bfloat16(), c['y1'].bfloat16(), c['g'].float(), a_l_bf16)


def _linear_fold_bf16(a_q1, p32, q32, gamma, c_lin):
    a_lin = a_q1 + (p32 / gamma[:32][None, :]) @ c_lin[:32] @ q32.T
    return Qb(a_lin)


def bf16_endpoint(c, a_q1, p64, p32, q32, b, sigma, gamma_e, c_e):
    """Endpoint: fold first-32 linear; nonlinear centered endpoint tanh difference branch.
    Anchor contribution lives in the single residual cache (no extra resident h0)."""
    a_lin_bf16 = _linear_fold_bf16(a_q1, p32, q32, gamma_e, c_e)
    c_n_bf16 = Qb(c_e[32:] / gamma_e[32:, None])
    p32_bf16 = Qb(p32)
    q32_bf16 = Qb(q32)
    b_fp32 = b.float()
    sigma_fp32 = sigma.float()
    g = c['g'].float()
    u0 = c['u0'].bfloat16()
    u1 = c['u1'].bfloat16()
    y0 = c['y0'].bfloat16()
    y1 = c['y1'].bfloat16()

    def endpoint_b(u):
        h = u @ p32_bf16
        v = torch.tanh((h.float() - b_fp32) / sigma_fp32).bfloat16()
        n = (v @ c_n_bf16) @ q32_bf16.T
        return Qb((u @ a_lin_bf16).float() + n.float())

    cache = Qb(y0.float() - endpoint_b(u0).float())
    base = Qb(endpoint_b(u1).float() + cache.float())
    gated = Qb(base.float() * g)
    target = Qb(y1.float() * g)
    return base, gated, target


def bf16_delta(c, a_q1, p64, p32, q32, b, sigma, gamma_d, c_d):
    """Delta-only: fold first-32 linear; retain anchor projection h0=u0@P32; odd tanh of
    endpoint projection difference (no b). Not expressible as a single-endpoint B(u) diff."""
    a_lin_bf16 = _linear_fold_bf16(a_q1, p32, q32, gamma_d, c_d)
    c_n_bf16 = Qb(c_d[32:] / gamma_d[32:, None])
    p32_bf16 = Qb(p32)
    q32_bf16 = Qb(q32)
    sigma_fp32 = sigma.float()
    g = c['g'].float()
    u0 = c['u0'].bfloat16()
    u1 = c['u1'].bfloat16()
    y0 = c['y0'].bfloat16()
    y1 = c['y1'].bfloat16()
    base, _, _ = bf16_replay(u0, u1, y0, y1, g, a_lin_bf16)
    h0 = u0 @ p32_bf16
    h1 = u1 @ p32_bf16
    v = torch.tanh((h1.float() - h0.float()) / sigma_fp32).bfloat16()
    n = (v @ c_n_bf16) @ q32_bf16.T
    base = Qb(base.float() + n.float())
    gated = Qb(base.float() * g)
    target = Qb(y1.float() * g)
    return base, gated, target


def bf16_predict(arm, c, params):
    """Dispatch BF16 replay for an arm using a reloaded parameter dict. Returns (base,gated,target)."""
    a_q1 = params['A_q1']
    a_q1_f32 = params['A_q1_f32']
    p64 = params['P64']
    p32 = p64[:, :32]
    q32 = params['Q32']
    b = params['b']
    sigma = params['sigma']
    if arm == 'CURRENT_Q1':
        return bf16_q1(c, a_q1_f32)
    if arm == 'Q1_PLUS_LINEAR64_Q32':
        return bf16_l64(c, a_q1, p64, q32, params['gamma_L'], params['C_L'])
    if arm == 'Q1_PLUS_DELTA_N32':
        return bf16_delta(c, a_q1, p64, p32, q32, b, sigma, params['gamma_D'], params['C_D'])
    if arm == 'Q1_PLUS_ENDPOINT_N32':
        return bf16_endpoint(c, a_q1, p64, p32, q32, b, sigma, params['gamma_E'], params['C_E'])
    raise ValueError('no BF16 path for ' + arm)