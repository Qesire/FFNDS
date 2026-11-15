"""B common replay (NEW author code, rev5 handoff B2-B5). Shared loader, own-W
native targets, and the three fixed method calls (Q1 / ICC / GLOBAL_P64) plus F64
risk scoring. Both weights W_O and W_D recompute their OWN anchor/target on the
identical 512-row native_ffn recipe; no mixing of old full-M D target with sampled
O target. G_ref is the same-source/site D4 hit gate, fixed across W and mu.

Reuses VERBATIM:
  ridge_fit_core.native_ffn / bf16_replay / _f64_sse (R1b sha b22ac372).
  v4_repr.q0_prior / fit_q1 (V4 sha 8ead06e5) copied below unchanged for Q1.
  icc_decompose.factorize + icc_ffn.Factors (ordinary SVD r512, gesvd, BF16 factors).
  chipmunk_fullm_proxy_math.fixed_prefix_permutation / rms_score / selection.
  chipmunk_wan_packed_pregate.packed_pregate.

Primary risk (B3): sum(((pred_BF16 - own_y1_BF16)*G_ref)^2)/(512*5120) in F64,
pred the real BF16 PRE-GATE prediction. Auxiliary: F64 MSE of BF16(pred*G_ref) vs
BF16(own_y1*G_ref). This is sampled local FFN function replay, not full-M video/sys.
"""
import math

import torch
import torch.nn.functional as F

from ridge_fit_core import _f64_sse, bf16_replay, native_ffn
from q1_fit_core import Problem
from icc_decompose import factorize
from icc_ffn import Factors
from chipmunk_fullm_proxy_math import fixed_prefix_permutation, rms_score, selection
from chipmunk_wan_packed_pregate import packed_pregate

D = 5120
M_FULL = 31668
RANK_ICC = 512
P64_K = 5120
SITES = (('high', 4), ('high', 19), ('high', 34), ('low', 4), ('low', 19), ('low', 34))
EXPERT_STEP = {'high': (0, 1), 'low': (2, 3)}
Q1_RIDGE = 1e-3
Q1_MAXITER = 240
Q1_TOL = 1e-5


def source_weight(n_sources, n_rows):
    return 1.0 / (n_sources * n_rows)


def site_key(expert, layer):
    return '%s%02d' % (expert, layer)


# ---- Q1 reuse (verbatim v4_repr.q0_prior / fit_q1) ----

def q0_prior(cal, w1, b1, w2):
    """Fresh per-config Q0 prior. U/W1/b1 lifted FP32, FP32 addmm + GELU(tanh),
    dz/dh lifted F64 for weighted moments; den clamp1e-30, alpha FP32, A0 F64."""
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


def fit_q1(cal, w1, w2, a0, alpha_q0, lam=Q1_RIDGE, maxiter=Q1_MAXITER, tol=Q1_TOL):
    """Q1 via R2 Problem PCG (verbatim). Fresh fit, init alpha_Q0. Returns A_Q1 F64,
    alpha FP32, info. Caller requires status CONVERGED + true_relative_residual<=tol."""
    sources = [{'x': c['X'].float(), 'y': c['Y'].float(), 'g': c['g'].float()} for c in cal]
    prob = Problem(w1, w2, a0.float(), sources, ridge=lam)
    alpha, info = prob.fit(alpha_q0.clone(), maxiter, tol)
    a_q1 = prob.matrix(alpha).double()
    return a_q1, alpha, info


# ---- loader + own-W targets ----

def load_site(bank_root, source, step, layer):
    name = '%s__seed42__s%02d__b%02d.pt' % (source, step, layer)
    d = torch.load(bank_root / source / name, map_location='cpu', weights_only=True)
    g = d['gate'].reshape(-1)
    return d['U'], d['Y_pre_gate'], g


def build_pack(bank_root, sources, expert, layer, device):
    """Load PANEL512 U0/U1 (BF16) and the D4 hit gate G_ref. X = U1-U0 (FP32).
    own-W y0/y1 and Y are filled by own_w_targets per weight."""
    a_step, h_step = EXPERT_STEP[expert]
    pack = []
    for src in sources:
        u0, _, _ = load_site(bank_root, src, a_step, layer)
        u1, _, g = load_site(bank_root, src, h_step, layer)
        u0b = u0[:512].to(device=device, dtype=torch.bfloat16)
        u1b = u1[:512].to(device=device, dtype=torch.bfloat16)
        gf = g.to(device=device, dtype=torch.float32)
        u0f = u0b.float()
        u1f = u1b.float()
        pack.append({'source': src, 'u0_bf16': u0b, 'u1_bf16': u1b,
                     'u0': u0f, 'u1': u1f, 'X': u1f - u0f, 'g': gf, 'n': u0b.shape[0]})
    return pack


def own_w_targets(pack, w1, b1, w2, b2):
    """B3: each W recomputes own anchor/hit on identical 512-row native_ffn (BF16).
    Fills z0/h0/y0/z1/h1/y1 (BF16) and Y=y1-y0 (FP32). Returns pack."""
    for c in pack:
        z0, h0, y0 = native_ffn(c['u0_bf16'], w1, b1, w2, b2)
        z1, h1, y1 = native_ffn(c['u1_bf16'], w1, b1, w2, b2)
        c['z0'], c['h0'], c['y0'] = z0, h0, y0
        c['z1'], c['h1'], c['y1'] = z1, h1, y1
        c['Y'] = y1.float() - y0.float()
    return pack


# ---- ICC ----

def icc_predict(c, fc1_left, fc1_right, fc2_left, fc2_right):
    """B4 ICC two-residual prediction accepting already-computed own-W z0/h0/y0.
    Arithmetic copied unchanged from icc_two_residual_decomposed; BF16 ops."""
    def first(x):
        return F.linear(F.linear(x, fc1_right), fc1_left)

    def second(x):
        return F.linear(F.linear(x, fc2_right), fc2_left)

    q1 = c['z0'] - first(c['u0_bf16'])
    q2 = c['y0'] - second(c['h0'])
    z_pred = first(c['u1_bf16']) + q1
    h_pred = F.gelu(z_pred, approximate='tanh')
    pred = second(h_pred) + q2
    if not torch.isfinite(pred).all():
        raise ValueError('nonfinite ICC prediction')
    return pred


def make_w_o_factors(w1, w2, device):
    """Ordinary SVD r512 of W_O W1/W2 (BF16 weights -> FP32 gesvd), split left=U*S,
    right=Vh, cast BF16. Same fixed scheme as the W_D factor producer; no CD-SVD."""
    f1 = factorize(w1, RANK_ICC, driver='gesvd').to(device=device, dtype=torch.bfloat16)
    f2 = factorize(w2, RANK_ICC, driver='gesvd').to(device=device, dtype=torch.bfloat16)
    return f1, f2


def load_w_d_factors(path, device):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    fc1, fc2 = obj['fc1'], obj['fc2']
    return (fc1['left'].to(device), fc1['right'].to(device),
            fc2['left'].to(device), fc2['right'].to(device)), obj


# ---- Chip GLOBAL_P64 (OWNW_LOCAL512_P64ANCHOR64_V1) ----

def chip_fixed64():
    return fixed_prefix_permutation(M_FULL, 42)


def chip_predict(c, w1, b1, w2, u0_64, u1_64, fixed64):
    """B5: own-W eval512 anchor cache; selector from fixed64 own-W proxy; packed
    pre-gate action on eval512. u0_64/u1_64 are the fixed64 rows (BF16) for this
    source/site/endpoint. Returns (pred, indices)."""
    h0_sel = F.gelu(F.linear(u0_64, w1, b1), approximate='tanh')   # own-W anchor64 hidden
    z1_proxy = F.linear(u1_64, w1, b1)                              # hit64 preact
    scores = rms_score(z1_proxy, h0_sel)                            # rms_score does GELU once
    indices = selection(scores, P64_K)
    pred = packed_pregate(c['u1_bf16'], w1, b1, w2, c['h0'], c['y0'], indices)
    return pred, indices


# ---- risk scoring ----

def primary_risk(pred_bf16, c):
    """B3 primary: F64 gated SSE/(512*5120), pred real BF16 pre-gate vs own y1 BF16."""
    d = (pred_bf16.double() - c['y1'].double()) * c['g'].double()
    v = d.square().sum()
    if not torch.isfinite(v):
        raise ValueError('nonfinite primary risk')
    return float(v) / (c['n'] * D)


def auxiliary_risk(pred_bf16, c):
    """B3 auxiliary: F64 MSE between BF16(pred*G_ref) and BF16(own_y1*G_ref)."""
    pg = (pred_bf16.float() * c['g']).bfloat16()
    tg = (c['y1'].float() * c['g']).bfloat16()
    return float(_f64_sse(pg.float(), tg.float(), torch.ones_like(c['g']))) / (c['n'] * D)


def q1_predict(c, a_q1_f32):
    """Q1 pre-gate base via original bf16_replay with Qb(A_q1_f32)."""
    a_bf16 = a_q1_f32.float().bfloat16()
    base, _, _ = bf16_replay(c['u0_bf16'], c['u1_bf16'], c['y0'], c['y1'], c['g'], a_bf16)
    return base