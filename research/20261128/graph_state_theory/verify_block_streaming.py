"""CPU algebra witness for a P64-first, row-tiled Wan-shaped block graph.

The frozen attention projection matrices are real CAL assets, hash-checked.
All hidden states, Q/K/V and FFN weights are explicitly synthetic. This is
neither a Wan inference run nor a performance/quality measurement. Float64
checks establish algebraic identities, not production BF16 equivalence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def norm(x, eps=1e-6):
    c = x - x.mean(axis=-1, keepdims=True)
    return c / np.sqrt((c * c).mean(axis=-1, keepdims=True) + eps)


def gelu(x):
    return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x**3)))


def softmax(x):
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cal-dir', type=Path, required=True)
    ap.add_argument('--pv', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    expected = {'high': '8d6f7651ce40d2f6f389ba7f981c573a84b2b3d3d9c8aa1fbeae95d9914ef864',
                'low': '04c15a63f6e4ffd34b61714a1be79d2651c195d104e4007aa5c8c22e2dd356e1'}
    assert digest(args.pv) == 'cfc5a6febf99b37f04ba2989038546bda0d404d9e15db659793bbc88125ca3ff'
    pv = np.load(args.pv)[:, :64].astype(np.float64)
    readers, cal_report = {}, {}
    for phase, key in [('high', 'out_current'), ('low', 'out_delta')]:
        path = args.cal_dir / f'self_{phase}_transform.npz'
        assert digest(path) == expected[phase]
        with np.load(path) as data:
            readers[phase] = pv @ data[key + '_basis'].astype(np.float64)
            sv = np.linalg.svd(readers[phase], compute_uv=False)
            spectrum = data[key + '_spectrum'].astype(np.float64)
            cal_report[phase] = dict(sha256=digest(path), reader_shape=list(readers[phase].shape),
                                    rank=int(np.linalg.matrix_rank(readers[phase])),
                                    condition=float(sv[0] / sv[-1]),
                                    normalized_feature_spectrum_top31_fraction=float(spectrum[:31].sum() / spectrum.sum()))

    rng = np.random.default_rng(20261128)
    n, h, dh, p, modes, width, selected_width = 257, 2, 128, 64, 16, 640, 128
    d = h * dh
    probes = np.sort(rng.choice(n, p, replace=False))
    rest = np.setdiff1d(np.arange(n), probes)
    def weight(a, b):
        return rng.normal(size=(a, b)) / np.sqrt(a)
    hidden = rng.normal(size=(n, d))
    hidden_anchor = hidden + .3 * rng.normal(size=(n, d))
    wq, wk, wv, wo = [weight(d, d) for _ in range(4)]
    cq, co = weight(d, d), weight(d, d)
    ck, cv = rng.normal(size=(h, 23, dh)), rng.normal(size=(h, 23, dh))
    g1, g2 = rng.uniform(.05, .3, size=(2, d))
    fscale, fshift = 1 + .05 * rng.normal(size=d), .05 * rng.normal(size=d)
    w1, w2 = weight(d, width), weight(width, d)
    w1k, w2k = w1[:, :selected_width], w2[:selected_width]
    a0 = .5 * w1[:, selected_width:] @ w2[selected_width:]
    zmap, vmode = weight(selected_width, modes), weight(modes, d)
    jmap = .5 * w1k @ zmap
    support = np.zeros((h, n, n), dtype=bool)
    for head in range(h):
        for start in range(0, n, 16):
            keys = rng.choice(n, 61, replace=False)
            support[head, start:start + 16, keys] = True
    ubase, uanchor = norm(hidden), norm(hidden_anchor)
    keys = (ubase @ wk).reshape(n, h, dh)
    values = (ubase @ wv).reshape(n, h, dh)
    anchor_q, anchor_k, anchor_v = [(uanchor @ w).reshape(n, h, dh) for w in (wq, wk, wv)]

    def attention(q, k, v, ids, dense=False):
        scores = np.einsum('ihc,jhc->hij', q, k) / np.sqrt(dh)
        probabilities = softmax(scores if dense else np.where(support[:, ids], scores, -np.inf))
        return np.einsum('hij,jhc->hic', probabilities, v)

    all_ids = np.arange(n)
    native_anchor = attention(anchor_q, anchor_k, anchor_v, all_ids, dense=True)
    selected_anchor = attention(anchor_q, anchor_k, anchor_v, all_ids)
    scores_a = np.einsum('ihc,jhc->hij', anchor_q, anchor_k) / np.sqrt(dh)
    mass = (softmax(scores_a) * support).sum(-1, keepdims=True)
    rho = 1 - mass
    mu_a, sd_a = anchor_v.mean(0)[:, None], anchor_v.std(0)[:, None]
    mu, sd = values.mean(0)[:, None], values.std(0)[:, None]
    history = (native_anchor - mass * selected_anchor - rho * mu_a) / sd_a
    anchor_design = selected_anchor @ readers['low']

    def tail_input(ids, attention_output):
        joined = attention_output.transpose(1, 0, 2).reshape(len(ids), d)
        residual = hidden[ids] + (joined @ wo) * g1
        cross_q = (norm(residual) @ cq).reshape(len(ids), h, dh)
        logits = np.einsum('ihc,hjc->hij', cross_q, ck) / np.sqrt(dh)
        cross = np.einsum('hij,hjc->ihc', softmax(logits), cv).reshape(len(ids), d) @ co
        residual = residual + cross
        return residual, norm(residual) * fscale + fshift

    _, cache_u = tail_input(all_ids, native_anchor)
    cache_z = cache_u @ w1k
    cache_y = gelu(cache_u @ w1) @ w2

    def ffn_operands(ids, u):
        delta_u = u - cache_u[ids]
        z = u @ w1k
        delta_z = z - cache_z[ids]
        delta_h = gelu(z) - gelu(cache_z[ids])
        base = cache_y[ids] + delta_u @ a0 + delta_h @ w2k
        q = delta_h @ zmap - delta_u @ jmap
        features = delta_h - .4 * delta_z
        return base, q, features

    def fit_ffn(u_p):
        base_p, q_p, x_p = ffn_operands(probes, u_p)
        truth = gelu(u_p @ w1) @ w2
        target = truth - base_p
        system = x_p @ x_p.T / selected_width + .75 * np.eye(p)
        iq = np.linalg.solve(system, q_p)
        gain = np.linalg.solve((q_p.T @ iq) * (vmode @ vmode.T),
                               (iq * (target @ vmode.T)).sum(0))
        coefficient = np.linalg.solve(system, target - (q_p * gain) @ vmode)
        return truth, x_p, gain, coefficient

    def read_ffn(ids, u, fit):
        truth, x_p, gain, coefficient = fit
        base, q, x = ffn_operands(ids, u)
        out = base + (q * gain) @ vmode + .5 * (x @ x_p.T / selected_width) @ coefficient
        location = {int(v): j for j, v in enumerate(probes)}
        for j, token in enumerate(ids):
            if int(token) in location:
                out[j] = truth[location[int(token)]]
        return out

    def attention_operands(ids, selected):
        design = np.concatenate([np.ones((h, len(ids), 1)), selected @ readers['low'] - anchor_design[:, ids]], axis=-1)
        base = (1 - rho[:, ids]) * selected + rho[:, ids] * mu + history[:, ids] * sd
        return design, base

    def fit_attention(design, base, truth):
        return np.linalg.solve(np.swapaxes(design, -1, -2) @ design + 6.4 * np.eye(32),
                               np.swapaxes(design, -1, -2) @ (truth - base))

    # Conventional full-field graph, preserving exact P64 writebacks.
    full_q = (ubase @ wq).reshape(n, h, dh)
    full_selected = attention(full_q, keys, values, all_ids)
    truth_p = attention(full_q[probes], keys, values, probes, dense=True)
    design, base = attention_operands(all_ids, full_selected)
    att_coef = fit_attention(design[:, probes], base[:, probes], truth_p)
    full_att = base + design @ att_coef
    full_att[:, probes] = truth_p
    full_residual, full_u = tail_input(all_ids, full_att)
    ff_fit = fit_ffn(full_u[probes])
    full_out = full_residual + read_ffn(all_ids, full_u, ff_fit) * g2

    # Both coefficients obtained before evaluating non-probe attention/tail.
    q_p = (ubase[probes] @ wq).reshape(p, h, dh)
    teacher_p = attention(q_p, keys, values, probes, dense=True)
    selected_p = attention(q_p, keys, values, probes)
    xp, bp = attention_operands(probes, selected_p)
    stream_coef = fit_attention(xp, bp, teacher_p)
    residual_p, u_p = tail_input(probes, teacher_p)
    stream_ff_fit = fit_ffn(u_p)
    streamed = np.empty_like(full_out)
    streamed[probes] = residual_p + stream_ff_fit[0] * g2
    tile_count = 0
    for start in range(0, len(rest), 71):
        ids = rest[start:start + 71]
        qt = (ubase[ids] @ wq).reshape(len(ids), h, dh)
        selected = attention(qt, keys, values, ids)
        xt, bt = attention_operands(ids, selected)
        restored = bt + xt @ stream_coef
        residual_t, u_t = tail_input(ids, restored)
        streamed[ids] = residual_t + read_ffn(ids, u_t, stream_ff_fit) * g2
        tile_count += 1

    # LayerNorm + following linear can be evaluated from moments of B + F U.
    b, f, u = rng.normal(size=(47, d)), rng.normal(size=(47, 13)), rng.normal(size=(13, d))
    w = weight(d, 59)
    gamma, beta = rng.normal(size=(2, d))
    z = b + f @ u
    mean = (b.sum(-1) + f @ u.sum(-1))[:, None] / d
    second = ((b*b).sum(-1) + 2 * np.einsum('ir,ir->i', f, b @ u.T)
              + np.einsum('ir,rs,is->i', f, u @ u.T, f))[:, None] / d
    scale = np.sqrt(np.maximum(second - mean**2, 0) + 1e-6)
    direct_ln_linear = (norm(z) * gamma + beta) @ w
    deferred_ln_linear = ((b * gamma) @ w + f @ ((u * gamma) @ w) - mean * (gamma @ w)) / scale + beta @ w

    errors = {'attention_coefficients': float(np.max(np.abs(stream_coef-att_coef))),
              'FFN_probe_input': float(np.max(np.abs(u_p-full_u[probes]))),
              'FFN_gain': float(np.max(np.abs(stream_ff_fit[2]-ff_fit[2]))),
              'FFN_coefficients': float(np.max(np.abs(stream_ff_fit[3]-ff_fit[3]))),
              'whole_block_output': float(np.max(np.abs(streamed-full_out))),
              'LayerNorm_then_linear_deferred_identity': float(np.max(np.abs(direct_ln_linear-deferred_ln_linear)))}
    assert max(errors.values()) < 1e-9, errors
    N, D, L, H, K, R = 31668, 5120, 40, 40, 2560, 31
    report = dict(status='CPU_ALGEBRA_VERIFIED', actual_CAL=cal_report, pv_sha256=digest(args.pv),
                  synthetic_graph=dict(rows=n, heads=h, head_dim=dh, probes=p, FFN_width=width,
                                       FFN_selected=selected_width, FFN_modes=modes, nonprobe_tiles=tile_count),
                  maximum_absolute_errors=errors,
                  production_shape_accounting=dict(N=N,D=D,H=H,L=L,
                      one_full_BF16_ND_GiB=2*N*D/2**30,
                      one_full_FP32_ND_GiB=4*N*D/2**30,
                      tile8192_FP32_ND_GiB=4*8192*D/2**30,
                      one_full_FP32_attention_design_GiB=4*N*H*(R+1)/2**30,
                      persistent_main_fields_GiB=(2*L*N*(2*D+K)+2*L*N*D+2*L*N*H*R+4*L*N*H)/2**30),
                  evidence_scope=['real CAL matrices match frozen high and low hashes',
                                  'all other weights/hidden states are synthetic float64',
                                  'no production BF16 bitwise, quality, GPU or timing claim',
                                  'streaming changes materialization, not persistent historical payload',
                                  'row tiling may change GEMM kernel/reduction choices; speed must not be inferred from bytes alone'],
                  new_gpu_jobs=0, model_generations=0)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
