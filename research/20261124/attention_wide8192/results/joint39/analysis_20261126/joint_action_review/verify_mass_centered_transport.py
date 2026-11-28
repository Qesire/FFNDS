"""Algebra and synthetic witnesses for transporting frozen-C low OUT-D state.

Actual model code and dimensions are bound below. Q/K/V witnesses are synthetic;
no Wan inference, candidate performance measurement or model-quality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp


CODE_BLOB = '7e7c611ec2aa7e8c8754bd19963cbce6ed49ded4'


def moments(v):
    mu = v.mean(axis=1, keepdims=True)
    sigma = np.sqrt(np.mean((v - mu) ** 2, axis=1, keepdims=True))
    inv = np.divide(1., sigma, out=np.zeros_like(sigma), where=sigma > 1e-12)
    return mu, sigma, inv


def softmax(logits):
    z = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def encode(a, b, rho, v):
    mu, sigma, inv = moments(v)
    u = a - (1 - rho) * b
    c = (u - rho * mu) * inv
    return c, u, mu, sigma


def decode(b, rho, c, v):
    mu, sigma, _ = moments(v)
    return (1 - rho) * b + rho * mu + c * sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('MASS_CENTERED_TRANSPORT_VERIFIED.json'))
    args = parser.parse_args()
    code = args.code.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(code)).encode() + b'\0' + code).hexdigest()
    assert blob == CODE_BLOB
    decoded = code.decode()
    assert 'u0=(native.float()-(1-rho)*sel.float()).bfloat16()' in decoded
    assert 'base=(1-rho)*selected+u0.float()' in decoded

    rng = np.random.default_rng(201276)
    heads, nquery, nkey, dim = 2, 96, 768, 128
    q = rng.normal(size=(heads, nquery, dim))
    k = rng.normal(size=(heads, nkey, dim))
    va = rng.normal(size=(heads, nkey, dim)) * np.exp(rng.normal(size=(heads, 1, dim)))
    va += rng.normal(size=(heads, 1, dim))
    va[..., 0] = 0.; va[..., 1] = 2.
    logits = q @ k.transpose(0, 2, 1) / np.sqrt(dim)
    s = softmax(logits)
    chosen = np.zeros_like(logits, dtype=bool)
    for h in range(heads):
        for row in range(nquery):
            chosen[h, row, rng.choice(nkey, 96, replace=False)] = True
    sb = softmax(np.where(chosen, logits, -np.inf))
    rho = 1 - np.sum(np.where(chosen, s, 0), axis=-1, keepdims=True)
    aa, ba = s @ va, sb @ va
    ca, ua, mua, da = encode(aa, ba, rho, va)
    tests = {'anchor_identity_max_abs': float(np.max(np.abs(decode(ba, rho, ca, va) - aa)))}

    gain = np.exp(rng.uniform(-1.2, .9, size=(heads, 1, dim)))
    shift = rng.normal(size=(heads, 1, dim))
    gain[..., :2] = 2.; shift[..., :2] = .5
    vs = va * gain + shift
    bs, target = sb @ vs, s @ vs
    mus, ds, _ = moments(vs)
    transported = decode(bs, rho, ca, vs)
    raw = (1 - rho) * bs + ua
    naive = (1 - rho) * bs + ua * np.divide(ds, da, out=np.zeros_like(ds), where=da > 1e-12)
    tests.update(positive_diagonal_affine_exact_max_abs=float(np.max(np.abs(transported - target))),
                 raw_base_error_l2=float(np.linalg.norm(raw - target)),
                 uncentered_scale_only_error_l2=float(np.linalg.norm(naive - target)),
                 known_update_identity_max_abs=float(np.max(np.abs(transported - raw - rho * (mus - mua) - ca * (ds - da)))))

    # Independent anchor/current affine coordinate frames, with different routes.
    s2 = softmax(rng.normal(size=(heads, nquery, nkey)))
    sb2 = softmax(np.where(chosen, rng.normal(size=(heads, nquery, nkey)), -np.inf))
    vb = rng.normal(size=va.shape)
    base = decode(sb2 @ vb, rho, ca, vb)
    ga, gs = (np.exp(rng.normal(size=(heads, 1, dim))) for _ in range(2))
    ta, ts = (rng.normal(size=(heads, 1, dim)) for _ in range(2))
    ga[..., :2] = 2.; ta[..., :2] = .5
    vat = va * ga + ta
    cat, _, _, _ = encode(s @ vat, sb @ vat, rho, vat)
    vbt = vb * gs + ts
    frame = decode(sb2 @ vbt, rho, cat, vbt)
    tests.update(anchor_frame_invariant_cache_max_abs=float(np.max(np.abs(cat - ca))),
                 independent_frame_base_equivariance_max_abs=float(np.max(np.abs(frame - (base * gs + ts)))))

    # Fixed-feature P64 correction: new base yields a known residual-null update.
    p = np.arange(64)
    f = 32
    features = rng.normal(size=(heads, nquery, f))
    xp = features[:, p]
    recovery = features @ np.linalg.solve(xp.transpose(0, 2, 1) @ xp + 6.4 * np.eye(f), xp.transpose(0, 2, 1))
    truth = s2 @ vb
    base_raw = (1 - rho) * (sb2 @ vb) + ua
    base_new = decode(sb2 @ vb, rho, ca, vb)
    def recover(bottom):
        answer = bottom + recovery @ (truth[:, p] - bottom[:, p])
        answer[:, p] = truth[:, p]
        return answer
    update = base_new - base_raw
    remaining_update = update - recovery @ update[:, p]
    remaining_update[:, p] = 0
    corrected_difference = recover(base_new) - recover(base_raw)
    tests.update(fixed_recovery_update_max_abs=float(np.max(np.abs(corrected_difference - remaining_update))),
                 corrected_probe_difference_max_abs=float(np.max(np.abs(corrected_difference[:, p]))))

    constant_current = vb.copy(); constant_current[..., :3] = np.array([0., 2., .5])
    constant_output = decode(sb2 @ constant_current, rho, ca, constant_current)
    tests['constant_current_columns_exact_max_abs'] = float(np.max(np.abs(constant_output[..., :3] - constant_current[:, :1, :3])))
    tests['constant_anchor_to_variable_current_finite'] = bool(np.isfinite(decode(sb2 @ vb, rho, ca, vb)).all())

    # Symbolic finite probability-mass centering and affine reconstruction.
    A, B, r, m, D, t, b = sp.symbols('A B r m D t b', nonzero=True)
    C = (A - (1-r)*B - r*m) / D
    anchor = (1-r)*B + r*m + C*D
    shifted = (1-r)*(B*t+b) + r*(m*t+b) + C*D*t
    tests['symbolic_anchor_identity'] = bool(sp.simplify(anchor-A) == 0)
    tests['symbolic_affine_reconstruction'] = bool(sp.simplify(shifted-(A*t+b)) == 0)
    for key, value in tests.items():
        if key.endswith('max_abs'):
            assert value < 5e-12, (key, value)
    assert tests['raw_base_error_l2'] > 1 and tests['uncentered_scale_only_error_l2'] > 1
    assert tests['symbolic_anchor_identity'] and tests['symbolic_affine_reconstruction']

    n, h, d, layers = 31668, 40, 128, 40
    field_elements = n*h*d
    stats_bytes = h*d*2*4
    result = {
        'status':'ALGEBRA_AND_SYNTHETIC_REFERENCE_VERIFIED',
        'scope':'Actual frozen-C source and dimensions; explicitly synthetic float64 Q/K/V plus SymPy; no real-candidate Wan execution',
        'frozen_c_git_blob':blob,
        'frozen_c_sha256':hashlib.sha256(code).hexdigest(),
        'synthetic_shape':{'heads':heads,'queries':nquery,'keys':nkey,'value_channels':dim,'probes':64,'features':32},
        'tests':tests,
        'model_shape_static_cost':{
            'positions':n,'heads':h,'head_dim':d,'low_layers':layers,
            'persistent_full_field_replacement_elements_per_layer':field_elements,
            'persistent_field_net_new_elements':0,
            'current_mean_std_fp32_bytes':stats_bytes,
            'optional_retained_mean_std_all_low_layers_MiB':stats_bytes*layers/2**20,
            'minimum_extra_BF16_V_read_GiB_both_low_anchor_and_hit_if_one_unfused_pass':2*layers*field_elements*2/2**30,
            'low_hits_channel_decode_multiply_count':layers*field_elements,
            'full_graph_query_count_change':0,
            'scope':'Static count only; actual reductions, fusion, temporary allocation and end-to-end time unmeasured'},
        'important_limits':[
            'Positive affine exactness assumes fixed full and selected routing; general route change remains in probe-recovered innovation.',
            'Independent affine equivariance proven for the base only; the retained OUT-D features and ridge are not proved covariant.',
            'Existing rho and u0 finite precision introduce anchor reconstruction/encoding residuals that need actual BF16 validation.',
            'Current job201276 uses a different selected-plus-groups base; its measured effect sizes do not estimate this candidate.',
            'This construction applies to existing low OUT-D historical slots; high OUT-ABS has no corresponding history field.']}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
