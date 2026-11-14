"""V4 representation / density comparison runner. Consumes the landed VB_DENSE22 bank.

0 new D4 full executions: pure local fitting + FFN-weight read + scoring. For each
row-prefix (256/512) x six primary site (high/low x layer 4/19/34), fit six models on
CAL_A11 (pooled, source-equal) and score DEV11 (full 512 panel) + CAL per source.

Anti-leakage: 11x256 and 11x512 are SEPARATE fits (no shared statistics); DEV never
enters any fit; A0/P/Q/alpha/readout all rebuilt from the CAL training subset only.
Primary estimand (route section 6): M(c,e,m)=mean over layers of gatedSSE/(rows*5120);
r=(M_m-M_Q1)/M_Q1; expert-equal within source, then source-equal. Six-site primary
table; all-80 is a pre-registered SEPARATE later table (V4D full-80 export). Primary
score is the F64 gated pre-gate increment math lift; BF16 replay is a V4D check.
"""
import datetime
import json
from pathlib import Path

import torch

import v4_repr as R
from ridge_fit_core import bf16_replay, _f64_sse
from runtime_guard import sha256_file as sha

SITES = (('high', 4), ('high', 19), ('high', 34), ('low', 4), ('low', 19), ('low', 34))
EXPERT_STEP = {'high': (0, 1), 'low': (2, 3)}   # (anchor, hit)
PREFIXES = (256, 512)
LAM_MAIN = 1e-3
D = R.D


def exclusive_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def site_key(expert, layer):
    return '%s%02d' % (expert, layer)


def load_site(bank_root, source, step, layer):
    name = '%s__seed42__s%02d__b%02d.pt' % (source, step, layer)
    d = torch.load(bank_root / source / name, map_location='cpu', weights_only=True)
    g = d['gate'].reshape(-1)                      # (1,1,D) per-channel -> (D,)
    return d['U'], d['Y_pre_gate'], g


def build_pack(bank_root, sources, expert, layer, prefix, device):
    a_step, h_step = EXPERT_STEP[expert]
    pack = []
    for src in sources:
        u0, y0, _ = load_site(bank_root, src, a_step, layer)
        u1, y1, g = load_site(bank_root, src, h_step, layer)
        u0 = u0[:prefix].to(device=device, dtype=torch.float32)
        u1 = u1[:prefix].to(device=device, dtype=torch.float32)
        y0 = y0[:prefix].to(device=device, dtype=torch.float32)
        y1 = y1[:prefix].to(device=device, dtype=torch.float32)
        g = g.to(device=device, dtype=torch.float32)
        pack.append({'source': src, 'u0': u0, 'u1': u1, 'y0': y0, 'y1': y1,
                     'X': u1 - u0, 'Y': y1 - y0, 'g': g, 'n': u0.shape[0]})
    return pack


def load_layer_weights(scope, expert, layer, device):
    from safetensors import safe_open
    ck = scope['checkpoints'][expert]
    if sha(ck['path']) != ck['sha256']:
        raise RuntimeError('checkpoint provenance ' + expert)
    with safe_open(ck['path'], framework='pt', device='cpu') as f:
        w1, b1, w2, b2 = [f.get_tensor('blocks.%d.ffn.%s' % (layer, n))
                          for n in ('0.weight', '0.bias', '2.weight', '2.bias')]
    shapes = [(13824, 5120), (13824,), (5120, 13824), (5120,)]
    if [tuple(t.shape) for t in (w1, b1, w2, b2)] != shapes:
        raise RuntimeError('ffn weight shape ' + expert)
    if not all(t.dtype == torch.bfloat16 for t in (w1, b1, w2, b2)):
        raise RuntimeError('ffn weight dtype')
    return [t.to(device) for t in (w1, b1, w2, b2)]


def fit_config(cal, w1, b1, w2, b2):
    """Fit all six models on one CAL pack. Returns params + diagnostics."""
    alpha_q0, a0, z_norm = R.q0_prior(cal, w1, b1, w2)
    p, q = R.select_pq(cal, a0, r_in=64, r_out=32)
    a_q1, alpha_q1, q1_info = R.fit_q1(cal, w1, w2, a0, alpha_q0, LAM_MAIN)
    d_free, free_info = R.fit_stable_free_linear(cal, a_q1, LAM_MAIN)
    a_free = a_q1 + d_free
    params = {'CURRENT_Q1': {'A': a_q1}, 'STABLE_FREE_LINEAR': {'A': a_free}}
    diag = {'Z': z_norm, 'A0_fro2': float(a0.square().sum()),
            'alpha_Q0': alpha_q0.detach().cpu().tolist(),
            'alpha_Q1': alpha_q1.detach().cpu().tolist(),
            'q1_fit': q1_info, 'free_ridge_fit': free_info,
            'P_sha256': sha_tensor(p), 'Q_sha256': sha_tensor(q)}
    for model in ('L32', 'DELTA_N32', 'ENDPOINT_N32', 'L64_Q32'):
        cout, rinfo = R.fit_readout(model, cal, a0, p, q)
        params[model] = {'C': cout}
        diag[model + '_readout'] = rinfo
        diag[model + '_C_fro2'] = float(cout.square().sum())
    params['_bases'] = {'P': p, 'Q': q, 'A0': a0}
    return params, diag


def sha_tensor(t):
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def predict(model, params, c):
    if model in ('CURRENT_Q1', 'STABLE_FREE_LINEAR'):
        return c['X'].double() @ params[model]['A']
    b = params['_bases']
    return R.predict_repr(model, c, b['A0'], b['P'], b['Q'], params[model]['C'])


def score_pack(model, params, pack):
    out = {}
    for c in pack:
        pred = predict(model, params, c)
        out[c['source']] = R.gated_mse(pred, c['Y'].double(), c['g'].double(), c['n'])
    return out


def cal_numerical_floor(cal, params):
    """CAL neutral numerical-error floor: BF16 cache/gate replay rounding of Q1.
    Mean gated MSE between BF16-replay gated prediction and gated target on CAL."""
    a_bf16 = params['CURRENT_Q1']['A'].float().bfloat16()
    vals = []
    for c in cal:
        # bf16_replay reproduces the deployed BF16 cache/gate rounding: u0/u1/y0/y1 must be
        # bf16 to match a_bf16 in u0@a; gate is fp32 for the base.float()*g promotion.
        _, gated, target = bf16_replay(c['u0'].bfloat16(), c['u1'].bfloat16(),
                                       c['y0'].bfloat16(), c['y1'].bfloat16(),
                                       c['g'].float(), a_bf16)
        sse = float(_f64_sse(gated.float(), target.float(), torch.ones_like(c['g'])))
        vals.append(sse / (c['n'] * D))
    return sum(vals) / len(vals)


def aggregate(dev_scores, cal_scores, floor):
    """dev_scores[model][site_key][source] = MSE. Build per-source/expert estimand."""
    models = R.MODELS
    experts = ('high', 'low')
    layers = (4, 19, 34)
    sources = sorted(next(iter(dev_scores.values()))[site_key('high', 4)].keys())
    per_source = {}
    for src in sources:
        m_ce = {}
        for m in models:
            for e in experts:
                mse_layers = [dev_scores[m][site_key(e, L)][src] for L in layers]
                m_ce[(m, e)] = sum(mse_layers) / len(mse_layers)
        rel = {}
        for m in models:
            for e in experts:
                q1 = m_ce[('CURRENT_Q1', e)]
                rel[(m, e)] = (m_ce[(m, e)] - q1) / q1 if q1 > 0 else None
        rel_source = {m: sum(rel[(m, e)] for e in experts) / len(experts) for m in models}
        denom_below_floor = any(m_ce[('CURRENT_Q1', e)] < floor for e in experts)
        per_source[src] = {'M': {f'{m}|{e}': m_ce[(m, e)] for m in models for e in experts},
                           'relative_to_Q1': rel_source,
                           'q1_denominator_below_floor': denom_below_floor}
    summary_rel = {}
    ratio_of_means = {}
    for m in models:
        summary_rel[m] = sum(per_source[s]['relative_to_Q1'][m] for s in sources) / len(sources)
        mean_m = sum(sum(per_source[s]['M'][f'{m}|{e}'] for e in experts) / 2 for s in sources) / len(sources)
        mean_q1 = sum(sum(per_source[s]['M'][f'CURRENT_Q1|{e}'] for e in experts) / 2 for s in sources) / len(sources)
        ratio_of_means[m] = (mean_m - mean_q1) / mean_q1 if mean_q1 > 0 else None
    loo = {}
    for m in models:
        loo[m] = {drop: sum(per_source[s]['relative_to_Q1'][m] for s in sources if s != drop) / (len(sources) - 1)
                  for drop in sources}
    return {'per_source': per_source, 'source_equal_relative': summary_rel,
            'ratio_of_mean_risks': ratio_of_means, 'leave_one_source_out': loo,
            'cal_numerical_floor_mse': floor}


def run(stage, scope, scope_sha):
    out = stage / 'output'
    out.mkdir(exist_ok=False)
    bank_root = Path(scope['bank_root'])
    manifest = json.loads((bank_root.parent / 'BANK_MANIFEST.json').read_text())
    if sha(bank_root.parent / 'BANK_MANIFEST.json') != scope['bank_manifest_sha256']:
        raise RuntimeError('bank manifest provenance')
    cal_sources = sorted(k for k, v in manifest.items() if v['cycle_role'] == 'CAL')
    dev_sources = sorted(k for k, v in manifest.items() if v['cycle_role'] == 'DEV')
    if len(cal_sources) != 11 or len(dev_sources) != 11:
        raise RuntimeError('cohort split')
    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dev_scores = {m: {} for m in R.MODELS}
    cal_scores = {m: {} for m in R.MODELS}
    fits = {}
    floors = {}
    t0 = datetime.datetime.now(datetime.timezone.utc)
    for prefix in PREFIXES:
        for expert, layer in SITES:
            sk = site_key(expert, layer)
            w1, b1, w2, b2 = load_layer_weights(scope, expert, layer, device)
            cal = build_pack(bank_root, cal_sources, expert, layer, prefix, device)
            dev = build_pack(bank_root, dev_sources, expert, layer, 512, device)
            params, diag = fit_config(cal, w1, b1, w2, b2)
            floor = cal_numerical_floor(cal, params)
            floors['%d|%s' % (prefix, sk)] = floor
            for m in R.MODELS:
                dev_scores[m].setdefault(sk, {})
                cal_scores[m].setdefault(sk, {})
                # prefix-specific scores keyed by prefix|site to keep 256/512 separate
                ds = score_pack(m, params, dev)
                cs = score_pack(m, params, cal)
                for s, v in ds.items():
                    dev_scores[m][sk]['%d|%s' % (prefix, s)] = v
                for s, v in cs.items():
                    cal_scores[m][sk]['%d|%s' % (prefix, s)] = v
            fits['%d|%s' % (prefix, sk)] = diag
            del params, cal, dev, w1, b1, w2, b2
            torch.cuda.empty_cache()
            print('V4_FIT prefix=%d site=%s' % (prefix, sk), flush=True)

    # aggregate per prefix (separate fits -> separate tables)
    tables = {}
    for prefix in PREFIXES:
        ds = {m: {sk: {s: dev_scores[m][sk]['%d|%s' % (prefix, s)] for s in dev_sources}
                  for sk in dev_scores[m]} for m in R.MODELS}
        cs = {m: {sk: {s: cal_scores[m][sk]['%d|%s' % (prefix, s)] for s in cal_sources}
                  for sk in cal_scores[m]} for m in R.MODELS}
        fl = sum(floors['%d|%s' % (prefix, site_key(e, L))] for e, L in SITES) / len(SITES)
        tables[str(prefix)] = aggregate(ds, cs, fl)

    elapsed = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    summary = {'scope_id': scope['scope_id'], 'status': 'V4_REPR_DENSITY_COMPLETE',
               'new_D4_full_executions': 0, 'models': list(R.MODELS),
               'row_prefixes': list(PREFIXES), 'sites': [site_key(e, L) for e, L in SITES],
               'primary_estimand': 'source_equal_relative (expert-equal then source-equal)',
               'scoring': 'F64 gated pre-gate increment math lift; BF16 replay is V4D',
               'quality_metrics': 'NOT_MEASURED', 'performance_comparison': 'NOT_BENCHMARKED',
               'elapsed_seconds': elapsed,
               'source_density_delta': {m: tables['512']['source_equal_relative'][m] - tables['256']['source_equal_relative'][m]
                                        for m in R.MODELS}}
    exclusive_json(out / 'VB_LOCAL_FITS.json', {'fits': fits, 'floors': floors})
    exclusive_json(out / 'VB_SOURCE_DENSITY.json', {'tables': tables,
                   'dev_gated_mse': dev_scores, 'cal_gated_mse': cal_scores})
    exclusive_json(out / 'SUMMARY.json', summary)
    print('V4_COMPLETE elapsed=%.1fs' % elapsed, flush=True)
    return summary