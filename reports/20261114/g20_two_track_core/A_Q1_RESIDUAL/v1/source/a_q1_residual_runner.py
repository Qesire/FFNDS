"""A Q1-residual four-arm runner (NEW author code, rev5 handoff A5-A6).

0 new D4/O40: pure local fitting + FFN-weight read + scoring on the landed VB_DENSE22
bank. Rebuilds A_q1 from the published V4 alpha_Q1 (no PCG refit), fits three fixed
residual readouts on CAL_A11x512, scores DEV11 + CAL per source. Saves complete
reloadable parameters per site, reloads, then scores final math + BF16 outputs from
the reloaded tensors. Three check groups (ancestry/zero-correction, CAL objective,
finite BF16). No cal_numerical_floor / source_density_delta / unmeasurable path.
"""
import datetime
import json
from pathlib import Path

import torch

import a_q1_residual_repr as R
from runtime_guard import sha256_file as sha

SITES = (('high', 4), ('high', 19), ('high', 34), ('low', 4), ('low', 19), ('low', 34))
EXPERT_STEP = {'high': (0, 1), 'low': (2, 3)}
PREFIX = 512
D = R.D


def exclusive_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def site_key(expert, layer):
    return '%s%02d' % (expert, layer)


def load_site(bank_root, source, step, layer):
    name = '%s__seed42__s%02d__b%02d.pt' % (source, step, layer)
    d = torch.load(bank_root / source / name, map_location='cpu', weights_only=True)
    g = d['gate'].reshape(-1)
    return d['U'], d['Y_pre_gate'], g


def build_pack(bank_root, sources, expert, layer, device):
    a_step, h_step = EXPERT_STEP[expert]
    pack = []
    for src in sources:
        u0, y0, _ = load_site(bank_root, src, a_step, layer)
        u1, y1, g = load_site(bank_root, src, h_step, layer)
        u0 = u0[:PREFIX].to(device=device, dtype=torch.float32)
        u1 = u1[:PREFIX].to(device=device, dtype=torch.float32)
        y0 = y0[:PREFIX].to(device=device, dtype=torch.float32)
        y1 = y1[:PREFIX].to(device=device, dtype=torch.float32)
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


def sha_tensor(t):
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_param_tensors(a_q1, a_q1_f32, alpha_q1, p64, q32, b, sigma, gammas, cs):
    """Assemble the complete weights_only-reloadable tensor dict for one site (A6)."""
    p32 = p64[:, :32]
    kit = {'A_q1': a_q1, 'A_q1_f32': a_q1_f32, 'alpha_Q1': alpha_q1,
           'P64': p64, 'Q32': q32, 'b': b, 'sigma': sigma,
           'gamma_L': gammas['Q1_PLUS_LINEAR64_Q32'],
           'gamma_D': gammas['Q1_PLUS_DELTA_N32'],
           'gamma_E': gammas['Q1_PLUS_ENDPOINT_N32'],
           'C_L': cs['Q1_PLUS_LINEAR64_Q32'],
           'C_D': cs['Q1_PLUS_DELTA_N32'],
           'C_E': cs['Q1_PLUS_ENDPOINT_N32']}
    # actual BF16 tensors participating in replay (exported for downstream consumers)
    kit['A_Q1_BF16'] = R.Qb(a_q1_f32)
    a_l = a_q1 + (p64 / kit['gamma_L'][None, :]) @ kit['C_L'] @ q32.T
    kit['A_L_BF16'] = R.Qb(a_l)
    a_d = a_q1 + (p32 / kit['gamma_D'][:32][None, :]) @ kit['C_D'][:32] @ q32.T
    kit['A_D_linear_BF16'] = R.Qb(a_d)
    a_e = a_q1 + (p32 / kit['gamma_E'][:32][None, :]) @ kit['C_E'][:32] @ q32.T
    kit['A_E_linear_BF16'] = R.Qb(a_e)
    kit['P32_BF16'] = R.Qb(p32)
    kit['Q32_BF16'] = R.Qb(q32)
    kit['C_D_N_BF16'] = R.Qb(kit['C_D'][32:] / kit['gamma_D'][32:, None])
    kit['C_E_N_BF16'] = R.Qb(kit['C_E'][32:] / kit['gamma_E'][32:, None])
    kit['b_FP32'] = b.float()
    kit['sigma_FP32'] = sigma.float()
    return kit


def score_all_arms(pack, kit):
    """Score math + BF16 for all four arms on a pack, using reloaded parameter tensors.
    Returns {arm: {'math': {source: mse}, 'bf16': {source: mse}}}."""
    p64 = kit['P64']
    p32 = p64[:, :32]
    q32 = kit['Q32']
    b = kit['b']
    sigma = kit['sigma']
    a_q1 = kit['A_q1']
    gammas = {'Q1_PLUS_LINEAR64_Q32': kit['gamma_L'], 'Q1_PLUS_DELTA_N32': kit['gamma_D'],
              'Q1_PLUS_ENDPOINT_N32': kit['gamma_E']}
    cs = {'Q1_PLUS_LINEAR64_Q32': kit['C_L'], 'Q1_PLUS_DELTA_N32': kit['C_D'],
          'Q1_PLUS_ENDPOINT_N32': kit['C_E']}
    out = {arm: {'math': {}, 'bf16': {}} for arm in R.ARMS}
    for c in pack:
        for arm in R.ARMS:
            if arm == 'CURRENT_Q1':
                pm = c['X'].double() @ a_q1
            else:
                phi = R.feature_raw(arm, c, p64, p32, b, sigma) / gammas[arm]
                pm = c['X'].double() @ a_q1 + phi @ cs[arm] @ q32.T
            out[arm]['math'][c['source']] = R.gated_mse(pm, c['Y'].double(), c['g'].double(), c['n'])
            _, gated, target = R.bf16_predict(arm, c, kit)
            out[arm]['bf16'][c['source']] = R.gated_mse(gated.float(), target.float(),
                                                        torch.ones_like(c['g']), c['n'])
    return out


def aggregate(scores, sources):
    """scores[arm]['math'|'bf16'][source] = per-site MSE dict {sk: mse}. Build per-source
    layer-equal -> expert-equal risks, differences vs Q1, source-equal mean + LOO.
    high/low reported separately. scores structure: [arm][path][source][sk]=mse."""
    experts = ('high', 'low')
    layers = (4, 19, 34)
    result = {}
    for path in ('math', 'bf16'):
        per_source = {}
        for src in sources:
            m_e = {}
            for arm in R.ARMS:
                for e in experts:
                    vals = [scores[arm][path][src][site_key(e, L)] for L in layers]
                    m_e[(arm, e)] = sum(vals) / len(vals)
            arm_source = {arm: sum(m_e[(arm, e)] for e in experts) / len(experts) for arm in R.ARMS}
            arm_expert = {f'{arm}|{e}': m_e[(arm, e)] for arm in R.ARMS for e in experts}
            diffs = {}
            for arm in R.RESIDUAL_ARMS:
                diffs['R_Q1-R_' + arm] = arm_source['CURRENT_Q1'] - arm_source[arm]
            diffs['R_L-R_E'] = arm_source['Q1_PLUS_LINEAR64_Q32'] - arm_source['Q1_PLUS_ENDPOINT_N32']
            diffs['R_D-R_E'] = arm_source['Q1_PLUS_DELTA_N32'] - arm_source['Q1_PLUS_ENDPOINT_N32']
            rel = {}
            for arm in R.RESIDUAL_ARMS:
                q1 = arm_source['CURRENT_Q1']
                rel[arm] = (arm_source[arm] - q1) / q1 if q1 > 0 else None
            per_source[src] = {'risk': arm_source, 'risk_by_expert': arm_expert,
                               'differences': diffs, 'relative_to_Q1': rel}
        # source-equal means of risks and differences
        mean_risk = {arm: sum(per_source[s]['risk'][arm] for s in sources) / len(sources) for arm in R.ARMS}
        diff_keys = list(per_source[sources[0]]['differences'].keys())
        mean_diff = {k: sum(per_source[s]['differences'][k] for s in sources) / len(sources) for k in diff_keys}
        loo = {k: {drop: sum(per_source[s]['differences'][k] for s in sources if s != drop) / (len(sources) - 1)
                   for drop in sources} for k in diff_keys}
        # high/low separate source-equal mean risk
        by_expert = {}
        for e in experts:
            by_expert[e] = {arm: sum(per_source[s]['risk_by_expert'][f'{arm}|{e}'] for s in sources) / len(sources)
                            for arm in R.ARMS}
        result[path] = {'per_source': per_source, 'source_equal_mean_risk': mean_risk,
                        'source_equal_mean_difference': mean_diff, 'leave_one_source_out': loo,
                        'source_equal_mean_risk_by_expert': by_expert}
    return result


def run(stage, scope, scope_sha):
    out = stage / 'output'
    out.mkdir(exist_ok=False)
    (out / 'params').mkdir(exist_ok=False)
    bank_root = Path(scope['bank_root'])
    manifest = json.loads((bank_root.parent / 'BANK_MANIFEST.json').read_text())
    if sha(bank_root.parent / 'BANK_MANIFEST.json') != scope['bank_manifest_sha256']:
        raise RuntimeError('bank manifest provenance')
    cal_sources = sorted(k for k, v in manifest.items() if v['cycle_role'] == 'CAL')
    dev_sources = sorted(k for k, v in manifest.items() if v['cycle_role'] == 'DEV')
    if len(cal_sources) != 11 or len(dev_sources) != 11:
        raise RuntimeError('cohort split')

    fits_path = stage / 'VB_LOCAL_FITS.json'
    if sha(fits_path) != scope['frozen_inputs_sha256']['VB_LOCAL_FITS.json']:
        raise RuntimeError('published V4 fits provenance')
    fits = json.loads(fits_path.read_text())['fits']
    ref_path = stage / 'VB_SOURCE_DENSITY.json'
    if sha(ref_path) != scope['frozen_inputs_sha256']['VB_SOURCE_DENSITY.json']:
        raise RuntimeError('published V4 density provenance')
    ref = json.loads(ref_path.read_text())

    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checks = {'ancestry': {}, 'zero_correction': {}, 'cal_objective': {}, 'bf16_finite': True,
              'bf16_export_consistency': {}, 'reload': True}
    # scores[arm][path][source][sk] = mse
    scores = {arm: {'math': {}, 'bf16': {}} for arm in R.ARMS}
    for arm in R.ARMS:
        for src in cal_sources + dev_sources:
            scores[arm]['math'][src] = {}
            scores[arm]['bf16'][src] = {}
    params_manifest = {}
    readout_diag = {}
    t0 = datetime.datetime.now(datetime.timezone.utc)

    for expert, layer in SITES:
        sk = site_key(expert, layer)
        w1, b1, w2, b2 = load_layer_weights(scope, expert, layer, device)
        cal = build_pack(bank_root, cal_sources, expert, layer, device)
        dev = build_pack(bank_root, dev_sources, expert, layer, device)
        alpha_q1 = torch.tensor(fits['512|' + sk]['alpha_Q1'], dtype=torch.float32, device=device)
        a_q1, a_q1_f32 = R.rebuild_a_q1(w1, w2, alpha_q1)

        # check one (a): Q1 ancestry math risk vs published V4 512 per-source values
        anc = {}
        for c in cal + dev:
            pm = c['X'].double() @ a_q1
            mse = R.gated_mse(pm, c['Y'].double(), c['g'].double(), c['n'])
            role = 'cal' if c['source'] in cal_sources else 'dev'
            refv = ref[role + '_gated_mse']['CURRENT_Q1'][sk]['512|' + c['source']]
            ok = abs(mse - refv) <= 1e-12 + 1e-5 * abs(refv)
            anc[c['source']] = {'rebuilt': mse, 'published_v4': refv, 'match': ok}
            if not ok:
                raise RuntimeError('Q1 ancestry mismatch ' + sk + ' ' + c['source'])
        checks['ancestry'][sk] = anc

        p64, q32 = R.select_bases(cal, a_q1)
        p32 = p64[:, :32]
        b, sigma = R.center_scale(cal, p32)
        gammas = {}
        cs = {}
        for arm in R.RESIDUAL_ARMS:
            gammas[arm] = R.feature_gamma(arm, cal, p64, p32, b, sigma)
            cout, info = R.fit_readout(arm, cal, a_q1, p64, p32, q32, b, sigma, gammas[arm])
            cs[arm] = cout
            j0, jfit, risk_unreg = R.cal_objective(cal, arm, a_q1, p64, p32, q32, b, sigma,
                                                   gammas[arm], cout, info['readout_ridge_rho'])
            tol_j = 1e-8 * max(abs(j0), 1e-30)
            ok = jfit <= j0 + tol_j
            checks['cal_objective'][sk + '|' + arm] = {
                'J0': j0, 'Jfit': jfit, 'cal_risk_unregularized': risk_unreg,
                'tol_J': tol_j, 'pass': ok, 'readout': info}
            if not ok:
                raise RuntimeError('CAL objective increased ' + sk + ' ' + arm)
            if info['readout_normal_relative_residual'] > 1e-8 and info['status'] == 'SOLVED':
                raise RuntimeError('readout normal residual ' + sk + ' ' + arm)
            readout_diag[sk + '|' + arm] = {'C_fro2': float(cout.square().sum()), 'readout': info}

        # check one (b): zero correction == CURRENT_Q1 (math + BF16) on first sorted CAL
        c0 = cal[0]
        zero = {}
        q1_math = c0['X'].double() @ a_q1
        _, q1_gated, _ = R.bf16_q1(c0, a_q1_f32)
        for arm in R.RESIDUAL_ARMS:
            zero_c = torch.zeros_like(cs[arm])
            phi = R.feature_raw(arm, c0, p64, p32, b, sigma) / gammas[arm]
            pm = c0['X'].double() @ a_q1 + phi @ zero_c @ q32.T
            math_eq = torch.equal(pm, q1_math)
            kit_zero = build_param_tensors(a_q1, a_q1_f32, alpha_q1, p64, q32, b, sigma, gammas,
                                           {**cs, arm: zero_c})
            _, gated_z, _ = R.bf16_predict(arm, c0, kit_zero)
            bf16_eq = torch.equal(gated_z, q1_gated)
            zero[arm] = {'math_equal': math_eq, 'bf16_equal': bf16_eq}
            if not (math_eq and bf16_eq):
                raise RuntimeError('zero-correction identity failed ' + sk + ' ' + arm)
        checks['zero_correction'][sk] = zero

        # save complete params, reload, score from reloaded tensors
        kit = build_param_tensors(a_q1, a_q1_f32, alpha_q1, p64, q32, b, sigma, gammas, cs)
        ppath = out / 'params' / (sk + '.pt')
        torch.save(kit, ppath)
        reloaded = torch.load(ppath, map_location=device, weights_only=True)
        # verify exported BF16 A matrices match the replay recomputation from reloaded F64
        cons = {}
        cons['A_Q1'] = torch.equal(reloaded['A_Q1_BF16'], R.Qb(reloaded['A_q1_f32']))
        a_l = reloaded['A_q1'] + (reloaded['P64'] / reloaded['gamma_L'][None, :]) @ reloaded['C_L'] @ reloaded['Q32'].T
        cons['A_L'] = torch.equal(reloaded['A_L_BF16'], R.Qb(a_l))
        checks['bf16_export_consistency'][sk] = cons
        if not all(cons.values()):
            raise RuntimeError('BF16 export consistency ' + sk)

        cal_sc = score_all_arms(cal, reloaded)
        dev_sc = score_all_arms(dev, reloaded)
        for arm in R.ARMS:
            for path in ('math', 'bf16'):
                for src, v in cal_sc[arm][path].items():
                    scores[arm][path][src][sk] = v
                for src, v in dev_sc[arm][path].items():
                    scores[arm][path][src][sk] = v
                vals = list(cal_sc[arm][path].values()) + list(dev_sc[arm][path].values())
                if not torch.isfinite(torch.tensor(vals)).all():
                    checks['bf16_finite'] = False
                    raise RuntimeError('nonfinite risk ' + sk + ' ' + arm + ' ' + path)
        params_manifest[sk] = {'path': str(ppath), 'bytes': ppath.stat().st_size,
                               'sha256': sha(ppath),
                               'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                                           for k, v in reloaded.items()}}
        del kit, reloaded, cal, dev, w1, b1, w2, b2, a_q1, a_q1_f32, p64, q32, p32
        torch.cuda.empty_cache()
        print('A_FIT site=%s' % sk, flush=True)

    # readout: DEV primary (11 sources), CAL diagnostic (11 sources)
    dev_agg = aggregate({arm: {p: {s: scores[arm][p][s] for s in dev_sources} for p in ('math', 'bf16')}
                         for arm in R.ARMS}, dev_sources)
    cal_agg = aggregate({arm: {p: {s: scores[arm][p][s] for s in cal_sources} for p in ('math', 'bf16')}
                         for arm in R.ARMS}, cal_sources)

    elapsed = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    total_params_bytes = sum(v['bytes'] for v in params_manifest.values())
    summary = {'scope_id': scope['scope_id'], 'status': 'A_Q1_RESIDUAL_COMPLETE',
               'new_D4_full_executions': 0, 'new_O40_full_executions': 0,
               'arms': list(R.ARMS), 'sites': [site_key(e, L) for e, L in SITES],
               'rows': PREFIX, 'cal_count': 11, 'dev_count': 11,
               'primary_estimand': 'absolute gated MSE; layer-equal->expert-equal->source-equal; math/BF16 separate',
               'scoring': 'F64 gated pre-gate increment math lift + separate BF16 local cache/operator/gate replay',
               'quality_metrics': 'NOT_MEASURED', 'performance_comparison': 'NOT_BENCHMARKED',
               'expected_rows_per_table': 528, 'total_param_bytes': total_params_bytes,
               'elapsed_seconds': elapsed,
               'dev_source_equal_mean_risk': {p: dev_agg[p]['source_equal_mean_risk'] for p in ('math', 'bf16')},
               'dev_source_equal_mean_difference': {p: dev_agg[p]['source_equal_mean_difference'] for p in ('math', 'bf16')}}
    exclusive_json(out / 'A_Q1_RESIDUAL_PARAMS.json', {'params': params_manifest,
                   'total_bytes': total_params_bytes, 'ancestry': 'published V4 alpha_Q1 (512 config), no PCG refit'})
    exclusive_json(out / 'A_Q1_RESIDUAL_READOUT.json', {'dev': dev_agg, 'cal': cal_agg,
                   'raw_scores': scores, 'readout_diag': readout_diag})
    exclusive_json(out / 'CHECKS.json', checks)
    exclusive_json(out / 'SUMMARY.json', summary)
    print('A_COMPLETE elapsed=%.1fs param_bytes=%d' % (elapsed, total_params_bytes), flush=True)
    return summary