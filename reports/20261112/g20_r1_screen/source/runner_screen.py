"""G20 R1 screening engine. Import and invoke run() ONLY after the guarded
runner has verified allocation, review, ACTIVE, hashes, node and budget.

Pass 1: 80-site lightweight homomorphic landscape (CAL4 + DEV5), Q1-r2 vs
reuse-Y, FP32 math lift and BF16 replay. Pass 2: six pre-registered sites get the
free-ridge A_Q1+D exact fit (main + robustness lambda) and frozen ICC svd_r512
two-residual evaluation, with source-out DEV readouts and identification
diagnostics. A pre-registered trigger may add supplementary site high/0.
Chipmunk author block-mean arm is recorded MISSING_ASSET (no full-M groups).
"""
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch
from safetensors import safe_open

from selector import load_pair, sha
from fit_core import (cheap_site_stats, free_ridge_fit, omega_source,
                      icc_two_residual, bf16_replay, native_ffn, D)
from selfcheck import run as selfcheck_run

CAL = ('chip10_02', 'chip10_04', 'chip10_07', 'chip10_09')
DEV = ('chip10_01', 'chip10_03', 'chip10_05', 'chip10_06', 'chip10_08')
STEPS = {'high': (0, 1), 'low': (2, 3)}


def _mse(sse, n):
    return sse / (n * D)


def _group(per_source, sources, sse_key):
    """Source-equal-weight MSE and increment/full-Y normalized errors."""
    mses, inc, full = [], [], []
    dropped = 0
    for s in sources:
        e = per_source[s]
        n = e['rows']
        mses.append(_mse(e[sse_key], n))
        if e['gated_dy_fro2'] > 0:
            inc.append(e[sse_key] / e['gated_dy_fro2'])
        else:
            dropped += 1
        if e['gated_y1_fro2'] > 0:
            full.append(e[sse_key] / e['gated_y1_fro2'])
    return {'mse': sum(mses) / len(mses),
            'increment_normalized': sum(inc) / len(inc) if inc else None,
            'full_y_normalized': sum(full) / len(full) if full else None,
            'dropped_zero_increment_energy_sources': dropped}


def _paired(per_source, sources, arm_key, base_key='q1_math_sse'):
    rel, absd = [], []
    for s in sources:
        e = per_source[s]
        base = _mse(e[base_key], e['rows'])
        arm = _mse(e[arm_key], e['rows'])
        absd.append(arm - base)
        rel.append((arm - base) / base if base > 0 else None)
    rel = [v for v in rel if v is not None]
    return {'mean_abs_mse_change': sum(absd) / len(absd),
            'mean_rel_change': sum(rel) / len(rel) if rel else None}


def _load_weights(checkpoint, expert, layer):
    with safe_open(checkpoint, framework='pt', device='cpu') as f:
        w = [f.get_tensor('blocks.%d.ffn.%s' % (layer, n)).cuda()
             for n in ('0.weight', '0.bias', '2.weight', '2.bias')]
    if [tuple(t.shape) for t in w] != [(13824, 5120), (13824,), (5120, 13824), (5120,)] \
            or any(t.dtype != torch.bfloat16 or not torch.isfinite(t).all() for t in w):
        raise ValueError('weight shape/dtype/finiteness')
    return w


def _load_q1(entry, site):
    expert, layer = site.split('/')
    prov = entry.get('provenance', {})
    if (prov.get('expert'), prov.get('layer'), prov.get('arm')) != (expert, int(layer), 'Q1'):
        raise ValueError('Q1 manifest provenance mismatch ' + site)
    if sha(entry['path']) != entry['sha256']:
        raise RuntimeError('Q1 matrix SHA ' + entry['path'])
    raw = np.load(entry['path'], allow_pickle=False)
    if raw.shape != (5120, 5120) or raw.dtype != np.uint16:
        raise ValueError('Q1 matrix format')
    a = torch.from_numpy(raw.copy()).view(torch.bfloat16).cuda()
    if not torch.isfinite(a).all():
        raise ValueError('Q1 nonfinite')
    return a


def _load_icc(iccman, site):
    expert, layer = site.split('/')
    entry = iccman[site]
    if sha(entry['path']) != entry['sha256']:
        raise RuntimeError('ICC factor SHA ' + entry['path'])
    fd = torch.load(entry['path'], map_location='cpu', weights_only=True)
    if (fd['expert'], fd['layer'], fd['method'], fd['rank']) != (expert, int(layer), 'svd', 512):
        raise ValueError('ICC frozen config identity')
    fc1, fc2 = fd['fc1'], fd['fc2']
    if fc1['left'].shape != (13824, 512) or fc1['right'].shape != (512, 5120) \
            or fc2['left'].shape != (5120, 512) or fc2['right'].shape != (512, 13824):
        raise ValueError('ICC factor geometry')
    return (fc1['left'].cuda(), fc1['right'].cuda(), fc2['left'].cuda(), fc2['right'].cuda())


def run(stage, scope, scope_sha):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    stage = Path(stage)
    budget = scope['budget']
    assets = json.loads((stage / 'assets_g20.json').read_text())
    q1man = json.loads((stage / 'q1_matrix_manifest.json').read_text())
    iccman = json.loads((stage / 'icc_svd_manifest.json').read_text())
    if tuple(assets['cal_sources']) != CAL:
        raise ValueError('CAL cohort')
    if tuple(assets['dev_sources']) != DEV:
        raise ValueError('DEV cohort')
    out = stage / 'output'
    out.mkdir(exist_ok=False)
    started = time.monotonic()

    def check_budget():
        if time.monotonic() - started > budget['soft_seconds']:
            raise TimeoutError('soft time budget')
        import resource
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > budget['host_memory_gib'] * 1024 ** 3:
            raise MemoryError('host peak RSS budget')
        total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if total > budget['output_bytes']:
            raise RuntimeError('output limit')

    selfcheck = selfcheck_run()
    allsrc = list(CAL) + list(DEV)
    landscape = {}

    # ---------------- Pass 1: 80-site cheap landscape ----------------
    for expert in ('high', 'low'):
        ck = assets['checkpoints'][expert]
        if sha(ck['path']) != ck['sha256']:
            raise RuntimeError('checkpoint SHA ' + expert)
        for layer in range(40):
            check_budget()
            torch.cuda.reset_peak_memory_stats()
            site = '%s/%d' % (expert, layer)
            w1, b1, w2, b2 = _load_weights(ck['path'], expert, layer)
            a = _load_q1(q1man[site], site)
            per_source = {}
            for source in allsrc:
                p = load_pair(assets, source, 42, expert, STEPS[expert], ck['sha256'], layer)
                per_source[source] = cheap_site_stats(p, a, w1, b1, w2, b2)
                del p
            entry = {'site': site, 'expert': expert, 'layer': layer,
                     'per_source': per_source,
                     'cal4': {k: _group(per_source, CAL, k) for k in
                              ('q1_math_gated_sse', 'q1_replay_gated_sse', 'reuse_y_gated_sse')},
                     'dev5': {k: _group(per_source, DEV, k) for k in
                              ('q1_math_gated_sse', 'q1_replay_gated_sse', 'reuse_y_gated_sse')},
                     'matched_counts': {s: per_source[s]['rows'] for s in allsrc},
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            landscape[site] = entry
            write_json(out / ('cheap_%s_%02d.json' % (expert, layer)), entry)
            del w1, b1, w2, b2, a, per_source
            torch.cuda.empty_cache()
            print('CHEAP', site, flush=True)

    # ---------------- Pass 2: six-site expensive fits ----------------
    expensive = {}
    for site in scope['expensive_sites']:
        expensive[site] = _expensive_site(site, assets, q1man, iccman, scope, out, check_budget)
        print('EXPENSIVE', site, flush=True)

    # ---------------- pre-registered supplementary trigger ----------------
    trigger = _evaluate_trigger(landscape, expensive, scope)
    supplementary = None
    if trigger['fires']:
        supplementary = _expensive_site(scope['supplementary_site'], assets, q1man, iccman,
                                        scope, out, check_budget)
        print('SUPPLEMENTARY', scope['supplementary_site'], flush=True)

    summary = _summarize(landscape, expensive, supplementary, trigger, scope)
    summary['selfcheck'] = selfcheck
    summary['chipmunk_arm'] = {'status': 'MISSING_ASSET',
                               'reason': 'author block-mean proxy requires full-M (31668-row) anchor/hit U '
                                         'for real voxel-block means, masks and valid 52-row tail; only 2048-row '
                                         'uniform panels exist for CAL sources. Not scored with scattered proxies; '
                                         'deferred to R2 shared capture.',
                               'required_missing_fields': ['full_M_U_anchor', 'full_M_U_hit',
                                                           'voxel_block_means', 'author_topk_mask', 'valid_tail_rows']}
    write_json(out / 'SUMMARY.json', summary)
    check_budget()
    return {'sites_cheap': len(landscape), 'sites_expensive': list(expensive),
            'supplementary_triggered': trigger['fires'],
            'elapsed_seconds': time.monotonic() - started,
            'peak_rss_bytes': process_peak_rss(),
            'summary_sha256': sha(out / 'SUMMARY.json')}


def process_peak_rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    tmp.replace(path)


def _expensive_site(site, assets, q1man, iccman, scope, out, check_budget):
    expert, layer = site.split('/')
    layer = int(layer)
    ck = assets['checkpoints'][expert]
    w1, b1, w2, b2 = _load_weights(ck['path'], expert, layer)
    a = _load_q1(q1man[site], site)
    allsrc = list(CAL) + list(DEV)
    pairs = {}
    for source in allsrc:
        check_budget()
        pairs[source] = load_pair(assets, source, 42, expert, STEPS[expert], ck['sha256'], layer)
    # gate must be per-channel and shared across CAL sources for the exact solve.
    g0 = pairs[CAL[0]]['g'][0]
    for source in CAL:
        gp = pairs[source]['g']
        if gp.shape != (pairs[source]['u0'].shape[0], D) \
                or not torch.equal(gp, g0[None, :].expand(gp.shape[0], -1)):
            raise RuntimeError('CAL gate not per-channel shared across sources at ' + site)
    g = g0
    x_list, y_list, r_list, n_list = [], [], [], []
    for source in CAL:
        p = pairs[source]
        _, _, y0n = native_ffn(p['u0'], w1, b1, w2, b2)
        _, _, y1n = native_ffn(p['u1'], w1, b1, w2, b2)
        x = p['u1'].float() - p['u0'].float()
        y = y1n.float() - y0n.float()
        r = y - x @ a.float()
        x_list.append(x)
        y_list.append(y)
        r_list.append(r)
        n_list.append(x.shape[0])
    lam_main = scope['ridge']['lambda_main']
    lam_rob = scope['ridge']['lambda_robust']
    d_main, mu, v, info_main = free_ridge_fit(x_list, y_list, r_list, g, n_list, a, lam_main)
    _, _, _, info_rob = free_ridge_fit(x_list, y_list, r_list, g, n_list, a, lam_rob)
    a_free = (a.float() + d_main).bfloat16()
    if not torch.isfinite(a_free).all():
        raise ValueError('nonfinite exported A_free')
    icc_factors = _load_icc(iccman, site)
    per_source = {}
    for source in allsrc:
        p = pairs[source]
        u0, u1, gg = p['u0'], p['u1'], p['g']
        _, _, y0n = native_ffn(u0, w1, b1, w2, b2)
        _, _, y1n = native_ffn(u1, w1, b1, w2, b2)
        x = u1.float() - u0.float()
        n = u0.shape[0]
        # Q1
        q1_math = float(((y0n.float() + x @ a.float() - y1n.float()) * gg).double().square().sum())
        _, q1g, q1t = bf16_replay(u0, u1, y0n, y1n, gg, a)
        q1_rep = float((q1g.float() - q1t.float()).double().square().sum())
        # free ridge (FP32 math lift + BF16-exported replay)
        ridge_pred = y0n.float() + x @ (a.float() + d_main)
        ridge_math = float(((ridge_pred - y1n.float()) * gg).double().square().sum())
        _, rg, rt = bf16_replay(u0, u1, y0n, y1n, gg, a_free)
        ridge_rep = float((rg.float() - rt.float()).double().square().sum())
        # ICC frozen two-residual
        icc = icc_two_residual(p, *icc_factors, w1, b1, w2, b2)
        rec = {'rows': n,
               'gated_dy_fro2': float(((y1n.float() - y0n.float()) * gg).double().square().sum()),
               'gated_y1_fro2': float((y1n.double() * gg.double()).square().sum()),
               'q1_math_sse': q1_math, 'q1_replay_sse': q1_rep,
               'ridge_math_sse': ridge_math, 'ridge_replay_sse': ridge_rep,
               'icc_gated_sse': icc['math_gated_sse'],
               'icc_zero_delta_max_abs': icc['zero_delta_max_abs'],
               'reuse_y_gated_sse': float(((y0n.float() - y1n.float()) * gg).double().square().sum())}
        if source in DEV:
            rec['omega'] = omega_source(x, mu, v)
        per_source[source] = rec
        del p, u0, u1, gg, y0n, y1n, x
    # save main-ridge D for reuse; robustness ridge is metrics-only diagnostic.
    stem = '%s_%02d' % (expert, layer)
    dpath = out / ('ridgeD_%s.npy' % stem)
    np.save(dpath, d_main.bfloat16().view(torch.int16).cpu().numpy().view(np.uint16), allow_pickle=False)
    arms = ('q1_math_sse', 'q1_replay_sse', 'ridge_math_sse', 'ridge_replay_sse',
            'icc_gated_sse', 'reuse_y_gated_sse')
    result = {'site': site, 'expert': expert, 'layer': layer,
              'per_source': per_source,
              'cal4': {k: _group(per_source, CAL, k) for k in arms},
              'dev5': {k: _group(per_source, DEV, k) for k in arms},
              'dev5_paired_vs_q1': {k: _paired(per_source, DEV, k) for k in
                                    ('ridge_math_sse', 'ridge_replay_sse', 'icc_gated_sse')},
              'cal4_paired_vs_q1': {k: _paired(per_source, CAL, k) for k in
                                    ('ridge_math_sse', 'ridge_replay_sse', 'icc_gated_sse')},
              'omega_dev': {s: per_source[s]['omega'] for s in DEV},
              'ridge_main_info': info_main, 'ridge_robust_info': info_rob,
              'ridgeD_bf16_sha256': sha(dpath),
              'icc_factor_sha256': iccman[site]['sha256'],
              'q1_matrix_sha256': q1man[site]['sha256'],
              'matched_counts': {s: per_source[s]['rows'] for s in allsrc}}
    write_json(out / ('expensive_%s.json' % stem), result)
    del w1, b1, w2, b2, a, a_free, d_main, mu, v, pairs, x_list, y_list, r_list, icc_factors
    torch.cuda.empty_cache()
    check_budget()
    return result


def _evaluate_trigger(landscape, expensive, scope):
    eps = scope['epsilon_num']
    expert = 'high'
    diffs = [landscape['%s/%d' % (expert, l)]['cal4']['reuse_y_gated_sse']['mse']
             - landscape['%s/%d' % (expert, l)]['cal4']['q1_math_gated_sse']['mse']
             for l in range(40)]
    med = median(diffs)
    high_sites = [s for s in scope['expensive_sites'] if s.startswith('high/')]
    ridge_helps = [s for s in high_sites
                   if expensive[s]['dev5']['ridge_math_sse']['mse']
                   < expensive[s]['dev5']['q1_math_sse']['mse'] - eps]
    fires = (med <= 0.0) and len(ridge_helps) > 0
    return {'fires': fires, 'expert': expert,
            'median_reuse_minus_q1_mse': med,
            'ridge_helps_dev_sites': ridge_helps,
            'epsilon_num': eps,
            'rule': 'fit supplementary high/0 iff high-expert 40-layer median '
                    '(reuseY - Q1) CAL4 math MSE <= 0 AND >=1 high expensive site has '
                    'FREE_RIDGE main DEV5 math MSE below Q1 by > epsilon_num'}


def _summarize(landscape, expensive, supplementary, trigger, scope):
    def expert_landscape(expert):
        sites = ['%s/%d' % (expert, l) for l in range(40)]
        out = {}
        for cohort in ('cal4', 'dev5'):
            for arm in ('q1_math_gated_sse', 'q1_replay_gated_sse', 'reuse_y_gated_sse'):
                vals = [landscape[s][cohort][arm]['mse'] for s in sites]
                out['%s_%s_mean_mse' % (cohort, arm)] = sum(vals) / len(vals)
                out['%s_%s_median_mse' % (cohort, arm)] = median(vals)
        return out

    summary = {'scope_id': scope['scope_id'],
               'arms': scope['arms'],
               'landscape_high': expert_landscape('high'),
               'landscape_low': expert_landscape('low'),
               'expensive_sites': expensive,
               'supplementary_site': supplementary,
               'supplementary_trigger': trigger,
               'readout_note': 'source-equal-weight current-gate MSE; math lift (FP32) and '
                               'BF16 cache/gate replay reported separately; full-M residual '
                               'writeback NOT represented (small-M proxy boundary); '
                               'increment-target and full-Y normalization auxiliary; DEV5 is '
                               'development source-out, NOT blind confirmation.'}
    return summary