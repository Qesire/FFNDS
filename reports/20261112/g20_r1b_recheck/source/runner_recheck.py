"""G20 R1b numerical recheck engine. Import and invoke run() ONLY after the
guarded runner has verified allocation, review, ACTIVE, hashes, node and budget.

Six pre-registered R1 sites, same CAL4/DEV5 cohorts, seed 42, same objective and
normalization. Per site:
  1. recompute the ORIGINAL R1 FP32 ridge solve (both lambdas) and check the
     BF16(D_main) export SHA against the frozen R1 artifact SHA;
  2. stable F64 sample-space solve (both lambdas) with F64 eigh cross-check;
  3. out-of-row-space energy of the original D;
  4. score Q1 / original-main / stable-main / stable-robust (math lift + BF16
     replay each) and the decomposed frozen ICC on every CAL+DEV source;
  5. save the ACTUALLY TESTED BF16 A_free matrices (orig/stable-main/stable-rob)
     plus the recomputed BF16 D_orig, all SHA-recorded;
  6. per-source identity comparison of recomputed Q1/orig/ICC/reuse SSEs against
     the bound R1 reference JSONs.
No 80-site landscape rerun, no supplementary trigger, no new generation, no
video, no benchmark, no lambda selection by pilot."""
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch
from safetensors import safe_open

from selector import load_pair, sha
from fit_core_v2 import (free_ridge_fit_fp32_original, stable_ridge_fit,
                         out_of_space_rel, omega_source,
                         icc_two_residual_decomposed, bf16_replay, native_ffn, D)
from selfcheck_v2 import run as selfcheck_run

CAL = ('chip10_02', 'chip10_04', 'chip10_07', 'chip10_09')
DEV = ('chip10_01', 'chip10_03', 'chip10_05', 'chip10_06', 'chip10_08')
STEPS = {'high': (0, 1), 'low': (2, 3)}
ARM_SSE = ('q1_math_sse', 'q1_replay_sse',
           'orig_main_math_sse', 'orig_main_replay_sse',
           'stable_main_math_sse', 'stable_main_replay_sse',
           'stable_rob_math_sse', 'stable_rob_replay_sse',
           'icc_gated_sse', 'reuse_y_gated_sse')
IDENTITY_MAP = {'q1_math_sse': 'q1_math_sse', 'q1_replay_sse': 'q1_replay_sse',
                'orig_main_math_sse': 'ridge_math_sse',
                'orig_main_replay_sse': 'ridge_replay_sse',
                'icc_gated_sse': 'icc_gated_sse',
                'reuse_y_gated_sse': 'reuse_y_gated_sse'}


def _mse(sse, n):
    return sse / (n * D)


def _group(per_source, sources, sse_key):
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


def _save_bf16_npy(path, tensor):
    np.save(path, tensor.bfloat16().view(torch.int16).cpu().numpy().view(np.uint16),
            allow_pickle=False)
    return sha(path)


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
    for expert in ('high', 'low'):
        ck = assets['checkpoints'][expert]
        if sha(ck['path']) != ck['sha256']:
            raise RuntimeError('checkpoint SHA ' + expert)
    allsrc = list(CAL) + list(DEV)
    sites = {}
    for site in scope['expensive_sites']:
        sites[site] = _recheck_site(site, assets, q1man, iccman, scope, out, check_budget)
        print('RECHECK', site, flush=True)
    summary = _summarize(sites, scope)
    summary['selfcheck'] = selfcheck
    write_json(out / 'SUMMARY.json', summary)
    check_budget()
    return {'sites': list(sites),
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


def _recheck_site(site, assets, q1man, iccman, scope, out, check_budget):
    expert, layer = site.split('/')
    layer = int(layer)
    stem = '%s_%02d' % (expert, layer)
    ref = json.loads((Path(__file__).resolve().parent / 'reference_r1'
                      / ('expensive_%s.json' % stem)).read_text())
    ck = assets['checkpoints'][expert]
    w1, b1, w2, b2 = _load_weights(ck['path'], expert, layer)
    a = _load_q1(q1man[site], site)
    allsrc = list(CAL) + list(DEV)
    pairs = {}
    for source in allsrc:
        check_budget()
        pairs[source] = load_pair(assets, source, 42, expert, STEPS[expert], ck['sha256'], layer)
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
    # 1. Original R1 FP32 solver, verbatim recomputation (both lambdas).
    d_orig_main, mu_o, v_o, info_o_main = free_ridge_fit_fp32_original(
        x_list, y_list, r_list, g, n_list, a, lam_main)
    d_orig_rob, _, _, info_o_rob = free_ridge_fit_fp32_original(
        x_list, y_list, r_list, g, n_list, a, lam_rob)
    d_orig_sha = _save_bf16_npy(out / ('ridgeD_orig_recomputed_%s.npy' % stem), d_orig_main)
    sha_match = (d_orig_sha == ref['ridgeD_bf16_sha256']
                 == scope['r1_reference']['ridgeD_bf16_sha256'][site])
    # 2. Stable F64 solves (both lambdas).
    d_st_main, bases_main, info_st_main = stable_ridge_fit(
        x_list, y_list, r_list, g, n_list, a, lam_main)
    d_st_rob, bases_rob, info_st_rob = stable_ridge_fit(
        x_list, y_list, r_list, g, n_list, a, lam_rob)
    for info in (info_st_main, info_st_rob):
        if info['cross_check_f64_eigh_rel_l2'] > 1e-7:
            raise RuntimeError('stable solver cross-check failed at ' + site)
        if info['D_out_of_rowspace_rel_fro'] > 1e-10:
            raise RuntimeError('stable D outside full sample space at ' + site)
    # 3. Out-of-CAL-subspace energy of the ORIGINAL D (F64, R1 Gram criterion).
    oos_orig_main = out_of_space_rel(d_orig_main, bases_main['calgram'])
    oos_orig_rob = out_of_space_rel(d_orig_rob, bases_rob['calgram'])
    # 4. Export the actually-tested BF16 matrices.
    a_free_orig = (a.float() + d_orig_main).bfloat16()
    a_free_st_main = (a.float() + d_st_main.float()).bfloat16()
    a_free_st_rob = (a.float() + d_st_rob.float()).bfloat16()
    for t in (a_free_orig, a_free_st_main, a_free_st_rob):
        if not torch.isfinite(t).all():
            raise ValueError('nonfinite exported A_free at ' + site)
    mat_sha = {
        'A_free_orig_main': _save_bf16_npy(out / ('A_free_orig_main_%s.npy' % stem), a_free_orig),
        'A_free_stable_main': _save_bf16_npy(out / ('A_free_stable_main_%s.npy' % stem), a_free_st_main),
        'A_free_stable_rob': _save_bf16_npy(out / ('A_free_stable_rob_%s.npy' % stem), a_free_st_rob),
        'ridgeD_orig_recomputed': d_orig_sha,
    }
    icc_factors = _load_icc(iccman, site)
    per_source = {}
    for source in allsrc:
        check_budget()
        p = pairs[source]
        u0, u1, gg = p['u0'], p['u1'], p['g']
        _, _, y0n = native_ffn(u0, w1, b1, w2, b2)
        _, _, y1n = native_ffn(u1, w1, b1, w2, b2)
        x = u1.float() - u0.float()
        n = u0.shape[0]
        q1_math = float(((y0n.float() + x @ a.float() - y1n.float()) * gg).double().square().sum())
        _, q1g, q1t = bf16_replay(u0, u1, y0n, y1n, gg, a)
        q1_rep = float((q1g.float() - q1t.float()).double().square().sum())

        def ridge_scores(dmat, amat):
            pred = y0n.float() + x @ (a.float() + dmat)
            math = float(((pred - y1n.float()) * gg).double().square().sum())
            _, rg, rt = bf16_replay(u0, u1, y0n, y1n, gg, amat)
            rep = float((rg.float() - rt.float()).double().square().sum())
            return math, rep

        om, orp = ridge_scores(d_orig_main, a_free_orig)
        sm, srp = ridge_scores(d_st_main.float(), a_free_st_main)
        rm, rrp = ridge_scores(d_st_rob.float(), a_free_st_rob)
        icc = icc_two_residual_decomposed(p, *icc_factors, w1, b1, w2, b2)
        rec = {'rows': n,
               'gated_dy_fro2': float(((y1n.float() - y0n.float()) * gg).double().square().sum()),
               'gated_y1_fro2': float((y1n.double() * gg.double()).square().sum()),
               'q1_math_sse': q1_math, 'q1_replay_sse': q1_rep,
               'orig_main_math_sse': om, 'orig_main_replay_sse': orp,
               'stable_main_math_sse': sm, 'stable_main_replay_sse': srp,
               'stable_rob_math_sse': rm, 'stable_rob_replay_sse': rrp,
               'icc_gated_sse': icc['math_gated_sse'],
               'icc_decomposition': {k: icc[k] for k in
                                     ('total_f64_gated_sse', 'stage1_propagated_gated_sse',
                                      'stage2_native_h_gated_sse', 'cross_term_gated',
                                      'decomposition_rel_residual', 'bf16_pred_rounding_rel',
                                      'stage1_pre_gelu_fro2', 'hidden_err_fro2',
                                      'zero_delta_max_abs', 'zero_delta_rel_l2',
                                      'zero_delta_gated_sse')},
               'reuse_y_gated_sse': float(((y0n.float() - y1n.float()) * gg).double().square().sum())}
        if source in DEV:
            rec['omega_stable'] = omega_source(x, bases_main['calgram'])
            rec['omega_r1_reference'] = ref['omega_dev'][source]
        per_source[source] = rec
        del p, u0, u1, gg, y0n, y1n, x
    # 5. Identity comparison against the bound R1 reference.
    identity = {'ridgeD_bf16_sha256_match': sha_match,
                'identity_warning': None if sha_match else
                'recomputed original BF16(D_main) SHA differs from frozen R1 artifact; '
                'continuing with the recomputed original as the pre-registered fallback',
                'ridgeD_bf16_sha256_recomputed': d_orig_sha,
                'ridgeD_bf16_sha256_r1': ref['ridgeD_bf16_sha256'],
                'per_source_sse_max_rel_diff': {},
                'ridge_main_info_diff': {}}
    for new_key, ref_key in IDENTITY_MAP.items():
        worst = 0.0
        for s in allsrc:
            a_v, b_v = per_source[s][new_key], ref['per_source'][s][ref_key]
            worst = max(worst, abs(a_v - b_v) / max(abs(b_v), 1e-30))
        identity['per_source_sse_max_rel_diff'][new_key] = worst
    for k in ('D_numerical_rank', 'gram_numerical_rank'):
        identity['ridge_main_info_diff'][k] = {
            'recomputed': info_o_main[k], 'r1': ref['ridge_main_info'][k],
            'equal': info_o_main[k] == ref['ridge_main_info'][k]}
    for k in ('D_fro2', 'D_max_singular', 'normal_relative_residual', 'ridge_over_ZA'):
        a_v, b_v = info_o_main[k], ref['ridge_main_info'][k]
        identity['ridge_main_info_diff'][k] = {
            'recomputed': a_v, 'r1': b_v,
            'rel_diff': abs(a_v - b_v) / max(abs(b_v), 1e-30)}
    arms = ARM_SSE
    result = {'site': site, 'expert': expert, 'layer': layer,
              'per_source': per_source,
              'cal4': {k: _group(per_source, CAL, k) for k in arms},
              'dev5': {k: _group(per_source, DEV, k) for k in arms},
              'dev5_paired_vs_q1': {k: _paired(per_source, DEV, k) for k in
                                    ('orig_main_math_sse', 'orig_main_replay_sse',
                                     'stable_main_math_sse', 'stable_main_replay_sse',
                                     'stable_rob_math_sse', 'stable_rob_replay_sse',
                                     'icc_gated_sse')},
              'cal4_paired_vs_q1': {k: _paired(per_source, CAL, k) for k in
                                    ('orig_main_math_sse', 'stable_main_math_sse',
                                     'stable_rob_math_sse', 'icc_gated_sse')},
              'omega_dev': {s: per_source[s]['omega_stable'] for s in DEV},
              'ridge_original_main_info': info_o_main,
              'ridge_original_robust_info': info_o_rob,
              'ridge_stable_main_info': info_st_main,
              'ridge_stable_robust_info': info_st_rob,
              'original_D_out_of_rowspace_rel': {'main': oos_orig_main, 'robust': oos_orig_rob},
              'identity_vs_r1': identity,
              'matrix_sha256': mat_sha,
              'icc_factor_sha256': iccman[site]['sha256'],
              'q1_matrix_sha256': q1man[site]['sha256'],
              'matched_counts': {s: per_source[s]['rows'] for s in allsrc},
              'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    write_json(out / ('expensive_recheck_%s.json' % stem), result)
    del w1, b1, w2, b2, a, a_free_orig, a_free_st_main, a_free_st_rob
    del d_orig_main, d_orig_rob, d_st_main, d_st_rob, mu_o, v_o, bases_main, bases_rob
    del pairs, x_list, y_list, r_list, icc_factors
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    check_budget()
    return result


def _summarize(sites, scope):
    table = {}
    for site, e in sites.items():
        icc_share = {}
        for k in ('stage1_propagated_gated_sse', 'stage2_native_h_gated_sse', 'cross_term_gated'):
            vals = [e['per_source'][s]['icc_decomposition'][k]
                    / max(e['per_source'][s]['icc_decomposition']['total_f64_gated_sse'], 1e-30)
                    for s in DEV]
            icc_share[k + '_dev_mean_share'] = sum(vals) / len(vals)
        table[site] = {
            'identity_sha_match': e['identity_vs_r1']['ridgeD_bf16_sha256_match'],
            'identity_max_sse_rel_diff': max(e['identity_vs_r1']['per_source_sse_max_rel_diff'].values()),
            'original_D_out_of_rowspace_rel_main': e['original_D_out_of_rowspace_rel']['main'],
            'stable_main_D_out_of_calgram_rel': e['ridge_stable_main_info']['D_out_of_calgram_rel_fro'],
            'stable_main_D_rank': e['ridge_stable_main_info']['D_numerical_rank'],
            'stable_rob_D_rank': e['ridge_stable_robust_info']['D_numerical_rank'],
            'stable_cross_check_rel_l2_main': e['ridge_stable_main_info']['cross_check_f64_eigh_rel_l2'],
            'dev5_q1_math_mse': e['dev5']['q1_math_sse']['mse'],
            'dev5_orig_main_math_mse': e['dev5']['orig_main_math_sse']['mse'],
            'dev5_stable_main_math_mse': e['dev5']['stable_main_math_sse']['mse'],
            'dev5_stable_rob_math_mse': e['dev5']['stable_rob_math_sse']['mse'],
            'dev5_paired_rel_vs_q1': {k: e['dev5_paired_vs_q1'][k]['mean_rel_change'] for k in
                                      ('orig_main_math_sse', 'stable_main_math_sse',
                                       'stable_rob_math_sse', 'icc_gated_sse')},
            'icc_dev_stage_shares': icc_share,
        }
    return {'scope_id': scope['scope_id'],
            'basis_scope_id': scope['r1_reference']['scope_id'],
            'sites': list(sites),
            'site_table': table,
            'all_identity_sha_match': all(e['identity_vs_r1']['ridgeD_bf16_sha256_match']
                                          for e in sites.values()),
            'readout_note': 'same registered objective/normalization as R1; stable F64 '
                            'sample-space solve with F64 eigh cross-check; both lambdas '
                            'scored on CAL4 and DEV5 (no pilot selection); original FP32 '
                            'solve recomputed and SHA-checked; ICC total split into '
                            'propagated stage-1 / stage-2-at-native-h / cross in F64; '
                            'DEV5 remains exposed development source-out, NOT blind; '
                            'small-M proxy boundary unchanged (no full-M writeback).'}