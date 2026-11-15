"""G20 BWS composite native-state four-cell engine (rev5 handoff B7).

Invoked ONLY by guarded_runner_bws after allocation/review/ACTIVE/hash verification.
Fixed CAL3 (vbi_1005/0895/0648) + DEV3 (vbi_1013/0886/0800); two weights W_O/W_D;
two composite native states mu_D (landed D4 bank) and mu_O (this scope's original40
capture); three methods Q1/ICC/GLOBAL_P64. Per W/site fit ONE Q1 on the mixed CAL3
(each source D_first256 + O_first256, six source-state halves equal at 256 rows),
then fix params and score all four DEV cells with the same theta(W) and the
same-source D4 hit gate G_ref. ICC factors weight-only; Chip uses the same
OWNW_LOCAL512_P64ANCHOR64_V1 adapter (mu_D fixed64 from B_CHIP_P64_INPUTS, mu_O
fixed64 from this DEV3 O40 capture).

Estimand: I_m = [d_m(W_D,mu_D)-d_m(W_O,mu_D)] - [d_m(W_D,mu_O)-d_m(W_O,mu_O)],
d_m(W,mu)=R_m(W,mu)-R_Q1(W,mu). Aggregation layer-equal->expert-equal->source-equal;
3 paired source values + equal-source mean/range. Primary F64 gated pre-gate SSE;
auxiliary BF16 gate MSE separate. n=3 is mechanism development; no bootstrap/p-value.
Composite interaction reported; pure interval/CFG/state components UNIDENTIFIED
without matching controls. Sampled local FFN replay; quality NOT_MEASURED.

<=6 new O40 (one dense original40 pass per condition), 0 new D4, 0 video decode/
scoring, 0 new pairs. Module-level functions so the author smoke exercises the EXACT
formal integration path (O40 capture -> mixed-CAL fit -> four-cell score -> aggregate
-> serialize) on one real source before dispatch.
"""
import csv
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from runtime_guard import sha256_file as sha
import panel512
import b_common_replay as B
from b_capture_o40 import (NativeTapO40, O40Sink, regenerate_fixed64,
                           EXPECTED_GRID, FULL_M, FIXED64_COUNT, PANEL_COUNT, SITES)

DIM = 5120
WEIGHTS = ('W_O', 'W_D')
METHODS = ('Q1', 'ICC', 'Chip')
METHOD_ID = {'Q1': 'Q1R2_CAL3_MIXED_P512_OWNW', 'ICC': 'ICC_SVD_R512_TWO_RESIDUAL',
             'Chip': 'CHIPMUNK_GLOBAL_P64'}
METHOD_SHORT = {v: k for k, v in METHOD_ID.items()}
ICC_FACTOR_SHAPES = ((13824, 512), (512, 5120), (5120, 512), (512, 13824))
CAL_IDS = ['vbi_1005', 'vbi_0895', 'vbi_0648']
DEV_IDS = ['vbi_1013', 'vbi_0886', 'vbi_0800']
FIT_PREFIX = 256
EXPERTS = ('high', 'low')
LAYERS = (4, 19, 34)


def exclusive_json(path, data):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def sha_tensor(t):
    b = t.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8)
    return hashlib.sha256(b.numpy().tobytes()).hexdigest()


def finite_tensor_sha(t, torch):
    t = t.detach().contiguous()
    if not torch.isfinite(t).all().item():
        raise RuntimeError('nonfinite initial noise')
    return {'sha256': hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
            'shape': list(t.shape), 'dtype': str(t.dtype)}


def verify_bws_conditions(conds):
    if len(conds) != 6:
        raise RuntimeError('exact 6 BWS conditions (CAL3+DEV3) required')
    by_role = {'CAL': [], 'DEV': []}
    for c in conds:
        by_role[c['cycle_role']].append(c['record_id'])
        if c['seed'] != 42 or c['orient'] != 'landscape':
            raise RuntimeError('condition seed/orient ' + c['record_id'])
        if hashlib.sha256(c['prompt'].encode('utf8')).hexdigest() != c['prompt_sha256']:
            raise RuntimeError('prompt hash ' + c['record_id'])
        if sha(c['image_path']) != c['image_sha256']:
            raise RuntimeError('image changed ' + c['record_id'])
    if sorted(by_role['CAL']) != sorted(CAL_IDS) or sorted(by_role['DEV']) != sorted(DEV_IDS):
        raise RuntimeError('BWS cohort must be exactly CAL3+DEV3')
    if len({c['source_cluster_id'] for c in conds}) != 6:
        raise RuntimeError('clusters not separated')


def verify_pipeline_binding(scope):
    """Hash-gate the original-model base assets + 12 weight shards + LightX2V tree."""
    base = Path(scope['base_model_dir']).resolve()
    for relative, spec in scope['base_assets_sha256'].items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size != spec['size']:
            raise RuntimeError('base asset path/size ' + relative)
        if sha(path) != spec['sha256']:
            raise RuntimeError('base asset SHA ' + relative)
    for spec in scope['original_weight_files']:
        path = Path(spec['path'])
        if path.stat().st_size != spec['size'] or sha(path) != spec['sha256']:
            raise RuntimeError('original shard payload ' + spec['path'])
    expected = scope['external_sources_sha256']
    for tree in scope['python_source_roots']:
        paths = {str(p.resolve()) for p in Path(tree).rglob('*.py') if p.is_file()}
        expected_paths = {str(Path(p).resolve()) for p in expected
                          if Path(p).resolve().is_relative_to(Path(tree).resolve()) and p.endswith('.py')}
        if not paths or paths != expected_paths:
            raise RuntimeError('source tree inventory ' + tree)
    for p, d in expected.items():
        if sha(p) != d:
            raise RuntimeError('pipeline source SHA ' + p)


def build_checked_pipeline(scope, out):
    """Build the original-model dense pipeline through the proven producer builder
    (verbatim normal_pipeline, hash-gated). aspect_ratio 16:9 + width 832 forces
    actual height 464; the capture hook asserts grid 21x29x52 at runtime. Shared by
    run() and the smoke so the exact checked path is exercised before dispatch."""
    from bws_pipeline import build_normal_pipeline
    pipe, config = build_normal_pipeline(scope['base_model_dir'], scope['required_runtime_config'])
    for k, v in scope['required_runtime_config'].items():
        if config.get(k) != v:
            raise RuntimeError('runtime config differs: ' + k)
    exclusive_json(out / 'EFFECTIVE_CONFIG.json', dict(config))
    return pipe


def load_weights(scope, weight_id, expert, layer, device):
    from safetensors import safe_open
    sk = B.site_key(expert, layer)
    ck = scope['checkpoints_W_D'][expert] if weight_id == 'W_D' else scope['checkpoints_W_O'][sk]
    if sha(ck['path']) != ck['sha256']:
        raise RuntimeError('checkpoint provenance ' + weight_id + ' ' + sk)
    with safe_open(ck['path'], framework='pt', device='cpu') as f:
        w1, b1, w2, b2 = [f.get_tensor('blocks.%d.ffn.%s' % (layer, n))
                          for n in ('0.weight', '0.bias', '2.weight', '2.bias')]
    if [tuple(t.shape) for t in (w1, b1, w2, b2)] != [(13824, 5120), (13824,), (5120, 13824), (5120,)]:
        raise RuntimeError('ffn weight shape ' + weight_id + ' ' + sk)
    import torch
    if weight_id == 'W_D':
        if not all(t.dtype == torch.bfloat16 for t in (w1, b1, w2, b2)):
            raise RuntimeError('W_D dtype')
    else:
        if not all(t.dtype == torch.float32 for t in (w1, b1, w2, b2)):
            raise RuntimeError('W_O storage dtype must be F32')
        w1, b1, w2, b2 = [t.bfloat16() for t in (w1, b1, w2, b2)]
    return [t.to(device) for t in (w1, b1, w2, b2)]


def capture_source_o40(cond, scope, scope_sha, o40_root, byproduct, fixed64_ids, budget,
                       check_resources, pipe, scheduler):
    """One dense original40 trajectory for a BWS condition; capture PANEL512 U at the
    first adjacent same-expert endpoint pair (six sites) via NativeTapO40; DEV also
    captures fixed64. Records actual timestep/noise/gap. Byproduct mp4 NOT decoded."""
    import torch
    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from lightx2v.common.ops.mm.mm_weight import MMWeight
    if MMWeight.apply.__module__ != 'lightx2v.common.ops.mm.mm_weight' or MMWeight.apply.__qualname__ != 'MMWeight.apply':
        raise RuntimeError('dense MMWeight patched')
    check_resources()
    src = cond['record_id']
    is_dev = cond['cycle_role'] == 'DEV'
    video = byproduct / ('%s__o40.mp4' % src)
    cap_dir = o40_root / src
    sink = O40Sink(torch, cap_dir, budget['capture_bytes_per_source'])

    def guarded_sink(event, payload):
        check_resources()
        event.update(image_sha256=cond['image_sha256'], prompt_sha256=cond['prompt_sha256'],
                     cycle_role=cond['cycle_role'], source_cluster_id=cond['source_cluster_id'],
                     scope_sha256=scope_sha)
        sink(event, payload)
        check_resources()

    tap = NativeTapO40(torch, guarded_sink, fixed64_ids, is_dev, MMWeight, EXPECTED_GRID)
    tap.begin_source(src, cond['seed'])
    tap.install(WanTransformerInfer)
    dispatch = pipe.runner.model
    old_dispatch = dispatch.get_current_model_index
    noise = []
    old_prepare = scheduler.prepare_latents

    def prepare_tap(*a, **k):
        result = old_prepare(*a, **k)
        if not noise:
            noise.append(finite_tensor_sha(scheduler.latents, torch))
        return result

    def dispatched(*a, **k):
        value = old_dispatch(*a, **k)
        step = int(scheduler.step_index)
        expert = int(dispatch.cur_model_index)
        times = scheduler.timesteps
        ts = float(times[step].item() if hasattr(times[step], 'item') else times[step])
        tap.dispatch = expert
        tap.record_step(step, expert, ts)
        return value

    scheduler.prepare_latents = prepare_tap
    dispatch.get_current_model_index = dispatched
    scheduler.generator = None
    start = time.monotonic()
    try:
        with torch.no_grad():
            pipe.generate(seed=cond['seed'], image_path=cond['image_path'], prompt=cond['prompt'],
                          negative_prompt='', save_result_path=str(video))
        tap.finish_source()
    finally:
        tap.uninstall()
        dispatch.get_current_model_index = old_dispatch
        scheduler.prepare_latents = old_prepare
    elapsed = time.monotonic() - start
    if not video.is_file() or not video.stat().st_size:
        raise RuntimeError('byproduct video absent ' + src)
    if len(noise) != 1:
        raise RuntimeError('initial noise coverage ' + src)
    n_sites = len(list(cap_dir.glob('*.pt')))
    if n_sites != 12:
        raise RuntimeError('capture site count %d != 12 for %s' % (n_sites, src))
    # endpoint metadata (actual timestep/gap per expert)
    endpoints = {}
    for e in (0, 1):
        st = tap.expert_steps[e]
        name = 'high' if e == 0 else 'low'
        endpoints[name] = {'anchor_step': st[0], 'hit_step': st[1],
                           'anchor_timestep': tap.step_timestep[st[0]],
                           'hit_timestep': tap.step_timestep[st[1]],
                           'gap': tap.step_timestep[st[0]] - tap.step_timestep[st[1]]}
    check_resources()
    return {'record_id': src, 'cycle_role': cond['cycle_role'], 'wall_seconds': elapsed,
            'initial_noise': noise[0], 'endpoints': endpoints, 'capture_sites': n_sites,
            'expert_step_count': {('high' if e == 0 else 'low'): len(tap.expert_steps[e]) for e in (0, 1)}}


def load_mu_d_pack(bank_root, sources, expert, layer, device, prefix=None):
    """mu_D from the landed D4 bank; G_ref = D4 hit gate. own-W targets filled later."""
    import torch
    a_step, h_step = B.EXPERT_STEP[expert]
    pack = []
    for src in sources:
        u0, _, _ = B.load_site(bank_root, src, a_step, layer)
        u1, _, g = B.load_site(bank_root, src, h_step, layer)
        n = prefix or u0.shape[0]
        u0b = u0[:n].to(device=device, dtype=torch.bfloat16)
        u1b = u1[:n].to(device=device, dtype=torch.bfloat16)
        gf = g.to(device=device, dtype=torch.float32)
        pack.append({'source': src, 'state_mu': 'mu_D', 'u0_bf16': u0b, 'u1_bf16': u1b,
                     'u0': u0b.float(), 'u1': u1b.float(), 'X': u1b.float() - u0b.float(),
                     'g': gf, 'n': n})
    return pack


def load_mu_o_pack(o40_root, sources, expert, layer, g_ref_by_source, device, prefix=None):
    """mu_O from this scope's O40 capture; U from O40, G_ref = same-source D4 hit gate."""
    import torch
    a_step = None
    pack = []
    # endpoint steps come from the capture sidecar (first adjacent pair per expert)
    for src in sources:
        files = sorted((o40_root / src).glob('*__b%02d.pt' % layer))
        sides = [json.loads(Path(str(f) + '.json').read_text()) for f in files]
        sides = [s for s in sides if s['expert'] == expert]
        anchor = next(s for s in sides if s['endpoint_role'] == 'anchor')
        hit = next(s for s in sides if s['endpoint_role'] == 'hit')
        da = torch.load(o40_root / src / anchor['file'], map_location='cpu', weights_only=True)
        dh = torch.load(o40_root / src / hit['file'], map_location='cpu', weights_only=True)
        u0, u1 = da['U_panel'], dh['U_panel']
        n = prefix or u0.shape[0]
        u0b = u0[:n].to(device=device, dtype=torch.bfloat16)
        u1b = u1[:n].to(device=device, dtype=torch.bfloat16)
        gf = g_ref_by_source[src].to(device=device, dtype=torch.float32)
        pack.append({'source': src, 'state_mu': 'mu_O', 'u0_bf16': u0b, 'u1_bf16': u1b,
                     'u0': u0b.float(), 'u1': u1b.float(), 'X': u1b.float() - u0b.float(),
                     'g': gf, 'n': n,
                     'o40_anchor_timestep': anchor['timestep'], 'o40_hit_timestep': hit['timestep']})
    return pack


def g_ref_from_bank(bank_root, sources, expert, layer, device):
    import torch
    _, h_step = B.EXPERT_STEP[expert]
    out = {}
    for src in sources:
        _, _, g = B.load_site(bank_root, src, h_step, layer)
        out[src] = g.to(device=device, dtype=torch.float32)
    return out


def slice_pack(pack, n):
    import torch
    out = []
    row_keys = ('u0_bf16', 'u1_bf16', 'u0', 'u1', 'X', 'y0', 'y1', 'z0', 'h0', 'z1', 'h1')
    for c in pack:
        d = dict(c)
        for k in row_keys:
            if k in c:
                d[k] = c[k][:n]
        d['n'] = n
        out.append(d)
    return out


def persist_and_reload(out, sk, weight_id, tensors, device):
    import torch
    pdir = out / 'params'; pdir.mkdir(exist_ok=True)
    ppath = pdir / ('%s__%s.pt' % (sk, weight_id))
    cpu = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    torch.save(cpu, ppath)
    back = torch.load(ppath, map_location='cpu', weights_only=True)
    if set(back) != set(cpu):
        raise RuntimeError('param reload key mismatch ' + sk)
    for k in cpu:
        if not torch.equal(cpu[k], back[k]):
            raise RuntimeError('param reload mismatch ' + sk + ' ' + k)
    manifest = {'path': str(ppath), 'bytes': ppath.stat().st_size, 'sha256': sha(ppath),
                'weights_only_reloadable': True,
                'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in cpu.items()}}
    return {k: back[k].to(device) for k in back}, manifest


def cell_row(c, expert, layer, weight_id, method_id, pred, indices, scope_sha, fit_sha, row_indices_sha):
    prim = B.primary_risk(pred, c)
    aux = B.auxiliary_risk(pred, c)
    return {'source_id': c['source'], 'role': 'DEV', 'expert': expert, 'layer': layer,
            'state_mu': c['state_mu'], 'endpoint_steps': '%d->%d' % B.EXPERT_STEP[expert],
            'weight_id': weight_id, 'method_id': method_id, 'config_sha': scope_sha,
            'fit_sha': fit_sha, 'U0_sha': sha_tensor(c['u0_bf16']), 'U1_sha': sha_tensor(c['u1_bf16']),
            'row_indices_sha': row_indices_sha, 'G_ref_sha': sha_tensor(c['g']),
            'own_anchor_sha': sha_tensor(c['y0']), 'own_target_sha': sha_tensor(c['y1']),
            'numerical_recipe_sha': scope_sha, 'n_rows': c['n'], 'output_dim': DIM,
            'pred_sha': sha_tensor(pred), 'gated_sse_f64': prim * c['n'] * DIM, 'gated_mse_f64': prim,
            'bf16_gate_mse': aux,
            'chip_indices_sha': sha_tensor(indices) if indices is not None else None,
            'chip_selected_k': B.P64_K if indices is not None else None,
            'chip_proxy_rows_sha': None, 'cell_status': 'VALID'}


def aggregate_bws(cells, dev_sources, weights=WEIGHTS, experts=EXPERTS, layers=LAYERS):
    """Per-source/W/mu/method risk; d_m(W,mu); I_m interaction; source-equal mean/range.
    Grid dims default to the formal full grid; the smoke passes a reduced grid to
    exercise this exact aggregation path on real reduced cells."""
    risk = {}
    for row in cells:
        if row['cell_status'] != 'VALID':
            continue
        m = METHOD_SHORT[row['method_id']]
        risk[(row['source_id'], row['weight_id'], row['state_mu'], m, row['expert'], row['layer'])] = row['gated_mse_f64']
    per_source = {}
    for src in dev_sources:
        r = {}
        for w in weights:
            for mu in ('mu_D', 'mu_O'):
                for m in METHODS:
                    by_e = {}
                    for e in experts:
                        vals = [risk[(src, w, mu, m, e, L)] for L in layers]
                        by_e[e] = sum(vals) / len(vals)
                    r[(w, mu, m)] = sum(by_e.values()) / len(experts)
        d = {(m, w, mu): r[(w, mu, m)] - r[(w, mu, 'Q1')] for m in ('ICC', 'Chip') for w in weights for mu in ('mu_D', 'mu_O')}
        I = {m: (d[(m, 'W_D', 'mu_D')] - d[(m, 'W_O', 'mu_D')]) - (d[(m, 'W_D', 'mu_O')] - d[(m, 'W_O', 'mu_O')])
             for m in ('ICC', 'Chip')}
        per_source[src] = {'risk': {'%s|%s|%s' % (w, mu, m): r[(w, mu, m)] for w in weights for mu in ('mu_D', 'mu_O') for m in METHODS},
                           'd': {'%s|%s|%s' % (m, w, mu): d[(m, w, mu)] for m in ('ICC', 'Chip') for w in weights for mu in ('mu_D', 'mu_O')},
                           'I': I}
    mean_risk = {'%s|%s|%s' % (w, mu, m): sum(per_source[s]['risk']['%s|%s|%s' % (w, mu, m)] for s in dev_sources) / len(dev_sources)
                 for w in weights for mu in ('mu_D', 'mu_O') for m in METHODS}
    mean_I = {m: sum(per_source[s]['I'][m] for s in dev_sources) / len(dev_sources) for m in ('ICC', 'Chip')}
    range_I = {m: [min(per_source[s]['I'][m] for s in dev_sources), max(per_source[s]['I'][m] for s in dev_sources)] for m in ('ICC', 'Chip')}
    return {'per_source': per_source, 'source_equal_mean_risk': mean_risk,
            'source_equal_mean_I': mean_I, 'source_range_I': range_I}


def fit_and_score_site(expert, layer, scope, scope_sha, bank_root, o40_root, chip_mu_d,
                       fixed64_ids, icc_man, bindings, out, device, checks, params_manifest, cells):
    """Mixed-CAL3 Q1 fit per W (six source-state halves equal at 256 rows), ICC factors,
    then four-cell DEV3 scoring (2W x 2mu x Q1/ICC/Chip) with fixed params + D4 G_ref."""
    import torch
    sk = B.site_key(expert, layer)
    row_sha = {s: bindings['panel']['per_source'][s]['token_id_sha256_le_int64'] for s in CAL_IDS + DEV_IDS}
    g_ref = g_ref_from_bank(bank_root, CAL_IDS + DEV_IDS, expert, layer, device)
    for weight_id in WEIGHTS:
        w1, b1, w2, b2 = load_weights(scope, weight_id, expert, layer, device)
        # mixed CAL: per source D_first256 + O_first256; own-W on full 512 then prefix
        cal_d = load_mu_d_pack(bank_root, CAL_IDS, expert, layer, device)
        cal_o = load_mu_o_pack(o40_root, CAL_IDS, expert, layer, g_ref, device)
        B.own_w_targets(cal_d, w1, b1, w2, b2)
        B.own_w_targets(cal_o, w1, b1, w2, b2)
        mixed = slice_pack(cal_d, FIT_PREFIX) + slice_pack(cal_o, FIT_PREFIX)
        for c in mixed:
            c['Y'] = c['y1'].float() - c['y0'].float()
        alpha_q0, a0, z_norm = B.q0_prior(mixed, w1, b1, w2)
        a_q1, alpha_q1, q1_info = B.fit_q1(mixed, w1, w2, a0, alpha_q0)
        conv = q1_info['status'] == 'CONVERGED' and q1_info['true_relative_residual'] <= B.Q1_TOL
        checks['q1_convergence'][sk + '|' + weight_id] = {
            'status': q1_info['status'], 'true_relative_residual': q1_info['true_relative_residual'],
            'iterations': q1_info['iterations'], 'converged': conv, 'fit': 'CAL3_mixed_D256_O256_six_halves'}
        if not conv:
            raise RuntimeError('Q1 not converged ' + sk + ' ' + weight_id)
        a_q1_f32 = (w1.float().T * alpha_q1) @ w2.float().T
        if weight_id == 'W_D':
            frec = icc_man['%s/%d' % (expert, layer)]
            (f1l, f1r, f2l, f2r), fobj = B.load_w_d_factors(frec['path'], device)
            ident = fobj['checkpoint_sha256'] == scope['checkpoints_W_D'][expert]['sha256'] and fobj['method'] == 'svd' and fobj['rank'] == 512
            checks['icc_factor_identity'][sk + '|' + weight_id] = {'source': 'reused_W_D_factor', 'checkpoint_match': ident, 'method': fobj['method'], 'rank': fobj['rank']}
            if not ident or sha(frec['path']) != frec['sha256']:
                raise RuntimeError('ICC W_D factor identity/hash ' + sk)
            if [tuple(t.shape) for t in (f1l, f1r, f2l, f2r)] != list(ICC_FACTOR_SHAPES) or not all(t.dtype == torch.bfloat16 for t in (f1l, f1r, f2l, f2r)):
                raise RuntimeError('ICC W_D factor shape/dtype ' + sk)
        else:
            f1, f2 = B.make_w_o_factors(w1, w2, device)
            f1l, f1r, f2l, f2r = f1.left, f1.right, f2.left, f2.right
            checks['icc_factor_identity'][sk + '|' + weight_id] = {'source': 'fresh_W_O_ordinary_svd', 'method': 'svd', 'rank': 512, 'driver': 'gesvd'}
            if [tuple(t.shape) for t in (f1l, f1r, f2l, f2r)] != list(ICC_FACTOR_SHAPES):
                raise RuntimeError('ICC W_O factor shape ' + sk)
        tensors = {'alpha_Q1': alpha_q1.float(), 'A_q1_f32': a_q1_f32,
                   'icc_fc1_left': f1l, 'icc_fc1_right': f1r, 'icc_fc2_left': f2l, 'icc_fc2_right': f2r}
        reloaded, pman = persist_and_reload(out, sk, weight_id, tensors, device)
        a_q1_f32 = reloaded['A_q1_f32']
        f1l, f1r, f2l, f2r = (reloaded['icc_fc1_left'], reloaded['icc_fc1_right'], reloaded['icc_fc2_left'], reloaded['icc_fc2_right'])
        q1_fit_sha = sha_tensor(reloaded['alpha_Q1'])
        # four DEV cells per mu (full 512, own-W targets, D4 G_ref)
        for mu, loader in (('mu_D', lambda: load_mu_d_pack(bank_root, DEV_IDS, expert, layer, device)),
                           ('mu_O', lambda: load_mu_o_pack(o40_root, DEV_IDS, expert, layer, g_ref, device))):
            dev = loader()
            B.own_w_targets(dev, w1, b1, w2, b2)
            for c in dev:
                pred_q1 = B.q1_predict(c, a_q1_f32)
                cells.append(cell_row(c, expert, layer, weight_id, METHOD_ID['Q1'], pred_q1, None, scope_sha, q1_fit_sha, row_sha[c['source']]))
                pred_icc = B.icc_predict(c, f1l, f1r, f2l, f2r)
                cells.append(cell_row(c, expert, layer, weight_id, METHOD_ID['ICC'], pred_icc, None, scope_sha, None, row_sha[c['source']]))
                u0_64, u1_64 = chip_fixed64_rows(chip_mu_d, o40_root, c['source'], expert, layer, mu, c, device)
                pred_chip, indices = B.chip_predict(c, w1, b1, w2, u0_64, u1_64, fixed64_ids)
                cells.append(cell_row(c, expert, layer, weight_id, METHOD_ID['Chip'], pred_chip, indices, scope_sha, None, row_sha[c['source']]))
            del dev
        params_manifest[sk + '|' + weight_id] = {'params_file': pman,
            'Q1': {'alpha_Q1_sha256': q1_fit_sha, 'A_q1_fro2': float(a_q1_f32.square().sum()), 'q1_fit': q1_info},
            'ICC': checks['icc_factor_identity'][sk + '|' + weight_id]}
        del cal_d, cal_o, mixed, w1, b1, w2, b2, a_q1, a_q1_f32, a0, alpha_q1, alpha_q0, tensors, reloaded, f1l, f1r, f2l, f2r
        torch.cuda.empty_cache()
        print('BWS_FIT site=%s weight=%s params_bytes=%d' % (sk, weight_id, pman['bytes']), flush=True)


def chip_fixed64_rows(chip_mu_d, o40_root, src, expert, layer, mu, c, device):
    """Return (u0_64,u1_64) BF16 fixed64 rows for this source/site/mu. mu_D from
    B_CHIP_P64_INPUTS D4 capture; mu_O from this O40 capture."""
    import torch
    a_step, h_step = B.EXPERT_STEP[expert]
    if mu == 'mu_D':
        rec = chip_mu_d['sources'][src]['sites'][B.site_key(expert, layer)]
        rows = []
        for role in ('anchor', 'hit'):
            t = torch.load(rec[role]['path'], map_location='cpu', weights_only=True)
            rows.append(t['U_fixed64'].to(device=device, dtype=torch.bfloat16))
        return rows[0], rows[1]
    files = sorted((o40_root / src).glob('*__b%02d.pt' % layer))
    sides = [json.loads(Path(str(f) + '.json').read_text()) for f in files]
    sides = [s for s in sides if s['expert'] == expert]
    out = []
    for role in ('anchor', 'hit'):
        side = next(s for s in sides if s['endpoint_role'] == role)
        t = torch.load(o40_root / src / side['file'], map_location='cpu', weights_only=True)
        out.append(t['U_fixed64'].to(device=device, dtype=torch.bfloat16))
    return out[0], out[1]


def run(stage, scope, scope_sha):
    stage = Path(stage)
    budget = scope['budget']
    out = stage / 'output'; out.mkdir(exist_ok=False)
    started = time.monotonic()
    conds = json.loads((stage / 'bws_conditions.json').read_text())
    verify_bws_conditions(conds)
    verify_pipeline_binding(scope)
    bindings = json.loads((stage / 'B_BINDINGS.json').read_text())
    if sha(stage / 'B_BINDINGS.json') != scope['frozen_inputs_sha256']['B_BINDINGS.json']:
        raise RuntimeError('B_BINDINGS provenance')
    bank_root = Path(scope['bank_root'])
    if sha(bank_root.parent / 'BANK_MANIFEST.json') != scope['bank_manifest_sha256']:
        raise RuntimeError('bank manifest SHA')
    chip_mu_d = json.loads(Path(scope['chip_p64_inputs']).read_text())
    if sha(scope['chip_p64_inputs']) != scope['chip_p64_inputs_sha256']:
        raise RuntimeError('B_CHIP_P64_INPUTS provenance')
    if chip_mu_d['capture_scope_sha256'] != scope['chip_stats_scope_sha256']:
        raise RuntimeError('chip inputs scope identity')
    icc_man = json.loads(Path(scope['icc_factor_manifest']).read_text())
    if sha(scope['icc_factor_manifest']) != scope['icc_factor_manifest_sha256']:
        raise RuntimeError('ICC manifest provenance')
    p64 = bindings['p64']
    fixed64_ids = p64['fixed64_token_ids']

    def check_budget():
        if time.monotonic() - started > budget['soft_seconds']:
            raise TimeoutError('soft time budget')
        import resource
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > budget['host_memory_gib'] * 1024 ** 3:
            raise MemoryError('host peak RSS budget')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > budget['output_bytes']:
            raise RuntimeError('output limit')

    for path in reversed(scope['python_import_paths']):
        sys.path.insert(0, path)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA A100-SXM4-80GB':
        raise RuntimeError('requires one A100 80GB')
    if torch.__version__ != scope['runtime_versions']['torch'] or np.__version__ != scope['runtime_versions']['numpy']:
        raise RuntimeError('runtime version mismatch')
    if regenerate_fixed64(torch) != fixed64_ids:
        raise RuntimeError('regenerated fixed64 differs from binding')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    gpu_limit = budget['gpu_memory_gib'] * 1024 ** 3

    def check_resources():
        check_budget()
        if torch.cuda.max_memory_reserved() > gpu_limit:
            raise MemoryError('GPU_MEMORY_BUDGET_EXCEEDED')

    device = 'cuda'
    pipe = build_checked_pipeline(scope, out)
    check_resources()
    scheduler = pipe.runner.scheduler
    o40_root = out / 'o40_inputs'; o40_root.mkdir(exist_ok=False)
    byproduct = out / 'byproduct_video'; byproduct.mkdir(exist_ok=False)
    capture_meta = {}
    for cond in conds:
        meta = capture_source_o40(cond, scope, scope_sha, o40_root, byproduct, fixed64_ids, budget,
                                  check_resources, pipe, scheduler)
        capture_meta[cond['record_id']] = meta
        exclusive_json(out / ('O40_CAPTURE_%s.json' % cond['record_id']), meta)
        print('BWS_O40_CAPTURED', cond['record_id'], cond['cycle_role'], 'sites=%d' % meta['capture_sites'], flush=True)
    # release the capture pipeline before fitting (free GPU)
    del pipe, scheduler
    torch.cuda.empty_cache()

    cells = []
    checks = {'q1_convergence': {}, 'icc_factor_identity': {}, 'risks_finite': True,
              'params_reload_verified': True, 'own_w_recompute': True,
              'fit_identity': 'CAL3_mixed_D256_O256_six_halves_equal', 'G_ref': 'same-source D4 hit gate'}
    params_manifest = {}
    for expert, layer in SITES:
        fit_and_score_site(expert, layer, scope, scope_sha, bank_root, o40_root, chip_mu_d,
                           fixed64_ids, icc_man, bindings, out, device, checks, params_manifest, cells)
    for row in cells:
        if not torch.isfinite(torch.tensor([row['gated_mse_f64'], row['bf16_gate_mse']])).all():
            checks['risks_finite'] = False
            raise RuntimeError('nonfinite risk cell')
    agg = aggregate_bws(cells, DEV_IDS)
    elapsed = time.monotonic() - started
    fieldnames = list(cells[0].keys())
    with (out / 'BWS_CELL.csv').open('x', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for row in cells:
            w.writerow(row)
    with (out / 'BWS_SOURCE.csv').open('x', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        w.writerow(['source_id', 'I_ICC', 'I_Chip', 'd_ICC_W_D_mu_D', 'd_ICC_W_O_mu_D', 'd_ICC_W_D_mu_O', 'd_ICC_W_O_mu_O'])
        for src in DEV_IDS:
            ps = agg['per_source'][src]
            w.writerow([src, ps['I']['ICC'], ps['I']['Chip'], ps['d']['ICC|W_D|mu_D'], ps['d']['ICC|W_O|mu_D'],
                        ps['d']['ICC|W_D|mu_O'], ps['d']['ICC|W_O|mu_O']])
    n_q1icc = sum(1 for r in cells if METHOD_SHORT[r['method_id']] in ('Q1', 'ICC'))
    summary = {'scope_id': scope['scope_id'], 'status': 'BWS_THREE_METHODS_COMPLETE',
               'new_D4_full_executions': 0, 'new_O40_full_executions': len(conds),
               'new_video_decode': 0, 'new_video_scoring': 0,
               'trajectory_byproduct_videos_not_decoded': len(conds),
               'cal_ids': CAL_IDS, 'dev_ids': DEV_IDS, 'weights': list(WEIGHTS),
               'states': ['mu_D', 'mu_O'], 'methods_completed': list(METHODS),
               'sites': [B.site_key(e, L) for e, L in SITES],
               'dev_cells_valid': len(cells), 'dev_cells_expected_Q1ICC': 144, 'dev_cells_expected_all_methods': 216,
               'q1icc_cells': n_q1icc,
               'primary_estimand': 'F64 gated pre-gate SSE/(512*5120); layer-equal->expert-equal->source-equal',
               'contrast': 'I_m=[d_m(W_D,mu_D)-d_m(W_O,mu_D)]-[d_m(W_D,mu_O)-d_m(W_O,mu_O)]',
               'fit_identity': 'CAL3 mixed D256+O256 per source, six halves equal; params fixed across state cells',
               'G_ref': 'same-source/site D4 hit gate (both mu)',
               'component_claims': 'pure interval/CFG/state UNIDENTIFIED without matching controls; composite interaction reported',
               'quality_metrics': 'NOT_MEASURED', 'performance_comparison': 'NOT_BENCHMARKED',
               'elapsed_seconds': elapsed,
               'params_total_bytes': sum(v['params_file']['bytes'] for v in params_manifest.values()),
               'source_equal_mean_risk': agg['source_equal_mean_risk'],
               'source_equal_mean_I': agg['source_equal_mean_I'], 'source_range_I': agg['source_range_I'],
               'o40_endpoints': {s: m['endpoints'] for s, m in capture_meta.items()}}
    exclusive_json(out / 'PARAMS_MANIFEST.json', params_manifest)
    exclusive_json(out / 'CHECKS.json', checks)
    exclusive_json(out / 'BWS_AGGREGATE.json', agg)
    exclusive_json(out / 'SUMMARY.json', summary)
    print('BWS_COMPLETE elapsed=%.1fs cells=%d q1icc=%d O40=%d' % (elapsed, len(cells), n_q1icc, len(conds)), flush=True)
    return {'status': 'BWS_THREE_METHODS_COMPLETE', 'new_O40_full_executions': len(conds),
            'dev_cells_valid': len(cells), 'q1icc_cells': n_q1icc,
            'source_equal_mean_I': agg['source_equal_mean_I'],
            'elapsed_seconds': elapsed,
            'peak_rss_bytes': int(__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss) * 1024,
            'summary_sha256': sha(out / 'SUMMARY.json')}