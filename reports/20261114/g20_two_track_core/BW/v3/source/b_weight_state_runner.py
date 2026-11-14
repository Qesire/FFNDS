"""B weight-state runner (NEW author code, rev5 handoff B6). BW mode: shared D4
PANEL512 states (mu_D), two weights W_O/W_D each with own-W recomputed targets,
three fixed methods Q1/ICC/GLOBAL_P64. This scope completes the Q1/ICC DEV cells
(11x6x2x2=264); Chip DEV cells are blocked by the fixed64 input gap (B_CHIP_STATS,
separate scope) and recorded as precise missing cells, not silently dropped.

d_m(W)=R_m(W)-R_Q1(W); theta_m=d_m(W_D)-d_m(W_O). Aggregation: layer-equal within
expert, expert-equal within source, then source-equal; 11 paired source values +
leave-one-source-out. high/low separate. Primary risk is F64 gated pre-gate SSE;
auxiliary BF16 gate MSE separate. No bootstrap/p-value gate; negative is a result.

Per handoff §5/A6 the fitted Q1/ICC parameters are saved as real reloadable tensors
(weights_only=True) and RELOADED before final scoring; PARAMS_MANIFEST.json records
path/bytes/sha256/tensor shapes, never norm/hash alone.
"""
import csv
import datetime
import json
from pathlib import Path

import torch

import b_common_replay as B
from runtime_guard import sha256_file as sha

D = B.D
WEIGHTS = ('W_O', 'W_D')
METHODS = ('Q1', 'ICC')   # Chip handled as blocked cells in this scope
# Cells record the full method_id; aggregation/output use the short method name.
METHOD_SHORT = {'Q1R2_CAL11_P512_OWNW': 'Q1', 'ICC_SVD_R512_TWO_RESIDUAL': 'ICC'}
ICC_FACTOR_SHAPES = ((13824, 512), (512, 5120), (5120, 512), (512, 13824))


def exclusive_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def sha_tensor(t):
    import hashlib
    # Raw payload bytes for any dtype (BF16 included); numpy() rejects BFloat16,
    # so reinterpret the contiguous buffer as uint8. Identical bytes to .numpy()
    # for FP32/FP64, hence backward-compatible with previously recorded hashes.
    b = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(b.numpy().tobytes()).hexdigest()


def persist_and_reload(out, sk, weight_id, tensors, device):
    """A6/§5: save real fitted tensors, reload from file (weights_only=True),
    verify byte-identical round-trip, return reloaded device tensors + manifest.
    Downstream scoring consumes the reloaded artifact, not a transient tensor."""
    pdir = out / 'params'
    pdir.mkdir(exist_ok=True)
    ppath = pdir / ('%s__%s.pt' % (sk, weight_id))
    cpu = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    torch.save(cpu, ppath)
    back = torch.load(ppath, map_location='cpu', weights_only=True)
    if set(back) != set(cpu):
        raise RuntimeError('param reload key mismatch ' + sk + ' ' + weight_id)
    for k in cpu:
        if not torch.equal(cpu[k], back[k]):
            raise RuntimeError('param reload mismatch ' + sk + ' ' + weight_id + ' ' + k)
    manifest = {'path': str(ppath), 'bytes': ppath.stat().st_size, 'sha256': sha(ppath),
                'weights_only_reloadable': True,
                'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in cpu.items()}}
    return {k: back[k].to(device) for k in back}, manifest


def load_weights(scope, weight_id, expert, layer, device):
    from safetensors import safe_open
    sk = B.site_key(expert, layer)
    if weight_id == 'W_D':
        ck = scope['checkpoints_W_D'][expert]
    else:
        ck = scope['checkpoints_W_O'][sk]
    if sha(ck['path']) != ck['sha256']:
        raise RuntimeError('checkpoint provenance ' + weight_id + ' ' + sk)
    with safe_open(ck['path'], framework='pt', device='cpu') as f:
        w1, b1, w2, b2 = [f.get_tensor('blocks.%d.ffn.%s' % (layer, n))
                          for n in ('0.weight', '0.bias', '2.weight', '2.bias')]
    shapes = [(13824, 5120), (13824,), (5120, 13824), (5120,)]
    if [tuple(t.shape) for t in (w1, b1, w2, b2)] != shapes:
        raise RuntimeError('ffn weight shape ' + weight_id + ' ' + sk)
    if weight_id == 'W_D':
        if not all(t.dtype == torch.bfloat16 for t in (w1, b1, w2, b2)):
            raise RuntimeError('W_D dtype')
    else:
        if not all(t.dtype == torch.float32 for t in (w1, b1, w2, b2)):
            raise RuntimeError('W_O storage dtype must be F32')
        w1, b1, w2, b2 = [t.bfloat16() for t in (w1, b1, w2, b2)]   # F32_TO_BF16_RNE
    return [t.to(device) for t in (w1, b1, w2, b2)]


def cell_row(source, role, expert, layer, weight_id, method_id, c, pred, indices,
             scope_sha, fit_sha, row_indices_sha):
    sk = B.site_key(expert, layer)
    prim = B.primary_risk(pred, c)
    aux = B.auxiliary_risk(pred, c)
    gated_sse = prim * c['n'] * D
    return {'source_id': source, 'role': role, 'expert': expert, 'layer': layer,
            'state_mu': 'mu_D', 'endpoint_steps': '%d->%d' % B.EXPERT_STEP[expert],
            'weight_id': weight_id, 'method_id': method_id, 'config_sha': scope_sha,
            'fit_sha': fit_sha, 'U0_sha': sha_tensor(c['u0_bf16']), 'U1_sha': sha_tensor(c['u1_bf16']),
            'row_indices_sha': row_indices_sha, 'G_ref_sha': sha_tensor(c['g']),
            'own_anchor_sha': sha_tensor(c['y0']), 'own_target_sha': sha_tensor(c['y1']),
            'numerical_recipe_sha': scope_sha, 'n_rows': c['n'], 'output_dim': D,
            'pred_sha': sha_tensor(pred), 'gated_sse_f64': gated_sse, 'gated_mse_f64': prim,
            'bf16_gate_mse': aux,
            'chip_indices_sha': sha_tensor(indices) if indices is not None else None,
            'chip_selected_k': B.P64_K if indices is not None else None,
            'chip_proxy_rows_sha': None, 'cell_status': 'VALID'}


def aggregate_bw(cells, dev_sources):
    """cells: list of rows. Build per-source/expert/weight/method primary risk,
    d_m(W), theta_m, source-equal mean + LOO. high/low separate."""
    experts = ('high', 'low')
    layers = (4, 19, 34)
    methods = ('Q1', 'ICC')
    # index primary risk (normalize full method_id to the short name used below)
    risk = {}
    for row in cells:
        if row['cell_status'] != 'VALID':
            continue
        m = METHOD_SHORT[row['method_id']]
        key = (row['source_id'], row['weight_id'], m, row['expert'], row['layer'])
        risk[key] = row['gated_mse_f64']
    per_source = {}
    for src in dev_sources:
        r = {}
        for w in WEIGHTS:
            for m in methods:
                by_e = {}
                for e in experts:
                    vals = [risk[(src, w, m, e, L)] for L in layers]
                    by_e[e] = sum(vals) / len(vals)
                r[(w, m)] = sum(by_e.values()) / len(experts)
                r[(w, m, 'high')] = by_e['high']
                r[(w, m, 'low')] = by_e['low']
        d = {}
        for w in WEIGHTS:
            for m in ('ICC',):
                d[(m, w)] = r[(w, m)] - r[(w, 'Q1')]
        theta = {m: d[(m, 'W_D')] - d[(m, 'W_O')] for m in ('ICC',)}
        per_source[src] = {'risk': {f'{w}|{m}': r[(w, m)] for w in WEIGHTS for m in methods},
                           'risk_by_expert': {f'{w}|{m}|{e}': r[(w, m, e)] for w in WEIGHTS for m in methods for e in experts},
                           'd': {f'{m}|{w}': d[(m, w)] for m in ('ICC',) for w in WEIGHTS},
                           'theta': theta}
    mean_risk = {f'{w}|{m}': sum(per_source[s]['risk'][f'{w}|{m}'] for s in dev_sources) / len(dev_sources)
                 for w in WEIGHTS for m in methods}
    mean_d = {f'{m}|{w}': sum(per_source[s]['d'][f'{m}|{w}'] for s in dev_sources) / len(dev_sources)
              for m in ('ICC',) for w in WEIGHTS}
    mean_theta = {m: sum(per_source[s]['theta'][m] for s in dev_sources) / len(dev_sources) for m in ('ICC',)}
    loo_theta = {m: {drop: sum(per_source[s]['theta'][m] for s in dev_sources if s != drop) / (len(dev_sources) - 1)
                     for drop in dev_sources} for m in ('ICC',)}
    return {'per_source': per_source, 'source_equal_mean_risk': mean_risk,
            'source_equal_mean_d': mean_d, 'source_equal_mean_theta': mean_theta,
            'leave_one_source_out_theta': loo_theta}


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
    icc_man = json.loads(Path(scope['icc_factor_manifest']).read_text())
    if sha(scope['icc_factor_manifest']) != scope['icc_factor_manifest_sha256']:
        raise RuntimeError('ICC manifest provenance')
    # B6 row identity: per-source PANEL512 token-id hash from the frozen binding.
    bindings = json.loads((stage / 'B_BINDINGS.json').read_text())
    if sha(stage / 'B_BINDINGS.json') != scope['frozen_inputs_sha256']['B_BINDINGS.json']:
        raise RuntimeError('B_BINDINGS provenance')
    if bindings['panel']['count'] != 512:
        raise RuntimeError('panel count not 512')
    row_sha = {s: bindings['panel']['per_source'][s]['token_id_sha256_le_int64']
               for s in cal_sources + dev_sources}

    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cells = []
    checks = {'q1_convergence': {}, 'icc_factor_identity': {}, 'risks_finite': True,
              'chip_missing_cells': [], 'own_w_recompute': True, 'params_reload_verified': True}
    params_manifest = {}
    t0 = datetime.datetime.now(datetime.timezone.utc)
    fixed64 = B.chip_fixed64()

    for expert, layer in B.SITES:
        sk = B.site_key(expert, layer)
        for weight_id in WEIGHTS:
            w1, b1, w2, b2 = load_weights(scope, weight_id, expert, layer, device)
            cal = B.build_pack(bank_root, cal_sources, expert, layer, device)
            dev = B.build_pack(bank_root, dev_sources, expert, layer, device)
            B.own_w_targets(cal, w1, b1, w2, b2)
            B.own_w_targets(dev, w1, b1, w2, b2)

            # Q1 fit on CAL own-W targets
            alpha_q0, a0, z_norm = B.q0_prior(cal, w1, b1, w2)
            a_q1, alpha_q1, q1_info = B.fit_q1(cal, w1, w2, a0, alpha_q0)
            conv = q1_info['status'] == 'CONVERGED' and q1_info['true_relative_residual'] <= B.Q1_TOL
            checks['q1_convergence'][sk + '|' + weight_id] = {
                'status': q1_info['status'], 'true_relative_residual': q1_info['true_relative_residual'],
                'iterations': q1_info['iterations'], 'converged': conv}
            if not conv:
                raise RuntimeError('Q1 not converged ' + sk + ' ' + weight_id)
            a_q1_f32 = (w1.float().T * alpha_q1) @ w2.float().T

            # ICC factors
            if weight_id == 'W_D':
                frec = icc_man['%s/%d' % (expert, layer)]
                (f1l, f1r, f2l, f2r), fobj = B.load_w_d_factors(frec['path'], device)
                ident = fobj['checkpoint_sha256'] == scope['checkpoints_W_D'][expert]['sha256'] \
                    and fobj['method'] == 'svd' and fobj['rank'] == 512
                checks['icc_factor_identity'][sk + '|' + weight_id] = {
                    'source': 'reused_W_D_factor', 'checkpoint_match': ident,
                    'method': fobj['method'], 'rank': fobj['rank'], 'factor_sha256': frec['sha256']}
                if not ident:
                    raise RuntimeError('ICC W_D factor identity ' + sk)
                if sha(frec['path']) != frec['sha256']:
                    raise RuntimeError('ICC factor hash ' + sk)
                if [tuple(t.shape) for t in (f1l, f1r, f2l, f2r)] != list(ICC_FACTOR_SHAPES) \
                        or not all(t.dtype == torch.bfloat16 for t in (f1l, f1r, f2l, f2r)):
                    raise RuntimeError('ICC W_D factor shape/dtype ' + sk)
            else:
                f1, f2 = B.make_w_o_factors(w1, w2, device)
                f1l, f1r, f2l, f2r = f1.left, f1.right, f2.left, f2.right
                checks['icc_factor_identity'][sk + '|' + weight_id] = {
                    'source': 'fresh_W_O_ordinary_svd', 'method': 'svd', 'rank': 512, 'driver': 'gesvd'}
                if [tuple(t.shape) for t in (f1l, f1r, f2l, f2r)] != list(ICC_FACTOR_SHAPES):
                    raise RuntimeError('ICC W_O factor shape ' + sk)

            # §5/A6: persist real fitted tensors, reload from file, then score.
            tensors = {'alpha_Q1': alpha_q1.float(), 'A_q1_f32': a_q1_f32,
                       'icc_fc1_left': f1l, 'icc_fc1_right': f1r,
                       'icc_fc2_left': f2l, 'icc_fc2_right': f2r}
            reloaded, pman = persist_and_reload(out, sk, weight_id, tensors, device)
            a_q1_f32 = reloaded['A_q1_f32']
            f1l, f1r, f2l, f2r = (reloaded['icc_fc1_left'], reloaded['icc_fc1_right'],
                                  reloaded['icc_fc2_left'], reloaded['icc_fc2_right'])
            q1_fit_sha = sha_tensor(reloaded['alpha_Q1'])

            # score Q1 + ICC on DEV (primary main table, B6 DEV-only 264 cells)
            for c in dev:
                pred_q1 = B.q1_predict(c, a_q1_f32)
                cells.append(cell_row(c['source'], 'DEV', expert, layer, weight_id, 'Q1R2_CAL11_P512_OWNW',
                                      c, pred_q1, None, scope_sha, q1_fit_sha, row_sha[c['source']]))
                pred_icc = B.icc_predict(c, f1l, f1r, f2l, f2r)
                cells.append(cell_row(c['source'], 'DEV', expert, layer, weight_id, 'ICC_SVD_R512_TWO_RESIDUAL',
                                      c, pred_icc, None, scope_sha, None, row_sha[c['source']]))
            # Chip DEV cells blocked: fixed64 rows not in PANEL512 (B_CHIP_STATS separate scope)
            for c in dev:
                checks['chip_missing_cells'].append({
                    'source_id': c['source'], 'site': sk, 'weight_id': weight_id,
                    'method_id': 'CHIPMUNK_GLOBAL_P64', 'cell_status': 'BLOCKED_INPUT_GAP',
                    'missing': 'fixed64 U0/U1 rows not in PANEL512', 'unblock_action': 'B_CHIP_STATS'})

            params_manifest[sk + '|' + weight_id] = {
                'params_file': pman,
                'Q1': {'alpha_Q1_sha256': q1_fit_sha, 'A_q1_fro2': float(a_q1.square().sum()),
                       'q1_fit': q1_info},
                'ICC': checks['icc_factor_identity'][sk + '|' + weight_id]}
            del cal, dev, w1, b1, w2, b2, a_q1, a_q1_f32, a0, alpha_q1, alpha_q0
            del tensors, reloaded, f1l, f1r, f2l, f2r
            torch.cuda.empty_cache()
            print('BW_FIT site=%s weight=%s params_bytes=%d' % (sk, weight_id, pman['bytes']), flush=True)

    # finite check
    for row in cells:
        if not (torch.isfinite(torch.tensor([row['gated_mse_f64'], row['bf16_gate_mse']])).all()):
            checks['risks_finite'] = False
            raise RuntimeError('nonfinite risk cell')

    agg = aggregate_bw(cells, dev_sources)
    elapsed = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds()
    total_param_bytes = sum(v['params_file']['bytes'] for v in params_manifest.values())

    # write BW_CELL.csv
    fieldnames = list(cells[0].keys())
    with (out / 'BW_CELL.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in cells:
            w.writerow(row)
    # BW_SOURCE.csv
    with (out / 'BW_SOURCE.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['source_id', 'risk_W_O_Q1', 'risk_W_O_ICC', 'risk_W_D_Q1', 'risk_W_D_ICC',
                    'd_ICC_W_O', 'd_ICC_W_D', 'theta_ICC'])
        for src in dev_sources:
            ps = agg['per_source'][src]
            w.writerow([src, ps['risk']['W_O|Q1'], ps['risk']['W_O|ICC'], ps['risk']['W_D|Q1'], ps['risk']['W_D|ICC'],
                        ps['d']['ICC|W_O'], ps['d']['ICC|W_D'], ps['theta']['ICC']])
    # BW_COST.csv — all five spec cost_fields; cost reported, not matched across methods.
    q1_param_bytes = 13824 * 4 + 5120 * 5120 * 4          # alpha_Q1 FP32 + A_q1_f32 FP32
    icc_factor_bytes = (13824 * 512 + 512 * 5120 + 5120 * 512 + 512 * 13824) * 2   # BF16 factors
    consumption = 'shared_run_elapsed_seconds=%.3f; not native full-M latency' % elapsed
    cost_rows = [
        ('Q1', 'Q1R2_CAL11_P512_OWNW', q1_param_bytes, 0, 5120 * 2, 5120 * 5120, 'none', consumption),
        ('ICC', 'ICC_SVD_R512_TWO_RESIDUAL', 0, icc_factor_bytes, (13824 + 5120) * 2,
         2 * 512 * (5120 + 13824), 'none', consumption),
        ('Chip', 'CHIPMUNK_GLOBAL_P64', 0, 0, (13824 + 5120) * 2, 2 * 5120 * 5120,
         'rms_score+selection(K=5120)+packed_pregate; BLOCKED_INPUT_GAP this scope',
         'not_executed_this_scope')]
    with (out / 'BW_COST.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method', 'method_id', 'parameter_bytes_per_site', 'factor_bytes_per_site',
                    'cache_bytes_per_token', 'analytic_mac_per_token', 'selector_packing_cost',
                    'actual_compute_consumption', 'cost_contract'])
        for r in cost_rows:
            w.writerow(list(r) + ['IDENTICAL_WITHIN_METHOD_ACROSS_WEIGHTS;CROSS_METHOD_COST_REPORTED_NOT_MATCHED'])

    n_chip_blocked = len(checks['chip_missing_cells'])
    summary = {'scope_id': scope['scope_id'], 'status': 'BW_Q1ICC_COMPLETE',
               'new_D4_full_executions': 0, 'new_O40_full_executions': 0,
               'weights': list(WEIGHTS), 'methods_completed': ['Q1', 'ICC'],
               'sites': [B.site_key(e, L) for e, L in B.SITES], 'state': 'mu_D',
               'dev_cells_valid': len(cells), 'dev_cells_expected_Q1ICC': 264,
               'chip_cells_blocked': n_chip_blocked, 'dev_cells_expected_all_methods': 396,
               'primary_estimand': 'F64 gated pre-gate SSE/(512*5120); layer-equal->expert-equal->source-equal',
               'contrast': 'd_m(W)=R_m(W)-R_Q1(W); theta_m=d_m(W_D)-d_m(W_O)',
               'quality_metrics': 'NOT_MEASURED', 'performance_comparison': 'NOT_BENCHMARKED',
               'elapsed_seconds': elapsed, 'params_total_bytes': total_param_bytes,
               'params_reload_verified': checks['params_reload_verified'],
               'source_equal_mean_risk': agg['source_equal_mean_risk'],
               'source_equal_mean_d': agg['source_equal_mean_d'],
               'source_equal_mean_theta': agg['source_equal_mean_theta'],
               'chip_unblock': 'B_CHIP_STATS (separate concrete scope + ACTIVE); blocks only Chip cells'}
    exclusive_json(out / 'PARAMS_MANIFEST.json', params_manifest)
    exclusive_json(out / 'CHECKS.json', checks)
    exclusive_json(out / 'BW_AGGREGATE.json', agg)
    exclusive_json(out / 'SUMMARY.json', summary)
    print('BW_COMPLETE elapsed=%.1fs dev_cells=%d chip_blocked=%d params_bytes=%d'
          % (elapsed, len(cells), n_chip_blocked, total_param_bytes), flush=True)
    return summary