"""G20 BW_CHIP_SCORE: consume the produced B_CHIP_P64_INPUTS and score ONLY the
132 blocked BW Chip cells (11 DEV x 6 sites x 2 weights) via the actual
CHIPMUNK_GLOBAL_P64 path. Reuses the completed 264 BW Q1/ICC results and fits
verbatim (no Q1/ICC refit or replay); diffs Chip against the existing Q1 table.

ZERO new trajectories: no D4, no O40, no source/image/prompt, no video decode or
scoring, no full-M block stats, no full-M F1. This is sampled local FFN function
replay on the existing PANEL512 mu_D bank with the produced fixed64 U0/U1 rows.
Primary risk F64 gated pre-gate SSE/(512*5120); auxiliary BF16 gate MSE separate.
G_ref is the same-source/site D4 hit gate, fixed across weights (identical to BW).
No quality claim, no production timing, no deployment claim.
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
METHOD_ID = 'CHIPMUNK_GLOBAL_P64'
EXPERTS = ('high', 'low')
LAYERS = (4, 19, 34)


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


def load_weights(scope, weight_id, expert, layer, device):
    from safetensors import safe_open
    sk = B.site_key(expert, layer)
    ck = scope['checkpoints_W_D'][expert] if weight_id == 'W_D' else scope['checkpoints_W_O'][sk]
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
        w1, b1, w2, b2 = [t.bfloat16() for t in (w1, b1, w2, b2)]
    return [t.to(device) for t in (w1, b1, w2, b2)]


def cell_row(source, expert, layer, weight_id, c, pred, indices, scope_sha, row_indices_sha,
             r_q1, r_icc):
    prim = B.primary_risk(pred, c)
    aux = B.auxiliary_risk(pred, c)
    gated_sse = prim * c['n'] * D
    return {'source_id': source, 'role': 'DEV', 'expert': expert, 'layer': layer,
            'state_mu': 'mu_D', 'endpoint_steps': '%d->%d' % B.EXPERT_STEP[expert],
            'weight_id': weight_id, 'method_id': METHOD_ID, 'config_sha': scope_sha,
            'fit_sha': None, 'U0_sha': sha_tensor(c['u0_bf16']), 'U1_sha': sha_tensor(c['u1_bf16']),
            'row_indices_sha': row_indices_sha, 'G_ref_sha': sha_tensor(c['g']),
            'own_anchor_sha': sha_tensor(c['y0']), 'own_target_sha': sha_tensor(c['y1']),
            'numerical_recipe_sha': scope_sha, 'n_rows': c['n'], 'output_dim': D,
            'pred_sha': sha_tensor(pred), 'gated_sse_f64': gated_sse, 'gated_mse_f64': prim,
            'bf16_gate_mse': aux, 'reused_q1_gated_mse_f64': r_q1, 'reused_icc_gated_mse_f64': r_icc,
            'd_chip_minus_q1_f64': prim - r_q1, 'd_icc_minus_q1_f64': r_icc - r_q1,
            'chip_indices_sha': sha_tensor(indices), 'chip_selected_k': B.P64_K,
            'chip_proxy_rows_sha': None, 'cell_status': 'VALID'}


def aggregate_chip(cells, dev_sources, q1icc):
    """Per-source Chip risk (layer-equal -> expert-equal), d_Chip=R_Chip-R_Q1 using
    the REUSED BW Q1 risks, theta_Chip=d(W_D)-d(W_O); source-equal mean + range + LOO."""
    risk = {}
    for row in cells:
        if row['cell_status'] != 'VALID':
            continue
        risk[(row['source_id'], row['weight_id'], row['expert'], row['layer'])] = row['gated_mse_f64']
    per_source = {}
    for src in dev_sources:
        r = {}
        for w in WEIGHTS:
            by_e = {}
            for e in EXPERTS:
                vals = [risk[(src, w, e, L)] for L in LAYERS]
                by_e[e] = sum(vals) / len(vals)
            r[w] = sum(by_e.values()) / len(EXPERTS)
            r[(w, 'high')] = by_e['high']
            r[(w, 'low')] = by_e['low']
        # reused Q1/ICC source risks (layer-equal -> expert-equal) from the BW table
        reused = {}
        for w in WEIGHTS:
            for m in ('Q1', 'ICC'):
                by_e = {}
                for e in EXPERTS:
                    vals = [q1icc[(src, w, m, e, L)] for L in LAYERS]
                    by_e[e] = sum(vals) / len(vals)
                reused[(w, m)] = sum(by_e.values()) / len(EXPERTS)
        d = {w: r[w] - reused[(w, 'Q1')] for w in WEIGHTS}
        d_icc = {w: reused[(w, 'ICC')] - reused[(w, 'Q1')] for w in WEIGHTS}
        theta = d['W_D'] - d['W_O']
        per_source[src] = {
            'risk': {w: r[w] for w in WEIGHTS},
            'risk_by_expert': {f'{w}|{e}': r[(w, e)] for w in WEIGHTS for e in EXPERTS},
            'reused_q1_risk': {w: reused[(w, 'Q1')] for w in WEIGHTS},
            'reused_icc_risk': {w: reused[(w, 'ICC')] for w in WEIGHTS},
            'd_chip': d, 'd_icc': d_icc, 'theta_chip': theta}
    mean_risk = {w: sum(per_source[s]['risk'][w] for s in dev_sources) / len(dev_sources) for w in WEIGHTS}
    mean_d = {w: sum(per_source[s]['d_chip'][w] for s in dev_sources) / len(dev_sources) for w in WEIGHTS}
    mean_d_icc = {w: sum(per_source[s]['d_icc'][w] for s in dev_sources) / len(dev_sources) for w in WEIGHTS}
    mean_theta = sum(per_source[s]['theta_chip'] for s in dev_sources) / len(dev_sources)
    range_d = {w: [min(per_source[s]['d_chip'][w] for s in dev_sources),
                   max(per_source[s]['d_chip'][w] for s in dev_sources)] for w in WEIGHTS}
    loo_theta = {drop: sum(per_source[s]['theta_chip'] for s in dev_sources if s != drop) / (len(dev_sources) - 1)
                 for drop in dev_sources}
    return {'per_source': per_source, 'source_equal_mean_risk': mean_risk,
            'source_equal_mean_d_chip': mean_d, 'source_equal_mean_d_icc_reused': mean_d_icc,
            'source_equal_mean_theta_chip': mean_theta, 'source_range_d_chip': range_d,
            'leave_one_source_out_theta_chip': loo_theta}


def run(stage, scope, scope_sha):
    stage = Path(stage)
    out = stage / 'output'
    out.mkdir(exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc)
    bank_root = Path(scope['bank_root'])
    manifest = json.loads((bank_root.parent / 'BANK_MANIFEST.json').read_text())
    if sha(bank_root.parent / 'BANK_MANIFEST.json') != scope['bank_manifest_sha256']:
        raise RuntimeError('bank manifest provenance')
    dev_sources = sorted(k for k, v in manifest.items() if v['cycle_role'] == 'DEV')
    if len(dev_sources) != 11:
        raise RuntimeError('DEV cohort must be 11')
    bindings = json.loads((stage / 'B_BINDINGS.json').read_text())
    if sha(stage / 'B_BINDINGS.json') != scope['frozen_inputs_sha256']['B_BINDINGS.json']:
        raise RuntimeError('B_BINDINGS provenance')
    chip_path = Path(scope['chip_p64_inputs'])
    if sha(chip_path) != scope['chip_p64_inputs_sha256']:
        raise RuntimeError('B_CHIP_P64_INPUTS provenance')
    chip = json.loads(chip_path.read_text())
    if chip['capture_scope_sha256'] != scope['chip_stats_scope_sha256']:
        raise RuntimeError('chip inputs scope identity')
    if sorted(chip['sources']) != dev_sources:
        raise RuntimeError('chip inputs cohort != DEV11')
    if chip['selected_k'] != B.P64_K or chip['grid'] != [21, 29, 52] or chip['flattened_tokens'] != B.M_FULL:
        raise RuntimeError('chip inputs geometry/selector')
    # reuse the completed BW 264 Q1/ICC results (frozen, hash-gated); no refit/replay
    bw_cell_path = Path(scope['bw_cell_csv'])
    if sha(bw_cell_path) != scope['bw_cell_csv_sha256']:
        raise RuntimeError('BW_CELL provenance')
    bw_rows = list(csv.DictReader(bw_cell_path.open(newline='', encoding='utf-8')))
    if len(bw_rows) != 264:
        raise RuntimeError('BW_CELL must have 264 Q1/ICC cells')
    short = {'Q1R2_CAL11_P512_OWNW': 'Q1', 'ICC_SVD_R512_TWO_RESIDUAL': 'ICC'}
    q1icc = {}
    for r in bw_rows:
        if r['cell_status'] != 'VALID' or r['state_mu'] != 'mu_D':
            raise RuntimeError('BW_CELL row not VALID mu_D')
        q1icc[(r['source_id'], r['weight_id'], short[r['method_id']], r['expert'], int(r['layer']))] = float(r['gated_mse_f64'])
    if len(q1icc) != 264:
        raise RuntimeError('BW_CELL distinct keys != 264')
    row_sha = {s: bindings['panel']['per_source'][s]['token_id_sha256_le_int64'] for s in dev_sources}
    fixed64 = B.chip_fixed64()
    if fixed64.tolist() != bindings['p64']['fixed64_token_ids']:
        raise RuntimeError('regenerated fixed64 differs from binding')

    device = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA A100-SXM4-80GB':
        raise RuntimeError('requires one A100 80GB')
    if torch.__version__ != scope['runtime_versions']['torch']:
        raise RuntimeError('torch version mismatch')
    gpu_limit = scope['budget']['gpu_memory_gib'] * 1024 ** 3

    def check_resources():
        if torch.cuda.max_memory_reserved() > gpu_limit:
            raise MemoryError('GPU_MEMORY_BUDGET_EXCEEDED')
        import resource
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > scope['budget']['host_memory_gib'] * 1024 ** 3:
            raise MemoryError('host peak RSS budget')

    cells = []
    checks = {'risks_finite': True, 'own_w_recompute': True, 'fixed64_payloads_verified': 0,
              'reused_bw_q1icc_cells': 264, 'new_trajectories': 0, 'q1icc_refit': False,
              'G_ref': 'same-source/site D4 hit gate (identical to BW)',
              'fit_identity': 'REUSED_BW_264_Q1ICC_no_refit'}
    for expert, layer in B.SITES:
        sk = B.site_key(expert, layer)
        for weight_id in WEIGHTS:
            w1, b1, w2, b2 = load_weights(scope, weight_id, expert, layer, device)
            dev = B.build_pack(bank_root, dev_sources, expert, layer, device)
            B.own_w_targets(dev, w1, b1, w2, b2)
            for c in dev:
                src = c['source']
                rec = chip['sources'][src]['sites'][sk]
                rows = []
                for role in ('anchor', 'hit'):
                    e = rec[role]
                    p = Path(e['path'])
                    if not p.is_file() or p.stat().st_size != e['bytes'] or sha(p) != e['file_sha256']:
                        raise RuntimeError('fixed64 payload provenance ' + src + ' ' + sk + ' ' + role)
                    t = torch.load(p, map_location='cpu', weights_only=True)
                    uf = t['U_fixed64']
                    if tuple(uf.shape) != (64, D) or uf.dtype != torch.bfloat16 or not torch.isfinite(uf.float()).all():
                        raise RuntimeError('fixed64 payload shape/dtype/finite ' + src + ' ' + sk + ' ' + role)
                    rows.append(uf.to(device=device, dtype=torch.bfloat16))
                    checks['fixed64_payloads_verified'] += 1
                pred, indices = B.chip_predict(c, w1, b1, w2, rows[0], rows[1], fixed64)
                r_q1 = q1icc[(src, weight_id, 'Q1', expert, layer)]
                r_icc = q1icc[(src, weight_id, 'ICC', expert, layer)]
                cells.append(cell_row(src, expert, layer, weight_id, c, pred, indices, scope_sha, row_sha[src], r_q1, r_icc))
            del dev, w1, b1, w2, b2
            torch.cuda.empty_cache()
            check_resources()
            print('BW_CHIP_SCORE site=%s weight=%s cells=%d' % (sk, weight_id, len(cells)), flush=True)
    for row in cells:
        if not torch.isfinite(torch.tensor([row['gated_mse_f64'], row['bf16_gate_mse'], row['d_chip_minus_q1_f64']])).all():
            checks['risks_finite'] = False
            raise RuntimeError('nonfinite risk cell')
    if len(cells) != 132 or checks['fixed64_payloads_verified'] != 132 * 2:
        raise RuntimeError('expected 132 Chip cells and 264 fixed64 payloads')
    agg = aggregate_chip(cells, dev_sources, q1icc)
    elapsed = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()
    fieldnames = list(cells[0].keys())
    with (out / 'BW_CHIP_CELL.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in cells:
            w.writerow(row)
    with (out / 'BW_CHIP_SOURCE.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['source_id', 'R_Chip_W_O', 'R_Chip_W_D', 'd_Chip_W_O', 'd_Chip_W_D',
                    'd_ICC_W_O_reused', 'd_ICC_W_D_reused', 'theta_Chip'])
        for src in dev_sources:
            ps = agg['per_source'][src]
            w.writerow([src, ps['risk']['W_O'], ps['risk']['W_D'], ps['d_chip']['W_O'], ps['d_chip']['W_D'],
                        ps['d_icc']['W_O'], ps['d_icc']['W_D'], ps['theta_chip']])
    summary = {'scope_id': scope['scope_id'], 'status': 'BW_CHIP_SCORE_COMPLETE',
               'new_D4_full_executions': 0, 'new_O40_full_executions': 0, 'new_videos': 0,
               'new_video_decode': 0, 'new_video_scoring': 0, 'new_source_pairs': 0,
               'q1icc_refit_or_replay': False, 'reused_bw_q1icc_cells': 264,
               'dev_sources': dev_sources, 'weights': list(WEIGHTS), 'method': METHOD_ID,
               'sites': [B.site_key(e, L) for e, L in B.SITES],
               'chip_cells_valid': len(cells), 'chip_cells_expected': 132,
               'fixed64_payloads_verified': checks['fixed64_payloads_verified'],
               'primary_estimand': 'F64 gated pre-gate SSE/(512*5120); layer-equal->expert-equal->source-equal',
               'contrast': 'd_Chip(W)=R_Chip(W)-R_Q1(W) reused; theta_Chip=d(W_D)-d(W_O)',
               'G_ref': 'same-source/site D4 hit gate (identical to BW)',
               'quality_metrics': 'NOT_MEASURED', 'performance_comparison': 'NOT_BENCHMARKED',
               'elapsed_seconds': elapsed,
               'source_equal_mean_risk': agg['source_equal_mean_risk'],
               'source_equal_mean_d_chip': agg['source_equal_mean_d_chip'],
               'source_equal_mean_d_icc_reused': agg['source_equal_mean_d_icc_reused'],
               'source_equal_mean_theta_chip': agg['source_equal_mean_theta_chip'],
               'source_range_d_chip': agg['source_range_d_chip']}
    exclusive_json(out / 'CHECKS.json', checks)
    exclusive_json(out / 'BW_CHIP_AGGREGATE.json', agg)
    exclusive_json(out / 'SUMMARY.json', summary)
    print('BW_CHIP_SCORE_COMPLETE elapsed=%.1fs cells=%d payloads=%d' % (elapsed, len(cells), checks['fixed64_payloads_verified']), flush=True)
    return {'status': 'BW_CHIP_SCORE_COMPLETE', 'chip_cells_valid': len(cells),
            'new_trajectories': 0, 'reused_bw_q1icc_cells': 264,
            'source_equal_mean_d_chip': agg['source_equal_mean_d_chip'],
            'source_equal_mean_theta_chip': agg['source_equal_mean_theta_chip'],
            'elapsed_seconds': elapsed,
            'peak_rss_bytes': int(__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss) * 1024,
            'summary_sha256': sha(out / 'SUMMARY.json')}