#!/bin/bash
# Read-only post-run integrity verification for BWS v1 (job 166225).
set -euo pipefail
ST="/public/home/scc/pb23061276/projects/wan-quant/work/g20_two_track_core_20261114/BWS/v1/stage"
PY="/public/home/scc/pb23061276/miniconda3/envs/hif4/bin/python3.11"
cd "$ST"
"$PY" -B - "$ST" <<'PYEOF'
import hashlib, json, csv, sys, math
from pathlib import Path
st = Path(sys.argv[1])
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
scope = json.loads((st/'scope.json').read_text())
scope_sha = sha(st/'scope.json')
print('SCOPE_SHA', scope_sha)
assert scope_sha == '180382a7e78c5c542ac7bcbeb7505b34311f21c80f17c0018bb1bcd7abbebf13', 'scope sha'
bad = []
for name, want in {**scope['code_sha256'], **scope['frozen_inputs_sha256']}.items():
    got = sha(st/name); ok = got == want
    if not ok: bad.append(name)
print('BIND_FILES', len(scope['code_sha256'])+len(scope['frozen_inputs_sha256']), 'BAD', len(bad))
assert not bad, f'hash mismatch: {bad}'
audit_sha = sha(st/'ROOT_SELF_AUDIT.txt')
print('AUDIT_SHA', audit_sha)
assert audit_sha == 'c688a585e6bea8f1e74917737a2c7534c638ca95f3c358f5eeb0a0d81d0fb6be', 'audit sha'
out = st/'output'
summ = json.loads((out/'SUMMARY.json').read_text())
assert summ['status'] == 'BWS_THREE_METHODS_COMPLETE', summ['status']
assert summ['new_D4_full_executions'] == 0 and summ['new_O40_full_executions'] == 6
assert summ['new_video_decode'] == 0 and summ['new_video_scoring'] == 0
assert summ['dev_cells_valid'] == 216 and summ['dev_cells_expected_all_methods'] == 216
assert summ['q1icc_cells'] == 144 and summ['dev_cells_expected_Q1ICC'] == 144
assert summ['cal_ids'] == ['vbi_1005','vbi_0895','vbi_0648']
assert summ['dev_ids'] == ['vbi_1013','vbi_0886','vbi_0800']
assert summ['weights'] == ['W_O','W_D'] and summ['states'] == ['mu_D','mu_O']
assert summ['methods_completed'] == ['Q1','ICC','Chip']
print('SUMMARY_OK cells=%d q1icc=%d O40=%d elapsed=%.1fs' % (summ['dev_cells_valid'], summ['q1icc_cells'], summ['new_O40_full_executions'], summ['elapsed_seconds']))
# BWS_CELL.csv
rows = list(csv.DictReader((out/'BWS_CELL.csv').open(newline='', encoding='utf8')))
assert len(rows) == 216, len(rows)
need_cols = {'source_id','role','expert','layer','state_mu','weight_id','method_id','gated_mse_f64','bf16_gate_mse','cell_status','n_rows','output_dim','pred_sha','G_ref_sha'}
assert need_cols.issubset(rows[0].keys()), set(rows[0].keys())
seen = set(); nchip = 0
for r in rows:
    assert r['cell_status'] == 'VALID' and r['role'] == 'DEV'
    assert int(r['n_rows']) == 512 and int(r['output_dim']) == 5120
    g = float(r['gated_mse_f64']); a = float(r['bf16_gate_mse'])
    assert math.isfinite(g) and math.isfinite(a) and g >= 0, r
    key = (r['source_id'],r['weight_id'],r['state_mu'],r['method_id'],r['expert'],r['layer'])
    assert key not in seen, ('dup', key); seen.add(key)
    if r['method_id'] == 'CHIPMUNK_GLOBAL_P64':
        nchip += 1
        assert r['chip_selected_k'] == '5120' and r['chip_indices_sha'] not in ('','None')
assert len(seen) == 216 and nchip == 72, (len(seen), nchip)
print('BWS_CELL_OK rows=216 distinct=216 chip=72 all_finite_nonneg')
# BWS_SOURCE.csv
srows = list(csv.DictReader((out/'BWS_SOURCE.csv').open(newline='', encoding='utf8')))
assert len(srows) == 3 and {r['source_id'] for r in srows} == set(summ['dev_ids'])
for r in srows:
    for k in ('I_ICC','I_Chip'): assert math.isfinite(float(r[k])), r
print('BWS_SOURCE_OK rows=3')
# BWS_AGGREGATE.json + independent I_m re-derivation
agg = json.loads((out/'BWS_AGGREGATE.json').read_text())
assert set(agg['per_source']) == set(summ['dev_ids'])
for src, ps in agg['per_source'].items():
    for m in ('ICC','Chip'):
        d = ps['d']
        I = (d[f'{m}|W_D|mu_D'] - d[f'{m}|W_O|mu_D']) - (d[f'{m}|W_D|mu_O'] - d[f'{m}|W_O|mu_O'])
        assert abs(I - ps['I'][m]) < 1e-12, (src, m, I, ps['I'][m])
        assert math.isfinite(ps['I'][m])
for m in ('ICC','Chip'):
    mi = summ['source_equal_mean_I'][m]
    recompute = sum(agg['per_source'][s]['I'][m] for s in summ['dev_ids'])/3
    assert abs(mi - recompute) < 1e-12, (m, mi, recompute)
    lo, hi = summ['source_range_I'][m]
    vals = [agg['per_source'][s]['I'][m] for s in summ['dev_ids']]
    assert abs(lo-min(vals)) < 1e-12 and abs(hi-max(vals)) < 1e-12
    assert math.isfinite(mi)
print('AGGREGATE_OK I_m re-derived; mean_I', json.dumps(summ['source_equal_mean_I']), 'range_I', json.dumps(summ['source_range_I']))
# CHECKS.json
chk = json.loads((out/'CHECKS.json').read_text())
assert chk['risks_finite'] is True and chk['params_reload_verified'] is True and chk['own_w_recompute'] is True
assert len(chk['q1_convergence']) == 12 and len(chk['icc_factor_identity']) == 12
for k, v in chk['q1_convergence'].items():
    assert v['converged'] is True and v['status'] == 'CONVERGED' and v['true_relative_residual'] <= 1e-5, (k, v)
for k, v in chk['icc_factor_identity'].items():
    assert v['method'] == 'svd' and v['rank'] == 512, (k, v)
    if k.endswith('W_D'): assert v['checkpoint_match'] is True
print('CHECKS_OK q1_converged=12/12 icc_identity=12 risks_finite params_reload')
# PARAMS_MANIFEST.json
pm = json.loads((out/'PARAMS_MANIFEST.json').read_text())
assert len(pm) == 12, len(pm)
tot = 0
for k, v in pm.items():
    pf = v['params_file']; p = Path(pf['path'])
    assert p.is_file() and p.stat().st_size == pf['bytes'] and sha(p) == pf['sha256'], k
    tot += pf['bytes']
assert tot == summ['params_total_bytes'], (tot, summ['params_total_bytes'])
print('PARAMS_OK entries=12 total_bytes=%d hash_match' % tot)
# O40 capture sidecars
caps = sorted(out.glob('O40_CAPTURE_vbi_*.json')); assert len(caps) == 6, len(caps)
for c in caps:
    cd = json.loads(c.read_text())
    assert cd['capture_sites'] == 12, c.name
    for e in ('high','low'):
        ep = cd['endpoints'][e]
        assert ep['hit_step'] == ep['anchor_step'] + 1, (c.name, e, ep)
    assert cd['record_id'] == c.stem.replace('O40_CAPTURE_','')
print('O40_SIDECARS_OK n=6 sites=12 adjacent_endpoints')
# o40_inputs payloads: 6 sources x 12 .pt = 72
o40 = out/'o40_inputs'
srcs = sorted(p.name for p in o40.iterdir() if p.is_dir())
assert srcs == sorted(summ['cal_ids']+summ['dev_ids']), srcs
n_pt = 0
for s in srcs:
    pts = sorted((o40/s).glob('*.pt')); assert len(pts) == 12, (s, len(pts))
    for f in pts:
        side = json.loads(Path(str(f)+'.json').read_text())
        assert side['grid'] == [21,29,52] and side['flattened_tokens'] == 31668, (f.name, side.get('grid'))
        n_pt += 1
assert n_pt == 72, n_pt
print('O40_PAYLOADS_OK n_pt=72 grid=[21,29,52] M=31668')
# torch spot-check ONE DEV mu_O payload: U_panel (512,5120) + U_fixed64 (64,5120) bf16 finite
import torch
dev0 = summ['dev_ids'][0]
f0 = sorted((o40/dev0).glob('*.pt'))[0]
t = torch.load(f0, map_location='cpu', weights_only=True)
up = t['U_panel']; assert tuple(up.shape) == (512,5120) and up.dtype == torch.bfloat16 and torch.isfinite(up.float()).all()
uf = t['U_fixed64']; assert tuple(uf.shape) == (64,5120) and uf.dtype == torch.bfloat16 and torch.isfinite(uf.float()).all()
print('O40_SPOTCHECK_OK', f0.name, 'U_panel', tuple(up.shape), 'U_fixed64', tuple(uf.shape))
print('VERIFY_POST_BWS_PASS')
PYEOF