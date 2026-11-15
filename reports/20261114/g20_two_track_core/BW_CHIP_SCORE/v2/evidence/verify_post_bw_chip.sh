#!/bin/bash
# Read-only post-run integrity verification for BW_CHIP_SCORE V2 (scoring outputs).
# Pure hashlib/json/csv + aggregation cross-check; no torch, no GPU, no trajectories.
set -euo pipefail
ST="/public/home/scc/pb23061276/projects/wan-quant/work/g20_two_track_core_20261114/BW_CHIP_SCORE/v2/stage"
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
assert scope_sha == '202bbbe3c09da160490db5994f648893948f5ff2f3daef61e1d07843e47e6899', 'scope sha'
bad = []
for name, want in {**scope['code_sha256'], **scope['frozen_inputs_sha256']}.items():
    got = sha(st/name)
    if got != want: bad.append(name)
assert not bad, f'hash mismatch: {bad}'
print('BOUND_HASHES_OK', len(scope['code_sha256'])+len(scope['frozen_inputs_sha256']))
assert sha(st/'ROOT_SELF_AUDIT.txt') == 'c85d02670db374fa8085904b8817b1e5d99107df305b3b57af1d5282e8a37ab3', 'audit sha'
ex = json.loads((st/'EXIT.json').read_text())
assert ex['exit_code'] == 0, f"EXIT {ex['exit_code']}"
assert ex['scope_sha256'] == scope_sha
print('EXIT_CODE', ex['exit_code'])
out = st/'output'
summ = json.loads((out/'SUMMARY.json').read_text())
checks = json.loads((out/'CHECKS.json').read_text())
agg = json.loads((out/'BW_CHIP_AGGREGATE.json').read_text())
assert summ['status'] == 'BW_CHIP_SCORE_COMPLETE', summ['status']
for k in ('new_D4_full_executions','new_O40_full_executions','new_videos','new_video_decode','new_video_scoring','new_source_pairs'):
    assert summ[k] == 0, (k, summ[k])
assert summ['q1icc_refit_or_replay'] is False
assert summ['reused_bw_q1icc_cells'] == 264
assert summ['chip_cells_valid'] == 132 == summ['chip_cells_expected']
assert summ['fixed64_payloads_verified'] == 264
assert checks['risks_finite'] is True and checks['new_trajectories'] == 0 and checks['q1icc_refit'] is False
assert checks['fixed64_payloads_verified'] == 264 and checks['reused_bw_q1icc_cells'] == 264
print('SUMMARY_CHECKS_OK status=%s cells=%d payloads=%d zero_consumption=True' % (summ['status'], summ['chip_cells_valid'], summ['fixed64_payloads_verified']))
# CELL csv: 132 rows, all VALID, distinct keys, finite, d_chip identity
rows = list(csv.DictReader((out/'BW_CHIP_CELL.csv').open(newline='', encoding='utf-8')))
assert len(rows) == 132, len(rows)
keys = set(); risk = {}
W = ('W_O','W_D'); E = ('high','low'); L = (4,19,34)
for r in rows:
    assert r['cell_status'] == 'VALID' and r['state_mu'] == 'mu_D' and r['role'] == 'DEV'
    assert r['method_id'] == 'CHIPMUNK_GLOBAL_P64'
    k = (r['source_id'], r['weight_id'], r['expert'], int(r['layer']))
    assert k not in keys, f'dup {k}'; keys.add(k)
    prim = float(r['gated_mse_f64']); rq1 = float(r['reused_q1_gated_mse_f64']); ricc = float(r['reused_icc_gated_mse_f64'])
    assert all(math.isfinite(x) for x in (prim, rq1, ricc, float(r['bf16_gate_mse']), float(r['gated_sse_f64'])))
    assert abs(float(r['d_chip_minus_q1_f64']) - (prim - rq1)) < 1e-12
    assert abs(float(r['d_icc_minus_q1_f64']) - (ricc - rq1)) < 1e-12
    assert abs(float(r['gated_sse_f64']) - prim*int(r['n_rows'])*int(r['output_dim'])) < 1e-6
    risk[k] = prim
assert len(keys) == 132
srcs = sorted({r['source_id'] for r in rows}); assert len(srcs) == 11
print('CELL_CSV_OK 132 VALID distinct keys; d_chip/d_icc/sse identities hold')
# SOURCE csv: 11 rows
srows = list(csv.DictReader((out/'BW_CHIP_SOURCE.csv').open(newline='', encoding='utf-8')))
assert len(srows) == 11
print('SOURCE_CSV_OK 11 rows')
# AGGREGATE cross-check from CELL csv: layer-equal -> expert-equal -> source-equal
assert len(agg['per_source']) == 11
q1icc = {}
for r in rows:
    q1icc[(r['source_id'], r['weight_id'], 'Q1', r['expert'], int(r['layer']))] = float(r['reused_q1_gated_mse_f64'])
    q1icc[(r['source_id'], r['weight_id'], 'ICC', r['expert'], int(r['layer']))] = float(r['reused_icc_gated_mse_f64'])
for s in srcs:
    ps = agg['per_source'][s]
    rchip = {}; rq1 = {}; ricc = {}
    for w in W:
        by_e = {}
        for e in E:
            by_e[e] = sum(risk[(s,w,e,lay)] for lay in L)/3
        rchip[w] = sum(by_e.values())/2
        qe = {e: sum(q1icc[(s,w,'Q1',e,lay)] for lay in L)/3 for e in E}
        rq1[w] = sum(qe.values())/2
        ie = {e: sum(q1icc[(s,w,'ICC',e,lay)] for lay in L)/3 for e in E}
        ricc[w] = sum(ie.values())/2
        assert abs(ps['risk'][w] - rchip[w]) < 1e-12, (s,w,'risk')
        assert abs(ps['reused_q1_risk'][w] - rq1[w]) < 1e-12
        assert abs(ps['reused_icc_risk'][w] - ricc[w]) < 1e-12
        assert abs(ps['d_chip'][w] - (rchip[w]-rq1[w])) < 1e-12
        assert abs(ps['d_icc'][w] - (ricc[w]-rq1[w])) < 1e-12
    assert abs(ps['theta_chip'] - (ps['d_chip']['W_D']-ps['d_chip']['W_O'])) < 1e-12
for w in W:
    md = sum(agg['per_source'][s]['d_chip'][w] for s in srcs)/11
    assert abs(agg['source_equal_mean_d_chip'][w] - md) < 1e-12
    lo, hi = agg['source_range_d_chip'][w]
    vals = [agg['per_source'][s]['d_chip'][w] for s in srcs]
    assert abs(lo-min(vals)) < 1e-12 and abs(hi-max(vals)) < 1e-12
mt = sum(agg['per_source'][s]['theta_chip'] for s in srcs)/11
assert abs(agg['source_equal_mean_theta_chip'] - mt) < 1e-12
assert len(agg['leave_one_source_out_theta_chip']) == 11
for drop in srcs:
    loo = sum(agg['per_source'][s]['theta_chip'] for s in srcs if s != drop)/10
    assert abs(agg['leave_one_source_out_theta_chip'][drop] - loo) < 1e-12
print('AGGREGATE_XCHECK_OK per_source=11 risk/d_chip/d_icc/theta/mean/range/LOO match CELL csv to 1e-12')
print('MEAN_D_CHIP', json.dumps(agg['source_equal_mean_d_chip']))
print('MEAN_THETA_CHIP', agg['source_equal_mean_theta_chip'])
print('VERIFY_POST_BW_CHIP_PASS')
PYEOF