#!/bin/bash
# Read-only post-run integrity verification for B_CHIP_STATS v1 (job 164928).
set -euo pipefail
ST="/public/home/scc/pb23061276/projects/wan-quant/work/g20_two_track_core_20261114/B_CHIP_STATS/v1/stage"
PY="/public/home/scc/pb23061276/miniconda3/envs/hif4/bin/python3.11"
cd "$ST"
"$PY" -B - "$ST" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
st = Path(sys.argv[1])
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
scope = json.loads((st/'scope.json').read_text())
scope_sha = sha(st/'scope.json')
print('SCOPE_SHA', scope_sha)
assert scope_sha == '13df65288ca31df682ebb041070ee53603212730d714c9bdc9f187d4b10fa2e9', 'scope sha'
bad = []
for name, want in {**scope['code_sha256'], **scope['frozen_inputs_sha256']}.items():
    got = sha(st/name); ok = got == want
    print('BIND', name, 'OK' if ok else 'MISMATCH')
    if not ok: bad.append(name)
assert not bad, f'hash mismatch: {bad}'
audit_sha = sha(st/'ROOT_SELF_AUDIT.txt')
print('AUDIT_SHA', audit_sha)
assert audit_sha == '30069cd2472782bb5c036edd2280ab1c7dc845c0f23dbd22965c4ef8ca192d72', 'audit sha'
out = st/'output'
summ = json.loads((out/'SUMMARY.json').read_text())
inputs = out/'B_CHIP_P64_INPUTS.json'
manifest_sha = sha(inputs)
print('INPUTS_MANIFEST_SHA', manifest_sha)
assert manifest_sha == summ['inputs_manifest_sha256'], 'manifest sha != SUMMARY'
doc = json.loads(inputs.read_text())
assert doc['fixed64_token_id_sha256_le_int64'] == 'd53fcf1b0930d6b1ea78b3f20b11dccb7b1c9d27abcad922866b2f26c78bf895'
assert doc['capture_scope_sha256'] == scope_sha
assert doc['parent_recipe_sha256'] == 'ee780352c8cf1c1e4efc8aea7c354b3be225c525141942cbd9c15d24cabae5b6'
assert len(doc['fixed64_token_ids']) == 64 and doc['rows_are_panel_first64'] is False
assert doc['selected_k'] == 5120 and doc['grid'] == [21,29,52] and doc['flattened_tokens'] == 31668
srcs = doc['sources']; assert len(srcs) == 11
assert {s['cycle_role'] for s in srcs.values()} == {'DEV'}
SITES = ['high04','high19','high34','low04','low19','low34']
n_pt = 0; tot_rows = 0; pt_bytes = 0
for rid, rec in srcs.items():
    assert rec['same_state_all_match'] is True, f'{rid} source same_state'
    assert rec['chip_cells_unblocked_count'] == 6, f'{rid} cells'
    assert sorted(rec['sites'].keys()) == SITES, f'{rid} site keys'
    for sk in SITES:
        sv = rec['sites'][sk]
        assert sv['chip_cell_unblocked'] is True, f'{rid}/{sk} cell'
        ss = sv['same_state']
        assert all(ss[k] is True for k in ('anchor_panel_u_bit_match','hit_panel_u_bit_match','anchor_gate_bit_match','hit_gate_bit_match')), f'{rid}/{sk} same_state {ss}'
        for ep in ('anchor','hit'):
            e = sv[ep]
            assert e['U_fixed64_shape'] == [64,5120] and e['U_fixed64_dtype'] == 'bfloat16', f'{rid}/{sk}/{ep} shape'
            p = Path(e['path']); assert p.is_file(), f'missing {p}'
            assert p.stat().st_size == e['bytes'], f'{rid}/{sk}/{ep} bytes'
            assert sha(p) == e['file_sha256'], f'{rid}/{sk}/{ep} pt sha'
            n_pt += 1; tot_rows += 64; pt_bytes += e['bytes']
print('N_PT_FILES', n_pt, 'EXPECTED', 11*6*2)
assert n_pt == 132
print('TOTAL_FIXED64_U_ROWS', tot_rows, 'EXPECTED', 11*64*12)
assert tot_rows == 8448 == summ['total_fixed64_U_rows_captured']
print('PT_TOTAL_BYTES', pt_bytes)
assert summ['status'] == 'B_CHIP_STATS_COMPLETE'
assert summ['new_D4_full_executions'] == 11 and summ['same_state_pass_sources'] == 11
assert summ['bw_chip_cells_unblocked'] == 66 == summ['bw_chip_cells_expected']
assert summ['same_state_fail_sources'] == [] and summ['new_O40_full_executions'] == 0
caps = sorted(out.glob('CHIP_CAPTURE_vbi_*.json')); assert len(caps) == 11
for c in caps:
    cd = json.loads(c.read_text())
    assert cd['capture_sites'] == 12, f'{c.name} capture_sites'
    assert cd['record_id'] == c.stem.replace('CHIP_CAPTURE_',''), f'{c.name} record_id'
print('N_CAPTURE_SIDECARS', len(caps))
print('BANK_MANIFEST_SHA', summ['bank_manifest_sha256'])
# torch spot-check ONE tensor: U_fixed64 (64,5120) bf16 finite + U_panel_check (512,5120)
import torch
rec0 = srcs['vbi_1013']['sites']['high04']['anchor']
t = torch.load(rec0['path'], map_location='cpu', weights_only=False)
keys = sorted(t.keys()) if isinstance(t, dict) else 'TENSOR'
print('PT_KEYS', keys)
uf = t['U_fixed64'] if isinstance(t, dict) else t
assert tuple(uf.shape) == (64,5120) and uf.dtype == torch.bfloat16 and torch.isfinite(uf.float()).all()
print('U_FIXED64_SPOTCHECK_OK', tuple(uf.shape), str(uf.dtype))
if isinstance(t, dict) and 'U_panel_check' in t:
    up = t['U_panel_check']; assert tuple(up.shape) == (512,5120)
    print('U_PANEL_CHECK_SPOTCHECK_OK', tuple(up.shape))
print('VERIFY_POST_CHIP_PASS')
PYEOF