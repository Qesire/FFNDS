"""Fixed R12 D0 allocation, run budget and diagnostic decisions (stdlib only)."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'R12E1_D0_EQUAL_BUDGET_NATIVE_VALUE'
STATE = 'R4_CLOSED_R12_D0_NATIVE_VALUE_REGISTERED'
PROTOCOL = ROOT / 'research/20261022/R12_D0_registered_protocol_worker_migration.json'
INDICES = [0, 5, 10, 15]
ACTIONS = {'N0': [162, 162], 'NA': [216, 108], 'NB': [108, 216]}
GATES = {'oracle_over_best_fixed_max': .99, 'oracle_over_UB1_max': .99,
         'headroom_over_eta_min_exclusive': 10}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jobs(phase, baseline='UB1', shard=0):
    if phase == 'controls' and baseline in ('DQ', 'UB1'):
        return f'control_{baseline}', [(i, baseline) for i in INDICES]
    if phase in ('baselines', 'interventions') and shard in (0, 1):
        arms = ['N0', 'N0_repeat'] if phase == 'baselines' else ['NA', 'NB']
        return f'{phase}_shard{shard}', [(i, a) for j, i in enumerate(INDICES)
                                      if j % 2 == shard for a in arms]
    raise ValueError('Unregistered D0 job')


def groups(arm, step, block):
    if arm == 'DQ':
        return 216
    if arm not in ('UB1', 'N0_repeat', *ACTIONS):
        raise ValueError('Unknown D0 action')
    if step == 3 and block in (30, 31) and arm != 'UB1':
        return ACTIONS['N0' if arm == 'N0_repeat' else arm][block - 30]
    return [54, 108, 108, 162][step]


def residual_count(arm):
    return 0 if arm == 'DQ' else 160 if arm == 'UB1' else 158


def decide(matrix, ub1, repeats):
    """All three raw actions and the separate UB1 comparator; no fitted policy."""
    if set(matrix) != set(INDICES) or set(ub1) != set(INDICES) or set(repeats) != set(INDICES):
        raise ValueError('Incomplete fixed source matrix')
    for i in INDICES:
        values = [*matrix[i].values(), ub1[i], repeats[i]]
        if set(matrix[i]) != set(ACTIONS) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid fixed action losses')
        if ub1[i] <= 0:
            raise ValueError('UB1 noise denominator must be positive')
    totals = {a: math.fsum(matrix[i][a] for i in INDICES) for a in ACTIONS}
    fixed = min(ACTIONS, key=totals.get)
    oracle = math.fsum(min(matrix[i].values()) for i in INDICES)
    incumbent = math.fsum(ub1.values())
    eta = max(abs(matrix[i]['N0'] - repeats[i]) / ub1[i] for i in INDICES)
    headroom = 1 - oracle / totals[fixed] if totals[fixed] else 0.
    controls = {'oracle_beats_best_fixed_1pct': oracle <= .99 * totals[fixed],
                'oracle_beats_UB1_1pct': oracle <= .99 * incumbent,
                'headroom_exceeds_10eta': headroom > 10 * eta}
    return {'total_loss_by_action': totals, 'best_fixed_action': fixed,
            'best_fixed_total_loss': totals[fixed], 'oracle_total_loss': oracle,
            'frozen_UB1_total_loss': incumbent, 'oracle_reduction_over_best_fixed': headroom,
            'oracle_reduction_over_UB1': 1 - oracle / incumbent, 'noise_eta': eta,
            'gates': controls, 'diagnostic_gate_pass': all(controls.values()),
            'oracle_actions': {str(i): min(ACTIONS, key=matrix[i].get) for i in INDICES}}


def completed_cases(out, names, identity):
    """Validate preceding phase receipts and all their flow payloads."""
    bindings, cases = {}, {}
    for name in names:
        path = Path(out) / name / 'RUN.json'
        job = read(path)
        if job['status'] != 'PASS_ALL_REGISTERED_RUNS' or job['scientific_identity'] != identity:
            raise RuntimeError('Prior phase did not pass with identical frozen bindings')
        if name.startswith('control_'):
            _, expected = jobs('controls', name.removeprefix('control_'))
        else:
            phase, shard = name.split('_shard')
            _, expected = jobs(phase, shard=int(shard))
        if len(job['completed']) != len(expected) or {(r['index'], r['arm']) for r in job['completed']} != set(expected):
            raise RuntimeError('Incomplete or duplicate previous phase')
        for row in job['completed']:
            if sha(row['path']) != row['sha256']:
                raise RuntimeError('Case receipt changed')
            case = read(row['path'])
            if case['status'] != 'PASS_R12_D0_NATIVE_CASE' or (case['index'], case['arm']) != (row['index'], row['arm']):
                raise RuntimeError('Failed/mislabeled previous case')
            final = case['steps'][3]
            if sha(final['path']) != final['sha256']:
                raise RuntimeError('Final flow payload changed')
            cases[row['index'], row['arm']] = case
        bindings[name] = sha(path)
    return bindings, cases


def require_repeats(cases):
    for i in INDICES:
        a, b = cases[i, 'N0'], cases[i, 'N0_repeat']
        if (a['steps'] != b['steps'] or a['scopes'] != b['scopes']):
            # File locations are intentionally distinct; compare numerical receipts.
            clean = lambda c: [{k: v for k, v in r.items() if k not in ('path', 'sha256')} for r in c['steps']]
            if clean(a) != clean(b) or a['scopes'] != b['scopes']:
                raise RuntimeError('Repeated N0 native bytes/controls differ')


def prerequisites(out, phase, identity):
    if phase == 'controls':
        return {}
    names = ['control_DQ', 'control_UB1']
    if phase == 'interventions':
        names += ['baselines_shard0', 'baselines_shard1']
    bindings, cases = completed_cases(out, names, identity)
    if phase == 'interventions':
        require_repeats(cases)
    return bindings
