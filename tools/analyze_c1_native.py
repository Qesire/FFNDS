"""Source-held-out CPU analysis of executed WHOLE-SITE finite actions."""
import json
from pathlib import Path
import numpy as np


def analyze(root):
    run = json.loads((root / 'RUN.json').read_text())
    if run['status'] != 'COMPLETE_NATIVE_C1_FIXED_SCOPE' or len(run['finite_scopes']) != 72:
        raise RuntimeError('Require complete C1 native data before diagnostic fitting')
    from c1_contract import SOURCES, BLOCKS, atomic, sha
    arms = ['UB1', 'swap_A', 'swap_B']; rows = []
    for s in run['finite_scopes']:
        path = root / Path(s['payload_HF_relative']).name
        if sha(path) != s['payload_sha256']: raise RuntimeError('Native C1 feature payload changed')
        with np.load(path, allow_pickle=False) as data:
            h = (data['h256'].astype(np.uint32) << 16).view(np.float32)
            current = np.concatenate([h[:, :8].mean(0), np.sqrt(np.mean(h[:, :8] ** 2, axis=0))])
            pre = data['pre_features'].astype(np.float64)
            added = np.concatenate([pre.mean(0), [np.log(np.maximum(pre[:, 1], 1e-12)).mean(),
                                                   data['norm_rstd'].astype(np.float64).mean()]])
            losses = np.array([data['nextnorm_' + a].sum() for a in arms], dtype=np.float64)
            temporal = {zone: {a: float(data['nextnorm_' + a].reshape(21, -1)[sl].sum()) for a in arms}
                        for zone, sl in [('conditioned_t0', slice(0, 1)), ('later20', slice(1, 21))]}
        if not np.isfinite(np.r_[current, added, losses]).all(): raise RuntimeError('Nonfinite C1 analysis input')
        rows.append({'source': s['source'], 'step': s['step'], 'block': s['block'],
                     'current': current.tolist(), 'added': added.tolist(), 'losses': losses.tolist(), 'zones': temporal})
    losses = np.array([r['losses'] for r in rows]); src = np.array([r['source'] for r in rows])
    position = np.array([r['step'] * 3 + BLOCKS.index(r['block']) for r in rows])
    onehot = np.eye(12)[position]
    current = np.array([r['current'] for r in rows]); added = np.array([r['added'] for r in rows])
    folds = [[0, 8, 16], [2, 10, 20]]
    predictions = {name: np.zeros(len(rows), dtype=int) for name in ['fixed_global', 'fixed_position', 'current', 'plus_scale_attention', 'shuffled_added']}
    for training in folds:
        train = np.isin(src, training); test = ~train
        predictions['fixed_global'][test] = int(losses[train].sum(0).argmin())
        for p in range(12):
            predictions['fixed_position'][test & (position == p)] = int(losses[train & (position == p)].sum(0).argmin())
        scales = np.array([losses[train & (position == p), 0].mean() for p in range(12)])
        if not np.all(scales > 0): raise RuntimeError('Undefined normalized action target')
        targets = (losses[:, [0]] - losses[:, 1:]) / scales[position, None]
        shuffled = added.copy()
        # Shuffle within position, separately in train/test source groups.
        # No held-out features or labels enter training.
        for flag in (train, test):
            for p in range(12):
                ids = np.flatnonzero(flag & (position == p)); shuffled[ids] = added[np.roll(ids, 1)]
        for name, extra in [('current', None), ('plus_scale_attention', added), ('shuffled_added', shuffled)]:
            x = np.column_stack([onehot, current] + ([] if extra is None else [extra]))
            mean, std = x[train].mean(0), x[train].std(0); std[std < 1e-8] = 1
            x = (x - mean) / std
            x = np.column_stack([np.ones(len(x)), x])
            penalty = np.eye(x.shape[1]); penalty[0, 0] = 0
            coef = np.linalg.solve(x[train].T @ x[train] + penalty, x[train].T @ targets[train])
            gains = x[test] @ coef
            predictions[name][test] = np.column_stack([np.zeros(test.sum()), gains]).argmax(1)
    oracle = losses.argmin(1); result = {}
    for name, decisions in {**predictions, 'oracle_whole_site': oracle, 'UB1': np.zeros(len(rows), dtype=int)}.items():
        selected = losses[np.arange(len(rows)), decisions]
        result[name] = {'total_nextnorm_SSE': float(selected.sum()),
                        'whole_site_action_counts': {a: int((decisions == i).sum()) for i, a in enumerate(arms)},
                        'per_source_SSE': {str(s): float(selected[src == s].sum()) for s in SOURCES},
                        'total_regret_to_same_action_oracle': float((selected - losses.min(1)).sum())}
    out = {'status': 'COMPLETE_C1_NATIVE_WHOLE_SITE_SOURCE_HELD_OUT_DIAGNOSTIC',
           'actual_sources': 6, 'actual_positions_per_source': 12, 'actual_complete_action_scopes': 72,
           'arms': arms, 'fit_protocol': 'Two complementary3source training/3source test folds; ridge1;16currentR256summary and8pre-norm/attention features; source-separated within-position shuffle',
           'CPU_closed_form_diagnostic_regressions': 6, 'neural_training_updates': 0,
           'decision_unit': 'WHOLE31668ROW_FFN_SITE; no per-row or per-bin output-splice policy evaluation',
           'horizon': 'Actual next block norm1 before modulation/attention, not next full block/flow/video',
           'scope': 'Six previously selected/exposed discovery sources, two per subject/background/camera; six independent source groups, not72independent samples. Offline decisions not an executed dynamic policy or speed result.',
           'results': result, 'native_whole_site_rows': rows,
           'official_video_quality_or_Pareto_or_native_policy_speed_result': False,
           'automatic_followup': False}
    atomic(root / 'ANALYSIS.json', out)
    print(json.dumps({'C1_actual_native_analysis': {k: v['total_nextnorm_SSE'] for k, v in result.items()}}), flush=True)
    return out
