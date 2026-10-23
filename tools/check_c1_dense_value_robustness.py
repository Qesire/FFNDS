"""One stronger current-information and fixed-risk control on exposed C1 data."""
import argparse
from pathlib import Path
import numpy as np
from c1_contract import BLOCKS, ROOT, SOURCES, atomic, read, sha


def main(root):
    run = read(root / 'RUN.json')
    plan_path = ROOT / 'research/20261023/C1_dense_value_robustness_plan.json'
    plan = read(plan_path)
    if run['status'] != 'COMPLETE_NATIVE_C1_FIXED_SCOPE':
        raise RuntimeError('Complete C1 data required')
    xx, zz, yy, ss, pp, tt, ii = [], [], [], [], [], [], []
    for i, r in enumerate(run['finite_scopes']):
        path = root / Path(r['payload_HF_relative']).name
        if sha(path) != r['payload_sha256']:
            raise RuntimeError('Changed C1 payload')
        with np.load(path, allow_pickle=False) as d:
            bf = lambda a: (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            h, v = bf(d['h256']), bf(d['V1']); ids = d['row_ids']
            xx.append(np.c_[h, h ** 2, (h ** 2).sum(1), v.mean(1), (v ** 2).sum(1), v[:, :8]])
            zz.append(np.c_[d['pre_features'], np.log(np.maximum(d['pre_features'][:, 1], 1e-12)),
                               np.log(np.maximum(d['norm_rstd'], 1e-12))])
            yy.append(d['nextnorm_UB1'][ids]); ss += [r['source']] * 336
            pp += [r['step'] * 3 + BLOCKS.index(r['block'])] * 336
            tt.extend(ids // 1508); ii += [i] * 336
    x, z, y = map(np.concatenate, [xx, zz, yy]); s, p, t, sites = map(np.array, [ss, pp, tt, ii])
    if not np.isfinite(np.c_[x, z, y]).all() or (y <= 0).any():
        raise RuntimeError('Invalid saved input')
    group = p * 21 + t; structural = np.c_[np.eye(12)[p], np.eye(21)[t]]
    names = ['fixed_arithmetic_mean_risk', 'rich_current', 'rich_current_plus_pre', 'rich_current_shuffled_pre']
    scores = {name: np.zeros(len(y)) for name in names}
    for sources in plan['source_folds']:
        train = np.isin(s, sources); test = ~train
        geometric = np.array([np.log(y[train & (group == j)]).mean() for j in range(252)])
        arithmetic = np.array([y[train & (group == j)].mean() for j in range(252)])
        target = np.log(y) - geometric[group]
        scores[names[0]][test] = arithmetic[group[test]]
        shuffled = z.copy()
        for flag in (train, test):
            for j in range(252):
                ids = np.flatnonzero(flag & (group == j)); shuffled[ids] = z[np.roll(ids, 16)]
        for name, extra in [('rich_current', None), ('rich_current_plus_pre', z), ('rich_current_shuffled_pre', shuffled)]:
            a = np.c_[structural, x] if extra is None else np.c_[structural, x, extra]
            mean, std = a[train].mean(0), a[train].std(0); std[std < 1e-8] = 1
            a = np.c_[np.ones(len(a)), (a - mean) / std]
            penalty = np.eye(a.shape[1]); penalty[0, 0] = 0
            coef = np.linalg.solve(a[train].T @ a[train] + penalty, a[train].T @ target[train])
            scores[name][test] = geometric[group[test]] + a[test] @ coef
    result = {}
    for name, score in scores.items():
        selected = np.zeros(len(y), dtype=bool)
        for i in range(72):
            ids = np.flatnonzero(sites == i); selected[ids[np.argsort(-score[ids], kind='stable')[:84]]] = True
        result[name] = {'captured_measured_nextnorm_loss': float(y[selected].sum()),
                        'fraction_of_all_sampled_measured_nextnorm_loss': float(y[selected].sum() / y.sum()),
                        'per_source_captured_loss': {str(j): float(y[selected & (s == j)].sum()) for j in SOURCES}}
    out = {'status': 'COMPLETE_EXPOSED_C1_STRONGER_CURRENT_INFORMATION_CONTROL',
           'plan_sha256': sha(plan_path), 'source_sha256': sha(__file__), 'current_feature_dimensions': x.shape[1],
           'added_feature_dimensions': z.shape[1], 'source_clusters': 6, 'saved_rows': len(y),
           'additional_CPU_closed_form_regressions': 6, 'new_model_calls': 0, 'neural_training_updates': 0,
           'results': result, 'interpretation': plan['exposure'],
           'video_quality_or_actual_policy_speed_result': False, 'automatic_model_followup': False}
    atomic(root / 'DENSE_VALUE_ROBUSTNESS.json', out)
    print({k: v['fraction_of_all_sampled_measured_nextnorm_loss'] for k, v in result.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
