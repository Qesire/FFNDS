"""Frozen saved-only dense-exception risk diagnostic; never runs a model."""
import argparse
from pathlib import Path
import numpy as np

from c1_contract import BLOCKS, ROOT, SOURCES, atomic, read, sha


def analyze(root):
    plan_path = ROOT / 'research/20261023/C1_saved_dense_value_CPU_plan.json'
    plan = read(plan_path); run = read(root / 'RUN.json')
    if run['status'] != 'COMPLETE_NATIVE_C1_FIXED_SCOPE' or len(run['finite_scopes']) != 72:
        raise RuntimeError('Complete C1 source data required')
    current, pre, loss, source, position, timebin, site, rowids = [], [], [], [], [], [], [], []
    for i, r in enumerate(run['finite_scopes']):
        p = root / Path(r['payload_HF_relative']).name
        if sha(p) != r['payload_sha256']:
            raise RuntimeError('Changed C1 payload')
        with np.load(p, allow_pickle=False) as d:
            ids = d['row_ids']; h = (d['h256'].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            current.append(np.column_stack([h[:, :8], h[:, :8] ** 2, (h ** 2).sum(1)]))
            pre.append(np.column_stack([d['pre_features'], np.log(np.maximum(d['pre_features'][:, 1], 1e-12)),
                                        np.log(np.maximum(d['norm_rstd'], 1e-12))]))
            loss.append(d['nextnorm_UB1'][ids]); source += [r['source']] * len(ids)
            position += [r['step'] * 3 + BLOCKS.index(r['block'])] * len(ids)
            timebin.extend(ids // 1508); site += [i] * len(ids); rowids.extend(ids)
    current, pre, loss = map(np.concatenate, [current, pre, loss])
    source, position, timebin, site, rowids = map(np.array, [source, position, timebin, site, rowids])
    if not np.isfinite(np.c_[current, pre, loss]).all() or (loss <= 0).any():
        raise RuntimeError('Invalid dense-value labels/features')
    logloss = np.log(loss); strata = position * 21 + timebin
    structural = np.c_[np.eye(12)[position], np.eye(21)[timebin]]
    names = ['training_position_timebin_fixed_risk', 'current_features', 'plus_pre_norm_attention', 'shuffled_added']
    scores = {n: np.zeros(len(loss)) for n in names}
    for train_sources in plan['source_folds']:
        train = np.isin(source, train_sources); test = ~train
        baseline = np.array([logloss[train & (strata == j)].mean() for j in range(252)])
        target = logloss - baseline[strata]
        scores[names[0]][test] = baseline[strata[test]]
        shuffled = pre.copy()
        for flag in (train, test):
            for j in range(252):
                ids = np.flatnonzero(flag & (strata == j))
                # One complete16-row source bin cycles to a different source.
                shuffled[ids] = pre[np.roll(ids, 16)]
        for name, extra in [('current_features', None), ('plus_pre_norm_attention', pre), ('shuffled_added', shuffled)]:
            x = np.c_[structural, current] if extra is None else np.c_[structural, current, extra]
            mean, std = x[train].mean(0), x[train].std(0); std[std < 1e-8] = 1
            x = np.c_[np.ones(len(x)), (x - mean) / std]
            penalty = np.eye(x.shape[1]); penalty[0, 0] = 0
            coef = np.linalg.solve(x[train].T @ x[train] + penalty, x[train].T @ target[train])
            scores[name][test] = baseline[strata[test]] + x[test] @ coef
    scores['sample_order_stride'] = -np.array([j % 336 % 4 for j in range(len(loss))], dtype=np.float64)
    scores['native_rowID_hash'] = -((rowids * 1103515245 + 12345) % (1 << 31)).astype(np.float64)
    scores['same336row_top84_oracle'] = loss
    output = {}
    for name, values in scores.items():
        saved = np.zeros(len(loss), dtype=bool)
        for i in range(72):
            ids = np.flatnonzero(site == i)
            selected = ids[np.argsort(-values[ids], kind='stable')[:84]]
            saved[selected] = True
        output[name] = {'sampled_rows_selected': int(saved.sum()),
                        'captured_measured_nextnorm_loss': float(loss[saved].sum()),
                        'fraction_of_all_sampled_measured_nextnorm_loss': float(loss[saved].sum() / loss.sum()),
                        'per_source_captured_loss': {str(s): float(loss[saved & (source == s)].sum()) for s in SOURCES}}
    out = {'status': 'COMPLETE_SAVED_ONLY_C1_DENSE_EXCEPTION_RISK_DIAGNOSTIC',
           'plan_sha256': sha(plan_path), 'source_sha256': sha(__file__), 'actual_source_clusters': 6,
           'actual_saved_feature_rows': len(loss), 'actual_sampled_rows_per_site': 336,
           'hypothetical_selected_sample_rows_per_site': 84,
           'total_sampled_measured_nextnorm_loss': float(loss.sum()), 'results': output,
           'new_model_calls': 0, 'additional_CPU_closed_form_regressions': 6, 'neural_training_updates': 0,
           'interpretation': plan['decision'], 'current_baseline_limit':
               'h256 features are not the exact R16 gated correction-energy selector. This is a conditional diagnostic, not a head-to-head reproduction or an executed dense-exception policy.',
           'automatic_model_followup': False, 'video_quality_or_actual_policy_speed_result': False}
    atomic(root / 'DENSE_VALUE_CPU.json', out)
    print({n: v['fraction_of_all_sampled_measured_nextnorm_loss'] for n, v in output.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    analyze(p.parse_args().root)
