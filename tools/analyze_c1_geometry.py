"""Describe measured finite normalization transport on saved C1 data only."""
import argparse
from pathlib import Path
import numpy as np

from c1_contract import ARMS, atomic, read, sha


def analyze(root):
    run = read(root / 'RUN.json')
    if run['status'] != 'COMPLETE_NATIVE_C1_FIXED_SCOPE' or len(run['finite_scopes']) != 72:
        raise RuntimeError('Complete original C1 scope required')
    sparse = ARMS[1:]; rows = []
    for r in run['finite_scopes']:
        path = root / Path(r['payload_HF_relative']).name
        if sha(path) != r['payload_sha256']:
            raise RuntimeError('Changed native C1 payload')
        with np.load(path, allow_pickle=False) as d:
            post = {a: d['post_' + a] for a in sparse}
            norm = {a: d['nextnorm_' + a] for a in sparse}
            p = np.array([post[a].sum() for a in sparse]); n = np.array([norm[a].sum() for a in sparse])
            changes = {}
            for a in sparse[1:]:
                local = post[a] - post['UB1']; downstream = norm[a] - norm['UB1']
                zones = {}
                for label, sl in [('conditioned_t0', slice(0, 1508)), ('later20', slice(1508, None))]:
                    zones[label] = {'post_delta_vs_UB1': float(local[sl].sum()),
                                    'nextnorm_delta_vs_UB1': float(downstream[sl].sum()),
                                    'rows_nextnorm_better_than_UB1': int((downstream[sl] < 0).sum()),
                                    'rows_local_vs_nextnorm_sign_reversed': int((local[sl] * downstream[sl] < 0).sum())}
                changes[a] = {'post_delta_vs_UB1': float(local.sum()),
                              'nextnorm_delta_vs_UB1': float(downstream.sum()),
                              'whole_site_sign_reversed': bool(local.sum() * downstream.sum() < 0),
                              'zones': zones}
            ratio = norm['UB1'] / np.maximum(post['UB1'], np.finfo(np.float64).tiny)
            ids = d['row_ids']
            xstd = d['pre_features'][:, 1].astype(np.float64)
            nrstd = d['norm_rstd'].astype(np.float64)
            actual = ratio[ids]
            corr = lambda a, b: None if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])
            row = {'source': r['source'], 'step': r['step'], 'block': r['block'],
                   'best_whole_site_post_action': sparse[int(p.argmin())],
                   'best_whole_site_nextnorm_action': sparse[int(n.argmin())],
                   'post_vs_nextnorm_best_action_changed': bool(p.argmin() != n.argmin()),
                   'actions': changes, 'UB1_finite_loss_ratio_nextnorm_over_post':
                       {str(q): float(np.quantile(ratio, q)) for q in (.1, .5, .9)},
                   'sampled_current_rstd_squared_vs_actual_finite_ratio_Pearson': corr(nrstd ** 2, actual),
                   'sampled_post_attention_std_min_max': [float(xstd.min()), float(xstd.max())]}
            rows.append(row)
    out = {'status': 'SAVED_NATIVE_C1_FINITE_GEOMETRY_DESCRIPTION', 'source_sha256': sha(__file__),
           'actual_scopes': 72, 'source_clusters': 6,
           'whole_site_best_action_changed_after_nextnorm': sum(r['post_vs_nextnorm_best_action_changed'] for r in rows),
           'whole_site_delta_sign_reversals_vs_UB1': sum(a['whole_site_sign_reversed'] for r in rows for a in r['actions'].values()),
           'rows': rows,
           'interpretation': 'Actual finite error transport and ranking on identical complete compensated FFNs. Current FFN norm statistics are upstream of the measured next norm; their squared inverse standard deviation is a proxy, not that next norms exact Jacobian. Correlation is descriptive, not incremental information or causal sufficiency.',
           'row_counts_are_not_executed_rowwise_policy_or_extra_independent_sources': True,
           'new_model_calls': 0, 'video_quality_or_speed_acceptance': False}
    atomic(root / 'GEOMETRY.json', out)
    print({k: out[k] for k in ['actual_scopes', 'whole_site_best_action_changed_after_nextnorm', 'whole_site_delta_sign_reversals_vs_UB1']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    analyze(parser.parse_args().root)
