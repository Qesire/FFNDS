"""Read native C1 scalar labels to compare two finite causal boundaries.

No Torch, weights, reconstructed selector or model execution. The label-wise
row rankings describe target disagreement, not an executed tile/row policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'research/20261024/C1_boundary_target_audit_plan.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ranks(x):
    order = np.argsort(x, kind='stable')
    sorted_x = x[order]
    starts = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    result = np.empty(len(x), dtype=np.float64)
    result[order] = np.repeat((starts + ends - 1) / 2, ends - starts)
    return result


def summary(rows):
    p = sum(r['post_total_SSE'] for r in rows)
    n = sum(r['nextnorm_total_SSE'] for r in rows)
    return {
        'scopes': len(rows),
        'mean_post_nextnorm_Spearman': float(np.mean([r['Spearman'] for r in rows])),
        'mean_top_quarter_overlap_fraction': float(np.mean([r['selected_overlap_fraction'] for r in rows])),
        'post_ranking_capture_of_post_loss': sum(r['post_selected_post_SSE'] for r in rows) / p,
        'nextnorm_ranking_capture_of_post_loss': sum(r['nextnorm_selected_post_SSE'] for r in rows) / p,
        'post_ranking_capture_of_nextnorm_loss': sum(r['post_selected_nextnorm_SSE'] for r in rows) / n,
        'nextnorm_ranking_capture_of_nextnorm_loss': sum(r['nextnorm_selected_nextnorm_SSE'] for r in rows) / n,
        'mean_conditioned_t0_fraction_in_post_selection': float(np.mean([r['post_selected_t0_fraction'] for r in rows])),
        'mean_conditioned_t0_fraction_in_nextnorm_selection': float(np.mean([r['nextnorm_selected_t0_fraction'] for r in rows])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve first saved-data target audit')
    plan = read(ROOT / PLAN)
    assert sha(ROOT / 'research/20261023/C1N0_native_results.json') == plan['native_results_sha256']
    assert sha(ROOT / 'tools/c1_native_probe.py') == plan['native_capture_source_sha256']
    assert sha(__file__) == plan['analysis_source_sha256']
    assert np.array_equal(ranks(np.array([3, 1, 1, 4.])), [2, .5, .5, 3])
    run = read(args.root / 'RUN.json')
    assert run['status'] == 'COMPLETE_NATIVE_C1_FIXED_SCOPE' and len(run['finite_scopes']) == 72
    rows = []
    for scope in run['finite_scopes']:
        path = args.root / Path(scope['payload_HF_relative']).name
        assert sha(path) == scope['payload_sha256']
        with np.load(path, allow_pickle=False) as data:
            p, n = data['post_UB1'].astype(np.float64), data['nextnorm_UB1'].astype(np.float64)
        assert p.shape == n.shape == (31668,) and np.isfinite(p).all() and np.isfinite(n).all()
        assert (p >= 0).all() and (n >= 0).all() and p.sum() > 0 and n.sum() > 0
        ids_p = np.argsort(-p, kind='stable')[:7917]
        ids_n = np.argsort(-n, kind='stable')[:7917]
        rows.append({
            'source': scope['source'], 'step': scope['step'], 'block': scope['block'],
            'payload_sha256': scope['payload_sha256'],
            'rows': 31668, 'selected_rows': 7917,
            'Spearman': float(np.corrcoef(ranks(p), ranks(n))[0, 1]),
            'selected_overlap_fraction': len(np.intersect1d(ids_p, ids_n, assume_unique=True)) / 7917,
            'post_total_SSE': float(p.sum()), 'nextnorm_total_SSE': float(n.sum()),
            'post_selected_post_SSE': float(p[ids_p].sum()),
            'nextnorm_selected_post_SSE': float(p[ids_n].sum()),
            'post_selected_nextnorm_SSE': float(n[ids_p].sum()),
            'nextnorm_selected_nextnorm_SSE': float(n[ids_n].sum()),
            'post_selected_t0_fraction': float((ids_p < 1508).mean()),
            'nextnorm_selected_t0_fraction': float((ids_n < 1508).mean()),
        })
    output = {
        'status': 'COMPLETE_SAVED_NATIVE_C1_BOUNDARY_TARGET_AUDIT',
        'plan_sha256': sha(ROOT / PLAN), 'source_sha256': sha(__file__),
        'independent_source_clusters': len({r['source'] for r in rows}),
        'scope': '72 same-prefix complete compensated dense-versus-UB1 native FFN observations, 31668 scalar row losses per scope; all discovery-exposed.',
        'post_target': 'Actual BF16 residual writeback state SSE to the matched-prefix dense action at the current FFN boundary.',
        'nextnorm_target': 'Actual next-block norm1 SSE before modulation/attention. It omits further attention/FFN and identity-path propagation.',
        'selection_scope': 'Post-hoc scalar-label top7917 rows, independently per scope. Not a feasible pre-W1 selector, executed row/tile action, complete-block/step value, video or population oracle bound.',
        'pooled': summary(rows),
        'per_source': {str(s): summary([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
        'per_step': {str(s): summary([r for r in rows if r['step'] == s]) for s in range(4)},
        'rows': rows,
        'new_model_GPU_FFN_norm_selector_calls': 0,
        'C2N1_score_actions_data_and_scope_unchanged': True,
        'causal_interpretation': 'Disagreement establishes sensitivity to the finite target boundary. Scalar SSE alone cannot identify radial error, its direction, full downstream value, or why a video differs.',
        'automatic_model_followup': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'status': output['status'], 'pooled': output['pooled'], 'source_clusters': output['independent_source_clusters']}))


if __name__ == '__main__':
    main()
