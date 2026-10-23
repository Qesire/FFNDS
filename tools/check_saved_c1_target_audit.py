"""Validate public C1 target-audit provenance and scalar reductions, CPU only."""
import json
import math
from pathlib import Path
from analyze_saved_c1_target_disagreement import PLAN, ROOT, read, sha, summary


def equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def main():
    plan = read(ROOT / PLAN)
    result = read(ROOT / 'research/20261024/C1_boundary_target_audit.json')
    assert result['plan_sha256'] == sha(ROOT / PLAN)
    assert result['source_sha256'] == plan['analysis_source_sha256'] == sha(ROOT / 'tools/analyze_saved_c1_target_disagreement.py')
    assert plan['native_results_sha256'] == sha(ROOT / 'research/20261023/C1N0_native_results.json')
    assert plan['native_capture_source_sha256'] == sha(ROOT / 'tools/c1_native_probe.py')
    native = read(ROOT / 'research/20261023/C1N0_native_results.json')
    bindings = {(r['source'], r['step'], r['block']): r['payload_sha256'] for r in native['finite_scopes']}
    rows = result['rows']
    assert len(rows) == len(bindings) == 72
    assert {(r['source'], r['step'], r['block']): r['payload_sha256'] for r in rows} == bindings
    for row in rows:
        assert row['rows'] == 31668 and row['selected_rows'] == 7917
        assert -1 <= row['Spearman'] <= 1 and 0 <= row['selected_overlap_fraction'] <= 1
        assert row['post_selected_post_SSE'] >= row['nextnorm_selected_post_SSE']
        assert row['nextnorm_selected_nextnorm_SSE'] >= row['post_selected_nextnorm_SSE']
    assert equal(result['pooled'], summary(rows))
    for field, key in [('per_source', 'source'), ('per_step', 'step')]:
        for label, expected in result[field].items():
            assert equal(expected, summary([r for r in rows if r[key] == int(label)]))
    assert result['independent_source_clusters'] == 6
    assert result['new_model_GPU_FFN_norm_selector_calls'] == 0
    assert result['C2N1_score_actions_data_and_scope_unchanged'] is True
    print(json.dumps({'status': 'PASS_SAVED_C1_TARGET_AUDIT', 'native_scopes': 72, 'source_clusters': 6, 'new_model_calls': 0}))


if __name__ == '__main__':
    main()
