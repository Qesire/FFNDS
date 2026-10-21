"""Exact two-arm whole-source quality oracle, with one shared choice across metrics."""
import numpy as np


def enumerate_assignments(dense, first, second, *, block_size=4096):
    dense, first, second = [np.asarray(v, dtype=np.float64) for v in [dense, first, second]]
    if (dense.ndim != 2 or dense.shape[1] != 3 or not 1 <= len(dense) <= 20
            or first.shape != dense.shape or second.shape != dense.shape
            or not all(np.isfinite(v).all() for v in [dense, first, second])
            or np.any(dense <= 0) or block_size < 1):
        raise ValueError('Paired finite n-by3 scores with1<=n<=20 and positive baselines required')
    count = len(dense)
    baseline = dense.mean(0)
    start_scores, delta = first.mean(0), (second - first) / count
    best_value, best_mask, best_scores = float('inf'), None, None
    feasible = 0
    for start in range(0, 1 << count, block_size):
        masks = np.arange(start, min(start + block_size, 1 << count), dtype=np.uint64)
        bits = ((masks[:, None] >> np.arange(count, dtype=np.uint64)) & 1).astype(np.float64)
        scores = start_scores + bits @ delta
        drops = 1 - scores / baseline
        worst = drops.max(1)
        feasible += int(np.count_nonzero(np.all(scores >= .98 * baseline, axis=1)))
        local = int(np.argmin(worst))
        if worst[local] < best_value:
            best_value, best_mask, best_scores = float(worst[local]), int(masks[local]), scores[local].copy()
    chosen = np.array([(best_mask >> i) & 1 for i in range(count)], dtype=bool)
    direct_scores = np.where(chosen[:, None], second, first).mean(0)
    if not np.allclose(best_scores, direct_scores, rtol=0, atol=2e-15):
        raise RuntimeError('Oracle additive enumeration differs from direct whole-source selection')
    passed = bool(np.all(direct_scores >= .98 * baseline))
    if passed != bool(feasible):
        raise RuntimeError('Numerically ambiguous feasibility boundary; preserve outcome for review')
    return {'assignments_enumerated': 1 << count, 'joint_feasible_assignments': feasible,
        'oracle_joint_gate': passed, 'best_mask': best_mask,
        'second_arm_indices': np.flatnonzero(chosen).tolist(),
        'dense_source_means': baseline.tolist(), 'oracle_source_means': direct_scores.tolist(),
        'oracle_relative_drops': (1 - direct_scores / baseline).tolist(),
        'minimax_relative_drop': best_value,
        'best_unconstrained_imaging_relative_drop': float(1 - np.maximum(first[:, 1], second[:, 1]).mean() / baseline[1])}
