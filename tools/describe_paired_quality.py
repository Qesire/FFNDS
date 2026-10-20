"""Describe source-level variability without changing the registered quality gate."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ar2_checkpoint import atomic_json, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparison', type=Path, required=True)
    ap.add_argument('--paired-csv', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    comparison = json.loads(args.comparison.read_text())
    if (comparison['status'] != 'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON'
            or comparison['unique_source_count'] != 20
            or comparison['paired_csv_sha256'] != digest(args.paired_csv)):
        raise RuntimeError('Complete same-identity20-source comparison required')
    rows = list(csv.DictReader(args.paired_csv.open()))
    metrics = ['aesthetic_quality', 'imaging_quality', 'motion_smoothness']
    sources = sorted({v['source_group'] for v in rows})
    if len(rows) != 60 or len(sources) != 20:
        raise RuntimeError('Exactly60 metric/source rows required; filename copies are not samples')
    dense, candidate = np.empty((20, 3)), np.empty((20, 3))
    for i, source in enumerate(sources):
        for j, metric in enumerate(metrics):
            records = [v for v in rows if v['source_group'] == source and v['dimension'] == metric]
            if len(records) != 1:
                raise RuntimeError('Missing/duplicate source/metric pair')
            dense[i, j], candidate[i, j] = [float(records[0][k]) for k in ['D_Q', 'candidate']]
    if not np.isfinite(dense).all() or not np.isfinite(candidate).all() or np.any(dense <= 0):
        raise RuntimeError('Finite scores and positive paired baseline required')
    for j, metric in enumerate(metrics):
        for values, name in [(dense, 'D_Q'), (candidate, 'candidate')]:
            if not np.isclose(values[:, j].mean(), comparison['metrics'][metric][name], rtol=1e-6, atol=1e-8):
                raise RuntimeError('Source scores do not reproduce the official aggregate units')
    strata = [[i for i, source in enumerate(sources) if source.startswith(prefix)]
              for prefix in ['i2v_subject:', 'i2v_background:']]
    if [len(v) for v in strata] != [10, 10] or set(sum(strata, [])) != set(range(20)):
        raise RuntimeError('Fixed subject10/background10 strata required')
    seed, repeats = 20261021, 10000
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(v, size=(repeats, 10), replace=True) for v in strata], axis=1)
    boot = 1 - candidate[indices].mean(axis=1) / dense[indices].mean(axis=1)
    per_source = 1 - candidate / dense
    summaries = {}
    for j, metric in enumerate(metrics):
        summaries[metric] = {
            'official_point_relative_drop': comparison['metrics'][metric]['relative_drop'],
            'paired_stratified_percentile95_interval': np.quantile(boot[:, j], [.025, .975]).tolist(),
            'individual_source_drop_min_median_max': np.quantile(per_source[:, j], [0, .5, 1]).tolist(),
            'sources_with_individual_drop_above2percent': int((per_source[:, j] > .02).sum()),
            'sources_with_improvement': int((per_source[:, j] < 0).sum()),
            'source_count': 20}
    result = {'status': 'COMPLETED_DESCRIPTIVE_PAIRED20_SOURCE_VARIABILITY',
              'comparison_sha256': digest(args.comparison), 'paired_csv_sha256': digest(args.paired_csv),
              'source_sha256': digest(Path(__file__)), 'numpy': np.__version__,
              'bootstrap': {'seed': seed, 'repeats': repeats,
                            'strata': 'resample10 subject and10 background paired sources independently with replacement; keep all3 metrics together',
                            'joint_fraction_with_each_drop_le2percent': float(np.all(boot <= .02, axis=1).mean())},
              'metrics': summaries, 'registered_point_estimate_quality_pass': comparison['formal20_quality_pass'],
              'interpretation': 'Descriptive benchmark resampling only; not a p-value, population success probability, untouched-test evidence or a replacement acceptance gate',
              'changes_registered_acceptance': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
