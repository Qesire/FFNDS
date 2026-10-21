"""Plot two separate quality cohorts for exactly the same frozen UB1 policy."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--original', type=Path, required=True)
    ap.add_argument('--supplement', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    paths = [args.original, args.supplement]
    records = [json.loads(p.read_text()) for p in paths]
    metrics = ['aesthetic_quality', 'imaging_quality', 'motion_smoothness']
    if (records[0]['candidate_policy_sha256'] != records[1]['candidate_policy_sha256']
            or records[0]['candidate_runtime_sources'] != records[1]['candidate_runtime_sources']
            or records[0]['dense_protocol_sha256'] == records[1]['dense_protocol_sha256']):
        raise RuntimeError('Require the same frozen implementation and separately evaluated cohorts')
    for record in records:
        if (record['status'] != 'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON'
                or record['unique_source_count'] != 20
                or any(record['metrics'][k]['max_relative_drop'] != .02 for k in metrics)):
            raise RuntimeError('Require two complete20-source three-metric comparisons')
    fig, axes = plt.subplots(1, 3, figsize=(10.7, 4.5), layout='constrained')
    labels = ['Original20\nreused', 'Supplement20\nsource-disjoint']
    for ax, key, title in zip(axes, metrics, ['Aesthetic quality', 'Imaging quality', 'Motion smoothness']):
        values = [100 * row['metrics'][key]['relative_drop'] for row in records]
        bars = ax.bar([0, 1], values, color=['#4e718c', '#b77749'], width=.58)
        ax.axhline(2, color='#b63d3a', linestyle='--', linewidth=1.1)
        ax.axhline(0, color='#555555', linewidth=.6)
        ax.set_xticks([0, 1], labels)
        ax.set_ylim(-.45, 3.85)
        ax.set_title(title)
        ax.set_ylabel('Relative drop from own paired D-Q (%)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.3f}%', (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 5 if value >= 0 else -14), textcoords='offset points', ha='center')
    fig.suptitle('Same frozen UB1: original20 pass; source-disjoint supplement20 fails imaging\n'
                 'Separate paired D-Q references; no pooling; each metric must remain ≤2%', fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.pdf']:
        fig.savefig(args.output.with_suffix(suffix), dpi=180)
    args.output.with_suffix('.json').write_text(json.dumps({
        'source_sha256': sha(Path(__file__)), 'policy_sha256': records[0]['candidate_policy_sha256'],
        'cohorts': [{'label': label, 'path': str(p.resolve()), 'sha256': sha(p),
                     'dense_protocol_sha256': r['dense_protocol_sha256']} for p, r, label in zip(paths, records, labels)],
        'scope': 'separate cohort results; supplementary sources now observed; no pooled acceptance or robustness claim',
        'matplotlib': matplotlib.__version__, 'quality_or_speed_acceptance': False}, indent=2)+'\n')


if __name__ == '__main__':
    main()
