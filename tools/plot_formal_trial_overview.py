"""Compare closed formal trials on the same frozen D-Q benchmark and metric units."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparisons', type=Path, nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if len(args.comparisons) != len(args.labels): ap.error('One label per comparison required')
    records = [json.loads(p.read_text()) for p in args.comparisons]
    names = ['aesthetic_quality', 'imaging_quality', 'motion_smoothness']
    baseline = records[0]
    for record in records:
        if (record['status'] != 'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON'
                or record['unique_source_count'] != 20
                or record['dense_protocol_sha256'] != baseline['dense_protocol_sha256']
                or any(record['metrics'][k]['D_Q'] != baseline['metrics'][k]['D_Q']
                    or record['metrics'][k]['max_relative_drop'] != .02 for k in names)):
            raise RuntimeError('Trials must share the same complete20-source D-Q evaluation baseline')
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.3), layout='constrained')
    colors = ['#536c8b', '#cb8243', '#508b78', '#976893']
    for ax, metric, title in zip(axes, names, ['Aesthetic quality', 'Imaging quality', 'Motion smoothness']):
        values = [100 * r['metrics'][metric]['relative_drop'] for r in records]
        bars = ax.bar(range(len(records)), values, color=[colors[i % len(colors)] for i in range(len(records))], width=.6)
        ax.axhline(2, color='#b13c3c', linewidth=1.2, linestyle='--')
        ax.set_xticks(range(len(records)), args.labels); ax.set_title(title)
        ax.set_ylim(min(0., min(values) * 1.25), max(2.8, max(values) * 1.2))
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.3f}%', (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
        ax.set_ylabel('Relative drop from D-Q (%)')
    fig.suptitle('Frozen formal20 trials: each metric must stay at or below the 2% dashed line\n'
        '20 paired independent sources; reused benchmark, seed42; camera excluded', fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=180); fig.savefig(args.output.with_suffix('.pdf'))
    args.output.with_suffix('.json').write_text(json.dumps({'source_sha256': sha(Path(__file__)),
        'inputs': [{'label': label, 'path': str(path.resolve()), 'sha256': sha(path),
            'policy_sha256': record['candidate_policy_sha256']} for label, path, record in zip(args.labels, args.comparisons, records)],
        'dense_protocol_sha256': baseline['dense_protocol_sha256'], 'matplotlib': matplotlib.__version__,
        'quality_or_speed_acceptance': False}, indent=2) + '\n')


if __name__ == '__main__': main()
