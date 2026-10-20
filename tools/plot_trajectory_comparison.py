"""Compare complete paired trajectories without equating latent error with quality."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ar2_checkpoint import atomic_json, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reports', type=Path, nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if len(args.reports) != len(args.labels) or len(set(args.labels)) != len(args.labels):
        ap.error('Distinct label required for every report')
    data = [json.loads(p.read_text()) for p in args.reports]
    for d in data:
        if (d['status'] != 'COMPLETED_PAIRED_FORMAL20_LATENT_DRIFT_DESCRIPTION'
                or d['dense_suite_sha256'] != data[0]['dense_suite_sha256']
                or [v['step'] for v in d['summary']] != list(range(4))
                or any(v['input_count'] != 20 for v in d['summary'])):
            raise RuntimeError('Complete paired20 trajectories against the same D-Q required')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    colors = ['#68788c', '#d07531', '#286c9c', '#328063', '#a34f6c']
    for i, (d, label) in enumerate(zip(data, args.labels)):
        for ax, key, multiplier in zip(axes,
                ['pooled_relative_RMSE', 'norm_ratio_mean', 'centered_variance_ratio_mean'], [100, 1, 1]):
            ax.plot(range(4), [multiplier * v[key] for v in d['summary']], '-o',
                    label=label, color=colors[i % len(colors)], lw=1.8, ms=4)
    for ax, ylabel in zip(axes, ['Pooled relative latent RMSE (%)', 'Mean candidate / D-Q norm',
                                'Mean centered energy ratio']):
        ax.set_xticks(range(4), ['0 / 1000', '1 / 750', '2 / 500', '3 / 250'])
        ax.set_xlabel('Native step / configured schedule index')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[1:]:
        ax.axhline(1, color='#555555', lw=.8, ls='--')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Paired formal20 trajectories against the same D-Q', fontsize=12)
    fig.text(.5, -.04, 'Descriptive state differences; lower latent RMSE does not establish better video quality.',
             ha='center', fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ['.png', '.pdf']:
        p = args.output.with_suffix(suffix)
        fig.savefig(p, dpi=180, bbox_inches='tight')
        files.append({'path': str(p.resolve()), 'sha256': digest(p)})
    atomic_json(args.output.with_suffix('.json'), {
        'reports': [{'label': label, 'path': str(p.resolve()), 'sha256': digest(p)}
                    for label, p in zip(args.labels, args.reports)],
        'source_sha256': digest(Path(__file__)), 'files': files, 'quality_or_speed_acceptance': False})


if __name__ == '__main__':
    main()
