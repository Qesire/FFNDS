"""Static research figure from the completed R11T1 public result only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--result', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    d = json.loads(args.result.read_text())
    if d['status'] != 'COMPLETED_FIXED20_MATCHED_NATIVE_OBJECTIVE':
        raise RuntimeError('Complete fixed comparison required')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    colors = {'local': '#db8c35', 'flow': '#287ca4'}
    labels = {'local': 'Local FFN objective', 'flow': 'Native flow objective'}
    x = np.arange(4)
    for offset, arm in [(-.18, 'local'), (.18, 'flow')]:
        values = [r['arms'][arm]['flow_SSE_over_UB1'] for r in d['evaluation']]
        bars = axes[0].bar(x + offset, values, .34, label=labels[arm], color=colors[arm])
        for bar, value in zip(bars, values):
            axes[0].annotate(f'{value:.3f}', (bar.get_x() + bar.get_width() / 2, value),
                             xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
        fit = d['fits'][arm]
        curve = [r['train_objective_before_update'] for r in fit['iterations']] + [fit['final_train_objective']]
        axes[1].plot(range(21), np.array(curve) / curve[0], color=colors[arm], label=labels[arm])
    axes[0].axhline(1, color='#555555', linestyle='--', linewidth=1, label='Frozen UB1')
    axes[0].set_xticks(x, ['00 train', '05 train', '10 train', '15 dev'])
    axes[0].set_ylabel('Complete flow SSE / frozen UB1')
    axes[0].set_title('Actual complete native deployment checked')
    axes[0].legend(fontsize=8, loc='lower left')
    axes[1].set_xlabel('Fixed normalized updates')
    axes[1].set_ylabel('Own training objective / its initial value')
    axes[1].set_title('Same parameters, sources and 20-step budget')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('R11T1: all sources development-exposed; no fresh quality or free rollout', fontsize=11)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, metadata={'Software': 'FFNDS matplotlib research plot'})
    plt.close(fig)


if __name__ == '__main__':
    main()
