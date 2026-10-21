"""Render saved scientific evidence with explicit, separate denominators."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    base = args.root.resolve() / 'research/20261021'
    synthesis = json.loads((base / 'failure_synthesis_R10.json').read_text())
    gain = json.loads((base / 'R7D2_results.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    blue, orange, green = '#34699a', '#d57936', '#328278'

    ax = axes[0, 0]
    x = np.arange(2)
    positions = synthesis['hessian_transfer']
    ax.bar(x - .17, [p['training_ratio'] for p in positions], .32, label='Training', color=green)
    ax.bar(x + .17, [p['validation_ratio'] for p in positions], .32, label='Validation', color=orange)
    ax.axhline(1, color='#666', ls='--', lw=1)
    ax.set_xticks(x, ['Step 2', 'Step 3'])
    ax.set_ylabel('Affine SSE / original RTN SSE')
    ax.set_title('R8F2: empirical fit does not transfer')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.18)

    ax = axes[0, 1]
    labels, inside, outside, cross = [], [], [], []
    for p in positions:
        for arm in ['RTN', 'Hessian']:
            v = p['validation_decomposition'][arm]
            labels.append(f'Step {p["step"]}\n{arm}')
            inside.append(v['in_span_SSE'])
            outside.append(v['outside_span_SSE'])
            cross.append(v['cross_term'])
    x = np.arange(4)
    ax.bar(x, inside, label='In training span', color=blue)
    ax.bar(x, outside, bottom=inside, label='Outside span', color=orange)
    ax.bar(x, cross, bottom=np.array(inside) + outside, label='Cross term', color='#aaaaaa')
    ax.set_xticks(x, labels)
    ax.set_title('Validation error: useful fit, costly extrapolation')
    ax.set_ylabel('FP64 affine output SSE')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    scopes = ['train', 'holdout']
    for j, (arm, color) in enumerate([('local_fit', blue), ('flow_fit', orange)]):
        v = [100 * gain['pooled_flow_SSE_reduction_fraction'][scope][arm] for scope in scopes]
        ax.bar(np.arange(2) + (j - .5) * .32, v, .31, color=color, label=arm.replace('_', ' '))
    ax.axhline(0, color='#444', lw=.7)
    ax.set_xticks(np.arange(2), ['Three fit sources', 'One fit-excluded source'])
    ax.set_ylabel('Actual native flow SSE reduction (%)')
    ax.set_title('R7D2: downstream target is not sufficient')
    ax.legend(frameon=False)
    ax.text(.02, .96, 'All sources development-exposed', transform=ax.transAxes, va='top', fontsize=8)

    ax = axes[1, 1]
    g = synthesis['gain_fit']
    v = [g['train_predicted_improvement'], g['train_observed_improvement']]
    bars = ax.bar(['Secant prediction', 'Actual native forward'], v, color=[green, orange], width=.55)
    ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.set_ylim(0, max(v) * 1.2)
    ax.set_ylabel('Training flow SSE reduction (absolute)')
    ax.set_title('R7D2: only 1.806% of predicted gain realized')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.12)
        ax.set_axisbelow(True)
    fig.suptitle('Saved-operand failure analysis: no new fit, encoding, or model generation', fontsize=12)
    out = base / 'failure_synthesis_R10'
    for ext in ['png', 'pdf']:
        fig.savefig(out.with_suffix('.' + ext), dpi=180)
    out.with_suffix('.figure.json').write_text(json.dumps({
        'source_sha256': sha(Path(__file__)),
        'inputs': {n: sha(base / n) for n in ['failure_synthesis_R10.json', 'R7D2_results.json']},
        'matplotlib': matplotlib.__version__, 'quality_or_speed_acceptance': False,
        'new_model_runs': 0, 'independent_quality_cohort': False,
        'outputs': {ext: sha(out.with_suffix('.' + ext)) for ext in ['png', 'pdf']},
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
