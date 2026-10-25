"""Publication-style descriptive plots of completed fixed E0 diagnostics."""
import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    d = json.loads(args.result.read_text())['studies']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.7), layout='constrained')
    ax = axes[0, 0]
    stat = d['statistical_gelu']['forms']['absolute']['overall']['methods']
    names = ['half_slope', 'empirical_mean_derivative', 'gaussian_moment_gain', 'GELU_OLS', 'Q2_OLS']
    labels = ['Half slope', 'Empirical derivative', 'Gaussian moment gain', 'GELU OLS', 'Native-Q2 OLS']
    vals = [100 * stat[n]['uniform32']['relative_target_SSE']['mean'] for n in names]
    ax.barh(labels[::-1], vals[::-1], color=['#147d92'] * 4 + ['#bd604d'])
    for i, val in enumerate(vals[::-1]): ax.text(val + 1, i, f'{val:.1f}%', va='center')
    ax.set(xlim=(0, 100), xlabel='Gated output SSE / target energy (%)',
           title='a  Half-linear GELU is insufficient (320 cases)')
    ax = axes[0, 1]
    basis = d['output_basis']['forms']['innovation']['overall']['methods']
    ranks = [1, 4, 16, 32, 64]
    for name, label, color, style in [
        ('cross_source_basis_true_coeff', 'Other-source basis, true coefficients', '#147d92', '-'),
        ('own_center_basis_true_coeff', 'Own-center basis, true coefficients', '#70a38f', '--'),
        ('input_predictive_RRR', 'Input predictive RRR', '#bd604d', '-'),
        ('random_basis_true_coeff', 'Random basis, true coefficients', '#999999', ':')]:
        ax.plot(ranks, [100 * basis[f'{name}_r{r}']['uniform_disjoint']['relative_target_SSE']['mean']
                        for r in ranks], label=label, color=color, linestyle=style, marker='o', markersize=4)
    ax.set(xlabel='Fixed rank (centered training rank <=127)', ylabel='Innovation SSE / target energy (%)',
           title='b  Remote-token basis and prediction errors (160 cases)')
    ax.legend(fontsize=8, loc='lower left')
    ax = axes[1, 0]
    blocks = [0, 10, 20, 30, 39]
    data = d['nonlinear_reconstruction']['forms']['absolute']['by_block']
    for name, label, color, style in [
        ('calibration_affine', 'Affine only (40/216 MAC)', '#999999', ':'),
        ('whole_g94_ridge', '94 groups + ridge (94/216 MAC)', '#bd604d', '-'),
        ('affine_residual_g54_ridge', 'Affine +54 groups + ridge (94/216 MAC)', '#147d92', '-')]:
        ax.plot(blocks, [100 * data[str(b)]['methods'][name]['uniform_disjoint']['relative_target_SSE']['mean']
                         for b in blocks], label=label, color=color, linestyle=style, marker='o')
    ax.set(xlabel='Block', ylabel='Gated output SSE / target energy (%)',
           title='c  Four calibration sources, two held sources (40 cases)')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    cases = d['native_finite']['numerical_cases']
    for name, label, color in [('whole', '94 native groups + ridge', '#bd604d'),
                               ('affine_residual', 'Affine +54 native groups + ridge', '#147d92')]:
        for block, marker in [(0, 'o'), (30, '^')]:
            subset = [c for c in cases if c['block'] == block]
            ax.scatter([median(c['methods'][name]['full_M_ms']) for c in subset],
                       [100 * c['methods'][name]['native_vs_saved']['uniform_disjoint']['relative_target_SSE'] for c in subset],
                       color=color, marker=marker, s=36, alpha=.8, label=f'{label}; block{block}')
    dense_ms = np.mean([median(c['dense_ms']) for c in cases])
    ax.scatter([dense_ms], [0], color='black', marker='s', label='Resident original dense')
    ax.set(xlabel='Standalone after-Q1 FFN median time (ms)', ylabel='Gated output SSE / target energy (%)',
           title='d  Native BF16/HiF4, repeated rows at M=31668')
    ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('Function representation, nonlinear information and real local computation', fontsize=14)
    fig.supxlabel('Case means are descriptive; six exposed image/seed clusters. No approximate free-running video or end-to-end timing.', fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'), dpi=180)
    fig.savefig(args.out.with_suffix('.pdf'))
    plt.close(fig)


if __name__ == '__main__': main()
