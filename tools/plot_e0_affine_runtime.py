"""Standalone scientific figures from the completed public E0 summaries."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save(fig, directory, name):
    fig.savefig(directory / f'{name}.png', dpi=190, bbox_inches='tight')
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    plt.close(fig)


def plot(directory):
    directory.mkdir(parents=True, exist_ok=True)
    runtime = json.loads((ROOT / 'research/20261025/E0_DENSE_WEIGHT_RESIDENCY.json').read_text())
    scientific = json.loads((ROOT / 'research/20261025/E0_AFFINE_FFN_DISCOVERY.json').read_text())
    colors = ['#48556a', '#3686ad', '#d17d30', '#318563']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={'width_ratios': [1.6, 1]})
    modes = ['original', 'block_no_flush', 'all_resident_no_flush']
    b_runs = [next(r for r in runtime['runs'] if r['anonymous_card'] == 'B' and r['mode'] == m) for m in modes]
    x = np.arange(3)
    for phase, label, offset, alpha in [(1, 'source0, seed42 (repeat)', -.17, 1),
                                         (2, 'source12, seed54 (warm)', .17, .55)]:
        values = [r['phases'][phase]['wall_seconds_including_video_save'] for r in b_runs]
        bars = axes[0].bar(x + offset, values, width=.31, alpha=alpha, color=[colors[0], colors[1], colors[3]], label=label)
        axes[0].bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3, fontsize=9)
    axes[0].set_xticks(x, ['Original', 'No explicit flush', 'All decoded resident'])
    axes[0].set_ylim(0, 91)
    axes[0].set_ylabel('Complete generation + video save (s)')
    axes[0].set_title('A. Same A100: observed warm latency')
    axes[0].legend(loc='upper right', fontsize=8, frameon=False)
    mode_memory = modes[:2] + ['ffn_resident_no_flush', modes[2]]
    memory = [max(r['sampled_peak_physical_MiB'] for r in runtime['runs'] if r['mode'] == m) / 1024 for m in mode_memory]
    bars = axes[1].bar(np.arange(4), memory, color=colors)
    axes[1].bar_label(bars, labels=[f'{v:.2f}' for v in memory], padding=3)
    axes[1].axhline(80, ls='--', c='#777777', lw=1)
    axes[1].set_ylim(0, 88)
    axes[1].set_xticks(np.arange(4), ['Original', 'No flush', 'FFN\nresident', 'All\nresident'])
    axes[1].set_ylabel('Sampled physical memory (GiB)')
    axes[1].set_title('B. Memory cost (2 s sampling)')
    fig.suptitle('18 complete dense videos: 72 latent hashes + 18 video-file hashes match', y=1.02, fontsize=12)
    fig.text(.5, -.04, 'Two exposed image/seed clusters; one sequence per mode/card. Model setup excluded. FFN MACs unchanged.\nFFN-resident memory from card A; panel A uses only card B. No confidence intervals implied.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, directory, 'dense_residency')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    tangent = scientific['studies']['all_block_tangent']
    increment = scientific['studies']['cross_source_increment']
    for ax, step, title in zip(axes[:2], [1, 3], ['A. High-noise 0→1', 'B. Low-noise 2→3']):
        for study, name, label, color in [
            (tangent, 'old_full_M_mean_derivative', 'Own old full-M slope', colors[1]),
            (increment, 'fixed_linear_only', 'Other-source fixed slope', colors[0]),
            (increment, 'linear_plus_27_source_fixed', 'Fixed base + 27 residual groups', colors[3]),
        ]:
            values = []
            for b in range(40):
                cases = [c for c in study['cases'] if c['step'] == step and c['block'] == b]
                values.append(100 * np.mean([(c['results'][name]['panels']['uniform32'] if study is tangent
                                             else c['results'][name]['uniform32'])['relative_current_output_SSE'] for c in cases]))
            ax.plot(np.arange(40), values, c=color, label=label, lw=1.6)
        ax.axhline(1, c='#888888', ls=':', lw=1)
        ax.set_yscale('log')
        ax.set_ylim(.3, 160)
        ax.set_xlabel('Block')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Local current-output relative SSE (%)')
    axes[0].legend(fontsize=7, loc='upper left', frameon=False)
    absolute = scientific['studies']['absolute_affine']
    blocks = [0, 10, 20, 30, 39]
    for name, label, color in [
        ('Q2_OLS_plus27_source_fixed', 'Affine + 27 groups (67/216 MAC)', colors[3]),
        ('group_only67_source_fixed', 'Group-only 67 (67/216 MAC)', colors[0]),
    ]:
        values = [100 * absolute['by_block'][str(b)]['means'][name]['uniform32']['relative_current_output_SSE'] for b in blocks]
        axes[2].plot(blocks, values, 'o-', c=color, label=label, lw=1.7)
    axes[2].set_title('C. Absolute FFN: exact matched MAC')
    axes[2].set_xlabel('Block (five sampled depths)')
    axes[2].set_ylim(0, 60)
    axes[2].legend(fontsize=7, loc='upper left', frameon=False)
    axes[2].grid(axis='y', alpha=.2)
    fig.suptitle('Function representations retain more output than direct group pruning, but leave substantial error', y=1.02, fontsize=12)
    fig.text(.5, -.035, 'Uniform32 evaluation; arithmetic means across two exposed source/seed cases (A/B), or all four steps (C).\nFull-M means old derivative statistics only. Saved-data diagnostics; no native folded video or quality acceptance.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, directory, 'affine_ffn_discovery')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    plot(parser.parse_args().out_dir)
