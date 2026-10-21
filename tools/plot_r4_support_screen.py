"""Render the completed R4 native finite-screen comparisons in their own baselines."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report['status'] != 'COMPLETED_R4E1_FINITE_NATIVE_SUPPORT_GATE_REJECTED' or len(report['CV']) != 4:
        raise RuntimeError('Completed four-fold R4E1 result required')
    rows = report['CV']
    gain = np.array([[100 * (1 - v['native_risks']['K1_frozen'] / v['native_risks']['K0_frozen']),
                      100 * (1 - v['native_risks']['K1_matched'] / v['native_risks']['K0_matched'])] for v in rows])
    fig, ax = plt.subplots(figsize=(9.1, 4.8))
    x, width = np.arange(4), .34
    for j, (label, color) in enumerate([('Original frozen R256', '#4c78a8'), ('Matched fixed-h readout', '#72b7b2')]):
        bars = ax.bar(x + (j - .5) * width, gain[:, j], width=width, label=label, color=color)
        ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.axhline(0, color='#444444', linewidth=.8)
    ax.axhline(2, color='#c33b31', linestyle='--', linewidth=1.1, label='2% matched-vs-K0 screen floor')
    ax.set_xticks(x, [f'Step{v["step"]} / fold{v["fold"]}' for v in rows])
    ax.set_ylabel('K1 native sampled-row SSE improvement over K0 (%)')
    ax.set_ylim(-6.8, 3.1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Finite support swaps fail after matched-compensation controls', fontsize=12)
    ax.legend(loc='lower left', frameon=False, fontsize=8)
    fig.text(.5, .01, 'Each bar uses its corresponding K0 readout. Known12/12 source folds,256 saved rows; no video-quality acceptance.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    outputs = {}
    for suffix in ['png', 'pdf']:
        path = args.output_prefix.with_suffix('.' + suffix)
        fig.savefig(path, dpi=220, bbox_inches='tight')
        outputs[suffix] = {'path': str(path), 'sha256': sha(path)}
    plt.close(fig)
    args.output_prefix.with_suffix('.json').write_text(json.dumps({'status': 'RENDERED_COMPLETE_R4E1_NATIVE_FINITE_SCREEN',
        'input_sha256': sha(args.report), 'source_sha256': sha(Path(__file__)),
        'native_improvement_percent': gain.tolist(), 'outputs': outputs,
        'quality_or_speed_acceptance': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
