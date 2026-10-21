"""Standalone scientific figure for the fixed R7D1 diagnostic, no model work."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    if result['status'] != 'PASS_COMPLETE_NATIVE_FIXED_PAIR_DIAGNOSTIC':
        raise RuntimeError('Complete verified native results required')
    rows = result['rows']
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = ['#167c80', '#e59529', '#5555a7']
    for j, arm in enumerate(['A', 'B', 'AB']):
        values = [100 * r['flow_SSE_reduction'][arm] / r['flow_SSE']['none'] for r in rows]
        axes[0].bar(x + (j - 1) * .24, values, .23, label=arm, color=colors[j])
    axes[0].set_title('Full downstream flow recovery')
    axes[0].set_ylabel('Flow SSE reduction vs frozen UB1 (%)')
    axes[0].legend(frameon=False, ncol=3)
    for j, (key, label, color) in enumerate([
            ('linear_direction_cross_term', 'Direction term', '#377eb8'),
            ('nonlinear_transport_term', 'Nonlinear term', '#e69f00'),
            ('interaction_advantage', 'Total interaction', '#555555')]):
        values = [100 * r[key] / r['flow_SSE']['none'] for r in rows]
        axes[1].bar(x + (j - 1) * .24, values, .23, label=label, color=color)
    axes[1].set_title('Joint gain minus the two separate gains')
    axes[1].set_ylabel('Interaction / baseline flow SSE (%)')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axhline(0, color='#444444', linewidth=.7)
        ax.set_xticks(x, [f'Input {r["index"]:02d}' for r in rows])
        ax.set_xlabel('Four development-exposed TF2 sources')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Native step 3, blocks 30/31: diagnostic dense restoration only', fontsize=11)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(args.out.with_suffix('.' + extension), dpi=180)
    meta = {'input': str(args.input), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'matplotlib': matplotlib.__version__, 'quality_or_speed_acceptance': False}
    args.out.with_suffix('.figure.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
