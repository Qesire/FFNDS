"""Render BO1's completed retrospective imaging bound, never a deployed router."""
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
    if (report['status'] != 'COMPLETED_BO1_JOINT_ORACLE_INFEASIBLE'
            or len(report['cohorts']) != 2
            or any(v['assignments_enumerated'] != 1 << 20 for v in report['cohorts'])):
        raise RuntimeError('Both complete20-source oracle enumerations required')
    values = np.array([[v['UB1_relative_drops']['imaging_quality'],
        v['UB2_relative_drops']['imaging_quality'],
        v['best_unconstrained_imaging_relative_drop']] for v in report['cohorts']]) * 100
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    width, x = .23, np.arange(2)
    for j, (label, color) in enumerate([('Frozen UB1', '#4c78a8'), ('Frozen UB2', '#8c8c8c'),
            ('Best possible imaging choice (oracle)', '#72b7b2')]):
        bars = ax.bar(x + (j - 1) * width, values[:, j], width=width, label=label,
            color=color, edgecolor='#333333', linewidth=.6, hatch='//' if j == 2 else None)
        ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.axhline(2, color='#c33b31', linestyle='--', linewidth=1.2, label='2% maximum drop')
    ax.set_xticks(x, ['Original20 (development)', 'Supplemental20 (development)'])
    ax.set_ylabel('Imaging quality relative drop (%)')
    ax.set_ylim(0, 5.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Even outcome-aware UB1/UB2 selection misses the supplemental gate', fontsize=12)
    ax.legend(fontsize=8, loc='upper left', frameon=False)
    fig.text(.5, .015, '20 independent sources per cohort. Oracle uses outcomes; no trained router or new official evaluation.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ['png', 'pdf']:
        path = args.output_prefix.with_suffix('.' + suffix)
        fig.savefig(path, dpi=220, bbox_inches='tight')
        outputs[suffix] = {'path': str(path), 'sha256': sha(path)}
    plt.close(fig)
    args.output_prefix.with_suffix('.json').write_text(json.dumps({
        'status': 'RENDERED_COMPLETE_BO1_RETROSPECTIVE_IMAGING_BOUND',
        'input_sha256': sha(args.report), 'source_sha256': sha(Path(__file__)),
        'matplotlib': matplotlib.__version__, 'values_percent': values.tolist(),
        'outputs': outputs, 'quality_or_speed_acceptance': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
