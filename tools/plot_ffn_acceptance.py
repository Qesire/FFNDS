"""Plot all measured complete-FFN repeats and per-step costs from accepted records."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ar2_checkpoint import atomic_json, digest
from compare_ffn_benchmark import load_run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparison', type=Path, required=True)
    ap.add_argument('--dense', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    comparison = json.loads(args.comparison.read_text())
    if (comparison['status'] != 'COMPLETED_FULL160_FFN_PAIRED_THREE_REPEAT_BENCHMARK'
            or digest(args.dense) != comparison['dense_run_sha256']
            or digest(args.candidate) != comparison['candidate_run_sha256']):
        raise RuntimeError('Same complete paired benchmark records required')
    dense, dt, _ = load_run(args.dense, False)
    candidate, ct, _ = load_run(args.candidate, True)
    if dt != comparison['dense']['repeats_ms'] or ct != comparison['candidate']['repeats_ms']:
        raise RuntimeError('Measured repeats differ from primary comparison')
    per_step = {}
    for label, meta, times in [('D-Q', dense, dt), ('UB1', candidate, ct)]:
        rows = []
        for entry in meta['requests'][1:]:
            request = json.loads(Path(entry['report']).read_text())
            timing = request['FFN_timing']
            if any(v['step'] not in range(4) for v in timing['rows']):
                raise RuntimeError('Unknown timing step')
            values = [sum(v['wall_ms'] for v in timing['rows'] if v['step'] == s) for s in range(4)]
            if abs(sum(values) - timing['wall_ms']) > 1e-6:
                raise RuntimeError('Per-step breakdown must include every charged operation')
            rows.append(values)
        per_step[label] = np.mean(rows, axis=0).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    colors = ['#657a91', '#287f75']
    for i, (values, color) in enumerate(zip([dt, ct], colors)):
        axes[0].bar(i, statistics.mean(values), yerr=statistics.stdev(values), capsize=5, color=color, width=.5)
        axes[0].scatter(np.linspace(i - .1, i + .1, 3), values, color='#202a32', s=18, zorder=3)
    axes[0].set_xticks([0, 1], ['D-Q', 'UB1'])
    axes[0].set_ylabel('Complete FFN wall time (ms)')
    axes[0].set_title('All measured repeats (mean and sample SD)')
    x = np.arange(4)
    for offset, (label, color) in zip([-.18, .18], zip(['D-Q', 'UB1'], colors)):
        axes[1].bar(x + offset, per_step[label], width=.35, color=color, label=label)
    axes[1].set_xticks(x, ['0 / 25%', '1 / 50%', '2 / 50%', '3 / 75%'])
    axes[1].set_xlabel('Native step / UB1 main GEMM density')
    axes[1].set_ylabel('Mean charged complete FFN time (ms)')
    axes[1].set_title('All40 blocks and preparation/prefetch costs per step')
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    speed = sum(dt) / sum(ct)
    verdict = 'PASS' if comparison['full_FFN_speed_pass'] else 'FAIL'
    fig.suptitle(f'Complete FFN ratio of sums: {speed:.4f}x ({verdict}; threshold1.5x)\n'
                 'Physical worker GPU0; one warmup + three measured repeats per arm; all160 positions', fontsize=11)
    fig.text(.5, -.04, 'Includes main/auxiliary GEMMs, QDQ, normalization, gain, gate/residual and conservative preparation/prefetch. '
             'Not an end-to-end model speed.', ha='center', fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ['.png', '.pdf']:
        p = args.output.with_suffix(suffix)
        fig.savefig(p, dpi=180, bbox_inches='tight')
        files.append({'path': str(p.resolve()), 'sha256': digest(p)})
    atomic_json(args.output.with_suffix('.json'), {
        'comparison_sha256': digest(args.comparison), 'source_sha256': digest(Path(__file__)),
        'full_FFN_wall_speed': speed, 'per_step_mean_ms_including_all_charges': per_step,
        'files': files, 'end_to_end_model_speed_claim': False})


if __name__ == '__main__':
    main()
