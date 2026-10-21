"""Render the complete OP2D state description; never plot partial source cohorts."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from ar2_checkpoint import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output basename without suffix')
    parser.add_argument('--wait-and-render', action='store_true', help='Wait locally, then use the worker CPU plotting environment')
    args = parser.parse_args()
    if args.wait_and_render:
        from run_on_worker import ROOT, SSH
        controller = ROOT / 'runs/worker_restart_20261020/op2d_figure_controller'
        controller.mkdir(parents=True, exist_ok=True)
        import fcntl
        lock = (controller / 'JOB.lock').open('w')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        meta = {'status': 'WAITING_COMPLETE_NATIVE_STATE_DESCRIPTION', 'source_sha256': digest(Path(__file__)),
                'quality_or_speed_acceptance': False}
        atomic_json(controller / 'RUN.json', meta)
        try:
            while True:
                parent = ROOT / 'runs/worker_restart_20261020/op2d_distribution_controller/RUN.json'
                if parent.exists():
                    state = json.loads(parent.read_text())['status']
                    if state.startswith('FAILED'):
                        raise RuntimeError('Native state description failed; figure is not rendered')
                    if state == 'COMPLETED_OP2D_CPU_DESCRIPTION_ONLY':
                        break
                time.sleep(30)
            command = ['bash', str(ROOT / 'scripts/worker_cpu_exec.sh'),
                str(ROOT / '.runtime/envs/vbench_official/bin/python'), str(Path(__file__)),
                '--input', str(args.input), '--output', str(args.output)]
            with (controller / 'remote.log').open('wb') as handle:
                subprocess.run(SSH + [shlex.join(command)], stdout=handle, stderr=subprocess.STDOUT, check=True)
            meta.update(status='RENDERED_COMPLETE_NATIVE_STATE_DESCRIPTION', numeric_evidence_sha256=digest(args.input))
        except BaseException as error:
            meta.update(status='FAILED_PRESERVED', error=str(error))
            raise
        finally:
            atomic_json(controller / 'RUN.json', meta)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    report = json.loads(args.input.read_text())
    if (report['status'] != 'COMPLETED_OP2D_NATIVE_STATE_DISTRIBUTION_DESCRIPTION'
            or len(report['prefix_controls']) != 24 or len(report['positions']) != 4
            or not all(c['V1_bitwise'] and c['TeacherY_bitwise'] and c['gate_and_raw_bias_bitwise']
                       for c in report['prefix_controls'])):
        raise RuntimeError('Only a completed description with all24 prefix controls may be rendered')
    positions = sorted(report['positions'], key=lambda p: p['step'])
    if [p['step'] for p in positions] != list(range(4)) or any(
            p['block'] != 20 or p['source_count'] != 24 or {c['index'] for c in p['cases']} != set(range(24))
            for p in positions):
        raise RuntimeError('Plot requires every registered source and position')
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharex=True)
    for axis, tensor, title in zip(axes, ['V1', 'TeacherY'], ['Actual post-Q1 input V1', 'Native full teacher output Y']):
        for index in range(24):
            values = [next(c for c in p['cases'] if c['index'] == index)[tensor]['relative_RMSE'] for p in positions]
            axis.plot(range(4), values, color='#aab5c2', alpha=.38, linewidth=.8)
        mean = [p['summary'][tensor]['equal_source_mean_relative_RMSE'] for p in positions]
        axis.plot(range(4), mean, 'o-', color='#2068ab', linewidth=2.5, label='Equal-source mean')
        axis.set(title=title, xlabel='Denoising step (block 20)', xticks=range(4), ylim=(0, None))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.18)
        axis.legend(frameon=False, loc='upper left')
    axes[0].set_ylabel('Relative RMSE versus the recorded C24 state')
    figure.suptitle('Actual UB1 versus earlier C24 native states', fontsize=14)
    figure.text(.5, .02, '24 known calibration sources; gray lines show every source. First-FFN prefix: 24/24 exact.\n'
                'Different internal states; this is neither a same-input policy comparison nor video quality.',
                ha='center', fontsize=9, color='#444444')
    figure.tight_layout(rect=(0, .11, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for extension in ['png', 'pdf']:
        path = args.output.with_suffix('.' + extension)
        if path.exists():
            raise RuntimeError('Refusing to overwrite an archived figure')
        figure.savefig(path, dpi=240, bbox_inches='tight')
        files.append({'path': str(path), 'sha256': digest(path)})
    plt.close(figure)
    atomic_json(args.output.with_suffix('.json'), {'status': 'RENDERED_COMPLETE_OP2D_STATE_DESCRIPTION',
        'input_sha256': digest(args.input), 'source_sha256': digest(Path(__file__)), 'files': files,
        'statistical_transform': 'display registered per-source values and equal-source means only',
        'quality_or_speed_acceptance': False})


if __name__ == '__main__':
    main()
