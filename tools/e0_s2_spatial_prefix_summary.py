"""Registered scalar-only source-equal summary of the closed V2/V3 lineage."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
import recompute_s2_spatial_prefix_replay as summary
native = summary.native
ID = 'E0_S2_SPATIAL52_FULL160_LPI_SCALAR_SUMMARY_CPU_20261029'
OUTPUT = 'ffnds-derived/e0_s2_spatial52_full160_lpi_summary_cpu_20261029/attempt1/worker/results'
RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 300, 'outer_seconds': 360,
    'new_HF_bytes': 64 << 20, 'max_RSS_bytes': 8 << 30, 'scalar_stream_unique_bytes_max': 512 << 20,
    'automatic_retry': False}
SOURCES = ('tools/e0_s2_spatial_prefix_summary.py', 'tools/recompute_s2_spatial_prefix_replay.py',
    'tests/test_recompute_s2_spatial_prefix_replay.py', 'tools/e0_s2_spatial_prefix_replay_batched_v3.py')


def draft(hf):
    parents = []; total = 0; paths = set(SOURCES)
    for part in (0, 1):
        path = ROOT/f'config/e0_s2_spatial_prefix_replay_batched_part{part}_20261029_v3.json'
        plan = native.read(path); out = hf/plan['output_HF_relative']
        paths |= {b['path'] for b in plan['repo_sources']} | {str(path.relative_to(ROOT))}
        proof = {'protocol': native.binding(path, ROOT), 'RUN': native.binding(out/'RUN.json', hf),
                 'exit': native.binding(out.parent/'EXIT_RECEIPT.json', hf)}
        run = native.read(out/'RUN.json')
        summary.m.core.vm.exited(native.read(out.parent/'EXIT_RECEIPT.json'), proof['RUN'], proof['protocol']['sha256'], plan['id'])
        native.need(run['status'] == 'COMPLETE_V3_MERGED_160_POSITIONS', 'V3 must actually exit complete before freezing scalar scope')
        prior = native.read(hf/plan['parent']['RUN']['path'])
        total += sum(run[k]['bytes'] for k in ('positions', 'controls', 'merged_positions'))+prior['positions']['bytes']
        parents.append(proof)
    native.need(total <= RESOURCES['scalar_stream_unique_bytes_max'], 'Scalar input footprint exceeded')
    return {'id': ID, 'status': 'DRAFT', 'assets_finalized': False, 'automatic_retry': False,
        'parents': parents, 'resources': RESOURCES, 'unique_scalar_stream_bytes': total,
        'output_HF_relative': OUTPUT, 'limits': {'closed_expert_parts': 2, 'unique_merged_records': 640,
            'source_equal_aggregations': 2, 'plots': 1, 'GPU_calls': 0, 'model_calls': 0},
        'repo_sources': [native.binding(ROOT/p, ROOT) for p in sorted(paths)]}


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = result['depth_profile_metric_labels']; rows = result['depth_profile']
    keys = [('teacher_local_error_fraction', 'local L'), ('prefix_error_fraction', 'prefix P'),
            ('interaction_error_fraction', 'interaction I'), ('total_error_fraction', 'total L+P+I')]
    figure, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for row, family in enumerate(result['families']):
        for step in range(4):
            axis = axes[row, step]; selected = sorted((x for x in rows if x['family'] == family and x['step'] == step), key=lambda x:x['block'])
            for key, name in keys:
                column = labels.index('gY/tau1_to20/central25/AC/'+key)
                axis.plot([x['block'] for x in selected], [x['mean'][column] for x in selected], label=name)
            axis.set_title(f'{family} / step {step}'); axis.set_xlabel('block'); axis.grid(alpha=.2)
            if step == 0: axis.set_ylabel('AC error energy / teacher AC energy')
    axes[0, 0].legend(fontsize=8)
    figure.suptitle('Two exposed diagnostic images; matched fixed-state decomposition, central ROI, tau 1-20\nSigned cross terms retained in total; curves are not causal shares or video-quality scores', fontsize=11)
    figure.tight_layout(); figure.savefig(path, dpi=150); plt.close(figure)


def run(args):
    start = time.monotonic(); plan = native.read(args.protocol); digest = native.sha(args.protocol)
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    rows = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    native.need(plan['id'] == ID and plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True
        and plan['resources'] == RESOURCES and plan['automatic_retry'] is False
        and active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True
        and len(rows) == 1 and rows[0]['protocol_sha256'] == digest and rows[0]['max_gpu_cards'] == 0
        and rows[0]['resources'] == RESOURCES and rows[0]['limits'] == plan['limits']
        and rows[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and len(os.sched_getaffinity(0)) <= 4,
        'Exact bounded scalar CPU scope required')
    for item in plan['repo_sources']: native.bound(ROOT, item)
    current = draft(args.hf_root)
    native.need(all(current[k] == plan[k] for k in ('parents', 'unique_scalar_stream_bytes', 'limits', 'repo_sources')), 'Frozen scalar inputs changed')
    out = (args.hf_root/OUTPUT).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined scalar output required')
    out.mkdir(parents=True)
    report = {'id': ID, 'protocol_sha256': digest, 'status': 'STARTED_CPU_SCALAR_RECOMPUTATION',
        'GPU_calls': 0, 'model_calls': 0, 'automatic_retry': False}
    native.write(out/'RUN.json', report)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Scalar CPU wall exceeded'))); signal.alarm(300)
    try:
        result = summary.rebuild(args.hf_root, 'v3')
        native.write(out/'SUMMARY.json', result); plot(result, out/'DEPTH_PROFILE.png')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        native.need(rss <= RESOURCES['max_RSS_bytes'], 'Scalar CPU RSS8GiB exceeded')
        native.need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) <= RESOURCES['new_HF_bytes']//2, 'Scalar output budget exceeded')
        report.update(status='COMPLETE_TWO_SOURCE_FULL160_LPI_SCALAR_SUMMARY', actual_unique_records=640,
            actual_independent_images=2, summary=native.binding(out/'SUMMARY.json', args.hf_root),
            plot=native.binding(out/'DEPTH_PROFILE.png', args.hf_root), max_RSS_bytes=rss,
            inherited_prefix_status='V2_INTENTIONAL_BLOCK_BOUNDARY_EXIT1_NOT_COMPLETE',
            numerical_controls='V3_ALL_THREE_PER_EXPERT_PASSED_RECHECKED', quality_or_speed_accepted=False)
    except BaseException as error:
        report.update(status='FAILED_SCALAR_SUMMARY_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-start; native.write(out/'RUN.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('draft', 'run'))
    p.add_argument('--protocol', type=Path); p.add_argument('--out', type=Path)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface')); args = p.parse_args()
    if args.action == 'draft':
        native.need(args.out is not None and not args.out.exists(), 'Fresh scalar draft required'); native.write(args.out, draft(args.hf_root))
    else: run(args)


if __name__ == '__main__': main()
