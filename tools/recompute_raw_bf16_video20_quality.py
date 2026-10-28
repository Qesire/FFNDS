"""Replay completed official video20 summaries without model or metric calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_video20_quality as frozen


def binding(path, root):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def recompute(protocol, hf):
    plan = json.loads(protocol.read_text())
    frozen.validate(plan, frozen=True)
    folder = hf/plan['output_HF_relative']
    run_binding, exit_binding = binding(folder/'RUN.json', hf), binding(folder.parent/'EXIT_RECEIPT.json', hf)
    report = json.loads((folder/'RUN.json').read_text())
    exited = json.loads((folder.parent/'EXIT_RECEIPT.json').read_text())
    frozen.base.exited(exited, run_binding, binding(protocol, ROOT)['sha256'], plan['id'])
    assert report['status'] == 'COMPLETE_RAW_BF16_VIDEO20_60_CLIPS_EIGHT_DIMENSIONS_QUALITY_NOT_ACCEPTED'
    assert report['protocol_sha256'] == binding(protocol, ROOT)['sha256']
    assert report['predebited'] == report['invoked'] == report['completed'] == frozen.LIMITS
    assert not report['assigned_GPU_resources']['violations']
    summary = frozen.summarize(plan, hf, report['completed_dimensions'], report['RAFT_observations'])
    assert summary == report['source_summary'] and summary['quality_accepted'] is False
    assert len(report['completed_dimensions']) == 8 and len(report['RAFT_observations']) == 60
    strata = {}
    for stratum in ('subject', 'background', 'camera'):
        rows = [s for s in summary['per_source'] if s['stratum'] == stratum]
        strata[stratum] = {'image_clusters': len(rows), 'continuous_RAFT_source_equal_means': {
            mode: sum(s['continuous_RAFT'][mode]['mean_top5percent_flow_px_per_2frame_pair'] for s in rows)/len(rows)
            for mode in ('dense', 'shared', 'step')}}
    return {'status': report['status'], 'protocol': binding(protocol, ROOT), 'RUN': run_binding, 'exit': exit_binding,
            'actual_child_returncode': exited['child_returncode'], 'actual_outer_seconds': exited['outer_seconds'],
            'numeric_path': report['numeric_path'], 'operation_counts': report['completed'],
            'model_work_accounting': report['model_work_accounting'],
            'official_module_functions': report['official_module_functions'],
            'image_clusters': 20, 'genuine_videos': 60, 'samples_per_source_per_arm': 1,
            'four_native_parents': report['producers'], 'prefix_CPU_CORE_repair': report['prefix_CPU_CORE_repair'],
            'source_summary': summary, 'strata': strata,
            'peak_GPU_reserved_bytes': report['peak_GPU_reserved_bytes'],
            'official_five_independent_samples': False, 'temporal_flickering': report['temporal_flickering'],
            'semantic_action_or_camera_acceptance': False, 'human_fullplay_review': report['human_fullplay_review'],
            'quality_accepted': False, 'replay_new_model_metric_tensor_GPU_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = recompute(args.protocol.resolve(), args.hf_root.resolve())
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'genuine_videos': result['genuine_videos'],
                      'continuous_RAFT': result['source_summary']['source_equal_continuous_RAFT_means']}))


if __name__ == '__main__':
    main()
