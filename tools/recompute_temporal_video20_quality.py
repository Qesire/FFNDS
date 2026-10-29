"""Rebuild actual forty-new/twenty-reused official scalar evidence."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_temporal_video20_quality as q


def recompute(protocol, hf):
    plan = q.base.read(protocol); q.validate(plan, True)
    out = hf/plan['output_HF_relative']; rb = q.base.binding(out/'RUN.json', hf)
    eb = q.base.binding(out.parent/'EXIT_RECEIPT.json', hf); pb = q.base.binding(protocol, ROOT)
    run = q.base.read_bound(hf, rb); receipt = q.base.read_bound(hf, eb)
    q.base.exited(receipt, rb, pb['sha256'], q.ID)
    assert run['status'] == 'COMPLETE_TEMPORAL40_EIGHT_AXES_REUSED_DENSE20_QUALITY_NOT_ACCEPTED'
    assert run['predebited'] == run['invoked'] == run['completed'] == q.LIMITS
    assert not run['assigned_GPU_resources']['violations']
    summary = q.summarize(plan, hf, run['completed_dimensions'], run['RAFT_observations'])
    assert summary == run['source_summary']
    strata = {}
    for group in ('subject', 'background', 'camera'):
        rows = [r for r in summary['per_source'] if r['stratum'] == group]
        strata[group] = {'image_clusters': len(rows), 'continuous_RAFT_source_equal_means': {
            mode: statistics.fmean(s['continuous_RAFT'][mode]['mean_top5percent_flow_px_per_2frame_pair'] for s in rows) for mode in q.MODES}}
    return {'status': run['status'], 'protocol': pb, 'RUN': rb, 'exit': eb, 'actual_exit': 0,
        'actual_outer_seconds': receipt['outer_seconds'], 'actual_counts': run['completed'],
        'actual_official_module_functions': run['official_module_functions'],
        'model_work_accounting': run['model_work_accounting'], 'new_scored_videos': 40,
        'reused_Dense_scored_videos': 20, 'image_clusters': 20, 'samples_per_arm_per_source': 1,
        'new_metric_axis_results': 280, 'reused_Dense_metric_axis_results': 140,
        'source_summary': summary, 'strata': strata, 'official_five_independent_samples': False,
        'temporal_flickering': run['temporal_flickering'], 'camera_motion': run['camera_motion'],
        'human_fullplay': run['human_fullplay_review'], 'quality_accepted': False,
        'fair_warmed_SYS': 'NOT_MEASURED', 'recompute_new_metric_tensor_model_GPU_calls': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface')); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); result = recompute(a.protocol.resolve(), a.hf_root.resolve())
    with a.out.open('x') as stream: json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'RAFT': result['source_summary']['source_equal_continuous_RAFT_means'],
                      'same_budget': result['source_summary']['same_budget_dcac10_vs_value10']}))


if __name__ == '__main__': main()
