"""Export the completed paired videos' scalar receipts; never executes a model."""
import argparse
import json
import os
from pathlib import Path
import shutil

from archive_public_json import archive
from rollout_contract import CAPS, METHOD, ROOT, read, sha


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from worker_identity import require_worker
    require_worker()
    hf = Path(os.environ['HF_HOME'])
    root = hf / 'derived/ffnds/R15V0_paired_rollout'
    protocol_path = ROOT / 'research/20261023/R15V0_registered_protocol.json'
    protocol = read(protocol_path)
    run, budget, review = [read(root / name) for name in
                          ('RUN.json', 'BUDGET.json', 'VIDEO_INSPECTION.json')]
    if (run['id'] != METHOD
            or run['status'] != 'COMPLETE_PAIRED_FULL_VIDEO_AWAITING_DECODE_AND_REVIEW'
            or run['protocol_sha256'] != sha(protocol_path)
            or run['budget'] != budget or budget['used'] != CAPS
            or any(d['status'] != 'COMPLETED' for d in budget['debits'])
            or [d['name'] for d in budget['debits']] != ['factors', *protocol['arms']]
            or [r['arm'] for r in run['requests']] != protocol['arms']
            or run['intermediate_latent_reset_to_DQ'] is not False
            or run['matched_initial_noise'] is not True):
        raise RuntimeError('Incomplete or changed paired generation/debit record')
    dense, candidate = run['requests']
    if (dense['initial_noise_sha256'] != candidate['initial_noise_sha256']
            or dense['candidate_events'] or len(candidate['candidate_events']) != 8):
        raise RuntimeError('Paired noise or actual candidate coverage differs')
    for request in run['requests']:
        c = request['counter_delta']
        if ([s['step'] for s in request['trajectory']] != list(range(4))
                or c['linear_forward_calls'] != 1600 or c['activation_qdq_calls'] != 1600
                or c['cache_misses'] or c['weight_transfers']
                or sha(hf / request['video_HF_relative']) != request['video_sha256']
                or request['peak_allocated_bytes'] > protocol['resources']['peak_allocated_bytes_max']):
            raise RuntimeError('Native trajectory, coverage, memory or video changed')
    if (review['raw_run_sha256'] != sha(root / 'RUN.json')
            or review['decoded_every_frame'] is not True or review['paired_frames'] != 81
            or review['automatic_visual_PASS'] is not False
            or sha(root / 'factors.safetensors') != run['factorization']['file_sha256']):
        raise RuntimeError('Decode evidence or frozen factors changed')
    for item in [*review['all_frame_contact_pages'], review['paired_video']]:
        if sha(hf / item['HF_relative']) != item['sha256']:
            raise RuntimeError('Review media changed')
    for name, expected in protocol['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Executed source changed before receipt export')
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if used > protocol['resources']['new_output_bytes_max']:
        raise RuntimeError('Combined generation/review output exceeds cap')
    base = ROOT / 'research/20261023'
    for source, target in [('RUN.json', 'R15V0_native_rollout_results.json'),
                           ('BUDGET.json', 'R15V0_budget_debit.json'),
                           ('VIDEO_INSPECTION.json', 'R15V0_video_inspection.json')]:
        archive(root / source, base / target)
    resources = {'status': 'COMPLETE_PAIRED_VIDEO_AND_CPU_DECODE',
                 'method_id': METHOD, 'actual_full_generations': 2,
                 'actual_offline_weight_factorizations': 2, 'training_steps': 0,
                 'actual_candidate_FFN_positions': 4, 'actual_candidate_linear_calls': 20,
                 'max_GPUs': 1, 'decoded_frames_per_video': 81,
                 'incremental_HF_output_bytes': used,
                 'free_HF_output_bytes_after': shutil.disk_usage(root).free,
                 'visual_verdict': 'PENDING_FULL_DURATION_REVIEW',
                 'population_quality_or_native_speed_acceptance': False}
    with (base / 'R15V0_completed_resources.json').open('x') as stream:
        json.dump(resources, stream, indent=2)
        stream.write('\n')
    print('Two full videos and every-frame decode exported; visual verdict remains separate.', flush=True)


if __name__ == '__main__':
    main()
