"""Export full160 actual-generation records, without running any model."""
import argparse
import json
import os
from pathlib import Path
import shutil

from archive_public_json import archive
from row_dispatch_contract import ARMS, CAPS, METHOD, ROOT, atomic, read, sha


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from worker_identity import require_worker
    require_worker()
    hf = Path(os.environ['HF_HOME'])
    root = hf / 'derived/ffnds/R16V0_full160_row_dispatch'
    protocol_path = ROOT / 'research/20261023/R16V0_registered_protocol.json'
    protocol = read(protocol_path)
    run, budget, review = [read(root / name) for name in ('RUN.json', 'BUDGET.json', 'VIDEO_REVIEW.json')]
    if (run['id'] != METHOD or
            run['status'] != 'COMPLETE_FULL160_GENERATION_AND_NATIVE_MICROBENCHMARK_AWAITING_VIDEO_REVIEW' or
            run['protocol_sha256'] != sha(protocol_path) or run['budget'] != budget or budget['used'] != CAPS or
            len(budget['debits']) != 169 or any(d['status'] != 'COMPLETED' for d in budget['debits']) or
            [r['arm'] for r in run['requests']] != ARMS or len(run['native_microbenchmarks']) != 4 or
            run['intermediate_latent_reset_to_DQ'] is not False):
        raise RuntimeError('Incomplete fixed execution or budget')
    if (len({r['initial_noise_sha256'] for r in run['requests']}) != 1 or
            not run['requests'][0]['original_dense_four_latents_bitwise_equal'] or
            any(b['UB1_reference_bit_mismatches'] for b in run['native_microbenchmarks'])):
        raise RuntimeError('Original native controls or matched initial noise differ')
    for request in run['requests']:
        if ([r['step'] for r in request['trajectory']] != list(range(4)) or
                request['counter_delta_excluding_debited_benchmarks'] != {
                    'linear_forward_calls': 1600, 'activation_qdq_calls': 1600, 'cache_misses': 0, 'weight_transfers': 0} or
                request['peak_allocated_bytes'] > 79 << 30 or
                sha(hf / request['video_HF_relative']) != request['video_sha256']):
            raise RuntimeError('Actual original native trajectory/counter/media/memory differs')
        if request['arm'] != 'dense_DQ' and (
                len(request['dispatch_events']) != 160 or
                {(e['step'], e['block']) for e in request['dispatch_events']} != {(s, b) for s in range(4) for b in range(40)}):
            raise RuntimeError('Missing full160 executed dispatch coverage')
    if (review['raw_run_sha256'] != sha(root / 'RUN.json') or not review['decoded_every_frame'] or
            review['frames_per_video'] != 81 or review['automatic_visual_PASS'] is not False):
        raise RuntimeError('Complete decode not bound to this execution')
    for item in [*review['all_frame_contact_pages'], review['paired_preview']]:
        if sha(hf / item['HF_relative']) != item['sha256']:
            raise RuntimeError('Review media changed')
    for name, expected in protocol['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Executed model source changed before export')
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if used > 1 << 29:
        raise RuntimeError('Actual HF output cap exceeded')
    base = ROOT / 'research/20261023'
    for source, destination in [('RUN.json', 'R16V0_native_results.json'),
                                 ('BUDGET.json', 'R16V0_budget_debit.json'),
                                 ('VIDEO_REVIEW.json', 'R16V0_video_review.json')]:
        archive(root / source, base / destination)
    resources = {'id': METHOD, 'status': 'COMPLETE_FIVE_NATIVE_VIDEOS_164_FFNS_AND_CPU_DECODE',
                 'full_generations': 5, 'standalone_FFN_forwards': 164, 'weight_factorizations': 0,
                 'training_steps': 0, 'official_quality_models': 0, 'max_GPUs': 1,
                 'actual_candidate_FFN_positions_per_video': 160, 'decoded_frames_per_video': 81,
                 'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in run['requests']),
                 'HF_output_bytes': used, 'free_HF_output_bytes_after': shutil.disk_usage(root).free,
                 'human_full_video_review_completed': False, 'population_quality_or_Pareto_acceptance': False,
                 'exporter_sha256': sha(__file__)}
    destination = base / 'R16V0_completed_resources.json'
    with destination.open('x') as stream:
        json.dump(resources, stream, indent=2)
        stream.write('\n')
    print('Five full160 native videos,164 FFN calls and complete CPU decode exported; no automatic visual promotion.')


if __name__ == '__main__':
    main()
