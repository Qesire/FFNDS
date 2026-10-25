"""Validate frozen dense runtime controls and export public, paired evidence.

This reads completed runs. It never loads a model or initializes CUDA.
"""
import argparse
import json
from pathlib import Path

from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def validate_plan(relative):
    path = ROOT / relative
    plan = json.loads(path.read_text())
    for group in ('driver_sources_sha256', 'native_source_sha256', 'bound_sources'):
        for name, digest in plan.get(group, {}).items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Frozen source changed: {name}')
    if 'accepted_observations' in plan:
        if sha256(ROOT / plan['accepted_observations']) != plan['accepted_observations_sha256']:
            raise ValueError('Accepted observations changed')
    return plan, sha256(path)


def analyze(hf_root, receipt_root=None):
    references = json.loads((ROOT / 'research/20261025/E0_NATIVE_OBSERVATIONS.json').read_text())
    source_refs = {s['index']: s for s in references['sources']}
    studies = [
        ('e0_dense_residency_20261025', 'config/e0_dense_weight_residency_20261025.json',
         [('original', 'A'), ('block_no_flush', 'B'), ('ffn_resident_no_flush', 'A')]),
        ('e0_dense_crossover_20261025', 'config/e0_dense_residency_crossover_20261025.json',
         [('original', 'B'), ('block_no_flush', 'A')]),
        ('e0_all_weight_residency_20261025', 'config/e0_all_weight_residency_20261025.json',
         [('all_resident_no_flush', 'B')]),
    ]
    runs = []
    card_bindings = {}
    all_receipts = []
    for study, protocol, modes in studies:
        plan, plan_sha = validate_plan(protocol)
        for mode, card in modes:
            relative = f'{study}/{mode}_attempt1/RUN.json'
            path = hf_root / relative
            raw = json.loads(path.read_text())
            if (not raw['status'].startswith('PASS_') or raw['protocol_sha256'] != plan_sha
                    or raw['generations_started'] != 3 or raw['generations_completed'] != 3
                    or len(raw['phases']) != 3 or raw['mode'] != mode):
                raise ValueError('Incomplete/mismatched native study')
            resources = raw['assigned_GPU_resources']
            if not resources['sampled_isolation_pass'] or resources['violations']:
                raise ValueError('Assigned GPU isolation failed')
            if any(s['foreign_process_count_on_assigned_GPU'] for s in resources['samples']):
                raise ValueError('Foreign GPU process recorded')
            if receipt_root is not None:
                receipt_path = receipt_root / study / f'{mode}_JOB_PRIVATE.json'
                receipt = json.loads(receipt_path.read_text())
                if (receipt['mode'] != mode or receipt['protocol_sha256'] != plan_sha
                        or receipt['GPU'] != resources['physical_GPU_index']):
                    raise ValueError('Private launch receipt mismatch')
                binding = (receipt['slot'], receipt['GPU'])
                if card in card_bindings and card_bindings[card] != binding:
                    raise ValueError('Same-card pairing failed')
                card_bindings[card] = binding
                all_receipts.append({'sha256': sha256(receipt_path), 'anonymous_card': card})
            phases = []
            for order, phase in enumerate(raw['phases']):
                index = plan['source_sequence'][order]
                ref = source_refs[index]
                if (phase['order'] != order or phase['source_index'] != index
                        or phase['source_id'] != ref['source_id'] or phase['seed'] != ref['seed']
                        or phase['trajectory'] != ref['trajectory']
                        or phase['video_sha256'] != ref['video_sha256']
                        or phase['identity'] != 'EXACT_HISTORICAL_LATENTS_AND_VIDEO'):
                    raise ValueError('Original dense identity failed')
                video = path.parent / f'phase{order:02d}_source{index:02d}.mp4'
                if sha256(video) != ref['video_sha256']:
                    raise ValueError('Encoded artifact changed')
                expected = dict(ref['native_counts_each_phase'])
                hit_key = 'all_retained_hits' if mode == 'all_resident_no_flush' else 'FFN_retained_hits'
                hits = phase['policy_delta'].get(hit_key, 0)
                expected['weight_transfers'] -= hits
                if phase['native_counts'] != expected:
                    raise ValueError('Unexpected arithmetic/copy counts')
                events = phase['FFN_boundary_timings']
                if (len(events) != 160 or {(e['step'], e['block']) for e in events}
                        != {(s, b) for s in range(4) for b in range(40)}):
                    raise ValueError('Missing native FFN boundary events')
                if abs(sum(e['elapsed_ms'] for e in events) - phase['FFN_boundary_total_ms']) > 1e-6:
                    raise ValueError('FFN event sum mismatch')
                phases.append({k: v for k, v in phase.items() if k != 'FFN_boundary_timings'})
            runs.append({
                'study': study, 'mode': mode, 'anonymous_card': card,
                'protocol': protocol, 'protocol_sha256': plan_sha,
                'raw_binding': {'HF_relative': relative, 'sha256': sha256(path)},
                'sampled_peak_physical_MiB': resources['peak_physical_used_memory_MiB'],
                'resource_sampling_interval_seconds': resources['sampling_interval_seconds'],
                'sampled_isolation_pass': True, 'phases': phases,
                'retained_weight_peak_bytes': raw['final_policy']['peak_retained_bytes'],
                'retained_bytes_after_context_exit': raw['final_policy']['retained_bytes'],
                'native_counts_observed': raw['native_counts_observed'],
                'HF_output_bytes': sum(f.stat().st_size for f in path.parent.rglob('*') if f.is_file()),
                'memory_precheck': raw.get('memory_precheck'),
            })
            if raw['final_policy']['retained_bytes'] != 0:
                raise ValueError('Policy did not release retained references')
    if receipt_root is not None and (len(card_bindings) != 2 or
            len({v[0] for v in card_bindings.values()}) != 1 or
            len(set(card_bindings.values())) != 2):
        raise ValueError('Reciprocal cards must be distinct and on the same worker')

    paired = []
    for card in ['A', 'B']:
        original = next(r for r in runs if r['anonymous_card'] == card and r['mode'] == 'original')
        for variant in [r for r in runs if r['anonymous_card'] == card and r['mode'] != 'original']:
            for order in range(3):
                base, target = original['phases'][order], variant['phases'][order]
                paired.append({
                    'anonymous_card': card, 'mode': variant['mode'], 'phase': order,
                    'source_index': base['source_index'], 'seed': base['seed'],
                    'original_wall_seconds': base['wall_seconds_including_video_save'],
                    'variant_wall_seconds': target['wall_seconds_including_video_save'],
                    'observed_speed_ratio': base['wall_seconds_including_video_save'] / target['wall_seconds_including_video_save'],
                    'observed_wall_reduction_fraction': 1 - target['wall_seconds_including_video_save'] / base['wall_seconds_including_video_save'],
                    'same_worker_card_verified_by_private_receipts': receipt_root is not None,
                })
    return {
        'id': 'E0_DENSE_WEIGHT_RESIDENCY_COMPLETED_20261025', 'status': 'COMPLETE',
        'analysis_sha256': sha256(Path(__file__)),
        'scope': {'exposed_source_clusters': 2, 'source_indices': [0, 12],
                  'actual_seeds_by_source': {'0': 42, '12': 54},
                  'base_seed_note': 'canonical source seed=42+source_index; not a common seed across sources',
                  'videos_per_run': 3, 'phase_order': ['source0_first', 'source0_repeat', 'source12_warm'],
                  'runs': 6, 'worker_count': 1, 'actual_GPU_peak': 2, 'user_GPU_ceiling': 4},
        'identity': {'matched_video_files': 18, 'matched_scheduler_latent_hashes': 72,
                     'linear_calls_each_generation': 1600, 'activation_QDQ_calls_each_generation': 1600,
                     'cache_misses_each_generation': 0, 'FFN_MAC_reduction': 0},
        'actual_budget': {'full_generations': 18, 'failed_generations': 0, 'retries': 0,
                          'training_updates': 0, 'new_decompositions': 0, 'official_quality_calls': 0},
        'private_pairing_receipt_bindings': all_receipts, 'runs': runs, 'same_card_comparisons': paired,
        'timing_boundary': 'Host wall around pipe.generate and post-generate CUDA synchronize, includes encode/save and common trajectory hashing; excludes asset validation, imports, model/pipeline construction and downloads.',
        'FFN_event_boundary': 'Native infer_ffn: attention residual add, norm2/modulation, Q1/W1/GELU/Q2/W2; excludes following gate/residual writeback. Events read after generation synchronization.',
        'limitations': [
            'Two exposed image/seed clusters, one three-generation sequence per policy/card; technical phases are not independent sources.',
            'Only original versus no-flush has reciprocal two-card crossover. All-weight residency has a same-card original control, but no reciprocal all-weight run.',
            'Sampled physical memory includes reserved allocator memory; it is not a continuous maximum or live-tensor measurement.',
            'Weight-fetch host wall may include synchronization/allocator waiting; it is not pure PCIe transfer time.',
            'No reduction of FFN arithmetic or general speed/quality acceptance; 80GB capacity is part of this runtime result.',
        ],
        'observed_native_identity': True, 'candidate_operator_selected': False,
        'formal_quality_or_speed_acceptance': False, 'stage_changed': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf-root', type=Path, required=True)
    parser.add_argument('--receipt-root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve the first validated summary')
    write_json(args.out, analyze(args.hf_root, args.receipt_root))
