"""Read-only OP2 metadata audit: cross-step row alignment and cache storage.

This never opens tensor payloads, loads a model, fits a predictor or selects a
deployment method. Equal sample offsets are not equal native token coordinates.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from ar2_checkpoint import atomic_json, digest
from ub1_native_data import load_ub1_collections

ROOT = Path(__file__).resolve().parents[1]


def paired_rows(left, right):
    """Return native coordinates and their distinct offsets in the two files."""
    a, b = left['selected_rows'], right['selected_rows']
    for scope, rows in [(left, a), (right, b)]:
        if (len(rows) != scope['rows'] or rows != sorted(set(rows))
                or not rows or rows[0] < 0 or rows[-1] >= scope['full_rows']):
            raise RuntimeError('Invalid sorted, unique native sample coordinates')
    if (left['grid'] != right['grid'] or left['full_rows'] != right['full_rows']
            or left['native_V1_shape'] != right['native_V1_shape']
            or left['native_V1_stride'] != right['native_V1_stride']):
        raise RuntimeError('Cross-step native geometry differs; cannot pair rows')
    a_offsets, b_offsets = {r: i for i, r in enumerate(a)}, {r: i for i, r in enumerate(b)}
    common = sorted(a_offsets.keys() & b_offsets.keys())
    return {'native_rows': common, 'anchor_offsets': [a_offsets[r] for r in common],
            'later_offsets': [b_offsets[r] for r in common]}


def summarize(rows):
    counts = [len(r['coordinates']['native_rows']) for r in rows]
    return {'source_pairs': len(rows), 'paired_native_rows': sum(counts),
            'pairs_with_no_shared_row': counts.count(0),
            'minimum_per_source_pair': min(counts), 'maximum_per_source_pair': max(counts),
            'histogram': {str(k): v for k, v in sorted(Counter(counts).items())}}


def audit(records):
    pairs, positions, crossing = [], [], []
    for block in range(40):
        for anchor, later in [(0, 1), (2, 3)]:
            entries = []
            for index in sorted(records):
                scopes = {(s['step'], s['block']): s for s in records[index]['scopes']}
                a, b = scopes[anchor, block], scopes[later, block]
                invariant_fields = ['expert', 'teacher_shard_sha256', 'original_b0_sha256', 'original_b2_sha256']
                if not all(a[k] == b[k] for k in invariant_fields):
                    raise RuntimeError('Same-expert FFN function metadata changed across steps')
                for s in [a, b]:
                    if (s['native_V1_shape'] != [s['full_rows'], 5120]
                            or s['native_V1_stride'] != [5120, 1]
                            or math.prod(s['grid']) != s['full_rows']):
                        raise RuntimeError('Expected original contiguous 5120-channel geometry')
                entry = {'input_index': index, 'block': block, 'anchor_step': anchor, 'later_step': later,
                         'coordinates': paired_rows(a, b),
                         'identical_sample_offsets': sum(x == y for x, y in zip(a['selected_rows'], b['selected_rows'])),
                         'same_gate_sha256': a['gate_sha256'] == b['gate_sha256'],
                         'anchor_payload_sha256': a['sha256'], 'later_payload_sha256': b['sha256']}
                entries.append(entry)
            folds = []
            for validation_parity in [0, 1]:
                train = [r for r in entries if r['input_index'] % 2 != validation_parity]
                valid = [r for r in entries if r['input_index'] % 2 == validation_parity]
                n = summarize(train)['paired_native_rows']
                folds.append({'validation_source_parity': validation_parity,
                              'train_paired_rows': n, 'validation_paired_rows': summarize(valid)['paired_native_rows'],
                              'centered_training_design_rank_upper_bound': max(0, min(5120, n - 1))})
            positions.append({'block': block, 'anchor_step': anchor, 'later_step': later,
                              **summarize(entries), 'existing_source_folds': folds})
            pairs.extend(entries)
        for index in sorted(records):
            scopes = {(s['step'], s['block']): s for s in records[index]['scopes']}
            a, b = scopes[1, block], scopes[2, block]
            crossing.append({'input_index': index, 'block': block,
                             'same_expert': a['expert'] == b['expert'],
                             'same_teacher_shard': a['teacher_shard_sha256'] == b['teacher_shard_sha256']})
    geometries = sorted({(tuple(s['grid']), s['full_rows']) for c in records.values() for s in c['scopes']})
    memory = []
    for grid, m in geometries:
        y = 40 * m * 5120 * 2
        h = 40 * m * 256 * 2
        memory.append({'grid': list(grid), 'M': m, 'BF16_Y_cache_all40_blocks_bytes': y,
                       'BF16_X_plus_Y_cache_all40_blocks_bytes': 2 * y,
                       'BF16_Y_plus_projected_h256_cache_all40_blocks_bytes': y + h,
                       'Y_only_cache_GiB': y / 2**30, 'X_plus_Y_cache_GiB': 2 * y / 2**30,
                       'Y_plus_h256_cache_GiB': (y + h) / 2**30,
                       'uniform_independent_draw_reference_expected_overlap_per_source_pair': 256**2 / m,
                       'pair_inclusion_probability_under_independent_uniform_reference': (256 / m)**2})
    return {'cases': len(records), 'metadata_scopes': sum(len(c['scopes']) for c in records.values()),
            'same_expert_pairs': {**summarize(pairs),
                                  'same_gate_pairs': sum(r['same_gate_sha256'] for r in pairs),
                                  'equal_native_rows_at_same_saved_offset': sum(r['identical_sample_offsets'] for r in pairs),
                                  'naively_zipped_saved_rows': len(pairs) * 256,
                                  'function_metadata_matches_all_pairs': True},
            'positions': positions, 'coordinate_pairs': pairs,
            'expert_boundary': {'source_block_pairs': len(crossing),
                                'same_expert_pairs': sum(r['same_expert'] for r in crossing),
                                'same_teacher_shard_pairs': sum(r['same_teacher_shard'] for r in crossing)},
            'storage_arithmetic': memory,
            'main_MAC_fraction_dense_at_0_2_skip_at_1_3': .5,
            'hypothetical_aux_MAC_fraction_of_four_dense_FFNs': {
                'full_X_cache_two_projections_at_each_of_two_reuse_steps': 256 / (2 * 13824),
                'h256_cache_one_projection_at_each_anchor_plus_two_at_each_reuse_step': 3 * 256 / (4 * 13824)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--evidence-base-commit', required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise RuntimeError('Preserve the first completed coordinate audit')
    if not re.fullmatch('[0-9a-f]{40}', args.evidence_base_commit):
        raise ValueError('A full evidence base commit is required')
    runs = ROOT / 'runs/worker_restart_20261020'
    records, bindings, capture = load_ub1_collections(
        [runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT)
    result = audit(records)
    result.update(status='COMPLETE_OP2_CROSS_STEP_COORDINATE_AUDIT_CPU_ONLY',
                  evidence_base_commit=args.evidence_base_commit,
                  collection_bindings=[{'path': str(Path(b['path']).relative_to(ROOT)), 'sha256': b['sha256']} for b in bindings],
                  input_report_bindings=[{'input_index': i, 'sha256': digest(next(
                      e['input_report'] for b in bindings for e in json.loads(Path(b['path']).read_text())['completed']
                      if e['index'] == i))} for i in sorted(records)],
                  original_UB1_policy_sha256=capture['original_UB1_policy_sha256'],
                  sources={f'tools/{name}': digest(ROOT / 'tools' / name) for name in [
                      'audit_cross_step_coordinates.py', 'ub1_native_data.py', 'ar2_checkpoint.py']},
                  tensor_payloads_opened=0, models_loaded=0, fitted_parameters=0,
                  GPU_jobs_started=0, quality_or_speed_acceptance=False,
                  limitations=[
                      'Metadata/source/no-op/video bindings checked by existing OP2 loader; tensor bodies not reread.',
                      'No tensor values, cache errors, cross-step numerical closeness or predictable gains measured.',
                      'All24 sources and their saved draws already entered development/fitting; not fresh evaluation.',
                      'Same spatial coordinate is not optical-flow correspondence or bitwise activation equality.',
                      'Pair inclusion formula is a design reference, not proof of unbiased fitted-trajectory risk.',
                      'Storage counts cache arrays only; live activations, weights, allocators and copies are extra.',
                      'Projected-cache memory assumes a newly specified shared linear map; native rounding identity unproved.',
                      'MAC arithmetic excludes quantization, normalization, cache transfer and launch cost; not measured speed.'
                  ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, result)
    print(json.dumps({'status': result['status'], 'same_expert_pairs': result['same_expert_pairs'],
                      'block20': [p for p in result['positions'] if p['block'] == 20],
                      'storage_arithmetic': result['storage_arithmetic']}))


if __name__ == '__main__':
    main()
