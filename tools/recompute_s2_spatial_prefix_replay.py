"""Rebuild source-equal fixed-state results from closed four-corner replays.

Only scalar output records are read. No model/tensor replay and no phase angle
is averaged. Original per-bin complex phases remain in immutable HF records.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
import e0_s2_spatial_prefix_replay_v2 as replay
m, native = replay.core, replay.native


def scalar_metrics(record):
    result = {}
    def flatten(value, prefix):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('frequency_bins', 'candidate_phase_radians'): continue
                flatten(item, prefix+'/'+key)
        elif value is None or type(value) in (int, float):
            native.need(value is None or math.isfinite(value), 'Finite scalar outputs required')
            result[prefix] = value
    for key in ('gY', 'u', 'rowLN_u', 'native_fullM_anchor_bridge',
                'observed_prefix_input_relative_SSE', 'M1092_vs_fullM_Y_relative_SSE'):
        flatten(record[key], key)
    return result


def mean_vectors(rows, labels):
    native.need(rows, 'At least one real observation required')
    values, counts = [], []
    for label in labels:
        valid = [row[label] for row in rows if row[label] is not None]
        values.append(math.fsum(valid)/len(valid) if valid else None); counts.append(len(valid))
    return {'mean': values, 'valid_position_counts': counts, 'position_N': len(rows)}


def equal_image_vectors(source_vectors):
    """Average each source once; never weight an image by its eligible rows."""
    native.need(source_vectors and len({len(v['mean']) for v in source_vectors}) == 1, 'Matched source vectors required')
    means, counts = [], []
    for column in zip(*(v['mean'] for v in source_vectors)):
        eligible = [v for v in column if v is not None]
        means.append(math.fsum(eligible)/len(eligible) if eligible else None); counts.append(len(eligible))
    return {'mean': means, 'valid_independent_image_counts': counts, 'independent_image_N': len(source_vectors)}


def record_stream(hf, binding, count_expected):
    path = (hf/binding['path']).resolve()
    native.need(path.is_relative_to(hf.resolve()) and path.stat().st_size == binding['bytes'], 'Scalar record stream size differs')
    before = native.stat_record(path); digest = hashlib.sha256(); count = 0; rows = []
    with path.open('rb') as stream:
        for line in stream:
            count += len(line); digest.update(line); rows.append(json.loads(line))
    native.need(count == binding['bytes'] and digest.hexdigest() == binding['sha256']
        and native.stat_record(path) == before and len(rows) == count_expected, 'Scalar record SHA/stat/count differs')
    return rows


def checked_records(hf, proof, plan):
    run = native.read(native.bound(hf, proof['RUN']))
    receipt = native.read(native.bound(hf, proof['exit']))
    m.core.vm.exited(receipt, proof['RUN'], proof['protocol']['sha256'], plan['id'])
    native.need(run['status'] == 'COMPLETE_TWO_SOURCE_ONE_EXPERT_LOCAL_PREFIX_FULL_SPATIAL_REPLAY'
        and run['completed_positions'] == 160 and run['all_original_u_neutrals_passed'] is True
        and not run['assigned_GPU_resources']['violations'], 'Replay not complete and exited')
    expected = {k: v for k, v in plan['limits'].items() if v}
    native.need(run['operation_ledger']['predebited'] == run['operation_ledger']['observed'] == expected,
                'Closed replay exact costs differ')
    return run, record_stream(hf, run['positions'], 320)


def checked_v3_records(hf, proof, plan):
    import e0_s2_spatial_prefix_replay_batched_v3 as v3
    v3.validate(plan, True)
    parent = v3.verify_parent(hf, plan['parent'])
    run = native.read(native.bound(hf, proof['RUN']))
    receipt = native.read(native.bound(hf, proof['exit']))
    m.core.vm.exited(receipt, proof['RUN'], proof['protocol']['sha256'], plan['id'])
    native.need(run['status'] == 'COMPLETE_V3_MERGED_160_POSITIONS'
        and run['parent'] == plan['parent'] and run['parent_completed_positions'] == parent['completed_positions']
        and run['controls_all_passed'] is True and run['controls_completed'] == 3
        and run['remaining_completed'] == len(plan['remaining']) and run['merged_records'] == 320
        and not run['assigned_GPU_resources']['violations'], 'V3 full merge/control/exit is incomplete')
    expected = {k: v for k, v in plan['limits'].items() if v}
    native.need(run['operation_ledger']['predebited'] == run['operation_ledger']['observed'] == expected,
                'V3 exact operation ledger differs')
    controls = record_stream(hf, run['controls'], 6)
    by_key = {(r['block'], r['source_id'], r['step'], r['family']): r for r in parent['records']}
    for i, position in enumerate(plan['controls']):
        for j, family in enumerate(m.FAMILIES):
            control = controls[2*i+j]
            native.need(control['position'] == position and control['family'] == family
                and control['comparison']['passed'] is True
                and v3.compare_records(by_key[(*position, family)], control['record'])['passed'], 'Recomputed V3 control differs')
    added = record_stream(hf, run['positions'], 2*len(plan['remaining']))
    merged = record_stream(hf, run['merged_positions'], 320)
    native.need(merged == v3.merge_records(plan['part'], plan['source_order'], parent['records'],
        parent['completed_positions'], added), 'Published merged records differ from immutable parent plus new tail')
    return run, merged, parent['parent_report']['operation_ledger']['observed']


def rebuild(hf, version='v3'):
    native.need(version in ('v2', 'v3'), 'Explicit replay version required')
    records, parents, operation_totals = [], [], defaultdict(int)
    inherited_totals, current_totals = defaultdict(int), defaultdict(int)
    for part in (0, 1):
        name = f'e0_s2_spatial_prefix_replay_batched_part{part}_20261029_v3.json' if version == 'v3' else f'e0_s2_spatial_prefix_replay_part{part}_20261029_v2.json'
        path = ROOT/'config'/name
        plan = native.read(path)
        out = hf/plan['output_HF_relative']
        proof = {'protocol': native.binding(path, ROOT), 'RUN': native.binding(out/'RUN.json', hf),
                 'exit': native.binding(out.parent/'EXIT_RECEIPT.json', hf)}
        if version == 'v3':
            run, rows, inherited = checked_v3_records(hf, proof, plan)
        else:
            m.validate(plan, True); run, rows = checked_records(hf, proof, plan); inherited = {}
            native.need(run['input_index_sha256'] == plan['input_index_sha256'], 'Frozen operand index differs')
        records.extend(rows); parents.append({**proof, 'positions': run.get('merged_positions', run['positions']),
            'inherited_parent': plan.get('parent'), 'controls': run.get('controls'),
            'wall_seconds': run['wall_seconds'], 'packet_bytes_read': run['packet_bytes_read'],
            'output_accounting': run['output_accounting']})
        for k, v in run['operation_ledger']['observed'].items(): current_totals[k] += v; operation_totals[k] += v
        for k, v in inherited.items(): inherited_totals[k] += v; operation_totals[k] += v
    ids = {(r['source'], r['family'], r['step'], r['block']) for r in records}
    expected = {(s, f, t, b) for s in (898, 96) for f in m.FAMILIES for t in range(4) for b in range(40)}
    native.need(ids == expected and len(records) == 640, 'All two-source/two-family/full160 unique scalar records required')
    metrics = {key: scalar_metrics(r) for key, r in zip(
        ((r['source'], r['family'], r['step'], r['block']) for r in records), records)}
    labels = sorted(next(iter(metrics.values())))
    native.need(all(sorted(v) == labels for v in metrics.values()), 'Metric schema differs across real positions')
    by_source = {}
    for source in (898, 96):
        by_source[str(source)] = {}
        for family in m.FAMILIES:
            def rows(step=None):
                return [metrics[source, family, t, b] for t in range(4) for b in range(40) if step is None or t == step]
            by_source[str(source)][family] = {'all_positions': mean_vectors(rows(), labels),
                'by_step': [mean_vectors(rows(t), labels) for t in range(4)]}
    equal = {}
    for family in m.FAMILIES:
        equal[family] = {'all_positions': equal_image_vectors([by_source[str(s)][family]['all_positions'] for s in (898, 96)]),
            'by_step': [equal_image_vectors([by_source[str(s)][family]['by_step'][t] for s in (898, 96)]) for t in range(4)]}
    # Keep a compact full-depth profile useful for selecting future *new*
    # causal intervention scopes, with no claim these are causal effects.
    profile_labels = [label for label in labels if label.startswith(('gY/tau1_to20/', 'rowLN_u/tau1_to20/'))
        and '/frequency/' not in label and label.rsplit('/', 1)[-1] in (
        'teacher_local_error_fraction', 'prefix_error_fraction', 'interaction_error_fraction', 'total_error_fraction')]
    profile = []
    for family in m.FAMILIES:
        for t in range(4):
            for b in range(40):
                profile.append({'family': family, 'step': t, 'block': b,
                    **equal_image_vectors([mean_vectors([metrics[s, family, t, b]], profile_labels) for s in (898, 96)])})
    first = [r for r in records if r['step'] == r['block'] == 0]
    return {'status': 'COMPLETE_TWO_SOURCE_FULL160_SAME_H_LOCAL_PREFIX_REPLAY_ACTUAL_EXIT0',
        'evidence': 'NATIVE_BF16_FIXED_STATE_COUNTERFACTUAL_AND_SCALAR_RECOMPUTATION_NOT_VIDEO_CAUSALITY',
        'independent_images': 2, 'exposure': 'TWO_EXPOSED_HIGH_MOTION_DIAGNOSTICS_NOT_POPULATION_CONFIRMATION',
        'source_order': [898, 96], 'families': list(m.FAMILIES), 'parents': parents,
        'actual_records': 640, 'actual_operations': dict(operation_totals),
        'statistics_version': version, 'inherited_original_statistics_operations': dict(inherited_totals),
        'new_statistics_operations_including_separate_controls': dict(current_totals),
        'distinct_complete_replay_positions': 320, 'extra_control_positions': 6 if version == 'v3' else 0,
        'all_gate_shift_scale_equal': all(all(r['gate_shift_scale_equal'].values()) for r in records),
        'first_site_controls': [{'source': r['source'], 'family': r['family'], **r['same_first_site_V1_h_raw_equal']} for r in first],
        'metric_labels': labels, 'by_source': by_source, 'source_equal': equal,
        'depth_profile_metric_labels': profile_labels, 'depth_profile': profile,
        'averaging': 'POSITION_EQUAL_WITHIN_IMAGE_THEN_EQUAL_IMAGE; NULLS_HAVE_EXPLICIT_ELIGIBLE_COUNTS; NO_TOKEN_N_OR_CI',
        'phase': 'PER_BIN_AND_BAND_AGGREGATE_COMPLEX_RESULTS_REMAIN_IN_BOUND_HF_POSITIONS; PHASE_ANGLES_ARE_NOT_AVERAGED',
        'interpretation': 'P_AND_I_INCLUDE_OBSERVED_UPSTREAM_AND_TYPED_NUMERICAL_CHANGES; SIGNED_CROSS_RETAINED; NOT_CAUSAL_ATTRIBUTION',
        'new_model_calls_by_this_recomputation': 0, 'quality_or_speed_accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    result = rebuild(args.hf_root); native.write(args.out, result)
    print(json.dumps({k: result[k] for k in ('status', 'actual_records', 'actual_operations', 'all_gate_shift_scale_equal')}))


if __name__ == '__main__': main()
