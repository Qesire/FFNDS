"""Validate the completed all-depth FFN representation studies, without CUDA."""
import argparse
import json
from pathlib import Path
from statistics import mean, median

from analyze_e0_dense_residency import validate_plan
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
PANELS = ['contiguous128', 'uniform32']


def aggregate(cases, tangent=False):
    output = {'case_count': len(cases), 'means': {}, 'ranges': {}}
    for name in cases[0]['results']:
        output['means'][name] = {}
        output['ranges'][name] = {}
        for panel in PANELS:
            metrics = (cases[0]['results'][name]['panels'][panel] if tangent
                       else cases[0]['results'][name][panel])
            output['means'][name][panel] = {}
            output['ranges'][name][panel] = {}
            for metric in metrics:
                values = [(c['results'][name]['panels'][panel] if tangent
                           else c['results'][name][panel])[metric] for c in cases]
                output['means'][name][panel][metric] = mean(values)
                output['ranges'][name][panel][metric] = [min(values), max(values)]
    return output


def analyze(hf_root):
    root = hf_root / 'e0_all_block_tangent_20261025'
    definitions = [
        ('all_block_tangent', 'saved_gpu_attempt1', 'e0_all_block_tangent_saved_20261025',
         'PASS_ALL_BLOCK_FULL_M_TANGENT_DISCRIMINATION', 160),
        ('cross_source_increment', 'fold_group_gpu_attempt1', 'e0_fold_group_residual_20261025',
         'PASS_CROSS_SOURCE_FOLD_GROUP_RESIDUAL', 160),
        ('absolute_affine', 'absolute_fold_gpu_attempt1', 'e0_absolute_affine_fold_20261025',
         'PASS_ABSOLUTE_AFFINE_GROUP_FOLD', 40),
        ('isolated_microbenchmark', 'fold_benchmark_attempt1', 'e0_fold_microbenchmark_20261025',
         'PASS_ISOLATED_NATIVE_FOLD_MICROBENCHMARK', 8),
    ]
    studies = {}
    for key, directory, filename, status, count in definitions:
        relative = f'config/{filename}.json'
        plan, digest = validate_plan(relative)
        path = root / directory / 'RUN.json'
        raw = json.loads(path.read_text())
        if raw['status'] != status or raw['protocol_sha256'] != digest or len(raw['cases']) != count:
            raise ValueError('Incomplete frozen study')
        resources = raw['assigned_GPU_resources']
        if (not resources['sampled_isolation_pass'] or resources['violations'] or
                any(s['foreign_process_count_on_assigned_GPU'] for s in resources['samples'])):
            raise ValueError('Assigned GPU isolation failed')
        expected = plan.get('projection_limits', plan.get('call_limits'))
        actual = raw.get('diagnostic_projections', raw.get('counters'))
        if actual != expected:
            raise ValueError('Diagnostic budget mismatch')
        cases = raw['cases']
        expected_keys = {(i, s, b) for i in [0, 12]
                         for s in ([0, 1, 2, 3] if key == 'absolute_affine' else [1, 3])
                         for b in ([30, 39] if key == 'isolated_microbenchmark' else
                                   [0, 10, 20, 30, 39] if key == 'absolute_affine' else range(40))}
        if len({(c['source_index'], c['step'], c['block']) for c in cases}) != count or {
                (c['source_index'], c['step'], c['block']) for c in cases} != expected_keys:
            raise ValueError('Missing/duplicate source-step-block case')
        for c in cases:
            if 'calibration_source' in c:
                if c['calibration_source'] != 12 - c['source_index']:
                    raise ValueError('Calibration is not held-source')
                if c.get('kappa_calibration_source', c['calibration_source']) != c['calibration_source']:
                    raise ValueError('Slope and support calibration differ')
        entry = {'protocol': relative, 'protocol_sha256': digest,
                 'raw_binding': {'HF_relative': str(path.relative_to(hf_root)), 'sha256': sha256(path)},
                 'status': raw['status'], 'actual_diagnostic_counts': actual,
                 'sampled_isolation_pass': True, 'cases': cases}
        if key != 'isolated_microbenchmark':
            tangent = key == 'all_block_tangent'
            entry['overall'] = aggregate(cases, tangent)
            entry['by_block'] = {str(b): aggregate([c for c in cases if c['block'] == b], tangent)
                                 for b in sorted({c['block'] for c in cases})}
            entry['by_step'] = {str(s): aggregate([c for c in cases if c['step'] == s], tangent)
                                for s in sorted({c['step'] for c in cases})}
            if key != 'absolute_affine':
                entry['descriptive_block_threshold_counts'] = {}
                for s in [1, 3]:
                    subset = [c for c in cases if c['step'] == s]
                    per_block = {b: aggregate([c for c in subset if c['block'] == b], tangent)
                                 for b in range(40)}
                    entry['descriptive_block_threshold_counts'][str(s)] = {
                        name: {panel: {
                            'mean_of_two_sources_below_1pct': sum(v['means'][name][panel]['relative_current_output_SSE'] < .01 for v in per_block.values()),
                            'mean_of_two_sources_below_5pct': sum(v['means'][name][panel]['relative_current_output_SSE'] < .05 for v in per_block.values()),
                        } for panel in PANELS} for name in subset[0]['results']}
        else:
            entry['environment'] = raw['environment']
            summary = []
            for c in cases:
                native = median(c['full_M_native_after_Q1_FFN_ms'])
                derivative = median(c['full_M_old_derivative_ms'])
                record = {'source_index': c['source_index'], 'step': c['step'], 'block': c['block'],
                          'native_after_Q1_ms_median': native, 'old_derivative_ms_median': derivative,
                          'native_128_vs_saved_full_M_relative_SSE': c['native_128_vs_saved_full_M_relative_SSE'],
                          'methods': {}}
                for dtype, values in c['methods'].items():
                    build = median(values['matrix_build_ms'])
                    apply = median(values['full_M_application_ms'])
                    record['methods'][dtype] = {
                        'matrix_build_ms_median': build, 'application_ms_median': apply,
                        'current_unfused_observer_plus_build_plus_apply_over_native': (derivative + build + apply) / native,
                        'sample128_native_output_relative_SSE': values['sample128_native_output_relative_SSE'],
                    }
                summary.append(record)
            entry['timing_summary'] = summary
        studies[key] = entry
    observations = ROOT / 'research/20261025/E0_ALL_BLOCK_TANGENT_OBSERVATIONS.json'
    accepted = json.loads(observations.read_text())
    return {
        'id': 'E0_AFFINE_FFN_DISCOVERY_20261025', 'status': 'COMPLETE',
        'analysis_sha256': sha256(Path(__file__)),
        'validation_dependency_sha256': sha256(ROOT / 'tools/analyze_e0_dense_residency.py'),
        'native_observation_binding': {'path': str(observations.relative_to(ROOT)), 'sha256': sha256(observations)},
        'scope': {'exposed_source_clusters': 2, 'actual_seeds_by_source': {'0': 42, '12': 54},
                  'native_observed_blocks': 40, 'native_observed_steps': 4,
                  'error_evaluation_panels': PANELS,
                  'full_M_note': 'Full-M means 31668 old rows only for derivative statistics; output errors use separately reported center128 and original random32 rows.',
                  'aggregation': 'Arithmetic mean of case-level squared-error/energy ratios; not pooled SSE and not independent replications across blocks/tokens.'},
        'native_observation_budget': accepted['actual_budget'],
        'native_identity': accepted['native_identity'], 'studies': studies,
        'ideal_projection_cost': {
            'absolute_affine_base_equivalent_groups_of_216': 40,
            'absolute_affine_plus27_vs_group_only67': [67 / 216, 67 / 216],
            'absolute_affine_plus54_vs_group_only94': [94 / 216, 94 / 216],
            'increment_fixed_linear_plus27_vs_group_only54': [5120 / (2 * 13824) + 1.5 * 27 / 216, 1.5 * 54 / 216],
            'exclusions': 'Ideal projection MAC only; excludes Q1/Q2, layout, routing, fetching, matrix creation, storage and error propagation. Increment controls recompute old/current selected W1 instead of free cached old hidden.',
        },
        'limitations': [
            'All data come from two exposed image/seed clusters; no new source/seed quality evidence.',
            'Saved-data representations are not returned to the network; no candidate video or packed partial Q2/FFN exists.',
            'Source-fixed parameters are calibrated only on the other source; posthoc energy ranking is label-informed and not a net-error-optimal oracle.',
            'Microbenchmark uses repeated saved rows at native full-M shape, not natural full-M tensors or end-to-end generation.',
            'No general exclusion of affine bases, nonlinear residuals, offline fixed slopes or fused observers follows from these local errors.',
        ],
        'candidate_operator_selected': False, 'quality_or_speed_acceptance': False, 'stage_changed': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve the first validated summary')
    write_json(args.out, analyze(args.hf_root))
