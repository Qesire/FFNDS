"""Validate and summarize the four frozen function-reconstruction studies.

No model, tensor fitting, or CUDA operation. Only completed scalar receipts.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from analyze_e0_dense_residency import validate_plan
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def summarize(cases, panels=None):
    names = set.intersection(*(set(c['results']) for c in cases))
    result = {'cases': len(cases), 'methods': {}}
    for name in sorted(names):
        result['methods'][name] = {}
        for panel in (panels or cases[0]['results'][name]):
            metrics = {}
            for metric in ['relative_target_SSE', 'relative_current_output_SSE', 'relative_centered_target_SSE']:
                values = [c['results'][name][panel][metric] for c in cases]
                metrics[metric] = {'mean': mean(values), 'median': median(values), 'min': min(values), 'max': max(values)}
            metrics['rows_per_case_range'] = [min(c['results'][name][panel]['rows'] for c in cases),
                                             max(c['results'][name][panel]['rows'] for c in cases)]
            result['methods'][name][panel] = metrics
    return result


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def analyze(hf):
    definitions = [
        ('statistical_gelu', 'statistical_gelu', 'statistical_gelu', 'PASS_STATISTICAL_GELU_GAINS'),
        ('output_basis', 'output_basis_prediction', 'output_basis', 'PASS_OUTPUT_BASIS_PREDICTION'),
        ('nonlinear_reconstruction', 'nonlinear_reconstruction', 'nonlinear_reconstruction', 'PASS_NONLINEAR_RECONSTRUCTION'),
        ('native_finite', 'reconstructed_native', 'reconstructed_native', 'PASS_RECONSTRUCTED_NATIVE_FINITE'),
    ]
    studies = {}
    for key, config, directory, status in definitions:
        relative = f'config/e0_{config}_20261026.json'
        plan, digest = validate_plan(relative)
        path = hf / f'e0_function_basis_20261026/{directory}_attempt1/RUN.json'
        raw = json.loads(path.read_text())
        if not finite(raw) or raw['status'] != status or raw['protocol_sha256'] != digest:
            raise ValueError('Receipt incomplete or wrong protocol')
        if raw['counts'] != plan['count_limits'] or len(raw['cases']) != plan['max_cases']:
            raise ValueError('Incomplete counts or cases')
        sources = plan.get('held_sources', plan.get('source_indices'))
        expected = {(i, step, b, form) for i in sources for b in plan['blocks'] for step in range(4)
                    for form in (['absolute', 'innovation'] if key == 'output_basis' and step in [1, 3] else ['absolute'])}
        actual = [(c['source_index'], c['step'], c['block'], c.get('form', 'absolute')) for c in raw['cases']]
        if set(actual) != expected or len(set(actual)) != len(actual):
            raise ValueError('Missing or repeated case')
        if any(c['seed'] != 42 + c['source_index'] for c in raw['cases']):
            raise ValueError('Actual source seed mismatch')
        if any(c.get('calibration_source', 12 - c['source_index']) != 12 - c['source_index'] for c in raw['cases']):
            raise ValueError('Source holdout failed')
        resources = raw['assigned_GPU_resources']
        if not resources['sampled_isolation_pass'] or resources['violations'] or any(
                s['foreign_process_count_on_assigned_GPU'] for s in resources['samples']):
            raise ValueError('Assigned GPU isolation failed')
        if raw['new_generations'] or raw['optimizer_updates'] or raw['persisted_fitted_weights']:
            raise ValueError('Scope exceeded')
        size = sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())
        if size > plan['max_HF_output_gib'] * (1 << 30):
            raise ValueError('HF output cap exceeded')
        # Verify the exact pre-execution snapshot, including the protocol itself.
        snapshot = path.parent / 'EXECUTED_SOURCES'
        if sha256(snapshot / Path(relative).name) != digest:
            raise ValueError('Executed protocol differs')
        for name, bound in plan['bound_sources'].items():
            if sha256(snapshot / name) != bound:
                raise ValueError('Executed source snapshot differs')
        entry = {'protocol': relative, 'protocol_sha256': digest,
                 'raw_binding': {'HF_relative': str(path.relative_to(hf)), 'sha256': sha256(path)},
                 'status': status, 'actual_counts': raw['counts'], 'cases': len(raw['cases']),
                 'sampled_assigned_GPU_isolation_pass': True, 'HF_output_bytes': size,
                 'peak_physical_GPU_MiB': resources['peak_physical_used_memory_MiB']}
        cases = raw['cases']
        if key != 'native_finite':
            forms = sorted({c.get('form', 'absolute') for c in cases})
            entry['forms'] = {}
            for form in forms:
                subset = [c for c in cases if c.get('form', 'absolute') == form]
                panel = 'uniform32' if key == 'statistical_gelu' else 'uniform_disjoint'
                entry['forms'][form] = {
                    'overall': summarize(subset),
                    'by_block': {str(b): summarize([c for c in subset if c['block'] == b], [panel]) for b in plan['blocks']},
                    'by_step': {str(s): summarize([c for c in subset if c['step'] == s]) for s in sorted({c['step'] for c in subset})},
                    'by_source': {str(i): summarize([c for c in subset if c['source_index'] == i], [panel]) for i in sources}}
            if key == 'statistical_gelu':
                entry['continuous_decomposition_means'] = {panel: {
                    metric: mean(c['decomposition'][panel][metric] for c in cases)
                    for metric in cases[0]['decomposition'][panel]} for panel in cases[0]['decomposition']}
                entry['slope_discrepancy_means'] = {name: mean(c['calibration_slope_discrepancy'][name] for c in cases)
                                                   for name in cases[0]['calibration_slope_discrepancy']}
                entry['native_decode_relative_SSE_max'] = max(c['continuous_native_decode_SSE'] for c in cases)
            if key == 'nonlinear_reconstruction':
                entry['calibrations'] = raw['calibrations']
                if len(raw['calibrations']) != 10 or any(c['rows'] != 256 or c['sources'] != [4, 8, 16, 20] for c in raw['calibrations']):
                    raise ValueError('Calibration identity mismatch')
            if key == 'output_basis':
                entry['ungated_decomposition_means_by_form'] = {form: {
                    str(rank): {panel: {metric: mean(c['ungated_error_decomposition'][str(rank)][panel][metric]
                                                    for c in cases if c['form'] == form)
                                       for metric in ['tail_over_target_energy', 'coefficient_error_over_target_energy']}
                                for panel in ['contiguous128', 'uniform_disjoint']}
                    for rank in plan['ranks']} for form in forms}
                entry['all_source_gates_equal'] = all(c['source_gate_equal'] for c in cases)
                entry['uniform_overlap_removed_range'] = [min(c['uniform_overlap_removed'] for c in cases), max(c['uniform_overlap_removed'] for c in cases)]
        else:
            if raw['native_FFN_calls'] != 432 or raw['native_QDQ_calls'] != 432:
                raise ValueError('Native calls not accounted')
            entry.update(environment=raw['environment'], preparation=raw['preparation'], numerical_cases=cases)
            entry['overall'] = {'dense_ms_mean_of_case_medians': mean(median(c['dense_ms']) for c in cases), 'methods': {}}
            for method in cases[0]['methods']:
                entry['overall']['methods'][method] = {
                    'ms_mean_of_case_medians': mean(median(c['methods'][method]['full_M_ms']) for c in cases),
                    'dense_over_method_mean_of_paired_ratios': mean(median(c['dense_ms']) / median(c['methods'][method]['full_M_ms']) for c in cases),
                    'native_vs_continuous_relative_SSE_mean': mean(c['methods'][method]['native_vs_continuous_relative_SSE'] for c in cases)}
            entry['native_dense_vs_saved_SSE_max'] = max(v['relative_target_SSE'] for c in cases for v in c['native_dense_vs_saved'].values())
        studies[key] = entry
    return {'id': 'E0_FUNCTION_RECONSTRUCTION_20261026', 'status': 'COMPLETE',
            'analysis_sha256': sha256(Path(__file__)), 'base_commit': '087893f31a5718f108f7fba13a4bf41a8ebe1fc4',
            'actual_seeds_by_source': {str(i): 42+i for i in [0, 4, 8, 12, 16, 20]},
            'aggregation': 'Equal-case arithmetic mean of SSE/energy ratios; panels and target types separate. Independent image/seed clusters are2 held and4 calibration, not layers/tokens. All six sources were historically exposed.',
            'new_full_generations': 0, 'peak_concurrent_GPU_cards': 2, 'user_GPU_ceiling': 4,
            'optimizer_updates': 0, 'persisted_fitted_weights': 0, 'model_weight_decompositions': 0,
            'native_standalone_after_Q1_FFN_calls': 432, 'native_Q2_calls': 432,
            'studies': studies, 'quality_or_speed_acceptance': False, 'stage_changed': False,
            'candidate_operator_selected': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve the first validated summary')
    write_json(args.out, analyze(args.hf_root))
