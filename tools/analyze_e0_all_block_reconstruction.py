"""All-depth matched-calibration function controls, scalar validation only."""
import argparse
import json
from pathlib import Path

from analyze_e0_dense_residency import validate_plan
from analyze_e0_function_reconstruction import finite, summarize
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def analyze(hf):
    relative = 'config/e0_all_block_reconstruction_20261026.json'
    plan, digest = validate_plan(relative)
    path = hf / 'e0_function_basis_20261026/all_block_reconstruction_attempt1/RUN.json'
    raw = json.loads(path.read_text()); cases = raw['cases']
    if not finite(raw) or raw['status'] != 'PASS_ALL_BLOCK_RECONSTRUCTION' or raw['protocol_sha256'] != digest:
        raise ValueError('Incomplete all-layer study')
    if len(cases) != 320 or raw['counts'] != plan['count_limits'] or raw['new_generations'] or raw['native_FFN_calls']:
        raise ValueError('Budget mismatch')
    keys = [(c['source_index'], c['step'], c['block']) for c in cases]
    if len(set(keys)) != 320 or set(keys) != {(i, s, b) for i in [0, 12] for s in range(4) for b in range(40)}:
        raise ValueError('Missing or repeated case')
    if any(c['seed'] != 42 + c['source_index'] for c in cases):
        raise ValueError('Seed mismatch')
    resources = raw['assigned_GPU_resources']
    if not resources['sampled_isolation_pass'] or resources['violations'] or any(
            v['foreign_process_count_on_assigned_GPU'] for v in resources['samples']):
        raise ValueError('Assigned GPU isolation failed')
    snapshot = path.parent / 'EXECUTED_SOURCES'
    if sha256(snapshot / Path(relative).name) != digest or any(
            sha256(snapshot / name) != expected for name, expected in plan['bound_sources'].items()):
        raise ValueError('Executed snapshot mismatch')
    size = sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())
    if size > plan['max_HF_output_gib'] * (1 << 30):
        raise ValueError('Output cap exceeded')
    if len(raw['calibrations']) != 80 or any(c['rows'] != 256 or c['sources'] != [4, 8, 16, 20] for c in raw['calibrations']):
        raise ValueError('Calibration split changed')
    old_path = hf / 'e0_function_basis_20261026/nonlinear_reconstruction_attempt1/RUN.json'
    old = json.loads(old_path.read_text())
    maximum = 0.
    for c in old['cases']:
        current = next(v for v in cases if (v['source_index'], v['step'], v['block']) == (c['source_index'], c['step'], c['block']))
        for method, panels in c['results'].items():
            for panel, metrics in panels.items():
                for metric, value in metrics.items():
                    maximum = max(maximum, abs(current['results'][method][panel][metric] - value))
    if maximum > 1e-10:
        raise ValueError('Original five-block results changed')
    selected = ['calibration_affine', 'calibration_input_ridge', 'calibration_input_token_mismatch',
                'affine_plus_input_ridge', 'affine_plus_input_token_mismatch',
                'whole_g67_ridge', 'whole_g94_ridge', 'affine_residual_g27_ridge', 'affine_residual_g54_ridge']
    projected = [{**c, 'results': {k: v for k, v in c['results'].items() if k in selected}} for c in cases]
    def grouped(field):
        return {str(value): summarize([c for c in projected if c[field] == value], ['uniform_disjoint'])
                for value in sorted({c[field] for c in projected})}
    return {'id': 'E0_ALL_BLOCK_RECONSTRUCTION_20261026', 'status': 'COMPLETE',
            'protocol': relative, 'protocol_sha256': digest, 'analysis_sha256': sha256(Path(__file__)),
            'raw_binding': {'HF_relative': str(path.relative_to(hf)), 'sha256': sha256(path)},
            'actual_counts': raw['counts'], 'cases': 320, 'calibration_rows_per_expert_block': 256,
            'calibration_sources': [4, 8, 16, 20], 'held_sources': [0, 12],
            'actual_seeds_by_source': {str(i): i + 42 for i in [0, 4, 8, 12, 16, 20]},
            'original_five_block_scalar_reproduction_max_abs_difference': maximum,
            'sampled_assigned_GPU_isolation_pass': True, 'HF_output_bytes': size,
            'peak_physical_GPU_MiB': resources['peak_physical_used_memory_MiB'],
            'overall_all_controls': summarize(cases), 'by_block': grouped('block'),
            'by_step': grouped('step'), 'by_source': grouped('source_index'),
            'aggregation': 'Equal-case ratio means; matched calibration/evaluation sources, rows, labels and fixed penalty fraction. All six historical source clusters exposed; not320 independent samples.',
            'candidate_operator_selected': False, 'quality_or_speed_acceptance': False, 'stage_changed': False,
            'new_generations': 0, 'native_FFN_calls': 0, 'optimizer_updates': 0, 'persisted_fitted_weights': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve completed result')
    write_json(args.out, analyze(args.hf_root))
