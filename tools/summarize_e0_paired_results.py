"""CPU JSON-only audit of completed paired traces. No tensors or model imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from ffnds_paired_video_strict_gate import inspect


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(study, hf_derived):
    ledger = json.loads((study / 'STUDY.json').read_text())
    if [a['pair'] for a in ledger['attempts']] != ['dc_source00', 'dc_source12']:
        raise ValueError('Exactly the two fixed v3 pairs required')
    runs = []
    for entry in ledger['attempts']:
        out = study / entry['pair']
        run = json.loads((out / 'RUN.json').read_text())
        if (entry['status'] != 'PASS_NATIVE_AND_CORE_METRICS_NOT_QUALITY'
                or run['status'] != entry['status'] or sha(out / 'RUN.json') != entry['RUN_sha256']
                or run['protocol_sha256'] != ledger['protocol_sha256']
                or sha(out / 'NATIVE_RUN.json') != run['native_RUN_sha256']
                or sha(out / run['observation']['path']) != run['observation']['sha256']):
            raise ValueError('Actual fixed receipt/status binding failed')
        failures = inspect(out / 'STRICT_BUNDLE.json')
        if failures:
            raise ValueError('; '.join(failures))
        trace = json.loads((out / 'TRACE.json').read_text())
        if (sha(out / 'TRACE.json') != run['trace_sha256']
                or trace['status'] != 'NATIVE_VECTOR_ACCOUNTING_COMPLETE_NOT_QUALITY_ACCEPTANCE'
                or trace['trajectory_pair_modes'] != ['dense', 'candidate']
                or {(s['step'], s['block']) for s in trace['sites']} != {(i, j) for i in range(4) for j in range(40)}
                or len(trace['sites']) != 160 or len(trace['steps']) != 4):
            raise ValueError('Complete actual native 40x4 coverage required')
        def q_energy(site):
            q = site['input_geometry']
            return q['reference_energy'] + q['difference_SSE'] + 2 * q['reference_difference_dot']
        per_step = []
        for step in range(4):
            sites = [s for s in trace['sites'] if s['step'] == step]
            state = next(s for s in trace['steps'] if s['step'] == step)
            per_step.append({
                'step': step,
                'teacher_operator_relative_to_teacher_energy_median': median(s['FFN']['teacher_operator_SSE'] / s['FFN']['reference_energy'] for s in sites),
                'student_operator_relative_to_teacher_energy_median': median(s['FFN']['student_operator_SSE'] / s['FFN']['reference_energy'] for s in sites),
                'student_to_teacher_operator_SSE_ratio_median': median(s['FFN']['student_operator_SSE'] / s['FFN']['teacher_operator_SSE'] for s in sites),
                'qS_to_qT_energy_ratio_median': median(q_energy(s) / s['input_geometry']['reference_energy'] for s in sites),
                'qT_qS_cosine_median': median((s['input_geometry']['reference_energy'] + s['input_geometry']['reference_difference_dot']) / math.sqrt(s['input_geometry']['reference_energy'] * q_energy(s)) for s in sites),
                'FFN_operator_prefix_cosine_median': median(s['FFN']['operator_prefix_cosine'] for s in sites if s['FFN']['operator_prefix_cosine'] is not None),
                'latent_delta_before_SSE': state['delta_before_SSE'],
                'latent_delta_after_SSE': state['delta_after_SSE'],
                'flow_difference_SSE': state['flow_difference_SSE'],
                'delta_flow_dot': state['delta_flow_dot'],
                'delta_flow_cosine': state['delta_flow_dot'] / math.sqrt(state['delta_before_SSE'] * state['flow_difference_SSE']) if state['delta_before_SSE'] else None,
                'h': state['h'],
                'ideal_positive_flow_energy_term': state['h'] ** 2 * state['flow_difference_SSE'],
                'ideal_signed_cross_term': -2 * state['h'] * state['delta_flow_dot'],
                'actual_energy_change': state['actual_energy_change'],
                'ideal_energy_change': state['ideal_energy_change'],
                'step_post_defect_SSE': state['ideal_update_defect_SSE'],
            })
        vector_bindings = [s['sampled_vectors'] for s in trace['sites']]
        if any(sha(out / v['path']) != v['sha256'] for v in vector_bindings):
            raise ValueError('Saved vector drift')
        bound = ['RUN.json', 'NATIVE_RUN.json', 'OBSERVATION.json', 'TRACE.json',
                 'BUNDLE.json', 'STRICT_BUNDLE.json', 'teacher_metrics.json', 'student_metrics.json']
        runs.append({
            'pair': entry['pair'], 'source_index': trace['source_index'], 'seed': trace['seed'],
            'raw': {'HF_relative': str(out.relative_to(hf_derived)), 'files_sha256': {name: sha(out / name) for name in bound}},
            'sites': 160, 'saved_vector_files_verified': len(vector_bindings),
            'student_operator_SSE_strictly_smaller_sites': sum(s['FFN']['student_operator_SSE'] < s['FFN']['teacher_operator_SSE'] for s in trace['sites']),
            'operator_SSE_equal_sites': [[s['step'], s['block']] for s in trace['sites'] if s['FFN']['student_operator_SSE'] == s['FFN']['teacher_operator_SSE']],
            'per_step': per_step, 'cost': run['cost'], 'cost_scope': run['cost_scope'],
            'GPU_allocated_bytes_before_CPU_metrics': run['GPU_allocated_bytes_before_CPU_metrics'],
            'cost_deduplication_key': run['cost_deduplication_key'],
            'observer_counts': run['observer_counts'], 'native_runtime_counts': run['native_runtime_counts'],
            'sampled_isolation_pass': run['assigned_GPU_resources']['sampled_isolation_pass'],
            'max_signed_energy_identity_relative_gap': max(abs(s[k]['signed_energy_identity_gap']) / max(s[k]['total_SSE'], 1e-30) for s in trace['sites'] for k in ['FFN', 'native_writeback']),
            'native_vs_continuous_32row_teacher_relative_SSE_median': median(s['numerical_check']['teacher_native_vs_continuous']['relative_SSE'] for s in trace['sites']),
            'native_vs_continuous_scope': 'Same deployed BF16 coefficients promoted FP32, no Q2/intermediate BF16. Denominator is continuous candidate energy on32 rows, not dense fullM.',
            'full_depth_site_metric': 'Ungated raw FFN output, plus actual residual/gate writebacks; not old gated FFN panel objective.',
            'core_and_strict_gate_pass': True, 'quality_accepted': False,
        })
    reports = [json.loads((study / r['pair'] / (branch + '_metrics.json')).read_text())
               for r in runs for branch in ['teacher', 'student']]
    frames = [m['metrics']['artifact']['verified_full_decode_frames'] for m in reports]
    native_gemm = sum(r['native_runtime_counts']['linear_forward_calls']
                      + r['observer_counts']['candidate_main_GEMM']
                      + r['observer_counts']['candidate_shadow_GEMM'] for r in runs)
    native_qdq = sum(r['native_runtime_counts']['activation_qdq_calls']
                     + r['observer_counts']['native_QDQ']
                     - r['observer_counts']['dense_main_GEMM']
                     - r['observer_counts']['dense_shadow_GEMM'] for r in runs)
    return {'id': 'E0_PAIRED_FULL_DEPTH_RESULTS_20261026',
            'status': 'COMPLETE_FIXED_V3_NATIVE_OBSERVATIONS_NOT_QUALITY_ACCEPTANCE',
            'protocol': 'config/e0_paired_full_depth_20261026_v3.json',
            'protocol_sha256': ledger['protocol_sha256'], 'STUDY_sha256': sha(study / 'STUDY.json'),
            'analysis_source_sha256': sha(Path(__file__)), 'runs': runs,
            'actual_budget': {'full_generations': 4, 'videos': len(reports), 'decoded_frames': sum(frames),
                             'frame_transitions': sum(f - 1 for f in frames),
                             'paired_sites': 320, 'saved_vector_files': 320,
                             'native_GEMM_all_target_linear': native_gemm,
                             'native_QDQ_all_target_linear': native_qdq,
                             'continuous_FP32_GEMM': sum(r['observer_counts']['continuous_GEMM'] for r in runs),
                             'peak_simultaneous_GPU_cards': 1, 'training_updates': 0,
                             'weight_fits': 0, 'official_quality_calls': 0,
                             'HF_output_bytes': sum(p.stat().st_size for p in study.rglob('*') if p.is_file())},
            'source_scope': 'Two previously exposed sources, seeds42/54; no population generalization.',
            'causal_attribution': 'NOT_IDENTIFIED: per-FFN dense shadow is not whole-model dense flow at student latent.',
            'official_quality': 'NOT_RUN', 'human_full_video_review': 'NOT_RUN',
            'quality_accepted': False, 'stage_promotion': False, 'new_model_jobs_started_by_analysis': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--hf-derived', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.study.resolve(), args.hf_derived.resolve())
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'paired_sites': 320, 'new_model_calls': 0}))
