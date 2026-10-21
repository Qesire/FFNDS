"""Verify all frozen R11T1 controls/fits/native deployment and export public data."""
import argparse
import json
import math
from pathlib import Path

from ar2_checkpoint import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    before = ROOT / 'research/20261021/R11T1_before_execution.json'
    reg = json.loads(before.read_text())['registration']
    protocol = ROOT / reg['protocol']
    frozen = reg['source_sha256']
    for path, sha in frozen.items():
        if digest(ROOT / path) != sha:
            raise RuntimeError('Frozen matched source changed')
    receipts = {}

    def read(name, expected):
        path = args.run / name / 'RUN.json'
        d = json.loads(path.read_text())
        if d['status'] != expected or d['sources'] != frozen or d['protocol_sha256'] != digest(protocol):
            raise RuntimeError('Missing complete frozen job: ' + name)
        receipts[name] = digest(path)
        return d

    rows, collection = {}, []
    for shard in (0, 1):
        d = read(f'collect{shard}', 'PASS_FIXED_WINDOW_CAPTURE_SHARD')
        collection.append(d)
        for row in d['completed']:
            i = row['index']
            if i in rows or digest(row['path']) != row['sha256'] or row['bytes'] > 2.25 * (1 << 30):
                raise RuntimeError('Incomplete/mutated/oversize native capture')
            if (not all(row['zero_update_byte_controls'].values())
                    or not row['teacher']['teacher_Q1_byte_equals_actual']
                    or not row['teacher']['native_RNG_unchanged']
                    or row['teacher']['counter_delta']['linear_forward_calls'] != 2
                    or row['teacher']['counter_delta']['activation_qdq_calls'] != 2):
                raise RuntimeError('Complete unchanged student and same-input teacher controls required')
            if (len(row['steps']) != 4 or any(not all(s[k] for k in ('input_byte_equal', 'flow_byte_equal', 'original_post_byte_equal')) for s in row['steps'])
                    or row['counter_delta']['linear_forward_calls'] != 1600 or row['counter_delta']['activation_qdq_calls'] != 1600
                    or row['tensors']['u']['shape'] != [31668, 5120] or row['tensors']['teacher_y']['shape'] != [31668, 5120]):
                raise RuntimeError('Original full-M/four-step controls required')
            rows[i] = row
    if set(rows) != {0, 5, 10, 15} or not rows[0]['source00_R11G1_repeat_equal']:
        raise RuntimeError('Frozen data membership/repeated source00 changed')
    fitted = {arm: read(f'train_{arm}', 'PASS_FIXED20_PARAMETER_FIT_FROZEN_BEFORE_DEVELOPMENT') for arm in ('local', 'flow')}
    for arm, fit in fitted.items():
        if (len(fit['iterations']) != 20 or fit['training_indices'] != [0, 5, 10] or fit['development_loss_read']
                or digest(fit['parameters_path']) != fit['parameters_sha256']
                or any(s['update']['relative_global_L2_step'] != .001 for s in fit['iterations'])
                or any(fit['capture_receipt_sha256'][str(i)] != receipts[f'collect{i}'] for i in (0, 1))):
            raise RuntimeError('Both equal-budget fits must be frozen before development')
    evaluation = read('evaluate', 'PASS_FIXED_MATCHED_OBJECTIVE_EVALUATION')
    if any(evaluation['fitted_receipt_sha256'][a] != receipts[f'train_{a}'] for a in fitted):
        raise RuntimeError('Development result not bound to both frozen fits')
    for arm in fitted:
        v = read(f'validate_{arm}', 'PASS_UPDATED_COMPLETE_NATIVE_FLOW_BYTE_EQUALS_REPLAY')
        if (v['evaluation_sha256'] != receipts['evaluate'] or v['parameters_sha256'] != fitted[arm]['parameters_sha256']
                or v['native']['index'] != 15 or len(v['native']['steps']) != 4
                or any(not s['input_byte_equal'] or not s['flow_byte_equal'] for s in v['native']['steps'])):
            raise RuntimeError('Actual complete native deployment differs from frozen replay')
    if [r['index'] for r in evaluation['rows']] != [0, 5, 10, 15]:
        raise RuntimeError('Evaluation membership changed')
    for row in evaluation['rows']:
        if not math.isfinite(row['UB1_flow_SSE']) or row['UB1_flow_SSE'] <= 0:
            raise RuntimeError('Invalid frozen UB1 denominator')
        for arm in fitted:
            a = row['arms'][arm]
            if not math.isfinite(a['flow_SSE']) or not math.isclose(a['flow_SSE_over_UB1'], a['flow_SSE'] / row['UB1_flow_SSE'], rel_tol=1e-14):
                raise RuntimeError('Flow SSE ratio mismatch')
    dev = evaluation['rows'][-1]
    over_ub1 = dev['arms']['flow']['flow_SSE'] / dev['UB1_flow_SSE']
    over_local = dev['arms']['flow']['flow_SSE'] / dev['arms']['local']['flow_SSE']
    passed = over_ub1 <= .98 and over_local <= .99
    if passed != evaluation['development_gate']:
        raise RuntimeError('Saved fixed gate mismatch')
    train = evaluation['rows'][:3]
    pooled = {'UB1': sum(r['UB1_flow_SSE'] for r in train),
              **{a: sum(r['arms'][a]['flow_SSE'] for r in train) for a in fitted}}
    curves = {a: [s['train_objective_before_update'] for s in f['iterations']] + [f['final_train_objective']]
              for a, f in fitted.items()}
    descent = {a: {'own_objective_relative_improvement': 1 - c[-1] / c[0],
                   'strictly_decreasing_updates': sum(y < x for x, y in zip(c, c[1:])),
                   'last_update_relative_improvement': 1 - c[-1] / c[-2]} for a, c in curves.items()}
    local_tradeoff = sum(r['arms']['flow']['local_gated_MSE'] for r in train) / 3 / curves['local'][0]
    result = {'id': reg['method_id'], 'status': 'COMPLETED_FIXED20_MATCHED_NATIVE_OBJECTIVE',
              'disposition': 'PASSED_FIXED_OBJECTIVE_SCREEN_NO_AUTOMATIC_ROLLOUT' if passed else 'CLOSED_FIXED20_MATCHED_HEAD_OBJECTIVE_REJECTED',
              'analysis_source_sha256': digest(Path(__file__)), 'before_execution_sha256': digest(before),
              'scientific_identity': {'protocol_sha256': digest(protocol), 'sources': frozen, 'assets': collection[0]['assets'],
                                      'original_policy_sha256': collection[0]['original_policy_sha256'],
                                      'original_TF2_reports': collection[0]['reference_bindings']},
              'raw_job_receipts_sha256': receipts, 'complete_native_generations': 6, 'optimizer_updates_per_arm': 20,
              'original_full_step_byte_comparisons': 48, 'complete_window_zero_update_controls': 16,
              'same_input_teacher_Q1_matches': 4, 'updated_complete_native_deployment_flow_matches': 2,
              'captured_payloads': [{'index': i, 'sha256': r['sha256'], 'bytes': r['bytes'], 'tensors': r['tensors'],
                                     'teacher': r['teacher']} for i, r in sorted(rows.items())],
              'fits': {a: {'parameters_sha256': f['parameters_sha256'], 'parameter_tensor_sha256': f['parameter_tensor_sha256'],
                           'iterations': f['iterations'], 'final_train_objective': f['final_train_objective'],
                           'train_objective_decreased': f['train_objective_decreased'],
                           'peak_allocated_bytes': f['peak_allocated_bytes'], 'elapsed_train_seconds': f['elapsed_train_seconds']} for a, f in fitted.items()},
              'evaluation': evaluation['rows'], 'pooled_train_flow_SSE': pooled,
              'pooled_train_relative_improvement': {a: 1 - pooled[a] / pooled['UB1'] for a in fitted},
              'training_descent_audit': descent,
              'flow_fit_local_train_MSE_over_initial_local_MSE': local_tradeoff,
              'development_flow_over_UB1': over_ub1, 'development_flow_over_local': over_local, 'development_gate': passed,
              'main_MAC_fraction': .5, 'rank': 256, 'additional_inference_parameters': 0, 'changed_positions': [[3, 39]],
              'quality_or_speed_acceptance': False,
              'data_scope': 'TF2 train00/05/10, optimization development15; all4 previously exposed. Whole-step DQ inputs, student within-step states; not free student rollout.',
              'limitations': ['Fixed20 normalized-gradient recipe and one final window only; no optimizer/family impossibility claim.',
                              'FullM data and gradient optimization differ from historical sampled-row OP3; both current arms share those changes. This is not an OP3 rerun or novelty from more rows.',
                              'No hyperparameter/checkpoint/source/layer/rank selection after development.',
                              'No fresh official quality or full160 speed measurement; same inference capacity does not establish quality.',
                              'Forced-DQ-post diagnostic videos are DQ by construction. Updated deployment checks concern complete flow, not candidate video quality.',
                              'Old finite rejected methods remain closed. A pass requires a separate rollout/fresh evaluation protocol.']}
    atomic_json(args.out, result)
    print(json.dumps({k: result[k] for k in ('status', 'disposition', 'development_flow_over_UB1', 'development_flow_over_local', 'development_gate')}))


if __name__ == '__main__':
    main()
