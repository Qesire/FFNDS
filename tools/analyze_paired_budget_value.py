"""Validate all24 R12 D0 native cases and compute the frozen action decision."""
import argparse
import json
from pathlib import Path

from r12d0_contract import (ROOT, PROTOCOL, INDICES, ACTIONS, completed_cases,
                           decide, read, require_repeats, sha, groups, residual_count)


def analyze(out):
    import torch
    from safetensors.torch import load_file
    from run_ar2_restart import tensor_sha
    from historical_source_snapshot import HistoricalSources
    from restored_assets import verified_restored_assets
    from run_cross_layer_transport import references
    history = HistoricalSources()
    _, _, assets = verified_restored_assets(history)
    refs, ref_binding = references(history, assets)
    identity = read(out / 'control_DQ/RUN.json')['scientific_identity']
    if identity['protocol_sha256'] != sha(PROTOCOL) or identity['assets'] != assets or identity['original_TF2_reports'] != ref_binding:
        raise RuntimeError('Frozen protocol/assets/TF2 reference identity changed')
    for path, expected in identity['sources'].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError('Frozen executed source changed')
    names = ['control_DQ', 'control_UB1', 'baselines_shard0', 'baselines_shard1',
             'interventions_shard0', 'interventions_shard1']
    bindings, cases = completed_cases(out, names, identity)
    if len(cases) != 24:
        raise RuntimeError('Not exactly24 registered cases')
    require_repeats(cases)
    matrix, ub1, repeats, receipts = {}, {}, {}, []
    full_identities = offline_linears = 0
    for index in INDICES:
        matrix[index] = {}
        target = refs['DQ'][index]['payloads'][3]['flow'].double()
        incumbent = refs['UB1'][index]['payloads'][3]['flow'].double()
        e = incumbent - target
        ub1[index] = float(e.square().sum())
        baseline_input = cases[index, 'N0']['scopes'][0]
        for arm in ['DQ', 'UB1', 'N0', 'N0_repeat', 'NA', 'NB']:
            case = cases[index, arm]
            execution, scopes = case['execution'], case['scopes']
            if (case['counter_delta']['linear_forward_calls'] != 1600
                    or case['counter_delta']['activation_qdq_calls'] != 1600
                    or case['counter_delta']['cache_misses'] or len(scopes) != 2
                    or execution['actual_residual_positions'] != residual_count(arm)
                    or execution['pending_residual_predictions']
                    or execution['actual_main_linear_calls'] != 320
                    or execution['actual_global_main_MAC_fraction'] != (1. if arm == 'DQ' else .5)):
                raise RuntimeError('Native model/cost coverage failed')
            for step in range(4):
                row = case['steps'][step]
                ref = refs['DQ'][index]['payloads'][step]
                control = refs['DQ' if arm == 'DQ' else 'UB1'][index]['payloads'][step]
                if row['step'] != step or row['input_sha256'] != tensor_sha(ref['input']):
                    raise RuntimeError('Incoming complete step state mismatch')
                if arm in ('DQ', 'UB1') or step != 3:
                    if row['flow_sha256'] != tensor_sha(control['flow']) or row['post_sha256'] != tensor_sha(control['post']):
                        raise RuntimeError('Original TF2 flow/post control failed')
                for block in range(40):
                    width = groups(arm, step, block) * 64
                    for slot, shape in [(0, [width, 5120]), (2, [5120, width])]:
                        r = execution['main_coverage'][f'step{step}_block{block}_ffn{slot}']
                        if r['actual_weight_shape'] != shape or r['rows'] != 31668:
                            raise RuntimeError('Recorded actual main GEMM shape mismatch')
            for block, scope in zip((30, 31), scopes):
                if (scope['block'] != block or scope['groups'] != groups(arm, 3, block)
                        or not scope['native_gate_residual_bitwise_equal'] or not scope['native_RNG_unchanged']):
                    raise RuntimeError('Actual pair/gate/RNG controls failed')
                calls = scope['offline_teacher_counter_delta']['linear_forward_calls']
                if scope['groups'] == 216:
                    if not scope['full216_original_teacher_bitwise_equal'] or not scope['original_Q1_Q2_bitwise_equal'] or calls != 2:
                        raise RuntimeError('Original full216 numerical identity failed')
                    full_identities += 1
                elif calls:
                    raise RuntimeError('Unexpected offline teacher')
                offline_linears += calls
            if arm in ACTIONS:
                if any(scopes[0][key] != baseline_input[key] for key in ('norm_input_sha256', 'Q1_sha256', 'gate_sha256', 'b0_sha256', 'b2_sha256')):
                    raise RuntimeError('Block30 decision input/Q1/raw bias differs across actions')
            if arm in ('DQ', 'UB1') and not case['original_R7_pair_operands_and_Y_bitwise_equal']:
                raise RuntimeError('Original native pair operand/Y control failed')
            final = case['steps'][3]
            payload = load_file(final['path'], device='cpu')
            if set(payload) != {'flow'} or payload['flow'].dtype != torch.float32 or payload['flow'].shape != target.shape:
                raise RuntimeError('Final full flow geometry/dtype mismatch')
            if tensor_sha(payload['flow']) != final['flow_sha256']:
                raise RuntimeError('Final decoded flow SHA mismatch')
            flow = payload['flow'].double()
            loss = float((flow - target).square().sum())
            delta = flow - incumbent
            signed, penalty = float(-2 * (e * delta).sum()), float(delta.square().sum())
            if abs((ub1[index] - loss) - (signed - penalty)) > 1e-9 * max(1., ub1[index], loss):
                raise RuntimeError('Signed downstream gain identity failed')
            if arm in ACTIONS:
                matrix[index][arm] = loss
            elif arm == 'N0_repeat':
                repeats[index] = loss
            if arm == 'DQ' and loss != 0. or arm == 'UB1' and loss != ub1[index]:
                raise RuntimeError('Original native comparator final loss mismatch')
            job = next(n for n in names if (out / n / 'RUN.json').exists()
                       and any(r['index'] == index and r['arm'] == arm for r in read(out / n / 'RUN.json')['completed']))
            raw_case = next(r for r in read(out / job / 'RUN.json')['completed'] if r['index'] == index and r['arm'] == arm)
            receipts.append({'index': index, 'arm': arm, 'flow_SSE': loss,
                'signed_alignment_gain': signed, 'direction_norm2_penalty': penalty,
                'gain_over_UB1': ub1[index] - loss, 'scopes': scopes,
                'block31_input_changed_vs_N0': scopes[1]['norm_input_sha256'] != cases[index, 'N0']['scopes'][1]['norm_input_sha256'],
                'steps': [{k: v for k, v in s.items() if k != 'path'} for s in case['steps']],
                'case_sha256': raw_case['sha256'], 'raw_locator': f'{job}/input{index:02d}_{arm}',
                'counter_delta': case['counter_delta'],
                'execution': {k: v for k, v in execution.items() if k not in ('main_coverage', 'residual_coverage')},
                'peak_allocated_bytes': case['peak_allocated_bytes'], 'peak_reserved_bytes': case['peak_reserved_bytes'],
                'elapsed_diagnostic_seconds': case['elapsed_seconds'],
                'diagnostic_video_is_DQ_by_construction': True})
    if full_identities != 16 or offline_linears != 32:
        raise RuntimeError('Expected16 original full216 identities/32 separately charged teacher linears')
    decision = decide(matrix, ub1, repeats)
    for name in names:
        job = read(out / name / 'RUN.json')
        if job['raw_placeholder_fallback_calls'] or job['resource_after_case']['free_bytes'] < 80 << 30:
            raise RuntimeError('Fallback/storage resource control failed')
    return {'id': 'R12_D0_NATIVE_EQUAL_BUDGET_RESULT',
        'status': 'COMPLETED_FIXED_R12_D0_NATIVE_VALUE', 'evidence_level': 'FIXED_INPUT_NATIVE_FLOW_ONLY',
        'protocol_sha256': sha(PROTOCOL), 'source_sha256': identity['sources'],
        'original_TF2_reports': ref_binding, 'job_sha256': bindings, 'completed_generations': 24,
        'numeric_and_cost_controls_pass': True, 'repeated_N0_bitwise_equal': True,
        'full216_original_teacher_identities': full_identities, 'offline_teacher_linears': offline_linears,
        'offline_teacher_QDQ_calls': offline_linears, 'action_matrix': matrix, 'frozen_UB1_losses': ub1,
        'N0_repeat_losses': repeats, 'decision': decision, 'rows': receipts,
        'deployment_policy_fitted': False, 'new_training_updates': 0,
        'free_student_quality_or_FFN_speed_acceptance': False,
        'scope': 'Four development-exposed sources; zero pair correction; oracle is post-action and nondeployable. No D1, predictor or rollout trigger.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from ar2_checkpoint import atomic_json
    result = analyze(args.out.resolve())
    if args.output.exists():
        raise RuntimeError('Preserve prior analysis')
    atomic_json(args.output, result)
    print(json.dumps(result['decision'], indent=2))


if __name__ == '__main__':
    main()
