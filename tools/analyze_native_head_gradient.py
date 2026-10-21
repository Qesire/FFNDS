"""Verify R11G1 receipts and export only public scientific metadata."""
import argparse
import json
import math
from pathlib import Path

from ar2_checkpoint import atomic_json, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    before_path = ROOT / 'research/20261021/R11G1_before_execution.json'
    registration = json.loads(before_path.read_text())['registration']
    protocol = ROOT / registration['protocol']
    capture_path, gradient_path = [args.run / phase / 'RUN.json' for phase in ('capture', 'gradient')]
    capture, gradient = [json.loads(p.read_text()) for p in (capture_path, gradient_path)]
    bytes_path = args.run / 'STRICT_BYTES.json'
    exact = json.loads(bytes_path.read_text())
    if (exact['status'] != 'PASS_COMPLETE_NATIVE_INTEGER_BYTES'
            or exact['capture_sha256'] != digest(capture_path)
            or exact['source_sha256'] != digest(ROOT / 'tools/verify_native_head_bytes.py')
            or any(not c['equal_integer_bytes'] or c['replay_sha256'] != c['native_sha256'] for c in exact['checks'].values())):
        raise RuntimeError('Strict native integer-byte/SHA controls required')
    if (capture['status'] != 'PASS_FULL_NATIVE_HEAD_CAPTURE'
            or gradient['status'] != 'PASS_NATIVE_HEAD_GRADIENT_REACHABILITY'
            or digest(capture_path) != gradient['capture_sha256']):
        raise RuntimeError('Both complete bound native receipts required')
    for report in (capture, gradient):
        if report['protocol_sha256'] != digest(protocol) or report['source_sha256'] != registration['source_sha256']:
            raise RuntimeError('Frozen protocol/source mismatch')
    for file, sha in registration['source_sha256'].items():
        if digest(ROOT / file) != sha:
            raise RuntimeError('Frozen source changed')
    if digest(capture['payload_path']) != capture['payload_sha256']:
        raise RuntimeError('Complete HF payload changed')
    if not all(gradient['controls'].values()) or len(gradient['controls']) != 5:
        raise RuntimeError('Complete zero-update boundary controls required')
    if [s['step'] for s in capture['steps']] != list(range(4)) or any(
            not all(s[k] for k in ('input_bitwise_equal', 'flow_bitwise_equal', 'post_bitwise_equal')) for s in capture['steps']):
        raise RuntimeError('All original full-step controls required')
    if (gradient['additional_inference_parameters'] != 0 or gradient['inference_parameter_count'] != 2626816
            or gradient['update']['relative_global_L2_step'] != .001
            or sum(gradient['update']['BF16_changed_elements'].values()) <= 0
            or gradient['actual_flow_change_SSE'] <= 0):
        raise RuntimeError('Fixed deployed parameter/update contract changed')
    for row in gradient['gradients'].values():
        if not row['connected'] or not row['finite'] or row['nonzero_count'] <= 0 or not math.isfinite(row['L2']):
            raise RuntimeError('Finite connected gradient required')
    improvement = 1 - gradient['one_fixed_update_flow_MSE'] / gradient['initial_flow_MSE']
    if not math.isfinite(improvement) or not math.isclose(improvement, gradient['one_fixed_update_relative_improvement'], abs_tol=1e-15):
        raise RuntimeError('Loss arithmetic mismatch')
    delta = capture['counter_delta']
    if delta['linear_forward_calls'] != 1600 or delta['activation_qdq_calls'] != 1600 or delta['cache_misses']:
        raise RuntimeError('Original native linear/QDQ coverage changed')
    public_gradient = {k: gradient[k] for k in (
        'controls', 'gradients', 'initial_flow_MSE', 'one_fixed_update_flow_MSE',
        'one_fixed_update_relative_improvement', 'actual_flow_change_SSE', 'update',
        'fullM_forward_backward_seconds', 'updated_forward_seconds', 'peak_allocated_bytes',
        'peak_reserved_bytes', 'inference_parameter_count', 'additional_inference_parameters')}
    result = {'id': registration['method_id'], 'status': 'COMPLETED_FIXED_NATIVE_HEAD_GRADIENT_READINESS',
              'disposition': 'COMPLETED_READINESS_ONLY_NO_TRAINING_TRIGGER',
              'analysis_source_sha256': digest(Path(__file__)), 'before_execution_sha256': digest(before_path),
              'scientific_identity': {'protocol_sha256': digest(protocol), 'sources': registration['source_sha256'],
                                      'assets': capture['assets'], 'original_policy_sha256': capture['original_policy_sha256'],
                                      'original_TF2_report_sha256': capture['reference_bindings']},
              'raw_receipts_sha256': {'capture': digest(capture_path), 'gradient': digest(gradient_path)},
              'strict_byte_audit': exact, 'strict_byte_audit_receipt_sha256': digest(bytes_path),
              'payload': {'storage': 'HF cache; private location omitted', 'sha256': capture['payload_sha256'],
                          'bytes': capture['payload_bytes'], 'tensors': capture['tensors']},
              'completed_pipeline_generations': 1, 'completed_gradient_steps': 1,
              'original_native_control_comparisons': 12, 'native_steps': capture['steps'],
              'counter_delta': delta, 'gradient_readiness': public_gradient,
              'one_exposed_source_loss_decreased': improvement > 0, 'quality_or_speed_acceptance': False,
              'scope': 'Original TF2 source00, step3/block39, full31668 rows. DQ states restored only between full steps. Not student free rollout or fresh validation.',
              'timing_scope': 'One isolated fullM zero-control/forward/backward plus one updated forward; first timing includes identity checks and extra correction prediction. Not warmup-controlled throughput or full160 timing.',
              'limitations': ['One fixed gradient step on one already exposed source; not a candidate fit or generalization result.',
                              'Local-vs-native matched training, free student rollout, fresh official quality and complete speed remain unrun.',
                              'All prior finite closures retained; no automatic rank/width/position/LR/step expansion.',
                              'No gradient through HiF4 Q1 or fixed main. BF16 cast derivatives are judged only by this actual rounded update.']}
    atomic_json(args.out, result)
    print(json.dumps({'status': result['status'], 'loss_relative_improvement': improvement,
                      'peak_allocated_bytes': gradient['peak_allocated_bytes']}))


if __name__ == '__main__':
    main()
