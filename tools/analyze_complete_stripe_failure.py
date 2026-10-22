"""Descriptive saved-data analysis after the fixed R14N0 result.

Only the existing small affine A is evaluated on CPU. No original W1/W2,
quantizer, complete FFN, model generation, training or new selector is run.
"""
import argparse
import json
import math
from pathlib import Path

from r14_stripe_contract import ORDER, ROOT, execution_root, read, sha
from worker_identity import require_worker


def correlation(x, y):
    a, b = math.fsum(x) / len(x), math.fsum(y) / len(y)
    dx, dy = [v - a for v in x], [v - b for v in y]
    norm = math.sqrt(math.fsum(v * v for v in dx) * math.fsum(v * v for v in dy))
    return math.fsum(v * w for v, w in zip(dx, dy)) / norm if norm else None


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [None] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for offset in order[start:end]:
            result[offset] = (start + end - 1) / 2
        start = end
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_worker()
    if args.output.exists():
        raise RuntimeError('Preserve first descriptive decomposition')
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from stripe_teacher_capture import writeback
    from onpolicy_residual import same_bits
    torch.set_num_threads(8)
    protocol_path = ROOT / 'research/20261022/R14N0_saved_failure_analysis_protocol.json'
    protocol = read(protocol_path)
    native_path = ROOT / 'research/20261022/R14N0_native_results.json'
    native = read(native_path)
    if sha(native_path) != protocol['native_result_sha256'] or torch.cuda.is_initialized():
        raise RuntimeError('Fixed saved evidence and CPU-only process required')
    root = execution_root(read(ROOT / 'research/20261022/R14N0_registered_protocol.json'))
    capture = read(root / 'CAPTURE.json')
    entries = {entry['index']: entry for entry in capture['completed']}
    cache = read(ROOT / '.runtime/hif4_rebuilt_cache.json')
    affine_record = next(a for a in read(ROOT / 'research/20261021/R9A1_native_affine_results.json')['affine_matrices']
                         if 2 in a['shared_steps'])
    matrix_path = Path(cache['path']).parents[1] / affine_record['HF_derived_relative']
    if sha(matrix_path) != protocol['frozen_affine_sha256']:
        raise RuntimeError('Original affine changed')
    matrix = load_file(str(matrix_path), device='cpu')['matrix_BF16']
    double_matrix = matrix.double()
    scopes = []
    with torch.inference_mode():
        for index in ORDER:
            entry = entries[index]
            if sha(entry['report']) != entry['sha256']:
                raise RuntimeError('New complete-stripe source receipt changed')
            scope = read(entry['report'])['scope']
            if sha(scope['path']) != scope['sha256']:
                raise RuntimeError('Saved complete-stripe payload changed')
            data = load_file(scope['path'], device='cpu')
            native_case = next(case for case in native['cases'] if case['index'] == index)
            x = data['V1'].reshape(64, 4, 5120)
            truth = data['TeacherY'].reshape_as(x).double()
            gate = data['gate'].reshape(1, 1, 5120).double()
            residual = data['Residual']
            infer_dtype = getattr(torch, scope['infer_dtype'].split('.')[-1])
            sensitive_dtype = getattr(torch, scope['sensitive_dtype'].split('.')[-1])
            actual_student_post = writeback(residual, data['StudentY'], data['gate'], infer_dtype, sensitive_dtype)
            if not same_bits(actual_student_post, data['NativePostStudent']):
                raise RuntimeError('CPU exact elementwise writeback differs from saved native student')
            actual_teacher_post = writeback(residual, data['TeacherY'], data['gate'], infer_dtype, sensitive_dtype)
            baseline = float((actual_teacher_post.double() - actual_student_post.double()).square().sum())
            if not math.isclose(baseline, math.fsum(native_case['UB1_SSE']), rel_tol=1e-12, abs_tol=1e-8):
                raise RuntimeError('Saved native UB1 SSE does not reproduce on CPU')
            centered_x = x.double() - x.double().mean(1, keepdim=True)
            real_contrast = F.linear(centered_x.reshape(256, 5120), double_matrix).reshape_as(x)
            real_mean_response_reference = real_contrast + truth.mean(1, keepdim=True)
            gated_reference_loss = float(((real_mean_response_reference - truth) * gate).square().sum())
            # An exact minimum only in this smooth additive-post reference class.
            # Native BF16 before-gate rounding does not obey that same class.
            predicted_post_contrast = residual.double().reshape_as(x) + real_contrast * gate
            difference = actual_teacher_post.double().reshape_as(x) - predicted_post_contrast
            smooth_common_post = predicted_post_contrast + difference.mean(1, keepdim=True)
            smooth_common_post_loss = float((smooth_common_post - actual_teacher_post.double().reshape_as(x)).square().sum())
            a, b, c, e = x.float().unbind(1)
            h = torch.stack(((a + b - c - e) * .5, (a - b + c - e) * .5,
                              (a - b - c + e) * .5), 1).bfloat16()
            ah = F.linear(h.reshape(192, 5120), matrix).reshape(64, 3, 5120).float()
            aa, bb, cc = ah.unbind(1)
            own = torch.stack(((aa + bb + cc) * .5, (aa - bb - cc) * .5,
                               (-aa + bb - cc) * .5, (-aa - bb + cc) * .5), 1).bfloat16()
            teacher_mean = data['TeacherY'].reshape_as(x).float().mean(1).bfloat16()
            y_reference = (own + teacher_mean[:, None]).reshape(256, 5120)
            post_reference = writeback(residual, y_reference, data['gate'], infer_dtype, sensitive_dtype)
            cpu_BF16_reference_loss = float((post_reference.double() - actual_teacher_post.double()).square().sum())
            benefit = [one - two for one, two in zip(native_case['one_point_SSE'], native_case['two_point_SSE'])]
            scores = native_case['pre_W1_contrast_norm2']
            analyzed = next(row for row in native['analysis']['per_source'] if row['index'] == index)
            scopes.append({'index': index, 'role': analyzed['role'], 'saved_payload_sha256': scope['sha256'],
                'actual_native_UB1_SSE': baseline,
                'FP64_affine_contrast_exact_teacher_mean_gated_reference_SSE': gated_reference_loss,
                'FP64_smooth_additive_common_post_reference_SSE': smooth_common_post_loss,
                'CPU_BF16_three_direction_teacher_mean_post_reference_SSE': cpu_BF16_reference_loss,
                'CPU_BF16_reference_over_UB1': cpu_BF16_reference_loss / baseline,
                'FP64_smooth_reference_over_UB1': smooth_common_post_loss / baseline,
                'extra_point_beneficial_groups_of64': sum(value > 0 for value in benefit),
                'extra_point_harmful_groups_of64': sum(value < 0 for value in benefit),
                'pre_W1_norm_vs_actual_incremental_benefit_Pearson': correlation(scores, benefit),
                'pre_W1_norm_vs_actual_incremental_benefit_Spearman': correlation(ranks(scores), ranks(benefit)),
                'dynamic_oracle_mask_overlap_of32': len(set(analyzed['dynamic_two_point_stripes']) & set(analyzed['oracle_two_point_stripes'])),
                'mean_incremental_benefit': math.fsum(benefit) / 64,
                'CPU_student_native_post_bitwise': True})
            print(f'R14N0 savedCPU source{index:02d} reference and signed-value decomposition complete', flush=True)
    summaries = []
    for name, indices in (('training', (1, 11)), ('old_nonfit_development', (0, 10)), ('all_exposed', ORDER)):
        chosen = [scope for scope in scopes if scope['index'] in indices]
        baseline = math.fsum(scope['actual_native_UB1_SSE'] for scope in chosen)
        cpu = math.fsum(scope['CPU_BF16_three_direction_teacher_mean_post_reference_SSE'] for scope in chosen)
        smooth = math.fsum(scope['FP64_smooth_additive_common_post_reference_SSE'] for scope in chosen)
        summaries.append({'cohort': name, 'UB1_SSE': baseline, 'CPU_BF16_teacher_mean_post_reference_SSE': cpu,
                          'CPU_BF16_teacher_mean_reference_over_UB1': cpu / baseline,
                          'FP64_smooth_common_post_reference_SSE': smooth,
                          'FP64_smooth_common_post_reference_over_UB1': smooth / baseline})
    if torch.cuda.is_initialized():
        raise RuntimeError('Saved CPU analysis initialized CUDA')
    result = {'id': 'R14N0_SAVED_CONTRAST_MEAN_AND_SIGNED_VALUE_ANALYSIS',
              'status': 'DESCRIPTIVE_SAVED_DATA_CPU_AFFINE_ONLY_NOT_NEW_NATIVE_ACTION',
              'protocol_sha256': sha(protocol_path), 'source_sha256': sha(__file__),
              'native_result_sha256': sha(native_path), 'original_affine_sha256': sha(matrix_path),
              'scopes': scopes, 'summary': summaries, 'CPU_existing_affine_projection_calls': 8,
              'CUDA_initialized': False, 'new_model_generations': 0, 'new_native_FFN_calls': 0,
              'new_original_W1_W2_calls': 0, 'new_quantizer_calls': 0, 'new_fitted_parameters': 0,
              'limits': 'Post-hoc descriptive CPU references on all4alreadyexposed sources. The teacher mean is an unavailable design fixture, not a policy. CPU192/256-row affine GEMMs are not the originalGPU23751-row projection; smooth additive-post minimum is not a bound on the BF16 native action class. Correlations are descriptive, not a new validated selector. No changed masks, signs or native calls.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
