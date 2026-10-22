"""Test spatial sharing of a frozen affine remainder on saved native outputs.

Evaluates existing R9 affine maps on CPU only. No original FFN, quantizer,
new fit, model pipeline or CUDA operation is run.
"""
import argparse
import json
import math
from pathlib import Path
import time

from audit_spatial_residual_coherence import ROOT, pairs

PROTOCOL = ROOT / 'research/20261022/R13_anchored_affine_audit_protocol.json'


def stripe_reference_error(directed_pair_error_sum, n, m, group_size=4):
    if not 1 < n <= m or group_size < 2:
        raise ValueError('Distinct-pair sampling and a nontrivial stripe required')
    pi2 = n * (n - 1) / (m * (m - 1))
    return directed_pair_error_sum / (group_size * pi2)


def aggregate(scopes):
    old = math.fsum(s['reference_UB1_SSE'] for s in scopes)
    shared = math.fsum(s['reference_random_anchor_BF16_SSE'] for s in scopes)
    affine = math.fsum(s['reference_affine_only_SSE'] for s in scopes)
    return {'sources': len(scopes), 'observed_undirected_stripe_pairs': sum(s['stripe_pairs'] for s in scopes),
        'reference_UB1_SSE': old, 'reference_random_anchor_BF16_SSE': shared,
        'reference_affine_only_SSE': affine, 'reference_random_anchor_over_UB1': shared / old,
        'scope': 'Descriptive inclusion-weighted estimate on saved trajectories, not a complete observed stripe or native rollout; sparse pair coverage can give high variance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scalar-out', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.scalar_out.exists():
        raise RuntimeError('Preserve the first anchored-affine audit')
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from ar2_checkpoint import atomic_json, digest
    from audit_r12_saved_operands import verified_safetensors
    from run_ar2_restart import tensor_sha
    from ub1_native_data import load_ub1_collections
    torch.set_num_threads(8)
    began = time.monotonic()
    protocol, protocol_sha = json.loads(PROTOCOL.read_text()), digest(PROTOCOL)
    r9_path = ROOT / 'research/20261021/R9A1_native_affine_results.json'
    r9 = json.loads(r9_path.read_text())
    cache = json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())
    cache_root = Path(cache['path']).parents[1]
    affines, affine_bindings = {}, []
    names = {'matrix': 'matrix_BF16', 'bias': 'bias_total_BF16', 'slope': 'slope_FP32', 'intercept': 'intercept_FP32'}
    for original in r9['affine_matrices']:
        path = cache_root / original['HF_derived_relative']
        if digest(path) != original['sha256']:
            raise RuntimeError('Original R9 affine bytes changed')
        data = load_file(str(path), device='cpu')
        if any(tensor_sha(data[name]) != original['tensor_sha256'][key] for key, name in names.items()):
            raise RuntimeError('Original affine decoded identity differs')
        for step in original['shared_steps']:
            affines[step] = (data, original)
        affine_bindings.append({'locator': original['HF_derived_relative'], 'sha256': original['sha256'],
                                'tensor_sha256': original['tensor_sha256'], 'original_training_indices': original['training_indices']})
    records, bindings, identity = load_ub1_collections(
        [ROOT / f'reports/worker_20261020/OP2_UB1_calibration24_shard{s}_RUN.json' for s in (0, 1)], ROOT)
    scopes, scalar_rows = [], []
    for step, block in protocol['positions']:
        affine, original = affines[step]
        for index in sorted(records):
            if digest(PROTOCOL) != protocol_sha:
                raise RuntimeError('CPU audit scope changed')
            scope = next(s for s in records[index]['scopes'] if (s['step'], s['block']) == (step, block))
            if scope['teacher_shard_sha256'] != original['teacher_shard_sha256']:
                raise RuntimeError('Affine and saved teacher use different FFN weights')
            verified_safetensors(scope['path'], scope['sha256'])
            data = load_file(scope['path'], device='cpu')
            gate, teacher, student = data['gate'].reshape(1, 5120), data['TeacherY'], data['StudentY']
            if data['V1'].shape != (256, 5120) or data['V1'].dtype != torch.bfloat16:
                raise RuntimeError('Unexpected original post-Q1 input')
            linear = F.linear(data['V1'], affine['matrix_BF16'], affine['bias_total_BF16'])
            residual = teacher.double() - linear.double()
            encoded = (teacher.float() - linear.float()).bfloat16()
            truth = (teacher * gate).double()
            old_error = (truth - (student * gate).double()).square().sum(1)
            affine_error = (truth - (linear * gate).double()).square().sum(1)
            local = {'near': [], 'stripe': []}
            for graph in local:
                for a, b in pairs(scope['selected_rows'], scope['grid'], graph):
                    ideal = float(((residual[a] - residual[b]) * gate[0].double()).square().sum())
                    for donor, target in ((a, b), (b, a)):
                        candidate = linear[target] + encoded[donor]
                        loss = float(((candidate * gate[0]).double() - truth[target]).square().sum())
                        row = {'index': index, 'step': step, 'graph': graph,
                            'donor_native_row': scope['selected_rows'][donor], 'target_native_row': scope['selected_rows'][target],
                            'ideal_remainder_difference_SSE': ideal, 'BF16_anchored_SSE': loss,
                            'UB1_SSE': float(old_error[target])}
                        local[graph].append(row)
                        scalar_rows.append(row)
            m, n = scope['full_rows'], scope['rows']
            scopes.append({'index': index, 'step': step, 'block': block,
                'R9_fit_source': index in original['training_indices'], 'original_payload_sha256': scope['sha256'],
                'affine_output_sha256': tensor_sha(linear), 'residual_BF16_sha256': tensor_sha(encoded),
                'stripe_pairs': len(local['stripe']) // 2, 'near_pairs': len(local['near']) // 2,
                'near_BF16_copy_SSE': math.fsum(r['BF16_anchored_SSE'] for r in local['near']),
                'near_UB1_SSE': math.fsum(r['UB1_SSE'] for r in local['near']),
                'reference_UB1_SSE': float(old_error.sum()) * m / n,
                'reference_affine_only_SSE': float(affine_error.sum()) * m / n,
                'reference_random_anchor_BF16_SSE': stripe_reference_error(math.fsum(r['BF16_anchored_SSE'] for r in local['stripe']), n, m),
                'reference_random_anchor_FP64_SSE': stripe_reference_error(math.fsum(r['ideal_remainder_difference_SSE'] for r in local['stripe']), n, m)})
            print(f'CPU anchored-affine step{step} source{index:02d}: near{scopes[-1]["near_pairs"]}, stripe{scopes[-1]["stripe_pairs"]}', flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU affine audit initialized CUDA')
    args.scalar_out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.scalar_out, {'protocol_sha256': protocol_sha, 'rows': scalar_rows})
    source_files = ['tools/audit_anchored_affine_remainder.py', 'tools/audit_spatial_residual_coherence.py',
                    'tools/ub1_native_data.py', 'tools/affine_remainder.py']
    result = {'id': protocol['id'], 'status': 'COMPLETED_CPU_SAVED_ANCHORED_AFFINE_REMAINDER',
        'evidence_level': 'CPU_FROZEN_AFFINE_ON_SAVED_NATIVE_OPERANDS_LOCAL_ONLY',
        'protocol_sha256': protocol_sha, 'source_sha256': {p: digest(ROOT / p) for p in source_files},
        'original_R9_result_sha256': digest(r9_path), 'original_affine_bindings': affine_bindings,
        'original_collection_bindings': [{'record': str(Path(b['path']).relative_to(ROOT)), 'sha256': b['sha256']} for b in bindings],
        'scopes': scopes,
        'summary': [{'step': s, 'cohort': cohort, **aggregate([r for r in scopes if r['step'] == s and
                     (cohort == 'all_exposed24' or r['R9_fit_source'] == (cohort == 'original_R9_fit_odd'))])}
                    for s in range(4) for cohort in ('all_exposed24', 'original_R9_fit_odd', 'nonfit_even_development')],
        'scalar_evidence': {'locator': str(args.scalar_out.relative_to(ROOT)), 'sha256': digest(args.scalar_out), 'rows': len(scalar_rows)},
        'new_model_generations': 0, 'new_training_updates': 0, 'new_large_tensor_writes': 0,
        'CPU_existing_affine_linear_calls': len(scopes), 'CUDA_initialized': False,
        'all_projection_MAC_fraction_for_quarter_anchors': .25 + 5120 / (2 * 13824),
        'elapsed_seconds': time.monotonic() - began, 'limitations': protocol['limits']}
    atomic_json(args.output, result)
    print('Completed frozen-affine spatial remainder audit;0 new model generations.', flush=True)


if __name__ == '__main__':
    main()
