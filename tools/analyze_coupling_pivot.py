"""Offline coupling discriminators: recorded scalars, headers and routing masks.

No model, weight tensor, learned evaluator, Torch import or CUDA call. Normal-
scale examples are synthetic counterexamples, separate from native evidence.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'research/20261023/C0_coupling_offline_results.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equivalent(a, b):
    if type(a) is float and type(b) is float:
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def dot(x, y):
    return math.fsum(a * b for a, b in zip(x, y))


def ln(x, eps=1e-6):
    mean = math.fsum(x) / len(x)
    centered = [v - mean for v in x]
    scale = math.sqrt(dot(centered, centered) / len(x) + eps)
    return [v / scale for v in centered]


def bf16_bits(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16


def scale_counterexample():
    u, v = [1., -1., 1., -1.], [1., 1., -1., -1.]
    outputs = {'dense': v, 'A': [a + b for a, b in zip(u, v)],
               'B': [1.5 * b for b in v]}
    rows = []
    for scale in (1., 4.):
        x = [scale * a for a in u]
        downstream = {k: ln([a + b for a, b in zip(x, f)]) for k, f in outputs.items()}
        losses = {k: math.fsum((a - b) ** 2 for a, b in zip(downstream[k], downstream['dense']))
                  for k in ('A', 'B')}
        rows.append({'residual_scale': scale,
                     'current_normalized_BF16_bits': [bf16_bits(a) for a in ln(x)],
                     'next_normalized_SSE': losses, 'best_action': min(losses, key=losses.get)})
    assert rows[0]['current_normalized_BF16_bits'] == rows[1]['current_normalized_BF16_bits']
    assert [r['best_action'] for r in rows] == ['B', 'A']
    # At a fixed normalized direction, a pure common 1/scale^2 loss factor
    # cannot change an equal-cost action ranking. Losing scale alone is not
    # sufficient: the fixed residual update rotates the next normalization.
    common_scale = [{'scale': s, 'A': 4 / s ** 2, 'B': 1 / s ** 2} for s in (1., 4.)]
    assert all(r['B'] < r['A'] for r in common_scale)
    return {'evidence': 'SYNTHETIC_FINITE_RESIDUAL_PLUS_LAYER_NORM_COUNTEREXAMPLE',
            'u': u, 'v': v, 'fixed_gated_FFN_outputs': outputs, 'epsilon': 1e-6,
            'rows': rows, 'common_scalar_metric_negative_control': common_scale,
            'scope': 'Same BF16 normalized input at unit modulation/smoothing; hence any deterministic Q1/R256 sees identical input. Changed pre-norm scale plus a fixed residual update changes the next norm direction and reverses finite-action ranking. No Wan frequency, predictor advantage or full-video observation.'}


def compensation_controls():
    # P is a fixed-feature least-squares projection in output/sample space.
    # Fit(T - D_S) is linear even though its coefficients depend on S.
    target = [4., 2., 3., -1.]
    groups = [[1., 2., 0., 1.], [2., -1., 1., 2.]]
    project = lambda x: [x[0], x[1], 0., 0.]
    outputs = {}
    for bits in itertools.product((0, 1), repeat=2):
        main = [math.fsum(bits[g] * groups[g][j] for g in range(2)) for j in range(4)]
        correction = project([a - b for a, b in zip(target, main)])
        outputs[bits] = [a + b for a, b in zip(main, correction)]
    xi = [outputs[1, 1][j] - outputs[1, 0][j] - outputs[0, 1][j] + outputs[0, 0][j]
          for j in range(4)]
    assert dot(xi, xi) == 0
    deltas = [[outputs[bits][j] - outputs[0, 0][j] for j in range(4)] for bits in ((1, 0), (0, 1))]
    error = [a - b for a, b in zip(outputs[0, 0], target)]
    def gain(bits):
        return dot(error, error) - math.fsum((a - b) ** 2 for a, b in zip(outputs[bits], target))
    interaction = gain((1, 1)) - gain((1, 0)) - gain((0, 1))
    expected = -2 * dot(*deltas)
    assert interaction == expected
    # A deliberately nonlinear, set-dependent compensation is a separate
    # positive control; it must not be attributed to GELU coordinate coupling.
    nonlinear_outputs = {bits: [value - (3. if bits == (1, 1) and j == 0 else 0.)
                                for j, value in enumerate(y)] for bits, y in outputs.items()}
    nonlinear_xi = [nonlinear_outputs[1, 1][j] - nonlinear_outputs[1, 0][j]
                    - nonlinear_outputs[0, 1][j] + nonlinear_outputs[0, 0][j] for j in range(4)]
    assert dot(nonlinear_xi, nonlinear_xi) == 9
    return {'evidence': 'SYNTHETIC_EXACT_FIXED_PROJECTION_ALGEBRA',
            'fixed_projection_output_mixed_difference': xi,
            'squared_loss_interaction': interaction, 'negative_twice_action_inner_product': expected,
            'nonlinear_set_compensation_control_Xi_norm2': dot(nonlinear_xi, nonlinear_xi),
            'deduction': 'A fixed-feature exact OLS refit remains additive: Y(S)=P*T+(I-P)*D_S. Mask-dependent compensation alone does not imply nonadditivity. Mask-dependent features/alpha/bias, coefficient encoding and native accumulation can break this identity.',
            'sign_convention': 'Native Y=M+R; increment=D_g+R_new-R_old. TACR C denotes compensation REMOVED from frozen UB1, so C=R_old-R_new and increment=D_g-delta(C).'}


def r7_readout():
    source = ROOT / 'research/20261021/R7D1_results.json'
    original = read(source)
    rows = []
    for r in original['rows']:
        linear, curved = r['linear_direction_cross_term'], r['nonlinear_transport_term']
        residual = r['interaction_advantage'] - linear - curved
        assert abs(residual) <= r['identity_absolute_tolerance']
        sum_gain = r['flow_SSE_reduction']['A'] + r['flow_SSE_reduction']['B']
        additive_norm = r['A_flow_change_norm2'] + r['B_flow_change_norm2'] - linear
        rows.append({'source_index': r['index'], 'gain_A_plus_gain_B': sum_gain,
                     'gain_AB': r['flow_SSE_reduction']['AB'],
                     'interaction_advantage': r['interaction_advantage'],
                     'linear_cross_term': linear, 'finite_nonlinear_transport_term': curved,
                     'curvature_norm2': r['curvature_norm2'], 'additive_response_norm2': additive_norm,
                     'curvature_norm2_over_additive_response_norm2': r['curvature_norm2'] / additive_norm,
                     'absolute_interaction_over_sum_single_gains': abs(r['interaction_advantage']) / abs(sum_gain),
                     'cross_and_curvature_cancellation_fraction': 1 - abs(linear + curved) / (abs(linear) + abs(curved)),
                     'local_downstream_rank_disagreement': r['local_downstream_rank_disagreement'],
                     'identity_residual': residual})
    sums = {k: math.fsum(r[k] for r in rows) for k in
            ('gain_A_plus_gain_B', 'gain_AB', 'interaction_advantage', 'linear_cross_term',
             'finite_nonlinear_transport_term', 'curvature_norm2', 'additive_response_norm2')}
    sums['curvature_norm2_over_additive_response_norm2'] = sums['curvature_norm2'] / sums['additive_response_norm2']
    sums['interaction_over_sum_single_gains'] = sums['interaction_advantage'] / sums['gain_A_plus_gain_B']
    sums['cross_and_curvature_cancellation_fraction'] = 1 - abs(sums['interaction_advantage']) / (
        abs(sums['linear_cross_term']) + abs(sums['finite_nonlinear_transport_term']))
    return {'evidence': 'NEW_SCALAR_REANALYSIS_OF_PUBLISHED_NATIVE_R7_NOT_NEW_FORWARD',
            'source_path': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
            'rows': rows, 'pooled': sums,
            'scope': 'Four already exposed sources, step3 blocks30/31, full-teacher restorations with dense resets BETWEEN steps. No affordable group action, free student video, infinitesimal Jacobian or cross-step claim.'}


def tensor_header(path):
    with Path(path).open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        if size > 1 << 20:
            raise ValueError('Unexpected safetensors header length')
        header = json.loads(stream.read(size))
    return {k: {'dtype': v['dtype'], 'shape': v['shape']} for k, v in header.items() if k != '__metadata__'}


def asset_audit(hf):
    base = ROOT / 'runs/worker_restart_20261020/op2_ub1_calibration24'
    entries, bindings, headers = [], [], []
    for shard in (0, 1):
        path = base / f'shard{shard}/RUN.json'
        run = read(path)
        assert run['status'] == 'PASS_OP2_UB1_CALIBRATION_SHARD'
        bindings.append({'record': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        for entry in run['completed']:
            case_path = Path(entry['input_report'])
            assert sha(case_path) == entry['input_report_sha256']
            case = read(case_path)
            assert len(case['scopes']) == 160
            entries.extend((case['index'], scope) for scope in case['scopes'])
    def inspect(item):
        index, scope = item
        path = Path(scope['path'])
        if not path.exists():
            return {'missing': True}
        schema = tensor_header(path)
        expected = {'V1', 'BaseY', 'TeacherY', 'StudentY', 'b2', 'gate'}
        assert set(schema) == expected
        assert all(x['dtype'] == 'BF16' for x in schema.values())
        return {'missing': False, 'schema': schema}
    with ThreadPoolExecutor(max_workers=8) as pool:
        headers = list(pool.map(inspect, entries))
    schemas = Counter(json.dumps(h['schema'], sort_keys=True) for h in headers if not h['missing'])
    r7 = read(ROOT / 'research/20261021/R7D1_results.json')
    r7_payloads = []
    for row in r7['rows']:
        for arm, payload in row['payloads'].items():
            rel = payload['HF_derived_relative']
            candidates = [hf / 'ffnds-derived' / rel, hf / 'derived/ffnds' / rel]
            exists = next((p for p in candidates if p.exists()), None)
            r7_payloads.append({'source_index': row['index'], 'arm': arm,
                                'available_in_audited_HF_roots': exists is not None,
                                'original_sha256': payload['sha256']})
    return {'evidence': 'ACTUAL_SAVED_ASSET_HEADER_AND_RECEIPT_AUDIT', 'OP2_run_bindings': bindings,
            'case_receipts_hash_checked': 24, 'declared_scopes': len(entries),
            'payload_headers_inspected': len(headers) - sum(h['missing'] for h in headers),
            'missing_payloads': sum(h['missing'] for h in headers),
            'unique_schemas': [{'schema': json.loads(k), 'count': v} for k, v in schemas.items()],
            'full_OP2_payload_hashes_replayed': False,
            'pre_norm_residual_or_attention_update_saved': False,
            'new_complete_group_action_finite_continuation_labels_saved': False,
            'E1_native_conditional_information_result': None, 'R7_raw_payload_availability': r7_payloads,
            'R7_four_arm_vector_replay_performed': False}


def routing_readout(hf):
    import numpy as np
    root = hf / 'derived/ffnds/R17E0_fresh_quality'
    report = read(root / 'GENERATIONS.json')
    mask_rows, masks = [], []
    for source in range(5):
        path = root / f'quality{source:02d}_dynamic_residual/REQUEST.json'
        request = read(path)
        binding = next(x for x in report['completed'] if x['job']['name'] == request['job']['name'])
        assert sha(path) == binding['request_sha256']
        trace = request['routing_trace']; payload = hf / trace['HF_relative']
        assert sha(payload) == trace['sha256']
        with np.load(payload, allow_pickle=False) as data:
            assert tuple(data['grid']) == (21, 29, 52)
            unpacked = np.unpackbits(data['packed'], axis=-1, bitorder='little')
        assert not unpacked[..., 31668:].any()
        mask = unpacked[..., :31668].astype(bool).reshape(4, 40, 21, 29, 52)
        assert np.all(mask.sum(axis=(-3, -2, -1)) == 7917)
        masks.append(mask)
        fraction = mask.mean(axis=(-2, -1))
        mask_rows.append({'source_index': source, 'stratum': 'i2v_subject', 'request_sha256': sha(path),
                          'trace_sha256': sha(payload), 'first_bin_dense_fraction': float(fraction[..., 0].mean()),
                          'later20_dense_fraction': float(fraction[..., 1:].mean()),
                          'first_bin_by_step': fraction[..., 0].mean(axis=1).tolist(),
                          'adjacent_time_bin_flip_fraction': float(np.mean(mask[..., 1:, :, :] != mask[..., :-1, :, :]))})
    pairs = []
    for i, j in itertools.combinations(range(5), 2):
        a, b = masks[i], masks[j]
        intersection = np.count_nonzero(a & b)
        union = np.count_nonzero(a | b)
        # Marginal-conditioned null: independent masks with the observed
        # separate per-position/per-time densities. Ratio of expectations.
        pa, pb = a.mean(axis=(-2, -1)), b.mean(axis=(-2, -1))
        expected_intersection = float((pa * pb).sum() * 29 * 52)
        budget = int(a.sum()) + int(b.sum())
        pairs.append({'source_pair': [i, j], 'dense_mask_Jaccard': intersection / union,
                      'dense_set_replaced_fraction': 1 - intersection / int(a.sum()),
                      'independent_per_time_marginal_expected_Jaccard': expected_intersection / (budget - expected_intersection)})
    return {'evidence': 'NEW_CPU_READOUT_OF_FIVE_EXECUTED_FRESH_SUBJECT_ROUTING_TRACES',
            'source_count': 5, 'per_source': mask_rows, 'source_pairs': pairs,
            'quality_or_action_value_labels': None, 'new_masks_executed': 0,
            'limits': 'Already exposed subject images only. Mask differences are decision variability, not beneficial conditional computation. Native21 latent bins are not81 decoded frames; no causal video-flicker conclusion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    proof, compensation, r7 = scale_counterexample(), compensation_controls(), r7_readout()
    if args.check:
        saved = read(OUTPUT)
        assert saved['source_sha256'] == sha(__file__)
        assert equivalent(saved['scale_counterexample'], proof)
        assert equivalent(saved['compensation_controls'], compensation)
        assert equivalent(saved['R7_scalar_reanalysis'], r7)
        for name, digest in saved['repository_source_bindings'].items():
            assert sha(ROOT / name) == digest
        assert saved['new_model_calls'] == saved['new_training_updates'] == saved['new_GPU_jobs'] == 0
        assert saved['asset_audit']['payload_headers_inspected'] == 3840
        assert saved['fresh_subject_mask_readout']['source_count'] == 5
        print('PASS: coupling algebra, saved native scalars, immutable bindings and zero model jobs')
        return
    hf = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    parent = read(ROOT / 'research/20261023/R17E0_registered_protocol.json')
    native = []
    for name in ['.runtime/code/LightX2V/lightx2v/models/networks/wan/infer/transformer_infer.py',
                 '.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py',
                 '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy/base/QType.py']:
        digest = sha(ROOT / name)
        bound = name in parent['source_sha256']
        assert not bound or digest == parent['source_sha256'][name]
        native.append({'source_relative_path': name, 'sha256': digest,
                       'also_bound_in_original_R17_protocol': bound})
    record = {'id': 'C0_COUPLING_PREW1_OFFLINE_DISCRIMINATORS', 'status': 'COMPLETE_SOURCE_SAVED_DATA_AND_CPU_ONLY',
              'source_sha256': sha(__file__), 'scale_counterexample': proof,
              'compensation_controls': compensation, 'R7_scalar_reanalysis': r7,
              'asset_audit': asset_audit(hf), 'fresh_subject_mask_readout': routing_readout(hf),
              'native_source_bindings': native,
              'repository_source_bindings': {name: sha(ROOT / name) for name in
                  ['tools/compensated_support.py', 'tools/cross_layer_transport.py', 'tools/ub1_row_dispatch.py',
                   'tools/variable_teacher_capture.py', 'research/20261023/TACR_sampling_evidence_v2.json']},
              'new_model_calls': 0, 'new_training_updates': 0, 'new_GPU_jobs': 0,
              'model_eligibility_changed': False, 'new_official_quality_result': None}
    OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': record['status'], 'OP2_headers': record['asset_audit']['payload_headers_inspected'],
                      'R7_pooled': r7['pooled'], 'new_model_calls': 0}))


if __name__ == '__main__':
    main()
