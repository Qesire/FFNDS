"""R8F2: one fixed64-group Hessian-aware HiF4 encoding of the folded matrix."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ar2_checkpoint import atomic_json, digest
from check_research_routes import load_records, validate_records, method_eligibility
from group_hessian_quant import encode_groups
from historical_source_snapshot import HistoricalSources
from native_affine_fold import folded_output, native_operands
from onpolicy_residual import native_sse, same_bits
from r4_native_support import full_parameters
from restored_assets import ROOT, verified_restored_assets
from run_ar2_restart import tensor_sha
from ub1_native_data import load_ub1_collections
from ub1_native_replay import UB1Positions, replay_position
from probe_native_affine_fold import SOURCES as BASE_SOURCES

METHOD = 'R8F2_GROUP64_HESSIAN_AFFINE'
PROTOCOL = ROOT / 'research/20261021/R8F2_group64_hessian_protocol.json'


def source_binding():
    return {f'tools/{name}': digest(ROOT / 'tools' / name) for name in sorted(set(BASE_SOURCES +
            ['probe_hessian_affine_fold.py', 'group_hessian_quant.py']))}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--public-report', type=Path, required=True)
    args = parser.parse_args()
    state = load_records(ROOT)
    if validate_records(*state) or not method_eligibility(state[0], state[1], METHOD)[0]:
        raise RuntimeError('Hessian-aware full-support control is not registered')
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'REGISTERED_FIXED_GROUP64_HESSIAN_ENCODING':
        raise RuntimeError('Unexpected repair protocol')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.out / 'RUN.json').exists() or args.public_report.exists():
        raise RuntimeError('Preserve previous encoding attempts')
    torch.set_num_threads(8); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    history = HistoricalSources(); _, cache, assets = verified_restored_assets(history)
    runs = ROOT / 'runs/worker_restart_20261020'
    records, bindings, capture = load_ub1_collections([runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in (0, 1)], ROOT)
    if capture['assets'] != assets:
        raise RuntimeError('Original native asset identity changed')
    positions = UB1Positions(runs / 'ub1_step_budget_fit/UB1_policy.json', history,
                            expected_policy_sha256=capture['original_UB1_policy_sha256'])
    prior_path = ROOT / 'research/20261021/R8F1_native_affine_results.json'
    prior = json.loads(prior_path.read_text())
    raw = Path(cache['path']).parents[1] / 'r8f2_group64_hessian_20261021' / args.out.name
    raw.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw).free < 82 << 30:
        raise RuntimeError('Preserve80GiB node reserve plus2GiB allowance')
    sources = source_binding()
    identity = {'protocol_sha256': digest(PROTOCOL), 'sources': sources, 'assets': assets,
                'prior_result_sha256': digest(prior_path), 'original_policy_sha256': positions.policy_sha256,
                'OP2_collections': [{'record': str(Path(b['path']).relative_to(ROOT)), 'sha256': b['sha256']} for b in bindings]}
    report = {'status': 'RUNNING', 'identity': identity, 'controls': [], 'encodings': [], 'rows': [],
              'pipeline_generations': 0, 'quality_or_speed_acceptance': False}
    atomic_json(args.out / 'RUN.json', report)
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    from quant.hif4_backend import hif4_qdq_tensor
    quant_weight = lambda value: hif4_qdq_tensor(value.bfloat16().contiguous(), force_fp32=True, kind='weight').bfloat16()
    try:
        prepared = []
        for step, block in protocol['positions']:
            position = positions.position(step, block)
            cases = replay_position(records, position, quantizer, require_captured_UB1=True)
            full = full_parameters(positions, position)
            for case in cases:
                if case['index'] in protocol['inputs']['indices']:
                    x, ids, phi = native_operands(case, full, quantizer); del x, ids, phi
            prepared.append((step, block, full, cases))
            report['controls'].append({'step': step, 'block': block, 'all24_UB1_main_and_R_bitwise': True,
                                       'full_native_teacher_bitwise_inputs': protocol['inputs']['indices']})
            atomic_json(args.out / 'RUN.json', report)
        for step, block, full, cases in prepared:
            if source_binding() != sources or digest(PROTOCOL) != identity['protocol_sha256']:
                raise RuntimeError('Frozen encoding source changed')
            baseline = next(v for v in prior['matrices'] if (v['step'], v['block']) == (step, block))
            path = Path(cache['path']).parents[1] / baseline['HF_derived_relative']
            if digest(path) != baseline['sha256']:
                raise RuntimeError('Original folded coefficient payload changed')
            values = load_file(str(path), device='cuda'); a, aq, bias = [values[k] for k in ['A_BF16', 'A_HiF4', 'biasA_BF16']]
            if tensor_sha(a) != baseline['A_BF16_sha256'] or tensor_sha(aq) != baseline['A_HiF4_sha256'] or tensor_sha(bias) != baseline['biasA_sha256']:
                raise RuntimeError('Decoded original folded matrices changed')
            groupwise_rtn = torch.cat([quant_weight(a[:, j:j + 64]) for j in range(0, 5120, 64)], dim=1).contiguous()
            if not same_bits(groupwise_rtn, aq):
                raise RuntimeError('Joint64-group quantizer does not equal original full-matrix HiF4 RTN')
            train = [c for c in cases if c['index'] in protocol['inputs']['training_indices']]
            if len(train) != 12 or sum(len(c['V1']) for c in train) != 3072:
                raise RuntimeError('Fixed train-only sampled covariance membership differs')
            inputs = torch.cat([c['V1'] for c in train]).double()
            covariance = inputs.T @ inputs / len(inputs)
            q, meta = encode_groups(a, covariance, quant_weight, group_size=64, damping=.01)
            encoded = q.bfloat16().contiguous(); del q, covariance
            output_path = raw / f'step{step}_block{block}_A.safetensors'
            save_file({'A_HiF4_group64_Hessian': encoded}, str(output_path))
            affine_loss = lambda weight: float((inputs @ (weight.double() - a.double()).T).square().sum())
            report['encodings'].append({'step': step, 'block': block, **meta,
                'training_indices': [c['index'] for c in train], 'training_payload_sha256': [c['scope']['sha256'] for c in train],
                'training_rows': len(inputs), 'groupwise_RTN_equals_prior_full_matrix_bitwise': True,
                'training_affine_SSE_RTN': affine_loss(aq), 'training_affine_SSE_Hessian': affine_loss(encoded),
                'HF_derived_relative': str(output_path.relative_to(Path(cache['path']).parents[1])),
                'sha256': digest(output_path), 'encoded_matrix_sha256': tensor_sha(encoded)})
            del inputs
            for case in cases:
                if case['index'] not in protocol['inputs']['indices']:
                    continue
                x, ids, phi = native_operands(case, full, quantizer)
                old = next(r for r in prior['rows'] if (r['step'], r['block'], r['index']) == (step, block, case['index']))
                target = case['TeacherY']; gate = case['gate']
                rtn = folded_output(x, ids, phi, full, case, aq, bias, quantizer, quantize_remainder=True)
                rtn_sse = native_sse(rtn, target, gate)
                if abs(rtn_sse - old['full_support_fold_gated_SSE']['HiF4_fold_HiF4_remainder']) > 1e-10 * max(1., rtn_sse):
                    raise RuntimeError('Original native full-support RTN loss does not reproduce')
                mixed = folded_output(x, ids, phi, full, case, encoded, bias, quantizer, quantize_remainder=False)
                native = folded_output(x, ids, phi, full, case, encoded, bias, quantizer, quantize_remainder=True)
                if not bool(torch.isfinite(mixed).all() and torch.isfinite(native).all()):
                    raise RuntimeError('Nonfinite encoded native outputs')
                errors = {'RTN_full_HiF4': rtn_sse, 'Hessian_HiF4_BF16_remainder': native_sse(mixed, target, gate),
                          'Hessian_full_HiF4': native_sse(native, target, gate)}
                reference = old['original_UB1_gated_SSE']; ratio = errors['Hessian_full_HiF4'] / reference
                row = {'step': step, 'block': block, 'index': case['index'], 'used_for_encoding': False,
                       'original_UB1_gated_SSE': reference, 'native_gated_SSE': errors,
                       'fold_error_over_UB1': {k: v / reference for k, v in errors.items()},
                       'native_precision_gate': ratio <= protocol['gates']['HiF4_error_over_UB1_max'],
                       'improves_prior_RTN': errors['Hessian_full_HiF4'] < rtn_sse,
                       'original_teacher_bitwise': True, 'original_RTN_loss_reproduced': True,
                       'OP2_payload_sha256': case['scope']['sha256']}
                report['rows'].append(row); atomic_json(args.out / 'RUN.json', report)
                print(f'R8F2 step{step} input{case["index"]:02d} native/UB1={ratio:.6f}, prior={rtn_sse / reference:.6f}', flush=True)
                del x, ids, phi, rtn, mixed, native
            del values, a, aq, bias, groupwise_rtn, encoded
        if len(report['rows']) != 8:
            raise RuntimeError('Exactly8 original held-out native cases required')
        report.update(status='COMPLETED_FIXED_GROUP64_HESSIAN_NATIVE_CHECK',
            native_precision_gate_all8=all(r['native_precision_gate'] for r in report['rows']),
            improves_RTN_all8=all(r['improves_prior_RTN'] for r in report['rows']),
            limitations='Known groupwise Hessian quantization transfer; fullsupport only, no pruning/SGD/video. Train-only odd12 covariance and original4 even validation sources, all development exposed. No threshold/damping/actorder sweep or accuracy/speed inheritance.')
        atomic_json(args.public_report, report)
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(args.out / 'RUN.json', report)


if __name__ == '__main__':
    main()
