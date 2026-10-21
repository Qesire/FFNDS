"""R8F1: qualify the existing R2 full-affine backbone at full support first."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file

from ar2_checkpoint import atomic_json, digest
from check_research_routes import load_records, validate_records, method_eligibility
from historical_source_snapshot import HistoricalSources
from native_affine_fold import fold_parameters, folded_output, native_operands
from onpolicy_residual import native_sse
from r4_native_support import full_parameters
from restored_assets import ROOT, verified_restored_assets
from run_ar2_restart import tensor_sha
from ub1_native_data import load_ub1_collections
from ub1_native_replay import UB1Positions, replay_position

METHOD = 'R8F1_FULL_SUPPORT_NATIVE_FOLD'
PROTOCOL = ROOT / 'research/20261021/R8F1_native_affine_protocol.json'
SOURCES = ['probe_native_affine_fold.py', 'native_affine_fold.py', 'r4_native_support.py',
           'ub1_native_data.py', 'ub1_native_replay.py', 'onpolicy_residual.py', 'folded_adapter.py',
           'historical_source_snapshot.py', 'restored_assets.py']


def source_binding():
    return {f'tools/{name}': digest(ROOT / 'tools' / name) for name in SOURCES}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--public-report', type=Path, required=True)
    parser.add_argument('--source-view-receipt', type=Path, required=True)
    args = parser.parse_args()
    state = load_records(ROOT)
    if validate_records(*state) or not method_eligibility(state[0], state[1], METHOD)[0]:
        raise RuntimeError('Full-support folding control is not registered')
    p = json.loads(PROTOCOL.read_text())
    if p['status'] != 'REGISTERED_FULL_SUPPORT_PRECISION_CONTROL':
        raise RuntimeError('Unexpected folding control phase')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'RUN.json').exists() or args.public_report.exists():
        raise RuntimeError('Preserve previous full-support attempts')
    lock = (args.out / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(8); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    runs = ROOT / 'runs/worker_restart_20261020'
    history = HistoricalSources(); _, cache, assets = verified_restored_assets(history)
    records, bindings, capture = load_ub1_collections([runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in (0, 1)], ROOT)
    if capture['assets'] != assets:
        raise RuntimeError('OP2/current original assets differ')
    original_view = json.loads((runs / 'source_view_native_equivalence/CONTROL.json').read_text())
    view = json.loads(args.source_view_receipt.read_text())
    if (original_view['status'] != 'PASS_EXACT_SOURCE_VIEW_ALL192_NATIVE_CONTROLS_BYTE_IDENTICAL'
            or view['status'] != 'RUNNING_CHILD_WITH_VERIFIED_SOURCE_VIEW'
            or view['view_manifest_sha256'] != original_view['view_manifest_sha256']
            or view['source_inventory_sha256'] != original_view['source_inventory_sha256']):
        raise RuntimeError('Exact original native dependency view required')
    positions = UB1Positions(runs / 'ub1_step_budget_fit/UB1_policy.json', history,
                            expected_policy_sha256=capture['original_UB1_policy_sha256'])
    raw = Path(cache['path']).parents[1] / 'r8f1_native_affine_20261021' / args.out.name
    raw.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw).free < 82 << 30:
        raise RuntimeError('Preserve80GiB reserve plus2GiB allowance')
    sources = source_binding()
    identity = {'protocol_sha256': digest(PROTOCOL), 'sources': sources, 'assets': assets,
                'original_policy_sha256': positions.policy_sha256,
                'OP2_collections': [{'record': str(Path(b['path']).relative_to(ROOT)), 'sha256': b['sha256']} for b in bindings],
                'source_inventory_sha256': view['source_inventory_sha256'], 'TF32': False}
    report = {'status': 'RUNNING', 'identity': identity, 'controls': [], 'rows': [], 'matrices': [],
              'model_generations': 0, 'quality_or_speed_acceptance': False}
    atomic_json(args.out / 'RUN.json', report)
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    from quant.hif4_backend import hif4_qdq_tensor
    try:
        prepared = []
        for step, block in p['positions']:
            if source_binding() != sources or digest(PROTOCOL) != identity['protocol_sha256']:
                raise RuntimeError('Frozen folding experiment source changed')
            position = positions.position(step, block)
            cases = replay_position(records, position, quantizer, require_captured_UB1=True)
            report['controls'].append({'step': step, 'block': block, 'all24_UB1_main_and_R_bitwise': True})
            full = full_parameters(positions, position)
            selected = [c for c in cases if c['index'] in p['inputs']['indices']]
            for case in selected:
                x, ids, phi = native_operands(case, full, quantizer)
                del x, ids, phi
            report['controls'][-1].update(full_native_teacher_bitwise_inputs=[c['index'] for c in selected])
            prepared.append((step, block, full, selected))
            del cases, position
            atomic_json(args.out / 'RUN.json', report)
        # Both positions/all48 incumbent replays and all8 full-teacher controls
        # pass before measuring any changed affine representation.
        for step, block, full, cases in prepared:
            matrix, bias = fold_parameters(full['w0'], full['w2'], full['b0'], full['smooth2'])
            a, ba = matrix.bfloat16().contiguous(), bias.bfloat16().contiguous()
            aq = hif4_qdq_tensor(a, force_fp32=True, kind='weight').bfloat16().contiguous()
            if not all(bool(torch.isfinite(v).all()) for v in [matrix, bias, a, aq, ba]):
                raise RuntimeError('Nonfinite folded coefficients')
            path = raw / f'step{step}_block{block}_matrices.safetensors'
            save_file({'A_BF16': a, 'A_HiF4': aq, 'biasA_BF16': ba}, str(path))
            report['matrices'].append({'step': step, 'block': block,
                'HF_derived_relative': str(path.relative_to(Path(cache['path']).parents[1])),
                'sha256': digest(path), 'FP64_A_sha256': tensor_sha(matrix),
                'A_BF16_sha256': tensor_sha(a), 'A_HiF4_sha256': tensor_sha(aq),
                'biasA_sha256': tensor_sha(ba), 'teacher_shard_sha256': full['shard_sha256']})
            del matrix, bias
            for case in cases:
                if case['index'] not in p['inputs']['indices']:
                    continue
                x, ids, phi = native_operands(case, full, quantizer)
                baseline = native_sse(case['StudentY'], case['TeacherY'], case['gate'])
                if baseline <= 0:
                    raise RuntimeError('Nondegenerate original UB1 error required')
                errors = {}
                for label, weight, quant_phi in [('BF16_fold_BF16_remainder', a, False),
                        ('HiF4_fold_BF16_remainder', aq, False), ('HiF4_fold_HiF4_remainder', aq, True)]:
                    output = folded_output(x, ids, phi, full, case, weight, ba, quantizer, quantize_remainder=quant_phi)
                    if not bool(torch.isfinite(output).all()):
                        raise RuntimeError('Nonfinite full-support folding output')
                    errors[label] = native_sse(output, case['TeacherY'], case['gate'])
                row = {'step': step, 'block': block, 'index': case['index'], 'full_rows': case['scope']['full_rows'],
                       'observed_sample_rows': len(case['V1']), 'OP2_payload_sha256': case['scope']['sha256'],
                       'original_full_native_teacher_bitwise': True, 'original_UB1_gated_SSE': baseline,
                       'full_support_fold_gated_SSE': errors,
                       'fold_error_over_UB1': {k: v / baseline for k, v in errors.items()},
                       'BF16_numerical_gate': errors['BF16_fold_BF16_remainder'] <= p['gates']['BF16_error_over_UB1_max'] * baseline,
                       'HiF4_precision_gate': errors['HiF4_fold_HiF4_remainder'] <= p['gates']['HiF4_error_over_UB1_max'] * baseline}
                report['rows'].append(row); atomic_json(args.out / 'RUN.json', report)
                print(f'R8F1 step{step} input{case["index"]:02d} fold/UB1={row["fold_error_over_UB1"]}', flush=True)
                del x, ids, phi
            del a, aq, ba
        if len(report['rows']) != 8:
            raise RuntimeError('Incomplete fixed eight full-support comparisons')
        report.update(status='COMPLETED_FULL_SUPPORT_NATIVE_FOLD_CHECK',
            BF16_numerical_gate_all8=all(r['BF16_numerical_gate'] for r in report['rows']),
            HiF4_precision_gate_all8=all(r['HiF4_precision_gate'] for r in report['rows']),
            limitations='Full support only, mainMAC1.185185x original dense; no pruning/fit/video. BF16 and mixed arms are attribution controls, not original W4A4 candidates. Failure does not eliminate all affine/quantization-aware methods.')
        atomic_json(args.public_report, report)
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(args.out / 'RUN.json', report)


if __name__ == '__main__':
    main()
