"""R9A1: original R2 learned affine/retained-remainder operator, no R256.

Two experts share one learned affine matrix across their two native steps.
BF16 derived matrix and post-Q2 remainder arithmetic are explicitly mixed
precision; this does not claim a fully re-quantized HiF4 representation.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from affine_remainder import fit_channel_affine, fold_learned, greedy_fixed_support, native_base, native_prediction, observe_native, remainder
from ar2_checkpoint import atomic_json, digest
from check_research_routes import load_records, validate_records, method_eligibility
from historical_source_snapshot import HistoricalSources
from onpolicy_residual import native_sse, same_bits
from r4_native_support import full_parameters
from restored_assets import ROOT, verified_restored_assets
from run_ar2_restart import tensor_sha
from ub1_native_data import load_ub1_collections
from ub1_native_replay import UB1Positions, replay_position

METHOD = 'R9A1_LEARNED_AFFINE_NATIVE_REMAINDER'
PROTOCOL = ROOT / 'research/20261021/R9A1_learned_affine_protocol.json'
SOURCES = ['screen_learned_affine_remainder.py', 'affine_remainder.py', 'r4_native_support.py',
           'ub1_native_data.py', 'ub1_native_replay.py', 'onpolicy_residual.py', 'folded_adapter.py',
           'historical_source_snapshot.py', 'restored_assets.py']


def source_binding():
    return {f'tools/{name}': digest(ROOT / 'tools' / name) for name in SOURCES}


def full_support_barrier(out, identity):
    deadline = time.monotonic() + 1200
    while True:
        reports = []
        for expert in (0, 1):
            path = out / f'expert{expert}/RUN.json'
            if not path.exists(): continue
            report = json.loads(path.read_text())
            if report['scientific_identity'] != identity or report['status'].startswith('FAILED'):
                raise RuntimeError('Paired affine job failed or scientific identity differs')
            if report.get('full_support_controls_complete'): reports.append(report)
        if len(reports) == 2:
            return all(r['full_support_gate_all8'] for r in reports)
        if time.monotonic() >= deadline:
            raise RuntimeError('Paired full-support control rendezvous timed out')
        time.sleep(2)


@torch.no_grad()
def select_groups(train, full, matrix, bias, slope, intercept, count):
    gate = train[0]['gate'].double()
    if any(not same_bits(c['gate'], train[0]['gate']) for c in train):
        raise RuntimeError('Original source-constant timestep gate required')
    r = torch.cat([c['remainder'] for c in train])
    target = torch.cat([(c['TeacherY'].double() - native_base(c, matrix, bias).double()) * gate for c in train])
    # Preallocate instead of stacking216 full output matrices twice.
    contributions = torch.empty((216, target.numel()), dtype=torch.float64, device='cuda')
    for group in range(216):
        start = group * 64
        value = F.linear(r[:, start:start + 64].double(), full['w2'][:, start:start + 64].double().contiguous()) * gate
        contributions[group].copy_(value.reshape(-1))
    gram = contributions @ contributions.T
    correlation = contributions @ target.reshape(-1)
    norm = float(target.square().sum())
    chosen = greedy_fixed_support(gram, correlation, count, norm)
    reconstruction = torch.zeros_like(target.reshape(-1))
    for group in chosen['groups']: reconstruction.add_(contributions[group])
    explicit = float((target.reshape(-1) - reconstruction).square().sum())
    if abs(explicit - chosen['ideal_training_risk']) > 1e-9 * max(1., explicit, norm):
        raise RuntimeError('Signed group Gram risk does not reproduce explicit training residual')
    chosen.update(explicit_training_risk=explicit, target_norm2=norm,
                  train_inputs=[c['index'] for c in train], all216_cross_terms=True)
    return chosen, gram.cpu().contiguous(), correlation.cpu().contiguous()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert', type=int, choices=[0, 1], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    state = load_records(ROOT)
    if validate_records(*state) or not method_eligibility(state[0], state[1], METHOD)[0]:
        raise RuntimeError('Learned affine operator is not registered')
    p = json.loads(PROTOCOL.read_text())
    if p['status'] != 'REGISTERED_LEARNED_AFFINE_FULL_SUPPORT_THEN_FIXED_COMPRESSION':
        raise RuntimeError('Unexpected learned-affine phase')
    torch.set_num_threads(8); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    args.out = args.out.resolve(); folder = args.out / f'expert{args.expert}'; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'RUN.json').exists(): raise RuntimeError('Preserve previous learned-affine attempts')
    history = HistoricalSources(); _, cache, assets = verified_restored_assets(history)
    runs = ROOT / 'runs/worker_restart_20261020'
    records, bindings, capture = load_ub1_collections([runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in (0, 1)], ROOT)
    if capture['assets'] != assets: raise RuntimeError('Original observed/native assets differ')
    positions = UB1Positions(runs / 'ub1_step_budget_fit/UB1_policy.json', history,
                            expected_policy_sha256=capture['original_UB1_policy_sha256'])
    sources = source_binding()
    identity = {'protocol_sha256': digest(PROTOCOL), 'sources': sources, 'assets': assets,
                'original_policy_sha256': positions.policy_sha256,
                'OP2_collections': [{'record': str(Path(b['path']).relative_to(ROOT)), 'sha256': b['sha256']} for b in bindings]}
    raw = Path(cache['path']).parents[1] / 'r9a1_learned_affine_20261021' / args.out.name / f'expert{args.expert}'
    raw.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw).free < 82 << 30: raise RuntimeError('Preserve80GiB reserve plus2GiB run allowance')
    steps = [0, 1] if args.expert == 0 else [2, 3]
    report = {'status': 'RUNNING', 'scientific_identity': identity, 'expert': args.expert,
              'native_controls': [], 'full_support': [], 'compression': [], 'pipeline_generations': 0,
              'quality_or_speed_acceptance': False, 'R256_active': False}
    atomic_json(folder / 'RUN.json', report)
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    try:
        by_step, full = {}, None
        for step in steps:
            position = positions.position(step, 20)
            cases = replay_position(records, position, quantizer, require_captured_UB1=True)
            parameters = full_parameters(positions, position)
            if full is not None and parameters['shard_sha256'] != full['shard_sha256']:
                raise RuntimeError('Two steps do not share the original expert/block weights')
            full = parameters
            for case in cases:
                z, v2 = observe_native(case, full, quantizer)
                case['native_z'], case['native_v2'] = z, v2
            by_step[step] = cases
            report['native_controls'].append({'step': step, 'block': 20, 'inputs': 24,
                'frozen_UB1_main_and_R_bitwise': True, 'original_full_TeacherY_bitwise': True})
        train_all = [c for step in steps for c in by_step[step] if c['index'] in p['inputs']['training_indices']]
        scaled_z = torch.cat([F.linear(c['V1'].double(), full['w0'].double(), full['b0'].double()) / full['smooth2'].double() for c in train_all])
        native_v = torch.cat([c['native_v2'] for c in train_all])
        slope, intercept = fit_channel_affine(scaled_z, native_v)
        del scaled_z, native_v, train_all
        b2 = by_step[steps[0]][0]['b2']
        if any(not same_bits(c['b2'], b2) for step in steps for c in by_step[step]):
            raise RuntimeError('Shared expert/block original bias differs across steps/sources')
        matrix, bias = fold_learned(full, b2, slope, intercept)
        if not all(bool(torch.isfinite(v).all()) for v in [slope, intercept, matrix, bias]):
            raise RuntimeError('Nonfinite learned affine coefficients')
        path = raw / 'shared_affine.safetensors'
        save_file({'matrix_BF16': matrix, 'bias_total_BF16': bias, 'slope_FP32': slope, 'intercept_FP32': intercept}, str(path))
        report['affine'] = {'shared_steps': steps, 'HF_derived_relative': str(path.relative_to(Path(cache['path']).parents[1])),
            'sha256': digest(path), 'tensor_sha256': {k: tensor_sha(v) for k, v in {'matrix': matrix, 'bias': bias, 'slope': slope, 'intercept': intercept}.items()},
            'parameter_bytes': sum(v.numel() * v.element_size() for v in [matrix, bias, slope, intercept]),
            'slope_min': float(slope.min()), 'slope_max': float(slope.max()),
            'training_indices': p['inputs']['training_indices'], 'training_native_rows': 6144,
            'teacher_shard_sha256': full['shard_sha256'], 'precision': p['precision']}
        for step in steps:
            for case in by_step[step]:
                case['remainder'] = remainder(case['native_z'], case['native_v2'], full, slope, intercept)
                if case['index'] not in p['inputs']['validation_indices']: continue
                prediction = native_prediction(case, full, matrix, bias, slope, intercept, range(216), quantizer)
                reference = native_sse(case['StudentY'], case['TeacherY'], case['gate'])
                error = native_sse(prediction, case['TeacherY'], case['gate'])
                report['full_support'].append({'step': step, 'block': 20, 'index': case['index'],
                    'native_SSE': error, 'frozen_UB1_SSE': reference, 'ratio': error / reference,
                    'gate_pass': error <= p['gates']['full_support_error_over_UB1_max'] * reference,
                    'original_payload_sha256': case['scope']['sha256']})
        report.update(full_support_controls_complete=True, full_support_gate_all8=all(r['gate_pass'] for r in report['full_support']))
        atomic_json(folder / 'RUN.json', report)
        if not full_support_barrier(args.out, identity):
            report.update(status='COMPLETED_NUMERICAL_FOLD_GATE_REJECTED', compression_status='NOT_RUN_FULL_SUPPORT_GATE_REJECTED')
            return
        for step in steps:
            if source_binding() != sources or digest(PROTOCOL) != identity['protocol_sha256']:
                raise RuntimeError('Frozen learned-affine source changed')
            train = [c for c in by_step[step] if c['index'] in p['inputs']['training_indices']]
            count = p['groups_by_step'][step]
            chosen, gram, correlation = select_groups(train, full, matrix, bias, slope, intercept, count)
            support_path = raw / f'step{step}_support.safetensors'
            save_file({'gram': gram, 'correlation': correlation,
                       'groups': torch.tensor(chosen['groups'], dtype=torch.int64)}, str(support_path))
            rows = []
            for case in by_step[step]:
                if case['index'] not in p['inputs']['validation_indices']: continue
                prediction = native_prediction(case, full, matrix, bias, slope, intercept, chosen['groups'], quantizer)
                error = native_sse(prediction, case['TeacherY'], case['gate'])
                reference = native_sse(case['StudentY'], case['TeacherY'], case['gate'])
                row = {'step': step, 'block': 20, 'index': case['index'], 'native_SSE': error,
                       'frozen_UB1_SSE': reference, 'ratio': error / reference,
                       'gate_pass': error <= p['gates']['compressed_error_over_UB1_max'] * reference,
                       'original_payload_sha256': case['scope']['sha256']}
                rows.append(row)
                print(f'R9A1 step{step} input{case["index"]:02d} candidate/UB1={row["ratio"]:.6f}', flush=True)
            report['compression'].append({'step': step, 'groups': count, 'width': count * 64,
                'actual_main_MAC_fraction': (40 + count) / 216, 'R256_active': False,
                'selection': chosen, 'support': {'HF_derived_relative': str(support_path.relative_to(Path(cache['path']).parents[1])),
                    'sha256': digest(support_path)}, 'validation': rows, 'gate_all4': all(r['gate_pass'] for r in rows)})
            atomic_json(folder / 'RUN.json', report)
        report.update(status='COMPLETED_LEARNED_AFFINE_COMPRESSION_SHARD',
                      native_compression_gate_all8=all(c['gate_all4'] for c in report['compression']))
    except BaseException as error:
        report.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(folder / 'RUN.json', report)


if __name__ == '__main__':
    main()
