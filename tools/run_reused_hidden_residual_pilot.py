"""NF2: one new native output GEMM reusing the frozen UB1 residual hidden state."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from hidden_profile_head import fit_head
from historical_source_snapshot import HistoricalSources
from onpolicy_residual import check_old_replay, native_sse, position_cases, replay_rows, same_bits
from reused_hidden_readout import readout_features, training_moments
from run_ar2_restart import ROOT, tensor_sha
from run_hidden_profile_pilot import hidden_full
from ub1_native_data import load_ub1_collections
from ub1_native_replay import UB1Positions


@torch.no_grad()
def head_risk(cases, residual, moments, head, *, nonlinear):
    total = 0.
    for case in cases:
        hidden, ids = hidden_full(case, residual)
        features = readout_features(hidden, moments, nonlinear=nonlinear)
        correction = F.linear(features, head['weight'], head['bias']).reshape(-1, 5120).index_select(0, ids)
        total += native_sse(case['StudentY'] + correction, case['TeacherY'], case['gate'])
    return total


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--source-view-receipt', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if (not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 82 << 30):
        raise RuntimeError('Node HF cache and80GiB reserve plus2GiB allowance required')
    runs = ROOT / 'runs/worker_restart_20261020'
    required = [
        ('op2a_dataset_integrity/RUN.json', 'PASS_OP2A_ALL24_FULL3840_NATIVE_PAYLOAD_INTEGRITY'),
        ('op2r_native_replay/RUN.json', 'PASS_OP2R_ALL_FOUR_STEPS_BOTH_NATIVE_STATE_CONTROLS'),
        ('source_view_native_equivalence/CONTROL.json', 'PASS_EXACT_SOURCE_VIEW_ALL192_NATIVE_CONTROLS_BYTE_IDENTICAL'),
        ('op3_state_matched_residual/RUN.json', 'COMPLETED_OP3_STATE_MATCHED_NATIVE_GATE_REJECTED')]
    conditions = []
    for name, status in required:
        path = runs / name
        if json.loads(path.read_text())['status'] != status:
            raise RuntimeError('NF2 requires completed original controls and the preserved OP3 rejection')
        conditions.append({'name': name, 'sha256': digest(path)})
    view_control = json.loads((runs / 'source_view_native_equivalence/CONTROL.json').read_text())
    execution = json.loads(args.source_view_receipt.read_text())
    if (execution['status'] != 'RUNNING_CHILD_WITH_VERIFIED_SOURCE_VIEW'
            or execution['view_manifest_sha256'] != view_control['view_manifest_sha256']
            or execution['source_inventory_sha256'] != view_control['source_inventory_sha256']):
        raise RuntimeError('NF2 must run under the source view with exact native equivalence controls')
    import transformers
    if (str(Path(transformers.__file__).resolve()) != execution['loaded_package_path']
            or transformers.__version__ != execution['version']):
        raise RuntimeError('Loaded dependency differs from its execution receipt')
    history = HistoricalSources()
    records, bindings, capture = load_ub1_collections(
        [runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT)
    positions = UB1Positions(runs / 'ub1_step_budget_fit/UB1_policy.json', history,
                            expected_policy_sha256=capture['original_UB1_policy_sha256'])
    sources = ['run_reused_hidden_residual_pilot.py', 'reused_hidden_readout.py',
        'hidden_profile_head.py', 'run_hidden_profile_pilot.py', 'ub1_native_replay.py',
        'ub1_native_data.py', 'historical_source_snapshot.py', 'onpolicy_residual.py',
        'with_transformers_source_view.py']
    protocol = ROOT / 'config/reused_hidden_residual_nf2.json'
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = CheckpointStore(args.out, {'protocol_sha256': digest(protocol),
        'protocol': json.loads(protocol.read_text()), 'conditions': conditions,
        'UB1_policy_sha256': positions.policy_sha256, 'collection_bindings': bindings,
        'source_view_manifest_sha256': view_control['view_manifest_sha256'],
        'source_inventory_sha256': view_control['source_inventory_sha256'],
        'sources': {str(ROOT / 'tools' / name): digest(ROOT / 'tools' / name) for name in sources},
        'torch': torch.__version__, 'TF32': False})
    meta = {'status': 'RUNNING', 'fingerprint': store.fingerprint, 'checkpoint_root': str(args.out),
        'conditions': conditions, 'source_view_execution_receipt': str(args.source_view_receipt),
        'controls': [], 'CV': [], 'quality_or_speed_acceptance': False,
        'scope': 'new-readout-only source folds on known row-exposed actual UB1 states; no whole-pipeline held-out claim'}
    atomic_json(args.report, meta)
    started = time.perf_counter()
    try:
        for step in range(4):
            position = positions.position(step, 20)
            residual = position['residual']
            cases = position_cases(records, step, 20, position['expected_capture'])
            check_old_replay(cases, residual)
            for case in cases:
                hidden, ids = hidden_full(case, residual)
                split = F.linear(hidden, residual['second'], residual['bias_second']).reshape(-1, 5120).index_select(0, ids)
                if (not same_bits(split, replay_rows(case['V1'], case['scope'], residual))
                        or not same_bits(case['BaseY'] + split, case['StudentY'])):
                    raise RuntimeError('Native split residual or observed student output bits changed')
                case['hidden'] = hidden.reshape(-1, 256).index_select(0, ids).contiguous()
            meta['controls'].append({'step': step, 'block': 20, 'inputs': len(cases),
                'split_frozen_R_bitwise': True, 'captured_UB1_StudentY_bitwise': True,
                'full_native_M_and_stride': True,
                'hidden_sha256': {str(c['index']): tensor_sha(c['hidden']) for c in cases}})
            atomic_json(args.report, meta)
            for fold in [0, 1]:
                key = f'step{step}_fold{fold}'
                part = store.load('heads', key)
                if part is None:
                    train = [c for c in cases if c['index'] % 2 != fold]
                    validation = [c for c in cases if c['index'] % 2 == fold]
                    if len(train) != 12 or len(validation) != 12 or sum(c['index'] < 12 for c in train) != 6:
                        raise RuntimeError('Fixed balanced12/12 source folds required')
                    moments = training_moments(torch.cat([c['hidden'] for c in train]))
                    samples = {False: [], True: []}
                    for case in train:
                        hidden, ids = hidden_full(case, residual)
                        for nonlinear in [False, True]:
                            features = readout_features(hidden, moments, nonlinear=nonlinear)
                            samples[nonlinear].append(features.reshape(-1, 256).index_select(0, ids).contiguous())
                    target = torch.cat([c['TeacherY'].float() - c['StudentY'].float() for c in train])
                    frozen = sum(native_sse(c['StudentY'], c['TeacherY'], c['gate']) for c in validation)
                    row = {'step': step, 'block': 20, 'fold': fold,
                        'train_inputs': [c['index'] for c in train],
                        'validation_inputs': [c['index'] for c in validation],
                        'frozen_UB1_native_SSE': frozen,
                        'moments_sha256': {k: tensor_sha(v) for k, v in moments.items()},
                        'full_native_M_and_stride': True, 'feature_width': 256,
                        'extra_output_GEMMs': 1, 'extra_input_GEMMs': 0,
                        'main_MAC_fraction': .5, 'aux_MAC_fraction_if_full160': 3 * 256 / (2 * 13824)}
                    arrays = {k: v.cpu().numpy() for k, v in moments.items()}
                    for nonlinear, label in [(False, 'linear'), (True, 'nonlinear')]:
                        fitted = fit_head(torch.cat(samples[nonlinear]), target, False)
                        head = {k: v.bfloat16().contiguous() for k, v in fitted.items()}
                        if any(not torch.isfinite(v).all() for v in head.values()):
                            raise RuntimeError('Nonfinite BF16 head coefficients')
                        row[label + '_native_SSE'] = head_risk(validation, residual, moments, head, nonlinear=nonlinear)
                        row[label + '_parameter_sha256'] = {k: tensor_sha(v) for k, v in head.items()}
                        arrays.update({label + '_' + k: v.float().cpu().numpy() for k, v in head.items()})
                    candidate, linear = row['nonlinear_native_SSE'], row['linear_native_SSE']
                    if min(frozen, linear) <= 0 or not all(math.isfinite(v) for v in [frozen, linear, candidate]):
                        raise RuntimeError('Finite nondegenerate native risks required')
                    row.update(nonlinear_over_frozen=candidate / frozen, nonlinear_over_linear=candidate / linear,
                        gate_pass=candidate <= .90 * frozen and candidate <= .95 * linear)
                    store.save('heads', key, arrays, row)
                    part = store.load('heads', key)
                for label in ['linear', 'nonlinear']:
                    restored = {k: torch.from_numpy(part['arrays'][label + '_' + k]).bfloat16()
                                for k in ['weight', 'bias']}
                    if {k: tensor_sha(v) for k, v in restored.items()} != part['metadata'][label + '_parameter_sha256']:
                        raise RuntimeError('Saved BF16 head coefficients are not lossless')
                if {k: tensor_sha(torch.from_numpy(part['arrays'][k])) for k in ['mean', 'scale']} != part['metadata']['moments_sha256']:
                    raise RuntimeError('Saved FP32 moments are not lossless')
                meta['CV'].append(part['metadata'])
                atomic_json(args.report, meta)
                print('NF2 ' + json.dumps(part['metadata']), flush=True)
            del cases, position, residual
        passed = len(meta['CV']) == 8 and all(row['gate_pass'] for row in meta['CV'])
        meta.update(status='COMPLETED_NF2_NATIVE_GATE_PASSED' if passed else 'COMPLETED_NF2_NATIVE_GATE_REJECTED',
            native_gate_pass=passed, elapsed_seconds=time.perf_counter() - started, full_fit_condition_met=passed)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(args.report, meta)
        atomic_json(args.out / 'run_meta.json', meta)


if __name__ == '__main__':
    main()
