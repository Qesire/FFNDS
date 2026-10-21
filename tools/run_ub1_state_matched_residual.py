"""OP3: fixed rank256 residual fitting on paired old and actual UB1 states."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import sys
import time

import torch

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from historical_source_snapshot import HistoricalSources
from onpolicy_residual import check_old_replay, native_sse, position_cases, replay_rows
from original_onpolicy_data import load_original_collections
from reduced_rank_residual import fit_rrr, native_parameters
from run_ar2_restart import ROOT, tensor_sha
from ub1_native_data import load_ub1_collections
from ub1_native_replay import UB1Positions, recompute_on_original_states


def checked_conditions(runs):
    requirements = [
        ('op2_native_controller/RUN.json', 'COMPLETED_OP2_UB1_NATIVE_CALIBRATION24_ONLY'),
        ('op2a_dataset_integrity/RUN.json', 'PASS_OP2A_ALL24_FULL3840_NATIVE_PAYLOAD_INTEGRITY'),
        ('op2d_native_distribution/RUN.json', 'COMPLETED_OP2D_NATIVE_STATE_DISTRIBUTION_DESCRIPTION'),
        ('op2r_native_replay/RUN.json', 'PASS_OP2R_ALL_FOUR_STEPS_BOTH_NATIVE_STATE_CONTROLS'),
        ('source_view_native_equivalence/CONTROL.json', 'PASS_EXACT_SOURCE_VIEW_ALL192_NATIVE_CONTROLS_BYTE_IDENTICAL')]
    bindings = []
    for name, status in requirements:
        path = runs / name
        if json.loads(path.read_text())['status'] != status:
            raise RuntimeError('Complete data, description, replay and source-view controls must precede OP3')
        bindings.append({'name': name, 'sha256': digest(path)})
    control = json.loads((runs / 'source_view_native_equivalence/CONTROL.json').read_text())
    if (control['original_native_report_sha256'] != digest(runs / 'op2r_native_replay/RUN.json')
            or control['view_native_report_sha256'] != control['original_native_report_sha256']):
        raise RuntimeError('Source-view numerical equivalence is not bound to this native baseline')
    return bindings, control


def native_risk(cases, parameters):
    return sum(native_sse(c['UB1_main'] + replay_rows(c['V1'], c['scope'], parameters),
                          c['TeacherY'], c['gate']) for c in cases)


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
        raise RuntimeError('OP3 arrays require node HF cache and80GiB reserve plus2GiB allowance')
    runs = ROOT / 'runs/worker_restart_20261020'
    conditions, view_control = checked_conditions(runs)
    execution = json.loads(args.source_view_receipt.read_text())
    if (execution['status'] != 'RUNNING_CHILD_WITH_VERIFIED_SOURCE_VIEW'
            or execution['view_manifest_sha256'] != view_control['view_manifest_sha256']
            or execution['source_inventory_sha256'] != view_control['source_inventory_sha256']):
        raise RuntimeError('OP3 must use the exact source view whose native controls passed')
    import transformers
    if (str(Path(transformers.__file__).resolve()) != execution['loaded_package_path']
            or transformers.__version__ != execution['version']):
        raise RuntimeError('Actual OP3 dependency differs from its execution receipt')
    history = HistoricalSources()
    original, old_bindings, old_identity = load_original_collections(
        [runs / f'onpolicy_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT, history)
    current, new_bindings, new_identity = load_ub1_collections(
        [runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT)
    if old_identity['assets'] != new_identity['assets']:
        raise RuntimeError('Paired state assets differ')
    positions = UB1Positions(runs / 'ub1_step_budget_fit/UB1_policy.json', history,
                            expected_policy_sha256=new_identity['original_UB1_policy_sha256'])
    if positions.policy['parent_policy']['sha256'] != old_identity['student_policy_sha256']:
        raise RuntimeError('Old-state collection is not the original frozen C24 parent')
    protocol = ROOT / 'config/onpolicy_ub1_residual_op3.json'
    sources = ['run_ub1_state_matched_residual.py', 'ub1_native_replay.py', 'ub1_native_data.py',
        'original_onpolicy_data.py', 'historical_source_snapshot.py', 'onpolicy_residual.py',
        'reduced_rank_residual.py', 'folded_adapter.py', 'run_folded_adapter_cv.py',
        'with_transformers_source_view.py']
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {'protocol_sha256': digest(protocol), 'protocol': json.loads(protocol.read_text()),
        'conditions': conditions, 'old_bindings': old_bindings, 'new_bindings': new_bindings,
        'original_UB1_policy_sha256': positions.policy_sha256,
        'source_view_manifest_sha256': view_control['view_manifest_sha256'],
        'source_inventory_sha256': view_control['source_inventory_sha256'],
        'sources': {str(ROOT / 'tools' / name): digest(ROOT / 'tools' / name) for name in sources},
        'torch': torch.__version__, 'TF32': False}
    store = CheckpointStore(args.out, identity)
    meta = {'status': 'RUNNING', 'fingerprint': store.fingerprint, 'checkpoint_root': str(args.out),
        'conditions': conditions, 'source_view_execution_receipt': str(args.source_view_receipt),
        'controls': [], 'CV': [], 'quality_or_speed_acceptance': False,
        'scope': 'new-residual-only source folds on known, row-exposed calibration states; no whole-pipeline held-out claim'}
    atomic_json(args.report, meta)
    started = time.perf_counter()
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    try:
        for step in range(4):
            position = positions.position(step, 20)
            new_cases = position_cases(current, step, 20, position['expected_capture'])
            check_old_replay(new_cases, position['residual'])
            for case in new_cases:
                case['UB1_main'] = case['BaseY']
            old_cases, control = recompute_on_original_states(original, positions, position, quantizer)
            for old, new in zip(old_cases, new_cases):
                if old['index'] != new['index'] or any(old['scope'][key] != new['scope'][key]
                        for key in ['grid', 'full_rows', 'selected_rows', 'sampling_seed', 'inclusion_probability']):
                    raise RuntimeError('Old and new training/validation rows are not paired')
            meta['controls'].append({'step': step, 'block': 20, 'actual_new_UB1_main_captured': True,
                'all24_new_UB1_residual_replay_bitwise': True, 'original_C24_control': control})
            atomic_json(args.report, meta)
            for fold in [0, 1]:
                key = f'step{step}_fold{fold}'
                part = store.load('CV_models', key)
                if part is None:
                    train_new = [c for c in new_cases if c['index'] % 2 != fold]
                    train_old = [c for c in old_cases if c['index'] % 2 != fold]
                    validation = [c for c in new_cases if c['index'] % 2 == fold]
                    if (len(train_new) != 12 or len(validation) != 12
                            or [c['index'] for c in train_old] != [c['index'] for c in train_new]
                            or sum(c['index'] < 12 for c in train_new) != 6):
                        raise RuntimeError('Balanced paired12/12 source folds required')
                    frozen = sum(native_sse(c['StudentY'], c['TeacherY'], c['gate']) for c in validation)
                    row = {'step': step, 'block': 20, 'fold': fold, 'rank': 256, 'lambda_ridge': 1.0,
                        'train_inputs': [c['index'] for c in train_new], 'validation_inputs': [c['index'] for c in validation],
                        'frozen_UB1_on_actual_UB1_states_native_SSE': frozen,
                        'validation_states': 'actual UB1 for both refits', 'full_native_M_and_stride': True}
                    arrays = {}
                    for label, train in [('old_state_fit', train_old), ('new_state_fit', train_new)]:
                        x = torch.cat([c['V1'] for c in train])
                        y = torch.cat([(c['TeacherY'].float() - c['UB1_main'].float()) * c['gate'].float() for c in train])
                        fitted = fit_rrr(x, y, 1.0, 256)
                        parameters = native_parameters(fitted, new_cases[0]['gate'], 256)
                        row[label + '_native_SSE'] = native_risk(validation, parameters)
                        row[label + '_parameter_sha256'] = {k: tensor_sha(v) for k, v in parameters.items()}
                        arrays.update({label + '_' + k: v.float().cpu().numpy() for k, v in parameters.items()})
                        del fitted, parameters, x, y
                    candidate, matched = row['new_state_fit_native_SSE'], row['old_state_fit_native_SSE']
                    if min(frozen, matched) <= 0 or not all(math.isfinite(v) for v in [frozen, matched, candidate]):
                        raise RuntimeError('Finite nondegenerate native validation risks required')
                    row.update(candidate_over_frozen=candidate / frozen, candidate_over_matched_old_state=candidate / matched,
                        gate_pass=candidate <= .90 * frozen and candidate <= .95 * matched)
                    store.save('CV_models', key, arrays, row)
                    part = store.load('CV_models', key)
                for label in ['old_state_fit', 'new_state_fit']:
                    restored = {k: torch.from_numpy(part['arrays'][label + '_' + k]).bfloat16()
                                for k in ['first', 'second', 'bias_first', 'bias_second']}
                    if {k: tensor_sha(v) for k, v in restored.items()} != part['metadata'][label + '_parameter_sha256']:
                        raise RuntimeError('Saved native residual parameters are not lossless')
                meta['CV'].append(part['metadata'])
                atomic_json(args.report, meta)
                print('OP3 ' + json.dumps(part['metadata']), flush=True)
            del new_cases, old_cases, position
        passed = len(meta['CV']) == 8 and all(row['gate_pass'] for row in meta['CV'])
        meta.update(status='COMPLETED_OP3_STATE_MATCHED_NATIVE_GATE_PASSED' if passed else 'COMPLETED_OP3_STATE_MATCHED_NATIVE_GATE_REJECTED',
            native_gate_pass=passed, elapsed_seconds=time.perf_counter() - started,
            full_fit_condition_met=passed)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(args.report, meta)
        atomic_json(args.out / 'run_meta.json', meta)


if __name__ == '__main__':
    main()
