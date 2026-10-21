"""OP2R control only: all four UB1 widths on actual new and original old states."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil

import torch

from ar2_checkpoint import atomic_json, digest
from historical_source_snapshot import HistoricalSources
from onpolicy_residual import native_sse
from original_onpolicy_data import load_original_collections
from run_ar2_restart import ROOT
from ub1_native_data import load_ub1_collections
from ub1_native_replay import UB1Positions, recompute_on_original_states, replay_position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if shutil.disk_usage('/var/tmp').free < 80 << 30:
        raise RuntimeError('Preserve80GiB worker node reserve')
    runs = ROOT / 'runs/worker_restart_20261020'
    condition_path = runs / 'op2d_native_distribution/RUN.json'
    condition = json.loads(condition_path.read_text())
    if condition['status'] != 'COMPLETED_OP2D_NATIVE_STATE_DISTRIBUTION_DESCRIPTION':
        raise RuntimeError('Complete all24 native state descriptions before replay controls')
    history = HistoricalSources()
    original, old_bindings, old_identity = load_original_collections(
        [runs / f'onpolicy_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT, history)
    current, new_bindings, new_identity = load_ub1_collections(
        [runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT)
    if old_identity['assets'] != new_identity['assets']:
        raise RuntimeError('New and original native collection assets differ')
    policy_path = runs / 'ub1_step_budget_fit/UB1_policy.json'
    positions = UB1Positions(policy_path, history, expected_policy_sha256=new_identity['original_UB1_policy_sha256'])
    if (positions.policy['parent_policy']['sha256'] != old_identity['student_policy_sha256']
            or positions.policy['cache_manifest']['sha256'] != new_identity['assets']['cache_manifest_sha256']):
        raise RuntimeError('Frozen UB1 coefficients and the two collections differ')
    prior_path = runs / 'nonlinear_residual_nf1/RUN.json'
    prior = json.loads(prior_path.read_text())
    if prior['status'] != 'COMPLETED_NF1_NATIVE_GATE_REJECTED':
        raise RuntimeError('Complete preserved NF1 baseline controls are required')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / 'RUN.json'
    if path.exists():
        raise RuntimeError('Native replay control records are immutable')
    sources = ['ub1_native_replay.py', 'verify_ub1_native_replay.py', 'folded_adapter.py',
        'onpolicy_residual.py', 'run_folded_adapter_cv.py', 'reduced_rank_residual.py',
        'ub1_native_data.py', 'original_onpolicy_data.py', 'historical_source_snapshot.py']
    meta = {'status': 'RUNNING', 'protocol_sha256': digest(ROOT / 'config/ub1_native_replay_op2r.json'),
        'condition_sha256': digest(condition_path), 'original_policy_sha256': positions.policy_sha256,
        'old_bindings': old_bindings, 'new_bindings': new_bindings, 'prior_NF1_sha256': digest(prior_path),
        'sources': {str(ROOT / 'tools' / name): digest(ROOT / 'tools' / name) for name in sources},
        'controls': [], 'prior_risk_controls': [], 'quality_or_speed_acceptance': False,
        'scope': 'full-native reconstruction controls only; no coefficient fitting or new policy acceptance'}
    atomic_json(path, meta)
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    try:
        for step in range(4):
            position = positions.position(step, 20)
            new_cases = replay_position(current, position, quantizer, require_captured_UB1=True)
            old_cases, old_control = recompute_on_original_states(original, positions, position, quantizer)
            meta['controls'].append({'step': step, 'block': 20, 'new_actual_UB1_sources': len(new_cases),
                'new_BaseY_and_StudentY_bitwise': True, 'original_C24_sources': len(old_cases),
                'original_C24_control': old_control, 'keep_groups': len(position['groups']),
                'frozen_R_parameter_sha256': position['expected_residual_sha256']})
            if step in [2, 3]:
                for fold in [0, 1]:
                    reference = next(v for v in prior['CV'] if v['step'] == step and v['fold'] == fold)
                    observed = sum(native_sse(c['UB1_main'] + c['UB1_frozen_R'], c['TeacherY'], c['gate'])
                                   for c in old_cases if c['index'] % 2 == fold)
                    expected = reference['frozen_UB1_native_SSE']
                    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-9):
                        raise RuntimeError('Original NF1 frozen baseline native risk differs')
                    meta['prior_risk_controls'].append({'step': step, 'fold': fold,
                        'observed_native_SSE': observed, 'expected_NF1_native_SSE': expected,
                        'exact_float_equal': observed == expected, 'within_registered_tolerance': True})
            atomic_json(path, meta)
            print(f'OP2R step{step}: 24 actual UB1 and24 original C24 native controls passed', flush=True)
            del position, new_cases, old_cases
        if len(meta['controls']) != 4 or len(meta['prior_risk_controls']) != 4:
            raise RuntimeError('Incomplete registered native replay scope')
        meta.update(status='PASS_OP2R_ALL_FOUR_STEPS_BOTH_NATIVE_STATE_CONTROLS',
                    actual_UB1_native_positions_checked=96, original_C24_native_positions_checked=96)
    except BaseException as error:
        meta.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(path, meta)


if __name__ == '__main__':
    main()
