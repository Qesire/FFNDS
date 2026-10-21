"""R4E1: small equal-MAC original-group swaps with matched fixed-h compensation."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from anchored_reconstruction import bf16_bits, from_bf16_bits
from ar2_checkpoint import CheckpointStore, atomic_json, digest
from compensated_support import FixedDesign, discrepancy_certificate, finite_swaps, linear_upper_bound, proposal_pool
from historical_source_snapshot import HistoricalSources
from onpolicy_residual import native_sse, replay_rows, same_bits
from r4_native_support import action_delta, full_parameters, group_contribution, matched_parameters, native_mains, reconstruct_teacher_v2
from run_ar2_restart import ROOT, tensor_sha
from run_hidden_profile_pilot import hidden_full
from ub1_native_data import load_ub1_collections
from ub1_native_replay import UB1Positions, replay_position


def selected_support(kept, action):
    return np.array(sorted((set(map(int, kept)) - set(action['removed'])) | set(action['added'])), dtype=np.int64)


def predictions(cases, mains, parameters):
    return {c['index']: mains[c['index']] + replay_rows(c['V1'], c['scope'], parameters) for c in cases}


def risk(cases, values):
    return sum(native_sse(values[c['index']], c['TeacherY'], c['gate']) for c in cases)


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
        raise RuntimeError('HF derived arrays and80GiB node reserve plus2GiB allowance required')
    runs = ROOT / 'runs/worker_restart_20261020'
    readiness_path = runs / 'r4e0_operand_readiness/RUN.json'
    readiness = json.loads(readiness_path.read_text())
    if readiness['status'] != 'PASS_R4E0_ALL48_NATIVE_OPERANDS_AND_COEFFICIENTS_BOUND_CPU_ONLY':
        raise RuntimeError('R4E0 current operand binding must pass first')
    view = json.loads((runs / 'source_view_native_equivalence/CONTROL.json').read_text())
    receipt = json.loads(args.source_view_receipt.read_text())
    if (view['status'] != 'PASS_EXACT_SOURCE_VIEW_ALL192_NATIVE_CONTROLS_BYTE_IDENTICAL'
            or receipt['status'] != 'RUNNING_CHILD_WITH_VERIFIED_SOURCE_VIEW'
            or receipt['view_manifest_sha256'] != view['view_manifest_sha256']
            or receipt['source_inventory_sha256'] != view['source_inventory_sha256']):
        raise RuntimeError('Exact dependency source-view controls required')
    records, bindings, capture = load_ub1_collections(
        [runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT)
    if bindings != readiness['collection_bindings']:
        raise RuntimeError('OP2 operands changed after R4E0')
    positions = UB1Positions(runs / 'ub1_step_budget_fit/UB1_policy.json', HistoricalSources(),
                            expected_policy_sha256=capture['original_UB1_policy_sha256'])
    protocol_path = ROOT / 'config/r4_compensated_support_screen.json'
    protocol = json.loads(protocol_path.read_text())
    if digest(protocol_path) != readiness['protocol_sha256']:
        raise RuntimeError('R4 registered finite scope changed after readiness')
    sources = ['run_r4_support_screen.py', 'r4_native_support.py', 'compensated_support.py',
        'ub1_native_replay.py', 'ub1_native_data.py', 'onpolicy_residual.py', 'folded_adapter.py',
        'run_hidden_profile_pilot.py', 'historical_source_snapshot.py', 'with_transformers_source_view.py']
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {'protocol_sha256': digest(protocol_path), 'protocol': protocol,
        'readiness_sha256': digest(readiness_path), 'collection_bindings': bindings,
        'UB1_policy_sha256': positions.policy_sha256,
        'view_manifest_sha256': view['view_manifest_sha256'],
        'sources': {str(ROOT / 'tools' / n): digest(ROOT / 'tools' / n) for n in sources},
        'native_quantizer_sha256': digest(ROOT / '.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py'),
        'torch': torch.__version__, 'numpy': np.__version__, 'TF32': False}
    store = CheckpointStore(args.out, identity)
    meta = {'status': 'RUNNING', 'fingerprint': store.fingerprint, 'checkpoint_root': str(args.out),
        'protocol_sha256': digest(protocol_path), 'readiness_sha256': digest(readiness_path),
        'source_view_execution_receipt': str(args.source_view_receipt), 'controls': [], 'CV': [],
        'quality_or_speed_acceptance': False,
        'scope': 'finite native fixed-h support/readout screening on known sampled OP2 states; not whole-M population, whole-pipeline holdout or video acceptance'}
    atomic_json(args.report, meta)
    started = time.perf_counter()
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as quantizer
    try:
        for step, block in protocol['positions']:
            position = positions.position(step, block)
            cases = replay_position(records, position, quantizer, require_captured_UB1=True)
            full = full_parameters(positions, position)
            for case in cases:
                key = f'step{step}_block{block}_input{case["index"]}'
                part = store.load('teacher_V2', key)
                if part is None:
                    v2 = reconstruct_teacher_v2(case, full, quantizer)
                    store.save('teacher_V2', key, {'V2_bits': bf16_bits(v2)}, {
                        'native_full_TeacherY_bitwise': True, 'V2_sha256': tensor_sha(v2),
                        'observed_payload_sha256': case['scope']['sha256'], 'shard_sha256': full['shard_sha256']})
                    part = store.load('teacher_V2', key)
                v2 = from_bf16_bits(part['arrays']['V2_bits'], device='cuda')
                if (tensor_sha(v2) != part['metadata']['V2_sha256']
                        or part['metadata']['observed_payload_sha256'] != case['scope']['sha256']):
                    raise RuntimeError('Recovered teacher V2 cache differs')
                case['A'] = v2
                hidden, ids = hidden_full(case, position['residual'])
                split = F.linear(hidden, position['residual']['second'], position['residual']['bias_second']).reshape(-1, 5120).index_select(0, ids)
                if not same_bits(split, case['UB1_frozen_R']):
                    raise RuntimeError('Frozen split R256 bits changed')
                case['hidden'] = hidden.reshape(-1, 256).index_select(0, ids).contiguous()
            meta['controls'].append({'step': step, 'block': block, 'inputs': len(cases),
                'full_native_teacher_Y_bitwise': True, 'frozen_UB1_main_and_student_bitwise': True,
                'split_R256_bitwise': True, 'full_native_M_and_stride': True})
            atomic_json(args.report, meta)
            print(f'R4E1 step{step} all24 native teacher/main/R controls passed', flush=True)
            kept = list(map(int, position['groups']))
            for fold in [0, 1]:
                key = f'step{step}_block{block}_fold{fold}'
                part = store.load('CV', key)
                if part is None:
                    train, val = [[c for c in cases if (c['index'] % 2 == fold) == validation] for validation in [False, True]]
                    if len(train) != 12 or len(val) != 12 or sum(c['index'] < 12 for c in train) != 6:
                        raise RuntimeError('Fixed balanced source folds required')
                    gate = train[0]['gate'].double()
                    if any(not same_bits(c['gate'], train[0]['gate']) for c in cases):
                        raise RuntimeError('Projection requires the original source-constant output gate')
                    atrain, aval = [torch.cat([c['A'] for c in group]) for group in [train, val]]
                    htrain, hval = [torch.cat([c['hidden'] for c in group]) for group in [train, val]]
                    design = FixedDesign(htrain)
                    teacher_train = torch.cat([c['TeacherY'] for c in train]).double()
                    main0_train = torch.cat([c['UB1_main'] for c in train]).double()
                    error = design.residual(main0_train - teacher_train) * gate
                    ideal_risk = float(error.square().sum())
                    if not math.isfinite(ideal_risk) or ideal_risk <= 0:
                        raise RuntimeError('Nondegenerate projected proxy risk required')
                    linear, c_scores = [], []
                    for group in range(216):
                        contribution = group_contribution(atrain, full, group, position['alpha']) * gate
                        projected = design.residual(contribution)
                        linear.append(float(-2 * (error * projected).sum()))
                        c_scores.append(float(contribution.square().sum()) / len(atrain))
                    removed, added = proposal_pool(kept, linear, width=8)
                    pool = removed + added
                    projected = torch.stack([design.residual(group_contribution(atrain, full, g, position['alpha']) * gate).reshape(-1) for g in pool])
                    gram = (projected @ projected.T).cpu().numpy()
                    del projected
                    search = finite_swaps(removed, added, linear, gram, q=2)
                    candidate = search['best_nonzero']
                    q = candidate['swap_count']
                    omitted = sorted(set(range(216)) - set(kept))
                    c_action = {'removed': sorted(sorted(kept, key=lambda g: (c_scores[g], g))[:q]),
                        'added': sorted(sorted(omitted, key=lambda g: (-c_scores[g], g))[:q])}
                    rng = np.random.default_rng(20261021 + 100 * step + fold)
                    random_action = {'removed': sorted(map(int, rng.choice(kept, q, replace=False))),
                        'added': sorted(map(int, rng.choice(omitted, q, replace=False)))}
                    row = {'step': step, 'block': block, 'fold': fold,
                        'train_inputs': [c['index'] for c in train], 'validation_inputs': [c['index'] for c in val],
                        'design_rank': design.rank, 'design_condition': design.condition, 'SVD_rtol': design.rtol,
                        'ideal_K0_training_projected_SSE': ideal_risk,
                        'linear_relaxation_upper_bound': linear_upper_bound(kept, linear, q=2),
                        'proxy_best_including_identity': search['best'], 'candidate': candidate,
                        'finite_candidates': search['candidate_count_including_identity'], 'proposal_pool': pool,
                        'all216_linear_scores': linear, 'C_control': c_action, 'random_control': random_action,
                        'native_risks': {}, 'numeric_discrepancy': {}}
                    actions = {'K1': candidate, 'C': c_action, 'random': random_action}
                    all_mains = {'K0': {c['index']: c['UB1_main'] for c in cases}}
                    for label, action in actions.items():
                        all_mains[label] = native_mains(cases, selected_support(kept, action), full, position, quantizer)
                    values, arrays = {}, {'proxy_gram': gram, 'linear_scores': np.asarray(linear)}
                    for label in ['K0', 'K1', 'C', 'random']:
                        mains = all_mains[label]
                        target = teacher_train - torch.cat([mains[c['index']] for c in train]).double()
                        _, parameters = matched_parameters(design, target, position['residual'])
                        values[label + '_matched'] = predictions(val, mains, parameters)
                        values[label + '_frozen'] = {c['index']: mains[c['index']] + c['UB1_frozen_R'] for c in val}
                        for suffix in ['frozen', 'matched']:
                            name = label + '_' + suffix
                            row['native_risks'][name] = risk(val, values[name])
                        row[label + '_readout_sha256'] = {k: tensor_sha(parameters[k]) for k in ['second', 'bias_second']}
                        arrays.update({label + '_' + k + '_bits': bf16_bits(parameters[k]) for k in ['second', 'bias_second']})
                    baseline_frozen = torch.cat([values['K0_frozen'][c['index']] * c['gate'] for c in val]).double()
                    baseline_matched = torch.cat([values['K0_matched'][c['index']] * c['gate'] for c in val]).double()
                    target_val = torch.cat([c['TeacherY'] * c['gate'] for c in val]).double()
                    phi_val = FixedDesign.features(hval)
                    for label, action in actions.items():
                        dtrain = action_delta(atrain, full, action, position['alpha'])
                        dval = action_delta(aval, full, action, position['alpha'])
                        proxy_matched_delta = (dval - phi_val @ design.fit(dtrain)) * gate
                        row['numeric_discrepancy'][label] = {}
                        for suffix, baseline, proxy_delta in [('frozen', baseline_frozen, dval * gate),
                                ('matched', baseline_matched, proxy_matched_delta)]:
                            output = torch.cat([values[label + '_' + suffix][c['index']] * c['gate'] for c in val]).double()
                            row['numeric_discrepancy'][label][suffix] = discrepancy_certificate(
                                baseline - target_val, proxy_delta, output - baseline)
                    risks = row['native_risks']
                    if not all(math.isfinite(v) and v > 0 for v in risks.values()):
                        raise RuntimeError('Finite positive native risks required')
                    row.update(K1_matched_over_K0_matched=risks['K1_matched'] / risks['K0_matched'],
                        K1_matched_over_frozen_UB1=risks['K1_matched'] / risks['K0_frozen'],
                        K1_matched_over_C_matched=risks['K1_matched'] / risks['C_matched'],
                        K1_matched_over_random_matched=risks['K1_matched'] / risks['random_matched'],
                        gate_pass=candidate['proxy_advantage'] > 0 and risks['K1_matched'] <= .98 * risks['K0_matched']
                            and risks['K1_matched'] <= .98 * risks['K0_frozen']
                            and risks['K1_matched'] <= .99 * risks['C_matched']
                            and risks['K1_matched'] <= .99 * risks['random_matched'],
                        actual_native_original_fullM=True, same_main_and_aux_GEMM_budget=True)
                    row['per_source_K0_K1_matched_native_SSE'] = [{'index': c['index'],
                        'K0': native_sse(values['K0_matched'][c['index']], c['TeacherY'], c['gate']),
                        'K1': native_sse(values['K1_matched'][c['index']], c['TeacherY'], c['gate'])} for c in val]
                    store.save('CV', key, arrays, row)
                    part = store.load('CV', key)
                    del design, atrain, aval, htrain, hval, error, all_mains, values, arrays
                for label in ['K0', 'K1', 'C', 'random']:
                    restored = {k: from_bf16_bits(part['arrays'][label + '_' + k + '_bits']) for k in ['second', 'bias_second']}
                    if {k: tensor_sha(v) for k, v in restored.items()} != part['metadata'][label + '_readout_sha256']:
                        raise RuntimeError('Saved native readout coefficient bits changed')
                meta['CV'].append(part['metadata'])
                atomic_json(args.report, meta)
                print('R4E1 ' + json.dumps({k: part['metadata'][k] for k in ['step', 'fold', 'K1_matched_over_K0_matched', 'K1_matched_over_frozen_UB1', 'gate_pass']}), flush=True)
            del cases, full, position
        passed = len(meta['CV']) == 4 and all(v['gate_pass'] for v in meta['CV'])
        meta.update(status='COMPLETED_R4E1_FINITE_NATIVE_SUPPORT_GATE_PASSED' if passed else 'COMPLETED_R4E1_FINITE_NATIVE_SUPPORT_GATE_REJECTED',
            native_gate_pass=passed, elapsed_seconds=time.perf_counter() - started,
            route_training='NOT_RUN_REQUIRES_SEPARATE_WHOLE_M_ACTION_VALUE_PROTOCOL' if passed else 'NOT_RUN_R4E1_FINITE_SCREEN_REJECTED')
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(args.report, meta)
        atomic_json(args.out / 'run_meta.json', meta)


if __name__ == '__main__':
    main()
