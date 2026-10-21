"""Test a fixed per-group head on frozen UB1 residual features, offline only."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from evaluate_residual_rr1 import open_verified
from hidden_profile_head import fit_head, predict_head, retained_fraction, support
from onpolicy_residual import check_old_replay, load_collections, position_cases, replay_rows, same_bits
from router_bank_feasibility import normalized_profiles
from run_ar2_restart import ROOT, tensor_sha


@torch.no_grad()
def hidden_full(case, parameters):
    scope, x = case['scope'], case['V1']
    shape, stride = tuple(scope['native_V1_shape']), tuple(scope['native_V1_stride'])
    if (shape[-1] != 5120 or len(shape) not in [2, 3]
            or int(np.prod(shape[:-1])) != scope['full_rows']
            or (len(shape) == 3 and shape[0] != 1)):
        raise RuntimeError('Unsupported original residual geometry')
    ids = torch.tensor(scope['selected_rows'], device=x.device, dtype=torch.long)
    if (len(ids) != 256 or len(ids.unique()) != 256 or int(ids.min()) < 0
            or int(ids.max()) >= scope['full_rows']):
        raise RuntimeError('Invalid frozen sampled rows')
    full = torch.empty_strided(shape, stride, dtype=x.dtype, device=x.device).zero_()
    (full if len(shape) == 2 else full[0]).index_copy_(0, ids, x)
    hidden = F.linear(full, parameters['first'], parameters['bias_first'])
    return hidden, ids


def read_parameters(store, step, changed=False):
    part = store.load('positions', f'step{step}_block20')
    if part is None:
        raise RuntimeError('Missing frozen residual position')
    arrays = part['arrays']
    parameters = {(k[3:] if changed else k): torch.from_numpy(v).to(device='cuda', dtype=torch.bfloat16)
                  for k, v in arrays.items() if not changed or k.startswith('rr_')}
    expected = part['metadata']['RR_parameter_sha256' if changed else 'parameter_sha256']
    if {k: tensor_sha(v) for k, v in parameters.items()} != expected:
        raise RuntimeError('Frozen residual parameter bits changed')
    return parameters


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    r = ROOT / 'runs/worker_restart_20261020'
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if (not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 82 * 1024**3):
        raise RuntimeError('HF derived cache and80GiB reserve plus2GiB allowance required')
    rb2path = r / 'router_bank_rb2/RUN.json'
    rb2 = json.loads(rb2path.read_text())
    if rb2['status'] != 'COMPLETED_RB2_PROXY_GATE_REJECTED':
        raise RuntimeError('Finish the frozen balanced-bank trial first')
    records, bindings, capture = load_collections([r / f'onpolicy_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT)
    policy_path = r / 'calibration24_residual_fit/C24_RR_policy.json'
    policy = json.loads(policy_path.read_text())
    if digest(policy_path) != capture['student_policy_sha256']:
        raise RuntimeError('Original C24 observation policy changed')
    prior_path = r / 'router_bank_rb1/RUN.json'
    prior, _, prior_id = open_verified(prior_path, 'COMPLETED_RB1_PROXY_GATE_REJECTED')
    if (prior_id['collection_bindings'] != bindings
            or prior_id['student_policy_sha256'] != digest(policy_path)
            or prior_id['derived_report_sha256'] != digest(r / 'dense_v2_native_replay/RUN.json')):
        raise RuntimeError('Frozen teacher profile source changed')
    for path, sha in prior_id['sources'].items():
        if digest(path) != sha:
            raise RuntimeError('Frozen teacher profile source code changed')
    def bound_store(binding, status):
        if digest(binding['path']) != binding['sha256']:
            raise RuntimeError('Frozen parameter report changed')
        store, _, _ = open_verified(Path(binding['path']), status)
        if store.fingerprint != binding['fingerprint']:
            raise RuntimeError('Frozen parameter fingerprint changed')
        return store
    base = bound_store(policy['fit_reports']['low_noise_model'], 'COMPLETED_CALIBRATION_PARAMETER_FIT')
    old = bound_store(policy['residual_fit'], 'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    ub1path = r / 'ub1_step_budget_fit/UB1_policy.json'
    ub1 = json.loads(ub1path.read_text())
    quality = json.loads((r / 'disjoint20_UB1_comparison/COMPARISON.json').read_text())
    if (digest(ub1path) != quality['candidate_policy_sha256']
            or ub1['parent_policy']['sha256'] != digest(policy_path)):
        raise RuntimeError('Frozen UB1 feature policy differs')
    changed = bound_store(ub1['changed_fits']['3'], 'COMPLETED_UB1_STEP40_FIT')
    protocol = ROOT / 'config/hidden_profile_hp1.json'
    sources = [Path(__file__), ROOT / 'tools/hidden_profile_head.py', ROOT / 'tools/onpolicy_residual.py',
               ROOT / 'tools/reduced_rank_residual.py', ROOT / 'tools/router_bank_feasibility.py']
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = CheckpointStore(args.out, {'protocol_sha256': digest(protocol), 'protocol': json.loads(protocol.read_text()),
        'collection_bindings': bindings, 'UB1_policy_sha256': digest(ub1path),
        'RB1_profile_fingerprint': prior.fingerprint, 'RB1_report_sha256': digest(prior_path),
        'RB2_report_sha256': digest(rb2path), 'sources': {str(p): digest(p) for p in sources},
        'torch': torch.__version__, 'numpy': np.__version__, 'TF32': False})
    meta = {'status': 'RUNNING', 'fingerprint': store.fingerprint, 'checkpoint_root': str(args.out.resolve()),
            'controls': [], 'CV': [], 'scope': 'per-row C-score proxy only; no sparse deployment or native FFN risk/quality acceptance',
            'feature_source_exposure': 'frozen student and residual parameters already saw all24 sources; not whole-pipeline held-out',
            'quality_or_speed_acceptance': False}
    started = time.perf_counter()
    atomic_json(args.report, meta)
    try:
        for step, keep in [(2, 108), (3, 162)]:
            info = base.load('positions', f'step{step}_block20')['metadata']
            cases = position_cases(records, step, 20, info)
            old_parameters = read_parameters(old, step)
            check_old_replay(cases, old_parameters)
            parameters = old_parameters if step == 2 else read_parameters(changed, step, changed=True)
            part = prior.load('C_profiles', f'step{step}_block20')
            if (part is None or part['metadata']['gate_sha256'] != tensor_sha(cases[0]['gate'])
                    or part['metadata']['weight_sha256'] != info['weight_sha256']
                    or [b['observed_payload_sha256'] for b in part['metadata']['bindings']] != [c['scope']['sha256'] for c in cases]):
                raise RuntimeError('Exact frozen teacher-profile rows/operands required')
            profiles = torch.from_numpy(normalized_profiles(part['arrays']['scores'])).cuda()
            for case in cases:
                hidden, ids = hidden_full(case, parameters)
                split = F.linear(hidden, parameters['second'], parameters['bias_second']).reshape(-1, 5120).index_select(0, ids)
                reference = replay_rows(case['V1'], case['scope'], parameters)
                if not same_bits(split, reference):
                    raise RuntimeError('Splitting the native residual changed its output bits')
                case['hidden'] = hidden.reshape(-1, 256).index_select(0, ids).contiguous()
                meta['controls'].append({'step': step, 'index': case['index'], 'original_C24_StudentY_bitwise': True,
                    'UB1_split_residual_replay_bitwise': True, 'hidden_sha256': tensor_sha(case['hidden'])})
            for fold in [0, 1]:
                key = f'step{step}_fold{fold}'
                previous = store.load('heads', key)
                if previous is None:
                    train = [c for c in cases if c['index'] % 2 != fold]
                    val = [c for c in cases if c['index'] % 2 == fold]
                    if len(train) != 12 or len(val) != 12 or sum(c['index'] < 12 for c in train) != 6:
                        raise RuntimeError('Fixed balanced12/12 source split required')
                    train_ids = torch.tensor(np.concatenate([np.arange(c['index']*256, (c['index']+1)*256) for c in train]), device='cuda')
                    val_ids = torch.tensor(np.concatenate([np.arange(c['index']*256, (c['index']+1)*256) for c in val]), device='cuda')
                    train_profiles, val_profiles = profiles[train_ids], profiles[val_ids]
                    static_ids = support(train_profiles.mean(0, keepdim=True), keep).expand(len(val_profiles), -1)
                    static = retained_fraction(val_profiles, static_ids)
                    oracle = retained_fraction(val_profiles, support(val_profiles, keep))
                    hidden_train = torch.cat([c['hidden'] for c in train])
                    row = {'step': step, 'block': 20, 'fold': fold, 'keep_groups': keep,
                           'train_inputs': [c['index'] for c in train], 'validation_inputs': [c['index'] for c in val],
                           'static_retained_C_fraction': static, 'unrestricted_oracle_retained_C_fraction': oracle}
                    arrays = {}
                    for quadratic, label in [(False, 'linear'), (True, 'quadratic')]:
                        head = fit_head(hidden_train, train_profiles, quadratic)
                        selected = []
                        for case in val:
                            full, ids = hidden_full(case, parameters)
                            scores = predict_head(full, head, quadratic).reshape(-1, 216).index_select(0, ids)
                            selected.append(support(scores, keep))
                        value = retained_fraction(val_profiles, torch.cat(selected))
                        row[label + '_native_retained_C_fraction'] = value
                        row[label + '_native_parameter_sha256'] = {k: tensor_sha(v) for k, v in head.items()}
                        arrays.update({label+'_'+k: v.cpu().numpy() for k, v in head.items()})
                    improvement = oracle - static
                    capture = (row['quadratic_native_retained_C_fraction'] - static) / improvement if improvement > 0 else 0.
                    extra = row['quadratic_native_retained_C_fraction'] - row['linear_native_retained_C_fraction']
                    row.update(status='COMPLETED_HP1_FIXED_HEAD_FOLD', oracle_minus_static=improvement,
                               quadratic_capture_fraction=capture, quadratic_minus_linear=extra,
                               gate_pass=improvement >= .02 and capture >= .5 and extra >= .002,
                               original_fullM_native_inference=True)
                    store.save('heads', key, arrays, row)
                    previous = store.load('heads', key)
                meta['CV'].append(previous['metadata'])
                atomic_json(args.report, meta)
                print('HP1 '+json.dumps(previous['metadata']), flush=True)
            del cases, profiles
        passed = len(meta['CV']) == 4 and all(v['gate_pass'] for v in meta['CV'])
        meta.update(status='COMPLETED_HP1_PROXY_GATE_PASSED' if passed else 'COMPLETED_HP1_PROXY_GATE_REJECTED',
                    proxy_gate_pass=passed, elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(args.report, meta)
        atomic_json(args.out / 'run_meta.json', meta)


if __name__ == '__main__':
    main()
