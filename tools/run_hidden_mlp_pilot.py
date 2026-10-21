"""HP2 fixed nonlinear head, reusing independently bound HP1 features and controls."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from evaluate_residual_rr1 import open_verified
from hidden_profile_head import predict_head, retained_fraction, support
from hidden_profile_mlp import predict, train
from onpolicy_residual import load_collections, position_cases
from router_bank_feasibility import normalized_profiles
from run_ar2_restart import ROOT, tensor_sha
from run_hidden_profile_pilot import hidden_full, read_parameters


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
    hp1path = r / 'hidden_profile_hp1/RUN.json'
    hp1, report, hp1id = open_verified(hp1path, 'COMPLETED_HP1_PROXY_GATE_REJECTED')
    if len(report['controls']) != 48 or len(report['CV']) != 4 or report['proxy_gate_pass']:
        raise RuntimeError('Complete frozen HP1 rejection and48 controls required')
    for path, sha in hp1id['sources'].items():
        if digest(path) != sha:
            raise RuntimeError('Frozen HP1 feature source changed')
    controls = {(c['step'], c['index']): c for c in report['controls']}
    records, bindings, capture = load_collections([r / f'onpolicy_calibration24/shard{s}/RUN.json' for s in [0,1]], ROOT)
    policy_path = r / 'calibration24_residual_fit/C24_RR_policy.json'
    policy = json.loads(policy_path.read_text())
    ub1path = r / 'ub1_step_budget_fit/UB1_policy.json'
    ub1 = json.loads(ub1path.read_text())
    if (bindings != hp1id['collection_bindings'] or digest(ub1path) != hp1id['UB1_policy_sha256']
            or digest(policy_path) != capture['student_policy_sha256']):
        raise RuntimeError('Exact same source-exposed HP1 operands/feature policy required')
    def bound(binding, status):
        if digest(binding['path']) != binding['sha256']:
            raise RuntimeError('Frozen feature parameter report changed')
        store, _, _ = open_verified(Path(binding['path']), status)
        if store.fingerprint != binding['fingerprint']:
            raise RuntimeError('Frozen feature fingerprint changed')
        return store
    base = bound(policy['fit_reports']['low_noise_model'], 'COMPLETED_CALIBRATION_PARAMETER_FIT')
    old = bound(policy['residual_fit'], 'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    changed = bound(ub1['changed_fits']['3'], 'COMPLETED_UB1_STEP40_FIT')
    priorpath = r / 'router_bank_rb1/RUN.json'
    prior, _, _ = open_verified(priorpath, 'COMPLETED_RB1_PROXY_GATE_REJECTED')
    if prior.fingerprint != hp1id['RB1_profile_fingerprint'] or digest(priorpath) != hp1id['RB1_report_sha256']:
        raise RuntimeError('HP1 teacher profiles changed')
    protocol = ROOT / 'config/hidden_profile_hp2.json'
    sources = [Path(__file__), ROOT / 'tools/hidden_profile_mlp.py', ROOT / 'tools/hidden_profile_head.py',
               ROOT / 'tools/run_hidden_profile_pilot.py']
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = CheckpointStore(args.out, {'protocol_sha256':digest(protocol), 'protocol':json.loads(protocol.read_text()),
        'HP1_report_sha256':digest(hp1path), 'HP1_fingerprint':hp1.fingerprint,
        'sources':{str(p):digest(p) for p in sources}, 'torch':torch.__version__, 'numpy':np.__version__, 'TF32':False})
    meta = {'status':'RUNNING', 'fingerprint':store.fingerprint, 'checkpoint_root':str(args.out.resolve()),
            'controls':[], 'CV':[], 'quality_or_speed_acceptance':False,
            'scope':'C-score only; source-exposed features, no native FFN risk or sparse deployment/quality acceptance'}
    started = time.perf_counter()
    atomic_json(args.report, meta)
    try:
        for step, keep in [(2,108),(3,162)]:
            info = base.load('positions', f'step{step}_block20')['metadata']
            cases = position_cases(records, step, 20, info)
            parameters = read_parameters(old, step) if step == 2 else read_parameters(changed, step, changed=True)
            part = prior.load('C_profiles', f'step{step}_block20')
            profiles = torch.from_numpy(normalized_profiles(part['arrays']['scores'])).cuda()
            with torch.no_grad():
                for case in cases:
                    full, ids = hidden_full(case, parameters)
                    case['hidden'] = full.reshape(-1,256).index_select(0,ids).contiguous()
                    if tensor_sha(case['hidden']) != controls[step,case['index']]['hidden_sha256']:
                        raise RuntimeError('Residual hidden values differ from frozen HP1')
                    meta['controls'].append({'step':step, 'index':case['index'], 'hidden_equals_HP1_bitwise':True})
            for fold in [0,1]:
                key = f'step{step}_fold{fold}'
                previous = store.load('heads', key)
                if previous is None:
                    training = [c for c in cases if c['index'] % 2 != fold]
                    val = [c for c in cases if c['index'] % 2 == fold]
                    oldpart = hp1.load('heads',key)
                    oldrow = oldpart['metadata']
                    if (oldrow['train_inputs'] != [c['index'] for c in training]
                            or oldrow['validation_inputs'] != [c['index'] for c in val]):
                        raise RuntimeError('HP1 source split changed')
                    def row_ids(subset):
                        return torch.tensor(np.concatenate([np.arange(c['index']*256,(c['index']+1)*256) for c in subset]),device='cuda')
                    train_profiles, val_profiles = profiles[row_ids(training)], profiles[row_ids(val)]
                    hidden_train = torch.cat([c['hidden'] for c in training])
                    head, model, mean, scale, optimization = train(hidden_train,train_profiles,20261021+100*step+fold)
                    linear = {k:torch.from_numpy(oldpart['arrays']['linear_'+k]).cuda() for k in ['weight','bias']}
                    routed, linear_routed, unnormalized = [], [], []
                    with torch.no_grad():
                        for case in val:
                            full, ids = hidden_full(case,parameters)
                            routed.append(support(predict(full,head).reshape(-1,216).index_select(0,ids),keep))
                            linear_routed.append(support(predict_head(full,linear,False).reshape(-1,216).index_select(0,ids),keep))
                            unnormalized.append(support(model((case['hidden'].float()-mean)/scale),keep))
                        selected = torch.cat(routed)
                        value = retained_fraction(val_profiles,selected)
                        control = retained_fraction(val_profiles,torch.cat(linear_routed))
                        if control != oldrow['linear_native_retained_C_fraction']:
                            raise RuntimeError('Frozen HP1 linear head retained C changed')
                        static, oracle = oldrow['static_retained_C_fraction'], oldrow['unrestricted_oracle_retained_C_fraction']
                        gain = oracle-static
                        capture = (value-static)/gain if gain > 0 else 0.
                        extra = value-control
                        disagreement = float((selected != torch.cat(unnormalized)).any(1).float().mean())
                    row = {'step':step, 'block':20, 'fold':fold, 'keep_groups':keep,
                           'train_inputs':oldrow['train_inputs'], 'validation_inputs':oldrow['validation_inputs'],
                           'static_retained_C_fraction':static, 'unrestricted_oracle_retained_C_fraction':oracle,
                           'linear_native_retained_C_fraction':control, 'MLP_native_retained_C_fraction':value,
                           'oracle_minus_static':gain, 'MLP_capture_fraction':capture, 'MLP_minus_linear':extra,
                           'gate_pass':gain>=.02 and capture>=.5 and extra>=.002,
                           'original_fullM_native_inference':True, 'HP1_linear_control_exact':True,
                           'folded_vs_unfolded_support_row_disagreement':disagreement,
                           'optimization':optimization, 'native_parameter_sha256':{k:tensor_sha(v) for k,v in head.items()}}
                    store.save('heads',key,{k:v.detach().cpu().numpy() for k,v in head.items()},row)
                    previous = store.load('heads',key)
                meta['CV'].append(previous['metadata'])
                atomic_json(args.report,meta)
                print('HP2 '+json.dumps(previous['metadata']),flush=True)
            del cases,profiles
        passed = len(meta['CV'])==4 and all(v['gate_pass'] for v in meta['CV'])
        meta.update(status='COMPLETED_HP2_PROXY_GATE_PASSED' if passed else 'COMPLETED_HP2_PROXY_GATE_REJECTED',
                    proxy_gate_pass=passed,elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE',error=str(error))
        raise
    finally:
        atomic_json(args.report,meta)
        atomic_json(args.out/'run_meta.json',meta)


if __name__ == '__main__':
    main()
