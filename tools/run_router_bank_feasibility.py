"""Run the preregistered RB1 C-score routing feasibility experiment, offline only."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from anchored_reconstruction import bf16_bits
from ar2_checkpoint import CheckpointStore, atomic_json, canonical_json, digest
from evaluate_residual_rr1 import open_verified
from onpolicy_residual import load_collections, position_cases
from router_bank_feasibility import (DegenerateBanks, bank_energy, describe_proxy, fit_banks,
                                     fit_router, folded_parameters, native_predict_rows, normalized_profiles)
from run_ar2_restart import ROOT, tensor_sha


def position_profiles(cases, derived, weight):
    activations, bindings = [], []
    for case in cases:
        row = derived[case['index']]
        if (digest(row['path']) != row['sha256'] or not row['Y_bitwise_equals_actual']
                or row['source_operand_sha256'] != case['scope']['sha256']
                or row['Y_sha256'] != tensor_sha(case['TeacherY'])):
            raise RuntimeError('Derived teacher V2 does not match the observed native teacher')
        with safe_open(row['path'], framework='pt', device='cpu') as handle:
            if set(handle.keys()) != {'V2'}:
                raise RuntimeError('Unexpected derived teacher fields')
            value = handle.get_tensor('V2')
        if (value.dtype != torch.bfloat16 or value.shape != (256, 13824)
                or not torch.isfinite(value).all() or tensor_sha(value) != row['V2_sha256']):
            raise RuntimeError('Derived teacher V2 shape/bits changed')
        activations.append(value.cuda())
        bindings.append({'index': case['index'], 'V2_payload_sha256': row['sha256'],
                         'observed_payload_sha256': case['scope']['sha256']})
    activation = torch.cat(activations).float()
    w, gate = weight.float(), cases[0]['gate'].float()
    scores = np.empty((6144, 216), dtype=np.float64)
    with torch.no_grad():
        for group in range(216):
            sl = slice(group * 64, (group + 1) * 64)
            signal = F.linear(activation[:, sl].contiguous(), w[:, sl].contiguous(), None) * gate
            scores[:, group] = signal.double().square().sum(1).cpu().numpy()
    normalized_profiles(scores)  # Reject zero/nonfinite energy before storing.
    return scores, {'bindings': bindings, 'shape': list(scores.shape),
                    'weight_sha256': tensor_sha(weight), 'gate_sha256': tensor_sha(cases[0]['gate'])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections', type=Path, nargs=2, required=True)
    ap.add_argument('--student-policy', type=Path, required=True)
    ap.add_argument('--derived-report', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    run_root = ROOT / 'runs/worker_restart_20261020'
    supplemental_path = run_root / 'disjoint20_pipeline/RUN.json'
    supplemental = json.loads(supplemental_path.read_text())
    if supplemental['status'] != 'COMPLETED_FROZEN_SOURCE_DISJOINT20_SUPPLEMENT':
        raise RuntimeError('RB1 waits for the existing supplemental evaluation to finish')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if (not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 82 * 1024 ** 3):
        raise RuntimeError('RB1 checkpoints require node HF cache and80GiB reserve plus2GiB allowance')
    records, bindings, capture = load_collections(args.collections, ROOT)
    if digest(args.student_policy) != capture['student_policy_sha256']:
        raise RuntimeError('Original C24 observation policy changed')
    policy = json.loads(args.student_policy.read_text())
    cache_binding = policy['cache_manifest']
    if digest(cache_binding['path']) != cache_binding['sha256']:
        raise RuntimeError('Original cache manifest changed')
    cache = json.loads(Path(cache_binding['path']).read_text())
    replay = json.loads(args.derived_report.read_text())
    replay_root = Path(replay['checkpoint_root'])
    replay_identity = json.loads((replay_root / 'identity.json').read_text())
    source_id = replay_identity['identity']
    if (replay['status'] != 'PASS_DENSE_FULL_GEOMETRY_V2_Y_CONTROLS_AND48_STUDENT_TEACHER_Y_REPLAYS'
            or len(replay['derived']) != 48 or len(replay['dense_controls']) != 4
            or not all(v['Y_bitwise_equals_actual'] and v['V2_bitwise_equals_actual_dense_capture'] for v in replay['dense_controls'])
            or replay_identity['fingerprint'] != replay['fingerprint']
            or hashlib.sha256(canonical_json(source_id).encode()).hexdigest() != replay['fingerprint']
            or source_id['onpolicy_collections'] != bindings
            or source_id['capture_identity'] != capture
            or source_id['protocol_sha256'] != digest(ROOT / 'config/dense_v2_replay_control.json')
            or source_id['cache_manifest_sha256'] != cache_binding['sha256']):
        raise RuntimeError('Complete same-source native teacher replay controls required')
    for path, sha in source_id['sources'].items():
        if digest(path) != sha:
            raise RuntimeError('Derived teacher numerical source changed')
    shard = next(v for v in cache['shards'] if v['expert'] == 'low_noise_model' and v['block'] == 20)
    if digest(shard['path']) != shard['sha256'] or source_id['cache_shard_sha256'] != shard['sha256']:
        raise RuntimeError('Original effective low-noise block20 weights changed')
    payload = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)['low_noise_model']
    weight = payload['blocks.20.ffn.2']['weight_qdq'].cuda()
    base_binding = policy['fit_reports']['low_noise_model']
    if digest(base_binding['path']) != base_binding['sha256']:
        raise RuntimeError('Original C24 base report changed')
    base, _, _ = open_verified(Path(base_binding['path']), 'COMPLETED_CALIBRATION_PARAMETER_FIT')
    if base.fingerprint != base_binding['fingerprint']:
        raise RuntimeError('C24 base identity changed')
    protocol_path = ROOT / 'config/router_bank_rb1.json'
    sources = [Path(__file__), ROOT / 'tools/router_bank_feasibility.py', ROOT / 'tools/onpolicy_residual.py',
               ROOT / 'tools/anchored_reconstruction.py']
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = CheckpointStore(args.out, {'protocol_sha256': digest(protocol_path),
        'protocol': json.loads(protocol_path.read_text()), 'collection_bindings': bindings,
        'student_policy_sha256': digest(args.student_policy), 'derived_report_sha256': digest(args.derived_report),
        'derived_fingerprint': replay['fingerprint'], 'supplemental_completion_sha256': digest(supplemental_path),
        'sources': {str(p): digest(p) for p in sources}, 'torch': torch.__version__, 'numpy': np.__version__, 'TF32': False})
    meta = {'status': 'RUNNING', 'fingerprint': store.fingerprint, 'checkpoint_root': str(args.out.resolve()),
            'profiles': [], 'CV': [], 'scope': 'C-score proxy only; native FFN risk and deployment not tested',
            'development_source_exposure': 'C24 student used all24 sources; source-split proxy training is not a whole-pipeline held-out test',
            'quality_or_speed_acceptance': False}
    atomic_json(args.report, meta)
    started = time.perf_counter()
    try:
        for step, keep in [(2, 108), (3, 162)]:
            info = base.load('positions', f'step{step}_block20')['metadata']
            if tensor_sha(weight) != info['weight_sha256']:
                raise RuntimeError('Full effective teacher W2 differs from original C24')
            cases = position_cases(records, step, 20, info)
            derived = {v['index']: v for v in replay['derived'] if v['step'] == step and v['block'] == 20}
            if set(derived) != set(range(24)):
                raise RuntimeError('Exactly24 matching teacher V2 payloads required')
            profile_key = f'step{step}_block20'
            cached = store.load('C_profiles', profile_key)
            if cached is None:
                scores, data = position_profiles(cases, derived, weight)
                store.save('C_profiles', profile_key, {'scores': scores}, data)
                cached = store.load('C_profiles', profile_key)
            profiles = normalized_profiles(cached['arrays']['scores'])
            meta['profiles'].append({'step': step, **cached['metadata']})
            atomic_json(args.report, meta)
            for fold in [0, 1]:
                key = f'step{step}_fold{fold}'
                previous = store.load('routers', key)
                if previous is None:
                    train_cases = [c for c in cases if c['index'] % 2 != fold]
                    validation = [c for c in cases if c['index'] % 2 == fold]
                    if (len(train_cases) != 12 or len(validation) != 12
                            or sum(c['index'] < 12 for c in train_cases) != 6):
                        raise RuntimeError('Balanced fixed12/12 source folds required')
                    train_rows = np.concatenate([np.arange(c['index'] * 256, (c['index'] + 1) * 256) for c in train_cases])
                    val_rows = np.concatenate([np.arange(c['index'] * 256, (c['index'] + 1) * 256) for c in validation])
                    seed = 20261021 + 100 * step + fold
                    row = {'step': step, 'block': 20, 'fold': fold, 'keep_groups': keep, 'seed': seed,
                           'train_inputs': [c['index'] for c in train_cases],
                           'validation_inputs': [c['index'] for c in validation]}
                    arrays = {}
                    try:
                        banks = fit_banks(profiles[train_rows], keep, seed)
                    except DegenerateBanks as error:
                        row.update(status='REJECTED_DEGENERATE_BANKS_WITHOUT_RESEEDING', error=str(error), gate_pass=False)
                    else:
                        train_x = torch.cat([c['V1'] for c in train_cases])
                        labels = torch.tensor(bank_energy(profiles[train_rows], banks['masks']).argmax(1), device='cuda', dtype=torch.long)
                        model, mean, scale, optimization = fit_router(train_x, labels, seed)
                        with torch.no_grad():
                            parameters = folded_parameters(model, mean, scale)
                            predictions = np.concatenate([native_predict_rows(c, parameters) for c in validation])
                            val_x = torch.cat([c['V1'] for c in validation]).float()
                            fp32_predictions = model((val_x - mean) / scale).argmax(1).cpu().numpy()
                        row.update(status='COMPLETED_FIXED_ROUTER_PROXY_FOLD', **describe_proxy(profiles[val_rows], banks, predictions),
                                   bank_training_counts=banks['counts'].tolist(), optimization=optimization,
                                   folded_native_parameter_sha256={k: tensor_sha(v) for k, v in parameters.items()},
                                   FP32_router_oracle_label_accuracy=float((fp32_predictions == bank_energy(profiles[val_rows], banks['masks']).argmax(1)).mean()),
                                   native_vs_FP32_route_disagreement=float((predictions != fp32_predictions).mean()),
                                   original_fullM_native_inference=True)
                        arrays = {'bank_masks': banks['masks'], 'static_mask': banks['static'], 'centers': banks['centers'],
                                  'training_mean': mean.cpu().numpy(), 'training_scale': scale.cpu().numpy(),
                                  **{'native_' + k: bf16_bits(v) for k, v in parameters.items()},
                                  **{'shadow_' + k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}}
                        del model, parameters, train_x, val_x
                    store.save('routers', key, arrays, row)
                    previous = store.load('routers', key)
                meta['CV'].append(previous['metadata'])
                atomic_json(args.report, meta)
                print('RB1 CV ' + json.dumps(previous['metadata']), flush=True)
            del cases
        passed = len(meta['CV']) == 4 and all(v['gate_pass'] for v in meta['CV'])
        meta.update(status='COMPLETED_RB1_PROXY_GATE_PASSED' if passed else 'COMPLETED_RB1_PROXY_GATE_REJECTED',
                    proxy_gate_pass=passed, elapsed_seconds=time.perf_counter() - started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error))
        raise
    finally:
        atomic_json(args.report, meta)
        atomic_json(args.out / 'run_meta.json', meta)


if __name__ == '__main__':
    main()
