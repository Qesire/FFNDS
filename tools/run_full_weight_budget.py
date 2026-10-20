"""FA2 fixed64epoch QAT CV with an exact native epoch8 FA1 replay control."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch

from anchored_reconstruction import bf16_bits
from ar2_checkpoint import CheckpointStore, atomic_json, digest
from evaluate_residual_rr1 import open_verified
from folded_adapter import packed_forward
from full_weight_adapter import FullMainWeights
from onpolicy_residual import load_collections, native_sse, same_bits
from run_ar2_restart import ROOT, tensor_sha
from run_folded_adapter_cv import prepare, rejected_op1, score


def fit(data, train, weight_quantizer, activation_quantizer, *, warm_expected=None):
    model = FullMainWeights(data['w0'], data['w2'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    x = torch.cat([c['V1'] for c in train]); residual = torch.cat([c['R'] for c in train])
    target = torch.cat([(c['TeacherY'] * c['gate']).float() for c in train]); denominator = target.square().mean()
    if not bool(torch.isfinite(denominator)) or denominator <= 0: raise RuntimeError('Degenerate FA2 target scale')
    generator = torch.Generator(device='cuda').manual_seed(20261021); history = []; updates = 0; warm_control = None
    for epoch in range(64):
        permutation = torch.randperm(len(x), generator=generator, device='cuda'); fit_sum, penalty_sum = 0., 0.
        for ids in permutation.split(128):
            optimizer.zero_grad(set_to_none=True); w0, w2 = model.weights(weight_quantizer)
            main = packed_forward(x[ids], w0, data['b0'], w2, data['b2'], data['smooth2'], data['alpha'],
                activation_quantizer, training=True)
            prediction = ((main + residual[ids]) * data['gate']).float()
            reconstruction = (prediction - target[ids]).square().mean() / denominator
            penalty = model.relative_deviation(); loss = reconstruction + .001 * penalty
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite FA2 training objective')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); updates += 1
            fit_sum += float(reconstruction.detach()) * len(ids); penalty_sum += float(penalty.detach()) * len(ids)
        history.append({'epoch': epoch + 1, 'normalized_reconstruction': fit_sum / len(x),
            'relative_deviation': penalty_sum / len(x)})
        if epoch == 7 and warm_expected is not None:
            with torch.no_grad():
                native8 = model.weights(weight_quantizer)
                native_hashes = [tensor_sha(w) for w in native8]
                shadow_hashes = {k: tensor_sha(v) for k, v in model.named_parameters()}
            warm_control = {'native_epoch8_sha256': native_hashes,
                'native_matches_FA1': native_hashes == warm_expected['folded_weight_sha256'],
                'shadow_matches_FA1': shadow_hashes == warm_expected['shadow_parameter_sha256']}
            if not warm_control['native_matches_FA1']:
                raise RuntimeError('FA2 epoch8 native weights differ from the original frozen FA1 fold')
            print('FA2 EPOCH8 CONTROL ' + json.dumps(warm_control), flush=True)
        print('FA2 EPOCH ' + json.dumps(history[-1]), flush=True)
    if len(train) not in [12, 24] or updates != 64 * len(train) * 256 // 128:
        raise RuntimeError('FA2 fixed optimization budget differs')
    with torch.no_grad():
        weights = tuple(w.detach().contiguous() for w in model.weights(weight_quantizer))
        deviation = float(model.relative_deviation())
    return model, weights, {'epochs': history, 'updates': updates, 'batch': 128,
        'training_target_mean_square': float(denominator), 'final_relative_deviation': deviation, 'epoch8_FA1_control': warm_control}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections', type=Path, nargs=2, required=True)
    ap.add_argument('--student-policy', type=Path, required=True)
    ap.add_argument('--la1-report', type=Path, required=True)
    ap.add_argument('--fa1-report', type=Path, required=True)
    ap.add_argument('--run-root', type=Path, default=ROOT / 'runs/worker_restart_20261020')
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args(); condition = rejected_op1(args.run_root)
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as aq, native_hif4_weight_qdq as wq
    torch.set_num_threads(8); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    if (not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 84 * 1024**3):
        raise RuntimeError('Node HF pilot requires80GiB reserve plus4GiB allowance')
    args.out.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    records, bindings, capture = load_collections(args.collections, ROOT)
    predecessor, la1, la1_id = open_verified(args.la1_report, 'COMPLETED_LA1_CV_GATE_REJECTED')
    if len(la1['CV']) != 4 or la1_id['collection_bindings'] != bindings:
        raise RuntimeError('Same-data completed LA1 rejection required')
    for name, sha in la1_id['sources'].items():
        if digest(name) != sha: raise RuntimeError('Frozen predecessor/control numerical source changed')
    short, short_report, short_id = open_verified(args.fa1_report, 'COMPLETED_FA1_CV_GATE_REJECTED')
    if len(short_report['CV']) != 4 or short_id['collection_bindings'] != bindings:
        raise RuntimeError('FA2 requires same-data frozen FA1 four-fold rejection')
    for name, sha in short_id['sources'].items():
        if digest(name) != sha: raise RuntimeError('Frozen FA1 numerical source changed')
    if digest(args.student_policy) != capture['student_policy_sha256']: raise RuntimeError('Frozen C24 student differs')
    policy = json.loads(args.student_policy.read_text()); cache_binding = policy['cache_manifest']
    if digest(cache_binding['path']) != cache_binding['sha256']: raise RuntimeError('Cache manifest changed')
    cache = json.loads(Path(cache_binding['path']).read_text()); old_binding = policy['residual_fit']
    if digest(old_binding['path']) != old_binding['sha256']: raise RuntimeError('Frozen C24 residual report changed')
    old, old_report, old_id = open_verified(Path(old_binding['path']), 'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    if old.fingerprint != old_binding['fingerprint']: raise RuntimeError('Frozen C24 residual differs')
    protocol = ROOT / 'config/full_weight_fa2.json'
    sources = [Path(__file__), ROOT / 'tools/full_weight_adapter.py', ROOT / 'tools/folded_adapter.py',
        ROOT / 'tools/run_folded_adapter_cv.py', ROOT / 'tools/onpolicy_residual.py', ROOT / 'tools/anchored_reconstruction.py']
    store = CheckpointStore(args.out, {'protocol_sha256': digest(protocol), 'protocol': json.loads(protocol.read_text()),
        'collection_bindings': bindings, 'capture_identity': capture, 'student_policy_sha256': digest(args.student_policy),
        'condition': condition, 'LA1_report_sha256': digest(args.la1_report), 'LA1_fingerprint': predecessor.fingerprint,
        'torch': torch.__version__, 'TF32': False, 'sources': {str(p): digest(p) for p in sources},
        'parent_numerical_sources': la1_id['sources'], 'FA1_report_sha256': digest(args.fa1_report),
        'FA1_fingerprint': short.fingerprint, 'FA1_numerical_sources': short_id['sources']})
    meta = {'status': 'RUNNING', 'checkpoint_root': str(args.out.resolve()), 'fingerprint': store.fingerprint,
        'native_controls': [], 'CV': [], 'quality_or_speed_acceptance': False, 'development_calibration_reused': True}
    atomic_json(args.report, meta); started = time.perf_counter()
    try:
        data = {step: prepare(records, cache, policy, old, step, aq) for step in [2, 3]}
        meta['native_controls'] = [d['control'] for d in data.values()]; atomic_json(args.report, meta)
        for step, position in data.items():
            for fold in [0, 1]:
                key = f'step{step}_fold{fold}'; part = store.load('CV_models', key)
                if part is None:
                    train = [c for c in position['cases'] if c['index'] % 2 != fold]
                    validation = [c for c in position['cases'] if c['index'] % 2 == fold]
                    original = sum(native_sse(c['StudentY'], c['TeacherY'], c['gate']) for c in validation)
                    initial_model = FullMainWeights(position['w0'], position['w2'])
                    with torch.no_grad():
                        initial = initial_model.weights(wq)
                        for slot, projected in zip(['0', '2'], initial):
                            effective = getattr(initial_model, 'weight' + slot)
                            if not same_bits(projected, wq(effective, force_fp32=True).bfloat16()):
                                raise RuntimeError('FA2 STE does not match native Q on identical input')
                        requantized = score(position, validation, initial, aq)
                    del initial_model, initial
                    short_part = short.load('CV_models', key)
                    if short_part is None: raise RuntimeError('Frozen FA1 fold checkpoint missing')
                    model, weights, optimization = fit(position, train, wq, aq, warm_expected=short_part['metadata'])
                    corrected = score(position, validation, weights, aq)
                    row = {'step': step, 'block': 20, 'fold': fold,
                        'train_inputs': [c['index'] for c in train], 'validation_inputs': [c['index'] for c in validation],
                        'original_captured_SSE': original, 'zero_update_requantized_SSE': requantized,
                        'requantized_over_original': requantized / original, 'FA2_native_SSE': corrected,
                        'FA2_over_original': corrected / original, 'gate_pass': corrected <= .8 * original,
                        'original_main_and_residual_replay_bitwise': True, 'STE_forward_native_bitwise': True,
                        'folded_weight_sha256': [tensor_sha(w) for w in weights], 'optimization': optimization,
                        'shadow_parameter_sha256': {k: tensor_sha(v) for k, v in model.named_parameters()}}
                    arrays = {'W0': bf16_bits(weights[0]), 'W2': bf16_bits(weights[1]),
                        **{'shadow_' + k: v.detach().cpu().numpy() for k, v in model.named_parameters()}}
                    store.save('CV_models', key, arrays, row); part = store.load('CV_models', key)
                    del model, weights, arrays
                meta['CV'].append(part['metadata']); atomic_json(args.report, meta)
                print('FA2 CV ' + json.dumps(part['metadata']), flush=True)
        meta['CV_gate_pass'] = len(meta['CV']) == 4 and all(r['gate_pass'] for r in meta['CV'])
        meta.update(status='COMPLETED_FA2_CV_GATE_PASSED' if meta['CV_gate_pass'] else 'COMPLETED_FA2_CV_GATE_REJECTED',
            elapsed_seconds=time.perf_counter() - started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error)); raise
    finally:
        atomic_json(args.report, meta); atomic_json(args.out / 'run_meta.json', meta)


if __name__ == '__main__': main()
