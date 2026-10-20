"""Conditional LA1 pilot: native HiF4 QAT, fixed rank32/8epochs, four development folds."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors import safe_open

from anchored_reconstruction import bf16_bits
from ar2_checkpoint import CheckpointStore, atomic_json, digest
from evaluate_residual_rr1 import open_verified
from folded_adapter import FoldedAdapters, packed_forward, replay_main_rows
from onpolicy_residual import check_old_replay, load_collections, native_sse, position_cases, replay_rows, same_bits
from run_ar2_restart import ROOT, tensor_sha


def rejected_op1(r):
    cv = r / 'onpolicy_residual_cv/RUN.json'
    if cv.exists() and json.loads(cv.read_text())['status'] == 'COMPLETED_OP1_CV_GATE_REJECTED':
        return {'path': str(cv), 'sha256': digest(cv), 'reason': 'CV rejected'}
    quality = r / 'formal20_OP1_comparison/COMPARISON.json'
    if quality.exists():
        d = json.loads(quality.read_text())
        if d['status'] == 'COMPLETED_PAIRED_FORMAL20_THREE_METRIC_COMPARISON' and not d['formal20_quality_pass']:
            return {'path': str(quality), 'sha256': digest(quality), 'reason': 'formal quality rejected'}
    raise RuntimeError('LA1 is conditional on OP1 CV or formal quality rejection')


def prepare(records, cache, policy, old, step, activation_quantizer, *, block=20):
    expert = 'high_noise_model' if step < 2 else 'low_noise_model'; name = f'step{step}_block{block:02d}'
    binding = policy['fit_reports'][expert]; path = Path(binding['path'])
    if digest(path) != binding['sha256']: raise RuntimeError('C24 base fit report changed')
    base, base_report, base_id = open_verified(path, 'COMPLETED_CALIBRATION_PARAMETER_FIT')
    if base.fingerprint != binding['fingerprint']: raise RuntimeError('C24 base fingerprint differs')
    part = base.load('positions', name); a, info = part['arrays'], part['metadata']
    ids = torch.tensor((a['C_groups'][:, None] * 64 + torch.arange(64).numpy()).reshape(-1), device='cuda')
    shard = next(s for s in cache['shards'] if s['expert'] == expert and s['block'] == block)
    if digest(shard['path']) != shard['sha256']: raise RuntimeError('Original effective cache shard changed')
    payload = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)[expert]
    w2_full = payload[f'blocks.{block}.ffn.2']['weight_qdq']
    if tensor_sha(w2_full) != info['weight_sha256']: raise RuntimeError('Frozen effective W2 differs')
    w0 = payload[f'blocks.{block}.ffn.0']['weight_qdq'].cuda().index_select(0, ids).contiguous()
    w2 = w2_full.cuda().index_select(1, ids).contiguous()
    smooth2 = payload[f'blocks.{block}.ffn.2']['smooth_scale'].cuda().index_select(0, ids).contiguous()
    view = json.loads((ROOT / '.runtime/node_local_model_view.json').read_text())
    source = next(f for f in view['files'] if f'/{expert}/' in f['view_path'] and f['filename'].endswith('.safetensors'))
    with safe_open(source['view_path'], framework='pt', device='cpu') as handle:
        b0_full = handle.get_tensor(f'blocks.{block}.ffn.0.bias').to(device='cuda', dtype=torch.bfloat16)
    cases = position_cases(records, step, block, info)
    if any(c['scope']['original_b0_sha256'] != tensor_sha(b0_full) for c in cases):
        raise RuntimeError('Original raw W0 bias differs from native observations')
    b0 = b0_full.index_select(0, ids).contiguous()
    b2 = torch.from_numpy(a['C_centered_gain_bias_b2']).to(device='cuda', dtype=torch.bfloat16)
    parameters = info['policies']['C']['modes']['centered_gain_bias']; alpha = parameters['alpha']
    if tensor_sha(b2) != parameters['b2_sha256']: raise RuntimeError('Frozen folded bias changed')
    old_part = old.load('positions', name)
    rr = {k: torch.from_numpy(v).to(device='cuda', dtype=torch.bfloat16) for k, v in old_part['arrays'].items()}
    if {k: tensor_sha(v) for k, v in rr.items()} != old_part['metadata']['parameter_sha256']:
        raise RuntimeError('Frozen residual parameter encoding differs')
    check_old_replay(cases, rr)
    with torch.no_grad():
        for case in cases:
            main = replay_main_rows(case['V1'], case['scope'], w0, b0, w2, b2, smooth2, alpha, activation_quantizer)
            if not same_bits(main, case['BaseY']):
                raise RuntimeError(f"LA1 original packed main did not replay bitwise: step{step} input{case['index']}")
            case['R'] = replay_rows(case['V1'], case['scope'], rr)
            if not same_bits(main + case['R'], case['StudentY']):
                raise RuntimeError('LA1 original complete student did not replay bitwise')
    return {'cases': cases, 'w0': w0, 'w2': w2, 'b0': b0, 'b2': b2, 'smooth2': smooth2,
        'alpha': alpha, 'base_info': info, 'gate': cases[0]['gate'],
        'control': {'step': step, 'block': block, 'expert': expert, 'inputs': 24, 'original_BaseY_bitwise': True,
            'original_StudentY_bitwise': True, 'full_native_geometry': True,
            'W0_sha256': tensor_sha(w0), 'W2_sha256': tensor_sha(w2),
            'b0_sha256': tensor_sha(b0), 'b2_sha256': tensor_sha(b2), 'smooth2_sha256': tensor_sha(smooth2)}}


@torch.no_grad()
def score(data, cases, weights, activation_quantizer):
    total = 0.
    for c in cases:
        main = replay_main_rows(c['V1'], c['scope'], weights[0], data['b0'], weights[1], data['b2'],
            data['smooth2'], data['alpha'], activation_quantizer)
        total += native_sse(main + c['R'], c['TeacherY'], c['gate'])
    return total


def fit(data, train, weight_quantizer, activation_quantizer):
    model = FoldedAdapters(data['w0'], data['w2'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    x = torch.cat([c['V1'] for c in train]); residual = torch.cat([c['R'] for c in train])
    target = torch.cat([(c['TeacherY'] * c['gate']).float() for c in train])
    denominator = target.square().mean()
    if not bool(torch.isfinite(denominator)) or denominator <= 0: raise RuntimeError('Degenerate LA1 target scale')
    generator = torch.Generator(device='cuda').manual_seed(20261021); losses = []; updates = 0
    for epoch in range(8):
        permutation = torch.randperm(len(x), generator=generator, device='cuda'); loss_sum = 0.
        for ids in permutation.split(128):
            optimizer.zero_grad(set_to_none=True)
            w0, w2 = model.weights(weight_quantizer)
            main = packed_forward(x[ids], w0, data['b0'], w2, data['b2'], data['smooth2'], data['alpha'],
                activation_quantizer, training=True)
            prediction = ((main + residual[ids]) * data['gate']).float()
            loss = (prediction - target[ids]).square().mean() / denominator
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite LA1 training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); updates += 1; loss_sum += float(loss.detach()) * len(ids)
        losses.append(loss_sum / len(x))
        print(f'LA1 epoch{epoch + 1}: normalized native loss={losses[-1]:.8g}', flush=True)
    if len(train) not in [12, 24] or updates != 8 * len(train) * 256 // 128:
        raise RuntimeError('LA1 fixed optimization budget differs')
    with torch.no_grad(): weights = tuple(w.detach().contiguous() for w in model.weights(weight_quantizer))
    return model, weights, {'epoch_losses': losses, 'updates': updates, 'epochs': 8, 'batch': 128,
        'training_target_mean_square': float(denominator)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections', type=Path, nargs=2, required=True)
    ap.add_argument('--student-policy', type=Path, required=True)
    ap.add_argument('--run-root', type=Path, default=ROOT / 'runs/worker_restart_20261020')
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args(); condition = rejected_op1(args.run_root)
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq as aq, native_hif4_weight_qdq as wq
    torch.set_num_threads(8); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    if (not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 83 * 1024**3):
        raise RuntimeError('Node HF pilot requires80GiB reserve plus3GiB allowance')
    args.out.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    records, bindings, capture = load_collections(args.collections, ROOT)
    if digest(args.student_policy) != capture['student_policy_sha256']: raise RuntimeError('Frozen C24 student differs')
    policy = json.loads(args.student_policy.read_text()); cache_binding = policy['cache_manifest']
    if digest(cache_binding['path']) != cache_binding['sha256']: raise RuntimeError('Cache manifest changed')
    cache = json.loads(Path(cache_binding['path']).read_text()); old_binding = policy['residual_fit']
    if digest(old_binding['path']) != old_binding['sha256']: raise RuntimeError('Frozen C24 residual report changed')
    old, old_report, old_id = open_verified(Path(old_binding['path']), 'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    if old.fingerprint != old_binding['fingerprint']: raise RuntimeError('Frozen C24 residual differs')
    protocol = ROOT / 'config/folded_adapter_la1.json'
    sources = [Path(__file__), ROOT / 'tools/folded_adapter.py', ROOT / 'tools/onpolicy_residual.py',
        ROOT / 'tools/anchored_reconstruction.py',
        ROOT / 'tools/reduced_rank_residual.py', ROOT / '.runtime/code/HiF4_Sp/HiFloat4/ptq/adapters/wan_i2v.py',
        ROOT / '.runtime/code/Wan2.2-I2V-A14B-W4A4/quant/hif4_backend.py',
        ROOT / '.runtime/code/HiF4_Sp/HiFloat4/main.py']
    sources += sorted(p for p in (ROOT / '.runtime/code/HiF4_Sp/HiFloat4/hif4_gpu/quant_cy').rglob('*')
        if p.suffix in ['.py', '.cu', '.cpp', '.h', '.hpp'])
    store = CheckpointStore(args.out, {'protocol_sha256': digest(protocol), 'protocol': json.loads(protocol.read_text()),
        'collection_bindings': bindings, 'capture_identity': capture, 'student_policy_sha256': digest(args.student_policy),
        'condition': condition, 'torch': torch.__version__, 'TF32': False, 'sources': {str(p): digest(p) for p in sources}})
    meta = {'status': 'RUNNING', 'checkpoint_root': str(args.out.resolve()), 'fingerprint': store.fingerprint,
        'native_controls': [], 'CV': [], 'quality_or_speed_acceptance': False, 'development_calibration_reused': True}
    atomic_json(args.report, meta); started = time.perf_counter()
    try:
        # Both scopes must pass the original-parent controls before any training.
        data = {step: prepare(records, cache, policy, old, step, aq) for step in [2, 3]}
        meta['native_controls'] = [d['control'] for d in data.values()]; atomic_json(args.report, meta)
        for step, position in data.items():
            for fold in [0, 1]:
                key = f'step{step}_fold{fold}'; part = store.load('CV_models', key)
                if part is None:
                    train = [c for c in position['cases'] if c['index'] % 2 != fold]
                    validation = [c for c in position['cases'] if c['index'] % 2 == fold]
                    original = sum(native_sse(c['StudentY'], c['TeacherY'], c['gate']) for c in validation)
                    initial_model = FoldedAdapters(position['w0'], position['w2'])
                    with torch.no_grad():
                        initial = initial_model.weights(wq)
                        for slot, projected, prior in zip(['0', '2'], initial, [position['w0'], position['w2']]):
                            # IEEE prior+zero can change a negative-zero sign.
                            # The STE control must give both functions precisely
                            # the same effective input, including its zero bits.
                            effective = prior.float() + getattr(initial_model, 'b' + slot) @ getattr(initial_model, 'a' + slot)
                            if not same_bits(projected, wq(effective, force_fp32=True).bfloat16()):
                                raise RuntimeError('Straight-through native weight forward differs')
                        requantized = score(position, validation, initial, aq)
                    del initial_model, initial
                    model, weights, optimization = fit(position, train, wq, aq)
                    corrected = score(position, validation, weights, aq)
                    row = {'step': step, 'block': 20, 'fold': fold, 'adapter_rank': 32,
                        'train_inputs': [c['index'] for c in train], 'validation_inputs': [c['index'] for c in validation],
                        'original_captured_SSE': original, 'zero_update_requantized_SSE': requantized,
                        'requantized_over_original': requantized / original, 'LA1_native_SSE': corrected,
                        'LA1_over_original': corrected / original, 'gate_pass': corrected <= .8 * original,
                        'original_main_and_residual_replay_bitwise': True, 'STE_forward_native_bitwise': True,
                        'folded_weight_sha256': [tensor_sha(w) for w in weights], 'optimization': optimization}
                    arrays = {'W0': bf16_bits(weights[0]), 'W2': bf16_bits(weights[1]),
                        **{'adapter_' + k: v.detach().cpu().numpy() for k, v in model.named_parameters()}}
                    store.save('CV_models', key, arrays, row); part = store.load('CV_models', key)
                    del model, weights, arrays
                meta['CV'].append(part['metadata']); atomic_json(args.report, meta)
                print('LA1 CV ' + json.dumps(part['metadata']), flush=True)
        meta['CV_gate_pass'] = len(meta['CV']) == 4 and all(r['gate_pass'] for r in meta['CV'])
        meta.update(status='COMPLETED_LA1_CV_GATE_PASSED' if meta['CV_gate_pass'] else 'COMPLETED_LA1_CV_GATE_REJECTED',
            elapsed_seconds=time.perf_counter() - started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error)); raise
    finally:
        atomic_json(args.report, meta); atomic_json(args.out / 'run_meta.json', meta)


if __name__ == '__main__': main()
