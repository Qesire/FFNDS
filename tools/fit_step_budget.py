"""Extend the frozen C24 centered/RR recipe to one UB1 step, with a108-group control."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ar2_checkpoint import CheckpointStore, atomic_json, digest
from ar2_core import top_groups
from centered_signal_fit import fit_signal_moments
from evaluate_residual_rr1 import open_verified
from fit_cstatic_centered import load_training_reports
from reduced_rank_residual import fit_rrr, native_parameters, predict_y
from run_ar2_restart import ROOT, tensor_sha


def fit_position(records, w, info, cmean, step, block, keep):
    groups = np.sort(top_groups(cmean, keep)); channels = torch.tensor(
        (groups[:, None] * 64 + np.arange(64)).reshape(-1), device='cuda')
    wf = w.float(); packed, packed_f = w.index_select(1, channels).contiguous(), wf.index_select(1, channels).contiguous()
    entries, bias, gate = [], None, None
    ss, sy = 0., 0.; sum_s = torch.zeros(5120, device='cuda', dtype=torch.float64); sum_y = torch.zeros_like(sum_s)
    for index in range(24):
        scope = next(s for s in records[index]['scopes'] if s['step'] == step and s['block'] == block)
        if scope['rows'] != 256 or digest(scope['path']) != scope['sha256']:
            raise RuntimeError('Frozen calibration operands changed')
        with safe_open(scope['path'], framework='pt', device='cpu') as handle:
            x, a, y, b, g = [handle.get_tensor(k).cuda() for k in ['V1', 'V2', 'Y', 'b2', 'gate']]
        b, g = b.reshape(-1), g.reshape(1, 5120)
        if (tensor_sha(b) != info['original_b2_sha256'] or tensor_sha(g) != info['gate_sha256']
                or x.shape != (256, 5120) or a.shape != (256, 13824) or y.shape != (256, 5120)):
            raise RuntimeError('Original native bias/gate/geometry differs')
        bias, gate = b, g; selected = a.index_select(1, channels)
        dense = F.linear(a.float(), wf, None); signal = F.linear(selected.float(), packed_f, None)
        sum_s += signal.double().sum(0); sum_y += dense.double().sum(0)
        sg, yg = (signal * gate.float()).double(), (dense * gate.float()).double()
        ss += float(sg.square().sum()); sy += float((sg * yg).sum())
        entries.append({'x': x, 'a': selected, 'y': y})
    fit = fit_signal_moments(6144, sum_s.cpu().numpy(), sum_y.cpu().numpy(), ss, sy, gate.float().cpu().numpy())
    if fit['centered_alpha'] is None: raise RuntimeError('Degenerate centered calibration, no fallback')
    alpha = float(np.float32(fit['centered_alpha']))
    correction = fit['mean_dense'] - alpha * fit['mean_signal']
    b2 = (bias.double() + torch.as_tensor(correction, device='cuda')).bfloat16()
    xs, ys, bases = [], [], []
    for entry in entries:
        base = F.linear((entry['a'].float() * alpha).bfloat16(), packed, b2)
        xs.append(entry['x']); ys.append(entry['y']); bases.append(base)
    x, y, base = [torch.cat(v) for v in [xs, ys, bases]]
    label = (y.float() - base.float()) * gate.float()
    model = fit_rrr(x, label, 1., 256); parameters = native_parameters(model, gate, 256)
    target = (y * gate).double(); candidate = (base + predict_y(x, parameters)) * gate
    data = {'groups': groups, 'b2': b2.float().cpu().numpy(),
        **{'rr_' + k: v.detach().cpu().float().numpy() for k, v in parameters.items()}}
    metadata = {'step': step, 'block': block, 'groups': keep, 'hidden_width': keep * 64,
        'alpha': alpha, 'alpha_fit_FP64': fit['centered_alpha'], 'b2_sha256': tensor_sha(b2),
        'RR_parameter_sha256': {k: tensor_sha(v) for k, v in parameters.items()},
        'gate_sha256': info['gate_sha256'], 'original_b2_sha256': info['original_b2_sha256'],
        'original_W2_sha256': info['weight_sha256'], 'rank': 256, 'lambda': 1.,
        'base_native_training_SSE': float(((base * gate).double() - target).square().sum()),
        'corrected_native_training_SSE': float((candidate.double() - target).square().sum())}
    return data, metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--step', type=int, choices=[0, 3], required=True)
    ap.add_argument('--collections', type=Path, nargs=2, required=True)
    ap.add_argument('--parent-policy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args(); torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    if (not args.out.resolve().is_relative_to(Path('/var/tmp/ffnds-xiang/huggingface/ffnds-derived'))
            or shutil.disk_usage('/var/tmp').free < 82 * 1024**3):
        raise RuntimeError('HF coefficient fit must preserve80GiB node reserve plus2GiB allowance')
    args.out.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    records, bindings, capture = load_training_reports(args.collections, expected_inputs=24, expected_rows=256)
    policy = json.loads(args.parent_policy.read_text()); expert = 'high_noise_model' if args.step == 0 else 'low_noise_model'
    b = policy['fit_reports'][expert]
    if digest(b['path']) != b['sha256']: raise RuntimeError('Frozen C24 base report changed')
    parent, parent_report, parent_id = open_verified(Path(b['path']), 'COMPLETED_CALIBRATION_PARAMETER_FIT')
    if parent_id['training_collections'] != bindings or parent.fingerprint != b['fingerprint']:
        raise RuntimeError('Parent and budget refit calibration identity differs')
    rr_binding = policy['residual_fit']
    if digest(rr_binding['path']) != rr_binding['sha256']: raise RuntimeError('C24 residual report changed')
    rr, rr_report, rr_id = open_verified(Path(rr_binding['path']), 'COMPLETED_RR1_FULL160_CALIBRATION_FIT')
    cache_binding = policy['cache_manifest']
    if digest(cache_binding['path']) != cache_binding['sha256'] or capture['assets']['cache_manifest_sha256'] != cache_binding['sha256']:
        raise RuntimeError('Original quantized cache changed')
    cache = json.loads(Path(cache_binding['path']).read_text()); protocol = ROOT / 'config/step_budget_ub1.json'
    counts = json.loads(protocol.read_text())['groups_by_step']; keep = counts[args.step]
    store = CheckpointStore(args.out, {'protocol_sha256': digest(protocol), 'step': args.step, 'keep_groups': keep,
        'training_collections': bindings, 'parent_policy_sha256': digest(args.parent_policy),
        'cache_manifest_sha256': cache_binding['sha256'], 'torch': torch.__version__, 'TF32': False,
        'sources': {str(p): digest(p) for p in [Path(__file__), ROOT / 'tools/centered_signal_fit.py',
            ROOT / 'tools/reduced_rank_residual.py', ROOT / 'tools/ar2_core.py']}})
    meta = {'status': 'RUNNING', 'fingerprint': store.fingerprint, 'checkpoint_root': str(args.out.resolve()),
        'step': args.step, 'groups': keep, 'control': None, 'positions': [], 'quality_or_speed_acceptance': False}
    atomic_json(args.report, meta); started = time.perf_counter()
    try:
        # Reproduce the exact old recipe before changing the selected count.
        control_name = f'step{args.step}_block20'; original = parent.load('positions', control_name)
        shard = next(s for s in cache['shards'] if s['expert'] == expert and s['block'] == 20)
        if digest(shard['path']) != shard['sha256']: raise RuntimeError('Effective control shard changed')
        payload = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)
        w = payload[expert]['blocks.20.ffn.2']['weight_qdq'].cuda()
        arrays, control = fit_position(records, w, original['metadata'], original['arrays']['C_mean'], args.step, 20, 108)
        prior = original['metadata']['policies']['C']['modes']['centered_gain_bias']; old_rr = rr.load('positions', control_name)
        if (control['alpha'] != prior['alpha'] or control['b2_sha256'] != prior['b2_sha256']
                or control['RR_parameter_sha256'] != old_rr['metadata']['parameter_sha256']
                or not np.array_equal(arrays['groups'], original['arrays']['C_groups'])):
            atomic_json(args.out / 'CONTROL_MISMATCH.json', {'refit': control, 'old_base': prior, 'old_RR': old_rr['metadata']})
            raise RuntimeError('108-group refit did not reproduce frozen C24 coefficient bits')
        meta['control'] = {'status': 'PASS_OLD_C24_ALPHA_BIAS_RR_PARAMETER_BITS', **control}; atomic_json(args.report, meta)
        del arrays, w, payload
        for block in range(40):
            name = f'step{args.step}_block{block:02d}'; previous = store.load('positions', name)
            if previous is None:
                base = parent.load('positions', name); shard = next(s for s in cache['shards'] if s['expert'] == expert and s['block'] == block)
                if digest(shard['path']) != shard['sha256']: raise RuntimeError('Effective weight shard changed')
                payload = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)
                w = payload[expert][f'blocks.{block}.ffn.2']['weight_qdq'].cuda()
                if tensor_sha(w) != base['metadata']['weight_sha256']: raise RuntimeError('Original effective W2 differs')
                arrays, info = fit_position(records, w, base['metadata'], base['arrays']['C_mean'], args.step, block, keep)
                store.save('positions', name, arrays, info); previous = store.load('positions', name)
                del w, payload, arrays
            meta['positions'].append(previous['metadata']); atomic_json(args.report, meta)
            print(f'UB1 fitted {name}, groups{keep}', flush=True)
        if len(meta['positions']) != 40: raise RuntimeError('All40 changed-step fits required')
        meta.update(status='COMPLETED_UB1_STEP40_FIT', elapsed_seconds=time.perf_counter() - started)
    except BaseException as error:
        meta.update(status='FAILED_RESUMABLE', error=str(error)); raise
    finally:
        atomic_json(args.report, meta); atomic_json(args.out / 'run_meta.json', meta)


if __name__ == '__main__': main()
