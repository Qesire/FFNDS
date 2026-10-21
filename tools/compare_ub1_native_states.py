"""OP2D: fixed-prefix identity and descriptive C24-to-UB1 state differences."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from ar2_checkpoint import atomic_json, digest
from historical_source_snapshot import HistoricalSources
from onpolicy_residual import native_sse, same_bits
from original_onpolicy_data import load_original_collections
from run_ar2_restart import ROOT
from ub1_native_data import load_ub1_collections


def scope(case, step, block):
    return next(s for s in case['scopes'] if s['step'] == step and s['block'] == block)


def read_pair(old, new):
    for field in ['grid', 'full_rows', 'selected_rows', 'sampling_seed', 'inclusion_probability',
                  'original_b2_sha256', 'gate_sha256', 'native_V1_shape', 'native_V1_stride']:
        if old[field] != new[field]:
            raise RuntimeError('Paired native source coordinates, gate or geometry differ')
    payloads = []
    for item in [old, new]:
        if digest(item['path']) != item['sha256']:
            raise RuntimeError('Original native state payload changed')
        with safe_open(item['path'], framework='pt', device='cpu') as handle:
            if set(handle.keys()) != {'V1', 'BaseY', 'TeacherY', 'StudentY', 'b2', 'gate'}:
                raise RuntimeError('Unexpected native payload fields')
            values = {k: handle.get_tensor(k) for k in handle.keys()}
        if any(v.dtype != torch.bfloat16 or not bool(torch.isfinite(v).all()) for v in values.values()):
            raise RuntimeError('Native payload is not finite BF16')
        if any(values[k].shape != (256, 5120) for k in ['V1', 'BaseY', 'TeacherY', 'StudentY']):
            raise RuntimeError('Native sampled row dimensions differ')
        payloads.append(values)
    for key in ['gate', 'b2']:
        if not same_bits(payloads[0][key], payloads[1][key]):
            raise RuntimeError('Paired original gate or raw bias changed')
    return payloads


def difference(current, reference):
    a, b = current.double(), reference.double()
    aa, bb = float(a.square().sum()), float(b.square().sum())
    sse = float((a - b).square().sum())
    if aa <= 0 or bb <= 0:
        raise RuntimeError('Degenerate native state norm')
    return {'SSE': sse, 'reference_energy': bb, 'relative_RMSE': math.sqrt(sse / bb),
            'cosine': float((a * b).sum()) / math.sqrt(aa * bb)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise RuntimeError('OP2D is a CPU-only description')
    torch.set_num_threads(8)
    runs = ROOT / 'runs/worker_restart_20261020'
    history = HistoricalSources()
    old, old_bindings, old_identity = load_original_collections(
        [runs / f'onpolicy_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT, history)
    new, new_bindings, new_identity = load_ub1_collections(
        [runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in [0, 1]], ROOT)
    if old_identity['assets'] != new_identity['assets']:
        raise RuntimeError('Paired state collections use different model or quantized assets')
    for index in range(24):
        if (old[index]['input']['input_provenance'] != new[index]['input']['input_provenance']
                or old[index]['input']['calibration_seed'] != new[index]['input']['seed']):
            raise RuntimeError('Paired original input identity differs')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'JOB.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = args.out / 'RUN.json'
    if report.exists():
        raise RuntimeError('Native distribution reports are immutable')
    meta = {'status': 'RUNNING', 'old_bindings': old_bindings, 'new_bindings': new_bindings,
        'historical_source_receipts': history.receipts,
        'protocol_sha256': digest(ROOT / 'config/onpolicy_ub1_distribution_op2d.json'),
        'sources': {str(ROOT / 'tools' / name): digest(ROOT / 'tools' / name) for name in [
            Path(__file__).name, 'original_onpolicy_data.py', 'ub1_native_data.py', 'historical_source_snapshot.py', 'onpolicy_residual.py']},
        'prefix_controls': [], 'positions': [], 'quality_or_speed_acceptance': False}
    atomic_json(report, meta)
    try:
        # All prefix controls precede any state-drift statistics.
        for index in range(24):
            a, b = read_pair(scope(old[index], 0, 0), scope(new[index], 0, 0))
            if not same_bits(a['V1'], b['V1']) or not same_bits(a['TeacherY'], b['TeacherY']):
                raise RuntimeError('Native first-FFN prefix differs; stop before drift interpretation')
            meta['prefix_controls'].append({'index': index, 'V1_bitwise': True, 'TeacherY_bitwise': True,
                                            'gate_and_raw_bias_bitwise': True})
        atomic_json(report, meta)
        for step in range(4):
            cases = []
            for index in range(24):
                a, b = read_pair(scope(old[index], step, 20), scope(new[index], step, 20))
                cases.append({'index': index, 'V1': difference(b['V1'], a['V1']),
                    'TeacherY': difference(b['TeacherY'], a['TeacherY']),
                    'C24_own_state_gated_native_SSE': native_sse(a['StudentY'], a['TeacherY'], a['gate']),
                    'UB1_own_state_gated_native_SSE': native_sse(b['StudentY'], b['TeacherY'], b['gate'])})
            summary = {}
            for tensor in ['V1', 'TeacherY']:
                summary[tensor] = {'equal_source_mean_relative_RMSE': float(np.mean([c[tensor]['relative_RMSE'] for c in cases])),
                    'pooled_relative_RMSE': math.sqrt(sum(c[tensor]['SSE'] for c in cases) / sum(c[tensor]['reference_energy'] for c in cases)),
                    'equal_source_mean_cosine': float(np.mean([c[tensor]['cosine'] for c in cases]))}
            meta['positions'].append({'step': step, 'block': 20, 'source_count': 24, 'summary': summary, 'cases': cases})
            atomic_json(report, meta)
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU description unexpectedly initialized CUDA')
        meta.update(status='COMPLETED_OP2D_NATIVE_STATE_DISTRIBUTION_DESCRIPTION', CUDA_initialized=False,
            scope='Known24 paired native states; own-state student errors have different inputs and widths. No same-input policy, held-out, speed or video-quality acceptance.')
    except BaseException as error:
        meta.update(status='FAILED_PRESERVED', error=str(error))
        raise
    finally:
        atomic_json(report, meta)


if __name__ == '__main__':
    main()
