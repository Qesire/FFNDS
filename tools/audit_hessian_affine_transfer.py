"""CPU-only saved-operand audit after R8F2; no new encoding or source selection."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from ar2_checkpoint import atomic_json, digest
from restored_assets import ROOT
from ub1_native_data import load_ub1_collections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if torch.cuda.is_initialized():
        raise RuntimeError('Saved-value audit must not initialize CUDA')
    old_path = ROOT / 'research/20261021/R8F1_native_affine_results.json'
    new_path = ROOT / 'research/20261021/R8F2_native_hessian_results.json'
    old, new = [json.loads(p.read_text()) for p in [old_path, new_path]]
    cache = json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())
    hf = Path(cache['path']).parents[1]
    runs = ROOT / 'runs/worker_restart_20261020'
    records, bindings, _ = load_ub1_collections([runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in (0, 1)], ROOT)
    protocol = json.loads((ROOT / 'research/20261021/R8F2_group64_hessian_protocol.json').read_text())
    train_ids, val_ids = protocol['inputs']['training_indices'], protocol['inputs']['indices']
    output = []
    for step, block in protocol['positions']:
        first = next(v for v in old['matrices'] if (v['step'], v['block']) == (step, block))
        second = next(v for v in new['encodings'] if (v['step'], v['block']) == (step, block))
        path0, path1 = hf / first['HF_derived_relative'], hf / second['HF_derived_relative']
        if digest(path0) != first['sha256'] or digest(path1) != second['sha256']:
            raise RuntimeError('Saved encoded matrices changed')
        values = load_file(str(path0), device='cpu')
        a = values['A_BF16'].double()
        errors = {'RTN': values['A_HiF4'].double() - a,
                  'Hessian': load_file(str(path1), device='cpu')['A_HiF4_group64_Hessian'].double() - a}
        inputs, gates, payloads = {}, {}, {}
        for index in train_ids + val_ids:
            scope = next(s for s in records[index]['scopes'] if (s['step'], s['block']) == (step, block))
            if digest(scope['path']) != scope['sha256']:
                raise RuntimeError('Original sampled post-Q1 operands changed')
            data = load_file(scope['path'], device='cpu')
            inputs[index], gates[index] = data['V1'].double(), data['gate'].double()
            payloads[index] = scope['sha256']
        train_x = torch.cat([inputs[i] for i in train_ids])
        covariance = train_x.T @ train_x / len(train_x)
        effective_dimension = float(covariance.trace().square() / covariance.square().sum())
        # The QR is descriptive only: no damping/encoding is selected from it.
        q, r = torch.linalg.qr(train_x.T, mode='reduced')
        diagonal = r.diag().abs()
        rows = []
        for index in train_ids + val_ids:
            x = inputs[index]
            losses, gated = {}, {}
            for label, error in errors.items():
                difference = x @ error.T
                losses[label] = float(difference.square().sum())
                gated[label] = float((difference * gates[index]).square().sum())
            orthogonal = x - (x @ q) @ q.T
            rows.append({'index': index, 'used_for_encoding': index in train_ids,
                         'affine_FP64_SSE': losses, 'affine_gated_FP64_SSE': gated,
                         'input_energy_outside_train_QR_span_fraction': float(orthogonal.square().sum() / x.square().sum())})
        totals = {scope: {label: sum(r['affine_FP64_SSE'][label] for r in rows if r['used_for_encoding'] == (scope == 'train'))
                          for label in errors} for scope in ['train', 'validation']}
        if abs(totals['train']['Hessian'] - second['training_affine_SSE_Hessian']) > 1e-8 * max(1., totals['train']['Hessian']):
            raise RuntimeError('GPU-reported training affine objective does not reproduce inFP64CPU')
        output.append({'step': step, 'block': block, 'input_columns': 5120, 'training_rows': len(train_x),
            'maximum_covariance_rank_from_row_count': len(train_x), 'trace_effective_dimension': effective_dimension,
            'QR_diagonal_min': float(diagonal.min()), 'QR_diagonal_max': float(diagonal.max()),
            'weight_error_Frobenius_norm2': {k: float(v.square().sum()) for k, v in errors.items()},
            'FP64_affine_SSE': totals, 'Hessian_over_RTN_affine_SSE': {k: v['Hessian'] / v['RTN'] for k, v in totals.items()},
            'source_payload_sha256': payloads, 'rows': rows})
        print(json.dumps({k: output[-1][k] for k in ['step', 'trace_effective_dimension', 'FP64_affine_SSE', 'Hessian_over_RTN_affine_SSE']}), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU audit unexpectedly initialized CUDA')
    atomic_json(args.out, {'status': 'PASS_CPU_SAVED_HESSIAN_TRANSFER_AUDIT', 'source_sha256': digest(Path(__file__)),
        'R8F1_sha256': digest(old_path), 'R8F2_sha256': digest(new_path), 'positions': output,
        'CUDA_initialized': False, 'new_model_runs': 0, 'new_fits_or_encodings': 0,
        'scope': 'Same12 train and4 pre-registered validation sources. Descriptive saved-value audit only; rank cap alone does not establish inadequate data or justify a sample-count sweep.'})


if __name__ == '__main__':
    main()
