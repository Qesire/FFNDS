"""Describe two failed surrogates using saved operands; no fit or model run."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from ar2_checkpoint import atomic_json, digest
from restored_assets import ROOT
from ub1_native_data import load_ub1_collections


def read(path):
    return json.loads(path.read_text())


def checked_values(record):
    path = Path(record['path'])
    if digest(path) != record['sha256']:
        raise RuntimeError('Saved operand receipt changed')
    return load_file(str(path), device='cpu')


def norm2(x):
    return float(x.square().sum())


def flow_audit():
    run = ROOT / 'runs/r7d2_gain_try1'
    fit = read(run / 'FIT.json')
    result_path = ROOT / 'research/20261021/R7D2_results.json'
    result = read(result_path)
    if digest(run / 'FIT.json') != result['raw_fit_sha256']:
        raise RuntimeError('Frozen fit changed')
    cases = {}
    for shard in (0, 1):
        receipt = result['job_receipts'][f'shard{shard}']
        path = ROOT / receipt['record']
        if digest(path) != receipt['sha256']:
            raise RuntimeError('Frozen generation receipt changed')
        for entry in read(path)['completed']:
            path = Path(entry['path'])
            if digest(path) != entry['sha256']:
                raise RuntimeError('Native generation record changed')
            case = read(path)
            cases[case['index'], case['label']] = case
    rows = []
    for index in fit['training_indices']:
        target_record = read(ROOT / f'reports/worker_20261020/TF2_DQ_input{index:02d}.json')['steps'][3]
        target = checked_values(target_record)['flow'].double()
        flows = {a: checked_values(cases[index, a]['steps'][3])['flow'].double()
                 for a in ['identity', 'direction_A', 'direction_B', 'local_fit', 'flow_fit']}
        error = flows['identity'] - target
        directions = [(flows['direction_' + a] - flows['identity']) / fit['epsilon'] for a in ['A', 'B']]
        for label in ['local_fit', 'flow_fit']:
            predicted = sum(d * scale for d, scale in zip(directions, fit[label]['delta']))
            actual = flows[label] - flows['identity']
            residual = actual - predicted
            pp, aa, rr = norm2(predicted), norm2(actual), norm2(residual)
            pred_sse, actual_sse = norm2(error + predicted), norm2(error + actual)
            cross = float(2 * ((error + predicted) * residual).sum())
            identity_residual = actual_sse - pred_sse - cross - rr
            if abs(identity_residual) > 1e-8 * max(1., actual_sse):
                raise RuntimeError('Secant failure decomposition identity failed')
            rows.append({'index': index, 'fit': label, 'identity_SSE': norm2(error),
                'predicted_SSE': pred_sse, 'actual_SSE': actual_sse,
                'predicted_change_norm2': pp, 'native_change_norm2': aa,
                'prediction_residual_norm2': rr,
                'native_predicted_change_cosine': float((actual * predicted).sum()) / (aa * pp) ** .5,
                'native_over_predicted_change_norm': (aa / pp) ** .5,
                'prediction_residual_over_prediction_norm': (rr / pp) ** .5,
                'SSE_gap_cross_term': cross, 'SSE_gap_residual_norm2': rr,
                'decomposition_identity_residual': identity_residual,
                'payload_sha256': {a: cases[index, a]['steps'][3]['sha256'] for a in flows},
                'target_payload_sha256': target_record['sha256']})
    return {'result_sha256': digest(result_path), 'rows': rows,
            'scope': 'Three existing fit sources only; no heldout directions were generated. Nonlinearity, casts and quantization are not separately identified by this decomposition.'}


def hessian_audit():
    old_path = ROOT / 'research/20261021/R8F1_native_affine_results.json'
    new_path = ROOT / 'research/20261021/R8F2_native_hessian_results.json'
    audit_path = ROOT / 'research/20261021/R8F2_transfer_audit.json'
    old, new, audit = [read(p) for p in [old_path, new_path, audit_path]]
    hf = Path(read(ROOT / '.runtime/hif4_rebuilt_cache.json')['path']).parents[1]
    runs = ROOT / 'runs/worker_restart_20261020'
    records, _, _ = load_ub1_collections([runs / f'op2_ub1_calibration24/shard{s}/RUN.json' for s in (0, 1)], ROOT)
    protocol = read(ROOT / 'research/20261021/R8F2_group64_hessian_protocol.json')
    train_ids, val_ids = protocol['inputs']['training_indices'], protocol['inputs']['indices']
    positions = []
    for prior in audit['positions']:
        step, block = prior['step'], prior['block']
        first = next(v for v in old['matrices'] if (v['step'], v['block']) == (step, block))
        second = next(v for v in new['encodings'] if (v['step'], v['block']) == (step, block))
        paths = [hf / first['HF_derived_relative'], hf / second['HF_derived_relative']]
        if any(digest(p) != r['sha256'] for p, r in zip(paths, [first, second])):
            raise RuntimeError('Frozen matrix payload changed')
        values = load_file(str(paths[0]), device='cpu')
        errors = {'RTN': values['A_HiF4'].double() - values['A_BF16'].double(),
            'Hessian': load_file(str(paths[1]), device='cpu')['A_HiF4_group64_Hessian'].double() - values['A_BF16'].double()}
        inputs = {}
        for index in train_ids + val_ids:
            scope = next(s for s in records[index]['scopes'] if (s['step'], s['block']) == (step, block))
            if scope['sha256'] != prior['source_payload_sha256'][str(index)]:
                raise RuntimeError('Prior input provenance changed')
            inputs[index] = checked_values(scope)['V1'].double()
        q, _ = torch.linalg.qr(torch.cat([inputs[i] for i in train_ids]).T, mode='reduced')
        weights = {}
        for label, error in errors.items():
            inside = (error @ q) @ q.T
            weights[label] = {'in_train_span_norm2': norm2(inside), 'outside_train_span_norm2': norm2(error - inside),
                              'total_norm2': norm2(error)}
        rows = []
        for index in val_ids:
            x = inputs[index]; inside = (x @ q) @ q.T
            arms = {}
            for label, error in errors.items():
                din, dout = inside @ error.T, (x - inside) @ error.T
                whole = norm2(x @ error.T); cross = float(2 * (din * dout).sum())
                residual = whole - norm2(din) - norm2(dout) - cross
                expected = next(r for r in prior['rows'] if r['index'] == index)['affine_FP64_SSE'][label]
                if abs(whole - expected) > 1e-8 * max(1., expected) or abs(residual) > 1e-8 * max(1., whole):
                    raise RuntimeError('Saved affine decomposition did not reproduce')
                arms[label] = {'total_SSE': whole, 'in_span_SSE': norm2(din), 'outside_span_SSE': norm2(dout),
                               'cross_term': cross, 'decomposition_identity_residual': residual}
            rows.append({'index': index, 'arms': arms})
        positions.append({'step': step, 'block': block, 'weight_error': weights, 'validation': rows,
            'matrix_payload_sha256': [first['sha256'], second['sha256']]})
        print(json.dumps({'step': step, 'weight_error': weights}), flush=True)
    return {'prior_transfer_audit_sha256': digest(audit_path), 'positions': positions,
            'scope': 'Fixed existing training row span; in/out output components are correlated and require the reported cross term. This is not a new fit or proof that more tokens fixes transfer.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('Do not overwrite an audit')
    torch.set_num_threads(8)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only audit')
    result = {'id': 'R10_SAVED_SURROGATE_FAILURE_GEOMETRY', 'flow': flow_audit(), 'hessian': hessian_audit(),
              'CUDA_initialized': torch.cuda.is_initialized(), 'new_model_runs': 0, 'new_fits_or_encodings': 0,
              'source_sha256': digest(Path(__file__)), 'status': 'PASS_SAVED_OPERAND_FAILURE_DECOMPOSITION'}
    if result['CUDA_initialized']:
        raise RuntimeError('CUDA was initialized')
    atomic_json(args.out, result)
    print(json.dumps({'status': result['status']}), flush=True)


if __name__ == '__main__':
    main()
