"""Source-held native group features with mean and ridge output reconstruction."""
import argparse
import json
from pathlib import Path

from analyze_ffnds_observations import load_arrays
from e0_affine_fold_math import fit_coefficients
from e0_nonlinear_reconstruction_math import dual_correction
from e0_saved_function_common import (ROOT, MATRIX_FIELDS, ObservationStore, prepare,
                                      describe, energy, check_output_budget)
from ffnds_observation_common import sha256, write_json


class CalibrationStore:
    def __init__(self, root, sources, torch):
        self.root, self.torch = root, torch
        accepted = json.loads((ROOT / 'research/20261025/E0_NATIVE_OBSERVATIONS.json').read_text())
        self.runs = {}
        for source in sources:
            path = root / f'source{source:02d}_attempt1/RUN.json'
            binding = next(s for s in accepted['sources'] if s['index'] == source)
            if sha256(path) != binding['run_sha256']:
                raise ValueError('Calibration run binding changed')
            self.runs[source] = path, json.loads(path.read_text())

    def load(self, source, step, block):
        path, run = self.runs[source]
        scope = next(s for s in run['scopes'] if (s['step'], s['block']) == (step, block))
        payload = path.parent / 'observed' / scope['path']
        if sha256(payload) != scope['sha256']:
            raise ValueError('Calibration operand changed')
        rows = scope['row_plan']
        ids = [rows['rows'].index(r) for r in rows['uniform_rows']]
        arrays = load_arrays(payload)
        return {name: self.torch.as_tensor(arrays[name][ids] if name in MATRIX_FIELDS else arrays[name])
                .to(device='cuda', dtype=self.torch.float64)
                for name in ['q1_post', 'q2_post', 'ffn_output', 'gate', 'b1', 'b2']}


def execute(args):
    plan, record = prepare(args, 'E0_NONLINEAR_RECONSTRUCTION_20261026',
                           {'W1': 0, 'W2': 0, 'calibration_Gram': 0,
                            'ridge_solve': 0, 'fitted_correction_maps': 0,
                            'affine_calibrations': 0, 'group_rankings': 0})
    monitor = None
    try:
        import torch
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, receipt = reserve_before_assets(torch)
        record['CUDA_reservation'] = receipt
        if 'A100' not in torch.cuda.get_device_name(0):
            raise ValueError('Registered A100 required')
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        held = ObservationStore(args.raw_root, torch)
        calibration = CalibrationStore(args.calibration_root, plan['calibration_sources'], torch)
        cache = json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())
        record['calibrations'] = []

        def charge(kind, count=1):
            record['counts'][kind] += count
            if record['counts'][kind] > plan['count_limits'][kind]:
                raise RuntimeError('Frozen calculation budget exceeded')

        def project(x, w, kind='W2'):
            charge(kind)
            return x @ w.T

        with torch.inference_mode():
            for block in plan['blocks']:
                for expert, steps in [('high_noise_model', [0, 1]), ('low_noise_model', [2, 3])]:
                    shard = next(s for s in cache['shards'] if s['expert'] == expert and s['block'] == block)
                    if sha256(Path(shard['path'])) != shard['sha256']:
                        raise ValueError('Native decoded weight changed')
                    params = torch.load(shard['path'], map_location='cpu', weights_only=False)[expert]
                    w1 = params[f'blocks.{block}.ffn.0']['weight_qdq'].to('cuda', dtype=torch.bfloat16).double()
                    w2 = params[f'blocks.{block}.ffn.2']['weight_qdq'].to('cuda', dtype=torch.bfloat16).double()
                    calibration_data, aa = [], []
                    for source in plan['calibration_sources']:
                        for step in steps:
                            c = calibration.load(source, step, block)
                            aa.append(project(c['q1_post'], w1, 'W1') + c['b1'].reshape(1, -1))
                            calibration_data.append(c)
                    ac = torch.cat(aa)
                    qc = torch.cat([c['q2_post'] for c in calibration_data])
                    yc = torch.cat([c['ffn_output'] for c in calibration_data])
                    if len(qc) != plan['calibration_rows_per_expert_block']:
                        raise ValueError('Wrong calibration row count')
                    slope, offset = fit_coefficients(ac, qc)
                    charge('affine_calibrations')
                    rc = qc - (ac * slope + offset)
                    b2 = calibration_data[0]['b2'].reshape(1, -1)
                    if any(not torch.equal(c['b2'].reshape(1, -1), b2) for c in calibration_data):
                        raise ValueError('Expert bias must be fixed')
                    affine_c = project(ac * slope + offset, w2) + b2
                    mean = yc.mean(0, keepdim=True)
                    mean_gate_squared = torch.stack([c['gate'].reshape(-1).square() for c in calibration_data]).mean(0)
                    norm = (w2.square() * mean_gate_squared[:, None]).sum(0)
                    orders = {}
                    for family, features in [('whole', qc), ('affine_residual', rc)]:
                        score = (features.var(0, unbiased=True) * norm).reshape(216, 64).sum(1)
                        orders[family] = torch.argsort(score, descending=True, stable=True)
                        charge('group_rankings')
                    routes, receipt_routes = [], []
                    for family, counts in [('whole', plan['whole_group_counts']),
                                           ('affine_residual', plan['affine_residual_group_counts'])]:
                        features = qc if family == 'whole' else rc
                        for groups in counts:
                            selected_groups = orders[family][:groups]
                            ids = (selected_groups[:, None] * 64 + torch.arange(64, device='cuda')[None]).reshape(-1)
                            fmean = features[:, ids].mean(0, keepdim=True)
                            fc = features[:, ids] - fmean
                            base_c = project(fc, w2[:, ids]) + (mean if family == 'whole' else affine_c)
                            error = yc - base_c
                            error_mean = error.mean(0, keepdim=True)
                            centered_error = error - error_mean
                            joint = torch.cat([centered_error, torch.roll(centered_error, len(yc) // 2, 0)], 1)
                            coefficients, gram, penalty = dual_correction(fc, joint, plan['ridge_fraction'], torch)
                            charge('calibration_Gram'); charge('ridge_solve'); charge('fitted_correction_maps', 2)
                            name = f'{family}_g{groups}'
                            routes.append((name, family, ids, fc, fmean, error_mean, coefficients))
                            fit = base_c + error_mean + (gram @ coefficients)[:, :5120]
                            receipt_routes.append({'name': name, 'groups': selected_groups.tolist(),
                                                   'ridge_lambda': penalty,
                                                   'ideal_projection_MAC_ratio': (groups + (40 if family != 'whole' else 0)) / 216,
                                                   'training_ungated_relative_SSE_before': energy(error) / max(energy(yc), 1e-30),
                                                   'training_ungated_relative_SSE_after': energy(fit - yc) / max(energy(yc), 1e-30)})
                    record['calibrations'].append({'expert': expert, 'block': block, 'rows': len(yc),
                                                   'sources': plan['calibration_sources'], 'steps': steps,
                                                   'routes': receipt_routes})
                    for source in plan['held_sources']:
                        for step in steps:
                            c = held.load(source, step, block, {'q1_post', 'q2_post', 'ffn_output', 'gate', 'b1', 'b2'})
                            if not torch.equal(c['b2'].reshape(1, -1), b2):
                                raise ValueError('Held expert bias changed')
                            a = project(c['q1_post'], w1, 'W1') + c['b1'].reshape(1, -1)
                            affine = project(a * slope + offset, w2) + b2
                            residual = c['q2_post'] - (a * slope + offset)
                            decoded = project(c['q2_post'], w2) + b2
                            item = {'source_index': source, 'seed': c['seed'], 'step': step, 'block': block,
                                    'results': {}, 'decode_relative_SSE':
                                    energy((decoded - c['ffn_output']) * c['gate']) /
                                    max(energy(c['ffn_output'] * c['gate']), 1e-30)}
                            def save(name, prediction):
                                item['results'][name] = describe(prediction, c['ffn_output'], c['ffn_output'],
                                                                 c['gate'], c['panels'])
                            save('calibration_mean', mean.expand_as(c['ffn_output']))
                            save('calibration_affine', affine)
                            for name, family, ids, fc, fmean, error_mean, coefficients in routes:
                                ft = (c['q2_post'] if family == 'whole' else residual)[:, ids] - fmean
                                base = project(ft, w2[:, ids]) + (mean if family == 'whole' else affine)
                                correction = (ft @ fc.T) @ coefficients
                                save(name + '_mean', base)
                                save(name + '_ridge', base + error_mean + correction[:, :5120])
                                save(name + '_token_mismatch', base + error_mean + correction[:, 5120:])
                            record['cases'].append(item)
                            monitor.check()
                    write_json(args.out / 'RUN.json', record)
                    print(f'nonlinear reconstruction block{block:02d} {expert}: four held cases complete', flush=True)
                    del params, w1, w2, routes, calibration_data, aa
        if record['counts'] != plan['count_limits'] or len(record['cases']) != plan['max_cases']:
            raise RuntimeError('Incomplete fixed scope')
        record['HF_output_bytes_before_final_receipt'] = check_output_budget(args.out, plan)
        record['status'] = 'PASS_NONLINEAR_RECONSTRUCTION'
    except BaseException as exc:
        record.update(status='FAILED_PRESERVED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if monitor is not None:
            monitor.close()
            record['assigned_GPU_resources'] = monitor.report()
        write_json(args.out / 'RUN.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'raw-root', 'calibration-root', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    execute(parser.parse_args())
