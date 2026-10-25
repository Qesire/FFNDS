"""Finite native BF16/HiF4 feature reconstruction, never a video experiment."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from e0_affine_fold_math import fit_coefficients
from e0_nonlinear_reconstruction_math import dual_correction
from e0_saved_function_common import (ROOT, ObservationStore, prepare, describe,
                                      energy, check_output_budget)
from ffnds_observation_common import sha256, write_json
from run_e0_nonlinear_reconstruction import CalibrationStore


def execute(args):
    names = ['W1', 'W2', 'calibration_Gram', 'ridge_solve', 'fitted_correction_maps',
             'FP32_matrix_compositions', 'transient_BF16_output_weights',
             'native_dense_after_Q1_FFN', 'native_partial_after_Q1_FFN',
             'native_Q2', 'folded_linear_applications']
    plan, record = prepare(args, 'E0_RECONSTRUCTED_NATIVE_20261026', dict.fromkeys(names, 0))
    monitor = None
    try:
        import torch
        import torch.nn.functional as F
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, receipt = reserve_before_assets(torch)
        record['CUDA_reservation'] = receipt
        if 'A100' not in torch.cuda.get_device_name(0):
            raise ValueError('Registered A100 required')
        for sub in ['HiF4_Sp', 'LightX2V', 'Wan2.2-I2V-A14B-W4A4']:
            sys.path.insert(0, str(ROOT / '.runtime/code' / sub))
        for sub in ['HiFloat4', 'HiFloat4/hif4_gpu']:
            sys.path.insert(0, str(ROOT / '.runtime/code/HiF4_Sp' / sub))
        from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        record['environment'] = {'torch': torch.__version__, 'CUDA': torch.version.cuda,
                                 'GPU': torch.cuda.get_device_name(0), 'TF32': False,
                                 'BF16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        held = ObservationStore(args.raw_root, torch)
        calibration = CalibrationStore(args.calibration_root, plan['calibration_sources'], torch)
        cache = json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())
        record['preparation'] = []

        def charge(name):
            record['counts'][name] += 1
            if record['counts'][name] > plan['count_limits'][name]:
                raise RuntimeError('Frozen call budget exceeded')

        def project(x, w, name='W2'):
            charge(name)
            return x @ w.T

        def compose(x, w):
            charge('FP32_matrix_compositions')
            return x @ w

        def repeat(x):
            return x.repeat(math.ceil(plan['M'] / len(x)), 1)[:plan['M']].contiguous()

        def timed(call):
            for _ in range(plan['warmup_calls']):
                value = call(); del value
            durations = []
            for _ in range(plan['timed_calls']):
                torch.cuda.synchronize(); start = time.perf_counter()
                value = call(); torch.cuda.synchronize()
                durations.append(1000 * (time.perf_counter() - start)); del value
            return durations

        with torch.inference_mode():
            for block in plan['blocks']:
                for expert, steps in [('high_noise_model', [0, 1]), ('low_noise_model', [2, 3])]:
                    shard = next(s for s in cache['shards'] if s['expert'] == expert and s['block'] == block)
                    if sha256(Path(shard['path'])) != shard['sha256']:
                        raise ValueError('Original weight shard changed')
                    params = torch.load(shard['path'], map_location='cpu', weights_only=False)[expert]
                    w1n = params[f'blocks.{block}.ffn.0']['weight_qdq'].to('cuda', dtype=torch.bfloat16)
                    w2n = params[f'blocks.{block}.ffn.2']['weight_qdq'].to('cuda', dtype=torch.bfloat16)
                    w1, w2 = w1n.double(), w2n.double()
                    smooth = params[f'blocks.{block}.ffn.2']['smooth_scale'].to('cuda').float().reshape(1, -1)
                    data, aa = [], []
                    for source in plan['calibration_sources']:
                        for step in steps:
                            c = calibration.load(source, step, block); data.append(c)
                            aa.append(project(c['q1_post'], w1, 'W1') + c['b1'].reshape(1, -1))
                    ac = torch.cat(aa); qc = torch.cat([c['q2_post'] for c in data])
                    yc = torch.cat([c['ffn_output'] for c in data])
                    slope, offset = fit_coefficients(ac, qc)
                    b1, b2 = data[0]['b1'].reshape(-1), data[0]['b2'].reshape(-1)
                    if any(not torch.equal(c['b1'].reshape(-1), b1) or
                           not torch.equal(c['b2'].reshape(-1), b2) for c in data):
                        raise ValueError('Expert biases changed')
                    affine_c = project(ac * slope + offset, w2) + b2
                    mean = yc.mean(0, keepdim=True)
                    rc = qc - ac * slope - offset
                    g2 = torch.stack([c['gate'].reshape(-1).square() for c in data]).mean(0)
                    norm = (w2.square() * g2[:, None]).sum(0)
                    routes = []
                    torch.cuda.synchronize(); prep_started = time.perf_counter()
                    # One fixed affine matrix, prepared offline. Subtraction below
                    # removes the selected groups' affine terms before adding q_S.
                    matrix0 = compose(w2n.float() * slope.float(), w1n.float())
                    bias0 = (w2n.float() @ (slope.reshape(-1) * b1 + offset.reshape(-1)).float()) + b2.float()
                    for family, groups in [('whole', plan['whole_groups']),
                                           ('affine_residual', plan['affine_residual_groups'])]:
                        features = qc if family == 'whole' else rc
                        score = (features.var(0, unbiased=True) * norm).reshape(216, 64).sum(1)
                        group_ids = torch.argsort(score, descending=True, stable=True)[:groups]
                        ids = (group_ids[:, None] * 64 + torch.arange(64, device='cuda')[None]).reshape(-1)
                        fmean = features[:, ids].mean(0, keepdim=True); fc = features[:, ids] - fmean
                        base = project(fc, w2[:, ids]) + (mean if family == 'whole' else affine_c)
                        error = yc - base; emean = error.mean(0, keepdim=True)
                        coefficients, _, penalty = dual_correction(fc, error - emean, plan['ridge_fraction'], torch)
                        for kind in ['calibration_Gram', 'ridge_solve', 'fitted_correction_maps']:
                            charge(kind)
                        v = w2[:, ids] + (fc.T @ coefficients).T
                        charge('transient_BF16_output_weights')
                        if family == 'whole':
                            matrix = None
                            bias = (mean + emean - fmean @ v.T).reshape(-1).bfloat16()
                        else:
                            matrix = (matrix0 - compose(v.float() * slope[:, ids].float(), w1n[ids].float())).bfloat16()
                            bias = (bias0 - v.float() @ (slope[:, ids].reshape(-1) * b1[ids] +
                                    offset[:, ids].reshape(-1) + fmean.reshape(-1)).float() + emean.reshape(-1).float()).bfloat16()
                        routes.append({'name': family, 'ids': ids, 'v': v.bfloat16(), 'matrix': matrix, 'bias': bias,
                                       'fc': fc, 'coefficients': coefficients, 'fmean': fmean, 'emean': emean,
                                       'group_ids': group_ids.tolist(), 'ridge_lambda': penalty,
                                       'packed_w1': w1n[ids].contiguous(), 'packed_b1': b1[ids].bfloat16().contiguous(),
                                       'packed_smooth': smooth[:, ids].contiguous()})
                    torch.cuda.synchronize()
                    record['preparation'].append({'block': block, 'expert': expert,
                                                   'composition_and_fit_seconds_excludes_data_loading_and_W1': time.perf_counter() - prep_started,
                                                   'routes': [{k: r[k] for k in ['name', 'group_ids', 'ridge_lambda']} for r in routes]})
                    for source in plan['held_sources']:
                        for step in steps:
                            c = held.load(source, step, block, {'q1_post', 'q2_post', 'ffn_output', 'gate', 'b1', 'b2'})
                            a = project(c['q1_post'], w1, 'W1') + c['b1'].reshape(1, -1)
                            affine = project(a * slope + offset, w2) + b2
                            resid = c['q2_post'] - a * slope - offset
                            xf = repeat(c['q1_post'].bfloat16()); gate = c['gate'].bfloat16()
                            def native(route=None):
                                charge('native_dense_after_Q1_FFN' if route is None else 'native_partial_after_Q1_FFN')
                                up = w1n if route is None else route['packed_w1']
                                upbias = b1.bfloat16() if route is None else route['packed_b1']
                                scale = smooth if route is None else route['packed_smooth']
                                hidden = F.gelu(F.linear(xf, up, upbias), approximate='tanh')
                                u = (hidden.float() / scale).bfloat16()
                                charge('native_Q2')
                                q = native_hif4_activation_qdq(u, force_fp32=True).bfloat16()
                                result = F.linear(q, w2n if route is None else route['v'],
                                                  b2.bfloat16() if route is None else route['bias'])
                                if route is not None and route['matrix'] is not None:
                                    charge('folded_linear_applications')
                                    result = result + F.linear(xf, route['matrix'])
                                return result
                            dense = native()[:160].clone()
                            item = {'source_index': source, 'seed': c['seed'], 'step': step, 'block': block,
                                    'native_dense_vs_saved': describe(dense.double(), c['ffn_output'], c['ffn_output'], c['gate'], c['panels']),
                                    'dense_ms': timed(native), 'methods': {}}
                            for route in routes:
                                family = route['name']; ids = route['ids']
                                f = (c['q2_post'] if family == 'whole' else resid)[:, ids] - route['fmean']
                                base = project(f, w2[:, ids]) + (mean if family == 'whole' else affine)
                                continuous = base + route['emean'] + (f @ route['fc'].T) @ route['coefficients']
                                actual = native(route)[:160].clone().double()
                                item['methods'][family] = {
                                    'continuous_vs_saved': describe(continuous, c['ffn_output'], c['ffn_output'], c['gate'], c['panels']),
                                    'native_vs_saved': describe(actual, c['ffn_output'], c['ffn_output'], c['gate'], c['panels']),
                                    'native_vs_continuous_relative_SSE': energy((actual - continuous) * c['gate']) / max(energy(continuous * c['gate']), 1e-30),
                                    'full_M_ms': timed(lambda: native(route)),
                                    'resident_candidate_weight_bytes': sum(t.numel() * t.element_size() for t in
                                        [route['packed_w1'], route['v']] + ([route['matrix']] if route['matrix'] is not None else []))}
                            record['cases'].append(item); monitor.check()
                            write_json(args.out / 'RUN.json', record)
                            print(f'native reconstruction source{source:02d} step{step} block{block:02d}', flush=True)
                            del xf, dense
                    del params, w1, w2, w1n, w2n, routes, data, aa
        if record['counts'] != plan['count_limits'] or len(record['cases']) != plan['max_cases']:
            raise RuntimeError('Incomplete fixed native scope')
        record['HF_output_bytes_before_final_receipt'] = check_output_budget(args.out, plan)
        record['status'] = 'PASS_RECONSTRUCTED_NATIVE_FINITE'
    except BaseException as exc:
        record.update(status='FAILED_PRESERVED', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        record['native_FFN_calls'] = record['counts']['native_dense_after_Q1_FFN'] + record['counts']['native_partial_after_Q1_FFN']
        record['native_QDQ_calls'] = record['counts']['native_Q2']
        if monitor is not None:
            monitor.close(); record['assigned_GPU_resources'] = monitor.report()
        write_json(args.out / 'RUN.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'raw-root', 'calibration-root', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    execute(parser.parse_args())
