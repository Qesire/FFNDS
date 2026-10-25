"""Saved-data discrimination of half-slope, empirical and Gaussian GELU gains."""
import argparse
import json
from pathlib import Path

import numpy as np

from e0_affine_fold_math import fit_coefficients
from e0_gelu_tangent import derivative
from e0_function_basis_math import gaussian_gain
from e0_saved_function_common import ROOT, ObservationStore, check_output_budget, describe, energy, prepare
from ffnds_observation_common import sha256, write_json


def execute(args):
    plan, record = prepare(args, 'E0_STATISTICAL_GELU_GAINS_20261026',
                           {'W1': 0, 'W2': 0, 'continuous_GELU_evaluations': 0,
                            'calibration_parameter_sets': 0})
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
        dev = lambda value: torch.as_tensor(value).to(device='cuda', dtype=torch.float64)
        store = ObservationStore(args.raw_root, torch)
        cache = json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())
        nodes, weights = [dev(v) for v in np.polynomial.hermite.hermgauss(plan['gaussian_quadrature_nodes'])]

        def charge(name):
            record['counts'][name] += 1
            if record['counts'][name] > plan['count_limits'][name]:
                raise RuntimeError('Frozen diagnostic budget exceeded')

        def project(x, weight, kind):
            charge(kind)
            return x @ weight.T

        with torch.inference_mode():
            for block in plan['blocks']:
                for expert, steps in [('high_noise_model', [0, 1]), ('low_noise_model', [2, 3])]:
                    shard = next(s for s in cache['shards'] if s['expert'] == expert and s['block'] == block)
                    if sha256(Path(shard['path'])) != shard['sha256']:
                        raise ValueError('Native decoded weight changed')
                    params = torch.load(shard['path'], map_location='cpu', weights_only=False)[expert]
                    w1 = params[f'blocks.{block}.ffn.0']['weight_qdq'].to('cuda', dtype=torch.bfloat16).double()
                    w2 = params[f'blocks.{block}.ffn.2']['weight_qdq'].to('cuda', dtype=torch.bfloat16).double()
                    smooth = dev(params[f'blocks.{block}.ffn.2']['smooth_scale']).reshape(1, -1)
                    for step in steps:
                        data = {}
                        for index in plan['source_indices']:
                            c = store.load(index, step, block, {'q1_post', 'gelu', 'q2_post', 'ffn_output', 'gate', 'b1', 'b2'})
                            a = project(c['q1_post'], w1, 'W1') + c['b1'].reshape(1, -1)
                            t = a[128:]
                            slopes = {
                                'half_slope': torch.full_like(smooth, .5) / smooth,
                                'Q2_OLS': fit_coefficients(t, c['q2_post'][128:])[0],
                                'GELU_OLS': fit_coefficients(t, c['gelu'][128:] / smooth)[0],
                                'empirical_mean_derivative': derivative(t).mean(0, keepdim=True) / smooth,
                                'gaussian_moment_gain': gaussian_gain(t.mean(0), t.var(0, unbiased=False), nodes, weights)[None] / smooth,
                            }
                            models = {}
                            for name, slope in slopes.items():
                                offset = c['q2_post'][128:].mean(0, keepdim=True) - slope * t.mean(0, keepdim=True)
                                charge('calibration_parameter_sets')
                                models[name] = (slope, offset)
                            data[index] = (c, a, models)
                        for index, (c, a, _) in data.items():
                            other = 12 - index
                            item = {'source_index': index, 'seed': c['seed'], 'step': step, 'block': block,
                                    'calibration_source': other, 'calibration_rows': 32, 'results': {}}
                            for name, (slope, offset) in data[other][2].items():
                                guess = project(a * slope + offset, w2, 'W2') + c['b2'].reshape(1, -1)
                                item['results'][name] = describe(guess, c['ffn_output'], c['ffn_output'], c['gate'],
                                                                  {k: c['panels'][k] for k in ['contiguous128', 'uniform32']})
                            decoded = project(c['q2_post'], w2, 'W2') + c['b2'].reshape(1, -1)
                            charge('continuous_GELU_evaluations')
                            h = torch.nn.functional.gelu(a, approximate='tanh')
                            odd = project(.5 * a / smooth, w2, 'W2') * c['gate']
                            even = project((h - .5 * a) / smooth, w2, 'W2') * c['gate']
                            item['decomposition'] = {}
                            for name in ['contiguous128', 'uniform32']:
                                ids = c['panels'][name]
                                u, v = odd[ids], even[ids]
                                total = u + v
                                norm = energy(total)
                                item['decomposition'][name] = {
                                    'odd_energy_over_continuous_no_bias_output': energy(u) / max(norm, 1e-30),
                                    'even_energy_over_continuous_no_bias_output': energy(v) / max(norm, 1e-30),
                                    'odd_even_cosine': float((u * v).sum()) / max((energy(u) * energy(v)) ** .5, 1e-30),
                                    'continuous_GELU_vs_actual_Q2_output_SSE': energy(total + c['b2'].reshape(1, -1) * c['gate'] - decoded[ids] * c['gate']) / max(energy(decoded[ids] * c['gate']), 1e-30),
                                }
                            item['continuous_native_decode_SSE'] = energy((decoded - c['ffn_output']) * c['gate']) / max(energy(c['ffn_output'] * c['gate']), 1e-30)
                            item['calibration_slope_discrepancy'] = {
                                name: energy((data[other][2][name][0] - data[other][2]['GELU_OLS'][0]) * smooth) /
                                max(energy(data[other][2]['GELU_OLS'][0] * smooth), 1e-30)
                                for name in ['half_slope', 'empirical_mean_derivative', 'gaussian_moment_gain']}
                            training_a = data[other][1][128:]
                            item['training_preactivation_negative_fraction'] = float((training_a < 0).double().mean())
                            record['cases'].append(item)
                            monitor.check()
                        del data
                        write_json(args.out / 'RUN.json', record)
                        print(f'statistical GELU block{block:02d} step{step}: two held-source cases complete', flush=True)
                    del params, w1, w2
        if record['counts'] != plan['count_limits'] or len(record['cases']) != plan['max_cases']:
            raise RuntimeError('Incomplete fixed scope')
        record['HF_output_bytes_before_final_receipt'] = check_output_budget(args.out, plan)
        record['status'] = 'PASS_STATISTICAL_GELU_GAINS'
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
    for name in ['protocol', 'raw-root', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    execute(parser.parse_args())
