"""Separate output-subspace capacity from early coefficient prediction."""
import argparse
from pathlib import Path

from e0_function_basis_math import ridge_factors, shuffled_factors
from e0_saved_function_common import ObservationStore, check_output_budget, describe, energy, prepare
from ffnds_observation_common import write_json


def execute(args):
    plan, record = prepare(args, 'E0_OUTPUT_BASIS_PREDICTION_20261026',
                           {'activation_SVD': 0, 'random_basis_QR': 0, 'fitted_diagnostic_maps': 0,
                            'temporal_affine_calibrations': 0})
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
        store = ObservationStore(args.raw_root, torch)

        def charge(name, count=1):
            record['counts'][name] += count
            if record['counts'][name] > plan['count_limits'][name]:
                raise RuntimeError('Frozen diagnostic budget exceeded')

        def svd(x):
            charge('activation_SVD')
            return torch.linalg.svd(x, full_matrices=False, driver='gesvd')

        generator = torch.Generator(device='cpu').manual_seed(plan['random_basis_seed'])
        random = torch.randn(5120, max(plan['ranks']), generator=generator, dtype=torch.float64).to('cuda')
        charge('random_basis_QR')
        random, _ = torch.linalg.qr(random, mode='reduced')
        with torch.inference_mode():
            for block in plan['blocks']:
                for step in range(4):
                    data = {i: store.load(i, step, block, {'q1_post', 'ffn_output', 'gate'}) for i in plan['source_indices']}
                    old = {i: store.load(i, step - 1, block, {'q1_post', 'ffn_output'}) for i in plan['source_indices']} if step in [1, 3] else None
                    if old is not None and any(data[i]['row_ids'] != old[i]['row_ids'] for i in plan['source_indices']):
                        raise ValueError('Temporal inputs and targets are not token-paired')
                    for form in (['absolute', 'innovation'] if old is not None else ['absolute']):
                        fields, fits = {}, {}
                        for index, c in data.items():
                            x = c['q1_post'] if form == 'absolute' else c['q1_post'] - old[index]['q1_post']
                            y = c['ffn_output'] if form == 'absolute' else c['ffn_output'] - old[index]['ffn_output']
                            fields[index] = (x, y)
                            fits[index] = ridge_factors(x[:128], y[:128], plan['ridge_fraction'], max(plan['ranks']), svd)
                            perm = torch.roll(torch.arange(128, device='cuda'), 64)
                            fits[index]['shuffled_left'], fits[index]['shuffled_right'] = shuffled_factors(fits[index], y[:128], perm, max(plan['ranks']))
                            charge('fitted_diagnostic_maps', 3)
                        for index, c in data.items():
                            other = 12 - index
                            x, y = fields[index]
                            fit, own = fits[other], fits[index]
                            panels = {k: c['panels'][k] for k in ['contiguous128', 'uniform_disjoint']}
                            item = {'source_index': index, 'seed': c['seed'], 'step': step, 'block': block,
                                    'form': form, 'calibration_source': other, 'calibration_rows': 128,
                                    'uniform_overlap_removed': 32 - len(panels['uniform_disjoint']),
                                    'source_gate_equal': torch.equal(c['gate'], data[other]['gate']),
                                    'ridge_lambda': fit['lambda'], 'results': {}}
                            constant = fit['mean_y'].expand_as(y)
                            item['results']['other_source_mean'] = describe(constant, y, c['ffn_output'], c['gate'], panels)
                            if form == 'innovation':
                                source_old = old[other]['ffn_output'][:128]
                                source_cur = data[other]['ffn_output'][:128]
                                xo = source_old - source_old.mean(0, keepdim=True)
                                yc = source_cur - source_cur.mean(0, keepdim=True)
                                g = data[other]['gate']
                                alpha = float(((xo * g) * (yc * g)).sum() / (xo * g).square().sum().clamp_min(1e-30))
                                beta = source_cur.mean(0, keepdim=True) - alpha * source_old.mean(0, keepdim=True)
                                charge('temporal_affine_calibrations')
                                guess = (alpha - 1) * old[index]['ffn_output'] + beta
                                item['results']['other_source_temporal_affine'] = describe(guess, y, c['ffn_output'], c['gate'], panels)
                            item['source_spectrum'] = {'singular_values': [float(v) for v in own['singular_values']],
                                                       'centered_sample_rank_upper_bound': 127,
                                                       'scope': 'Center128 own-source ungated target spectrum; posthoc, not global model rank.'}
                            for rank in plan['ranks']:
                                basis = fit['pca_right'][:rank]
                                own_basis = own['pca_right'][:rank]
                                models = {
                                    'cross_source_basis_true_coeff': fit['mean_y'] + (y - fit['mean_y']) @ basis.T @ basis,
                                    'own_center_basis_true_coeff': own['mean_y'] + (y - own['mean_y']) @ own_basis.T @ own_basis,
                                    'random_basis_true_coeff': fit['mean_y'] + (y - fit['mean_y']) @ random[:, :rank] @ random[:, :rank].T,
                                    'input_ridge_output_PCA': fit['mean_y'] + (x - fit['mean_x']) @ fit['pca_left'][:, :rank] @ basis,
                                    'input_predictive_RRR': fit['mean_y'] + (x - fit['mean_x']) @ fit['predictive_left'][:, :rank] @ fit['predictive_right'][:rank],
                                    'input_ridge_token_mismatch': fit['mean_y'] + (x - fit['mean_x']) @ fit['shuffled_left'][:, :rank] @ fit['shuffled_right'][:rank],
                                }
                                for name, guess in models.items():
                                    item['results'][f'{name}_r{rank}'] = describe(guess, y, c['ffn_output'], c['gate'], panels)
                                yc = y - fit['mean_y']
                                predicted_coeff = (x - fit['mean_x']) @ fit['pca_left'][:, :rank]
                                tail = yc - yc @ basis.T @ basis
                                coeff_error = yc @ basis.T - predicted_coeff
                                actual_error = yc - predicted_coeff @ basis
                                difference = abs(energy(actual_error) - energy(tail) - energy(coeff_error))
                                if difference > 1e-10 * max(energy(actual_error), 1):
                                    raise RuntimeError('Orthogonal capacity/prediction decomposition failed')
                                item.setdefault('ungated_error_decomposition', {})[str(rank)] = {
                                    panel: {'tail_over_target_energy': energy(tail[ids]) / max(energy(y[ids]), 1e-30),
                                            'coefficient_error_over_target_energy': energy(coeff_error[ids]) / max(energy(y[ids]), 1e-30),
                                            'scope': 'Euclidean ungated target; not an identity in a general gate-weighted metric.'}
                                    for panel, ids in panels.items()}
                            record['cases'].append(item)
                            monitor.check()
                        del fields, fits
                    del data, old
                    write_json(args.out / 'RUN.json', record)
                    print(f'output basis block{block:02d} step{step}: forms and held-source controls complete', flush=True)
        if record['counts'] != plan['count_limits'] or len(record['cases']) != plan['max_cases']:
            raise RuntimeError('Incomplete fixed scope')
        record['HF_output_bytes_before_final_receipt'] = check_output_budget(args.out, plan)
        record['status'] = 'PASS_OUTPUT_BASIS_PREDICTION'
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
