"""Build two fixed, source-calibrated prototype banks in the HF derived cache."""
import argparse
import json
from pathlib import Path

from e0_affine_fold_math import fit_coefficients
from e0_merged_calibration import MergedCalibrationStore
from e0_nonlinear_reconstruction_math import dual_correction
from e0_saved_function_common import ROOT, prepare, check_output_budget
from ffnds_observation_common import sha256, write_json


def execute(args):
    kinds = ['W1', 'W2', 'affine_calibrations', 'ridge_solve', 'calibration_Gram',
             'fitted_correction_maps', 'FP32_matrix_compositions', 'persisted_operator_packs']
    plan, record = prepare(args, 'E0_RECONSTRUCTION_BANK_20261026', dict.fromkeys(kinds, 0))
    monitor = None
    try:
        import torch
        from safetensors.torch import save_file
        from startup_gpu_reservation import reserve_before_assets
        reservation, monitor, receipt = reserve_before_assets(torch)
        record['CUDA_reservation'] = receipt
        if 'A100' not in torch.cuda.get_device_name(0):
            raise ValueError('Registered A100 required')
        torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
        store = MergedCalibrationStore(args.calibration_root, plan['calibration_sources'], torch)
        cache = json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())
        record['packs'] = []

        def charge(name):
            record['counts'][name] += 1
            if record['counts'][name] > plan['count_limits'][name]:
                raise RuntimeError('Frozen bank budget exceeded')

        def project(x, w, name='W2'):
            charge(name); return x @ w.T

        def compose(x, w):
            charge('FP32_matrix_compositions'); return x @ w

        with torch.inference_mode():
            for block in plan['blocks']:
                for expert, steps in [('high_noise_model', [0, 1]), ('low_noise_model', [2, 3])]:
                    shard = next(s for s in cache['shards'] if s['expert'] == expert and s['block'] == block)
                    if sha256(Path(shard['path'])) != shard['sha256']:
                        raise ValueError('Original decoded shard changed')
                    params = torch.load(shard['path'], map_location='cpu', weights_only=False)[expert]
                    up = params[f'blocks.{block}.ffn.0']; down = params[f'blocks.{block}.ffn.2']
                    w1n, w2n = [v['weight_qdq'].to('cuda', dtype=torch.bfloat16) for v in [up, down]]
                    w1, w2 = w1n.double(), w2n.double()
                    data, aa = [], []
                    for source in plan['calibration_sources']:
                        for step in steps:
                            c = store.load(source, step, block); data.append(c)
                            aa.append(project(c['q1_post'], w1, 'W1') + c['b1'].reshape(1, -1))
                    a = torch.cat(aa); q = torch.cat([c['q2_post'] for c in data]); y = torch.cat([c['ffn_output'] for c in data])
                    if len(a) != 256:
                        raise ValueError('Fixed calibration row count changed')
                    slope, offset = fit_coefficients(a, q); charge('affine_calibrations')
                    b1, b2 = data[0]['b1'].reshape(-1), data[0]['b2'].reshape(-1)
                    if any(not torch.equal(c['b1'].reshape(-1), b1) or not torch.equal(c['b2'].reshape(-1), b2) for c in data):
                        raise ValueError('Expert biases changed')
                    residual = q - (a * slope + offset)
                    affine = project(a * slope + offset, w2) + b2
                    mean = y.mean(0, keepdim=True)
                    g2 = torch.stack([c['gate'].reshape(-1).square() for c in data]).mean(0)
                    norm = (w2.square() * g2[:, None]).sum(0)
                    matrix0 = compose(w2n.float() * slope.float(), w1n.float())
                    bias0 = w2n.float() @ (slope.reshape(-1) * b1 + offset.reshape(-1)).float() + b2.float()
                    for family, groups in [('whole', plan['whole_groups']), ('affine_residual', plan['affine_residual_groups'])]:
                        f = q if family == 'whole' else residual
                        score = (f.var(0, unbiased=True) * norm).reshape(216, 64).sum(1)
                        selected = torch.argsort(score, descending=True, stable=True)[:groups]
                        ids = (selected[:, None] * 64 + torch.arange(64, device='cuda')[None]).reshape(-1)
                        fm = f[:, ids].mean(0, keepdim=True); fc = f[:, ids] - fm
                        base = project(fc, w2[:, ids]) + (mean if family == 'whole' else affine)
                        error = y - base; em = error.mean(0, keepdim=True)
                        coeff, _, penalty = dual_correction(fc, error - em, plan['ridge_fraction'], torch)
                        for name in ['calibration_Gram', 'ridge_solve', 'fitted_correction_maps']: charge(name)
                        v = w2[:, ids] + (fc.T @ coeff).T
                        pack = {'w1': w1n[ids].contiguous(), 'v': v.bfloat16().contiguous(),
                                'b1': b1[ids].bfloat16().contiguous(),
                                's1': up['smooth_scale'].float().reshape(1, -1).contiguous().to('cuda'),
                                's2': down['smooth_scale'].float().reshape(1, -1)[:, ids.cpu()].contiguous().to('cuda')}
                        if family == 'whole':
                            pack['bias'] = (mean + em - fm @ v.T).reshape(-1).bfloat16().contiguous()
                        else:
                            pack['matrix'] = (matrix0 - compose(v.float() * slope[:, ids].float(), w1n[ids].float())).bfloat16().contiguous()
                            pack['bias'] = (bias0 - v.float() @ (slope[:, ids].reshape(-1) * b1[ids] +
                                           offset[:, ids].reshape(-1) + fm.reshape(-1)).float() + em.reshape(-1).float()).bfloat16().contiguous()
                        directory = args.out / family; directory.mkdir(exist_ok=True)
                        path = directory / f'{expert}_block{block:02d}.safetensors'
                        save_file({k: v.cpu().contiguous() for k, v in pack.items()}, str(path))
                        charge('persisted_operator_packs')
                        record['persisted_fitted_weights'] = record['counts']['persisted_operator_packs']
                        record['packs'].append({'family': family, 'expert': expert, 'block': block,
                                                'groups': selected.tolist(), 'path': str(path.relative_to(args.out)),
                                                'sha256': sha256(path), 'bytes': path.stat().st_size,
                                                'ridge_lambda': penalty,
                                                'training_rows': 256, 'training_sources': plan['calibration_sources'],
                                                'new_weight_format': 'BF16 learned closed-form correction and FP32-composed/BF16 affine; no checkpoint re-HiF4 quantization'})
                        del pack
                    record['cases'].append({'block': block, 'expert': expert}); monitor.check()
                    check_output_budget(args.out, plan); write_json(args.out / 'RUN.json', record)
                    print(f'fixed prototype banks block{block:02d} {expert}', flush=True)
                    del params, w1, w2, w1n, w2n, data, aa
        if record['counts'] != plan['count_limits'] or len(record['cases']) != plan['max_cases']:
            raise RuntimeError('Incomplete fixed bank scope')
        record['status'] = 'PASS_FIXED_RECONSTRUCTION_BANKS'
    except BaseException as exc:
        record.update(status='FAILED_PRESERVED', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        if monitor is not None:
            monitor.close(); record['assigned_GPU_resources'] = monitor.report()
        write_json(args.out / 'RUN.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'calibration-root', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    execute(parser.parse_args())
