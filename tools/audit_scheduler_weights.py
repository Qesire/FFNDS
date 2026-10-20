"""Reconcile configured indices, native sigma/timestep traces and update coefficients."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import torch

from ar2_checkpoint import atomic_json, digest
from run_ar2_restart import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); torch.set_num_threads(8)
    ref = json.loads(args.reference.read_text())
    if ref['status'] != 'PASS' or len(ref['trajectory']) != 4: raise RuntimeError('Complete native trace required')
    config = ref['resolved_contract']; source = ROOT / '.runtime/code/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py'
    tree = ast.parse(source.read_text()); base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanStepDistillScheduler')
    method = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == 'set_denoising_timesteps')
    namespace = {'torch': torch, 'Union': Union}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    probe = SimpleNamespace(sigma_min=0., sigma_max=1., sample_shift=config['sample_shift'],
        num_train_timesteps=1000, denoising_step_list=config['denoising_step_list'])
    namespace['set_denoising_timesteps'](probe, device='cpu')
    sigmas = probe.sigmas.tolist(); timesteps = probe.timesteps.tolist()
    observed = ref['trajectory']
    if sigmas != [r['sigma'] for r in observed] or timesteps != [r['scheduler_timestep'] for r in observed]:
        raise RuntimeError('Native source CPU reproduction differs from actual recorded schedule')
    delta = [a - b for a, b in zip(sigmas, sigmas[1:] + [0.])]
    rows = [{'step': i, 'configured_index': config['denoising_step_list'][i], 'actual_sigma': sigmas[i],
        'actual_model_timestep': timesteps[i], 'flow_update_coefficient_magnitude': delta[i],
        'relative_to_first_coefficient': delta[i] / delta[0],
        'relative_squared_coefficient': (delta[i] / delta[0])**2,
        'observed_native_latent_dtype': observed[i]['dtype']} for i in range(4)]
    result = {'status': 'PASS_NATIVE_SOURCE_AND_RECORDED_SCHEDULE_RECONCILIATION',
        'reference_sha256': digest(args.reference), 'scheduler_source_sha256': digest(source),
        'audit_source_sha256': digest(Path(__file__)), 'sample_shift': config['sample_shift'],
        'rows': rows, 'scope': 'Continuous algebraic flow-prediction coefficient at fixed input; not an FFN Jacobian or final-video sensitivity',
        'rounding': 'Native step_post evaluates two FP32 products/additions and casts back to the observed latent dtype; do not claim bit identity with a fused delta-sigma formula',
        'configured_indices_are_not_shifted_model_timesteps': True,
        'UB1_policy_already_frozen_sha256': digest(ROOT / 'runs/worker_restart_20261020/ub1_step_budget_fit/UB1_policy.json'),
        'quality_or_speed_acceptance': False, 'cuda_initialized': torch.cuda.is_initialized()}
    if result['cuda_initialized']: raise RuntimeError('Schedule audit must remain CPU-only')
    args.out.parent.mkdir(parents=True, exist_ok=True); atomic_json(args.out, result)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
