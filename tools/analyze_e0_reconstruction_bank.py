"""Verify fixed BF16 prototype packs without any model execution."""
import argparse
import json
from pathlib import Path

from analyze_e0_dense_residency import validate_plan
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def analyze(hf):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(8)
    relative = 'config/e0_reconstruction_bank_20261026.json'
    plan, digest = validate_plan(relative)
    path = hf / 'e0_function_basis_20261026/prototype_banks_attempt1/RUN.json'
    raw = json.loads(path.read_text())
    if raw['status'] != 'PASS_FIXED_RECONSTRUCTION_BANKS' or raw['protocol_sha256'] != digest or raw['counts'] != plan['count_limits']:
        raise ValueError('Bank receipt incomplete')
    if raw['new_generations'] or raw['native_FFN_calls'] or raw['optimizer_updates'] or raw['persisted_fitted_weights'] != 160:
        raise ValueError('Bank scope mismatch')
    resources = raw['assigned_GPU_resources']
    if not resources['sampled_isolation_pass'] or resources['violations']:
        raise ValueError('Bank GPU isolation failed')
    snapshot = path.parent / 'EXECUTED_SOURCES'
    if sha256(snapshot / Path(relative).name) != digest or any(sha256(snapshot / name) != expected for name, expected in plan['bound_sources'].items()):
        raise ValueError('Bank source snapshot mismatch')
    keys = [(p['family'], p['expert'], p['block']) for p in raw['packs']]
    if len(set(keys)) != 160 or set(keys) != {(f, e, b) for f in ['whole', 'affine_residual'] for e in ['high_noise_model', 'low_noise_model'] for b in range(40)}:
        raise ValueError('Incomplete operator bank')
    parameters = 0
    for p in raw['packs']:
        target = path.parent / p['path']
        if sha256(target) != p['sha256'] or target.stat().st_size != p['bytes']:
            raise ValueError('Bank tensor bytes changed')
        tensors = load_file(str(target), device='cpu')
        affine = p['family'] == 'affine_residual'
        k = (54 if affine else 94) * 64
        expected = {'w1':(k,5120),'v':(5120,k),'b1':(k,),'bias':(5120,), 's1':(1,5120),'s2':(1,k)}
        if affine: expected['matrix'] = (5120,5120)
        if set(tensors) != set(expected) or any(tuple(tensors[n].shape) != shape for n, shape in expected.items()):
            raise ValueError('Operator shape mismatch')
        if any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
            raise ValueError('Nonfinite fitted parameter')
        if any(t.dtype != (torch.float32 if n in ['s1','s2'] else torch.bfloat16) for n,t in tensors.items()):
            raise ValueError('Operator dtype mismatch')
        parameters += sum(t.numel() for n,t in tensors.items() if n in ['v','matrix','bias'])
    size = sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())
    if size > plan['max_HF_output_gib'] * (1 << 30): raise ValueError('Bank output cap exceeded')
    return {'id':'E0_RECONSTRUCTION_BANK_20261026','status':'COMPLETE','protocol':relative,'protocol_sha256':digest,
            'analysis_sha256':sha256(Path(__file__)), 'raw_binding':{'HF_relative':str(path.relative_to(hf)),'sha256':sha256(path)},
            'actual_counts':raw['counts'],'operator_packs':160,'new_readout_affine_bias_scalar_parameters':parameters,
            'HF_output_bytes':size,'all_parameters_finite':True,'all_source_snapshots_match':True,
            'sampled_assigned_GPU_isolation_pass':True,'peak_physical_GPU_MiB':resources['peak_physical_used_memory_MiB'],
            'calibration_sources':[4,8,16,20],'held_sources_read_for_fit':[], 'packs':raw['packs'],
            'new_generations':0,'optimizer_updates':0,'native_FFN_calls':0,'model_weight_decompositions':0,
            'format':'Original selected decoded W1; NEW BF16 readout/bias and FP32-composed BF16 affine, no re-HiF4 weight quantization',
            'candidate_operator_selected':False,'quality_or_speed_acceptance':False,'stage_changed':False}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hf-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve accepted bank manifest')
    write_json(a.out,analyze(a.hf_root))
