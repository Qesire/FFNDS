"""Locate LA1 initial forward bit differences without fitting any coefficient."""
import json
from pathlib import Path

import torch

from ar2_checkpoint import atomic_json, digest
from evaluate_residual_rr1 import open_verified
from folded_adapter import FoldedAdapters, straight_through
from onpolicy_residual import same_bits
from run_ar2_restart import ROOT, tensor_sha


def differences(a, b):
    if a.shape != b.shape or a.dtype != b.dtype: raise RuntimeError('Mismatched probe tensors')
    bits = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    mask = a.view(bits) != b.view(bits)
    zero = (a == 0) & (b == 0)
    return {'bit_differences': int(mask.sum()), 'numerical_differences': int((a != b).sum()),
        'zero_sign_differences': int((mask & zero).sum()), 'max_abs_difference': float((a.float() - b.float()).abs().max()),
        'equal_bits': same_bits(a, b), 'dtype': str(a.dtype)}


def main():
    from HiFloat4.ptq.adapters.wan_i2v import native_hif4_weight_qdq as q
    torch.set_num_threads(8); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    r = ROOT / 'runs/worker_restart_20261020'; policy = json.loads((r / 'calibration24_residual_fit/C24_RR_policy.json').read_text())
    binding = policy['fit_reports']['low_noise_model']; base, report, identity = open_verified(
        Path(binding['path']), 'COMPLETED_CALIBRATION_PARAMETER_FIT')
    part = base.load('positions', 'step2_block20'); groups = torch.from_numpy(part['arrays']['C_groups']).long().cuda()
    ids = (groups[:, None] * 64 + torch.arange(64, device='cuda')).reshape(-1)
    cache = json.loads((ROOT / '.runtime/hif4_rebuilt_cache.json').read_text())
    shard = next(s for s in cache['shards'] if s['expert'] == 'low_noise_model' and s['block'] == 20)
    if digest(shard['path']) != shard['sha256']: raise RuntimeError('Cache shard changed')
    payload = torch.load(shard['path'], map_location='cpu', weights_only=True, mmap=True)['low_noise_model']
    w0 = payload['blocks.20.ffn.0']['weight_qdq'].cuda().index_select(0, ids).contiguous()
    w2 = payload['blocks.20.ffn.2']['weight_qdq'].cuda().index_select(1, ids).contiguous()
    model = FoldedAdapters(w0, w2); rows = []
    with torch.no_grad():
        actual = model.weights(q)
        for slot, prior, observed in zip(['0', '2'], [w0, w2], actual):
            delta = getattr(model, 'b' + slot) @ getattr(model, 'a' + slot)
            effective = prior.float() + delta
            original_q = q(prior.float(), force_fp32=True).bfloat16()
            effective_q = q(effective, force_fp32=True).bfloat16()
            repeated_q = q(effective, force_fp32=True).bfloat16()
            direct_ste = straight_through(effective, q).bfloat16()
            rows.append({'slot': slot, 'prior_sha256': tensor_sha(prior), 'delta_abs_max': float(delta.abs().max()),
                'effective_vs_prior': differences(effective, prior.float()),
                'model_vs_Q_prior': differences(observed, original_q),
                'model_vs_Q_same_effective': differences(observed, effective_q),
                'STE_vs_Q_same_effective': differences(direct_ste, effective_q),
                'native_Q_repeat': differences(effective_q, repeated_q)})
    out = r / 'folded_adapter_forward_probe'; out.mkdir(parents=True, exist_ok=True)
    result = {'status': 'COMPLETED_FORWARD_BIT_DIFFERENCE_DIAGNOSTIC', 'positions': rows,
        'coefficient_fitting': False, 'source_sha256': digest(Path(__file__)), 'adapter_source_sha256': digest(ROOT / 'tools/folded_adapter.py')}
    atomic_json(out / 'RUN.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
