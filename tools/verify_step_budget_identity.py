"""Require the budget runtime's108-group identity mode to preserve C24 exactly."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from ar2_checkpoint import atomic_json, digest
from run_ar2_restart import tensor_sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True); ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); args = ap.parse_args(); torch.set_num_threads(8)
    ref, candidate = [json.loads(p.read_text()) for p in [args.reference, args.candidate]]
    if any(v['status'] != 'PASS' for v in [ref, candidate]) or ref['input_record'] != candidate['input_record']:
        raise RuntimeError('Budget identity input/status differs')
    for field in ['arm', 'checkpoint_sha256', 'quantized_cache_identity', 'versions', 'contract', 'resolved_contract']:
        if ref[field] != candidate[field]: raise RuntimeError('Budget identity assets/environment differ: ' + field)
    packed = candidate['hif4_runtime']['packed_reference']
    if (not packed['identity_control'] or packed['groups_by_step'] != [108] * 4
            or packed['policy']['parent_policy']['sha256'] != ref['packed_policy_sha256']
            or packed['actual_FFN_linear_calls'] != 320 or packed['actual_global_main_MAC_density'] != .5
            or packed['residual_correction']['actual_positions'] != 160):
        raise RuntimeError('Incomplete or different-parent budget identity control')
    hashes = []
    if len(ref['trajectory']) != 4 or len(candidate['trajectory']) != 4: raise RuntimeError('Four native latents required')
    for old, new in zip(ref['trajectory'], candidate['trajectory']):
        if old['step'] != new['step']: raise RuntimeError('Step order differs')
        for row in [old, new]:
            if digest(row['path']) != row['sha256']: raise RuntimeError('Latent file changed')
        with safe_open(old['path'], framework='pt', device='cpu') as a, safe_open(new['path'], framework='pt', device='cpu') as b:
            x, y = a.get_tensor('latents'), b.get_tensor('latents')
            if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x.view(torch.uint8), y.view(torch.uint8)):
                raise RuntimeError('Budget identity changed raw latent bits')
            hashes.append({'step': old['step'], 'sha256': tensor_sha(x)})
    if digest(ref['video']['path']) != digest(candidate['video']['path']): raise RuntimeError('Budget identity video bytes differ')
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / 'RUN.json', {'status': 'PASS_UB1_IDENTITY_FULL160_FOUR_LATENTS_AND_VIDEO_BYTES',
        'reference_sha256': digest(args.reference), 'candidate_sha256': digest(args.candidate),
        'source_sha256': digest(Path(__file__)), 'latent_hashes': hashes, 'video_sha256': digest(candidate['video']['path']),
        'quality_or_speed_acceptance': False})


if __name__ == '__main__': main()
