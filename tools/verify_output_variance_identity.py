"""Verify identity affine wiring preserves every native latent and video byte."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from ar2_checkpoint import atomic_json,digest
from run_ar2_restart import tensor_sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(8)
    ref=json.loads(args.reference.read_text());candidate=json.loads(args.candidate.read_text())
    if any(v['status']!='PASS' for v in [ref,candidate]) or ref['input_record']!=candidate['input_record']:
        raise RuntimeError('Identity control input/status differs')
    runtime=candidate['hif4_runtime']['packed_reference'];affine=runtime['output_affine']
    if (not affine['identity_control'] or affine['actual_positions']!=160
            or any(v['pointwise_kernel_calls']!=0 for v in affine['coverage'])
            or runtime['base_policy_sha256']!=ref['packed_policy_sha256']):
        raise RuntimeError('Candidate did not exercise complete160 exact identity controls on the same base')
    hashes=[]
    if len(ref['trajectory'])!=4 or len(candidate['trajectory'])!=4:raise RuntimeError('Four native latents required')
    for old,new in zip(ref['trajectory'],candidate['trajectory']):
        if old['step']!=new['step']:raise RuntimeError('Step order differs')
        for row in [old,new]:
            if digest(row['path'])!=row['sha256']:raise RuntimeError('Latent file changed')
        with safe_open(old['path'],framework='pt',device='cpu') as a,safe_open(new['path'],framework='pt',device='cpu') as b:
            x,y=a.get_tensor('latents'),b.get_tensor('latents')
            if x.dtype!=y.dtype or x.shape!=y.shape or not torch.equal(x.view(torch.uint8),y.view(torch.uint8)):
                raise RuntimeError('Identity affine changed raw latent bits')
            hashes.append({'step':old['step'],'raw_tensor_sha256':tensor_sha(x)})
    if digest(ref['video']['path'])!=digest(candidate['video']['path']):raise RuntimeError('Identity video bytes differ')
    args.out.mkdir(parents=True,exist_ok=True)
    atomic_json(args.out/'RUN.json',{'status':'PASS_VM1_IDENTITY_FOUR_LATENTS_AND_VIDEO_BYTES',
        'reference_sha256':digest(args.reference),'candidate_sha256':digest(args.candidate),
        'source_sha256':digest(Path(__file__)),'latent_hashes':hashes,'all160_identity_controls':True,
        'video_sha256':digest(candidate['video']['path']),'quality_or_speed_acceptance':False})


if __name__=='__main__':main()
