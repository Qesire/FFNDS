"""Strict integer-view identity audit before R11G1 autograd, including signed zero."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from ar2_checkpoint import atomic_json, digest
from native_head_replay import PARAMETERS, patchify_flow, replay, unpatchify_flow
from reduced_rank_residual import predict_y
from run_ar2_restart import tensor_sha


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    output = args.run / 'STRICT_BYTES.json'
    if output.exists():
        raise RuntimeError('Preserve previous exact identity audit')
    capture_path = args.run / 'capture/RUN.json'
    control = json.loads(capture_path.read_text())
    if control['status'] != 'PASS_FULL_NATIVE_HEAD_CAPTURE' or digest(control['payload_path']) != control['payload_sha256']:
        raise RuntimeError('Complete unmodified native capture required')
    data = load_file(control['payload_path'], device='cuda')
    p = {k: data['p_' + k].float() for k in PARAMETERS}
    with torch.inference_mode():
        actual = replay(data, p)
        pairs = {'residual': (predict_y(data['u'], p), data['correction']),
                 'gate_writeback': (data['base'] + data['native_y'] * data['gate'], data['native_post']),
                 'complete_tokens': (actual, data['native_tokens']),
                 'complete_flow': (unpatchify_flow(actual, (21, 29, 52)).float(), data['native_flow']),
                 'permutation_inverse': (unpatchify_flow(patchify_flow(data['teacher_flow'], (21, 29, 52)), (21, 29, 52)), data['teacher_flow'])}
        checks = {k: {'equal_integer_bytes': same_bytes(a, b), 'replay_sha256': tensor_sha(a), 'native_sha256': tensor_sha(b)} for k, (a, b) in pairs.items()}
    passed = all(v['equal_integer_bytes'] and v['replay_sha256'] == v['native_sha256'] for v in checks.values())
    result = {'status': 'PASS_COMPLETE_NATIVE_INTEGER_BYTES' if passed else 'FAILED_NATIVE_INTEGER_BYTES_PRESERVED',
              'source_sha256': digest(Path(__file__)), 'capture_sha256': digest(capture_path),
              'protocol_sha256': control['protocol_sha256'], 'checks': checks, 'gradient_updates': 0,
              'purpose': 'Strengthen float equality with strict integer-byte and SHA controls, before gradient',
              'quality_or_speed_acceptance': False}
    atomic_json(output, result)
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('Exact native byte checks failed; do not proceed to gradient')


if __name__ == '__main__':
    main()
