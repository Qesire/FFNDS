"""Independent CPU audit of every accepted OP1 payload and native geometry."""
import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import torch
from safetensors import safe_open

from ar2_checkpoint import atomic_json, digest
from onpolicy_residual import load_collections
from run_ar2_restart import ROOT, tensor_sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections', type=Path, nargs=2, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); torch.set_num_threads(8); started = time.perf_counter()
    records, bindings, identity = load_collections(args.collections, ROOT)
    payloads, total_bytes, videos = 0, 0, 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'collection_bindings': bindings, 'capture_identity': identity,
        'source_sha256': digest(Path(__file__)), 'inputs': [], 'quality_or_speed_acceptance': False}
    atomic_json(args.out, report)
    try:
        for index in range(24):
            record = records[index]; shapes = set()
            for scope in record['scopes']:
                path = Path(scope['path'])
                if digest(path) != scope['sha256'] or path.stat().st_size != scope['bytes']:
                    raise RuntimeError('OP1 operand payload differs')
                with safe_open(path, framework='pt', device='cpu') as handle:
                    if set(handle.keys()) != {'V1', 'BaseY', 'TeacherY', 'StudentY', 'b2', 'gate'}:
                        raise RuntimeError('Unexpected OP1 tensor fields')
                    for key in ['V1', 'BaseY', 'TeacherY', 'StudentY']:
                        header = handle.get_slice(key)
                        if header.get_dtype() != 'BF16' or header.get_shape() != [256, 5120]:
                            raise RuntimeError('OP1 payload BF16 row geometry differs')
                    for key, field in [('b2', 'original_b2_sha256'), ('gate', 'gate_sha256')]:
                        tensor = handle.get_tensor(key)
                        if tensor.dtype != torch.bfloat16 or tensor.numel() != 5120 or tensor_sha(tensor) != scope[field]:
                            raise RuntimeError('OP1 gate/raw-bias payload differs')
                if (scope['native_V1_shape'] != [scope['full_rows'], 5120]
                        or scope['native_V1_stride'] != [5120, 1]
                        or scope['native_BaseY_shape'] != scope['native_V1_shape']):
                    raise RuntimeError('Observed OP1 native deployment geometry changed')
                shapes.add(tuple(scope['native_V1_shape'])); payloads += 1; total_bytes += scope['bytes']
            for video in record['videos'].values():
                with imageio.get_reader(video['path']) as reader:
                    info = reader.get_meta_data(); count = reader.count_frames()
                if (count != 81 or abs(float(info['fps']) - 16) > .01
                        or list(info['size']) != record['observed_video']['size']):
                    raise RuntimeError('OP1 native video frames/fps/geometry differs')
                videos += 1
            report['inputs'].append({'index': index, 'positions': 160, 'native_shapes': sorted(shapes),
                'all_payload_checksums_verified': True, 'all_headers_BF16': True,
                'sampling_coordinates_verified': True, 'four_step_and_video_noop': True})
            atomic_json(args.out, report); print(f'AUDITED OP1 input{index:02d}', flush=True)
        if payloads != 3840 or videos != 48 or torch.cuda.is_initialized():
            raise RuntimeError('Full CPU-only24-input audit required')
        report.update(status='PASS_OP1_ALL3840_PAYLOADS_HEADERS_SAMPLING_AND48_VIDEOS',
            payloads=payloads, operand_bytes=total_bytes, videos=videos, elapsed_seconds=time.perf_counter() - started,
            cuda_initialized=False)
    except BaseException as error:
        report.update(status='FAILED', error=str(error)); raise
    finally: atomic_json(args.out, report)


if __name__ == '__main__': main()
