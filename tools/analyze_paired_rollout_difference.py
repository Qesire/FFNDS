"""CPU-only, post-hoc RGB fidelity of the already completed R15V0 video pair.

Paired fidelity measures trajectory differences from D-Q. It is not a measure
of official video quality, perceptual equivalence or population uncertainty.
"""
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path

from archive_public_json import archive
from rollout_contract import ROOT, read, sha


def main():
    from worker_identity import require_worker
    require_worker()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Decoded-video analysis is CPU only')
    import imageio_ffmpeg
    import numpy as np
    hf = Path(os.environ['HF_HOME'])
    root = hf / 'derived/ffnds/R15V0_paired_rollout'
    run = read(root / 'RUN.json')
    if run['status'] != 'COMPLETE_PAIRED_FULL_VIDEO_AWAITING_DECODE_AND_REVIEW':
        raise RuntimeError('Two completed original videos required')
    readers, metadata, decoded_hash = [], [], [hashlib.sha256(), hashlib.sha256()]
    for request in run['requests']:
        path = hf / request['video_HF_relative']
        if sha(path) != request['video_sha256']:
            raise RuntimeError('Original paired video changed')
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        readers.append(reader)
        metadata.append(next(reader))
    if (len(readers) != 2 or metadata[0]['size'] != metadata[1]['size']
            or metadata[0]['fps'] != metadata[1]['fps']):
        raise RuntimeError('Actual paired geometry/FPS differs')
    width, height = metadata[0]['size']
    channels_per_frame = width * height * 3
    rows = []
    for index, pair in enumerate(zip_longest(*readers)):
        if any(frame is None for frame in pair):
            raise RuntimeError('Paired decode lengths differ')
        arrays = []
        for arm, frame in enumerate(pair):
            decoded_hash[arm].update(frame)
            arrays.append(np.frombuffer(frame, dtype=np.uint8).astype(np.int32))
        difference = arrays[1] - arrays[0]
        sse = int(np.square(difference).sum(dtype=np.int64))
        absolute_sum = int(np.abs(difference).sum(dtype=np.int64))
        mse = sse / channels_per_frame
        rows.append({'frame': index, 'time_seconds': index / metadata[0]['fps'],
                     'RGB_channel_count': channels_per_frame, 'RGB_SSE': sse,
                     'RGB_absolute_sum': absolute_sum,
                     'different_RGB_channels': int(np.count_nonzero(difference)),
                     'max_absolute_RGB_difference': int(np.abs(difference).max()),
                     'RGB_MSE_0_255': mse, 'exact_match': sse == 0,
                     'PSNR_dB': None if sse == 0 else 10 * math.log10(255 ** 2 / mse)})
    if len(rows) != 81:
        raise RuntimeError('Complete original81-frame pair required')
    total = channels_per_frame * len(rows)
    pooled_mse = sum(r['RGB_SSE'] for r in rows) / total
    finite = [r['PSNR_dB'] for r in rows if not r['exact_match']]
    a, b = run['requests']
    report = {'id': 'R15V0_ACTUAL_DECODED_PAIR_RGB_FIDELITY',
              'status': 'POSTHOC_DESCRIPTIVE_ACTUAL_VIDEO_FIDELITY',
              'source_sha256': sha(Path(__file__)), 'raw_run_sha256': sha(root / 'RUN.json'),
              'input_video_sha256': [r['video_sha256'] for r in run['requests']],
              'decoded_RGB_stream_sha256': [h.hexdigest() for h in decoded_hash],
              'size': [width, height], 'fps': metadata[0]['fps'], 'frames': len(rows),
              'protocol': {'color_space': 'ffmpeg RGB24 decoded display pixels',
                           'data_range': 255, 'alignment': 'same source/prompt/seed/initial noise/frame index; no warp',
                           'sample_unit': 'one old exposed source0/seed42 video pair',
                           'aggregation': 'all RGB channels equally weighted; separate pooled-MSE PSNR and mean finite per-frame PSNR',
                           'exact_match_PSNR': 'positive infinity; JSON null plus exact_match=true',
                           'exposure': 'post-hoc after complete frame-sequence visual review; no threshold/success gate'},
              'matched_initial_noise': a['initial_noise_sha256'] == b['initial_noise_sha256'],
              'free_latent_SHA_differs_at_steps': [x['raw_tensor_sha256'] != y['raw_tensor_sha256']
                                                  for x, y in zip(a['trajectory'], b['trajectory'])],
              'per_frame': rows, 'pooled_RGB_SSE': sum(r['RGB_SSE'] for r in rows),
              'pooled_RGB_MSE_0_255': pooled_mse,
              'pooled_RGB_MAE_0_255': sum(r['RGB_absolute_sum'] for r in rows) / total,
              'pooled_MSE_PSNR_dB': None if pooled_mse == 0 else 10 * math.log10(255 ** 2 / pooled_mse),
              'mean_finite_per_frame_PSNR_dB': math.fsum(finite) / len(finite) if finite else None,
              'exact_match_frames': sum(r['exact_match'] for r in rows),
              'worst_MSE_frame': max(rows, key=lambda r: r['RGB_SSE'])['frame'],
              'new_model_generations': 0, 'new_learned_evaluator_calls': 0,
              'official_quality_or_native_speed_claim': False,
              'limits': 'One old-source four-site integration. RGB fidelity mixes texture and legitimate motion differences; no perceptual quality, independent-source or confidence-interval claim.'}
    path = root / 'DECODED_PAIR_DIFFERENCE.json'
    with path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    archive(path, ROOT / 'research/20261023/R15V0_decoded_pair_difference.json')
    print(json.dumps({k: report[k] for k in ('frames', 'pooled_RGB_MSE_0_255', 'pooled_RGB_MAE_0_255',
                     'pooled_MSE_PSNR_dB', 'mean_finite_per_frame_PSNR_dB', 'exact_match_frames', 'worst_MSE_frame')}), flush=True)


if __name__ == '__main__':
    main()
