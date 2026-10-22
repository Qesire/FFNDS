"""CPU-only complete five-video decode, paired fidelity and review media.

This does not load learned quality networks or grant a human smoke verdict.
"""
import argparse
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import subprocess

from rollout_contract import atomic, read, sha


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from worker_identity import require_worker
    require_worker()
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    import numpy as np
    hf = Path(os.environ['HF_HOME'])
    root = hf / 'derived/ffnds/R16V0_full160_row_dispatch'
    run = read(root / 'RUN.json')
    if run['status'] != 'COMPLETE_FULL160_GENERATION_AND_NATIVE_MICROBENCHMARK_AWAITING_VIDEO_REVIEW':
        raise RuntimeError('Require all five completed model videos before decode')
    out = root / 'review'
    out.mkdir(exist_ok=False)
    readers, metadata, videos = [], [], []
    for request in run['requests']:
        video = hf / request['video_HF_relative']
        if sha(video) != request['video_sha256']:
            raise RuntimeError('Generated video bytes changed')
        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24', output_params=['-threads', '1'])
        readers.append(reader)
        metadata.append(next(reader))
        videos.append(video)
    if any(m['size'] != (832, 464) or m['fps'] != 16 for m in metadata):
        raise RuntimeError('Unexpected actual paired dimensions or frame rate')
    width, height, fps = 832, 464, 16
    arms = [r['arm'] for r in run['requests']]
    channel_count = width * height * 3
    pages, per_frame, canvas = [], [], None
    thumb_w, thumb_h = 240, 134
    # This upper strip is a descriptive fixed image-coordinate diagnostic.
    # No learned motion estimator or candidate-specific mask is used. Camera
    # and geometry changes may affect it; it cannot certify static-ROI flicker.
    upper_y1 = 92
    previous_upper = [None] * 5
    frame_count = 0
    for frame, values in enumerate(zip_longest(*readers)):
        if any(v is None for v in values):
            raise RuntimeError('Paired decoded frame counts differ')
        arrays = [np.frombuffer(v, dtype=np.uint8).reshape(height, width, 3) for v in values]
        if frame % 12 == 0:
            if canvas is not None:
                target = out / f'all_arms_frames_{len(pages):02d}.jpg'
                canvas.save(target, quality=93)
                pages.append({'HF_relative': str(target.relative_to(hf)), 'sha256': sha(target)})
            canvas = Image.new('RGB', (5 * thumb_w, 12 * (thumb_h + 27)), 'white')
        y = (frame % 12) * (thumb_h + 27)
        draw = ImageDraw.Draw(canvas)
        observations = {}
        for col, (arm, array) in enumerate(zip(arms, arrays)):
            draw.text((col * thumb_w + 3, y + 2), f'{arm} f{frame:02d} {frame/fps:.3f}s', fill='black')
            canvas.paste(Image.fromarray(array).resize((thumb_w, thumb_h)), (col * thumb_w, y + 25))
            delta = array.astype(np.int32) - arrays[0].astype(np.int32)
            sse = int(np.sum(delta.astype(np.int64) ** 2))
            sae = int(np.sum(np.abs(delta), dtype=np.int64))
            mse = sse / channel_count
            upper = (.299 * array[:upper_y1, :, 0] + .587 * array[:upper_y1, :, 1] + .114 * array[:upper_y1, :, 2])
            upper_change = None if previous_upper[col] is None else float(np.abs(upper - previous_upper[col]).mean())
            previous_upper[col] = upper
            observations[arm] = {'RGB_SSE': sse, 'RGB_SAE': sae,
                                  'PSNR_dB': None if not sse else 10 * math.log10(255 ** 2 / mse),
                                  'upper_strip_luma_mean': float(upper.mean()),
                                  'upper_strip_pixel_temporal_MAE': upper_change}
        per_frame.append({'frame': frame, 'time_seconds': frame / fps, 'RGB_channel_count': channel_count,
                          'observations': observations})
        frame_count += 1
    if canvas is not None:
        target = out / f'all_arms_frames_{len(pages):02d}.jpg'
        canvas.save(target, quality=93)
        pages.append({'HF_relative': str(target.relative_to(hf)), 'sha256': sha(target)})
    if frame_count != 81:
        raise RuntimeError('Require all81 decoded frames per video')
    aggregates = {}
    for arm in arms:
        sse = sum(row['observations'][arm]['RGB_SSE'] for row in per_frame)
        sae = sum(row['observations'][arm]['RGB_SAE'] for row in per_frame)
        mse = sse / (frame_count * channel_count)
        frame_psnr = [row['observations'][arm]['PSNR_dB'] for row in per_frame]
        aggregates[arm] = {'pooled_RGB_MSE_0_255': mse, 'pooled_RGB_MAE_0_255': sae / (frame_count * channel_count),
                           'pooled_MSE_PSNR_dB': None if not sse else 10 * math.log10(255 ** 2 / mse),
                           'mean_frame_PSNR_dB': None if any(v is None for v in frame_psnr) else math.fsum(frame_psnr) / frame_count,
                           'exact_match_frames': sum(row['observations'][arm]['RGB_SSE'] == 0 for row in per_frame)}
    preview = out / 'paired_dense_left_dynamic_right.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-i', str(videos[0]),
                    '-i', str(videos[-1]), '-filter_complex', 'hstack=inputs=2', '-an', '-c:v', 'libx264',
                    '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p', str(preview)], check=True)
    result = {'id': 'R16V0_COMPLETE_FIVE_VIDEO_CPU_DECODE', 'status': 'COMPLETE_DECODE_AWAITING_VISUAL_REVIEW',
              'raw_run_sha256': sha(root / 'RUN.json'), 'source_sha256': sha(__file__),
              'decoded_every_frame': True, 'frames_per_video': frame_count, 'size': [width, height],
              'fps': fps, 'duration_seconds': frame_count / fps, 'arms': arms,
              'input_video_sha256': [r['video_sha256'] for r in run['requests']],
              'all_frame_contact_pages': pages,
              'paired_preview': {'HF_relative': str(preview.relative_to(hf)), 'sha256': sha(preview)},
              'RGB_fidelity_to_dense': aggregates, 'per_frame': per_frame,
              'upper_strip': {'pixels': [0, 0, width, upper_y1], 'selection': 'fixed image-coordinate upper strip shared by all arms',
                              'motion_aligned': False, 'certifies_static_ROI_flicker': False},
              'automatic_visual_PASS': False, 'human_full_video_review_completed': False,
              'official_quality_or_native_speed_claim': False, 'new_model_generations': 0,
              'fidelity_interpretation': 'Decoded RGB trajectory fidelity includes texture and plausible motion divergence, not video-quality deterioration. One old-source cluster; frames are not independent sources.'}
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if used > 1 << 29:
        raise RuntimeError('Combined model/review HF output cap exceeded')
    result['HF_output_bytes_before_review_receipt'] = used
    atomic(root / 'VIDEO_REVIEW.json', result)
    print(json.dumps({'status': result['status'], 'frames_per_video': frame_count,
                      'contact_pages': len(pages), 'fidelity': aggregates}), flush=True)


if __name__ == '__main__':
    main()
