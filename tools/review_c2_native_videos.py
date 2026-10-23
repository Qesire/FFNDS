"""Decode the authorized C2 videos and prepare full-duration review media.

CPU only; no learned evaluator and no automatic human/quality verdict.
"""
from itertools import zip_longest
import math
import os
from pathlib import Path
import subprocess

from c2_contract import ARMS, CAPS, METHOD, atomic, read, sha


def pixel_summary(sse, sae, frame_psnr, channels):
    frames = len(frame_psnr); mse = sse / (frames * channels)
    return {'pooled_RGB_MSE_0_255': mse, 'pooled_RGB_MAE_0_255': sae / (frames * channels),
            'pooled_MSE_PSNR_dB': None if sse == 0 else 10 * math.log10(255 ** 2 / mse),
            'mean_frame_PSNR_dB': None if any(v is None for v in frame_psnr) else math.fsum(frame_psnr) / frames,
            'exact_match_frames': sum(v is None for v in frame_psnr)}


def main(*, root_relative='derived/ffnds/C2N0_norm_scaled_smoke',
         complete_status='COMPLETE_C2_NATIVE_VIDEOS_PENDING_FULL_REVIEW', method_id=METHOD):
    from worker_identity import require_worker
    require_worker()
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    import numpy as np
    hf = Path(os.environ['HF_HOME']); root = hf / root_relative
    run = read(root / 'RUN.json')
    if (run['status'] != complete_status or run['budget']['used'] != CAPS or run.get('id', method_id) != method_id
            or [r['name'] for r in run['requests']] != ARMS):
        raise RuntimeError('Require complete authorized C2 scope')
    out = root / 'review'; out.mkdir(exist_ok=False)
    readers, metadata, videos = {}, {}, {}
    for request in run['requests']:
        name = request['name']; video = hf / request['video_HF_relative']
        if sha(video) != request['video_sha256']: raise RuntimeError('Native video bytes changed')
        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24', output_params=['-threads', '1'])
        meta = next(reader)
        if meta['size'] != (832, 464) or meta['fps'] != 16: raise RuntimeError('Unexpected native media geometry')
        readers[name], metadata[name], videos[name] = reader, meta, video
    width, height, channels, fps = 832, 464, 832 * 464 * 3, 16
    fresh = ARMS[1:]; totals = {a: [0, 0, []] for a in fresh}
    frames, pages, pair_pages = [], [], []
    all_canvas, pair_canvas = None, None
    for frame, decoded in enumerate(zip_longest(*(readers[a] for a in ARMS))):
        if any(v is None for v in decoded): raise RuntimeError('Video lengths differ')
        arrays = {a: np.frombuffer(v, dtype=np.uint8).reshape(height, width, 3) for a, v in zip(ARMS, decoded)}
        if frame % 12 == 0:
            if all_canvas is not None:
                target = out / f'all_arms_{len(pages):02d}.jpg'; all_canvas.save(target, quality=93)
                pages.append({'HF_relative': str(target.relative_to(hf)), 'sha256': sha(target)})
            all_canvas = Image.new('RGB', (5 * 240, 12 * (134 + 26)), 'white')
        if frame % 8 == 0:
            if pair_canvas is not None:
                target = out / f'dense_candidate_{len(pair_pages):02d}.jpg'; pair_canvas.save(target, quality=94)
                pair_pages.append({'HF_relative': str(target.relative_to(hf)), 'sha256': sha(target)})
            pair_canvas = Image.new('RGB', (2 * 832, 4 * (232 + 24)), 'white')
        y = (frame % 12) * 160; observations = {}
        for col, arm in enumerate(fresh):
            array = arrays[arm]
            ImageDraw.Draw(all_canvas).text((col * 240 + 3, y + 2), f'{arm} f{frame:02d} {frame/fps:.3f}s', fill='black')
            all_canvas.paste(Image.fromarray(array).resize((240, 134)), (col * 240, y + 24))
            difference = array.astype(np.int32) - arrays['dense_DQ'].astype(np.int32)
            sse = int(np.sum(difference.astype(np.int64) ** 2)); sae = int(np.sum(np.abs(difference), dtype=np.int64))
            psnr = None if not sse else 10 * math.log10(255 ** 2 / (sse / channels))
            totals[arm][0] += sse; totals[arm][1] += sae; totals[arm][2].append(psnr)
            observations[arm] = {'RGB_SSE': sse, 'RGB_SAE': sae, 'PSNR_dB': psnr}
        px, py = (frame % 2) * 832, ((frame % 8) // 2) * 256
        ImageDraw.Draw(pair_canvas).text((px + 4, py + 3), f'f{frame:02d} {frame/fps:.3f}s | dense left | C2 right', fill='black')
        for col, arm in enumerate(('dense_DQ', 'norm_scaled_energy')):
            pair_canvas.paste(Image.fromarray(arrays[arm]).resize((416, 232)), (px + col * 416, py + 24))
        frames.append({'frame': frame, 'time_seconds': frame / fps, 'observations': observations})
    if len(frames) != 81: raise RuntimeError('All81frames required')
    for canvas, collection, prefix in [(all_canvas, pages, 'all_arms'), (pair_canvas, pair_pages, 'dense_candidate')]:
        target = out / f'{prefix}_{len(collection):02d}.jpg'; canvas.save(target, quality=94)
        collection.append({'HF_relative': str(target.relative_to(hf)), 'sha256': sha(target)})
    preview = out / 'dense_left_C2_right.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-n',
                    '-threads', '1', '-i', str(videos['dense_DQ']), '-threads', '1', '-i', str(videos['norm_scaled_energy']),
                    '-filter_complex_threads', '1', '-filter_complex', 'hstack=inputs=2', '-an',
                    '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p', str(preview)], check=True)
    routing, bits = {}, {}
    for arm in ('fixed_stride', 'dynamic_residual', 'norm_scaled_energy'):
        request = next(r for r in run['requests'] if r['name'] == arm); path = root / arm / 'ROUTING.npz'
        if sha(path) != request['routing_sha256']: raise RuntimeError('Native routing bytes changed')
        with np.load(path, allow_pickle=False) as data: selected = np.unpackbits(data['packed_dense_rows'], axis=1)[:, :31668]
        if selected.shape != (160, 31668) or not np.all(selected.sum(1) == 7917): raise RuntimeError('Matched physical row budget violated')
        bits[arm] = selected
        routing[arm] = {'sites': 160, 'dense_rows_per_site': 7917,
                        'conditioned_t0_dense_fraction': float(selected[:, :1508].mean()),
                        'later20_dense_fraction': float(selected[:, 1508:].mean())}
    overlap = (bits['dynamic_residual'] & bits['norm_scaled_energy']).sum(1) / 7917
    result = {'id': method_id, 'status': 'COMPLETE_C2_CPU_MEDIA_PREPARATION_PENDING_FULL_REVIEW',
              'raw_run_sha256': sha(root / 'RUN.json'), 'source_sha256': sha(__file__),
              'videos_decoded': 6, 'frames_per_video': 81, 'total_decoded_frames': 486,
              'fps': fps, 'size': [width, height], 'duration_seconds': 81 / fps,
              'source_cluster_count': 1, 'source_index': 6, 'per_frame': frames,
              'RGB_fidelity_to_dense': {a: pixel_summary(*totals[a], channels) for a in fresh},
              'all_frame_all_arm_pages': pages, 'all_frame_dense_candidate_pages': pair_pages,
              'paired_preview': {'HF_relative': str(preview.relative_to(hf)), 'sha256': sha(preview)},
              'actual_native_routing': routing, 'old_new_dense_selection_overlap_fraction': float(overlap.mean()),
              'routing_scope': 'Actual free-policy trajectories differ. Mask overlap does not isolate a same-prefix causal effect.',
              'automatic_visual_PASS': False, 'human_full_video_review_completed': False,
              'official_quality_or_repeated_system_speed_result': False, 'new_model_calls': 0,
              'fidelity_scope': 'Decoded RGB fidelity in0..255 onone new source, not perceptual quality. Frames are not independent source groups. Exact-frame PSNR is mathematical infinity encoded null; mean-framePSNR null if any exact frame.'}
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if used > 1 << 29: raise RuntimeError('Combined C2 media/model output cap exceeded')
    result['HF_output_bytes_before_review_receipt'] = used
    atomic(root / 'VIDEO_REVIEW.json', result)
    print({'status': result['status'], 'RGB_fidelity': result['RGB_fidelity_to_dense'], 'routing': routing}, flush=True)


if __name__ == '__main__': main()
