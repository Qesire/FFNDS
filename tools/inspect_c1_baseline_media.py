"""Decode-count C1 baseline media and reuse available old trajectory controls."""
import os
from pathlib import Path

from c1_contract import ROOT, atomic, read, sha


def main():
    import imageio_ffmpeg
    hf = Path(os.environ['HF_HOME']); root = hf / 'derived/ffnds/C1N0_native_information'
    run = read(root / 'RUN.json')
    if run['status'] != 'COMPLETE_NATIVE_C1_FIXED_SCOPE':
        raise RuntimeError('Complete C1 native result required')
    rows = []
    previous = read(ROOT / 'research/20261023/R17E0_partial_generations.json')
    for r in run['requests']:
        video = hf / r['video_HF_relative']
        if sha(video) != r['video_sha256']:
            raise RuntimeError('C1 video bytes changed')
        decoder = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24', output_params=['-threads', '1'])
        meta = next(decoder); count = 0
        for frame in decoder:
            if len(frame) != 832 * 464 * 3:
                raise RuntimeError('Unexpected decoded frame geometry')
            count += 1
        result = {'width': meta['size'][0], 'height': meta['size'][1], 'nb_read_frames': count,
                  'fps': meta['fps'], 'duration': meta['duration']}
        if (result['width'], result['height'], int(result['nb_read_frames'])) != (832, 464, 81):
            raise RuntimeError('Incomplete baseline video decode')
        row = {'name': r['name'], 'source_index': r['source_index'], 'video_sha256': r['video_sha256'],
               'decoded_video_stream': result, 'candidate_or_human_quality_verdict': False}
        if r['source_index'] in (0, 2):
            prior = next(x for x in previous['completed'] if x['job']['name'] == f'quality{r["source_index"]:02d}_UB1')
            path = hf / prior['request_HF_relative']
            if sha(path) != prior['request_sha256']:
                raise RuntimeError('Original R17 subject receipt changed')
            old = read(path)
            expected = [{'step': s['step'], 'sha256': s['sha256']} for s in old['trajectory']]
            if r['trajectory'] != expected or r['initial_noise_sha256'] != old['initial_noise_sha256']:
                raise RuntimeError('C1 diagnostic baseline differs from saved original R17 subject trajectory')
            row['original_R17_UB1_four_latent_and_initial_noise_identity'] = True
            row['original_R17_request_sha256'] = prior['request_sha256']
        rows.append(row)
    atomic(root / 'BASELINE_MEDIA.json', {'status': 'ALL8_BASELINES_DECODE81FRAMES_AND_TWO_REUSED_SUBJECT_IDENTITIES',
           'records': rows, 'new_model_calls': 0, 'candidate_video_smoke_or_official_quality_result': False,
           'source_sha256': sha(__file__)})
    print('All8 baseline videos decoded81frames; source0/2 diagnostic trajectories match saved R17 UB1')


if __name__ == '__main__':
    main()
