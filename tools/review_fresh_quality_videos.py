"""Decode all115 genuine fresh videos on CPU; no learned quality model."""
import json
import math
import os
from pathlib import Path
import subprocess

from fresh_quality_contract import ARMS, METHOD, ROOT, atomic, read, sha


def probe(path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height,avg_frame_rate,codec_name,duration',
               '-of', 'json', str(path)]
    streams = json.loads(subprocess.check_output(command))['streams']
    if len(streams) != 1:
        raise RuntimeError('One actual video stream required')
    row = streams[0]
    numerator, denominator = map(int, row['avg_frame_rate'].split('/'))
    if numerator / denominator != 16:
        raise RuntimeError('Frozen16fps generation changed')
    return {'width': int(row['width']), 'height': int(row['height']),
            'fps_numerator': numerator, 'fps_denominator': denominator,
            'codec': row['codec_name']}


def source_decode(requests, hf, error_folder):
    import hashlib
    import numpy as np
    paths = {arm: hf / requests[arm]['video_HF_relative'] for arm in ARMS}
    metadata = {arm: probe(path) for arm, path in paths.items()}
    grid = requests['dense_DQ']['grid']
    height, width = grid[1] * 16, grid[2] * 16
    if grid[0] != 21 or any(r['grid'] != grid for r in requests.values()):
        raise RuntimeError('Paired native temporal/spatial geometry differs')
    if any((m['height'], m['width']) != (height, width) for m in metadata.values()):
        raise RuntimeError('Encoded video does not match actual native grid')
    for arm, path in paths.items():
        if sha(path) != requests[arm]['video_sha256']:
            raise RuntimeError('Generated video bytes changed')
    processes, logs = {}, []
    hashes = {arm: hashlib.sha256() for arm in ARMS}
    errors = {arm: [] for arm in ARMS[1:]}
    frame_bytes = height * width * 3
    try:
        for arm, path in paths.items():
            log = (error_folder / (arm + '.log')).open('wb')
            logs.append(log)
            processes[arm] = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '1',
                '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=log)
        for frame in range(81):
            decoded = {}
            for arm, process in processes.items():
                data = process.stdout.read(frame_bytes)
                if len(data) != frame_bytes:
                    raise RuntimeError('Truncated actual full-duration video at frame' + str(frame))
                hashes[arm].update(data)
                decoded[arm] = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
            for arm in ARMS[1:]:
                difference = decoded[arm] - decoded['dense_DQ']
                errors[arm].append(float(np.sum(difference * difference, dtype=np.float64) / len(difference)))
        for process in processes.values():
            if process.stdout.read(1) or process.wait(timeout=30):
                raise RuntimeError('Extra frames or decoder failure')
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
        for log in logs:
            log.close()
    fidelity = {}
    for arm, values in errors.items():
        pooled = math.fsum(values) / 81
        psnr = [None if x == 0 else 10 * math.log10(255 ** 2 / x) for x in values]
        fidelity[arm] = {'frame_MSE_RGB_0_255': values, 'pooled_MSE_RGB_0_255': pooled,
                         'pooled_PSNR_dB': None if pooled == 0 else 10 * math.log10(255 ** 2 / pooled),
                         'mean_frame_PSNR_dB': None if any(p is None for p in psnr) else math.fsum(psnr) / 81,
                         'perfect_match_frame_count': sum(p is None for p in psnr)}
    videos = {arm: {**metadata[arm], 'decoded_frames': 81,
                    'video_HF_relative': requests[arm]['video_HF_relative'],
                    'video_sha256': requests[arm]['video_sha256'],
                    'decoded_all_RGB_bytes_sha256': hashes[arm].hexdigest()} for arm in ARMS}
    return {'videos': videos, 'paired_pixel_fidelity': fidelity}


def main():
    from worker_identity import require_worker
    require_worker()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Full video decode is CPU only; hide all GPUs')
    hf = Path(os.environ['HF_HOME'])
    root = hf / 'derived/ffnds/R17E0_fresh_quality'
    output = root / 'VIDEO_DECODE.json'
    if output.exists():
        raise RuntimeError('Preserve the first full decode receipt')
    generations_path = root / 'GENERATIONS.json'
    generations = read(generations_path)
    if generations['status'] != 'COMPLETE_145_GENERATIONS_AWAITING_FRESH_OFFICIAL_METRICS':
        raise RuntimeError('Require the complete actual fixed145generation batch')
    requests = {}
    for item in generations['completed']:
        if item['job']['phase'] != 'quality':
            continue
        path = hf / item['request_HF_relative']
        if sha(path) != item['request_sha256']:
            raise RuntimeError('Actual model request receipt changed')
        record = read(path)
        key = record['job']['source'], record['job']['arm']
        if key in requests:
            raise RuntimeError('Duplicate actual quality video identity')
        requests[key] = record
    if set(requests) != {(s, a) for s in range(23) for a in ARMS}:
        raise RuntimeError('Exactly115unique fresh-source/arm videos required')
    report = {'id': METHOD, 'status': 'CPU_DECODING_ALL_FRESH_VIDEOS',
              'generations_sha256': sha(generations_path), 'source_sha256': sha(__file__),
              'new_model_generations': 0, 'learned_quality_model_forwards': 0,
              'full_duration_human_or_agent_visual_review': False,
              'official_perceptual_quality_or_smoke_pass': False, 'sources': []}
    atomic(output, report)
    try:
        for source in range(23):
            folder = root / 'decode_logs' / f'source{source:02d}'
            folder.mkdir(parents=True)
            result = source_decode({a: requests[source, a] for a in ARMS}, hf, folder)
            report['sources'].append({'source': source, **result})
            atomic(output, report)
            print('Fully decoded fresh source', source + 1, '/23; five actual81-frame videos', flush=True)
        report['status'] = 'COMPLETE_ALL115_FRESH_VIDEOS_ALL81_FRAMES_DECODED_NO_VISUAL_VERDICT'
    except BaseException as error:
        report.update(status='FAILED_FULL_FRESH_VIDEO_DECODE', error_type=type(error).__name__)
        raise
    finally:
        atomic(output, report)
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
