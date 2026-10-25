"""Audit six frozen native videos; decode every frame without learned scoring."""
import argparse
import json
from pathlib import Path

from analyze_e0_dense_residency import validate_plan
from analyze_e0_function_reconstruction import finite
from ffnds_observation_common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def compare_pixels(paths, out, source):
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    captures = [cv2.VideoCapture(str(p)) for p in paths]
    fps = [c.get(cv2.CAP_PROP_FPS) for c in captures]
    if min(fps) <= 0 or max(fps) - min(fps) > 1e-6:
        raise ValueError('Inconsistent encoded frame rate')
    metrics = [[], []]
    previous = None
    contacts = []
    video = out / f'source{source:02d}_dense_whole94_affine54.mp4'
    writer = imageio.get_writer(str(video), fps=fps[0], codec='libx264', quality=7,
                               macro_block_size=2, ffmpeg_log_level='error')
    count = 0
    try:
        while True:
            reads = [c.read() for c in captures]
            if not any(ok for ok, _ in reads):
                break
            if not all(ok for ok, _ in reads):
                raise ValueError('Unequal decoded video lengths')
            frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in reads]
            if any(f.shape != frames[0].shape for f in frames):
                raise ValueError('Unequal decoded frame shapes')
            values = [f.astype(np.float64) / 255 for f in frames]
            luma = [v @ np.array([.2126, .7152, .0722]) for v in values]
            for i in range(2):
                difference = values[i + 1] - values[0]
                item = {'frame': count, 'RGB_MSE_0to1': float(np.mean(difference ** 2)),
                        'RGB_MAE_0to1': float(np.mean(np.abs(difference))),
                        'reference_mean_RGB_0to1': values[0].mean(axis=(0, 1)).tolist(),
                        'paired_signed_mean_RGB_drift_0to1': difference.mean(axis=(0, 1)).tolist(),
                        'encoded_luma_MSE_0to1': float(np.mean((luma[i + 1] - luma[0])**2)),
                        'encoded_luma_signed_mean_drift_0to1': float(np.mean(luma[i + 1] - luma[0])),
                        'candidate_mean_RGB_0to1': values[i + 1].mean(axis=(0, 1)).tolist()}
                if previous is not None:
                    reference_delta = values[0] - previous[0]
                    candidate_delta = values[i + 1] - previous[i + 1]
                    item.update(temporal_difference_MSE_0to1=float(np.mean((candidate_delta-reference_delta)**2)),
                                candidate_temporal_MAE_0to1=float(np.mean(np.abs(candidate_delta))),
                                reference_temporal_MAE_0to1=float(np.mean(np.abs(reference_delta))))
                metrics[i].append(item)
            previous = values
            thumbnails = [Image.fromarray(f).resize((416, 240)) for f in frames]
            canvas = Image.new('RGB', (1248, 266), 'white')
            draw = ImageDraw.Draw(canvas)
            for i, (label, image) in enumerate(zip(['Dense D-Q', 'Whole94 + recovery', 'Affine +54 + recovery'], thumbnails)):
                canvas.paste(image, (416*i, 26)); draw.text((416*i+8, 7), label, fill='black')
            writer.append_data(np.asarray(canvas))
            if count in [0, 20, 40, 60, 80]:
                contacts.append(canvas.copy())
            count += 1
    finally:
        writer.close()
        for c in captures:
            c.release()
    if count != 81 or len(contacts) != 5:
        raise ValueError('Expected all81 decoded frames')
    contact = Image.new('RGB', (1248, 266*len(contacts)), 'white')
    for i, canvas in enumerate(contacts):
        contact.paste(canvas, (0, 266*i))
    contact_path = out / f'source{source:02d}_contact.png'; contact.save(contact_path)
    result = []
    for mode, items in zip(['whole', 'affine_residual'], metrics):
        result.append({'mode': mode, 'all_frame_metrics': items,
                       'mean_frame_RGB_RMSE_0to1': float(np.mean([np.sqrt(m['RGB_MSE_0to1']) for m in items])),
                       'mean_temporal_difference_RMSE_0to1': float(np.mean([np.sqrt(m['temporal_difference_MSE_0to1']) for m in items[1:]]))})
    return {'source_index': source, 'decoded_frames_each': count, 'fps': fps[0],
            'original_frame_shape': list(frames[0].shape), 'paired_pixel_metrics': result,
            'comparison_video': video.name, 'comparison_video_sha256': sha256(video),
            'contact_sheet': contact_path.name, 'contact_sheet_sha256': sha256(contact_path),
            'interpretation': 'Encoded RGB and BT.709-weighted encoded-luma deviations (not linear-light luminance). Temporal difference is delta(candidate-dense), blind to constant offsets; none are perceptual/official quality scores or human full-video acceptance'}


def analyze(hf, receipts, media):
    relative = 'config/e0_reconstructed_videos_20261026.json'
    plan, digest = validate_plan(relative)
    refs = {v['index']: v for v in json.loads((ROOT / plan['accepted_observations']).read_text())['sources']}
    runs = []; videos = {}; bindings = []; intervals = []
    for mode in plan['modes']:
        path = hf / f'e0_function_basis_20261026/prototype_{mode}_attempt1/RUN.json'
        raw = json.loads(path.read_text())
        if (not finite(raw) or raw['status'] != 'PASS_FIXED_PROTOTYPE_VIDEO_EXECUTION_NOT_QUALITY_ACCEPTANCE'
                or raw['protocol_sha256'] != digest or raw['mode'] != mode
                or raw['bank_RUN_sha256'] != plan['bank_RUN_sha256']
                or raw['generations_started'] != 2 or len(raw['phases']) != 2):
            raise ValueError('Incomplete prototype execution')
        if raw['optimizer_updates'] or raw['new_weight_fits'] or raw['official_quality_calls']:
            raise ValueError('Unexpected model/parameter work')
        snapshot = path.parent / 'EXECUTED_SOURCES'
        if sha256(snapshot / Path(relative).name) != digest or any(
                sha256(snapshot / name) != expected for name, expected in plan['driver_sources_sha256'].items()):
            raise ValueError('Executed snapshot changed')
        resource = raw['assigned_GPU_resources']
        if not resource['sampled_isolation_pass'] or resource['violations'] or any(
                v['foreign_process_count_on_assigned_GPU'] for v in resource['samples']):
            raise ValueError('Assigned GPU isolation failed')
        receipt_path = receipts / f'prototype_{mode}_LAUNCH_PRIVATE.json'
        receipt = json.loads(receipt_path.read_text())
        if (receipt['protocol_sha256'] != digest or receipt['mode'] != mode
                or receipt['physical_GPU'] != resource['physical_GPU_index']):
            raise ValueError('Private launch receipt mismatch')
        bindings.append((receipt['slot'], receipt['physical_GPU']))
        intervals.append((min(s['monotonic_seconds'] for s in resource['samples']),
                          max(s['monotonic_seconds'] for s in resource['samples'])))
        controls = raw['neutral_controls']
        if mode == 'dense':
            if controls:
                raise ValueError('Unexpected dense neutral work')
        elif (len(controls) != 4 or {(c['step'], c['block']) for c in controls} != {(s,b) for s in [0,2] for b in [0,30]}
              or any(c['BF16_bit_mismatches'] or c['rows'] != 48 or c['extra_native_FFN_calls'] != 2 or c['extra_native_QDQ_calls'] != 4 for c in controls)):
            raise ValueError('Neutral control failed')
        phases = []
        for order, phase in enumerate(raw['phases']):
            source = plan['source_sequence'][order]; reference = refs[source]
            if phase['order'] != order or phase['source_index'] != source or phase['seed'] != reference['seed']:
                raise ValueError('Input sequence changed')
            if [v['step'] for v in phase['trajectory']] != [0,1,2,3]:
                raise ValueError('Incomplete free trajectory')
            native = phase['runtime_native_counts']; expected = 1600 if mode == 'dense' else 1280
            if native['linear_forward_calls'] != expected or native['activation_qdq_calls'] != expected or native['cache_misses'] or native['weight_transfers']:
                raise ValueError('Unexpected original native calls or generation transfers')
            hook = phase['prototype_native_counts']
            if hook != ({} if mode == 'dense' else {'up':160,'down':160,'Q1':160,'Q2':160,'affine':160 if mode == 'affine_residual' else 0}):
                raise ValueError('Unexpected prototype calls')
            events = phase['FFN_boundary_timings']
            if len(events) != 160 or {(e['step'],e['block']) for e in events} != {(s,b) for s in range(4) for b in range(40)}:
                raise ValueError('Incomplete FFN timers')
            if abs(sum(e['elapsed_ms'] for e in events) - phase['FFN_boundary_total_ms']) > 1e-6:
                raise ValueError('FFN timing sum changed')
            video = path.parent / phase['video']
            if sha256(video) != phase['video_sha256']:
                raise ValueError('Video artifact changed')
            if mode == 'dense' and (phase['trajectory'] != reference['trajectory'] or phase['video_sha256'] != reference['video_sha256']):
                raise ValueError('Dense historical identity failed')
            videos[mode, source] = video
            summary = {k:v for k,v in phase.items() if k != 'FFN_boundary_timings'}
            summary['generation_wall_minus_CUDA_FFN_boundary_seconds'] = (
                phase['wall_seconds_including_video_save_and_common_latent_hash'] - phase['FFN_boundary_total_ms']/1000)
            phases.append(summary)
        size = sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())
        if size > plan['max_HF_output_gib_per_mode'] * (1 << 30) or raw['final_residency']['retained_bytes']:
            raise ValueError('Output budget or release failed')
        runs.append({'mode': mode, 'raw_binding':{'HF_relative':str(path.relative_to(hf)), 'sha256':sha256(path)},
                     'launch_receipt_sha256':sha256(receipt_path), 'anonymous_card':'A',
                     'setup_seconds':raw['setup_seconds_including_assets_pipeline_packs_neutral_and_weight_preload'],
                     'original_weight_preload_seconds':raw['original_weight_preload_seconds'],
                     'fitted_pack_load_seconds':raw.get('fitted_pack_load_seconds',0),
                     'memory_precheck':raw['memory_precheck'],
                     'peak_physical_GPU_MiB':resource['peak_physical_used_memory_MiB'],
                     'resource_sampling_interval_seconds':resource['sampling_interval_seconds'],
                     'sampled_isolation_pass':True, 'HF_output_bytes':size,
                     'neutral_controls':controls, 'phases':phases})
    if len(set(bindings)) != 1:
        raise ValueError('Same-card comparison failed')
    if any(end >= following[0] for (_, end), following in zip(intervals, intervals[1:])):
        raise ValueError('Serial mode monitoring intervals overlap')
    neutral_ffn = sum(c['extra_native_FFN_calls'] for r in runs for c in r['neutral_controls'])
    neutral_qdq = sum(c['extra_native_QDQ_calls'] for r in runs for c in r['neutral_controls'])
    generation_qdq = sum(p['runtime_native_counts']['activation_qdq_calls'] +
                          p['prototype_native_counts'].get('Q1', 0) +
                          p['prototype_native_counts'].get('Q2', 0) for r in runs for p in r['phases'])
    gemms = neutral_ffn*2 + sum(p['runtime_native_counts']['linear_forward_calls'] +
                              p['prototype_native_counts'].get('up', 0) +
                              p['prototype_native_counts'].get('down', 0) +
                              p['prototype_native_counts'].get('affine', 0) for r in runs for p in r['phases'])
    if (neutral_ffn != plan['extra_native_FFN_calls'] or neutral_qdq != plan['extra_native_QDQ_calls']
            or generation_qdq != plan['full_generation_QDQ_calls']
            or generation_qdq+neutral_qdq != plan['total_QDQ_calls']
            or gemms != plan['total_linear_GEMMs_including_neutral']):
        raise ValueError('Summed cross-mode invocation budget mismatch')
    media.mkdir(parents=True, exist_ok=False)
    comparisons = [compare_pixels([videos[m,s] for m in plan['modes']],media,s) for s in plan['source_sequence']]
    media_size = sum(p.stat().st_size for p in media.rglob('*') if p.is_file())
    if media_size > .125*(1 << 30):
        raise ValueError('Derived media cap exceeded')
    paired = []
    for order, source in enumerate(plan['source_sequence']):
        dense = runs[0]['phases'][order]
        for run in runs[1:]:
            phase = run['phases'][order]
            paired.append({'source_index':source, 'mode':run['mode'],
                           'timing_state':'COMMON_WARM_SECOND_VIDEO' if order else 'FRESH_PROCESS_FIRST_VIDEO_KERNEL_STATES_NOT_EQUAL',
                           'generation_wall_ratio_dense_over_prototype':dense['wall_seconds_including_video_save_and_common_latent_hash']/phase['wall_seconds_including_video_save_and_common_latent_hash'],
                           'FFN_boundary_ratio_dense_over_prototype':dense['FFN_boundary_total_ms']/phase['FFN_boundary_total_ms']})
    warm = runs[0]['phases'][1]
    T = warm['wall_seconds_including_video_save_and_common_latent_hash']
    F = warm['FFN_boundary_total_ms']/1000
    return {'id':plan['id'], 'status':'COMPLETE_EXECUTION_NO_QUALITY_ACCEPTANCE',
            'protocol':relative,'protocol_sha256':digest,'analysis_sha256':sha256(Path(__file__)),
            'bank_manifest_sha256':plan['bank_manifest_sha256'], 'runs':runs,'paired_timing':paired,
            'videos':6,'historical_dense_video_hashes_matched':2,'historical_dense_latent_hashes_matched':8,
            'extra_neutral_FFN_calls':neutral_ffn,'extra_neutral_QDQ_calls':neutral_qdq,
            'total_QDQ_calls':generation_qdq+neutral_qdq,
            'total_linear_GEMMs_including_neutral':gemms,'planned_maximum_simultaneous_model_processes':1,
            'same_card_serial_monitoring_intervals_nonoverlapping':True,
            'conditional_warm_dense_Amdahl':{'FFN_boundary_fraction_of_generation_wall':F/T,
                'bound_if_entire_boundary_disappeared':T/(T-F),
                'fixed_non_FFN_remainder_seconds':T-F,
                'assumptions':'Non-FFN remainder fixed and CUDA boundary sum acts as serial removable component; not a universal speed limit'},
            'pixel_comparisons':comparisons,'HF_derived_media_bytes':media_size,
            'new_weight_fits':0,'optimizer_updates':0,'official_quality_calls':0,
            'candidate_operator_selected':False,'stage_changed':False,'quality_or_speed_acceptance':False,
            'human_full_video_review':'NOT_OBTAINED',
            'limitations':['Two previously exposed sources/seeds, one timed observation per mode/source',
                           'Generation wall excludes setup, bank fitting and weight preload',
                           'Neutral full-width check is48 saved rows, not full model equivalence',
                           'New readout/affine weights are BF16, not re-HiF4 quantized',
                           'Pixel metrics cannot establish perceptual or task quality']}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['hf-root','receipts','media','out']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError('Preserve accepted results')
    write_json(a.out,analyze(a.hf_root,a.receipts,a.media))
