"""Register and locate a real 5x5 native-space ROI on two existing Dense videos.

Select one subject and one camera source by their previously measured Dense
RAFT amplitude, without candidate score selection. This CPU-only producer
decodes actual full videos and produces capture coordinates, not FFN tensors.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_temporal_video20_quality as evidence
base = evidence.base
ID = 'E0_S2_DENSE_MOTION_ROI_CAPTURE_DESIGN_CPU_20261029'
PARENT = 'config/e0_raw_bf16_temporal_video20_quality_20261029_v1.json'
OUTPUT = 'ffnds-derived/e0_s2_dense_motion_roi_cpu_20261029/attempt1/worker/results'
RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 300, 'outer_seconds': 360,
             'new_HF_bytes': 16 << 20, 'automatic_retry': False}
LIMITS = {'source_video_decoders': 2, 'source_image_byte_checks': 2, 'video_decoded_frames': 162, 'adjacent_frame_Farneback': 160,
          'model_calls': 0, 'GPU_calls': 0, 'new_videos': 0, 'new_FFN_tensors': 0}
FLOW = {'pyr_scale': .5, 'levels': 3, 'winsize': 15, 'iterations': 3,
        'poly_n': 5, 'poly_sigma': 1.2, 'flags': 0}
DESIGN = {'source_selection': 'MAX_PREVIOUS_DENSE_CONTINUOUS_RAFT_WITHIN_SUBJECT_AND_CAMERA_LEXICAL_ORDINAL_TIES',
          'source_role': 'TWO_HIGH_MOTION_EXPOSED_DEVELOPMENT_DIAGNOSTIC_NOT_POPULATION_OR_BLIND',
          'spatial_rule': 'MAX_MEAN_FULLVIDEO_ADJACENT_FRAME_FARNEBACK_MAGNITUDE_5X5_NATIVE_CELLS',
          'control_rule': 'TWO_LOW_FLOW_CELLS_CHEBYSHEV_DISTANCE_GT5_FROM_ROI_CENTER_AND_GT8_FROM_EACH_OTHER',
          'coordinates': '25_CONTIGUOUS_NATIVE_XY_PLUS2_FAR_CONTROLS_ALL21TAU_567_ROWS_FOR_FUTURE_CAPTURE',
          'geometry': '832X464_PIXEL_VIDEO_TO52X29_NATIVE_SPATIAL_GRID_INTER_AREA',
          'online_action': 'NONE_OFFLINE_OBSERVER_DESIGN_ONLY_NO_FREE_MATCH_OR_MOTION_CLAIM'}


def anchors(heat, np):
    heat = np.asarray(heat, dtype=np.float64)
    base.require(heat.shape == (29, 52) and bool(np.isfinite(heat).all()) and bool((heat >= 0).all()), 'Expected finite29x52 nonnegative motion map')
    centers = [(float(heat[y-2:y+3, x-2:x+3].mean()), y, x) for y in range(2, 27) for x in range(2, 50)]
    score, y, x = min(centers, key=lambda p: (-p[0], p[1], p[2]))
    tile = [(a, b) for a in range(y-2, y+3) for b in range(x-2, x+3)]
    candidates = sorted([(float(heat[a, b]), a, b) for a in range(29) for b in range(52)
                         if max(abs(a-y), abs(b-x)) > 5])
    controls = []
    for _, a, b in candidates:
        if all(max(abs(a-c), abs(b-d)) > 8 for c, d in controls): controls.append((a, b))
        if len(controls) == 2: break
    base.require(len(controls) == 2 and len(set(tile+controls)) == 27, 'Spatial controls unavailable')
    points = sorted(tile+controls)
    rows = [tau*29*52+a*52+b for tau in range(21) for a, b in points]
    return {'roi_center_yx': [y, x], 'roi_mean_flow_pixels_per_adjacent_frame': score,
            'tile_yx': [list(p) for p in tile], 'far_controls_yx': [list(p) for p in controls],
            'anchors_yx': [list(p) for p in points], 'selected_rows': rows,
            'trajectory_shape_tau_anchor': [21, 27], 'grid_tau_y_x': [21, 29, 52]}


def inputs(hf):
    parent = base.read(ROOT/PARENT); old, quality = evidence.dense_quality(parent, hf)
    sources = base.read_bound(ROOT, parent['source_cohort'])['sources']
    selected = []
    for stratum in ('subject', 'camera'):
        rows = [s for s in quality['source_summary']['per_source'] if s['stratum'] == stratum]
        chosen = min(rows, key=lambda r: (-r['continuous_RAFT']['dense']['mean_top5percent_flow_px_per_2frame_pair'], r['source']))
        case = next(c for c in sources if c['native_source_ordinal'] == chosen['source'])
        dense = next(v for v in old['videos'] if v['source'] == chosen['source'] and v['mode'] == 'dense')
        selected.append({'case': case, 'Dense_video': dense['video'], 'Dense_generation': dense['generation'],
                         'selection_Dense_RAFT': chosen['continuous_RAFT']['dense']})
    return parent, selected


def draft(hf):
    parent, selected = inputs(hf)
    paths = {b['path'] for b in parent['repo_sources']} | {
        'tools/e0_s2_motion_roi_cpu.py', 'tests/test_e0_s2_motion_roi_cpu.py',
        'tools/durable_worker_job.py', 'tools/worker_identity.py', 'tools/e0_raw_bf16_temporal_video20_quality.py'}
    return {'id': ID, 'status': 'DRAFT_NOT_AUTHORIZED', 'parent_protocol': base.binding(ROOT/PARENT, ROOT),
            'sources': selected, 'resources': RESOURCES, 'limits': LIMITS, 'design': DESIGN, 'Farneback': FLOW,
            'execution_private_worker_config_sha256': base.read(ROOT/PARENT)['execution_private_worker_config_sha256'],
            'repo_sources': [base.binding(ROOT/p, ROOT) for p in sorted(paths)],
            'output_HF_relative': OUTPUT, 'automatic_retry': False}


def authorize(plan, sha):
    base.require(plan['id'] == ID and plan['status'] == 'FROZEN_REGISTERED' and plan['resources'] == RESOURCES
        and plan['limits'] == LIMITS and plan['design'] == DESIGN and plan['Farneback'] == FLOW, 'Fixed ROI design differs')
    active = base.read(ROOT/'research/ACTIVE_RESEARCH.json')
    rows = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    base.require(len(rows) == 1 and rows[0]['protocol_sha256'] == sha and rows[0]['max_gpu_cards'] == 0
        and rows[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE'), 'Exact CPU scope not registered')


def run(args):
    plan = base.read(args.protocol); sha = base.sha(args.protocol); authorize(plan, sha)
    evidence.q.verify_worker(plan)
    base.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'numpy', 'torch'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 4, 'Fresh hidden-CUDA four CPU process required')
    base.require(base.binding(ROOT/PARENT, ROOT) == plan['parent_protocol'], 'Parent protocol changed')
    _, selected = inputs(args.hf_root); base.require(selected == plan['sources'], 'Source selection changed')
    for binding in plan['repo_sources']: base.bound(ROOT, binding)
    out = (args.hf_root/OUTPUT).resolve()
    base.require(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined output only')
    out.mkdir(parents=True); started = time.monotonic()
    report = {'id': ID, 'protocol_sha256': sha, 'status': 'STARTED_TWO_DENSE_VIDEO_ROI_CPU',
              'predebited': {}, 'completed': {}, 'sources': [], 'model_calls': 0, 'GPU_calls': 0,
              'new_videos': 0, 'new_FFN_tensors': 0, 'automatic_retry': False}
    save = lambda: base.write(out/'RUN.json', report)
    def call(key, fn, *a, **kw):
        report['predebited'][key] = report['predebited'].get(key, 0)+1
        base.require(report['predebited'][key] <= LIMITS[key], 'CPU work budget exceeded')
        result = fn(*a, **kw); report['completed'][key] = report['completed'].get(key, 0)+1
        return result
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('ROI300s exhausted')))
    signal.alarm(RESOURCES['wall_seconds']); save()
    try:
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '4'
        import numpy as np
        import cv2
        cv2.setNumThreads(4)
        base.require('torch' not in sys.modules, 'Unexpected model library import')
        report['runtime'] = {'NumPy': np.__version__, 'OpenCV': cv2.__version__, 'Torch_imported': 'torch' in sys.modules}
        for selected in plan['sources']:
            case = selected['case']; path = base.bound(args.hf_root, selected['Dense_video'])
            call('source_image_byte_checks', base.bound, args.hf_root, case['image'])
            cap = call('source_video_decoders', cv2.VideoCapture, str(path))
            try:
                base.require(cap.isOpened(), 'Dense video decoder unavailable')
                heat = np.zeros((29, 52), np.float64); previous = None
                frame_hashes = []; frame_count = 0
                while True:
                    okay, frame = cap.read()
                    if not okay: break
                    base.require(frame.shape == (464, 832, 3) and frame_count < 81, 'Actual Dense frame geometry/count differs')
                    def decode():
                        frame_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
                        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = call('video_decoded_frames', decode); frame_count += 1
                    if previous is not None:
                        flow = call('adjacent_frame_Farneback', cv2.calcOpticalFlowFarneback, previous, gray, None,
                                    FLOW['pyr_scale'], FLOW['levels'], FLOW['winsize'], FLOW['iterations'], FLOW['poly_n'], FLOW['poly_sigma'], FLOW['flags'])
                        magnitude = np.sqrt(np.sum(flow.astype(np.float64)**2, axis=-1))
                        base.require(bool(np.isfinite(magnitude).all()), 'Nonfinite Dense flow')
                        heat += cv2.resize(magnitude, (52, 29), interpolation=cv2.INTER_AREA)
                    previous = gray
                base.require(frame_count == 81, 'Full81frame Dense playback required')
                heat /= 80
                report['sources'].append({**selected, **anchors(heat, np), 'mean_flow_heatmap_native_grid': heat.tolist(),
                    'decoded_Bgr_frame_hashes': frame_hashes, 'decoded_frames': frame_count,
                    'flow_identity': 'OFFLINE_FULLVIDEO_ADJACENT_FRAME_FARNEBACK_FOR_CAPTURE_COORDINATES_NOT_RAFT_OR_SEMANTIC_LABEL'})
                save()
            finally: cap.release()
        base.require(all(report[side].get(k, 0) == n for side in ('predebited', 'completed') for k, n in LIMITS.items()), 'Incomplete actual CPU work')
        report.update(status='COMPLETE_TWO_ACTUAL_DENSE_VIDEO_MOTION_ROI_CAPTURE_COORDINATES',
                      new_native_capture='NOT_RUN', new_student_capture='NOT_RUN')
    except BaseException as error:
        report.update(status='FAILED_CPU_ROI_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-started; save()
        base.require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= RESOURCES['new_HF_bytes'], 'Output bytes exceeded')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('draft', 'run'))
    p.add_argument('--protocol', type=Path); p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    p.add_argument('--out', type=Path); args = p.parse_args()
    if args.action == 'draft':
        base.require(args.out is not None and not args.out.exists(), 'Fresh draft required'); base.write(args.out, draft(args.hf_root))
    else: run(args)


if __name__ == '__main__': main()
