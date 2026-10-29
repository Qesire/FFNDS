"""Full160 same-coordinate Dense/value/DCAC spatial teacher/student capture.

Two exposed high-motion diagnostic images, each with a high-flow5x5 tile,
central5x5 tile and two far controls across all21 latent time slots. Existing
complete video states are exact passive-observer controls; no new decode.
"""
import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_temporal_video20 as video
import e0_raw_bf16_temporal_video20_quality as quality

vm, native, io = video.vm, video.native, video.io
legacy = io.legacy
BASE_ID = 'E0_S2_SPATIAL_TEACHER_STUDENT_FULL160_CAPTURE_20261029'
ROI = 'config/e0_s2_motion_roi_cpu_20261029_v1.json'
PARENT = 'config/e0_raw_bf16_temporal_video20_quality_20261029_v1.json'
MODES = ('dense', 'value10', 'dcac10')
ROWS = 21*52
SHAPES = {'V1': [ROWS, 5120], 'h': [ROWS, 13824], 'Y': [ROWS, 5120],
          'r': [ROWS, 5120], 'u': [ROWS, 5120], 'gate': [5120],
          'shift': [5120], 'scale': [5120], 'rows': [ROWS]}
PACKET_BYTES = ROWS*(4*5120+13824)*2+3*5120*2+ROWS*8
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 40 << 30,
             'host_available_min_bytes': 128 << 30, 'free_disk_min_bytes': 100 << 30,
             'wall_seconds': 3600, 'outer_seconds': 3900, 'new_HF_bytes': 40 << 30, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 300, 'outer_seconds': 360, 'new_HF_bytes': 8 << 20}
DESIGN = {'evidence': 'PASSIVE_FULL160_REAL_SPATIAL_DENSE_AND_STUDENT_INPUTS_WITH_PRIOR_TEN_STATE_BYTE_NEUTRALS',
          'sources': 'PREVIOUS_DENSE_RAFT_MAX_SUBJECT898_AND_CAMERA96_TWO_EXPOSED_DIAGNOSTICS_NOT_BLIND',
          'arms': list(MODES), 'rows': ROWS, 'spatial_anchors': 52,
          'spatial': 'ACTUAL_DENSE_FARNEBACK_HIGH_FLOW_5X5_PLUS_FIXED_CENTRAL_Y12TO16_X24TO28_5X5_PLUS2_FAR_LOW_FLOW_CONTROLS',
          'time': 'ALL21_LATENT_TAU_SAME_XY_GRID29X52_ALL4STEP40BLOCK',
          'boundaries': 'ORIGINAL_V1_POSTGELU_H_Y_GATE_R_AND_ACTUAL_U_SHIFT_SCALE; NO_FP4',
          'countermechanisms': ['TEACHER_LOCAL_DIRECTION_DOMAIN_GENERALIZATION', 'STUDENT_PREFIX_DISTRIBUTION_AMPLIFICATION',
                                'HIGH_FLOW_BACKGROUND_VS_CENTRAL_CONTENT_SPATIAL_DIFFERENCE'],
          'native_action': 'UNCHANGED_ORIGINAL_DENSE_OR_COMPLETED_R960_SHARED80_TWO_GEMM_BANK_ORIGINAL_B2',
          'capture': 'INDEX_SELECT_AND_CPU_SERIALIZATION_ONLY_NO_CHANGED_VALUES',
          'reference': 'EACH_ARM_ALL_TEN_ORIGINAL_COMPLETED_VIDEO_STATE_BYTES_REQUIRED_EQUAL_NO_REDECODE',
          'future_discriminator': 'SAME_STUDENT_H_RAW_W2_COUNTERFACTUAL_AND_FREQUENCY_CROSS_TO_SEPARATE_LOCAL_ERROR_FROM_PREFIX; SEPARATE_SCOPE',
          'semantic_ROI': 'NOT_CERTIFIED_HIGH_FLOW_AND_CENTRAL_ARE_GEOMETRIC_STRATA',
          'motion_alignment': 'NOT_YET_MEASURED_NEIGHBOR_COVERAGE_ONLY',
          'fitting': False, 'new_video_decode': False, 'quality_or_speed_accepted': False}
REQUIRED = {'tools/e0_s2_spatial_student_collect.py', 'tests/test_e0_s2_spatial_student_collect.py',
            'tools/e0_raw_bf16_temporal_video20.py', 'tools/e0_original_bf16_full_depth_collect.py',
            'tools/durable_worker_job.py', 'scripts/worker_wait_uuid_idle_exec.sh', ROI, PARENT}


def scope_id(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two single-source parts')
    return BASE_ID+'_PART'+str(part)


def output_relative(part):
    scope_id(part)
    return f'ffnds-derived/e0_s2_spatial_student_full160_20261029/part{part}/attempt1/worker/results'


def run_id(case, mode):
    native.need(mode in MODES, 'Unknown fixed captured arm')
    return f'spatial52_ord{case["native_source_ordinal"]}_seed42_{mode}'


def geometry(roi):
    high = [tuple(p) for p in roi['tile_yx']]
    center = [(y, x) for y in range(12, 17) for x in range(24, 29)]
    far = [tuple(p) for p in roi['far_controls_yx']]
    points = sorted(high+center+far)
    native.need(len(high) == len(center) == 25 and len(far) == 2 and len(set(points)) == 52
        and all(0 <= y < 29 and 0 <= x < 52 for y, x in points), 'Fixed two disjoint5x5 tiles plus far2 required')
    rows = [tau*1508+y*52+x for tau in range(21) for y, x in points]
    return {'grid_tau_y_x': [21, 29, 52], 'anchors_yx': [list(p) for p in points],
            'high_flow_tile_yx': [list(p) for p in high], 'central_tile_yx': [list(p) for p in center],
            'far_controls_yx': [list(p) for p in far], 'selected_rows': rows,
            'trajectory_shape_tau_anchor': [21, 52], 'semantics': 'GEOMETRIC_ROI_NOT_SEMANTIC_OR_MOTION_CORRESPONDENCE'}


def limits(part):
    scope_id(part)
    result = {k: v//2*3 for k, v in vm.native_original_limits.items()}
    result.update(original_expert_loads=2, raw_Default_MM_loads=812, buffer_Default_MM_loads=40,
        T5_raw_Default_MM_loads=168, T5_buffer_Default_MM_loads=7,
        video_generations=0, VAE_decodes=0, core_CPU_reports=0, native_prefix_replays_no_decode=3,
        BF16_factor_GEMM=640, original_W2_H2D_skips=324,
        original_W2_H2D_skipped_bytes=324*vm.ORIGINAL_W2_BYTES,
        factor_CPU_file_loads=160, factor_CPU_tensor_bytes=160*vm.FACTOR_BYTES,
        factor_GPU_family_preloads=2, factor_GPU_H2D_tensors=480,
        factor_GPU_H2D_bytes=2*80*vm.FACTOR_BYTES,
        row_index_uploads=1, row_index_upload_bytes=4*40*ROWS*8,
        raw_FFN_conditioning_captures=480, operand_row_selections=480*5,
        actual_grid_geometry_checks=480, raw_postwrite_packets=480,
        packet_tensor_D2H=480*9, packet_tensor_D2H_bytes=480*PACKET_BYTES,
        packet_payload_validations=480, packet_safetensor_writes=480, packet_metadata_writes=480,
        state_tensor_D2H=30, state_tensor_D2H_bytes=3*math.prod(native.VIDEO['latent_shape'])*(4+2+4*4+4*2),
        state_output_writes=30, historical_all_arm_state_byte_checks=30)
    return result


def load_inputs(plan, hf):
    rp = native.read(native.bound(ROOT, plan['ROI_producer']['protocol']))
    rr = native.read(native.bound(hf, plan['ROI_producer']['RUN']))
    vm.exited(native.read(native.bound(hf, plan['ROI_producer']['exit'])), plan['ROI_producer']['RUN'],
              plan['ROI_producer']['protocol']['sha256'], rp['id'])
    native.need(rr['status'] == 'COMPLETE_TWO_ACTUAL_DENSE_VIDEO_MOTION_ROI_CAPTURE_COORDINATES'
        and rr['predebited'] == rr['completed'] == {k: v for k, v in rp['limits'].items() if v}
        and rr['new_FFN_tensors'] == rr['model_calls'] == rr['GPU_calls'] == 0, 'Actual closed CPU ROI proof required')
    source = rr['sources'][plan['part']]
    native.need(source['case'] == plan['cases'][0] and geometry(source) == plan['geometry'], 'Actual ROI source/coordinates changed')
    qp = native.read(native.bound(ROOT, plan['completed_video_protocol']))
    quality.verify_generation(qp, hf)
    native.need(qp['source_cohort'] == plan['source_cohort'], 'Different video source cohort')
    controls = {}
    for mode in MODES:
        row = next(v for v in qp['videos'] if v['source'] == source['case']['native_source_ordinal'] and v['mode'] == mode)
        gen = native.read(native.bound(hf, row['generation']))
        native.need(gen['image_sha256'] == source['case']['image']['sha256']
            and gen['prompt_sha256'] == source['case']['prompt_sha256'] and gen['seed'] == 42
            and gen['states'] == row['states'] and len(row['states']) == 10, 'Actual arm control source/state differs')
        controls[mode] = {'generation': row['generation'], 'states': row['states'], 'reused_video': row['video']}
    return controls


def validate(plan, frozen=False):
    native.need(plan['id'] == scope_id(plan['part']) and plan['resources'] == RESOURCES
        and plan['CPU_preflight_resources'] == CPU_RESOURCES and plan['limits'] == limits(plan['part'])
        and plan['design'] == DESIGN and plan['numeric_path'] == 'BF16_ORIGINAL'
        and plan['runtime_overlay'] == native.recipe() and plan['automatic_retry'] is False
        and plan['output_HF_relative'] == output_relative(plan['part']) and len(plan['cases']) == 1
        and plan['cases'][0]['native_source_ordinal'] == (898 if plan['part'] == 0 else 96)
        and plan['run_ids'] == [run_id(plan['cases'][0], m) for m in MODES]
        and len(plan['geometry']['selected_rows']) == ROWS and REQUIRED <= {b['path'] for b in plan['repo_sources']},
        'Fixed spatial/student capture scope differs')
    if frozen: native.need(plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True, 'Scope not frozen')


def authorize(plan, sha):
    validate(plan, True); active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True, 'Current continuation inactive')
    ss = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    native.need(len(ss) == 1 and ss[0]['protocol_sha256'] == sha and ss[0]['max_gpu_cards'] == 1
        and ss[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and ss[0]['limits'] == plan['limits'] and ss[0]['resources'] == RESOURCES
        and ss[0]['run_ids'] == plan['run_ids'], 'Exact new capture scope not registered')
    return ss[0]


def cpu_identity(plan, sha):
    return {'id': plan['id'], 'protocol_sha256': sha, 'status': 'PASS_SPATIAL52_TEACHER_STUDENT_EXISTING_VIDEO_CONTROLS_FACTOR_SHA_CURRENT_MASTER',
        'source_closure_sha256': native.canonical(plan['repo_sources']), 'geometry_sha256': native.canonical(plan['geometry']),
        'factor_bank': plan['factor_bank'], 'ROI_producer': plan['ROI_producer'], 'factor_files': 160,
        'factor_payload_bytes': 160*vm.FACTOR_BYTES, 'new_tensor_model_GPU_calls': 0, 'numerical_imports': False}


def preflight(args):
    start = time.monotonic(); plan = native.read(args.protocol); sha = native.sha(args.protocol); authorize(plan, sha)
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 4, 'Fresh hidden-CUDA byte worker required')
    native.need(args.out is not None and not args.out.exists(), 'Fresh CPU proof only'); args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'id': plan['id'], 'protocol_sha256': sha, 'status': 'STARTED_CAPTURE_BYTE_PREFLIGHT'}
    native.write(args.out, report)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU300s exhausted'))); signal.alarm(300)
    try:
        io.require_execution_master(plan, args.hf_root)
        for b in plan['repo_sources']: native.bound(ROOT, b)
        native.model_metadata(plan); native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
        controls = load_inputs(plan, args.hf_root)
        for arm in controls.values():
            native.bound(args.hf_root, arm['reused_video'])
            for item in arm['states'].values(): native.bound(args.hf_root, item)
        factors = video.factor_index(plan, args.hf_root, True)
        native.need(not {'torch', 'numpy'} & set(sys.modules), 'Numeric import during byte proof')
        report.update(cpu_identity(plan, sha), factor_bindings=factors,
            factor_bindings_sha256=native.canonical(factors), controls_sha256=native.canonical(controls))
    except BaseException as error:
        report.update(status='FAILED_CPU_CAPTURE_PROOF_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-start; native.write(args.out, report)


def capture_phase(torch, pipe, guard, plan, mode, indices, controls, out, hf, sha, report, counter, persist, check):
    from safetensors.torch import save_file
    from ub1_consumer_runtime import tensor_sha
    from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
    from lightx2v.utils.utils import seed_all
    case = plan['cases'][0]; rid = run_id(case, mode); folder = out/rid; folder.mkdir()
    phase = {'id': plan['id'], 'protocol_sha256': sha, 'run_id': rid, 'native_source_ordinal': case['native_source_ordinal'],
        'source_id': case['source_id'], 'image_sha256': case['image']['sha256'], 'prompt_sha256': case['prompt_sha256'],
        'mode': mode, 'seed': 42, 'source_state': 'teacher_input' if mode == 'dense' else 'student_input',
        'numeric_path': 'BF16_ORIGINAL', 'status': 'STARTED_NO_RETRY', 'states': {}, 'packets': [],
        'original_video_control': controls[mode]}
    report['phases'].append(phase); persist(); ledger = guard.ledger
    def save(value, name, key):
        def output():
            expected = torch.float32 if key in ('raw_FP32_noise_files', 'native_FP32_flow_files') else torch.bfloat16
            native.need(name not in phase['states'] and value.dtype == expected and list(value.shape) == native.VIDEO['latent_shape'], 'Actual state dtype/geometry differs')
            counter.reserve_payload(value.numel()*value.element_size())
            cpu = ledger.call(rid, {'state_tensor_D2H': 1, 'state_tensor_D2H_bytes': value.numel()*value.element_size()}, lambda: value.detach().cpu().contiguous())
            native.need(bool(torch.isfinite(cpu).all()), 'Nonfinite native state')
            path = folder/(name+'.safetensors')
            try: ledger.call(rid, {'state_output_writes': 1}, save_file, {'flow' if key == 'native_FP32_flow_files' else 'latents': cpu}, str(path))
            finally:
                if path.exists(): counter.observe_external(path)
            item = {'root': 'hf', **counter.binding(path, hf), 'raw_tensor_sha256': tensor_sha(cpu),
                    'shape': list(cpu.shape), 'dtype': str(cpu.dtype).removeprefix('torch.')}
            ledger.call(rid, {'historical_all_arm_state_byte_checks': 1},
                lambda: native.need(vm.same_state(item, controls[mode]['states'][name]), 'Passive observer changed actual original video state bytes'))
            phase['states'][name] = item; persist(); return item
        return ledger.call(rid, {key: 1}, output)
    def packet(p, grid):
        native.need(grid == [21, 29, 52] and set(p['values']) == set(SHAPES)-{'rows'}, 'Actual capture boundaries/geometry differ')
        counter.reserve_payload(PACKET_BYTES)
        values = {**p['values'], 'rows': p['rows']}; cpu = {}
        for key, value in values.items():
            cpu[key] = ledger.call(rid, {'packet_tensor_D2H': 1, 'packet_tensor_D2H_bytes': value.numel()*value.element_size()}, lambda value=value: value.detach().cpu().contiguous())
        def validate_packet():
            native.need(all(list(t.shape) == SHAPES[k] and t.dtype == (torch.int64 if k == 'rows' else torch.bfloat16)
                and bool(torch.isfinite(t).all()) for k, t in cpu.items())
                and cpu['rows'].tolist() == plan['geometry']['selected_rows'], 'Actual saved packet coords/type/nonfinite differs')
            return {k: {'shape': list(t.shape), 'dtype': str(t.dtype).removeprefix('torch.'), 'raw_tensor_sha256': tensor_sha(t)} for k, t in cpu.items()}
        meta = ledger.call(rid, {'packet_payload_validations': 1}, validate_packet)
        stem = f'step{p["step"]}_block{p["block"]:02d}'; path = folder/(stem+'.safetensors')
        try: ledger.call(rid, {'packet_safetensor_writes': 1}, save_file, cpu, str(path))
        finally:
            if path.exists(): counter.observe_external(path)
        record = {'schema': 'ffnds.s2.spatial_student.operands.v1', 'run_id': rid, 'protocol_sha256': sha,
            'source_id': case['source_id'], 'native_source_ordinal': case['native_source_ordinal'],
            'image_sha256': case['image']['sha256'], 'prompt_sha256': case['prompt_sha256'], 'seed': 42,
            'numeric_path': 'BF16_ORIGINAL', 'source_state': phase['source_state'], 'mode': mode,
            'step': p['step'], 'block': p['block'], 'expert': 'high' if p['step'] < 2 else 'low',
            'grid': grid, 'population_rows': 31668, 'selected_rows': plan['geometry']['selected_rows'],
            'geometry_sha256': native.canonical(plan['geometry']), 'full_operand_layout': p['source_layout'],
            'tensor_metadata': meta, 'payload': {'root': 'hf', **counter.binding(path, hf)},
            'passive_observer': True, 'new_pre_GELU_activation': 'NOT_CAPTURED_NOT_REQUIRED_FOR_CURRENT_W2_PREFIX_DISCRIMINATOR'}
        mp = folder/(stem+'.json'); ledger.call(rid, {'packet_metadata_writes': 1}, counter.write_json, mp, record)
        phase['packets'].append({'step': p['step'], 'block': p['block'], 'metadata': counter.binding(mp, hf)})
        persist(); check()
    pipe.runner.scheduler.generator = None
    tap = legacy.CollectorTap(pipe, guard, rid, save, check, case, indices, packet)
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    def replay():
        runner = pipe.runner; info = init_empty_input_info('i2v', pipe.support_tasks)
        update_input_info_from_dict(info, {'seed': 42, 'prompt': case['prompt'], 'negative_prompt': '',
            'image_path': str((hf/case['image']['path']).resolve()), 'save_result_path': 'UNUSED_OBSERVER_NO_DECODE'})
        seed_all(42); runner.input_info = info
        with tap, torch.inference_mode():
            runner.inputs = runner.run_input_encoder(); runner.init_run()
            native.need(runner.video_segment_num == 1, 'One segment required')
            runner.init_run_segment(0); runner.run_segment(0); runner.end_run()
        torch.cuda.synchronize(); tap.finish()
    try:
        ledger.call(rid, {'native_prefix_replays_no_decode': 1}, replay)
        native.need(len(phase['states']) == 10 and len(phase['packets']) == 160, 'Incomplete real capture')
        phase.update(status='COMPLETE_ALL160_SPATIAL_TEACHER_OR_STUDENT_CAPTURE_TEN_BYTE_NEUTRALS',
            all_ten_prior_video_state_bytes_equal=True, completed_native_flow_steps=4, scheduler_generator_reset=True,
            FFN_boundary_CUDA_seconds=sum(a.elapsed_time(b) for a, b in tap.events)/1000,
            capture_wall_seconds=time.perf_counter()-started, peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            all160_FFN_sites=sorted(tap.seen), native_counter_delta={'linear_forward_calls': len(tap.lines), 'activation_qdq_calls': 0, 'weight_quantizations': 0})
        counter.write_json(folder/'GENERATION_OBSERVATION.json', phase)
        phase['observation_receipt'] = counter.binding(folder/'GENERATION_OBSERVATION.json', hf)
    except BaseException as error:
        phase.update(status='FAILED_OBSERVER_PHASE_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally: counter.write_json(folder/'PHASE.json', phase); persist()


def run(args):
    started = time.monotonic(); plan = native.read(args.protocol); sha = native.sha(args.protocol); scope = authorize(plan, sha)
    proof = scope['cpu_preflight']; cpu = native.read(native.bound(args.hf_root, proof['RUN']))
    vm.exited(native.read(native.bound(args.hf_root, proof['exit'])), proof['RUN'], sha, plan['id'])
    native.need(all(cpu.get(k) == v for k, v in cpu_identity(plan, sha).items())
        and cpu['factor_bindings_sha256'] == native.canonical(cpu['factor_bindings'])
        and 0 <= cpu['wall_seconds'] <= CPU_RESOURCES['wall_seconds'], 'Exited CPU proof differs')
    native.need(not {'torch', 'numpy'} & set(sys.modules), 'Fresh numeric worker required')
    io.require_execution_master(plan, args.hf_root)
    for item in plan['repo_sources']: native.bound(ROOT, item)
    controls = load_inputs(plan, args.hf_root)
    native.need(cpu['controls_sha256'] == native.canonical(controls), 'Controls changed after CPU proof')
    native.model_metadata(plan)
    local, package_rows = native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    native.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions']
        and len(os.sched_getaffinity(0)) <= 4 and not any(os.environ.get(k) == '1'
        for k in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')), 'Runtime/CPU/precision differs')
    native.ID, native.LIMITS, native.RUN_IDS, native.RESOURCE = plan['id'], limits(plan['part']), tuple(plan['run_ids']), RESOURCES
    out = (args.hf_root/plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined output only')
    out.mkdir(parents=True)
    report = {'id': plan['id'], 'protocol_sha256': sha, 'part': plan['part'], 'numeric_path': 'BF16_ORIGINAL',
        'status': 'STARTED_FULL160_SPATIAL_TEACHER_STUDENT_CAPTURE', 'CPU_preflight': proof,
        'design': DESIGN, 'geometry': plan['geometry'], 'phases': [], 'automatic_retry': False,
        'new_videos': 0, 'fits': 0, 'official_quality': 'NOT_RUN', 'quality_or_speed_accepted': False}
    monitor = None; counter = io.WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    journal = counter.open_journal(out/'OPERATIONS.jsonl'); persist = lambda: counter.write_json(out/'RUN.json', report)
    ledger = vm.Ledger(report.setdefault('operation_ledger', {}), lambda event: journal.write(json.dumps(event, separators=(',', ':'))+'\n'))
    meter = vm.FactorMeter(ledger); current = {'rid': None, 'skips': []}; persist()
    def emit(event):
        if event['event'] == 'original_W2_H2D_skipped':
            native.need(current['rid'] is not None and event['bytes'] == vm.ORIGINAL_W2_BYTES, 'Unexpected W2 skip')
            ledger.call(current['rid'], {'original_W2_H2D_skips': 1, 'original_W2_H2D_skipped_bytes': vm.ORIGINAL_W2_BYTES}, current['skips'].append, event)
        else: journal.write(json.dumps({'kind': 'runtime_dispatch', **event}, separators=(',', ':'))+'\n')
    def check():
        if monitor: monitor.check()
        counter.check()
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available >= RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes']
            and time.monotonic()-started <= RESOURCES['wall_seconds'], 'Host/disk/wall budget exceeded')
        if 'torch' in sys.modules: native.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'GPU40GiB exceeded')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Capture3600s exhausted'))); signal.alarm(3600)
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder, no_hif4 = CheckedPackageFinder(local, package_rows), native.NoHiF4()
    file_guard = io.single_file_bound(); file_guard.__enter__()
    try:
        sys.dont_write_bytecode = True; sys.path[:0] = [str(local), str(ROOT/'.runtime/code/LightX2V')]
        sys.meta_path[:0] = [no_hif4, finder]
        os.environ.update(DTYPE='BF16', SENSITIVE_LAYER_DTYPE='BF16', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                          OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        reservation, monitor, report['CUDA_reservation'] = vm.reserved_startup.reserve_before_assets(torch)
        native.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'One assigned A100 required')
        torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        report['runtime_precision'] = {'TF32': False, 'FP32_matmul': 'highest', 'BF16_reduced_precision_reduction': True}
        check(); snapshot = out/'EXECUTED_SOURCES'; snapshot.mkdir()
        copies = [(ROOT/b['path'], snapshot/b['path'], b) for b in plan['repo_sources']]
        copies.append((args.protocol, snapshot/'protocol.json', native.binding(args.protocol, ROOT)))
        counter.hold('frozen_metadata', sum(b['bytes'] for _, _, b in copies))
        for source, destination, binding in copies:
            destination.parent.mkdir(parents=True, exist_ok=True); counter.copy_bound(source, destination, binding)
        native.need(counter.holds.pop('frozen_metadata') == 0, 'Source snapshot bytes differ')
        with native.OriginalGuard(torch, plan, ledger) as guard:
            at = time.perf_counter(); pipe = native.build_pipeline(plan); torch.cuda.synchronize()
            report['raw_model_setup_seconds'] = time.perf_counter()-at
            banks = video.load_cpu_banks(torch, cpu['factor_bindings'], args.hf_root, ledger)
            case = plan['cases'][0]
            indices = ledger.call('setup', {'row_index_uploads': 1, 'row_index_upload_bytes': 4*40*ROWS*8}, torch.tensor,
                [[plan['geometry']['selected_rows'] for _ in range(40)] for _ in range(4)], dtype=torch.int64, device='cuda')
            gpu_bank = {}
            with vm.FactorW2Runtime(pipe, guard, gpu_bank, factor_forward=vm.core.factor_forward, meter=meter,
                synchronize=torch.cuda.synchronize, record_stream=lambda t: t.record_stream(torch.cuda.current_stream()), emit=emit) as runtime:
                for mode in MODES:
                    check(); authorize(plan, sha); rid = run_id(case, mode); meter.rid = current['rid'] = rid; current['skips'] = []
                    torch.cuda.synchronize(); runtime.bank = {}; gpu_bank.clear(); del gpu_bank; torch.cuda.empty_cache()
                    gpu_bank = {} if mode == 'dense' else video.to_gpu_bank(torch, banks[mode], ledger, rid)
                    runtime.bank = gpu_bank; runtime.set_mode('dense' if mode == 'dense' else 'factor')
                    capture_phase(torch, pipe, guard, plan, mode, indices, controls, out, args.hf_root, sha, report, counter, persist, check)
                    native.need(len(current['skips']) == (0 if mode == 'dense' else 162), 'Original W2 skip coverage differs')
                torch.cuda.synchronize(); runtime.bank = {}; gpu_bank.clear()
            ledger.finish(); native.need(len(report['phases']) == 3 and not counter.holds, 'Incomplete capture part')
            report.update(status='COMPLETE_ONE_SOURCE_THREE_ARMS_FULL160_SPATIAL_STUDENT_CAPTURE',
                all_thirty_prior_video_state_byte_neutrals=True, actual_native_packets=480,
                actual_packet_tensor_payload_bytes=480*PACKET_BYTES, checked_transformers_imports=finder.origins)
            check()
    except BaseException as error:
        report.update(status='FAILED_SPATIAL_STUDENT_CAPTURE_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); monitor_error = None
        if monitor:
            try:
                monitor.close(); report['assigned_GPU_resources'] = monitor.report()
                native.need(not report['assigned_GPU_resources']['violations'], 'Assigned GPU violation')
            except BaseException as error:
                monitor_error = error; report.update(status='FAILED_SPATIAL_STUDENT_CAPTURE_PRESERVED_NO_RETRY', resource_error=str(error))
        for item in (finder, no_hif4):
            if item in sys.meta_path: sys.meta_path.remove(item)
        report.update(wall_seconds=time.monotonic()-started, output_accounting=counter.summary(),
                      operation_journal=counter.binding(out/'OPERATIONS.jsonl', args.hf_root))
        persist(); journal.close(); file_guard.__exit__(None, None, None)
        if monitor_error: raise monitor_error


def draft(hf, part):
    scope_id(part); qp = native.read(ROOT/PARENT)
    producer = native.read(ROOT/'config/e0_raw_bf16_temporal_video20_part0_20261029_v1.json')
    inherit = ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions', 'local_transformers',
        'private_worker_config_sha256', 'execution_private_worker_config_sha256', 'execution_raw_master_CPU_proof',
        'runtime_overlay', 'source_cohort', 'factor_bank', 'reference_protocol')
    plan = {k: copy.deepcopy(producer[k]) for k in inherit}
    roi_p = native.read(ROOT/ROI); folder = hf/roi_p['output_HF_relative']; rr = native.read(folder/'RUN.json')
    paths = {b['path'] for b in producer['repo_sources']} | {b['path'] for b in qp['repo_sources']} | REQUIRED
    plan.update(id=scope_id(part), part=part, numeric_path='BF16_ORIGINAL', status='DRAFT_NOT_AUTHORIZED',
        assets_finalized=False, resources=RESOURCES, CPU_preflight_resources=CPU_RESOURCES, limits=limits(part), design=DESIGN,
        ROI_producer={'protocol': native.binding(ROOT/ROI, ROOT), 'RUN': native.binding(folder/'RUN.json', hf),
                      'exit': native.binding(folder.parent/'EXIT_RECEIPT.json', hf)},
        completed_video_protocol=native.binding(ROOT/PARENT, ROOT), geometry=geometry(rr['sources'][part]),
        cases=[rr['sources'][part]['case']], automatic_retry=False, output_HF_relative=output_relative(part),
        independent_review='NOT_PERFORMED_AGENT_SERVICE_USAGE_LIMIT')
    plan['run_ids'] = [run_id(plan['cases'][0], mode) for mode in MODES]
    plan['repo_sources'] = [native.binding(ROOT/p, ROOT) for p in sorted(paths)]
    validate(plan); load_inputs(plan, hf)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('draft', 'preflight', 'run'))
    p.add_argument('--part', type=int, choices=(0, 1)); p.add_argument('--protocol', type=Path)
    p.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    p.add_argument('--local-root', type=Path); p.add_argument('--out', type=Path); args = p.parse_args()
    if args.action == 'draft':
        native.need(args.out is not None and not args.out.exists(), 'Fresh draft only'); native.write(args.out, draft(args.hf_root, args.part))
    elif args.action == 'preflight': preflight(args)
    else: run(args)


if __name__ == '__main__': main()
