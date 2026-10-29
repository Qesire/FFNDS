"""Matched full160 free videos for the completed value10 and dcac10 banks.

Reuse the twenty actual original-BF16 Dense references, with their original
success/failure/CPU-repair lineage. New videos change only the frozen W2 bank.
The numerical producer/runtime are loaded in a private namespace, unchanged.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:
    sys.path.insert(0, str(ROOT/'tools'))
_spec = importlib.util.spec_from_file_location('_ffnds_temporal_video20_native', ROOT/'tools/e0_raw_bf16_full160_video20_v2.py')
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)
native, io = vm.native, vm.io

BASE_ID = 'E0_RAW_BF16_VALUE10_DCAC10_FULL160_VIDEO20_20261029'
COHORT = vm.COHORT
BANK = 'config/e0_raw_bf16_new10_temporal_representation_full80_bank_20261029_v1.json'
TEMPLATE = 'config/e0_raw_bf16_full160_video20_part0_20261029_v2.json'
REFERENCE = 'config/e0_raw_bf16_video20_quality_20261029_v1.json'
MODES = ('value10', 'dcac10')
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 40 << 30,
             'host_available_min_bytes': 128 << 30, 'wall_seconds': 2400,
             'outer_seconds': 2700, 'new_HF_bytes': 4 << 30,
             'free_disk_min_bytes': 5 << 30, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 300,
                 'outer_seconds': 360, 'new_HF_bytes': 8 << 20}
DESIGN = {'evidence': 'EXPLORATORY_S2_MATCHED_REPRESENTATION_FREE_STUDENT_VIDEO',
          'sources': 'SAME20_EXPOSED_DEVELOPMENT_IMAGE_CLUSTERS_DISJOINT_FROM_NEW10_FIT',
          'arms': list(MODES), 'rank': 960, 'factor_files_per_arm': 80, 'beta': 1.,
          'action': 'ORIGINAL_W1_TANH_GELU_THEN_TWO_BF16_GEMMS_ORIGINAL_B2_GATE_RESIDUAL_ALL160',
          'factor_selection': 'EXACT_COMPLETE_NEW10_EXPORTS_NO_REFIT_OR_TUNING',
          'prediction_1': 'DC_AC_BALANCING_IMPROVES_MOTION_VS_SAME_DATA_VALUE_OBJECTIVE',
          'prediction_2': 'TEACHER_AC_GAINS_DO_NOT_TRANSFER_OR_TAU0_DC_COST_DOMINATES_FREE_ROLLOUT',
          'reference': 'REUSE20_ACTUAL_DENSE_VIDEOS_10_NATIVE_STATES_CORE_AND_OFFICIAL_METRICS_WITH_ORIGINAL_LINEAGE',
          'noise': 'EACH_NEW_CLIP_REAL_FP32_RAW_AND_BF16_INITIAL_BYTES_EQUAL_ACTUAL_SAME_SOURCE_DENSE',
          'state': 'FOUR_STEP_FREE_STUDENT_NO_SHADOW_DENSE_NO_CACHE',
          'order': 'ALTERNATE_TWO_ARMS_WITHIN_EACH_PART_GLOBAL_INDEX_DIV2_MOD2',
          'timing': 'SINGLE_OBSERVED_ORDER_WITH_SETUP_PRELOAD_OBSERVERS_NOT_FAIR_WARMED_SYS',
          'quality': 'ALL81_FRAME_PAIRED_CORE_NOW_OFFICIAL_EIGHT_AXES_AND_HUMAN_SEPARATE',
          'new_Dense_generations': 0, 'new_fits': 0, 'official_quality_calls': 0,
          'formal_confirmation': False, 'quality_or_speed_accepted': False}
REQUIRED = {'tools/e0_raw_bf16_temporal_video20.py', 'tests/test_e0_raw_bf16_temporal_video20.py',
            'tools/e0_raw_bf16_full160_video20_v2.py', 'tools/raw_bf16_factor_w2_runtime.py',
            'tools/raw_bf16_full80_spectrum.py', 'tools/e0_raw_bf16_video20_quality.py',
            'tools/durable_worker_job.py', 'scripts/worker_wait_uuid_idle_exec.sh', BANK, REFERENCE, COHORT}


def scope_id(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two ten-source parts')
    return BASE_ID+'_PART'+str(part)


def output_relative(part):
    scope_id(part)
    return f'ffnds-derived/e0_raw_bf16_temporal_video20_20261029/part{part}/attempt1/worker/results'


def run_id(case, mode):
    native.need(mode in MODES, 'Unknown fixed representation arm')
    return f'temporal20_ord{case["native_source_ordinal"]}_seed42_{mode}'


def phase_order(case):
    offset = (case['index']//2) % 2
    return MODES[offset:]+MODES[:offset]


def limits(part):
    scope_id(part)
    result = {k: v//2*20 for k, v in vm.native_original_limits.items()}
    result.update(original_expert_loads=2, raw_Default_MM_loads=812, buffer_Default_MM_loads=40,
        T5_raw_Default_MM_loads=168, T5_buffer_Default_MM_loads=7,
        core_CPU_reports=20, BF16_factor_GEMM=6400, original_W2_H2D_skips=3240,
        original_W2_H2D_skipped_bytes=3240*vm.ORIGINAL_W2_BYTES,
        factor_CPU_file_loads=160, factor_CPU_tensor_bytes=160*vm.FACTOR_BYTES,
        factor_GPU_family_preloads=20, factor_GPU_H2D_tensors=4800,
        factor_GPU_H2D_bytes=20*80*vm.FACTOR_BYTES, prefetch_transfer_observations=3240,
        stage_CUDA_event_pairs=140, state_tensor_D2H=200,
        state_tensor_D2H_bytes=20*math.prod(native.VIDEO['latent_shape'])*(4+2+4*4+4*2),
        state_output_writes=200, video_geometry_checks=20,
        matched_reference_noise_byte_checks=40, historical_state_byte_checks=0,
        source_registration_writes=20, reused_Dense_registration_writes=10)
    return result


def configure(plan):
    # The private producer classes resolve these registered constants at call time.
    vm.MODES, vm.RESOURCES, vm.run_id, vm.phase_order = MODES, RESOURCES, run_id, phase_order
    native.ID, native.LIMITS, native.RUN_IDS, native.RESOURCE = plan['id'], limits(plan['part']), tuple(plan['run_ids']), RESOURCES


def validate(plan, frozen=False):
    part = plan['part']
    native.need(plan['id'] == scope_id(part) and plan['resources'] == RESOURCES
        and plan['CPU_preflight_resources'] == CPU_RESOURCES and plan['design'] == DESIGN
        and plan['limits'] == limits(part) and plan['output_HF_relative'] == output_relative(part)
        and plan['numeric_path'] == 'BF16_ORIGINAL' and plan['automatic_retry'] is False
        and plan['runtime_overlay'] == native.recipe(), 'Fixed full160 temporal video scope differs')
    cohort = native.read(native.bound(ROOT, plan['source_cohort']))
    vm.validate_cohort(cohort)
    native.need(plan['cases'] == [c for c in cohort['sources'] if c['part'] == part]
        and plan['run_ids'] == [run_id(c, m) for c in plan['cases'] for m in phase_order(c)]
        and len(plan['cases']) == 10 and len(set(plan['run_ids'])) == 20, 'Fixed twenty video jobs differ')
    native.need(REQUIRED <= {b['path'] for b in plan['repo_sources']}, 'Source closure incomplete')
    if frozen:
        native.need(plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True,
                    'Assets/review not frozen')


def authorize(plan, sha):
    validate(plan, True)
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,
                'Current continued execution inactive')
    rows = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    native.need(len(rows) == 1 and rows[0]['protocol_sha256'] == sha
        and rows[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and rows[0]['resources'] == RESOURCES and rows[0]['limits'] == plan['limits']
        and rows[0]['run_ids'] == plan['run_ids'] and rows[0]['max_gpu_cards'] == 1,
        'Exact video scope not registered')
    return rows[0]


def factor_index(plan, hf, payload=False):
    bank = native.read(native.bound(ROOT, plan['factor_bank']))
    native.need(bank['status'] == 'COMPLETE_ACTUAL_TWO_EXPERT_EXIT0_FULL80_BANKS_VIDEO_NOT_RUN'
        and bank['models'] == plan['models'] and bank['rank'] == 960
        and bank['factor_files'] == 160 and bank['factor_payload_bytes'] == 160*vm.FACTOR_BYTES,
        'Closed full80 matched factor bank required')
    rows = []
    for parent in bank['parents']:
        producer = native.read(native.bound(ROOT, parent['protocol']))
        report = native.read(native.bound(hf, parent['RUN']))
        vm.exited(native.read(native.bound(hf, parent['exit'])), parent['RUN'], parent['protocol']['sha256'], producer['id'])
        native.need(report['status'] == 'COMPLETE_EXPERT40_MATCHED_VALUE_DC_AC_R960_BANKS'
            and report['protocol_sha256'] == parent['protocol']['sha256']
            and len(report['functions']) == 40 and report['exported_factors'] == 80,
            'Completed exact fitting parent required')
        native.need(producer['models'] == plan['models'] and not report['assigned_GPU_resources']['violations']
            and not report['operation_ledger'].get('forbidden_attempts', [])
            and all(report['operation_ledger'][side].get(k, 0) == n for side in ('predebited', 'observed') for k, n in producer['limits'].items()),
            'Fit model/resource/count identity differs')
        native.need(not ({c['image']['sha256'] for c in plan['cases']}
            & {c['image_sha256'] for c in producer['fitting_images']}), 'Video development source leaked into this fit')
        for entry in report['functions']:
            record = native.read(native.bound(hf, entry['record']))
            native.need(record['expert'] == entry['expert'] and record['block'] == entry['block']
                and len(record['fits']) == 2 and {f['family'] for f in record['fits']} == set(MODES), 'Matched function pair differs')
            for fit in record['fits']:
                row = {'expert': 0 if entry['expert'] == 'high' else 1, 'block': entry['block'],
                       'family': fit['family'], 'training_source_ids': fit['training_source_ids'], **fit['export']}
                native.need(row['tensors']['right.bias']['raw_tensor_sha256'] == record['raw_tensors'][1]['raw_tensor_sha256'], 'Original b2 differs')
                path = (hf/row['binding']['path']).resolve()
                native.need(path.is_relative_to(hf.resolve()) and path.stat().st_size == row['binding']['bytes'], 'Factor escaped HF or changed size')
                checked = vm.factor_header(path, row['tensors'], payload=payload)
                native.need(not payload or checked['sha256'] == row['binding']['sha256'], 'Serialized factor SHA differs')
                rows.append({**row, **checked})
    identity = lambda r: {k: v for k, v in r.items() if k not in ('stat', 'sha256', 'actual_payload_read_bytes')}
    key = lambda r: (r['expert'], r['block'], r['family'])
    native.need(len(rows) == 160 and len({key(r) for r in rows}) == 160
        and sorted([identity(r) for r in rows], key=key) == sorted(bank['exported_factors'], key=key),
        'Compiled exports differ from actual joined bank')
    return sorted(rows, key=key)


def references(plan, hf, payload=False):
    # Retain original failed-V1 + CPU repair + successful-V2 provenance.
    import e0_raw_bf16_video20_quality as quality
    old = native.read(native.bound(ROOT, plan['reference_protocol']))
    phases, parent_shas, protocols, _ = quality.generation_metadata(old, hf)
    native.need(old['source_cohort'] == plan['source_cohort'], 'Different reference cohort')
    result = {}
    for case in plan['cases']:
        ordinal = case['native_source_ordinal']
        row = next(v for v in old['videos'] if v['source'] == ordinal and v['mode'] == 'dense')
        generation = native.read(native.bound(hf, row['generation']))
        native.need(protocols[ordinal, 'dense']['models'] == plan['models']
            and protocols[ordinal, 'dense']['runtime_overlay'] == plan['runtime_overlay']
            and generation['protocol_sha256'] == parent_shas[ordinal, 'dense']
            and generation['image_sha256'] == row['image_sha256'] == case['image']['sha256']
            and generation['prompt_sha256'] == row['prompt_sha256'] == case['prompt_sha256']
            and generation['seed'] == 42 and generation['states'] == row['states']
            and generation['status'] == 'PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE'
            and generation['all160_FFN_covered'] is True and generation['completed_native_flow_steps'] == 4
            and phases[ordinal, 'dense']['generation_receipt'] == row['generation'], 'Dense source/noise/full160 lineage differs')
        core = native.read(native.bound(hf, row['core']))
        native.need(core['identity']['run_sha256'] == row['generation']['sha256']
            and core['candidate']['sha256'] == row['video']['sha256'], 'Actual Dense CORE lineage differs')
        if payload:
            native.bound(hf, row['video'])
            for item in row['states'].values(): native.bound(hf, item)
        result[ordinal] = {**row, 'protocol_sha256': parent_shas[ordinal, 'dense']}
    return result


def cpu_identity(plan, sha):
    return {'id': plan['id'], 'protocol_sha256': sha, 'status': 'PASS_FULL160_TEMPORAL_TWO_BANK_SHA_AND_DENSE_REFERENCE_BYTES',
            'source_closure_sha256': native.canonical(plan['repo_sources']), 'factor_bank': plan['factor_bank'],
            'reference_protocol': plan['reference_protocol'], 'factor_files': 160,
            'factor_payload_bytes': 160*vm.FACTOR_BYTES, 'Dense_references': 10,
            'numerical_imports': False, 'model_calls': 0, 'GPU_calls': 0}


def preflight(args):
    start = time.monotonic(); plan = native.read(args.protocol); sha = native.sha(args.protocol)
    authorize(plan, sha)
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 4, 'New bounded hidden-CUDA byte worker required')
    native.need(args.out is not None and not args.out.exists() and args.out.resolve().is_relative_to(args.hf_root.resolve()), 'Fresh CPU output required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'id': plan['id'], 'protocol_sha256': sha, 'status': 'STARTED_FIXED_BYTE_PROOF'}
    native.write(args.out, report)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU fixed300s exhausted')))
    signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        io.require_execution_master(plan, args.hf_root)
        for item in plan['repo_sources']: native.bound(ROOT, item)
        for case in plan['cases']: native.bound(args.hf_root, case['image'])
        native.model_metadata(plan)
        native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
        rows = factor_index(plan, args.hf_root, True)
        refs = references(plan, args.hf_root, True)
        native.need(not {'torch', 'numpy'} & set(sys.modules), 'Numerical import in byte proof')
        report.update(cpu_identity(plan, sha), factor_bindings=rows, factor_bindings_sha256=native.canonical(rows),
                      references_sha256=native.canonical(refs))
    except BaseException as error:
        report.update(status='FAILED_CPU_BYTE_PROOF_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-start; native.write(args.out, report)
    return report


def require_cpu(scope, plan, sha, hf):
    proof = scope['cpu_preflight']; report = native.read(native.bound(hf, proof['RUN']))
    vm.exited(native.read(native.bound(hf, proof['exit'])), proof['RUN'], sha, plan['id'])
    native.need(all(report.get(k) == v for k, v in cpu_identity(plan, sha).items())
        and report['factor_bindings_sha256'] == native.canonical(report['factor_bindings'])
        and report['wall_seconds'] <= CPU_RESOURCES['wall_seconds'], 'Exited CPU proof differs')
    for row in report['factor_bindings']:
        native.need(native.stat_record(hf/row['binding']['path']) == row['stat'], 'Factor changed after full SHA CPU proof')
    return proof, report['factor_bindings'], report['references_sha256']


def coordinates(row):
    e, b = row['expert'], row['block']
    native.need(type(e) is int and e in (0, 1) and type(b) is int and b in range(40)
        and row['family'] in MODES, 'Factor dispatch coordinate differs')
    return [(e, 2*e, b), (e, 2*e+1, b)]


def load_cpu_banks(torch, rows, hf, ledger):
    from safetensors.torch import load_file
    banks = {m: {} for m in MODES}
    for row in rows:
        path = hf/row['binding']['path']
        def load():
            native.need(native.stat_record(path) == row['stat'], 'Factor stat changed')
            result = load_file(str(path), device='cpu')
            native.need(set(result) == set(vm.SHAPES) and all(t.dtype == torch.bfloat16 and list(t.shape) == vm.SHAPES[k]
                and t.device.type == 'cpu' and t.is_contiguous() for k, t in result.items())
                and native.stat_record(path) == row['stat'], 'Loaded compiled factor differs')
            return result
        factors = ledger.call('setup', {'factor_CPU_file_loads': 1, 'factor_CPU_tensor_bytes': vm.FACTOR_BYTES}, load)
        for coordinate in coordinates(row):
            native.need(coordinate not in banks[row['family']], 'Duplicate factor coordinate')
            banks[row['family']][coordinate] = factors
    expected = {(e, s, b) for e in (0, 1) for s in (2*e, 2*e+1) for b in range(40)}
    native.need(all(set(b) == expected for b in banks.values()), 'All160 factor dispatch required')
    return banks


def to_gpu_bank(torch, bank, ledger, rid):
    seen, result = {}, {}
    def move():
        for coordinate, factors in bank.items():
            if id(factors) not in seen: seen[id(factors)] = {k: t.to('cuda') for k, t in factors.items()}
            result[coordinate] = seen[id(factors)]
        torch.cuda.synchronize()
        native.need(len(seen) == 80, 'Shared80 physical storage required')
        return result
    return ledger.call(rid, {'factor_GPU_family_preloads': 1, 'factor_GPU_H2D_tensors': 240,
        'factor_GPU_H2D_bytes': 80*vm.FACTOR_BYTES}, move)


def match_noise(phase, dense, ledger):
    for name in ('raw_noise_FP32', 'initial_noise'):
        ledger.call(phase['run_id'], {'matched_reference_noise_byte_checks': 1},
                    lambda name=name: native.need(vm.same_state(phase['states'][name], dense['states'][name]), 'Actual Dense/candidate noise bytes differ'))


def registration(plan, hf, refs):
    from ffnds_cohort_preflight import audit
    # The three-arm verification includes true reused Dense files; only two
    # candidate arms are permitted native run IDs in this new model scope.
    base = vm.registration(plan, hf)
    jobs = []
    for case in plan['cases']:
        candidates = [j for j in base['samples'] if j['source_id'] == case['source_id']]
        dense = refs[case['native_source_ordinal']]
        generation = native.read(native.bound(hf, dense['generation']))
        jobs.append({**candidates[0], 'variant': 'dense', 'run_id': generation['run_id'],
                     'asset_id': 'REUSED_DENSE:'+dense['generation']['sha256']})
        jobs.extend(candidates)
    result = {**base, 'samples': jobs, 'required_variants': ['dense', *MODES]}
    native.need(not audit(result)['errors'], 'Reused Dense/actual candidate cohort invalid')
    return result


def registration_record(registered, index, phase, hf, *, reused=False):
    job = registered['samples'][index]
    return {'schema': 'ffnds.registered.native.video.v1', 'status': 'COMPLETE_NATIVE_GENERATION_ONLY',
        'job': job, 'source': next(s for s in registered['sources'] if s['source_id'] == job['source_id']),
        'native_ffn_sites': 160, 'scheduler_generator_reset': True,
        'native_ffn_counters': phase['native_counter_delta'],
        'actual_initial_noise_file': str(hf/phase['states']['initial_noise']['path']),
        'video_path': phase['video']['path'], 'decoded_frames': 81, 'decoded_size': [832, 464], 'decoded_fps': 16,
        'immutable_generation_receipt': phase['generation_receipt'], 'numeric_path': 'BF16_ORIGINAL',
        'generation_origin': 'REUSED_ORIGINAL_DENSE_NOT_A_NEW_GENERATION' if reused else 'NEW_ACTUAL_CANDIDATE'}


def core_report(plan, phase, case, dense, hf, out, counter, ledger, setup_seconds):
    from ffnds_video_metrics import evaluate, validate as validate_core
    import cv2
    cv2.setNumThreads(RESOURCES['CPU_threads'])
    common = {'source_id': case['image']['sha256'], 'seed': 42, 'expected_frames': 81,
              'input_prompt_digest': case['prompt_sha256'], 'model_digest': native.canonical(plan['models'])}
    context = {'identity': {**common, 'protocol_sha256': phase['protocol_sha256'], 'run_sha256': phase['generation_receipt']['sha256']},
        'reference_identity': {**common, 'protocol_sha256': dense['protocol_sha256'], 'run_sha256': dense['generation']['sha256']},
        'cost': {'generation_wall_seconds': phase['generation_wall_seconds_including_encoders_decode_observers'],
                 'ffn_boundary_seconds': phase['FFN_boundary_CUDA_seconds'], 'setup_seconds': setup_seconds,
                 'peak_gpu_memory_mib': phase['peak_cuda_allocated_bytes']/(1 << 20)}}
    started = time.perf_counter()
    report = ledger.call(phase['run_id'], {'core_CPU_reports': 1}, evaluate,
        Path(phase['video']['path']), (hf/dense['video']['path']).resolve(), context)
    native.need(not validate_core(report), 'Whole81frame paired CORE invalid')
    path = out/phase['run_id']/'CORE_METRICS.json'; counter.write_json(path, report)
    phase.update(core_metrics=counter.binding(path, hf), CORE_CPU_wall_seconds=time.perf_counter()-started,
                 reused_Dense_reference=dense)


def run(args):
    started = time.monotonic(); plan = native.read(args.protocol); sha = native.sha(args.protocol)
    scope = authorize(plan, sha); proof, rows, refs_sha = require_cpu(scope, plan, sha, args.hf_root)
    native.need(not {'torch', 'numpy'} & set(sys.modules), 'Fresh numeric worker required')
    io.require_execution_master(plan, args.hf_root)
    for item in plan['repo_sources']: native.bound(ROOT, item)
    refs = references(plan, args.hf_root)
    native.need(native.canonical(refs) == refs_sha, 'Dense reference changed after CPU proof')
    native.model_metadata(plan)
    local, package_rows = native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    native.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions']
        and len(os.sched_getaffinity(0)) <= 4 and not any(os.environ.get(k) == '1'
        for k in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')), 'Runtime/CPU/precision differs')
    out = (args.hf_root/plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh HF output only, no retry')
    out.mkdir(parents=True); configure(plan)
    report = {'id': plan['id'], 'protocol_sha256': sha, 'part': plan['part'], 'numeric_path': 'BF16_ORIGINAL',
        'status': 'STARTED_MATCHED_REPRESENTATION_FULL160_VIDEO20_PART', 'automatic_retry': False,
        'CPU_preflight': proof, 'factor_bank': plan['factor_bank'], 'source_cohort': plan['source_cohort'],
        'reference_protocol': plan['reference_protocol'], 'design': DESIGN, 'phases': [], 'pairs': [],
        'official_quality': 'NOT_RUN', 'human_video_review': 'PENDING', 'quality_or_speed_acceptance': False,
        'new_Dense_generations': 0, 'new_fits': 0, 'system_timing_identity': DESIGN['timing']}
    monitor = None
    counter = io.WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    for rid in plan['run_ids']: counter.hold(str(counter.path(out/rid/'output.mp4')), 128 << 20)
    journal = counter.open_journal(out/'OPERATIONS.jsonl')
    persist = lambda: counter.write_json(out/'RUN.json', report)
    ledger = vm.Ledger(report.setdefault('operation_ledger', {}), lambda event: journal.write(json.dumps(event, separators=(',', ':'))+'\n'))
    meter = vm.FactorMeter(ledger); current = {'rid': None, 'skips': []}; persist()
    def emit(event):
        if event['event'] == 'original_W2_H2D_skipped':
            native.need(current['rid'] is not None and event['bytes'] == vm.ORIGINAL_W2_BYTES, 'Unregistered W2 skip')
            ledger.call(current['rid'], {'original_W2_H2D_skips': 1, 'original_W2_H2D_skipped_bytes': vm.ORIGINAL_W2_BYTES}, current['skips'].append, event)
        else: journal.write(json.dumps({'kind': 'runtime_dispatch', **event}, separators=(',', ':'))+'\n')
    def check():
        if monitor: monitor.check()
        counter.check()
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available >= RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes']
            and time.monotonic()-started <= RESOURCES['wall_seconds'], 'Host/disk/wall reserve exceeded')
        if 'torch' in sys.modules: native.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'GPU40GiB exceeded')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Fixed video2400s exhausted')))
    signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder, no_hif4 = CheckedPackageFinder(local, package_rows), native.NoHiF4()
    file_guard = io.single_file_bound(); file_guard.__enter__()
    try:
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(local), str(ROOT/'.runtime/code/LightX2V')]
        sys.meta_path[:0] = [no_hif4, finder]
        os.environ.update(DTYPE='BF16', SENSITIVE_LAYER_DTYPE='BF16', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                          OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        reservation, monitor, report['CUDA_reservation'] = vm.reserved_startup.reserve_before_assets(torch)
        native.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'One idle assigned A100 required')
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
        native.need(counter.holds.pop('frozen_metadata') == 0, 'Snapshot bytes did not close')
        with native.OriginalGuard(torch, plan, ledger) as guard:
            begun = time.perf_counter(); pipe = native.build_pipeline(plan); torch.cuda.synchronize()
            report.update(raw_model_setup_seconds=time.perf_counter()-begun, original_loader_paths=guard.loaded,
                          matrix_residency='ORIGINAL_RAW_CPU_MASTERS_AND_ALL_FOUR_GPU_BUFFERS_RETAINED_PLUS_CURRENT_SHARED80_FACTOR_BANK')
            begun = time.perf_counter(); banks = load_cpu_banks(torch, rows, args.hf_root, ledger)
            report['factor_CPU_bank_load_seconds'] = time.perf_counter()-begun
            report['factor_CPU_tensor_bytes_resident'] = 160*vm.FACTOR_BYTES
            registered = registration(plan, args.hf_root, refs)
            counter.write_json(snapshot/'registered_video_plan.json', registered)
            gpu_bank = {}
            with vm.FactorW2Runtime(pipe, guard, gpu_bank, factor_forward=vm.core.factor_forward,
                    meter=meter, synchronize=torch.cuda.synchronize,
                    record_stream=lambda t: t.record_stream(torch.cuda.current_stream()), emit=emit) as runtime:
                for case in plan['cases']:
                    pair = []
                    dense = refs[case['native_source_ordinal']]
                    old_phase = native.read(native.bound(args.hf_root, dense['generation']))
                    old_phase['generation_receipt'] = dense['generation']
                    dense_index = next(i for i, j in enumerate(registered['samples'])
                        if j['source_id'] == case['source_id'] and j['variant'] == 'dense')
                    folder = out/f'job{dense_index:06d}'; folder.mkdir()
                    ledger.call('setup', {'reused_Dense_registration_writes': 1}, counter.write_json,
                        folder/'RUN.json', registration_record(registered, dense_index, old_phase, args.hf_root, reused=True))
                    for position, mode in enumerate(phase_order(case)):
                        check(); authorize(plan, sha)
                        rid = run_id(case, mode); meter.rid = current['rid'] = rid; current['skips'] = []
                        torch.cuda.synchronize(); runtime.bank = {}; gpu_bank.clear(); del gpu_bank
                        torch.cuda.empty_cache(); before = torch.cuda.memory_allocated(); begun = time.perf_counter()
                        gpu_bank = to_gpu_bank(torch, banks[mode], ledger, rid)
                        runtime.bank = gpu_bank; runtime.set_mode('factor'); preload = time.perf_counter()-begun
                        phase = vm.native_phase(torch, pipe, guard, runtime, plan, case, mode, out, args.hf_root,
                            sha, report, persist, check, counter, {})
                        native.need(len(current['skips']) == 162, 'All original W2 H2D skips required')
                        phase.update(order_within_pair=position, current_factor_family_gpu_preload_seconds=preload,
                            current_factor_family_GPU_tensor_bytes=80*vm.FACTOR_BYTES, GPU_allocated_before_family_preload=before,
                            original_GPU_buffers_retained=True, skipped_original_W2_transfers=len(current['skips']),
                            skipped_original_W2_H2D_bytes=sum(x['bytes'] for x in current['skips']))
                        counter.write_json(out/rid/'W2_DISPATCH.json', {'events': current['skips'], 'mode': mode,
                            'actual_factor_GEMMs': 320, 'original_W2_loaded': False})
                        match_noise(phase, dense, ledger)
                        core_report(plan, phase, case, dense, args.hf_root, out, counter, ledger, report['raw_model_setup_seconds'])
                        index = next(i for i, j in enumerate(registered['samples']) if j['run_id'] == rid)
                        folder = out/f'job{index:06d}'; folder.mkdir()
                        def save_registration():
                            counter.write_json(folder/'RUN.json', registration_record(registered, index, phase, args.hf_root))
                        ledger.call(rid, {'source_registration_writes': 1}, save_registration)
                        counter.write_json(out/rid/'PHASE.json', phase); pair.append(rid); persist(); check()
                    report['pairs'].append({'native_source_ordinal': case['native_source_ordinal'], 'source_id': case['source_id'],
                        'stratum': case['stratum'], 'order': list(phase_order(case)), 'run_ids': pair,
                        'both_candidate_noises_byte_equal_actual_Dense': True})
                    persist()
                torch.cuda.synchronize(); runtime.bank = {}; gpu_bank.clear()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified = collect(registered, out); counter.write_json(out/'VERIFIED_GENERATIONS.json', verified)
            native.need(len(report['pairs']) == 10 and len(report['phases']) == 20 and not counter.holds, 'Incomplete actual full-depth part')
            report.update(status='COMPLETE_MATCHED_TEMPORAL_REPRESENTATION_TEN_SOURCES_TWENTY_FULL160_VIDEOS',
                verified_generations=counter.binding(out/'VERIFIED_GENERATIONS.json', args.hf_root),
                checked_transformers_imports=finder.origins, actual_full160_videos=20, actual_flow_steps=80,
                actual_FFN_positions=3200, actual_factor_W2_GEMMs=6400)
            check()
    except BaseException as error:
        report.update(status='FAILED_TEMPORAL_VIDEO_PART_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); monitor_error = None
        if monitor:
            try:
                monitor.close(); report['assigned_GPU_resources'] = monitor.report()
                native.need(not report['assigned_GPU_resources']['violations'], 'Final assigned GPU violation')
            except BaseException as error:
                monitor_error = error; report.update(status='FAILED_TEMPORAL_VIDEO_PART_PRESERVED_NO_RETRY', resource_error=str(error))
        for item in (finder, no_hif4):
            if item in sys.meta_path: sys.meta_path.remove(item)
        for rid in plan['run_ids']:
            path = counter.path(out/rid/'output.mp4')
            if path not in counter.files and path.exists():
                try: counter.observe_external(path, hold=str(path))
                except BaseException as error: counter.accounting_errors.append(str(error))
        report.update(wall_seconds=time.monotonic()-started, output_accounting_checkpoint_before_final_RUN=counter.summary(),
                      operation_journal=counter.binding(out/'OPERATIONS.jsonl', args.hf_root))
        persist(); journal.close(); file_guard.__exit__(None, None, None)
        if monitor_error is not None: raise monitor_error
    return report


def draft(part):
    scope_id(part)
    old = native.read(ROOT/TEMPLATE)
    inherit = ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions', 'local_transformers',
        'private_worker_config_sha256', 'execution_private_worker_config_sha256', 'execution_raw_master_CPU_proof', 'runtime_overlay')
    plan = {k: copy.deepcopy(old[k]) for k in inherit}
    cohort = native.read(ROOT/COHORT)
    cases = [c for c in cohort['sources'] if c['part'] == part]
    paths = {b['path'] for b in old['repo_sources']} | REQUIRED | {TEMPLATE}
    plan.update(id=scope_id(part), part=part, status='DRAFT_NOT_AUTHORIZED', assets_finalized=False,
        numeric_path='BF16_ORIGINAL', automatic_retry=False, resources=RESOURCES, CPU_preflight_resources=CPU_RESOURCES,
        limits=limits(part), design=DESIGN, prior_development_criteria=old['prior_development_criteria'],
        source_cohort=native.binding(ROOT/COHORT, ROOT), factor_bank=native.binding(ROOT/BANK, ROOT),
        reference_protocol=native.binding(ROOT/REFERENCE, ROOT), cases=cases,
        run_ids=[run_id(c, m) for c in cases for m in phase_order(c)], output_HF_relative=output_relative(part),
        repo_sources=[native.binding(ROOT/p, ROOT) for p in sorted(paths)],
        independent_review='NOT_PERFORMED_AGENT_SERVICE_USAGE_LIMIT', quality_or_speed_accepted=False)
    validate(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'preflight', 'run'))
    parser.add_argument('--part', type=int, choices=(0, 1)); parser.add_argument('--protocol', type=Path)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--local-root', type=Path); parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.action == 'draft':
        native.need(args.out is not None and not args.out.exists(), 'Fresh draft output required')
        native.write(args.out, draft(args.part))
    elif args.action == 'preflight': preflight(args)
    else: run(args)


if __name__ == '__main__': main()
