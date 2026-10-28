"""Fixed 20-image full-depth raw-BF16 Dense/shared80/step160 development videos.

Draft and byte preflight import only the standard library. Native arithmetic is
the closed original producer plus the independently reviewed W2 factor adapter.
There are no shadow Dense calls, online fits, QDQ, warmups or official metrics.
"""
import argparse
from contextlib import ExitStack
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:
    sys.path.insert(0, str(ROOT/'tools'))
import importlib.util
_spec = importlib.util.spec_from_file_location('_ffnds_video20_io', ROOT/'tools/e0_original_bf16_full_depth_collect_v2.py')
io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(io)
native = io.native  # Private namespace; never rebind the frozen public producer.
import raw_bf16_full80_spectrum as core
from raw_bf16_factor_w2_runtime import FactorW2Runtime, ORIGINAL_W2_BYTES
from raw_bf16_assigned_gpu_monitor_v4 import startup as reserved_startup

BASE_ID = 'E0_RAW_BF16_FULL160_VIDEO20_20261029'
COHORT = 'config/e0_raw_bf16_full160_video20_sources_20261029_draft.json'
ENVIRONMENT = 'config/e0_original_bf16_full_depth_collect_part0_20261029_v3.json'
FACTOR_PARENT = 'config/e0_raw_bf16_full80_spectrum_20261029_v1.json'
FACTOR_RUN_SHA = '83363b4278ad1e47cfef612e3229e46d309f00f36db0c34df5312fd4809d65af'
FACTOR_EXIT_SHA = 'c0909ab76731197abe850536a90f00f343b8bc1009fc1483d722fe2d5fb2c3b2'
MODES = ('dense', 'shared', 'step')
CAL = [240, 288, 304, 896, 897, 899]
SHAPES = {'left.weight': [960, 13824], 'right.weight': [5120, 960], 'right.bias': [5120]}
FACTOR_BYTES = sum(2*math.prod(s) for s in SHAPES.values())
FAMILY_BYTES = {'shared': 80*FACTOR_BYTES, 'step': 160*FACTOR_BYTES}
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 40 << 30,
    'host_available_min_bytes': 128 << 30, 'wall_seconds': 7200, 'outer_seconds': 7500,
    'new_HF_bytes': 10 << 30, 'free_disk_min_bytes': 12 << 30, 'session_max_GPUs': 2,
    'repository_GPU_policy': 'DYNAMIC_MAX_ACTUALLY_CALLABLE', 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 600, 'outer_seconds': 900,
    'factor_payload_bytes': 240*FACTOR_BYTES, 'raw_master_payload_bytes_rehashed': 0, 'automatic_retry': False}
DESIGN = {'image_clusters_total': 20, 'image_clusters_per_part': 10, 'seed': 42,
    'arms': list(MODES), 'rank': 960, 'beta': 1., 'factor_selection': 'ALL6_FIXED_EXPORTS_NO_LOO_OR_GAIN_SELECTION',
    'online': 'FULL_ORIGINAL_BF16_W1_TANH_GELU_THEN_TWO_BF16_FACTOR_GEMMS_ORIGINAL_B2_GATE_RESIDUAL',
    'W2_storage': 'SHARED80_VS_STEP160_EQUAL_MAC_DISTINCT_FACTOR_STORAGE',
    'state': 'FREE_FOUR_STEP_STUDENT_TRAJECTORIES_NO_SHADOW_DENSE',
    'order': 'CYCLIC_BY_GLOBAL_SOURCE_INDEX_DIV2_EACH_PART_BALANCED_4_3_3',
    'noise': 'NEW_REAL_FP32_RAW_AND_BF16_INITIAL_BYTE_EQUAL_ACROSS_EACH_TRIAD',
    'old_neutrals': 'DENSE_ORD894_AND256_ALL_TEN_STATE_BYTES',
    'residency': 'ALL240_CPU_BANKS_SINGLE_CURRENT_GPU_FAMILY_OR_NONE_DENSE_ORIGINAL_BUFFERS_RETAINED',
    'timing': 'SINGLE_OBSERVED_ORDER_NO_WARMUP_OR_REPEAT_NOT_ACCEPTED_FAIR_WARMED_SYS',
    'quality': '30_FULL81_CORE_CPU_REPORTS_PER_PART_OFFICIAL_AND_HUMAN_SEPARATE',
    'source_history': 'HISTORICALLY_EXPOSED_DEVELOPMENT_NOT_BLIND', 'automatic_retry': False}
PRIOR_CRITERIA = {'primary_unit': 'IMAGE_CLUSTER_EQUAL_WEIGHT_SEED42_DEVELOPMENT',
    'fixed_vector': ['imaging', 'aesthetic', 'I2V_subject', 'I2V_background', 'temporal_flicker',
        'motion_smoothness', 'continuous_motion', 'subject_action', 'camera_motion', 'semantic_relation', 'paired_perceptual'],
    'generation_scope_measures': 'Whole81frame CORE continuous motion, texture, paired spatial/temporal/motion and real costs',
    'missing_axes': 'NOT_RUN_NOT_IMPUTED_FROM_LATENT_SSE',
    'reporting': 'SOURCE_EQUAL_MEAN_MEDIAN_AND_ALL_SOURCE_TAILS_STRATIFIED_NO_RESULT_BASED_SOURCE_REMOVAL',
    'formal_acceptance': 'PENDING_SEPARATELY_REGISTERED_FULL_QUALITY_VECTOR_AND_USER_COMPLETE_VIDEO_REVIEW',
    'thresholds': 'NO_NEW_PASS_THRESHOLD_SELECTED_AFTER_OUTPUT_NO_METHOD_ACCEPTANCE_BY_THIS_SCOPE'}
REQUIRED = {'tools/e0_raw_bf16_full160_video20.py', 'tests/test_e0_raw_bf16_full160_video20.py',
    'tools/raw_bf16_factor_w2_runtime.py', 'tests/test_raw_bf16_factor_w2_runtime.py',
    'tools/raw_bf16_full80_spectrum.py', 'tools/raw_bf16_assigned_gpu_monitor_v4.py', COHORT}


def scope_id(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two parts required')
    return BASE_ID+'_PART'+str(part)


def run_id(case, mode):
    native.need(mode in MODES, 'Unknown video arm')
    return 'raw160_ord'+str(case['native_source_ordinal'])+'_seed42_'+mode


def phase_order(case):
    # Sources alternate parts. A division by two balances order within each part.
    offset = (case['index']//2) % 3
    return MODES[offset:]+MODES[:offset]


def output_relative(part):
    scope_id(part)
    return f'ffnds-derived/e0_raw_bf16_full160_video20_20261029/part{part}/attempt1/worker/results'


def limits(part=0):
    scope_id(part)
    result = {k: (v//2)*30 for k, v in native_original_limits.items()}
    result.update(original_expert_loads=2, raw_Default_MM_loads=812, buffer_Default_MM_loads=40,
        T5_raw_Default_MM_loads=168, T5_buffer_Default_MM_loads=7, core_CPU_reports=30,
        BF16_factor_GEMM=6400, original_W2_H2D_skips=3240, original_W2_H2D_skipped_bytes=3240*ORIGINAL_W2_BYTES,
        factor_CPU_file_loads=240, factor_CPU_tensor_bytes=240*FACTOR_BYTES,
        factor_GPU_family_preloads=20, factor_GPU_H2D_tensors=7200,
        factor_GPU_H2D_bytes=10*sum(FAMILY_BYTES.values()), prefetch_transfer_observations=4860,
        stage_CUDA_event_pairs=210,
        state_tensor_D2H=300, state_tensor_D2H_bytes=30*math.prod(native.VIDEO['latent_shape'])*(4+2+4*4+4*2),
        state_output_writes=300, video_geometry_checks=30, matched_triad_state_byte_checks=40,
        historical_state_byte_checks=20 if part == 0 else 0, source_registration_writes=30)
    return result


native_original_limits = copy.deepcopy(native.LIMITS)


def exited(value, run, digest, identifier):
    io.legacy.exited(value, run, digest, identifier)


def validate(plan, frozen=False):
    part = plan.get('part'); scope_id(part)
    native.need(plan.get('id') == scope_id(part) and plan.get('numeric_path') == 'BF16_ORIGINAL'
        and plan.get('limits') == limits(part) and plan.get('resources') == RESOURCES
        and plan.get('CPU_preflight_resources') == CPU_RESOURCES and plan.get('design') == DESIGN
        and plan.get('prior_development_criteria') == PRIOR_CRITERIA and plan.get('runtime_overlay') == native.recipe()
        and plan.get('automatic_retry') is False and plan.get('output_HF_relative') == output_relative(part), 'Fixed video20 scope differs')
    cohort = native.read(native.bound(ROOT, plan['source_cohort']))
    validate_cohort(cohort)
    native.need(plan['cases'] == [c for c in cohort['sources'] if c['part'] == part]
        and plan['run_ids'] == [run_id(c, m) for c in plan['cases'] for m in phase_order(c)], 'Fixed source/order/run IDs differ')
    paths = [b['path'] for b in plan['repo_sources']]
    native.need(REQUIRED <= set(paths) and len(paths) == len(set(paths)) and all(b['bytes'] < 128 << 20 for b in plan['repo_sources']), 'Code closure missing or unsafe snapshot size')
    if frozen:
        native.need(plan.get('assets_finalized') is True and plan.get('binding_status') == 'BOUND_READY_FOR_REGISTRATION', 'Video20 assets NOT_BOUND')


def validate_cohort(cohort):
    cases = cohort['sources']
    native.need(cohort.get('id') == 'E0_RAW_BF16_FULL160_VIDEO20_SOURCES_20261029'
        and cohort.get('excluded_fit_ordinals') == CAL and len(cases) == 20
        and [c['index'] for c in cases] == list(range(20))
        and len({c['native_source_ordinal'] for c in cases}) == len({c['image']['sha256'] for c in cases}) == 20
        and not set(CAL) & {c['native_source_ordinal'] for c in cases}, 'Fixed 20 disjoint image clusters differ')
    for c in cases:
        native.need(c['part'] == c['index'] % 2 and c['seed'] == 42
            and c['prior_history'] == 'HISTORICALLY_EXPOSED_DEVELOPMENT_NOT_BLIND'
            and hashlib.sha256(c['prompt'].encode()).hexdigest() == c['prompt_sha256']
            and c['conditioning_identity'] in ('ORIGINAL_PROMPT', 'NEW_CAMERA_PROMPT_NOT_HISTORICAL_CONDITIONING'), 'Conditioning/history identity differs')


def factor_header(path, tensors, *, payload=False, io_record=None):
    """One stream checks serialized SHA and all three BF16 tensor raw SHAs."""
    before = native.stat_record(path)
    if payload and io_record is not None:
        io_record['predebited_serialized_factor_bytes'] += before['bytes']
    def read(stream, amount):
        data = stream.read(amount)
        if payload and io_record is not None:
            io_record['actual_serialized_factor_bytes_read'] += len(data)
        return data
    with Path(path).open('rb') as stream:
        prefix = read(stream, 8); native.need(len(prefix) == 8, 'Truncated factor header')
        n = struct.unpack('<Q', prefix)[0]; native.need(0 < n <= 1 << 20, 'Factor header bound differs')
        raw = read(stream, n); native.need(len(raw) == n, 'Truncated factor JSON')
        header = json.loads(raw); native.need(set(header)-{'__metadata__'} == set(SHAPES) == set(tensors), 'Factor tensor keys differ')
        whole = hashlib.sha256(prefix+raw); cursor = 0
        for key in sorted(SHAPES, key=lambda k: header[k]['data_offsets'][0]):
            info = header[key]; a, b = info['data_offsets']; count = 2*math.prod(SHAPES[key])
            native.need(info['dtype'] == 'BF16' and info['shape'] == tensors[key]['shape'] == SHAPES[key]
                and tensors[key]['dtype'] == 'bfloat16' and a == cursor and b-a == count
                and io.hex_sha(tensors[key]['raw_tensor_sha256']), 'Factor BF16 shape/offset/hash differs')
            if payload:
                digest = hashlib.sha256(); remaining = count
                while remaining:
                    data = read(stream, min(8 << 20, remaining)); native.need(data, 'Truncated factor payload')
                    digest.update(data); whole.update(data); remaining -= len(data)
                native.need(digest.hexdigest() == tensors[key]['raw_tensor_sha256'], 'Actual factor tensor SHA differs')
            cursor = b
        native.need(before['bytes'] == 8+n+cursor and cursor == FACTOR_BYTES
            and native.stat_record(path) == before, 'Factor file geometry/stat changed')
    return {'stat': before, 'sha256': whole.hexdigest() if payload else None,
        'actual_payload_read_bytes': FACTOR_BYTES if payload else 0}


def factor_index(plan, hf, verify_payloads=False, io_record=None):
    proof = plan['factor_parent']
    protocol = native.read(native.bound(ROOT, proof['protocol']))
    run = native.read(native.bound(hf, proof['RUN']))
    exited(native.read(native.bound(hf, proof['launch_exit'])), proof['RUN'], proof['protocol']['sha256'], protocol['id'])
    native.need(proof['protocol']['path'] == FACTOR_PARENT and proof['RUN']['sha256'] == FACTOR_RUN_SHA
        and proof['launch_exit']['sha256'] == FACTOR_EXIT_SHA
        and run.get('protocol_sha256') == proof['protocol']['sha256']
        and run.get('status') == 'COMPLETE_RAW_FULL80_FIXED_SPECTRUM_ANCHOR_LOO_BF16_FACTOR_BANKS'
        and run.get('teacher_postwrite_neutrals_complete_before_fit') is True
        and len(run['functions']) == 80 and protocol['models'] == plan['models'], 'Closed same-raw-master all80 factor producer required')
    native.need({(f['expert'], f['block']) for f in run['functions']} == {(e, b) for e in ('high', 'low') for b in range(40)}, 'All80 physical functions required')
    rows = []
    for item in run['functions']:
        record = native.read(native.bound(hf, item['record']))
        native.need(record['expert'] == item['expert'] and record['block'] == item['block'], 'Factor function record coordinate differs')
        fits = [f for f in record['fits'] if f['held_source'] is None]
        native.need(len(fits) == 3 and {f['family'] for f in fits} == {'shared', 'step0', 'step1'}, 'Exactly three all6 exports per function required')
        for fit in fits:
            native.need(fit['training_sources'] == CAL and fit.get('export') is not None, 'Factor trained source split differs')
            export = fit['export']; path = (hf/export['binding']['path']).resolve()
            native.need(path.is_relative_to(hf.resolve()) and path.is_file()
                and path.stat().st_size == export['binding']['bytes'], 'Factor path/size differs')
            checked = factor_header(path, export['tensors'], payload=verify_payloads, io_record=io_record)
            native.need(not verify_payloads or checked['sha256'] == export['binding']['sha256'], 'Factor serialized file SHA differs')
            native.need(export['tensors']['right.bias']['raw_tensor_sha256'] == record['raw_tensors'][1]['raw_tensor_sha256'], 'Original b2 anchor differs')
            rows.append({'expert': 0 if item['expert'] == 'high' else 1, 'block': item['block'],
                'family': fit['family'], **export, **checked})
    native.need(len(rows) == 240 and sum(r['actual_payload_read_bytes'] for r in rows) == (240*FACTOR_BYTES if verify_payloads else 0), 'Factor bank byte count differs')
    return sorted(rows, key=lambda r: (r['expert'], r['block'], r['family']))


def history_controls(plan, hf):
    proof = plan['raw_parent']; old = native.read(native.bound(ROOT, proof['protocol']))
    run = native.read(native.bound(hf, proof['RUN']))
    exited(native.read(native.bound(hf, proof['exit'])), proof['RUN'], proof['protocol']['sha256'], old['id'])
    native.need(proof['RUN']['sha256'] == io.legacy.PARENT_RUN_SHA and proof['protocol']['sha256'] == io.legacy.PARENT_SHA
        and run.get('status') == 'COMPLETE_TWO_ORIGINAL_BF16_NATIVE_VIDEOS_CORE_QUALITY_PENDING', 'Original raw Dense parent differs')
    controls = {}
    cohort = native.read(native.bound(ROOT, plan['source_cohort']))
    for ordinal, index in ((894, 0), (256, 12)):
        case = next(c for c in cohort['sources'] if c['native_source_ordinal'] == ordinal)
        phase = next(p for p in run['phases'] if p['source_index'] == index)
        native.need(case['image']['sha256'] == phase['image_sha256'] and case['prompt_sha256'] == phase['prompt_sha256']
            and phase['seed'] == 42 and len(phase['states']) == 10, 'Original olddev conditioning/state identity differs')
        controls[ordinal] = phase['states']
    return controls


def cpu_expected(plan, digest):
    return {'id': plan['id'], 'protocol_sha256': digest,
        'status': 'PASS_RAW160_VIDEO20_FULL240_FACTOR_SHA_20_IMAGE_CURRENT_MASTER_STAT_NO_NUMERIC_IMPORTS',
        'part': plan['part'], 'sources_sha256': native.canonical(plan['repo_sources']),
        'source_cohort': plan['source_cohort'], 'factor_parent': plan['factor_parent'],
        'execution_raw_master_CPU_proof': plan['execution_raw_master_CPU_proof'],
        'factor_files': 240, 'factor_payload_bytes': 240*FACTOR_BYTES, 'source_images': 20,
        'raw_master_payload_bytes_rehashed': 0, 'all_CUDA_devices_hidden': True,
        'Torch_or_NumPy_imported': False, 'model_calls': 0, 'GPU_calls': 0, 'automatic_retry': False}


def preflight(args):
    started = time.monotonic(); plan = native.read(args.protocol); validate(plan, True)
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 4, 'Independent hidden-CUDA stdlib CPU proof required')
    result = cpu_expected(plan, native.sha(args.protocol))
    result.update(status='STARTED_FACTOR_AND_IMAGE_BYTE_PROOF_NO_RETRY', actual_input_IO={
        'predebited_serialized_factor_bytes': 0, 'actual_serialized_factor_bytes_read': 0})
    native.need(args.out is not None and not args.out.exists() and args.out.resolve().is_relative_to(args.hf_root.resolve()), 'Fresh confined CPU proof output required')
    args.out.mkdir(parents=True); native.write(args.out/'RUN.json', result)
    def timeout(*_): raise TimeoutError('Fixed video20 factor CPU proof wall600 reached; no retry')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(CPU_RESOURCES['wall_seconds'])
    try:
        io.require_execution_master(plan, args.hf_root)
        history_controls(plan, args.hf_root)
        for item in plan['repo_sources']:
            native.bound(ROOT, item)
        cohort = native.read(native.bound(ROOT, plan['source_cohort']))
        for case in cohort['sources']:
            native.bound(args.hf_root, case['image'])
        native.model_metadata(plan)
        native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
        rows = factor_index(plan, args.hf_root, True, result['actual_input_IO'])
        result.update(cpu_expected(plan, native.sha(args.protocol)))
        result.update(factor_bindings=rows, factor_bindings_sha256=native.canonical(rows))
        native.need(not {'torch', 'numpy'} & set(sys.modules), 'Numerical import during byte proof')
    except BaseException as error:
        result.update(status='FAILED_FACTOR_AND_IMAGE_BYTE_PROOF_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); result['wall_seconds'] = time.monotonic()-started
        native.write(args.out/'RUN.json', result)
    return result


def authorize(plan, digest):
    validate(plan, True); active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    policy = active.get('resource_scheduling_policy_20261028', {})
    native.need(active.get('user_continuation_authorization_20261027', {}).get('new_scope_registration_allowed') is True
        and policy.get('mode') == 'DYNAMIC_MAX_ACTUALLY_CALLABLE' and policy.get('global_gpu_card_cap') is None
        and active.get('current_session_resource_constraint_20261028', {}).get('max_simultaneously_assigned_idle_GPUs') == 2, 'Current authority differs')
    rows = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    native.need(len(rows) == 1 and rows[0].get('protocol_sha256') == digest
        and rows[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and rows[0].get('limits') == limits(plan['part']) and rows[0].get('resources') == RESOURCES
        and rows[0].get('run_ids') == plan['run_ids'] and rows[0].get('max_gpu_cards') == 1, 'Unique exact video20 part not registered')
    return rows[0]


def require_cpu(scope, plan, digest, hf):
    proof = scope.get('cpu_preflight'); native.need(isinstance(proof, dict) and {'RUN', 'launch_exit'} <= set(proof), 'Exited independent CPU proof NOT_BOUND')
    report = native.read(native.bound(hf, proof['RUN']))
    native.need(all(report.get(k) == v for k, v in cpu_expected(plan, digest).items())
        and report.get('factor_bindings_sha256') == native.canonical(report['factor_bindings'])
        and 0 <= report.get('wall_seconds', -1) <= CPU_RESOURCES['wall_seconds'], 'CPU proof identity differs')
    exited(native.read(native.bound(hf, proof['launch_exit'])), proof['RUN'], digest, plan['id'])
    rows = factor_index(plan, hf, False)
    native.need([{k: v for k, v in r.items() if k not in ('sha256', 'actual_payload_read_bytes')} for r in rows]
        == [{k: v for k, v in r.items() if k not in ('sha256', 'actual_payload_read_bytes')} for r in report['factor_bindings']], 'Factor bytes/stat changed after CPU proof')
    return proof, report['factor_bindings']


class Ledger(native.Ledger):
    """Serialize bookkeeping only, retaining original asynchronous prefetch."""
    def __init__(self, record, journal):
        super().__init__(record, journal); self.lock = threading.RLock()
    def call(self, rid, costs, fn, *args, **kwargs):
        with self.lock:
            native.need(rid in native.RUN_IDS or rid == 'setup', 'Unregistered video operation')
            native.need(costs and all(k in native.LIMITS and type(n) is int and n > 0
                and self.record['predebited'].get(k, 0)+n <= native.LIMITS[k] for k, n in costs.items()), 'Video20 operation budget exceeded')
            for k, n in costs.items(): self.record['predebited'][k] = self.record['predebited'].get(k, 0)+n
            self.journal({'kind': 'reserved_before_call', 'run_id': rid, 'costs': costs})
        value = fn(*args, **kwargs)
        with self.lock:
            for k, n in costs.items(): self.record['observed'][k] = self.record['observed'].get(k, 0)+n
            self.journal({'kind': 'observed_return', 'run_id': rid, 'costs': costs})
        return value


class FactorMeter:
    def __init__(self, ledger): self.ledger, self.rid = ledger, None
    def call(self, key, fn, *a, **kw):
        native.need(key == 'BF16_factor_GEMM' and self.rid is not None, 'Unregistered factor arithmetic')
        return self.ledger.call(self.rid, {key: 1}, fn, *a, **kw)


def bank_coordinates(row):
    expert, block, family = row['expert'], row['block'], row['family']
    native.need(expert in (0, 1) and type(expert) is int and type(block) is int and block in range(40), 'Factor coordinate differs')
    native.need(family in ('shared', 'step0', 'step1'), 'Factor family differs')
    steps = (2*expert, 2*expert+1) if family == 'shared' else (2*expert+int(family[-1]),)
    return [(expert, s, block) for s in steps]


def load_cpu_banks(torch, rows, hf, ledger):
    from safetensors.torch import load_file
    result = {'shared': {}, 'step': {}}
    for row in rows:
        path = hf/row['binding']['path']
        native.need(native.stat_record(path) == row['stat'], 'CPU bank changed after exited proof')
        def load():
            tensors = load_file(str(path), device='cpu')
            native.need(set(tensors) == set(SHAPES) and all(t.dtype == torch.bfloat16 and list(t.shape) == SHAPES[k]
                and t.device.type == 'cpu' and t.is_contiguous() for k, t in tensors.items())
                and native.stat_record(path) == row['stat'], 'Actual loaded bank layout/stat differs')
            return tensors
        tensors = ledger.call('setup', {'factor_CPU_file_loads': 1, 'factor_CPU_tensor_bytes': FACTOR_BYTES}, load)
        family = 'shared' if row['family'] == 'shared' else 'step'
        for coordinate in bank_coordinates(row):
            native.need(coordinate not in result[family], 'Duplicate factor runtime coordinate')
            result[family][coordinate] = tensors
    native.need(all(set(bank) == {(e, s, b) for e in (0, 1) for s in (2*e, 2*e+1) for b in range(40)} for bank in result.values()), 'Full160 factor dispatch missing')
    return result


def to_gpu_bank(torch, bank, mode, ledger, rid):
    seen, result = {}, {}
    def move():
        for coordinate, factors in bank.items():
            if id(factors) not in seen:
                seen[id(factors)] = {k: t.to('cuda') for k, t in factors.items()}
            result[coordinate] = seen[id(factors)]
        torch.cuda.synchronize()
        return result
    n = 80 if mode == 'shared' else 160
    return ledger.call(rid, {'factor_GPU_family_preloads': 1, 'factor_GPU_H2D_tensors': n*3,
        'factor_GPU_H2D_bytes': FAMILY_BYTES[mode]}, move)


def transfer_geometry(buffer, destination, block, runtime):
    """Actual copy_ source metadata under the frozen utils.load_state_dict path.

    Counts requested H2D/D2D tensor bytes, not measured PCIe traffic or time.
    Source scalar RMS placeholders remain GPU-to-GPU and are never read.
    """
    from lightx2v.common.ops.utils import resolve_block_name
    h2d = d2d = copies = w2 = 0
    def owners(value):
        yield value
        for child in list(getattr(value, '_parameters', {}).values())+list(getattr(value, '_modules', {}).values()):
            if child is not None: yield from owners(child)
    for owner in owners(buffer):
        if runtime.mode == 'factor' and id(owner) in runtime.objects:
            continue
        specs = list(getattr(owner, 'base_attrs', []))
        if type(owner).__name__ == 'DefaultTensor':
            specs.append((owner.tensor_name, 'tensor', False))
        for name, attr, _transpose in specs:
            key = resolve_block_name(name, block, None, owner.is_post_adapter)
            if key in destination and hasattr(owner, attr+'_cuda_buffer'):
                source, target = destination[key], getattr(owner, attr+'_cuda_buffer')
                native.need(source.shape == target.shape and source.dtype == target.dtype and target.device.type == 'cuda', 'Copy source/destination geometry differs')
                size = source.numel()*source.element_size(); copies += 1
                if source.device.type == 'cpu': h2d += size
                elif source.device.type == 'cuda': d2d += size
                else: raise ValueError('Unsupported original prefetch source device')
                if key in (f'blocks.{block}.ffn.2.weight', f'blocks.{block}.ffn.2.bias'): w2 += size
        for attr, name_attr in getattr(owner, 'lora_attrs', {}).items():
            key = resolve_block_name(getattr(owner, name_attr), block)
            if key in destination:
                source = destination[key]; copies += 1; size = source.numel()*source.element_size()
                if source.device.type == 'cpu': h2d += size
                elif source.device.type == 'cuda': d2d += size
                else: raise ValueError('Unsupported auxiliary prefetch source')
    native.need(w2 == (ORIGINAL_W2_BYTES if runtime.mode == 'dense' else 0), 'Original W2 prefetch byte identity differs')
    return {'requested_H2D_tensor_bytes': h2d, 'requested_D2D_tensor_bytes': d2d,
        'copy_calls': copies, 'original_W2_H2D_bytes': w2}


class LightTap(native.NativeTap):
    def __init__(self, *a, runtime, **kw):
        super().__init__(*a, **kw); self.runtime = runtime; self.transfers = []; self.ffn_sites = set(); self.stages = []
    def mm_call(self, mm, value, meta, original):
        position = self.block
        result = super().mm_call(mm, value, meta, original)
        if position is not None:
            self.shapes[-1]['block'] = position[1]
            if self.mapping[id(mm)][2] == 'ffn.2': self.ffn_sites.add((self.step(position[0]), position[1]))
        return result
    def install(self):
        super().install()  # Mandatory: resets BOTH expert first buffers each phase.
        for owner, name in ((self.runner, 'run_input_encoder'), (self.runner, 'run_vae_encoder'),
                (self.runner, 'run_vae_decoder'), (self.runner.model, 'infer')):
            old = getattr(owner, name)
            def timed(*a, _old=old, _name=name, **kw):
                def measured():
                    torch = self.guard.torch
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    record = {'stage': _name, 'step': self.step() if _name == 'infer' else None,
                        'start': start, 'end': end}; self.stages.append(record)
                    begun = time.perf_counter(); start.record()
                    try: return _old(*a, **kw)
                    finally:
                        end.record(); record['wall_seconds'] = time.perf_counter()-begun
                return self.guard.ledger.call(self.rid, {'stage_CUDA_event_pairs': 1}, measured)
            self.stack.enter_context(native.patch_method(owner, name, timed))
        for expert, model in enumerate(self.runner.model.model):
            for buffer in model.transformer_infer.offload_manager.cuda_buffers:
                old = buffer.load_state_dict
                def load(destination, block_index, adapter_block_index=None, _old=old, _buffer=buffer, _expert=expert):
                    native.need(adapter_block_index is None, 'Post adapter outside scope')
                    def observed():
                        observed_at = time.perf_counter()
                        geometry = transfer_geometry(_buffer, destination, block_index, self.runtime)
                        started = time.perf_counter(); result = _old(destination, block_index, adapter_block_index)
                        self.transfers.append({'expert': _expert, 'step': self.step(_expert), 'block': block_index,
                            **geometry, 'enqueue_wall_seconds': time.perf_counter()-started,
                            'metadata_observer_wall_seconds': started-observed_at})
                        return result
                    return self.guard.ledger.call(self.rid, {'prefetch_transfer_observations': 1}, observed)
                self.stack.enter_context(native.patch_method(buffer, 'load_state_dict', load))
    def finish(self):
        super().finish()
        native.need(self.ffn_sites == {(s, b) for s in range(4) for b in range(40)} and len(self.transfers) == 162,
            'Genuine all160 FFN coverage/prefetch observations incomplete')
        native.need(len(self.stages) == 7, 'Encoder/decode/four-flow stage observations incomplete')


def same_state(a, b):
    return all(a[k] == b[k] for k in ('sha256', 'bytes', 'raw_tensor_sha256', 'shape', 'dtype'))


def verify_triad(phases, case, ledger=None):
    native.need(len(phases) == 3 and {p['mode'] for p in phases} == set(MODES)
        and all(p['native_source_ordinal'] == case['native_source_ordinal'] and p['seed'] == 42
            and p['completed_native_flow_steps'] == 4 for p in phases), 'Complete same-source triad required')
    dense = next(p for p in phases if p['mode'] == 'dense')
    for phase in phases:
        if phase['mode'] == 'dense': continue
        for name in ('raw_noise_FP32', 'initial_noise'):
            def check(): native.need(same_state(phase['states'][name], dense['states'][name]), 'Actual paired raw/initial noise bytes differ')
            if ledger: ledger.call(phase['run_id'], {'matched_triad_state_byte_checks': 1}, check)
            else: check()
    return {'raw_FP32_and_initial_BF16_same_bytes_all_arms': True, 'matched_source_ordinal': case['native_source_ordinal']}


def native_phase(torch, pipe, guard, runtime, plan, case, mode, out, hf, digest, report, persist, check, counter, controls):
    from safetensors.torch import save_file
    from ub1_consumer_runtime import tensor_sha
    from run_ub1_consumer_videos import inspect_video
    rid = run_id(case, mode); folder = out/rid; folder.mkdir()
    phase = {'id': plan['id'], 'run_id': rid, 'mode': mode, 'numeric_path': 'BF16_ORIGINAL',
        'native_source_ordinal': case['native_source_ordinal'], 'source_id': case['source_id'], 'source_index': case['index'],
        'image_sha256': case['image']['sha256'], 'prompt_sha256': case['prompt_sha256'], 'conditioning_id': case['source_id']+':'+case['prompt_sha256'],
        'seed': 42, 'protocol_sha256': digest, 'source_exposure': case['prior_history'], 'states': {},
        'status': 'STARTED_NO_RETRY', 'quality_or_speed_acceptance': False}
    report['phases'].append(phase); persist()
    observer = {'state_save_wall_seconds': 0., 'state_D2H_wall_seconds': 0.}
    def save(value, name, key):
        def output():
            begun = time.perf_counter()
            expected = torch.float32 if key in ('raw_FP32_noise_files', 'native_FP32_flow_files') else torch.bfloat16
            native.need(name not in phase['states'] and value.dtype == expected and list(value.shape) == native.VIDEO['latent_shape'], 'Native state shape/dtype/order differs')
            path = folder/(name+'.safetensors'); counter.reserve_payload(value.numel()*value.element_size())
            def d2h():
                at = time.perf_counter(); cpu = value.detach().cpu().contiguous(); observer['state_D2H_wall_seconds'] += time.perf_counter()-at
                native.need(bool(torch.isfinite(cpu).all()), 'Nonfinite native state'); return cpu
            cpu = guard.ledger.call(rid, {'state_tensor_D2H': 1, 'state_tensor_D2H_bytes': value.numel()*value.element_size()}, d2h)
            try:
                guard.ledger.call(rid, {'state_output_writes': 1}, save_file,
                    {'flow' if key == 'native_FP32_flow_files' else 'latents': cpu}, str(path))
            finally:
                if path.exists(): counter.observe_external(path)
            item = {'root': 'hf', **counter.binding(path, hf), 'raw_tensor_sha256': tensor_sha(cpu),
                'shape': list(cpu.shape), 'dtype': str(cpu.dtype).removeprefix('torch.')}
            phase['states'][name] = item
            if mode == 'dense' and case['native_source_ordinal'] in controls:
                guard.ledger.call(rid, {'historical_state_byte_checks': 1},
                    lambda: native.need(same_state(item, controls[case['native_source_ordinal']][name]), 'Historical raw Dense state byte neutral failed'))
            observer['state_save_wall_seconds'] += time.perf_counter()-begun
            counter.write_json(folder/'PROGRESS.json', {'run_id': rid, 'states': phase['states'], 'status': phase['status']})
            return item
        return guard.ledger.call(rid, {key: 1}, output)
    pipe.runner.scheduler.generator = None
    tap = LightTap(pipe, guard, rid, save, check, runtime=runtime)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    try:
        def generate():
            with tap, torch.inference_mode():
                pipe.generate(seed=42, image_path=str((hf/case['image']['path']).resolve()), prompt=case['prompt'],
                    negative_prompt='', save_result_path=str(folder/'output.mp4'))
            torch.cuda.synchronize(); tap.finish()
        guard.ledger.call(rid, {'video_generations': 1}, generate)
        inspected = guard.ledger.call(rid, {'video_geometry_checks': 1}, inspect_video, folder/'output.mp4', native.VIDEO)
        counter.observe_external(folder/'output.mp4', hold=str(counter.path(folder/'output.mp4')), known=inspected)
        counter.write_json(folder/'NATIVE_OPERANDS_AND_TRANSFERS.json', {'all160_FFN_sites': sorted(tap.ffn_sites),
            'Default_MM_actual_operand_metadata': tap.shapes, 'prefetch': tap.transfers,
            'FFN_boundary_CUDA_by_site': [{'step': i//40, 'block': i%40, 'seconds': a.elapsed_time(b)/1000}
                for i, (a, b) in enumerate(tap.events)],
            'transfer_byte_identity': 'SRC_RESOLVED_ACTUAL_COPY_SOURCE_METADATA_NOT_HARDWARE_BUS_TRAFFIC'})
        phase.update(status='PASS_FULL_DEPTH_GENERATION_NOT_QUALITY_ACCEPTANCE', completed_native_flow_steps=4,
            scheduler_generator_reset=True, timetable=tap.timetable, all160_FFN_covered=True,
            native_counter_delta={'linear_forward_calls': len(tap.lines), 'activation_qdq_calls': 0, 'weight_quantizations': 0},
            native_operand_transfer_evidence=counter.binding(folder/'NATIVE_OPERANDS_AND_TRANSFERS.json', hf),
            initial_noise=phase['states']['initial_noise'], raw_FP32_noise=phase['states']['raw_noise_FP32'], video=inspected,
            generation_wall_seconds_including_encoders_decode_observers=time.perf_counter()-started,
            FFN_boundary_CUDA_seconds=sum(a.elapsed_time(b) for a, b in tap.events)/1000,
            stages=[{'stage': s['stage'], 'step': s['step'], 'wall_seconds': s['wall_seconds'],
                'current_stream_CUDA_elapsed_seconds': s['start'].elapsed_time(s['end'])/1000} for s in tap.stages],
            observer_cost=observer, prefetch_enqueue_wall_seconds=sum(x['enqueue_wall_seconds'] for x in tap.transfers),
            requested_prefetch_H2D_bytes=sum(x['requested_H2D_tensor_bytes'] for x in tap.transfers),
            requested_prefetch_D2D_bytes=sum(x['requested_D2D_tensor_bytes'] for x in tap.transfers),
            original_W2_H2D_bytes=sum(x['original_W2_H2D_bytes'] for x in tap.transfers),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            matched_olddev_all_ten_states=(True if mode == 'dense' and case['native_source_ordinal'] in controls else 'NOT_APPLICABLE'))
        native.need(len(phase['states']) == 10 and phase['peak_cuda_reserved_bytes'] <= RESOURCES['GPU_reserved_max_bytes'], 'Incomplete state capture/GPU bound')
        counter.write_json(folder/'GENERATION.json', phase); phase['generation_receipt'] = counter.binding(folder/'GENERATION.json', hf)
        return phase
    except BaseException as error:
        phase.update(status='FAILED_PHASE_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        counter.write_json(folder/'PHASE.json', phase); persist()


def registration(plan, hf):
    from ffnds_cohort_preflight import audit
    sources, jobs = [], []
    for c in plan['cases']:
        condition = c['source_id']+':'+c['prompt_sha256']
        sources.append({'source_id': c['source_id'], 'image_id': c['image']['sha256'],
            'prompt_id': c['prompt_sha256'], 'prompt': c['prompt'], 'image_path': str((hf/c['image']['path']).resolve()),
            'image_name': Path(c['image']['path']).name, 'conditioning_id': condition, 'role': 'DEVELOPMENT',
            'exposure_history': {'fitting': False, 'method_selection': True, 'prior_evaluation': True}})
        for mode in phase_order(c):
            jobs.append({'source_id': c['source_id'], 'variant': mode, 'sample_index': 0, 'seed': 42,
                'run_id': run_id(c, mode), 'asset_id': run_id(c, mode)+':new_actual_video',
                'initial_noise_id': 'raw160_ord'+str(c['native_source_ordinal'])+'_shared_real_seed42',
                'conditioning_id': condition, 'scheduler_id': 'Wan22StepDistill_shift5_1000_750_500_250',
                'frames': 81, 'fps': 16, 'width': 832, 'height': 464})
    result = {'schema': 'ffnds.cohort.v1', 'phase': 'REGISTERED_REAL_GENERATION_PLAN',
        'mode': 'CUSTOM_PAIRED', 'purpose': 'DEVELOPMENT', 'sources': sources, 'samples': jobs,
        'required_variants': list(MODES), 'expected_samples_per_source_per_variant': 1,
        'source_bank_id': plan['source_cohort']['sha256'], 'native_protocol_id': plan['id']}
    native.need(not audit(result)['errors'], 'Registered paired triad manifest invalid')
    return result


def core_reports(plan, phases, case, hf, out, counter, ledger, setup_seconds):
    from ffnds_video_metrics import evaluate, validate as validate_core
    import cv2
    cv2.setNumThreads(RESOURCES['CPU_threads'])
    dense = next(p for p in phases if p['mode'] == 'dense')
    results = []
    for phase in phases:
        identity = {'source_id': case['image']['sha256'], 'seed': 42, 'expected_frames': 81,
            'input_prompt_digest': case['prompt_sha256'], 'model_digest': native.canonical(plan['models']),
            'protocol_sha256': phase['protocol_sha256'], 'run_sha256': phase['generation_receipt']['sha256']}
        cost = {'generation_wall_seconds': phase['generation_wall_seconds_including_encoders_decode_observers'],
            'ffn_boundary_seconds': phase['FFN_boundary_CUDA_seconds'], 'setup_seconds': setup_seconds,
            'peak_gpu_memory_mib': phase['peak_cuda_allocated_bytes']/(1 << 20)}
        begun = time.perf_counter()
        core_report = ledger.call(phase['run_id'], {'core_CPU_reports': 1}, evaluate,
            Path(phase['video']['path']), None if phase['mode'] == 'dense' else Path(dense['video']['path']),
            {'identity': identity, 'cost': cost})
        native.need(not validate_core(core_report), 'Whole81frame CORE report invalid')
        path = out/phase['run_id']/'CORE_METRICS.json'; counter.write_json(path, core_report)
        phase['core_metrics'] = counter.binding(path, hf); phase['CORE_CPU_wall_seconds'] = time.perf_counter()-begun
        results.append({'mode': phase['mode'], 'binding': phase['core_metrics']})
    return results


def run(args):
    started = time.monotonic(); plan = native.read(args.protocol); digest = native.sha(args.protocol)
    scope = authorize(plan, digest); proof, factor_rows = require_cpu(scope, plan, digest, args.hf_root)
    native.need(not {'torch', 'numpy'} & set(sys.modules), 'Fresh numerical worker required')
    io.require_execution_master(plan, args.hf_root)
    controls = history_controls(plan, args.hf_root)
    for item in plan['repo_sources']: native.bound(ROOT, item)
    native.model_metadata(plan)
    local, package_rows = native.verify_local_package(plan, {'repo': ROOT, 'hf': args.hf_root}, args.local_root)
    native.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions']
        and len(os.sched_getaffinity(0)) <= 4 and not any(os.environ.get(k) == '1'
            for k in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')), 'Runtime/CPU/precision differs')
    out = (args.hf_root/plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined output required; no retry')
    out.mkdir(parents=True)
    report = {'id': plan['id'], 'protocol_sha256': digest, 'part': plan['part'], 'numeric_path': 'BF16_ORIGINAL',
        'status': 'STARTED_FIXED_FULL160_VIDEO20_PART_NO_RETRY', 'automatic_retry': False,
        'CPU_preflight': proof, 'factor_parent': plan['factor_parent'], 'source_cohort': plan['source_cohort'],
        'design': DESIGN, 'prior_development_criteria': PRIOR_CRITERIA, 'phases': [], 'triads': [],
        'independent_image_clusters_this_part': 10, 'full20_cohort_status': 'REQUIRES_BOTH_ACTUAL_PART_EXIT0',
        'official_quality': 'NOT_RUN', 'human_video_review': 'PENDING', 'quality_or_speed_acceptance': False,
        'system_timing_identity': 'ONE_OBSERVED_ORDER_NO_WARMUPS_NO_REPEATS_INCLUDES_OBSERVERS_NOT_ACCEPTED_WARMED_SYS'}
    native.ID, native.LIMITS, native.RUN_IDS, native.RESOURCE = plan['id'], limits(plan['part']), tuple(plan['run_ids']), RESOURCES
    monitor = None
    counter = io.WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    # All30 opaque writers have hard128MiB single-file limits and prior holds.
    for rid in plan['run_ids']:
        counter.hold(str(counter.path(out/rid/'output.mp4')), 128 << 20)
    journal = counter.open_journal(out/'OPERATIONS.jsonl')
    persist = lambda: counter.write_json(out/'RUN.json', report)
    ledger = Ledger(report.setdefault('operation_ledger', {}), lambda event: journal.write(json.dumps(event, separators=(',', ':'))+'\n'))
    meter = FactorMeter(ledger); persist()
    current = {'rid': None, 'skips': []}
    def emit(event):
        if event['event'] == 'original_W2_H2D_skipped':
            native.need(current['rid'] is not None and event['bytes'] == ORIGINAL_W2_BYTES, 'W2 skip outside registered phase')
            ledger.call(current['rid'], {'original_W2_H2D_skips': 1, 'original_W2_H2D_skipped_bytes': ORIGINAL_W2_BYTES},
                current['skips'].append, event)
        else:
            journal.write(json.dumps({'kind': 'runtime_dispatch', **event}, separators=(',', ':'))+'\n')
    def check():
        if monitor: monitor.check()
        counter.check()
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(available >= RESOURCES['host_available_min_bytes'] and shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes'], 'Host/disk reserve exceeded')
        native.need(time.monotonic()-started <= RESOURCES['wall_seconds'], 'Fixed video20 wall exhausted')
        if 'torch' in sys.modules:
            native.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'GPU40GiB reserved bound exceeded')
    def timeout(*_): raise TimeoutError('Fixed full160 video20 wall7200 exhausted; no retry')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCES['wall_seconds'])
    from e0_b_final_block_capture_local_package import CheckedPackageFinder
    finder, no_hif4 = CheckedPackageFinder(local, package_rows), native.NoHiF4()
    file_guard = io.single_file_bound()
    try:
        file_guard.__enter__()
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(local), str(ROOT/'.runtime/code/LightX2V')]
        sys.meta_path[:0] = [no_hif4, finder]
        os.environ.update(DTYPE='BF16', SENSITIVE_LAYER_DTYPE='BF16', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        reservation, monitor, report['CUDA_reservation'] = reserved_startup.reserve_before_assets(torch)
        native.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'One assigned A100 required')
        torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        report['runtime_precision'] = {'TF32': False, 'FP32_matmul': 'highest', 'BF16_reduced_precision_reduction': True}
        check()
        snapshot = out/'EXECUTED_SOURCES'; snapshot.mkdir()
        copies = [(ROOT/b['path'], snapshot/b['path'], b) for b in plan['repo_sources']]
        copies.append((args.protocol, snapshot/'protocol.json', native.binding(args.protocol, ROOT)))
        counter.hold('frozen_metadata', sum(b['bytes'] for _, _, b in copies))
        at = time.perf_counter()
        for source, dest, binding in copies:
            dest.parent.mkdir(parents=True, exist_ok=True); counter.copy_bound(source, dest, binding)
        native.need(counter.holds.pop('frozen_metadata') == 0, 'Snapshot bytes did not close')
        report['source_snapshot_wall_seconds'] = time.perf_counter()-at
        with native.OriginalGuard(torch, plan, ledger) as guard:
            at = time.perf_counter(); pipe = native.build_pipeline(plan); torch.cuda.synchronize()
            report.update(raw_model_setup_seconds=time.perf_counter()-at, original_loader_paths=guard.loaded,
                matrix_residency='Original raw CPU masters and four original CUDA block buffers retained in all modes; only current candidate family additionally GPU resident')
            at = time.perf_counter(); cpu_banks = load_cpu_banks(torch, factor_rows, args.hf_root, ledger)
            report['factor_CPU_bank_load_seconds'] = time.perf_counter()-at
            report['factor_CPU_tensor_bytes_resident'] = 240*FACTOR_BYTES
            registered = registration(plan, args.hf_root); counter.write_json(snapshot/'registered_video_plan.json', registered)
            gpu_bank = {}
            with FactorW2Runtime(pipe, guard, gpu_bank, factor_forward=core.factor_forward, meter=meter,
                    synchronize=torch.cuda.synchronize, record_stream=lambda t: t.record_stream(torch.cuda.current_stream()), emit=emit) as runtime:
                for case in plan['cases']:
                    triad = []
                    for position, mode in enumerate(phase_order(case)):
                        check(); authorize(plan, digest)
                        rid = run_id(case, mode); meter.rid = current['rid'] = rid; current['skips'] = []
                        # Synchronize before dropping refs. Runtime retains no old bank.
                        torch.cuda.synchronize(); runtime.bank = {}; gpu_bank.clear(); del gpu_bank
                        torch.cuda.empty_cache()
                        before = torch.cuda.memory_allocated(); load_at = time.perf_counter()
                        gpu_bank = {} if mode == 'dense' else to_gpu_bank(torch, cpu_banks[mode], mode, ledger, rid)
                        runtime.bank = gpu_bank; runtime.set_mode('dense' if mode == 'dense' else 'factor')
                        preload_seconds = time.perf_counter()-load_at
                        phase = native_phase(torch, pipe, guard, runtime, plan, case, mode, out, args.hf_root,
                            digest, report, persist, check, counter, controls)
                        native.need(len(current['skips']) == (0 if mode == 'dense' else 162), 'Original W2 skip coverage differs')
                        phase.update(order_within_triad=position, nominal_cache_state='SINGLE_ORDER_OBSERVED_NOT_CONTROLLED_COLD_OR_WARM',
                            current_factor_family_gpu_preload_seconds=preload_seconds,
                            current_factor_family_GPU_tensor_bytes=0 if mode == 'dense' else FAMILY_BYTES[mode],
                            GPU_allocated_before_family_preload=before, original_GPU_buffers_retained=True,
                            skipped_original_W2_transfers=len(current['skips']), skipped_original_W2_H2D_bytes=sum(x['bytes'] for x in current['skips']))
                        counter.write_json(out/rid/'W2_DISPATCH.json', {'events': current['skips'], 'mode': mode,
                            'actual_factor_GEMMs': 0 if mode == 'dense' else 320, 'original_W2_loaded': mode == 'dense'})
                        triad.append(phase); persist(); check()
                    checks = verify_triad(triad, case, ledger)
                    begun = time.perf_counter()
                    reports = core_reports(plan, triad, case, args.hf_root, out, counter, ledger, report['raw_model_setup_seconds'])
                    report['triads'].append({'native_source_ordinal': case['native_source_ordinal'], 'source_id': case['source_id'],
                        'stratum': case['stratum'], 'image_sha256': case['image']['sha256'], 'order': list(phase_order(case)),
                        **checks, 'CORE': reports, 'CORE_triad_CPU_wall_seconds': time.perf_counter()-begun})
                    for phase in triad:
                        index = plan['run_ids'].index(phase['run_id']); folder = out/f'job{index:06d}'; folder.mkdir()
                        def save_registration():
                            counter.write_json(folder/'RUN.json', {'schema': 'ffnds.registered.native.video.v1',
                                'status': 'COMPLETE_NATIVE_GENERATION_ONLY', 'job': registered['samples'][index],
                                'source': next(s for s in registered['sources'] if s['source_id'] == case['source_id']),
                                'native_ffn_sites': 160, 'scheduler_generator_reset': True,
                                'native_ffn_counters': phase['native_counter_delta'],
                                'actual_initial_noise_file': str(args.hf_root/phase['states']['initial_noise']['path']),
                                'video_path': phase['video']['path'], 'decoded_frames': 81, 'decoded_size': [832, 464], 'decoded_fps': 16,
                                'immutable_generation_receipt': phase['generation_receipt'], 'numeric_path': 'BF16_ORIGINAL'})
                        ledger.call(phase['run_id'], {'source_registration_writes': 1}, save_registration)
                        counter.write_json(out/phase['run_id']/'PHASE.json', phase)
                    persist(); check()
                torch.cuda.synchronize(); runtime.bank = {}; gpu_bank.clear()
            ledger.finish()
            from ffnds_stage_registered_i2v import collect
            verified = collect(registered, out); counter.write_json(out/'VERIFIED_GENERATIONS.json', verified)
            native.need(len(report['triads']) == 10 and len(report['phases']) == 30 and not counter.holds, 'Incomplete full-depth part')
            report.update(status='COMPLETE_RAW_BF16_FULL160_VIDEO20_PART_TEN_TRIADS',
                verified_generations=counter.binding(out/'VERIFIED_GENERATIONS.json', args.hf_root),
                checked_transformers_imports=finder.origins, actual_full160_videos=30, actual_flow_steps=120,
                actual_FFN_positions=4800, actual_factor_W2_GEMMs=6400,
                all_local_olddev_neutrals_passed=all(p['matched_olddev_all_ten_states'] is True
                    for p in report['phases'] if p['mode'] == 'dense' and p['native_source_ordinal'] in controls))
            check()
    except BaseException as error:
        report.update(status='FAILED_FULL160_VIDEO20_PART_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); monitor_error = None
        if monitor:
            try:
                monitor.close(); report['assigned_GPU_resources'] = monitor.report()
                native.need(not report['assigned_GPU_resources'].get('violations'), 'Final assigned GPU monitor violation')
            except BaseException as error:
                monitor_error = error; report.update(status='FAILED_FULL160_VIDEO20_PART_PRESERVED_NO_RETRY', resource_error=str(error))
        for finder_item in (finder, no_hif4):
            if finder_item in sys.meta_path: sys.meta_path.remove(finder_item)
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


def draft(hf, part):
    scope_id(part)
    environment = native.read(ROOT/ENVIRONMENT)
    inherited = ('models', 'model_view_root', 'model_view_metadata', 'runtime_versions', 'local_transformers',
        'private_worker_config_sha256', 'execution_private_worker_config_sha256', 'execution_raw_master_CPU_proof', 'runtime_overlay')
    plan = {k: copy.deepcopy(environment[k]) for k in inherited}
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    scope = next(s for s in active['experiment_authority']['current_user_authorized_scopes'] if s['id'] == 'E0_RAW_BF16_FULL80_R960_SHARED_STEP_SPECTRUM_20261029')
    cohort_binding = native.binding(ROOT/COHORT, ROOT); cohort = native.read(ROOT/COHORT); validate_cohort(cohort)
    cases = [c for c in cohort['sources'] if c['part'] == part]
    plan.update(id=scope_id(part), part=part, status='DRAFT_NOT_AUTHORIZED', numeric_path='BF16_ORIGINAL',
        assets_finalized=False, binding_status='BOUND_ASSETS_REVIEW_AND_FREEZE_PENDING', automatic_retry=False,
        design=DESIGN, prior_development_criteria=PRIOR_CRITERIA, resources=RESOURCES, CPU_preflight_resources=CPU_RESOURCES,
        limits=limits(part), source_cohort=cohort_binding, cases=cases, run_ids=[run_id(c, m) for c in cases for m in phase_order(c)],
        output_HF_relative=output_relative(part), raw_parent=environment['parent'],
        factor_parent={'protocol': native.binding(ROOT/FACTOR_PARENT, ROOT), 'RUN': scope['actual_RUN'], 'launch_exit': scope['actual_exit_receipt']},
        independent_review='PENDING', coordinator_review='PENDING', official_quality='NOT_RUN', human_review='PENDING',
        quality_or_speed_accepted=False, repository_GPU_policy_global_cap=None,
        state_payload_bytes=limits(part)['state_tensor_D2H_bytes'],
        video_single_file_hard_limit_bytes=128 << 20, output_accounting='INCREMENTAL_ACTUAL_WRITES_NO_RECURSIVE_HOT_LOOP',
        factor_tensor_payload_bytes_cpu=240*FACTOR_BYTES, factor_GPU_bytes_by_family=FAMILY_BYTES,
        all20_cluster_completion='TWO_INDEPENDENT_PART_ACTUAL_EXIT0_AND_POSTRUN_COHORT_AUDIT_REQUIRED')
    inherited_paths = {b['path'] for b in environment['repo_sources']} | {b['path'] for b in native.read(ROOT/FACTOR_PARENT)['repo_sources']}
    paths = inherited_paths | REQUIRED | {ENVIRONMENT, FACTOR_PARENT}
    plan['repo_sources'] = [native.binding(ROOT/p, ROOT) for p in sorted(paths)]
    validate(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('draft', 'inspect', 'preflight', 'run'))
    parser.add_argument('--protocol', type=Path); parser.add_argument('--part', type=int, choices=(0, 1))
    parser.add_argument('--hf-root', type=Path, default=Path.home()/'.cache/huggingface')
    parser.add_argument('--local-root', type=Path); parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'draft':
        native.need(args.part is not None, 'Part required'); result = draft(args.hf_root, args.part)
    elif args.command == 'inspect':
        plan = native.read(args.protocol); validate(plan)
        result = {'id': plan['id'], 'status': 'STATIC_SOFTWARE_ONLY', 'resources': RESOURCES, 'limits': limits(plan['part']),
            'model_calls': 0, 'GPU_calls': 0}
    elif args.command == 'preflight': result = preflight(args)
    else: result = run(args)
    if args.command == 'preflight': pass  # Both success and failure RUN are owned by preflight.
    elif args.out:
        native.write(args.out, result)
    else: print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__': main()
