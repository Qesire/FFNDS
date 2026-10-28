"""Fixed raw-BF16 all80 output spaces, finite source LOO and norm baselines.

No full model, W1, encoder, video or SGD runs. Actual M256 BF16 factor replay
uses saved teacher h and is explicitly separate from full-M native deployment.
"""
import argparse
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
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path:
    sys.path.insert(0, str(ROOT/'tools'))
import raw_bf16_full80_spectrum as core
import e0_original_bf16_full_depth_collect_v4 as infrastructure
from raw_bf16_assigned_gpu_monitor_v4 import startup as reserved_startup

native = infrastructure.native
ID = 'E0_RAW_BF16_FULL80_R960_SHARED_STEP_SPECTRUM_20261029'
RANK, D, H, ROWS = 960, 5120, 13824, 256
CAL = [240, 288, 304, 896, 897, 899]
DEV = [256, 894]
ORDINALS = CAL+DEV
ANCHORS = [(expert, block) for expert in (0, 1) for block in (0, 13, 26, 39)]
NEUTRAL_BYTES = (3*ROWS*D+D)*2
PACKET_UPLOAD_BYTES = (ROWS*H+3*ROWS*D+D)*2
FACTOR_BYTES = (H*RANK+RANK*D+D)*2
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 24 << 30,
    'host_available_min_bytes': 32 << 30, 'wall_seconds': 2700, 'outer_seconds': 3000,
    'new_HF_bytes': 10 << 30, 'free_disk_min_bytes': 12 << 30,
    'session_max_GPUs': 2, 'automatic_retry': False}
LIMITS = {'packet_reads': 1280, 'raw_W2_tensor_reads': 80, 'raw_bias_tensor_reads': 80,
    'neutral_packet_header_reads': 1280, 'neutral_tensor_range_reads': 5120,
    'neutral_operand_payload_bytes': 1280*NEUTRAL_BYTES,
    'neutral_H2D_tensors': 5120, 'neutral_H2D_bytes': 1280*NEUTRAL_BYTES,
    'packet_H2D_tensors': 6400, 'packet_H2D_bytes': 1280*PACKET_UPLOAD_BYTES,
    'raw_H2D_tensors': 160, 'raw_H2D_bytes': 80*(D*H+D)*2,
    'factor_export_D2H_tensors': 720, 'factor_export_D2H_bytes': 240*FACTOR_BYTES,
    'factor_readback_H2D_tensors': 720, 'factor_readback_H2D_bytes': 240*FACTOR_BYTES,
    'raw_tensor_bytes': 80*(D*H+D)*2, 'raw_headers': 2,
    'FP64_sample_Gram_GEMM': 248, 'FP64_eigh': 384, 'FP64_eigvalsh_lower_bound': 8,
    'FP64_basis_GEMM': 384, 'FP64_basis_orthogonality_GEMM': 384,
    'FP64_weight_metric_scales': 240, 'FP64_factor_compile_GEMM': 384,
    'FP64_covariance_commutator_GEMM': 480,
    'BF16_original_W2_GEMM': 1280, 'FP64_original_W2_reference_GEMM': 1280, 'BF16_factor_GEMM': 7424,
    'fixed_LS_gain_fits': 512, 'fixed_variance_gain_fits': 512,
    'fixed_gate_gain_calls': 12416, 'BF16_postwrite_calls': 12416,
    'teacher_postwrite_byte_neutrals': 1280, 'source_case_metrics': 11136,
    'rounding_bridge_statistics': 392,
    'BF16_export_files': 240, 'BF16_export_readbacks': 240,
    'full_model_flows': 0, 'new_W1_calls': 0, 'new_QDQ': 0, 'training_updates': 0,
    'new_videos': 0, 'official_quality_calls': 0}
DESIGN = {'rank': RANK, 'rank_grid': False, 'calibration_ordinals': CAL, 'development_ordinals': DEV,
    'LOO_anchors': [list(x) for x in ANCHORS], 'LOO': 'TRAIN5_HELD1_ONLY_FIXED8_PHYSICAL_FUNCTIONS',
    'all80': 'ALL6_FIT_AND_TWO_EXPOSED_DEVELOPMENT_TRANSFER',
    'bias': 'ORIGINAL_B2_FIXED_NO_EMPIRICAL_CENTERING',
    'shared': 'RMS_OF_TWO_STATIC_STEP_GATE_METRICS_FEASIBLE_SURROGATE_NOT_GLOBAL_JOINT_OPTIMUM',
    'step': 'ONE_FACTOR_PAIR_PER_STEP_PHYSICAL_W2_SHARED_IDENTITY_RETAINED',
    'gain': 'STEP_BLOCK_FIXED_SCALAR_BETA_0_TO_2_CONTINUOUS_FIT_FP32_GATE_SCALE_BF16_REPLAY',
    'native_boundary': 'M256_OPERATOR_REPLAY_NOT_M31668_FULL_FFN_OR_VIDEO',
    'rounding': 'NATIVE_Y_MINUS_B2_SPECTRUM_IS_NOT_AN_EXACT_NATIVE_RANK_LOWER_BOUND',
    'online': 'FULL_ORIGINAL_W1_GELU_THEN_SAME_CORE_FACTOR_FORWARD_ORIGINAL_BIAS_GATE_RESIDUAL',
    'scope': 'TEACHER_INPUT_ONLY_STUDENT_STATES_NOT_MEASURED', 'automatic_retry': False}
DESIGN['neutral_order'] = 'ALL1280_DENSE_BETA1_BYTE_NEUTRALS_COMPLETE_BEFORE_ANY_FIT'
DESIGN['IO'] = 'FIRST_Y_R_U_GATE_RANGES_THEN_EACH_WHOLE_PACKET_ONCE_PHYSICAL_W2_ONCE'
DESIGN['shared_covariance'] = 'UNCENTERED_HIDDEN_MOMENT_COMMUTATOR_EXACT_SAMPLE_TRACE_6_GEMM_PER_PHYSICAL'
REQUIRED = {'tools/e0_raw_bf16_full80_spectrum.py', 'tools/raw_bf16_full80_spectrum.py',
    'tests/test_e0_raw_bf16_full80_spectrum.py', 'tests/test_raw_bf16_full80_spectrum.py'}


class Meter:
    def __init__(self, report, journal):
        self.report, self.journal = report, journal
        report.update(predebited={}, observed={})
    def call(self, key, fn, *args, **kwargs):
        return self.work({key: 1}, fn, *args, **kwargs)
    def work(self, costs, fn, *args, **kwargs):
        native.need(costs and all(k in LIMITS and type(n) is int and n > 0
            and self.report['predebited'].get(k, 0)+n <= LIMITS[k] for k, n in costs.items()), 'Fixed diagnostic operation exceeded')
        for key, count in costs.items():
            self.report['predebited'][key] = self.report['predebited'].get(key, 0)+count
        self.journal({'kind': 'reserved_before_call', 'costs': costs})
        result = fn(*args, **kwargs)
        for key, count in costs.items():
            self.report['observed'][key] = self.report['observed'].get(key, 0)+count
        self.journal({'kind': 'observed_return', 'costs': costs})
        return result
    def finish(self):
        native.need(all(self.report[side].get(k, 0) == n for side in ('predebited', 'observed') for k, n in LIMITS.items()), 'Incomplete exact diagnostic budget')


def validate(plan, frozen=False):
    native.need(plan.get('id') == ID and plan.get('design') == DESIGN and plan.get('resources') == RESOURCES
        and plan.get('limits') == LIMITS and plan.get('automatic_retry') is False, 'Fixed full80 diagnostic differs')
    paths = [b['path'] for b in plan['repo_sources']]
    native.need(REQUIRED <= set(paths) and len(paths) == len(set(paths)), 'Source closure incomplete')
    relative = Path(plan['output_HF_relative'])
    native.need(not relative.is_absolute() and relative.parts[:1] == ('ffnds-derived',) and '..' not in relative.parts, 'Output escaped HF')
    if frozen:
        native.need(plan.get('assets_finalized') is True and plan.get('binding_status') == 'BOUND_READY_FOR_REGISTRATION'
            and isinstance(plan.get('operand_audit'), dict) and {'protocol', 'RUN', 'launch_exit', 'manifest'} <= set(plan['operand_audit']),
            'Complete eight-source audit actual exit0 NOT_BOUND')


def load_index(plan, hf):
    proof = plan['operand_audit']
    protocol = native.read(native.bound(ROOT, proof['protocol']))
    run = native.read(native.bound(hf, proof['RUN']))
    ex = native.read(native.bound(hf, proof['launch_exit']))
    native.need(run.get('id') == protocol['id'] == ex.get('id') and run.get('protocol_sha256') == ex.get('protocol_sha256') == proof['protocol']['sha256']
        and run.get('status') == 'COMPLETE_ASSET_BYTES_AUDITED' and ex.get('process_exited') is True
        and type(ex.get('child_returncode')) is int and ex['child_returncode'] == 0 and ex.get('timeout') is False
        and ex.get('automatic_retry') is False and ex.get('run_sha256') == proof['RUN']['sha256']
        and run.get('manifest') == proof['manifest'], 'Independent complete source audit exit0 required')
    index = native.read(native.bound(hf, proof['manifest']))
    native.need(index.get('schema') == 'ffnds.raw_bf16.full_depth.operands_index.v1'
        and index.get('totals') == {'sources': 8, 'calibration': 6, 'development': 2, 'packets': 1280, 'native_positions': 160}, 'Complete raw index geometry differs')
    cases = index['cases']
    native.need(len(cases) == 8 and sorted(c['native_source_ordinal'] for c in cases if c['role'] == 'calibration') == CAL
        and sorted(c['native_source_ordinal'] for c in cases if c['role'] == 'development') == sorted(DEV)
        and len({c['image_sha256'] for c in cases}) == 8, 'Frozen source-role split differs')
    positions = {(s, b) for s in range(4) for b in range(40)}
    for case in cases:
        native.need(case['source_state'] == 'teacher_input' and len(case['packets']) == 160
            and {(p['step'], p['block']) for p in case['packets']} == positions, 'Incomplete teacher positions')
    observations = index['conditioning_SHA_observations']
    native.need(len(observations) == 480 and {(o['step'], o['block'], o['tensor']) for o in observations}
        == {(s, b, k) for s in range(4) for b in range(40) for k in ('gate', 'scale', 'shift')}
        and all(o['all_eight_same'] is True and o['distinct_raw_sha256_count'] == 1 for o in observations), 'Source-static FFN gate/scale/shift disproved')
    return {c['native_source_ordinal']: c for c in cases}


def execution_master(plan, hf):
    return infrastructure.parent_v3.parent_v2.require_execution_master(plan, hf)


def cpu_expected(plan, digest):
    return {'id': ID, 'protocol_sha256': digest, 'status': 'PASS_AUDITED_RAW8_FULL80_METADATA_CURRENT_MASTER_STAT_NO_NUMERIC_IMPORTS',
        'source_closure_sha256': native.canonical(plan['repo_sources']), 'operand_audit': plan['operand_audit'],
        'execution_raw_master_CPU_proof': plan['execution_raw_master_CPU_proof'], 'all_CUDA_devices_hidden': True,
        'Torch_or_NumPy_imported': False, 'model_calls': 0, 'GPU_calls': 0, 'raw_payload_bytes_rehashed': 0}


def preflight(args):
    start = time.monotonic(); plan = native.read(args.protocol); validate(plan, True)
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 4, 'Fresh hidden GPU stdlib proof required')
    execution_master(plan, args.hf_root); load_index(plan, args.hf_root)
    for b in plan['repo_sources']:
        native.bound(ROOT, b)
    result = cpu_expected(plan, native.sha(args.protocol)); result['wall_seconds'] = time.monotonic()-start
    native.need(result['wall_seconds'] <= 300 and not {'torch', 'numpy'} & set(sys.modules), 'Byte proof wall/import gate')
    return result


def authorize(plan, digest):
    validate(plan, True); active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    policy = active.get('resource_scheduling_policy_20261028', {})
    native.need(active.get('user_continuation_authorization_20261027', {}).get('new_scope_registration_allowed') is True
        and policy.get('mode') == 'DYNAMIC_MAX_ACTUALLY_CALLABLE' and policy.get('global_gpu_card_cap') is None
        and active.get('current_session_resource_constraint_20261028', {}).get('max_simultaneously_assigned_idle_GPUs') == 2, 'Current authority differs')
    rows = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    native.need(len(rows) == 1 and rows[0].get('protocol_sha256') == digest and rows[0].get('execution_status') in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
        and rows[0].get('limits') == LIMITS and rows[0].get('resources') == RESOURCES and rows[0].get('max_gpu_cards') == 1, 'Fixed full80 scope not registered')
    return rows[0]


def require_cpu(scope, plan, digest, hf):
    proof = scope.get('cpu_preflight'); native.need(isinstance(proof, dict) and {'RUN', 'launch_exit'} <= set(proof), 'Independent CPU exit NOT_BOUND')
    run = native.read(native.bound(hf, proof['RUN']))
    native.need(all(run.get(k) == v for k, v in cpu_expected(plan, digest).items()) and 0 <= run.get('wall_seconds', -1) <= 300, 'Actual CPU identity differs')
    infrastructure.legacy.exited(native.read(native.bound(hf, proof['launch_exit'])), proof['RUN'], digest, ID)
    return proof


def packet_payloads(hf, entry, wanted, *, whole, io=None):
    """Read only four neutral ranges, or one complete original audited packet."""
    binding = entry['payload']; path = (hf/binding['path']).resolve()
    native.need(path.is_relative_to(hf.resolve()), 'Packet path escaped HF')
    before = native.stat_record(path)
    native.need(before['bytes'] == binding['bytes'], 'Packet bytes changed')
    def read(stream, count):
        value = stream.read(count)
        if io is not None:
            io['packet_or_neutral_actual_read_bytes'] = io.get('packet_or_neutral_actual_read_bytes', 0)+len(value)
        native.need(len(value) == count, 'Truncated saved operand packet')
        return value
    with path.open('rb') as stream:
        prefix = read(stream, 8); size = struct.unpack('<Q', prefix)[0]
        native.need(0 < size <= 1 << 20, 'Packet header size differs')
        header_raw = read(stream, size); header = json.loads(header_raw)
        shapes = {'V1': [ROWS, D], 'h': [ROWS, H], 'Y': [ROWS, D], 'r': [ROWS, D],
            'u': [ROWS, D], 'gate': [D], 'shift': [D], 'scale': [D], 'rows': [ROWS]}
        native.need(set(header)-{'__metadata__'} == set(entry['tensor_metadata']) == set(shapes)
            and set(wanted) <= set(shapes), 'Saved operand names differ')
        end = 0
        for name in sorted(shapes, key=lambda key: header[key]['data_offsets'][0]):
            info = header[name]; metadata = entry['tensor_metadata'][name]; a, b = info['data_offsets']
            width = 8 if name == 'rows' else 2
            native.need(info['shape'] == metadata['shape'] == shapes[name] and a == end
                and b-a == width*math.prod(shapes[name]) and info['dtype'] == ('I64' if name == 'rows' else 'BF16')
                and metadata['dtype'] == ('int64' if name == 'rows' else 'bfloat16'), 'Saved operand layout/dtype differs')
            end = b
        native.need(binding['bytes'] == 8+size+end, 'Packet trailing bytes differ')
        payloads = {}
        if whole:
            body = read(stream, end)
            native.need(hashlib.sha256(prefix+header_raw+body).hexdigest() == binding['sha256'], 'Actual packet SHA changed')
            for name, metadata in entry['tensor_metadata'].items():
                a, b = header[name]['data_offsets']; payload = body[a:b]
                native.need(hashlib.sha256(payload).hexdigest() == metadata['raw_tensor_sha256'], 'Packet tensor rawSHA changed')
                if name in wanted:
                    payloads[name] = payload
        else:
            for name in sorted(wanted):
                a, b = header[name]['data_offsets']; stream.seek(8+size+a); payload = read(stream, b-a)
                native.need(hashlib.sha256(payload).hexdigest() == entry['tensor_metadata'][name]['raw_tensor_sha256'], 'Neutral tensor rawSHA changed')
                payloads[name] = payload
    native.need(native.stat_record(path) == before, 'Packet changed during read')
    return payloads, {key: header[key]['shape'] for key in wanted}


def packet(torch, hf, entry, *, neutral=False, io=None):
    wanted = {'Y', 'r', 'u', 'gate'} if neutral else {'h', 'Y', 'r', 'u', 'gate'}
    payloads, shapes = packet_payloads(hf, entry, wanted, whole=not neutral, io=io)
    result = {}
    for name in sorted(wanted):
        value = torch.frombuffer(bytearray(payloads[name]), dtype=torch.bfloat16).reshape(shapes[name]).clone()
        native.need(bool(torch.isfinite(value).all()), 'Nonfinite teacher operand')
        result[name] = value.to('cuda')
    return result


def read_raw_tensor(torch, model, header, key, expected_shape, expected_stat, io=None):
    before = native.stat_record(model['path'])
    native.need(before == expected_stat, 'Raw master changed after current full SHA proof')
    info = header['rows'][key]; a, b = info['data_offsets']
    native.need(info['dtype'] == 'BF16' and info['shape'] == expected_shape and b-a == 2*math.prod(expected_shape), 'Original tensor header differs')
    with Path(model['path']).open('rb') as stream:
        stream.seek(8+header['header_bytes']+a); payload = stream.read(b-a)
    if io is not None:
        io['raw_master_actual_tensor_read_bytes'] = io.get('raw_master_actual_tensor_read_bytes', 0)+len(payload)
    native.need(len(payload) == b-a and native.stat_record(model['path']) == before, 'Raw tensor read changed/truncated')
    value = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(expected_shape).clone()
    digest = hashlib.sha256(payload).hexdigest()
    native.need(hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest() == digest, 'Loaded original tensor rawSHA differs from file range')
    return value.to('cuda'), {'path': model['path'], 'key': key, 'header_sha256': header['header_sha256'],
        'data_offsets': [a, b], 'actual_payload_bytes': len(payload), 'raw_tensor_sha256': digest, 'master_stat': before,
        'master_sha256': model['sha256'], 'master_bytes': model['bytes'],
        'identity': 'RANGE_RAW_SHA_AND_LOADED_BYTES_FROM_CURRENT_FULL_SHA_PROVED_STAT_IMMUTABLE_MASTER'}


def export(torch, counter, hf, out, name, factors, meter):
    from safetensors.torch import save_file, load_file
    path = out/(name+'.safetensors')
    cpu = meter.work({'factor_export_D2H_tensors': 3, 'factor_export_D2H_bytes': FACTOR_BYTES},
        lambda: {k: v.detach().cpu().contiguous() for k, v in factors.items()})
    counter.reserve_payload(sum(v.numel()*v.element_size() for v in cpu.values()))
    def save():
        save_file(cpu, str(path)); counter.observe_external(path)
    meter.call('BF16_export_files', save)
    def read():
        loaded = load_file(str(path), device='cpu')
        native.need(set(loaded) == set(cpu) and all(torch.equal(loaded[k], cpu[k]) and loaded[k].dtype == torch.bfloat16 for k in cpu), 'Actual exported BF16 readback differs')
        return {k: v.to('cuda') for k, v in loaded.items()}
    loaded = meter.work({'BF16_export_readbacks': 1, 'factor_readback_H2D_tensors': 3,
        'factor_readback_H2D_bytes': FACTOR_BYTES}, read)
    record = {'binding': counter.binding(path, hf), 'tensors': {k: {'shape': list(v.shape), 'dtype': 'bfloat16',
        'raw_tensor_sha256': hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()} for k, v in cpu.items()}}
    return loaded, record


def evaluate_fit(torch, data, train, test, steps, factors, meter, fold, family):
    predictions = {}
    for source in sorted(set(train+test)):
        for t in steps:
            predictions[source, t] = core.factor_forward(data[source, t]['h'], factors, meter)
    gains, result = {}, []
    for t in steps:
        gains[t] = core.fit_gains([data[s, t] for s in train], [predictions[s, t] for s in train], meter)
        choices = {'one': 1., **{k: v['beta_deployed_FP32'] for k, v in gains[t].items()}}
        for source in sorted(set(train+test)):
            case = data[source, t]; yh = predictions[source, t]
            for name, beta in choices.items():
                uh, effective_gate = core.postwrite(case['r'], yh, case['gate'], beta, meter, return_gate=True)
                metrics = meter.call('source_case_metrics', core.row_statistics, case['r'], case['Y'], case['u'], case['gate'], yh, uh, effective_gate=effective_gate)
                result.append({'family': family, 'fold': fold, 'source_index': source, 'native_source_ordinal': ORDINALS[source],
                    'local_step': t, 'role': 'train' if source in train else ('held' if fold is not None else 'development_transfer'),
                    'gain': name, 'beta': beta, **metrics})
    return result, {str(t): v for t, v in gains.items()}


def summarize(functions):
    """Rows are nested: first average steps within a source, then sources."""
    keys = ('gated_Y_relative_SSE', 'post_u_relative_SSE', 'post_centered_relative_SSE',
        'std_ratio_mean', 'std_ratio_std', 'std_ratio_MSE', 'normalized_u_relative_SSE',
        'normalized_direct', 'normalized_radial', 'normalized_cross_signed')
    accum = {}
    bottlenecks = []
    for record in functions:
        cells = {}
        for fit in record['fits']:
            family = 'shared80' if fit['family'] == 'shared' else 'step160'
            for case in fit['cases']:
                # Train cases in LOO are fitting diagnostics, not another N.
                if case['fold'] is not None and case['role'] != 'held':
                    continue
                role = 'anchor_LOO_held' if case['fold'] is not None else 'calibration_train' if case['role'] == 'train' else 'development_transfer'
                cell = cells.setdefault((family, case['gain'], role, case['native_source_ordinal']), [])
                cell.append(case)
        for (family, gain, role, source), values in cells.items():
            source_summary = {k: sum(v[k] for v in values)/len(values) for k in keys}
            accum.setdefault((family, gain, role, source), []).append(source_summary)
        for family in ('shared80', 'step160'):
            chosen = [v for (f, g, role, s), values in cells.items() if f == family and g == 'one'
                and role == 'development_transfer' for v in values]
            bottlenecks.append({'expert': record['expert'], 'block': record['block'], 'family': family,
                'development_mean_normalized_u_relative_SSE': sum(v['normalized_u_relative_SSE'] for v in chosen)/len(chosen)})
    grouped = {}
    for (family, gain, role, source), values in accum.items():
        grouped.setdefault((family, gain, role), []).append({'native_source_ordinal': source,
            **{k: sum(v[k] for v in values)/len(values) for k in keys}, 'physical_functions': len(values)})
    return {'identity': 'STEP_WITHIN_SOURCE_THEN_PHYSICAL_FUNCTION_THEN_SOURCE_EQUAL',
        'independent_calibration_sources': 6, 'development_sources': 2, 'LOO_anchor_physical_functions': 8,
        'groups': [{'family': f, 'gain': g, 'role': role, 'sources': values,
            'source_equal': {k: sum(v[k] for v in values)/len(values) for k in keys}}
            for (f, g, role), values in sorted(grouped.items())],
        'development_bottlenecks_descriptive_only_no_selection': sorted(bottlenecks,
            key=lambda x: -x['development_mean_normalized_u_relative_SSE']),
        'deploy_cost': {'original_W2_MAC_per_token': H*D, 'factor_W2_MAC_per_token': RANK*(H+D),
            'factor_W2_MAC_fraction': RANK*(H+D)/(H*D),
            'full_W1_and_factor_W2_MAC_fraction': (H*D+RANK*(H+D))/(2*H*D),
            'shared80_export_tensor_bytes': 80*FACTOR_BYTES, 'step160_export_tensor_bytes': 160*FACTOR_BYTES,
            'system_speed': 'NOT_MEASURED', 'full_video_quality': 'NOT_MEASURED'}}


def run(args):
    started = time.monotonic(); plan = native.read(args.protocol); digest = native.sha(args.protocol)
    scope = authorize(plan, digest); proof = require_cpu(scope, plan, digest, args.hf_root)
    native.need(not {'torch', 'numpy'} & set(sys.modules) and len(os.sched_getaffinity(0)) <= 4, 'Fresh numerical worker/affinity gate')
    native.need(not any(os.environ.get(k) == '1' for k in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')), 'TF32 environment override')
    master_proof = execution_master(plan, args.hf_root); index = load_index(plan, args.hf_root)
    current_masters = native.read(native.bound(args.hf_root, master_proof['RUN']))['model_payloads']
    master_stats = {item['kind']: item['stat'] for item in current_masters}
    for b in plan['repo_sources']:
        native.bound(ROOT, b)
    native.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'], 'Native package versions differ')
    out = (args.hf_root/plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined diagnostic output required')
    out.mkdir(parents=True); monitor = None
    counter = infrastructure.WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    report = {'id': ID, 'protocol_sha256': digest, 'status': 'STARTED_FIXED_FULL80_SPECTRUM_NO_RETRY',
        'CPU_preflight': proof, 'design': DESIGN, 'automatic_retry': False, 'functions': [],
        'full_model_calls': 0, 'student_states': 'NOT_CAPTURED', 'quality_or_system_speed_accepted': False,
        'actual_input_IO': {}, 'teacher_postwrite_neutrals_complete_before_fit': False,
        'fixed_native_precision': {'allow_tf32': False, 'allow_bf16_reduced_precision_reduction': True,
            'factor_input_M': ROWS, 'captured_teacher_M': 31668, 'factor_forward': 'raw_bf16_full80_spectrum.factor_forward'}}
    persist = lambda: counter.write_json(out/'RUN.json', report)
    journal = counter.open_journal(out/'OPERATIONS.jsonl')
    meter = Meter(report.setdefault('operation_ledger', {}), lambda event: journal.write(json.dumps(event, separators=(',', ':'))+'\n'))
    persist()
    def check():
        if monitor:
            monitor.check()
        native.need(time.monotonic()-started <= RESOURCES['wall_seconds'], 'Fixed2700s diagnostic budget exhausted')
        counter.check()
        native.need(shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes'], 'Diagnostic free-disk reserve violated')
        if 'torch' in sys.modules:
            native.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'Diagnostic CUDA reserved24GiB exceeded')
    def timeout(*_):
        raise TimeoutError('Fixed full80 diagnostic wall2700 reached; no automatic retry')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(RESOURCES['wall_seconds'])
    try:
        import torch
        reservation, monitor, report['CUDA_reservation'] = reserved_startup.reserve_before_assets(torch)
        native.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'One assigned A100 required')
        torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        # Freeze one copy of every executed source and protocol, outside hot loops.
        source_bytes = sum(b['bytes'] for b in plan['repo_sources'])+args.protocol.stat().st_size
        counter.hold('frozen_metadata', source_bytes)
        for item in plan['repo_sources']:
            check()
            destination = out/'source_snapshots'/item['path']; destination.parent.mkdir(parents=True, exist_ok=True)
            counter.copy_bound(ROOT/item['path'], destination, item)
        counter.copy_bound(args.protocol, out/'EXECUTED_PROTOCOL.json',
            {'bytes': args.protocol.stat().st_size, 'sha256': digest})
        native.need(counter.holds.pop('frozen_metadata') == 0, 'Unconsumed source snapshot budget')
        report['source_snapshot_bytes'] = source_bytes
        with Path('/proc/meminfo').open() as stream:
            available = int(next(line.split()[1] for line in stream if line.startswith('MemAvailable:')))*1024
        native.need(available >= RESOURCES['host_available_min_bytes'], 'Insufficient available host memory for bounded function stream')
        models = {m['kind']: m for m in plan['models'] if m['kind'] in ('high', 'low')}
        headers = {}
        for kind, model in models.items():
            headers[kind] = meter.call('raw_headers', lambda: native.raw_header_check(native.header(model['path'])))
            native.need(headers[kind]['header_sha256'] == model['header_sha256'], 'Raw master header differs')
        # This global pass precedes every spectral fit, not merely each layer fit.
        neutral_started = time.monotonic()
        for ordinal in ORDINALS:
            for entry in sorted(index[ordinal]['packets'], key=lambda value: (value['step'], value['block'])):
                check()
                case = meter.work({'neutral_packet_header_reads': 1, 'neutral_tensor_range_reads': 4,
                    'neutral_operand_payload_bytes': NEUTRAL_BYTES, 'neutral_H2D_tensors': 4,
                    'neutral_H2D_bytes': NEUTRAL_BYTES}, packet, torch, args.hf_root, entry,
                    neutral=True, io=report['actual_input_IO'])
                restored = core.postwrite(case['r'], case['Y'], case['gate'], 1., meter)
                meter.call('teacher_postwrite_byte_neutrals', lambda: native.need(torch.equal(restored, case['u']),
                    'Saved Dense postwrite beta1 byte neutral failed before any fit'))
                del case, restored
            report['last_neutral_source_ordinal'] = ordinal; persist()
        native.need(report['operation_ledger']['observed'].get('teacher_postwrite_byte_neutrals') == 1280
            and report['operation_ledger']['predebited'].get('FP64_eigh', 0) == 0, 'Global neutral order violated')
        report.update(teacher_postwrite_neutrals_complete_before_fit=True,
            teacher_postwrite_neutral_wall_seconds=time.monotonic()-neutral_started)
        persist(); completed_records = []
        for expert in (0, 1):
            kind = 'high' if expert == 0 else 'low'
            for block in range(40):
                check(); function_started = time.monotonic()
                record = {'expert': kind, 'block': block, 'fits': [], 'raw_M256_rounding': []}
                data = {}
                for source, ordinal in enumerate(ORDINALS):
                    for local_step in (0, 1):
                        step = 2*expert+local_step
                        entry = next(p for p in index[ordinal]['packets'] if p['step'] == step and p['block'] == block)
                        data[source, local_step] = meter.work({'packet_reads': 1, 'packet_H2D_tensors': 5,
                            'packet_H2D_bytes': PACKET_UPLOAD_BYTES}, packet, torch, args.hf_root, entry, io=report['actual_input_IO'])
                for t in (0, 1):
                    native.need(all(torch.equal(data[s, t]['gate'], data[0, t]['gate']) for s in range(8)), 'Source-static actual gate mismatch')
                weight, weight_record = meter.work({'raw_W2_tensor_reads': 1, 'raw_tensor_bytes': D*H*2,
                    'raw_H2D_tensors': 1, 'raw_H2D_bytes': D*H*2}, read_raw_tensor, torch, models[kind], headers[kind],
                    f'blocks.{block}.ffn.2.weight', [D, H], master_stats[kind], report['actual_input_IO'])
                bias, bias_record = meter.work({'raw_bias_tensor_reads': 1, 'raw_tensor_bytes': D*2,
                    'raw_H2D_tensors': 1, 'raw_H2D_bytes': D*2}, read_raw_tensor, torch, models[kind], headers[kind],
                    f'blocks.{block}.ffn.2.bias', [D], master_stats[kind], report['actual_input_IO'])
                record['raw_tensors'] = [weight_record, bias_record]
                weight64, bias64 = weight.T.double(), bias.double()
                for s in range(8):
                    for t in (0, 1):
                        case = data[s, t]
                        raw = meter.call('BF16_original_W2_GEMM', torch.addmm, bias, case['h'], weight.T)
                        difference = raw.to(torch.float64)-case['Y'].to(torch.float64)
                        linear = meter.call('FP64_original_W2_reference_GEMM', torch.addmm, bias64, case['h'].double(), weight64)
                        case['continuous_rounding_error'] = case['Y'].double()-linear
                        record['raw_M256_rounding'].append({'source': ORDINALS[s], 'local_step': t,
                            'bitwise_equal_captured_fullM_rows': bool(torch.equal(raw, case['Y'])),
                            'relative_SSE': float(difference.square().sum()/case['Y'].to(torch.float64).square().sum().clamp_min(1e-30)),
                            'captured_addmm_vs_FP64_hW_bias_gate_relative_SSE': float((case['continuous_rounding_error']*case['gate'].double()).square().sum()/(case['Y'].double()*case['gate'].double()).square().sum().clamp_min(1e-30)),
                            'M256_addmm_vs_FP64_hW_bias_relative_SSE': float((raw.double()-linear).square().sum()/linear.square().sum().clamp_min(1e-30))})
                metric = core.metrics([data[0, t]['gate'] for t in (0, 1)])
                weighted_h = []
                for t in (0, 1):
                    chunks = []
                    for s in range(6):
                        case = data[s, t]
                        energy = max(float((case['Y'].double()*case['gate'].double()).square().sum()/ROWS), 1e-30)
                        chunks.append(case['h'].double()/math.sqrt(6*ROWS*energy))
                    weighted_h.append(torch.cat(chunks))
                record['hidden_moment_commutator'] = core.hidden_moment_commutator(*weighted_h, meter)
                for family, steps, metric_name in (('shared', [0, 1], 'shared'), ('step0', [0], 'step0'), ('step1', [1], 'step1')):
                    a = core.anchored_rows(data, bias, metric[metric_name], steps=steps, source_indices=list(range(6)))
                    gram = core.sample_gram(a, meter)
                    scaled = meter.call('FP64_weight_metric_scales', lambda: weight64*metric[metric_name])
                    held_values = [None]+(list(range(6)) if (expert, block) in ANCHORS else [])
                    for held in held_values:
                        check(); train = [s for s in range(6) if s != held]
                        aa, kk = (a, gram) if held is None else core.source_subset(a, gram, train, 6, ROWS*len(steps))
                        basis, spectral = core.fit_basis(aa, kk, RANK, meter)
                        spectral['rounding_bridge'] = meter.call('rounding_bridge_statistics', core.rounding_bridge, data, metric[metric_name], steps=steps,
                            source_indices=train, continuous_tail=spectral['continuous_tail'])
                        factors = core.compile_factors(scaled, bias, metric[metric_name], basis, meter)
                        exported = None
                        if held is None:
                            factors, exported = export(torch, counter, args.hf_root, out, f'{kind}_block{block:02d}_{family}', factors, meter)
                        test = [6, 7] if held is None else [held]
                        results, gains = evaluate_fit(torch, data, train, test, steps, factors, meter, held, family)
                        record['fits'].append({'family': family, 'held_source': None if held is None else ORDINALS[held],
                            'training_sources': [ORDINALS[s] for s in train], 'spectrum': spectral, 'export': exported,
                            'fixed_step_gains': gains, 'cases': results})
                if (expert, block) in ANCHORS:
                    amin = core.anchored_rows(data, bias, metric['minimum'], steps=[0, 1], source_indices=list(range(6)))
                    spectrum = core.minimum_spectrum(amin, meter)
                    record['continuous_min_gate_lower_spectrum'] = {'values': spectrum, 'rank': RANK,
                        'continuous_tail': sum(spectrum[RANK:]), 'trace': sum(spectrum),
                        'rounding_bridge': meter.call('rounding_bridge_statistics', core.rounding_bridge, data, metric['minimum'], steps=[0, 1],
                            source_indices=list(range(6)), continuous_tail=sum(spectrum[RANK:])),
                        'identity': 'ALG_CONTINUOUS_COMMON_RANK_LOWER_RELAXATION_WITH_ADDMM_ROUNDING_REMAINDER_NOT_NATIVE_BOUND'}
                record['wall_seconds'] = time.monotonic()-function_started
                name = f'{kind}_block{block:02d}.json'; counter.write_json(out/name, record)
                report['functions'].append({'expert': kind, 'block': block, 'record': counter.binding(out/name, args.hf_root)})
                completed_records.append(record)
                persist(); del data, weight, bias, weight64, bias64, weighted_h, record, metric, scaled, a, gram, factors, basis
                check()
        meter.finish()
        counter.write_json(out/'SUMMARY.json', summarize(completed_records))
        report.update(status='COMPLETE_RAW_FULL80_FIXED_SPECTRUM_ANCHOR_LOO_BF16_FACTOR_BANKS',
            summary=counter.binding(out/'SUMMARY.json', args.hf_root),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            evidence='6 independent calibration sources;2 historically exposed development transfers; anchors only6LOO;teacher states only')
    except BaseException as error:
        report.update(status='FAILED_FULL80_DIAGNOSTIC_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); monitor_error = None
        if monitor:
            monitor.close(); report['assigned_GPU_resources'] = monitor.report()
            if report['assigned_GPU_resources'].get('violations'):
                monitor_error = RuntimeError('Assigned GPU monitor violation')
                report.update(status='FAILED_FULL80_DIAGNOSTIC_PRESERVED_NO_RETRY', resource_error=str(monitor_error))
        journal.close(); report['operation_journal'] = counter.binding(out/'OPERATIONS.jsonl', args.hf_root)
        report['wall_seconds'] = time.monotonic()-started; report['output_accounting_checkpoint_before_final_RUN'] = counter.summary()
        persist()
        if monitor_error is not None and sys.exc_info()[0] is None:
            raise monitor_error
    return report


def draft():
    original = native.read(ROOT/infrastructure.PARENT)
    plan = {k: copy.deepcopy(original[k]) for k in ('models', 'runtime_versions', 'execution_raw_master_CPU_proof', 'execution_private_worker_config_sha256')}
    closure = {b['path'] for b in original['repo_sources']} | REQUIRED | infrastructure.EXTRA_SOURCES
    plan.update(id=ID, status='DRAFT_WAIT_COMPLETE_EIGHT_SOURCE_ASSET_AUDIT', assets_finalized=False,
        binding_status='NOT_BOUND_COMPLETE_EIGHT_SOURCE_AUDIT', automatic_retry=False,
        design=DESIGN, limits=LIMITS, resources=RESOURCES, operand_audit=None,
        output_HF_relative='ffnds-derived/e0_raw_bf16_full80_r960_shared_step_spectrum_20261029/attempt1/worker/results',
        CPU_preflight_resources={'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 300, 'outer_seconds': 360, 'new_raw_master_hash_bytes': 0},
        repo_sources=[native.binding(ROOT/p, ROOT) for p in sorted(closure)])
    summary = ROOT/'research/20261029/ORIGINAL_BF16_FULL_DEPTH_CORPUS_RESULTS_20261029.json'
    if summary.exists():
        result = native.read(summary)
        native.need(result.get('status') == 'COMPLETE_ASSET_AUDIT_ACTUAL_CPU_EXIT0', 'Corpus summary is not a completed actual audit')
        plan.update(status='DRAFT_AUDITED_ASSETS_INDEPENDENT_REVIEW_AND_FREEZE_PENDING',
            binding_status='AUDIT_BOUND_REVIEW_AND_FINAL_FREEZE_PENDING',
            operand_audit={key: copy.deepcopy(result[source]) for key, source in
                (('protocol', 'protocol'), ('RUN', 'RUN'), ('launch_exit', 'exit'), ('manifest', 'manifest'))})
    validate(plan); return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=('draft', 'inspect', 'preflight', 'run'))
    parser.add_argument('--protocol', type=Path, required=True); parser.add_argument('--hf-root', type=Path, required=True); parser.add_argument('--out', type=Path)
    args = parser.parse_args(); args.hf_root = args.hf_root.resolve()
    if args.command == 'draft':
        native.need(not args.protocol.exists(), 'Fresh draft path required'); native.write(args.protocol, draft()); return
    plan = native.read(args.protocol); validate(plan, args.command != 'inspect')
    if args.command == 'inspect':
        print(json.dumps({'id': ID, 'limits': LIMITS, 'resources': RESOURCES, 'status': plan['status']}, indent=2)); return
    if args.command == 'preflight':
        native.need(args.out is not None, 'Fresh CPU output required'); out = args.out.resolve()
        native.need(out.is_relative_to(args.hf_root) and not out.exists(), 'Fresh confined CPU output')
        out.mkdir(parents=True); value = {'id': ID, 'protocol_sha256': native.sha(args.protocol), 'status': 'STARTED_CPU_PREFLIGHT'}
        native.write(out/'RUN.json', value)
        try:
            value = preflight(args)
        except BaseException as error:
            value.update(status='FAILED_CPU_PREFLIGHT_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
        finally:
            native.write(out/'RUN.json', value)
        print(json.dumps({'status': value['status']})); return
    print(json.dumps({'status': run(args)['status']}))


if __name__ == '__main__':
    main()
