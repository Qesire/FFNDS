"""Registered same-h W2 replays separating current local and student-prefix error.

Consumes two closed three-arm spatial52 collectors. Each GPU owns one physical
expert, all40 blocks, both applicable steps and both exposed diagnostic images.
No training, factor fitting, full model, new video or quality-model invocation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'tools') not in sys.path: sys.path.insert(0, str(ROOT/'tools'))
import e0_s2_spatial_student_collect_v2 as capture
import e0_raw_bf16_full80_spectrum as spectrum
import e0_raw_bf16_full21tau_analysis as data
import raw_bf16_full21tau_geometry as typed
import raw_bf16_spatial_prefix_diagnostics as geometry

core, native = capture.core, capture.core.native
BASE = 'E0_S2_SPATIAL52_SAME_H_LOCAL_PREFIX_REPLAY_20261029'
FAMILIES = ('value10', 'dcac10')
POSITIONS = 2*2*40
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 24 << 30,
    'host_available_min_bytes': 64 << 30, 'free_disk_min_bytes': 4 << 30,
    'wall_seconds': 7200, 'outer_seconds': 7500, 'new_HF_bytes': 512 << 20,
    'packet_full_SHA_stream_bytes_max': 40 << 30, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 300,
    'outer_seconds': 360, 'new_HF_bytes': 16 << 20}
DESIGN = {'sources': 'EXPOSED_HIGH_MOTION898_AND_CAMERA96_DIAGNOSTICS_N2_NOT_CONFIRMATION',
    'geometry': 'ALL21TAU_HIGH_FLOW25_CENTRAL25_FAR2_AT_ALL160_NATIVE_SITES',
    'algebra': 'TOTAL=STUDENT_LOCAL+PREFIX=L+P+I; I=STUDENT_LOCAL-TEACHER_LOCAL; ALL_SIGNED_CROSS_RETAINED',
    'gate_and_residual': 'EACH_STATE_OWN_CAPTURED_GATE_AND_RESIDUAL_RETAINED',
    'LPI_targets': 'Y_TIMES_OWN_GATE_AND_TYPED_U_ROWLN; GATED_INTERACTION_INCLUDES_ANY_GATE_CHANGE; U_PREFIX_INCLUDES_RESIDUAL',
    'numerics': 'BF16_M1092_ORIGINAL_W2_AND_TWO_GEMM_EXISTING_FACTORS_TF32_OFF',
    'rounding': 'RAW_TEACHER_AND_STUDENT_CANDIDATE_M1092_VS_ORIGINAL_M31668_REPORTED',
    'postwrite': 'EACH_TRIPLE_ORIGINAL_U_EXACT_BYTE_RESTORED_BEFORE_POSITION_COMPARISONS',
    'temporal': 'TAU0_SEPARATE_REST20_DC_AC_AND_GY_COMPLEX_FREQUENCY',
    'frequency': 'LOW1TO2_MID3TO6_HIGH7PLUS_ONE_SIDED_PARSEVAL_REAL_NYQUIST',
    'phase': 'PER_BIN_POOLED_COMPLEX_CROSS_PHASE_AND_EXPLICIT_BAND_AGGREGATE; NOT_MEAN_LOCAL_PHASE; LOW_COHERENCE_PHASE_NULL',
    'interpretation': 'FIXED_STATE_LOCAL_COUNTERFACTUAL_NOT_DOWNSTREAM_VIDEO_CAUSALITY',
    'motion_correspondence': 'NOT_MEASURED_GEOMETRIC_STRATA_ONLY',
    'fit': False, 'new_video': False, 'quality_or_speed_accepted': False}
REQUIRED = {'tools/e0_s2_spatial_prefix_replay.py', 'tests/test_e0_s2_spatial_prefix_replay.py',
    'tools/raw_bf16_spatial_prefix_diagnostics.py', 'tests/test_raw_bf16_spatial_prefix_diagnostics.py',
    'tools/e0_raw_bf16_full80_spectrum.py', 'tools/raw_bf16_full80_spectrum.py',
    'tools/e0_raw_bf16_full21tau_analysis.py', 'tools/raw_bf16_full21tau_geometry.py',
    'tools/e0_raw_bf16_temporal_pair_diagnostic.py'}


def identity(part):
    native.need(type(part) is int and part in (0, 1), 'Exactly two physical-expert shards')
    return BASE+'_PART'+str(part)


def output(part):
    identity(part)
    return f'ffnds-derived/e0_s2_spatial52_same_h_local_prefix_replay_20261029/part{part}/attempt1/worker/results'


def limits():
    return {'raw_headers': 1, 'raw_W2_tensor_reads': 40, 'raw_bias_tensor_reads': 40,
        'raw_H2D_tensors': 80, 'raw_H2D_bytes': 40*(5120*13824+5120)*2,
        'factor_file_reads': 80, 'factor_tensor_SHA_checks': 240,
        'factor_H2D_tensors': 240, 'factor_H2D_bytes': 80*core.vm.FACTOR_BYTES,
        'packet_reads': POSITIONS*3, 'packet_tensor_SHA_checks': POSITIONS*27,
        'row_coordinate_checks': POSITIONS*3, 'original_u_byte_neutrals': POSITIONS*3,
        'h_H2D_tensors': POSITIONS*3, 'h_H2D_bytes': POSITIONS*3*1092*13824*2,
        'BF16_original_W2_GEMM': POSITIONS*3, 'BF16_factor_GEMM': POSITIONS*8,
        'predicted_Y_D2H_tensors': POSITIONS*7, 'predicted_Y_D2H_bytes': POSITIONS*7*1092*5120*2,
        'counterfactual_postwrite': POSITIONS*7, 'rowLN_calls': POSITIONS*10,
        'gY_FP64_products': POSITIONS*10, 'temporal_comparisons': POSITIONS*6,
        'native_anchor_bridges': POSITIONS*6,
        'rFFT_calls': POSITIONS*32, 'prefix_input_comparisons': POSITIONS*2,
        'source_position_records': POSITIONS*2,
        'full_model_calls': 0, 'W1_calls': 0, 'fits': 0, 'new_videos': 0, 'official_quality_calls': 0}


def collect_index(parents, hf, expert):
    """Metadata-only closure. Tensor values are read exclusively by registered run."""
    cases, factor_proofs = [], []
    native.need(len(parents) == 2, 'Two closed source parts required')
    for part, proof in enumerate(parents):
        plan = native.read(native.bound(ROOT, proof['protocol']))
        core.validate(plan, True)
        report = native.read(native.bound(hf, proof['RUN']))
        core.vm.exited(native.read(native.bound(hf, proof['exit'])), proof['RUN'],
                       proof['protocol']['sha256'], plan['id'])
        expected = {k: v for k, v in plan['limits'].items() if v}
        native.need(plan['part'] == part and report['protocol_sha256'] == proof['protocol']['sha256']
            and report['status'] == 'COMPLETE_ONE_SOURCE_THREE_ARMS_FULL160_SPATIAL_STUDENT_CAPTURE'
            and report['all_thirty_prior_video_state_byte_neutrals'] is True
            and report['actual_native_packets'] == 480 and len(report['phases']) == 3
            and report['operation_ledger']['predebited'] == report['operation_ledger']['observed'] == expected
            and not report['operation_ledger']['forbidden_attempts']
            and not report['assigned_GPU_resources']['violations'], 'Capture not closed and exact')
        cpu_proof = report['CPU_preflight']
        cpu = native.read(native.bound(hf, cpu_proof['RUN']))
        core.vm.exited(native.read(native.bound(hf, cpu_proof['exit'])), cpu_proof['RUN'],
                       proof['protocol']['sha256'], plan['id'])
        native.need(cpu['factor_bindings_sha256'] == native.canonical(cpu['factor_bindings'])
            and all(cpu.get(k) == v for k, v in core.cpu_identity(plan, proof['protocol']['sha256']).items()),
            'Actual closed capture factor proof differs')
        factor_proofs.append(cpu['factor_bindings'])
        case = plan['cases'][0]; arms = {}
        for mode, phase in zip(core.MODES, report['phases']):
            observed = native.read(native.bound(hf, phase['observation_receipt']))
            native.need(all(phase.get(k) == v for k, v in observed.items())
                and phase['mode'] == mode and phase['all_ten_prior_video_state_bytes_equal'] is True
                and len(phase['states']) == 10 and len(phase['packets']) == 160
                and {(e['step'], e['block']) for e in phase['packets']} == {(s, b) for s in range(4) for b in range(40)}
                and phase['source_id'] == case['source_id'], 'Closed phase identity differs')
            controls = phase['original_video_control']['states']
            native.need(set(phase['states']) == set(controls)
                and all(core.vm.same_state(v, controls[k]) for k, v in phase['states'].items()),
                'All ten genuine prior states must remain byte-neutral')
            rows = []
            for entry in phase['packets']:
                meta = native.read(native.bound(hf, entry['metadata']))
                s, b = entry['step'], entry['block']
                native.need(meta['schema'] == 'ffnds.s2.spatial_student.operands.v1'
                    and meta['protocol_sha256'] == proof['protocol']['sha256']
                    and meta['source_id'] == case['source_id'] and meta['image_sha256'] == case['image']['sha256']
                    and meta['prompt_sha256'] == case['prompt_sha256'] and meta['mode'] == mode
                    and meta['seed'] == 42 and meta['numeric_path'] == 'BF16_ORIGINAL'
                    and meta['step'] == s and meta['block'] == b and meta['grid'] == [21, 29, 52]
                    and meta['selected_rows'] == plan['geometry']['selected_rows']
                    and meta['geometry_sha256'] == native.canonical(plan['geometry'])
                    and meta['source_state'] == ('teacher_input' if mode == 'dense' else 'student_input')
                    and meta['passive_observer'] is True and set(meta['tensor_metadata']) == set(core.SHAPES),
                    'Real packet source/state/coordinates differ')
                native.need(all(v['shape'] == core.SHAPES[k] and v['dtype'] == ('int64' if k == 'rows' else 'bfloat16')
                    and len(v['raw_tensor_sha256']) == 64 for k, v in meta['tensor_metadata'].items()), 'Packet typed header differs')
                if s//2 == expert:
                    rows.append({'step': s, 'block': b, 'metadata': entry['metadata'],
                        'payload': meta['payload'], 'tensor_metadata': meta['tensor_metadata']})
            native.need({(r['step'], r['block']) for r in rows} == {(2*expert+s, b) for s in (0, 1) for b in range(40)}
                and len(rows) == 80, 'Full expert coverage differs')
            arms[mode] = sorted(rows, key=lambda x: (x['block'], x['step']))
        cases.append({'source_id': case['source_id'], 'source': case['native_source_ordinal'],
            'geometry': plan['geometry'], 'arms': arms})
    native.need(factor_proofs[0] == factor_proofs[1] and len({c['source_id'] for c in cases}) == 2,
                'Same compiled factors and two distinct images required')
    factors = [r for r in factor_proofs[0] if r['expert'] == expert]
    native.need(len(factors) == 80 and {(r['family'], r['block']) for r in factors}
        == {(f, b) for f in FAMILIES for b in range(40)}, 'Exactly80 expert factor files')
    return {'cases': cases, 'factors': factors}


def validate(plan, frozen=False):
    native.need(plan['id'] == identity(plan['part']) and plan['resources'] == RESOURCES
        and plan['CPU_preflight_resources'] == CPU_RESOURCES and plan['limits'] == limits()
        and plan['design'] == DESIGN and plan['output_HF_relative'] == output(plan['part'])
        and plan['automatic_retry'] is False and REQUIRED <= {b['path'] for b in plan['repo_sources']}, 'Replay scope differs')
    if frozen: native.need(plan['assets_finalized'] is True and plan['status'] == 'FROZEN_REGISTERED', 'Replay not frozen')


def authorize(plan, sha):
    validate(plan, True)
    active = native.read(ROOT/'research/ACTIVE_RESEARCH.json')
    native.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True, 'Continuation inactive')
    scopes = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    native.need(len(scopes) == 1 and scopes[0]['protocol_sha256'] == sha
        and scopes[0]['limits'] == plan['limits'] and scopes[0]['resources'] == RESOURCES
        and scopes[0]['max_gpu_cards'] == 1
        and scopes[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE'), 'Exact replay not registered')
    return scopes[0]


def preflight(args):
    started = time.monotonic(); plan = native.read(args.protocol); sha = native.sha(args.protocol); authorize(plan, sha)
    native.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
        and len(os.sched_getaffinity(0)) <= 4 and not args.out.exists(), 'Fresh bounded metadata-only CPU proof required')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Replay CPU300s exhausted'))); signal.alarm(300)
    report = {'id': plan['id'], 'protocol_sha256': sha, 'status': 'STARTED_REPLAY_METADATA_PREFLIGHT'}
    native.write(args.out, report)
    try:
        for b in plan['repo_sources']: native.bound(ROOT, b)
        core.io.require_execution_master(plan, args.hf_root)
        native.need(collect_index(plan['parents'], args.hf_root, plan['part']) == plan['input_index'], 'Closed index changed')
        native.need(plan['input_index_sha256'] == native.canonical(plan['input_index']), 'Index digest differs')
        for row in plan['input_index']['factors']:
            native.need(native.stat_record(args.hf_root/row['binding']['path']) == row['stat'], 'Factor stat changed')
        for case in plan['input_index']['cases']:
            for arm in case['arms'].values():
                for e in arm:
                    native.need((args.hf_root/e['payload']['path']).stat().st_size == e['payload']['bytes'], 'Packet size changed')
        native.need(not {'torch', 'numpy'} & set(sys.modules), 'Numeric import in metadata gate')
        report.update(status='PASS_CLOSED960_PACKETS60_BYTE_CONTROLS_FACTOR_STAT_MASTER_PROOF_METADATA_ONLY',
            input_index_sha256=plan['input_index_sha256'], source_closure_sha256=native.canonical(plan['repo_sources']),
            packet_value_bytes_read=0, model_calls=0, GPU_calls=0)
    except BaseException as error:
        report.update(status='FAILED_REPLAY_CPU_PREFLIGHT_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); report['wall_seconds'] = time.monotonic()-started; native.write(args.out, report)


def stream_packet(hf, item, audit):
    path = (hf/item['path']).resolve(); before = path.stat()
    native.need(path.is_relative_to(hf.resolve()) and stat.S_ISREG(before.st_mode)
        and before.st_size == item['bytes'] and item['bytes'] < 128 << 20, 'Confined packet size differs')
    sha, chunks, count = hashlib.sha256(), [], 0
    with path.open('rb', buffering=0) as stream:
        while True:
            chunk = stream.read(min(8 << 20, item['bytes']+1-count))
            if not chunk: break
            count += len(chunk); audit['packet_bytes_read'] += len(chunk)
            native.need(count <= item['bytes'] and audit['packet_bytes_read'] <= RESOURCES['packet_full_SHA_stream_bytes_max'], 'Replay stream budget exceeded')
            sha.update(chunk); chunks.append(chunk)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    native.need(tuple(getattr(before, k) for k in fields) == tuple(getattr(after, k) for k in fields)
        and count == item['bytes'] and sha.hexdigest() == item['sha256'], 'Full packet SHA/stat differs')
    return b''.join(chunks)


def prefix_inputs(teacher, student, masks, np):
    result = {}
    for name in ('V1', 'h', 'r', 'u'):
        t = teacher[name].reshape(21, 52, -1)
        s = student[name].reshape(21, 52, -1)
        result[name] = {view: {roi: geometry.relative_sse(s[sl][:, mask], t[sl][:, mask], np)
            for roi, mask in masks.items()} for view, sl in
            (('all21', slice(None)), ('tau1_to20', slice(1, None)), ('tau0', slice(0, 1)))}
    return result


class Meter:
    def __init__(self, report, emit):
        self.report, self.emit = report, emit; report.update(predebited={}, observed={})

    def work(self, costs, fn, *args, **kwargs):
        native.need(all(k in limits() and type(n) is int and n > 0
            and self.report['predebited'].get(k, 0)+n <= limits()[k] for k, n in costs.items()), 'Replay operation budget exceeded')
        for k, n in costs.items(): self.report['predebited'][k] = self.report['predebited'].get(k, 0)+n
        self.emit({'kind': 'predebit', 'costs': costs})
        result = fn(*args, **kwargs)
        for k, n in costs.items(): self.report['observed'][k] = self.report['observed'].get(k, 0)+n
        return result

    def call(self, key, fn, *args, **kwargs): return self.work({key: 1}, fn, *args, **kwargs)

    def finish(self):
        native.need(all(self.report[side].get(k, 0) == n for side in ('predebited', 'observed')
            for k, n in limits().items()), 'Incomplete exact replay counts')


def run(args):
    started = time.monotonic(); plan = native.read(args.protocol); sha = native.sha(args.protocol); scope = authorize(plan, sha)
    proof = scope['cpu_preflight']; cpu = native.read(native.bound(args.hf_root, proof['RUN']))
    core.vm.exited(native.read(native.bound(args.hf_root, proof['exit'])), proof['RUN'], sha, plan['id'])
    native.need(cpu['status'].startswith('PASS_CLOSED960') and cpu['input_index_sha256'] == plan['input_index_sha256']
        and cpu['source_closure_sha256'] == native.canonical(plan['repo_sources']), 'Exited CPU gate differs')
    native.need(not {'torch', 'numpy'} & set(sys.modules) and len(os.sched_getaffinity(0)) <= 4, 'Fresh numeric worker required')
    for b in plan['repo_sources']: native.bound(ROOT, b)
    core.io.require_execution_master(plan, args.hf_root)
    native.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'], 'Runtime versions differ')
    out = (args.hf_root/plan['output_HF_relative']).resolve()
    native.need(out.is_relative_to(args.hf_root.resolve()) and not out.exists(), 'Fresh confined replay output required'); out.mkdir(parents=True)
    report = {'id': plan['id'], 'protocol_sha256': sha, 'part': plan['part'], 'status': 'STARTED_SAME_H_LOCAL_PREFIX_REPLAY',
        'design': DESIGN, 'CPU_preflight': proof, 'input_index_sha256': plan['input_index_sha256'],
        'completed_positions': 0, 'packet_bytes_read': 0, 'raw_tensor_IO': {}, 'raw_tensors': [],
        'automatic_retry': False, 'quality_or_speed_accepted': False}
    monitor = None; counter = core.io.WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    journal = counter.open_journal(out/'OPERATIONS.jsonl'); records = counter.open_journal(out/'POSITIONS.jsonl')
    meter = Meter(report.setdefault('operation_ledger', {}), lambda e: journal.write(json.dumps(e, separators=(',', ':'))+'\n'))
    persist = lambda: counter.write_json(out/'RUN.json', report)
    persist()
    def check():
        if monitor: monitor.check()
        counter.check()
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        native.need(time.monotonic()-started <= RESOURCES['wall_seconds'] and available >= RESOURCES['host_available_min_bytes']
            and shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes'], 'Replay wall/host/disk budget exceeded')
        if 'torch' in sys.modules: native.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'Replay GPU24GiB exceeded')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Replay7200s exhausted'))); signal.alarm(7200)
    try:
        import torch
        reservation, monitor, report['CUDA_reservation'] = spectrum.reserved_startup.reserve_before_assets(torch)
        import numpy as np
        from safetensors.torch import load_file
        native.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'One assigned A100 required')
        torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        native.need(not any(os.environ.get(k) == '1' for k in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')), 'TF32 override')
        for item in plan['repo_sources']+[native.binding(args.protocol, ROOT)]:
            target = out/'EXECUTED_SOURCES'/item['path']; target.parent.mkdir(parents=True, exist_ok=True)
            counter.hold('source', item['bytes']); counter.copy_bound(ROOT/item['path'], target, item, hold='source')
            native.need(counter.holds.pop('source') == 0, 'Snapshot accounting differs')
        master = native.read(native.bound(args.hf_root, plan['execution_raw_master_CPU_proof']['RUN']))
        kind = 'high' if plan['part'] == 0 else 'low'
        model = next(m for m in plan['models'] if m['kind'] == kind)
        master_stat = next(m['stat'] for m in master['model_payloads'] if m['kind'] == kind)
        header = meter.call('raw_headers', lambda: native.raw_header_check(native.header(model['path'])))
        native.need(header['header_sha256'] == model['header_sha256'], 'Original W2 header differs')
        factors_by_key = {(r['family'], r['block']): r for r in plan['input_index']['factors']}
        base_np = np
        def counted_rfft(*args, **kwargs): return meter.call('rFFT_calls', base_np.fft.rfft, *args, **kwargs)
        class NPProxy:
            fft = SimpleNamespace(rfft=counted_rfft)
            def __getattr__(self, name): return getattr(base_np, name)
        np_counted = NPProxy()
        for block in range(40):
            check(); authorize(plan, sha)
            weight, wr = meter.work({'raw_W2_tensor_reads': 1, 'raw_H2D_tensors': 1, 'raw_H2D_bytes': 5120*13824*2},
                spectrum.read_raw_tensor, torch, model, header, f'blocks.{block}.ffn.2.weight', [5120, 13824], master_stat, report['raw_tensor_IO'])
            bias, br = meter.work({'raw_bias_tensor_reads': 1, 'raw_H2D_tensors': 1, 'raw_H2D_bytes': 5120*2},
                spectrum.read_raw_tensor, torch, model, header, f'blocks.{block}.ffn.2.bias', [5120], master_stat, report['raw_tensor_IO'])
            report['raw_tensors'].extend([wr, br]); banks = {}
            for family in FAMILIES:
                row = factors_by_key[family, block]; path = args.hf_root/row['binding']['path']
                native.need(row['tensors']['right.bias']['raw_tensor_sha256'] == br['raw_tensor_sha256'],
                    'Compiled candidate bias differs from original W2 bias')
                native.need(native.stat_record(path) == row['stat'], 'Factor stat differs from prior full SHA proof')
                bank = meter.call('factor_file_reads', load_file, str(path), device='cpu')
                native.need(native.stat_record(path) == row['stat'] and set(bank) == set(row['tensors']), 'Loaded factor changed')
                for name, tensor in bank.items():
                    expected = row['tensors'][name]
                    meter.call('factor_tensor_SHA_checks', lambda tensor=tensor, expected=expected:
                        native.need(tensor.dtype == torch.bfloat16 and list(tensor.shape) == expected['shape']
                            and hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest() == expected['raw_tensor_sha256'], 'Loaded factor raw SHA differs'))
                banks[family] = meter.work({'factor_H2D_tensors': 3, 'factor_H2D_bytes': core.vm.FACTOR_BYTES},
                    lambda: {k: t.to('cuda') for k, t in bank.items()})
            for case in plan['input_index']['cases']:
                masks = geometry.spatial_masks(case['geometry'])
                for step in (2*plan['part'], 2*plan['part']+1):
                    check(); packets, h_gpu, entries = {}, {}, {}
                    for mode in core.MODES:
                        entry = next(e for e in case['arms'][mode] if e['step'] == step and e['block'] == block)
                        raw = meter.call('packet_reads', stream_packet, args.hf_root, entry['payload'], report)
                        packet = meter.work({'packet_tensor_SHA_checks': 9, 'row_coordinate_checks': 1}, data.decode_packet,
                            raw, entry, case['geometry']['selected_rows'], np, core.SHAPES, whole_SHA_already_stream_verified=True)
                        meter.call('original_u_byte_neutrals', typed.neutral_postwrite, packet, np)
                        packets[mode] = packet; entries[mode] = entry; del raw
                        h_gpu[mode] = meter.work({'h_H2D_tensors': 1, 'h_H2D_bytes': 1092*13824*2},
                            lambda: torch.frombuffer(bytearray(typed.bf16_bytes(packet['h'], np)), dtype=torch.bfloat16).reshape(1092, 13824).to('cuda'))
                    predicted, owners = {}, {}
                    for mode in core.MODES:
                        key = 'raw_'+mode; owners[key] = mode
                        predicted[key] = meter.call('BF16_original_W2_GEMM', torch.addmm, bias, h_gpu[mode], weight.T)
                    for family in FAMILIES:
                        for owner in ('dense', family):
                            key = family+'_'+owner; owners[key] = owner
                            predicted[key] = spectrum.core.factor_forward(h_gpu[owner], banks[family], meter)
                    ys, gy, us, ln = {}, {}, {}, {}
                    for key, value in predicted.items():
                        y_cpu = meter.work({'predicted_Y_D2H_tensors': 1, 'predicted_Y_D2H_bytes': 1092*5120*2}, lambda: value.detach().cpu())
                        ys[key] = (y_cpu.view(torch.uint16).numpy().astype(np.uint32) << 16).view(np.float32)
                        packet = packets[owners[key]]
                        gy[key] = meter.call('gY_FP64_products', lambda: ys[key].astype(np.float64)*packet['gate'].astype(np.float64))
                        _, us[key] = meter.call('counterfactual_postwrite', typed.postwrite, packet['r'], ys[key], packet['gate'], np)
                        ln[key] = meter.call('rowLN_calls', typed.row_ln, us[key], np)
                    native_gy, native_ln = {}, {}
                    for mode, packet in packets.items():
                        native_gy[mode] = meter.call('gY_FP64_products', lambda packet=packet:
                            packet['Y'].astype(np.float64)*packet['gate'].astype(np.float64))
                        native_ln[mode] = meter.call('rowLN_calls', typed.row_ln, packet['u'], np)
                    for family in FAMILIES:
                        keys = ('raw_dense', 'raw_'+family, family+'_'+family, family+'_dense')
                        result = {'source_id': case['source_id'], 'source': case['source'], 'step': step, 'block': block,
                            'family': family, 'expert': kind, 'original_three_u_byte_neutrals': True,
                            'gate_shift_scale_equal': {n: entries['dense']['tensor_metadata'][n]['raw_tensor_sha256']
                                == entries[family]['tensor_metadata'][n]['raw_tensor_sha256'] for n in ('gate', 'shift', 'scale')},
                            'M1092_vs_fullM_Y_relative_SSE': {
                                'raw_teacher': geometry.relative_sse(ys['raw_dense'], packets['dense']['Y'], np),
                                'candidate_student': geometry.relative_sse(ys[family+'_'+family], packets[family]['Y'], np)},
                            'same_first_site_V1_h_raw_equal': {n: np.array_equal(packets['dense'][n], packets[family][n]) for n in ('V1', 'h')}
                                if step == 0 and block == 0 else None}
                        result['observed_prefix_input_relative_SSE'] = meter.call('prefix_input_comparisons',
                            prefix_inputs, packets['dense'], packets[family], masks, np)
                        for target, arrays in (('gY', gy), ('u', us), ('rowLN_u', ln)):
                            result[target] = meter.call('temporal_comparisons', geometry.temporal_views,
                                *(arrays[k].reshape(21, 52, 5120) for k in keys), masks, np_counted, spectrum=target == 'gY')
                            native_arrays = native_gy if target == 'gY' else native_ln if target == 'rowLN_u' else {m: p['u'] for m, p in packets.items()}
                            result.setdefault('native_fullM_anchor_bridge', {})[target] = meter.call('native_anchor_bridges',
                                geometry.native_rounding_bridge, native_arrays['dense'].reshape(21, 52, 5120),
                                native_arrays[family].reshape(21, 52, 5120), arrays['raw_dense'].reshape(21, 52, 5120),
                                arrays[family+'_'+family].reshape(21, 52, 5120), masks, np_counted, spectrum=target == 'gY')
                        meter.call('source_position_records', lambda: records.write(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n'))
                    report['completed_positions'] += 1; persist()
                    del packets, h_gpu, predicted, ys, gy, us, ln
            del weight, bias, banks
        torch.cuda.synchronize(); meter.finish()
        native.need(report['completed_positions'] == POSITIONS and not counter.holds, 'Incomplete replay positions')
        report.update(status='COMPLETE_TWO_SOURCE_ONE_EXPERT_LOCAL_PREFIX_FULL_SPATIAL_REPLAY',
            all_original_u_neutrals_passed=True, actual_independent_images=2, full_model_calls=0,
            new_videos=0, fits=0, official_quality_calls=0)
    except BaseException as error:
        report.update(status='FAILED_LOCAL_PREFIX_REPLAY_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); monitor_error = None
        if monitor:
            try:
                monitor.close(); report['assigned_GPU_resources'] = monitor.report()
                native.need(not report['assigned_GPU_resources']['violations'], 'Assigned GPU resource violation')
            except BaseException as error:
                monitor_error = error
                report.update(status='FAILED_LOCAL_PREFIX_REPLAY_PRESERVED_NO_RETRY', resource_error=str(error))
        report.update(wall_seconds=time.monotonic()-started, output_accounting=counter.summary(),
            positions=counter.binding(out/'POSITIONS.jsonl', args.hf_root), operations=counter.binding(out/'OPERATIONS.jsonl', args.hf_root))
        persist(); records.close(); journal.close()
        if monitor_error: raise monitor_error


def draft(hf, part):
    identity(part); parents = []
    for source_part in (0, 1):
        path = ROOT/f'config/e0_s2_spatial_student_collect_part{source_part}_20261029_v2.json'
        plan = native.read(path); folder = hf/plan['output_HF_relative']
        parents.append({'protocol': native.binding(path, ROOT), 'RUN': native.binding(folder/'RUN.json', hf),
                        'exit': native.binding(folder.parent/'EXIT_RECEIPT.json', hf)})
    inherited = native.read(ROOT/parents[0]['protocol']['path'])
    plan = {k: inherited[k] for k in ('models', 'private_worker_config_sha256',
        'execution_private_worker_config_sha256', 'execution_raw_master_CPU_proof', 'runtime_versions')}
    index = collect_index(parents, hf, part)
    paths = {b['path'] for b in inherited['repo_sources']} | REQUIRED | {p['protocol']['path'] for p in parents}
    plan.update(id=identity(part), part=part, status='DRAFT_NOT_AUTHORIZED', assets_finalized=False,
        parents=parents, input_index=index, input_index_sha256=native.canonical(index), design=DESIGN,
        resources=RESOURCES, CPU_preflight_resources=CPU_RESOURCES, limits=limits(),
        automatic_retry=False, output_HF_relative=output(part),
        repo_sources=[native.binding(ROOT/p, ROOT) for p in sorted(paths)],
        independent_review='NOT_PERFORMED_AGENT_SERVICE_USAGE_LIMIT_COORDINATOR_REVIEW_REQUIRED')
    validate(plan); return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'preflight', 'run'))
    parser.add_argument('--part', type=int, choices=(0, 1)); parser.add_argument('--protocol', type=Path)
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--out', type=Path); args = parser.parse_args()
    if args.action == 'draft':
        native.need(not args.out.exists(), 'Fresh draft only'); native.write(args.out, draft(args.hf_root, args.part))
    elif args.action == 'preflight': preflight(args)
    else: run(args)


if __name__ == '__main__': main()
