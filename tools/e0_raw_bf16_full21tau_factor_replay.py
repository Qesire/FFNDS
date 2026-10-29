"""Fixed same-Dense-h temporal error replay: original M168 / shared / step W2.

No fitting, W1, full model or new video. The original M168 W2 replay explicitly
measures shape-dependent rounding against the captured full-M Dense output.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_full21tau_analysis as data
import e0_raw_bf16_full80_spectrum as original
import e0_raw_bf16_full160_video20_v2 as video
import raw_bf16_full80_spectrum as factor
from raw_bf16_assigned_gpu_monitor_v4 import startup

native = original.native
ID = 'E0_RAW_BF16_FULL21TAU_NEW10_FIXED_FACTOR_REPLAY_20261029'
OUTPUT = 'ffnds-derived/e0_raw_bf16_full21tau_new10_fixed_factor_replay_20261029/attempt1/worker/results'
INPUT_PROTOCOL = 'config/e0_raw_bf16_full21tau_new10_geometry_obs_cpu_20261029_v1.json'
FACTOR_PROTOCOL = 'config/e0_raw_bf16_full160_video20_part0_20261029_v2.json'
FACTOR_CPU_DIR = 'ffnds-derived/e0_raw_bf16_full160_video20_v2_cpu_20261029/part0/attempt1/worker'
ARMS = ('raw_M168', 'shared', 'step')
TARGETS = ('gY_FP64', 'u_BF16_emulated', 'rowLN_u_FP64')
METRICS = (*data.core.ACTION_METRICS, 'DC_error_fraction', 'AC_aligned_gain', 'AC_cosine')
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'wall_seconds': 1800, 'outer_seconds': 2100,
             'GPU_reserved_max_bytes': 24 << 30, 'host_available_min_bytes': 32 << 30,
             'new_HF_bytes': 128 << 20, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 180, 'outer_seconds': 240,
                 'new_HF_bytes': 16 << 20}
DESIGN = {'unit': 'TEN_NEW_DEVELOPMENT_IMAGES_EQUAL_160_POSITIONS_WITHIN_SOURCE_THEN_EQUAL_IMAGES',
          'state': 'IDENTICAL_CAPTURED_DENSE_H_R_GATE_NOT_STUDENT_FREE_PREFIX',
          'geometry': 'ALL_21_LATENT_TIMES_X8_FIXED_XY; NOT_MOTION_ALIGNED', 'M': 168,
          'arms': list(ARMS), 'targets': list(TARGETS), 'views': list(data.core.VIEWS),
          'factors': 'EXISTING_ALL6_CAL_R960_EXPORTS_NO_REFIT_NO_GAIN_OR_HYPERPARAMETER_SELECTION',
          'raw_M168': 'PAID_ORIGINAL_W2_BIAS_REPLAY_SEPARATE_FROM_CAPTURED_M31668_ROUNDING',
          'factor_forward': 'EXACT_FROZEN_BF16_MM_THEN_ADDMM_ORIGINAL_B2',
          'postwrite': 'CPU_EXPLICIT_BF16_PRODUCT_THEN_ADD_VALIDATED_BY_EVERY_ORIGINAL_SAVED_U_BYTE',
          'motion': 'DC_AC_BOUNDARY_GEOMETRY_NOT_VIDEO_ACTION_CAUSALITY_OR_ACCEPTANCE',
          'precision': 'TF32_OFF_HIGHEST_BF16_REDUCED_PRECISION_REDUCTION_TRUE',
          'fits': 0, 'decompositions': 0, 'W1_calls': 0, 'model_flows': 0, 'new_videos': 0,
          'official_quality_calls': 0, 'quality_or_speed_accepted': False, 'automatic_retry': False}
REQUIRED = {'tools/e0_raw_bf16_full21tau_factor_replay.py',
            'tests/test_e0_raw_bf16_full21tau_factor_replay.py',
            'tools/durable_worker_job.py', 'scripts/worker_wait_uuid_idle_exec.sh'}


def labels():
    return [f'{arm}/{target}/{view}/{metric}' for arm in ARMS for target in TARGETS
            for view in data.core.VIEWS for metric in METRICS]


def error_components(candidate, reference, np):
    values = data.core.error_geometry(candidate, reference, np)
    dc_energy = values[1]-values[5]
    gain = None if values[8] is None else 1+values[8]
    cosine = gain/math.sqrt(values[7]) if gain is not None and values[7] is not None and values[7] > 0 else None
    data.need(cosine is None or abs(cosine) <= 1+1e-11, 'AC cosine outside rounding tolerance')
    return [*values, values[3]/dc_energy if dc_energy > 0 else None,
            gain, None if cosine is None else min(1., max(-1., cosine))]


def limits(index):
    packets = [p for c in index['cases'] for p in c['packets']]
    data.need(len(index['cases']) == 10 and len(packets) == 1600, 'Full ten-source acquisition required')
    return {'raw_headers': 2, 'raw_W2_tensors': 80, 'raw_bias_tensors': 80,
            'raw_tensor_bytes': 80*(5120*13824+5120)*2,
            'raw_H2D_tensors': 160, 'raw_H2D_bytes': 80*(5120*13824+5120)*2,
            'factor_files': 240, 'factor_tensor_raw_SHA_checks': 720, 'factor_H2D_tensors': 720,
            'factor_H2D_bytes': 240*video.FACTOR_BYTES, 'factor_bias_anchor_checks': 240,
            'packet_full_SHA_reads': 1600, 'packet_full_SHA_bytes': sum(p['payload']['bytes'] for p in packets),
            'tensor_raw_SHA_checks': 14400, 'row_payload_checks': 1600,
            'original_u_byte_neutrals': 1600, 'BF16_product_calls': 6400, 'BF16_add_calls': 6400,
            'h_CPU_BF16_reconstruction_checks': 1600, 'h_H2D_tensors': 1600,
            'h_H2D_bytes': 1600*168*13824*2,
            'BF16_original_W2_GEMM': 1600, 'BF16_factor_GEMM': 6400,
            'predicted_Y_D2H_tensors': 4800, 'predicted_Y_D2H_bytes': 4800*168*5120*2,
            'gY_FP64_products': 6400, 'rowLN_FP64_calls': 6400,
            'temporal_error_evaluations': 1600*3*3*3,
            'source_position_records': 1600, 'source_equal_finalizations': 1,
            'fits': 0, 'decompositions': 0, 'W1_calls': 0, 'full_model_flows': 0, 'new_videos': 0}


def source_inputs(plan, hf):
    raw = data.read(native.bound(ROOT, plan['source_input_protocol']))
    index, metadata = data.collect_inputs(raw, hf)
    data.need(index == plan['input_index'] and raw['input_index'] == index, 'Closed source index changed')
    data.need(index['models_sha256'] == native.canonical(plan['models']), 'Teacher operands and factor checkpoint differ')
    base = data.read(native.bound(ROOT, plan['factor_video_protocol']))
    data.need(base['models'] == plan['models'] and base['factor_parent'] == plan['factor_parent'], 'Factor raw model differs')
    return raw, index, base


def reused_factor_bytes(plan, hf):
    base = data.read(native.bound(ROOT, plan['factor_video_protocol']))
    proof = plan['reused_factor_CPU_proof']
    cpu = data.read(native.bound(hf, proof['RUN']))
    video.exited(data.read(native.bound(hf, proof['exit'])), proof['RUN'], plan['factor_video_protocol']['sha256'], base['id'])
    data.need(all(cpu.get(k) == v for k, v in video.cpu_expected(base, plan['factor_video_protocol']['sha256']).items())
              and cpu['factor_bindings_sha256'] == native.canonical(cpu['factor_bindings']), 'Exited real factor CPU proof differs')
    current = video.factor_index(base, hf, False)
    strip = lambda r: {k: v for k, v in r.items() if k not in ('sha256', 'actual_payload_read_bytes')}
    data.need([strip(r) for r in current] == [strip(r) for r in cpu['factor_bindings']], 'Factor current-worker stat/header changed')
    return cpu['factor_bindings']


def validate(plan, frozen=False):
    data.need(plan['id'] == ID and plan['design'] == DESIGN and plan['resources'] == RESOURCES
              and plan['limits'] == limits(plan['input_index']) and plan['output_HF_relative'] == OUTPUT
              and plan['automatic_retry'] is False, 'Exact factor replay contract differs')
    data.need(REQUIRED <= {b['path'] for b in plan['repo_sources']}, 'Replay software closure missing')
    if frozen:
        data.need(plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True
                  and plan['coordinator_review']['status'] == 'REVIEWED_CODE_DATA_SOFTWARE_CONTROLS', 'Unreviewed replay freeze')


def draft(hf):
    raw = data.read(ROOT/INPUT_PROTOCOL)
    base = data.read(ROOT/FACTOR_PROTOCOL)
    paths = {b['path'] for b in raw['repo_sources']} | {b['path'] for b in base['repo_sources']} | REQUIRED
    paths |= {INPUT_PROTOCOL, FACTOR_PROTOCOL, 'tools/e0_raw_bf16_full80_spectrum.py', 'tools/raw_bf16_full80_spectrum.py'}
    return {'id': ID, 'status': 'DRAFT_METADATA_BOUND_NOT_REGISTERED', 'assets_finalized': False,
            'automatic_retry': False, 'design': DESIGN, 'resources': RESOURCES, 'CPU_resources': CPU_RESOURCES,
            'limits': limits(raw['input_index']), 'output_HF_relative': OUTPUT,
            'source_input_protocol': data.binding(ROOT/INPUT_PROTOCOL, ROOT), 'input_index': raw['input_index'],
            'factor_video_protocol': data.binding(ROOT/FACTOR_PROTOCOL, ROOT),
            'factor_parent': base['factor_parent'], 'models': base['models'],
            'execution_raw_master_CPU_proof': base['execution_raw_master_CPU_proof'],
            'execution_private_worker_config_sha256': base['execution_private_worker_config_sha256'],
            'runtime_versions': base['runtime_versions'],
            'reused_factor_CPU_proof': {'RUN': data.binding(hf/FACTOR_CPU_DIR/'CPU/RUN.json', hf),
                                       'exit': data.binding(hf/FACTOR_CPU_DIR/'EXIT_RECEIPT.json', hf)},
            'repo_sources': [data.binding(ROOT/p, ROOT) for p in sorted(paths)],
            'competing_predictions': ['DISPROPORTIONATE_AC_LOSS_ON_SAME_DENSE_H_SUPPORTS_BOUNDARY_REPRESENTATION_LOSS',
                                     'PRESERVED_ALIGNED_AC_WEAKENS_LOCAL_LOSS_FAVORS_STUDENT_PREFIX_OR_MOTION_METRIC_ALTERNATIVES'],
            'source_N': 10, 'source_role': 'NEW_DEVELOPMENT_NO_BLIND_CONFIRMATION',
            'prior_CPU_geometry_result_reused': False, 'no_tensor_values_or_new_model_seen_during_design': True}


def authorize(plan, sha):
    validate(plan, True)
    active = data.read(ROOT/'research/ACTIVE_RESEARCH.json')
    data.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,
              'Current continuation authorization inactive')
    rows = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == ID]
    data.need(len(rows) == 1 and rows[0]['protocol_sha256'] == sha
              and rows[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
              and rows[0]['resources'] == RESOURCES and rows[0]['limits'] == plan['limits']
              and rows[0]['max_gpu_cards'] == 1, 'Unique exact fixed factor scope not registered')
    return rows[0]


def cpu_identity(plan, sha):
    return {'id': ID, 'protocol_sha256': sha, 'status': 'PASS_CLOSED_NEW10_MASTER_AND_REUSED_FACTOR_STAT_SOURCE_PROOF_NO_NUMERIC_IMPORTS',
            'source_input_protocol': plan['source_input_protocol'], 'reused_factor_CPU_proof': plan['reused_factor_CPU_proof'],
            'source_closure_sha256': native.canonical(plan['repo_sources']), 'source_N': 10,
            'model_calls': 0, 'GPU_calls': 0, 'numerical_imports': False}


def preflight(args):
    started = time.monotonic()
    plan, sha = data.read(args.protocol), native.sha(args.protocol)
    authorize(plan, sha)
    data.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
              and len(os.sched_getaffinity(0)) <= 4, 'Hidden CUDA fresh four CPU proof required')
    source_inputs(plan, args.hf_root)
    original.execution_master(plan, args.hf_root)
    rows = reused_factor_bytes(plan, args.hf_root)
    for b in plan['repo_sources']:
        native.bound(ROOT, b)
    result = {**cpu_identity(plan, sha), 'factor_bindings': rows,
              'factor_bindings_sha256': native.canonical(rows), 'wall_seconds': time.monotonic()-started}
    data.need(result['wall_seconds'] <= CPU_RESOURCES['wall_seconds'], 'CPU proof budget exceeded')
    return result


class FactorMeter:
    def __init__(self, meter):
        self.meter = meter
    def call(self, key, fn, *args, **kwargs):
        return self.meter.work({key: 1}, fn, *args, **kwargs)


def source_equal(values, np):
    valid = np.isfinite(values)
    def mean(a, axis):
        mask = np.isfinite(a)
        n = mask.sum(axis)
        result = np.full(n.shape, np.nan)
        np.divide(np.where(mask, a, 0).sum(axis), n, out=result, where=n > 0)
        return result, n
    def clean(x):
        return [clean(v) for v in x] if isinstance(x, list) else (None if isinstance(x, float) and not math.isfinite(x) else x)
    per_source, counts = mean(values.reshape(10, 160, -1), 1)
    all_source, source_n = mean(per_source, 0)
    joint, joint_n = mean(values, 0)
    return {'metric_labels': labels(), 'source_equal_all160': clean(all_source.tolist()),
            'source_equal_all160_source_N': source_n.tolist(), 'per_source_all160': clean(per_source.tolist()),
            'per_source_valid_positions': counts.tolist(), 'source_equal_step4_block40': clean(joint.tolist()),
            'joint_source_N': joint_n.tolist(), 'independent_image_clusters': 10,
            'weights': 'EQUAL_POSITION_WITHIN_SOURCE_THEN_EQUAL_UNIQUE_IMAGES; NULL_NOT_ZERO_IMPUTED'}


def run(args):
    started = time.monotonic()
    plan, sha = data.read(args.protocol), native.sha(args.protocol)
    scope = authorize(plan, sha)
    proof = scope['cpu_preflight']
    cpu = data.read(native.bound(args.hf_root, proof['RUN']))
    video.exited(data.read(native.bound(args.hf_root, proof['exit'])), proof['RUN'], sha, ID)
    data.need(all(cpu.get(k) == v for k, v in cpu_identity(plan, sha).items()), 'CPU proof identity differs')
    _, index, _ = source_inputs(plan, args.hf_root)
    master_proof = original.execution_master(plan, args.hf_root)
    rows = reused_factor_bytes(plan, args.hf_root)
    data.need(rows == cpu['factor_bindings'], 'Native worker differs from exited factor stat proof')
    for b in plan['repo_sources']:
        native.bound(ROOT, b)
    data.need(not {'torch', 'numpy'} & set(sys.modules) and len(os.sched_getaffinity(0)) <= 4,
              'Fresh four CPU native replay process required')
    out = args.hf_root/OUTPUT
    data.need(not out.exists(), 'Fresh output, no retry')
    out.mkdir(parents=True)
    output = data.OutputBudget(out, RESOURCES['new_HF_bytes']-(1 << 20))
    report = {'id': ID, 'protocol_sha256': sha, 'status': 'STARTED_FIXED_NEW10_SAME_DENSE_H_FACTOR_REPLAY_NO_RETRY',
              'automatic_retry': False, 'design': DESIGN, 'CPU_preflight': proof, 'completed_packets': 0,
              'actual_packet_bytes_read': 0, 'actual_input_IO': {}, 'passed_original_u_byte_neutrals': 0,
              'source_N': 10, 'full_model_flows': 0, 'new_videos': 0, 'fits': 0, 'decompositions': 0,
              'quality_or_speed_accepted': False, 'raw_weight_records': []}
    meter = data.Meter(report, output, plan['limits'])
    fm = FactorMeter(meter)
    monitor = None
    def save():
        report['wall_seconds'] = time.monotonic()-started
        output.json('RUN.json', report, emergency=True)
    def check():
        if monitor:
            monitor.check()
        data.need(time.monotonic()-started <= RESOURCES['wall_seconds'], 'Fixed replay wall1800 exhausted')
        if 'torch' in sys.modules:
            data.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'Replay GPU budget exceeded')
    def timeout(*_):
        raise TimeoutError('Fixed replay wall1800 exhausted; no retry')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(RESOURCES['wall_seconds'])
    save()
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file
        data.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'], 'Runtime versions differ')
        reservation, monitor, report['CUDA_reservation'] = startup.reserve_before_assets(torch)
        data.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'One assigned A100 required')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        output.json('EXECUTED_PROTOCOL.json', plan)
        output.json('SOURCE_BINDINGS.json', plan['repo_sources'])
        values = np.full((10, 4, 40, len(labels())), np.nan)
        models = {m['kind']: m for m in plan['models'] if m['kind'] in ('high', 'low')}
        master_run = data.read(native.bound(args.hf_root, master_proof['RUN']))
        stats = {r['kind']: r['stat'] for r in master_run['model_payloads']}
        headers = {kind: meter.work({'raw_headers': 1}, lambda model=model: native.raw_header_check(native.header(model['path'])))
                   for kind, model in models.items()}
        data.need(all(headers[k]['header_sha256'] == models[k]['header_sha256'] for k in models),
                  'Raw master actual header differs from current full-SHA proof')
        for expert, kind in enumerate(('high', 'low')):
            for block in range(40):
                check()
                weight, wr = meter.work({'raw_W2_tensors': 1, 'raw_tensor_bytes': 5120*13824*2,
                    'raw_H2D_tensors': 1, 'raw_H2D_bytes': 5120*13824*2}, original.read_raw_tensor,
                    torch, models[kind], headers[kind], f'blocks.{block}.ffn.2.weight', [5120, 13824], stats[kind], report['actual_input_IO'])
                bias, br = meter.work({'raw_bias_tensors': 1, 'raw_tensor_bytes': 5120*2,
                    'raw_H2D_tensors': 1, 'raw_H2D_bytes': 5120*2}, original.read_raw_tensor,
                    torch, models[kind], headers[kind], f'blocks.{block}.ffn.2.bias', [5120], stats[kind], report['actual_input_IO'])
                report['raw_weight_records'].extend((wr, br))
                banks = {}
                for row in (r for r in rows if r['expert'] == expert and r['block'] == block):
                    def load(row=row):
                        path = args.hf_root/row['binding']['path']
                        data.need(native.stat_record(path) == row['stat'], 'Factor changed before actual load')
                        tensors = load_file(str(path), device='cpu')
                        for key, tensor in tensors.items():
                            data.need(hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
                                      == row['tensors'][key]['raw_tensor_sha256'], 'Actual factor tensor bytes changed')
                        data.need(native.stat_record(path) == row['stat'], 'Factor changed during actual load')
                        return {k: t.to('cuda') for k, t in tensors.items()}
                    banks[row['family']] = meter.work({'factor_files': 1, 'factor_tensor_raw_SHA_checks': 3,
                        'factor_H2D_tensors': 3, 'factor_H2D_bytes': video.FACTOR_BYTES}, load)
                    meter.work({'factor_bias_anchor_checks': 1}, lambda family=row['family']:
                        data.need(torch.equal(banks[family]['right.bias'].view(torch.uint8), bias.view(torch.uint8)), 'Factor original b2 differs'))
                data.need(set(banks) == {'shared', 'step0', 'step1'}, 'Three all6 exports required')
                for step in (2*expert, 2*expert+1):
                    for case in index['cases']:
                        check()
                        entry = next(p for p in case['packets'] if p['step'] == step and p['block'] == block)
                        raw, _ = meter.work({'packet_full_SHA_reads': 1, 'packet_full_SHA_bytes': entry['payload']['bytes']},
                            data.stream_payload, args.hf_root, entry['payload'], report)
                        packet = meter.work({'tensor_raw_SHA_checks': 9, 'row_payload_checks': 1}, data.decode_packet,
                            raw, entry, data.core.rows(), np, data.SHAPES, whole_SHA_already_stream_verified=True)
                        meter.work({'original_u_byte_neutrals': 1, 'BF16_product_calls': 1, 'BF16_add_calls': 1},
                            data.core.neutral_postwrite, packet, np)
                        report['passed_original_u_byte_neutrals'] += 1
                        h_cpu = torch.from_numpy(packet['h']).to(torch.bfloat16)
                        meter.work({'h_CPU_BF16_reconstruction_checks': 1}, lambda:
                            data.need(hashlib.sha256(h_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
                                      == entry['tensor_metadata']['h']['raw_tensor_sha256'], 'Dense h reconstructed bytes differ'))
                        h = meter.work({'h_H2D_tensors': 1, 'h_H2D_bytes': 168*13824*2}, h_cpu.to, 'cuda')
                        ref = meter.work({'gY_FP64_products': 1, 'rowLN_FP64_calls': 1}, data.core.operands, packet, np)
                        scalar = []
                        for arm in ARMS:
                            yp = (meter.work({'BF16_original_W2_GEMM': 1}, torch.addmm, bias, h, weight.T)
                                  if arm == 'raw_M168' else factor.factor_forward(h, banks['shared' if arm == 'shared' else 'step'+str(step % 2)], fm))
                            host = meter.work({'predicted_Y_D2H_tensors': 1, 'predicted_Y_D2H_bytes': 168*5120*2}, lambda: yp.detach().cpu().contiguous())
                            y = (host.view(torch.uint16).numpy().astype(np.uint32) << 16).view(np.float32)
                            _, u = meter.work({'BF16_product_calls': 1, 'BF16_add_calls': 1}, data.core.postwrite, packet['r'], y, packet['gate'], np)
                            gy, ln = meter.work({'gY_FP64_products': 1, 'rowLN_FP64_calls': 1}, lambda:
                                (y.astype(np.float64)*packet['gate'].astype(np.float64), data.core.row_ln(u, np)))
                            for a, b in ((gy, ref['gY']), (u, packet['u']), (ln, ref['rowLN_u'])):
                                for view in data.core.VIEWS:
                                    scalar += meter.work({'temporal_error_evaluations': 1}, error_components,
                                        data.core.time_view(a.reshape(21, 8, -1), view), data.core.time_view(b.reshape(21, 8, -1), view), np)
                            del yp, host, y, u, gy, ln
                        data.need(len(scalar) == len(labels()) and all(x is None or math.isfinite(x) for x in scalar), 'Invalid finite error vector')
                        def record():
                            values[case['cohort_index'], step, block] = [np.nan if x is None else x for x in scalar]
                            output.append('POSITIONS.jsonl', (json.dumps({'source_id': case['source_id'], 'native_source_ordinal': case['native_source_ordinal'],
                                'step': step, 'block': block, 'payload': entry['payload'], 'values': scalar}, separators=(',', ':'), allow_nan=False)+'\n').encode())
                        meter.work({'source_position_records': 1}, record)
                        report['completed_packets'] += 1
                        report['last_completed_packet'] = [case['native_source_ordinal'], step, block]
                        save()
                        del raw, packet, h_cpu, h, ref, scalar
                torch.cuda.synchronize()
                del weight, bias, banks
        result = meter.work({'source_equal_finalizations': 1}, source_equal, values, np)
        meter.finish()
        data.need(report['completed_packets'] == report['passed_original_u_byte_neutrals'] == 1600, 'Incomplete fixed replay')
        result.update(id=ID, protocol_sha256=sha, design=DESIGN,
                      source_order=[{k: c[k] for k in ('source_id', 'native_source_ordinal', 'image_sha256')} for c in index['cases']])
        output.json('SUMMARY.json', result)
        report.update(status='COMPLETE_NEW10_SAME_DENSE_H_R960_TEMPORAL_COMPONENT_REPLAY',
                      result_files={name: data.binding(out/name, args.hf_root) for name in ('POSITIONS.jsonl', 'SUMMARY.json', 'operations.jsonl')},
                      peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        report.update(status='FAILED_FIXED_FACTOR_REPLAY_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        try:
            if monitor:
                monitor.close()
                report['assigned_GPU_resources'] = monitor.report()
                data.need(not report['assigned_GPU_resources']['violations'], 'Assigned GPU resource violation')
        except BaseException as error:
            report.update(status='FAILED_FIXED_FACTOR_REPLAY_PRESERVED_NO_RETRY', resource_error=str(error))
            raise
        finally:
            report['output_bytes_before_final_RUN_replace'] = output.final_audit()
            save()
            output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'preflight', 'run'))
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    args.hf_root = args.hf_root.resolve()
    if args.action == 'run':
        run(args)
    elif args.action == 'preflight':
        data.need(args.out is not None and not args.out.exists()
                  and args.out.resolve().is_relative_to(args.hf_root), 'Fresh confined CPU proof output required')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result = {'id': ID, 'protocol_sha256': native.sha(args.protocol), 'status': 'STARTED_FACTOR_REPLAY_CPU_PREFLIGHT'}
        args.out.write_text(json.dumps(result, indent=2)+'\n')
        try:
            result = preflight(args)
        except BaseException as error:
            result.update(status='FAILED_FACTOR_REPLAY_CPU_PREFLIGHT_NO_RETRY', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'status': result['status'], 'model_GPU_calls': 0}))
    else:
        result = draft(args.hf_root)
        data.need(args.out is not None and not args.out.exists(), 'Fresh output required')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps({'status': result['status'], 'model_GPU_calls': 0}))


if __name__ == '__main__':
    main()
