"""Matched r960 value/DC-AC calibration on new10; old8 image-disjoint diagnostics.

Two separately registered expert shards. No W1, flow, optimizer or video runs.
Source roles change only for this new fit; all earlier captures stay unchanged.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import e0_raw_bf16_full21tau_factor_replay as prior
import raw_bf16_temporal_balanced_fit as target

data, original, native, factor = prior.data, prior.original, prior.native, prior.factor
BASE_ID = 'E0_RAW_BF16_NEW10_VALUE_VS_TEMPORAL_BALANCED_R960_FIT_20261029'
FAMILIES = ('value10', 'dcac10')
RANK, FACTOR_BYTES = 960, prior.video.FACTOR_BYTES
PRIOR_PROTOCOL = 'config/e0_raw_bf16_full21tau_new10_fixed_factor_replay_20261029_v1.json'
OLD_PROTOCOL = 'config/e0_raw_bf16_full80_spectrum_20261029_v1.json'
CAL_BANK = 'config/e0_raw_bf16_new10_temporal_representation_calibration_bank_20261029_v1.json'
COVERAGE = 'research/20261029/RAW_BF16_NEW10_TEMPORAL_REPRESENTATION_COVERAGE_20261029.json'
RESOURCES = {'GPU': 1, 'CPU_threads': 4, 'GPU_reserved_max_bytes': 24 << 30,
             'host_available_min_bytes': 32 << 30, 'free_disk_min_bytes': 5 << 30,
             'wall_seconds': 1800, 'outer_seconds': 2100, 'new_HF_bytes': 4 << 30, 'automatic_retry': False}
CPU_RESOURCES = {'GPU': 0, 'CPU_threads': 4, 'wall_seconds': 180, 'outer_seconds': 240, 'new_HF_bytes': 8 << 20}
DESIGN = {'research_status': 'POST_HOC_EXPLORATORY_FROM_ACTUAL_NEW10_SAME_H_AC_LOSS',
          'fitting_images': 10, 'diagnostic_images': 8,
          'source_roles': 'NEW10_NOW_USED_FOR_CALIBRATION; OLD8_IMAGE_DISJOINT_FROM_THIS_FIT_ALREADY_EXPOSED_DIAGNOSTICS_NOT_BLIND',
          'functions': 'TWO_EXPERT_SHARDS_EACH40_BLOCKS_2_STEPS; JOIN_FULL80_PHYSICAL_AND160_SITES',
          'families': list(FAMILIES), 'rank': RANK, 'gain': 'FIXED_BETA1_NO_GAIN_OR_BIAS_FIT',
          'target': 'SAME_Y_MINUS_ORIGINAL_B2_SAME_SOURCE_STEP_RISK_WEIGHTS_SAME_RMS_TWO_STEP_GATE_SURROGATE',
          'value10': 'ORDINARY_OUTPUT_OBJECTIVE_ON_ALL168_ROWS',
          'dcac10': 'TAU0_ROW_WEIGHT_UNCHANGED; REST20_DC_AND_AC_EQUAL_RELATIVE_REFERENCE_RISK_MASS_WITH_ROWSPACE_TRANSFORM',
          'reference_normalization': 'ORIGINAL_Y_INCLUDING_B2_TIMES_ACTUAL_STEP_GATE; CANDIDATE_ERRORS_NEVER_SET_WEIGHTS',
          'deployment': 'ORIGINAL_W1_GELU_RETAINED_TWO_SAME_R960_BF16_GEMMS_ORIGINAL_B2_GATE_WRITE',
          'native_loss': 'BF16_FACTOR_EVALUATION_SEPARATE_FROM_CONTINUOUS_SAMPLE_GRAM_SURROGATE',
          'new10_rows': 'COMPLETE21TAU_8FIXEDXY_168_MATCHED_ALL160',
          'old8_rows': 'HISTORICAL256RANDOM_ROWS_NO_COMPLETE21TAU; NO_OLD8_TEMPORAL_COMPONENT_CLAIM',
          'model_flows': 0, 'new_videos': 0, 'SGD_updates': 0, 'rank_grid': False,
          'motion_correspondence': 'NOT_MEASURED', 'quality_or_speed_accepted': False, 'automatic_retry': False}
REQUIRED = {'tools/e0_raw_bf16_temporal_balanced_fit.py', 'tools/raw_bf16_temporal_balanced_fit.py',
            'tests/test_raw_bf16_temporal_balanced_fit.py', 'tests/test_e0_raw_bf16_temporal_balanced_fit.py',
            CAL_BANK, COVERAGE}


def scope_id(part):
    data.need(type(part) is int and part in (0, 1), 'Exactly high/low expert shards')
    return BASE_ID+'_PART'+str(part)


def output_relative(part):
    scope_id(part)
    return f'ffnds-derived/e0_raw_bf16_new10_temporal_balanced_fit_20261029/part{part}/attempt1/worker/results'


def load_inputs(plan, hf):
    source_plan = data.read(native.bound(ROOT, plan['prior_replay_protocol']))
    prior.validate(source_plan, True)
    proof = plan['prior_replay_result']
    run = data.read(native.bound(hf, proof['RUN']))
    prior.video.exited(data.read(native.bound(hf, proof['exit'])), proof['RUN'], plan['prior_replay_protocol']['sha256'], prior.ID)
    data.need(run['status'] == 'COMPLETE_NEW10_SAME_DENSE_H_R960_TEMPORAL_COMPONENT_REPLAY'
              and run['completed_packets'] == run['passed_original_u_byte_neutrals'] == 1600
              and not run['assigned_GPU_resources']['violations']
              and all(run['operation_ledger'][k] == source_plan['limits'] for k in ('predebited', 'observed')),
              'Closed prior complete same-h evidence required')
    _, train, _ = prior.source_inputs(source_plan, hf)
    old_plan = data.read(native.bound(ROOT, plan['old_corpus_protocol']))
    old = original.load_index(old_plan, hf)
    cal = data.read(native.bound(ROOT, plan['calibration_bank']))
    coverage = data.read(native.bound(ROOT, plan['coverage']))
    data.need(len(cal['sources']) == 10 and all(s['role'] == 'CALIBRATION' for s in cal['sources'])
              and {s['image_id'] for s in cal['sources']} == {s['image_sha256'] for s in train['cases']}
              and coverage['status'] == 'STATIC_BANK_COVERAGE_ONLY'
              and coverage['arms'][0]['status'] == 'SOURCE_CANDIDATES_PRESENT', 'New fitting role/source coverage differs')
    data.need(not ({c['image_sha256'] for c in train['cases']} & {c['image_sha256'] for c in old.values()})
              and source_plan['models'] == old_plan['models'] == plan['models'], 'Fit/diagnostic image leak or numeric checkpoint differs')
    return train, old


def limits(part, train, old):
    scope_id(part)
    new_packets = [p for c in train['cases'] for p in c['packets'] if p['step']//2 == part]
    old_packets = [p for c in old.values() for p in c['packets'] if p['step']//2 == part]
    data.need(len(new_packets) == 800 and len(old_packets) == 640, 'Exact per-expert data count required')
    return {'raw_headers': 1, 'raw_W2_tensor_reads': 40, 'raw_bias_tensor_reads': 40,
            'raw_tensor_bytes': 40*(5120*13824+5120)*2,
            'raw_H2D_tensors': 80, 'raw_H2D_bytes': 40*(5120*13824+5120)*2,
            'new_packet_reads': 800, 'new_packet_bytes': sum(p['payload']['bytes'] for p in new_packets),
            'new_tensor_raw_SHA_checks': 7200, 'new_row_checks': 800,
            'new_packet_H2D_tensors': 4000, 'new_packet_H2D_bytes': 800*(168*(13824+3*5120)+5120)*2,
            'old_packet_reads': 640, 'old_packet_bytes': sum(p['payload']['bytes'] for p in old_packets),
            'old_packet_H2D_tensors': 3200, 'old_packet_H2D_bytes': 640*original.PACKET_UPLOAD_BYTES,
            'teacher_postwrite_byte_neutrals': 1440,
            'fixed_gate_gain_calls': 4320, 'BF16_postwrite_calls': 4320,
            'row_target_constructions': 80, 'FP64_sample_Gram_GEMM': 80,
            'FP64_eigh': 80, 'FP64_basis_GEMM': 80, 'FP64_basis_orthogonality_GEMM': 80,
            'FP64_factor_compile_GEMM': 80, 'BF16_factor_GEMM': 5760,
            'factor_export_D2H_tensors': 240, 'factor_export_D2H_bytes': 80*FACTOR_BYTES,
            'BF16_export_files': 80, 'BF16_export_readbacks': 80,
            'factor_readback_H2D_tensors': 240, 'factor_readback_H2D_bytes': 80*FACTOR_BYTES,
            'source_case_metrics': 2880, 'training_temporal_component_metrics': 1600,
            'full_model_flows': 0, 'W1_calls': 0, 'new_videos': 0, 'SGD_updates': 0}


def draft(hf, part):
    base = data.read(ROOT/PRIOR_PROTOCOL)
    plan = {k: copy.deepcopy(base[k]) for k in
            ('models', 'execution_raw_master_CPU_proof', 'execution_private_worker_config_sha256', 'runtime_versions')}
    prior_out = hf/base['output_HF_relative']
    paths = {b['path'] for b in base['repo_sources']} | REQUIRED | {PRIOR_PROTOCOL, OLD_PROTOCOL}
    plan.update(id=scope_id(part), part=part, status='DRAFT_REVIEW_NOT_REGISTERED', assets_finalized=False,
                automatic_retry=False, design=DESIGN, resources=RESOURCES, CPU_resources=CPU_RESOURCES,
                output_HF_relative=output_relative(part), prior_replay_protocol=data.binding(ROOT/PRIOR_PROTOCOL, ROOT),
                prior_replay_result={'RUN': data.binding(prior_out/'RUN.json', hf),
                                     'exit': data.binding(prior_out.parent/'EXIT_RECEIPT.json', hf)},
                old_corpus_protocol=data.binding(ROOT/OLD_PROTOCOL, ROOT), calibration_bank=data.binding(ROOT/CAL_BANK, ROOT),
                coverage=data.binding(ROOT/COVERAGE, ROOT),
                repo_sources=[data.binding(ROOT/p, ROOT) for p in sorted(paths)])
    train, old = load_inputs(plan, hf)
    plan.update(limits=limits(part, train, old),
                fitting_images=[{k: c[k] for k in ('source_id', 'native_source_ordinal', 'image_sha256')} for c in train['cases']],
                diagnostic_images=[{k: old[o][k] for k in ('source_id', 'native_source_ordinal', 'image_sha256')} for o in original.ORDINALS])
    return plan


def validate(plan, frozen=False):
    data.need(plan['id'] == scope_id(plan['part']) and plan['design'] == DESIGN and plan['resources'] == RESOURCES
              and plan['output_HF_relative'] == output_relative(plan['part']) and plan['automatic_retry'] is False,
              'Fixed matched temporal representation scope differs')
    data.need(REQUIRED <= {b['path'] for b in plan['repo_sources']}, 'Fit closure incomplete')
    if frozen:
        data.need(plan['status'] == 'FROZEN_REGISTERED' and plan['assets_finalized'] is True
                  and plan['coordinator_review']['status'] == 'REVIEWED_CODE_DATA_SOFTWARE_CONTROLS', 'Fit not frozen after review')


def authorize(plan, sha):
    validate(plan, True)
    active = data.read(ROOT/'research/ACTIVE_RESEARCH.json')
    data.need(active['user_continuation_authorization_20261027']['new_scope_registration_allowed'] is True,
              'Current continuation inactive')
    rows = [s for s in active['experiment_authority']['current_user_authorized_scopes'] if s.get('id') == plan['id']]
    data.need(len(rows) == 1 and rows[0]['protocol_sha256'] == sha
              and rows[0]['execution_status'] in ('AUTHORIZED_NOT_STARTED', 'RUNNING_FIXED_SCOPE')
              and rows[0]['resources'] == RESOURCES and rows[0]['limits'] == plan['limits']
              and rows[0]['max_gpu_cards'] == 1, 'Exact per-expert fit scope not registered')
    return rows[0]


def cpu_identity(plan, sha):
    return {'id': plan['id'], 'protocol_sha256': sha, 'status': 'PASS_NEW10_CAL_OLD8_DIAGNOSTIC_DISJOINT_CURRENT_MASTER_STAT_SOURCE_ONLY',
            'source_closure_sha256': native.canonical(plan['repo_sources']), 'prior_replay_result': plan['prior_replay_result'],
            'fitting_images': plan['fitting_images'], 'diagnostic_images': plan['diagnostic_images'],
            'model_GPU_calls': 0, 'numerical_imports': False}


def preflight(args):
    start = time.monotonic()
    plan, sha = data.read(args.protocol), native.sha(args.protocol)
    authorize(plan, sha)
    data.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not {'torch', 'numpy'} & set(sys.modules)
              and len(os.sched_getaffinity(0)) <= 4, 'Fresh hidden-CUDA four CPU proof required')
    original.execution_master(plan, args.hf_root)
    train, old = load_inputs(plan, args.hf_root)
    data.need(limits(plan['part'], train, old) == plan['limits'], 'Source budgets changed')
    for b in plan['repo_sources']:
        native.bound(ROOT, b)
    result = {**cpu_identity(plan, sha), 'wall_seconds': time.monotonic()-start}
    data.need(result['wall_seconds'] <= CPU_RESOURCES['wall_seconds'], 'CPU wall budget exceeded')
    return result


def new_packet(torch, np, hf, entry, meter, report):
    raw, _ = meter.work({'new_packet_reads': 1, 'new_packet_bytes': entry['payload']['bytes']},
                         data.stream_payload, hf, entry['payload'], report)
    packet = meter.work({'new_tensor_raw_SHA_checks': 9, 'new_row_checks': 1}, data.decode_packet,
                         raw, entry, data.core.rows(), np, data.SHAPES, whole_SHA_already_stream_verified=True)
    def move():
        result = {}
        for k in ('h', 'Y', 'r', 'u', 'gate'):
            cpu = torch.from_numpy(packet[k]).to(torch.bfloat16)
            data.need(hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
                      == entry['tensor_metadata'][k]['raw_tensor_sha256'], 'Reconstructed train operand bytes differ')
            result[k] = cpu.to('cuda')
        return result
    return meter.work({'new_packet_H2D_tensors': 5,
                       'new_packet_H2D_bytes': (168*(13824+3*5120)+5120)*2}, move)


def run(args):
    start = time.monotonic()
    plan, sha = data.read(args.protocol), native.sha(args.protocol)
    scope = authorize(plan, sha)
    proof = scope['cpu_preflight']
    cpu = data.read(native.bound(args.hf_root, proof['RUN']))
    prior.video.exited(data.read(native.bound(args.hf_root, proof['exit'])), proof['RUN'], sha, plan['id'])
    data.need(all(cpu.get(k) == v for k, v in cpu_identity(plan, sha).items()), 'Actual CPU proof differs')
    train, old = load_inputs(plan, args.hf_root)
    master_proof = original.execution_master(plan, args.hf_root)
    data.need(limits(plan['part'], train, old) == plan['limits'] and not {'torch', 'numpy'} & set(sys.modules), 'Input budget/import differs')
    for b in plan['repo_sources']:
        native.bound(ROOT, b)
    out = args.hf_root/plan['output_HF_relative']
    data.need(not out.exists(), 'Fresh output, no automatic retry')
    out.mkdir(parents=True)
    monitor = None
    counter = original.infrastructure.WrittenBytes(out, RESOURCES['new_HF_bytes'], lambda: monitor.check() if monitor else None)
    journal = counter.open_journal(out/'OPERATIONS.jsonl')
    report = {'id': plan['id'], 'part': plan['part'], 'protocol_sha256': sha, 'status': 'STARTED_FIXED_MATCHED_VALUE_DC_AC_FIT_NO_RETRY',
              'automatic_retry': False, 'CPU_preflight': proof, 'design': DESIGN, 'functions': [],
              'actual_packet_bytes_read': 0, 'actual_input_IO': {}, 'full_model_flows': 0, 'new_videos': 0,
              'quality_or_speed_accepted': False}
    meter = data.Meter(report, journal_adapter(journal), plan['limits'])
    fm = NumericalMeter(meter)
    def save():
        report['wall_seconds'] = time.monotonic()-start
        counter.write_json(out/'RUN.json', report)
    def check():
        if monitor:
            monitor.check()
        counter.check()
        data.need(time.monotonic()-start <= RESOURCES['wall_seconds']
                  and shutil.disk_usage(out).free >= RESOURCES['free_disk_min_bytes'], 'Fit wall/disk budget exceeded')
        available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                             if s.startswith('MemAvailable:')))*1024
        data.need(available >= RESOURCES['host_available_min_bytes'], 'Fit available host RAM below reserve')
        if 'torch' in sys.modules:
            data.need(sys.modules['torch'].cuda.max_memory_reserved() <= RESOURCES['GPU_reserved_max_bytes'], 'Fit GPU24GiB exceeded')
    def timeout(*_):
        raise TimeoutError('Fixed fit wall1800 reached; no retry')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(RESOURCES['wall_seconds'])
    save()
    try:
        import numpy as np
        import torch
        data.need({k: importlib.metadata.version(k) for k in plan['runtime_versions']} == plan['runtime_versions'], 'Runtime versions differ')
        reservation, monitor, report['CUDA_reservation'] = prior.startup.reserve_before_assets(torch)
        data.need(torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0), 'One assigned A100 required')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        report['actual_runtime_precision'] = {'allow_tf32': False, 'float32_matmul_precision': 'highest',
                                              'bf16_reduced_precision_reduction': True}
        counter.write_json(out/'EXECUTED_PROTOCOL.json', plan)
        part = plan['part']
        kind = 'high' if part == 0 else 'low'
        model = next(m for m in plan['models'] if m['kind'] == kind)
        master = data.read(native.bound(args.hf_root, master_proof['RUN']))
        stat = next(m['stat'] for m in master['model_payloads'] if m['kind'] == kind)
        header = meter.work({'raw_headers': 1}, lambda: native.raw_header_check(native.header(model['path'])))
        data.need(header['header_sha256'] == model['header_sha256'], 'Raw header differs')
        for block in range(40):
            check()
            cases, identities = {}, {}
            for source, case in enumerate(train['cases']):
                identities[source] = {**{k: case[k] for k in ('source_id', 'native_source_ordinal', 'image_sha256')}, 'role': 'TRAIN_NEW10'}
                for local_step in (0, 1):
                    step = 2*part+local_step
                    entry = next(p for p in case['packets'] if p['step'] == step and p['block'] == block)
                    cases[source, local_step] = new_packet(torch, np, args.hf_root, entry, meter, report)
            for source, ordinal in enumerate(original.ORDINALS, 10):
                case = old[ordinal]
                identities[source] = {**{k: case[k] for k in ('source_id', 'native_source_ordinal', 'image_sha256')}, 'role': 'IMAGE_DISJOINT_EXPOSED_DIAGNOSTIC_OLD8'}
                for local_step in (0, 1):
                    step = 2*part+local_step
                    entry = next(p for p in case['packets'] if p['step'] == step and p['block'] == block)
                    cases[source, local_step] = meter.work({'old_packet_reads': 1, 'old_packet_bytes': entry['payload']['bytes'],
                        'old_packet_H2D_tensors': 5, 'old_packet_H2D_bytes': original.PACKET_UPLOAD_BYTES}, original.packet,
                        torch, args.hf_root, entry, io=report['actual_input_IO'])
            for case in cases.values():
                u = factor.postwrite(case['r'], case['Y'], case['gate'], 1., fm)
                meter.work({'teacher_postwrite_byte_neutrals': 1}, lambda case=case, u=u:
                    data.need(torch.equal(u.view(torch.uint8), case['u'].view(torch.uint8)), 'Original u byte neutral failed before this function fit'))
            for local_step in (0, 1):
                data.need(all(torch.equal(cases[s, local_step]['gate'].view(torch.uint8), cases[0, local_step]['gate'].view(torch.uint8))
                              for s in range(18)), 'Actual source-static gate differs')
            weight, wr = meter.work({'raw_W2_tensor_reads': 1, 'raw_tensor_bytes': 5120*13824*2,
                'raw_H2D_tensors': 1, 'raw_H2D_bytes': 5120*13824*2}, original.read_raw_tensor,
                torch, model, header, f'blocks.{block}.ffn.2.weight', [5120, 13824], stat, report['actual_input_IO'])
            bias, br = meter.work({'raw_bias_tensor_reads': 1, 'raw_tensor_bytes': 5120*2,
                'raw_H2D_tensors': 1, 'raw_H2D_bytes': 5120*2}, original.read_raw_tensor,
                torch, model, header, f'blocks.{block}.ffn.2.bias', [5120], stat, report['actual_input_IO'])
            metric = factor.metrics([cases[0, t]['gate'] for t in (0, 1)])['shared']
            scaled = weight.T.double()*metric
            record = {'expert': kind, 'block': block, 'raw_tensors': [wr, br], 'fits': []}
            for family in FAMILIES:
                z, weights = meter.work({'row_target_constructions': 1}, target.anchored_rows,
                    cases, bias, metric, steps=(0, 1), source_indices=list(range(10)), balance=family == 'dcac10')
                gram = factor.sample_gram(z, fm)
                basis, spectrum = factor.fit_basis(z, gram, RANK, fm)
                factors = factor.compile_factors(scaled, bias, metric, basis, fm)
                deployed, exported = original.export(torch, counter, args.hf_root, out, f'{kind}_block{block:02d}_{family}', factors, fm)
                fit = {'family': family, 'training_source_ids': [identities[s]['source_id'] for s in range(10)],
                       'source_normalization_and_temporal_weights': weights, 'spectrum': spectrum, 'export': exported, 'evaluations': []}
                for source in range(18):
                    for local_step in (0, 1):
                        case = cases[source, local_step]
                        y = factor.factor_forward(case['h'], deployed, fm)
                        u = factor.postwrite(case['r'], y, case['gate'], 1., fm)
                        metrics = meter.work({'source_case_metrics': 1}, factor.row_statistics,
                            case['r'], case['Y'], case['u'], case['gate'], y, u)
                        if source < 10:
                            metrics.update(meter.work({'training_temporal_component_metrics': 1}, target.temporal_metrics,
                                                      case['Y'], y, case['gate']))
                        fit['evaluations'].append({**identities[source], 'global_step': 2*part+local_step, **metrics})
                        del y, u
                record['fits'].append(fit)
                del z, gram, basis, spectrum, factors, deployed, fit
            path = out/f'{kind}_block{block:02d}.json'
            counter.write_json(path, record)
            report['functions'].append({'expert': kind, 'block': block, 'record': counter.binding(path, args.hf_root)})
            save()
            torch.cuda.synchronize()
            del cases, weight, bias, metric, scaled, record
        meter.finish()
        data.need(len(report['functions']) == 40, 'Incomplete full40-function expert shard')
        report.update(status='COMPLETE_EXPERT40_MATCHED_VALUE_DC_AC_R960_BANKS', exported_factors=80,
                      peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(), peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
        check()
    except BaseException as error:
        report.update(status='FAILED_TEMPORAL_MATCHED_FIT_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        try:
            if monitor:
                monitor.close()
                report['assigned_GPU_resources'] = monitor.report()
                data.need(not report['assigned_GPU_resources']['violations'], 'Assigned fit GPU resource violation')
        except BaseException as error:
            report.update(status='FAILED_TEMPORAL_MATCHED_FIT_PRESERVED_NO_RETRY', resource_error=str(error))
            raise
        finally:
            journal.close()
            report['output_accounting'] = counter.summary()
            save()


class journal_adapter:
    """Use the existing predebit CPU Meter with incremental hashed native I/O."""
    def __init__(self, journal):
        self.journal = journal
    def append(self, name, raw):
        data.need(name == 'operations.jsonl', 'Unexpected meter journal')
        self.journal.write(raw.decode())


class NumericalMeter(prior.FactorMeter):
    def work(self, costs, fn, *args, **kwargs):
        return self.meter.work(costs, fn, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'preflight', 'run'))
    parser.add_argument('--part', type=int, choices=(0, 1))
    parser.add_argument('--hf-root', type=Path, default=Path('/home/xiang/.cache/huggingface'))
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    args.hf_root = args.hf_root.resolve()
    if args.action == 'run':
        run(args)
    elif args.action == 'draft':
        data.need(args.out is not None and not args.out.exists(), 'Fresh fit draft required')
        value = draft(args.hf_root, args.part)
        args.out.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps({'id': value['id'], 'status': value['status'], 'model_GPU_calls': 0}))
    else:
        data.need(args.out is not None and not args.out.exists() and args.out.resolve().is_relative_to(args.hf_root), 'Fresh confined CPU proof required')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        value = {'id': data.read(args.protocol)['id'], 'protocol_sha256': native.sha(args.protocol), 'status': 'STARTED_MATCHED_TEMPORAL_FIT_CPU_PROOF'}
        args.out.write_text(json.dumps(value)+'\n')
        try:
            value = preflight(args)
        except BaseException as error:
            value.update(status='FAILED_MATCHED_TEMPORAL_FIT_CPU_PROOF_NO_RETRY', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            args.out.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'id': value['id'], 'status': value['status'], 'model_GPU_calls': 0}))


if __name__ == '__main__':
    main()
