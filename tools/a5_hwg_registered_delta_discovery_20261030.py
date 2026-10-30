"""Exact-reviewed finite full-M teacher-state discovery, never a video run.

No registration, hardware selection or retry occurs here. Each source needs
its own reviewed immutable protocol and actual verified producer closure.
All projections below are paid dense FP32 diagnostics, not native skipping.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import a5_hwg_matched_capture_20261030 as guard
from a5_hwg_registered_static_fit_20261030 import authorize_common, validate_parent, CALIBRATION_SOURCES, CALIBRATION_KIND
from a5_hwg_evidence_io_20261030 import bound_json, bound_tensors, checkpoint_biases, CallLedger
from a5_hwg_matched_capture_namespace_v2_20261030 import validate_import_origins

SELF = 'tools/a5_hwg_registered_delta_discovery_20261030.py'
RES = {
    'GPU': 1, 'CPU_threads': 4, 'session_max_GPUs': 4,
    'GPU_reserved_max_bytes': 76 << 30, 'host_available_min_bytes': 128 << 30,
    'peak_RSS_max_bytes': 96 << 30, 'model_forecast_bytes': 64 << 30,
    'host_max_models': 2, 'new_HF_bytes': 32 << 30,
    'free_disk_min_bytes': 36 << 30, 'maximum_single_file_bytes': 512 << 20,
    'asset_validation_read_max_bytes': 256 << 30,
    'wall_seconds': 21600, 'outer_seconds': 21720, 'automatic_retry': False,
}
CALL_LIMITS = {name: count * 80 for name, count in {
    'effective_operands_to_GPU': 1,
    'native_W1': 2, 'native_GELU': 2, 'native_smooth2': 2, 'native_Q2': 2, 'native_W2': 2,
    'native_sample_verify': 2, 'proxy_weight_parameters': 1,
    'native_input_H2D': 2, 'current_gate_H2D': 1,
    'proxy_old_W1': 1, 'proxy_new_W1': 1, 'proxy_old_GELU': 1, 'proxy_new_GELU': 1,
    'current_gate_W2_sensitivity': 1,
    'proxy_score_field_algebra': 1,
    'p64_input_gather': 1, 'p64_history_gather': 1, 'p64_W1': 1, 'p64_GELU': 1, 'p64_rawh_RMS': 1,
    'mean_m64_old_input': 1, 'mean_m64_new_input': 1,
    'mean_m64_old_W1': 1, 'mean_m64_new_W1': 1,
    'mean_m64_old_GELU': 1, 'mean_m64_new_GELU': 1,
    'mean_score_field_and_expand': 1, 'proxy_parameters_D2H': 1,
    'paid_delta_and_auxiliary_scores': 1, 'global_ranking_diagnostic': 1,
    'full_delta_W2_FP32': 1,
    'support_select': 58, 'support_statistics': 58,
    'candidate_mask_pack_and_D2H': 57, 'candidate_W2_FP32': 57,
    'signed_output_diagnostic': 58, 'latent_feature_AC': 58, 'sample_postwrite': 58,
}.items()}
CALL_LIMITS['coordinate_setup_H2D'] = 1
guard.RES = RES


def authorize(protocol, protocol_path, job_path):
    authorize_common(protocol, protocol_path, job_path, RES, CALL_LIMITS,
                     'FULLM_ALL80_58ARM_HIF4_TEACHER_DISCOVERY_V1')
    guard.need(protocol['keep_channels'] == 5120 and protocol['grid'] == [21, 29, 52]
               and protocol['P64_seed'] == 42 and protocol['shuffle_seed'] == 101
               and protocol['random_group_seed_base'] == 42000,
               'exact full-M scientific scope')
    from a5_hwg_e0_native_replay_20261030 import arm_specs
    guard.need(protocol['arms'] == arm_specs(), 'exact arm identities')


def load_static(protocol, hf, budget):
    """Require the actual independent calibration RUN/EXIT before dev reads."""
    p = protocol['static_calibration']
    run = bound_json(hf, p['RUN'], budget)
    receipt = bound_json(hf, p['EXIT'], budget)
    original = guard.read(guard.check_binding(ROOT, p['protocol']))
    guard.need(original['id'] == p['id'] and original['kind'] == CALIBRATION_KIND
               and original['calibration_source_ordinals'] == CALIBRATION_SOURCES
               and original['cache'] == protocol['cache']
               and run['id'] == receipt['id'] == p['id']
               and run['protocol_sha256'] == receipt['protocol_sha256'] == p['protocol']['sha256']
               and receipt['spec_sha256'] == p['job_spec_sha256']
               and receipt['process_exited'] is True and receipt['child_returncode'] == 0
               and receipt['timeout'] is False and receipt['automatic_retry'] is False
               and receipt['run_sha256'] == p['RUN']['sha256']
               and run['status'] == 'COMPLETE_CAL2_SIGNED_GREEDY_FULL80_EXIT_PENDING'
               and run['completed_sites'] == 80 and run['dev_tensor_files_read'] == 0,
               'actual frozen CAL2 artifact closure')
    guard.need(run['static_support'] == p['artifact']
               and run['static_support_schema'] == p['schema'], 'frozen static artifact differs')
    values = bound_tensors(hf, p['artifact'], budget, p['schema'])
    import torch
    selected = values['selected_groups']
    guard.need(selected.dtype == torch.bool and list(selected.shape) == [2, 40, 216]
               and bool((selected.sum(-1) == 80).all()), 'exact whole-K64 static support')
    return selected


def ranking_diagnostic(scores):
    """Whole-M K64 score-rank agreement with the paid diagonal selector."""
    import torch
    energies = {name: score.float().square().sum(0).reshape(216, 64).sum(-1)
                for name, score in scores.items() if name in
                ('a5_current_hwg', 'a5_delta_hwg', 'a5_delta_nohwg', 'p64_rawh', 'paid_delta_diagonal')}
    orders = {name: torch.argsort(value, descending=True, stable=True) for name, value in energies.items()}
    oracle_order = orders['paid_delta_diagonal']
    reference_rank = oracle_order.argsort().double()
    result = {}
    for name, order in orders.items():
        rank = order.argsort().double()
        a, b = rank - rank.mean(), reference_rank - reference_rank.mean()
        selected, paid = torch.zeros_like(rank, dtype=torch.bool), torch.zeros_like(rank, dtype=torch.bool)
        selected[order[:80]], paid[oracle_order[:80]] = True, True
        result[name] = {'stable_tie_rank_correlation': float((a * b).sum() / (a.square().sum() * b.square().sum()).sqrt()),
                        'K64_top80_support_Jaccard': float((selected & paid).sum() / (selected | paid).sum()),
                        'paid_reference_is_signed_optimal': False}
    return result


def sample_postwrite(old_y, retained, rows, sampled):
    """Actual sampled BF16 gate/residual branch, not a full-flow consumer."""
    import torch
    predicted_y = (old_y.index_select(0, rows).float() + retained.index_select(0, rows)).to(torch.bfloat16).cpu()
    gate = sampled['gate_broadcast'].reshape(1, -1)
    predicted_delta = predicted_y * gate
    predicted_next = sampled['R'].clone().add_(predicted_delta)
    def relative(left, right):
        denominator = right.double().square().sum()
        return float(((left.double() - right.double()).square().sum() / denominator).sqrt()) if float(denominator) else None
    return {'rows': rows.numel(), 'original_dtype': 'bfloat16',
            'gated_writeback_relative_L2': relative(predicted_delta, sampled['Delta_writeback_operand']),
            'post_residual_relative_L2': relative(predicted_next, sampled['X_next']),
            'evidence': 'ACTUAL_1344_TEACHER_RESIDUAL_ROWS_NOT_DOWNSTREAM_FLOW_OR_VIDEO'}


def preflight_native_root(name, expected_root):
    """Reject a shadow package before importing any native adapter callable."""
    spec = importlib.util.find_spec(name)
    guard.need(spec is not None and spec.submodule_search_locations is not None,
               'native root is not a package: ' + name)
    roots = [Path(v).resolve() for v in spec.submodule_search_locations]
    guard.need(roots == [Path(expected_root).resolve()], 'native root search path differs: ' + name)
    if spec.origin is not None:
        guard.need(Path(spec.origin).resolve().parent == Path(expected_root).resolve(),
                   'native root origin differs: ' + name)


@guard.execution_bootstrap('run')
def run(args):
    need = guard.need
    started, budget = guard.CURRENT_BOOTSTRAP.started, guard.CURRENT_BOOTSTRAP.budget
    p = guard.read(args.protocol)
    authorize(p, args.protocol, args.job_spec)
    private = guard.private_assets(p, args.private_assets)
    need(sorted(os.sched_getaffinity(0)) == p['cpu_affinity'], 'registered CPU affinity')
    hf = Path(private['hf_root']).resolve()
    out = hf / p['output_HF_relative']
    need(not out.exists(), 'fresh single attempt result directory')
    out.mkdir()
    report = {'id': p['id'], 'protocol_sha256': guard.sha(args.protocol), 'status': 'RUNNING',
              'kind': p['kind'], 'source': p['source']['case'], 'completed_pairs': 0,
              'numeric_family': 'HIF4_FAKE_QDQ_BF16_NATIVE_REFERENCE_PAID_FP32_DELTA_DIAGNOSTIC',
              'automatic_retry': False, 'new_videos': 0, 'native_skip_or_speed_claim': False,
              'actual_fullM_original_Y_observed': False, 'actual_original_boundary_rows_per_site': 1344,
              'site_manifest': [], 'call_record_files': []}
    monitor = None
    def check():
        need(time.monotonic() - started <= RES['wall_seconds'], 'full-M discovery wall budget')
        need(guard.available_memory() >= RES['host_available_min_bytes'], 'full-M host reserve')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= RES['peak_RSS_max_bytes'], 'full-M RSS')
        need(shutil.disk_usage(out).free >= 4 << 30, 'full-M disk reserve')
        if monitor: monitor.check()
        if monitor and 'torch' in sys.modules:
            need(sys.modules['torch'].cuda.max_memory_reserved() <= RES['GPU_reserved_max_bytes'], 'full-M CUDA reserved')
    budget.check = check
    writer = guard.Writer(out, hf, check)
    writer.holds['journal'] = 128 << 20
    persist = lambda: writer.json(out / 'RUN.json', report)
    persist()
    try:
        with guard.HostReservation(), guard.bounded_files():
            need(shutil.disk_usage(out).free >= RES['free_disk_min_bytes'], 'full-M initial disk reserve')
            for name, version in p['runtime_versions'].items():
                need(importlib.metadata.version(name) == version, 'runtime version: ' + name)
            import torch
            import torch.nn.functional as F
            torch.set_num_threads(4)
            torch.set_num_interop_threads(4)
            torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            torch.set_float32_matmul_precision('highest')
            from startup_gpu_reservation import reserve_before_assets
            reservation, monitor, report['CUDA_reservation'] = reserve_before_assets(torch)
            from a5_hwg_e0_native_replay_20261030 import (native_replay, verify_actual_samples, proxy_scores,
                p64_score, mean_m64_score, arm_specs, arm_mask, pack_selected, latent_ac_metrics)
            from a5_hwg_support_regularity_math_20261030 import support_summary, signed_output_metrics, four_step_cost_identity
            static = load_static(p, hf, budget)
            validate_parent(p['source'], hf, budget, p['cache'])
            cache = Path(private['cache_path'])
            need(cache.stat().st_size == p['cache']['bytes'] and guard.sha(cache) == p['cache']['sha256'], 'actual cache SHA')
            writer.note('CACHE_MMAP_LOAD_BEGIN_INTERNAL_IO_UNKNOWN', declared_file_bytes=p['cache']['bytes'])
            cache_values = torch.load(cache, map_location='cpu', weights_only=False, mmap=True)
            writer.note('CACHE_MMAP_LOAD_RETURN_INTERNAL_IO_UNKNOWN', status='COMPLETE')
            report['cache_loader'] = {'calls': 1, 'mmap': True, 'internal_IO_bytes': 'UNKNOWN_SEPARATE_FROM_EXPLICIT_SHA'}
            view = guard.read(guard.check_binding(ROOT, p['model_view']))
            biases = {}
            for expert in ['high_noise_model', 'low_noise_model']:
                expected = p['bias_checkpoints'][expert]
                actual = next(r for r in view['files'] if r['sha256'] == expected['sha256'])
                need(actual['bytes'] == expected['bytes'], 'original checkpoint binding')
                biases[expert] = checkpoint_biases(Path(actual['view_path']), expected, budget, guard.sha)
            for path in [ROOT / '.runtime/code/HiF4_Sp', ROOT / '.runtime/code/Wan2.2-I2V-A14B-W4A4']:
                sys.path.insert(0, str(path))
            os.environ['HIF4_MAIN_REPO'] = str(ROOT / '.runtime/code/HiF4_Sp')
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            preflight_native_root('HiFloat4', ROOT / '.runtime/code/HiF4_Sp/HiFloat4')
            preflight_native_root('quant', ROOT / '.runtime/code/Wan2.2-I2V-A14B-W4A4/quant')
            from HiFloat4.ptq.adapters.wan_i2v import native_hif4_activation_qdq
            from quant import hif4_backend
            hif4_backend.get_hif4_api()
            report['native_import_origins_before_first_replay'] = validate_import_origins(p)
            ledger = CallLedger(CALL_LIMITS, writer.note, check, torch)
            rows_cpu = torch.tensor(guard.matched_rows(p['grid'], p['source']['case']['native_source_ordinal'], 42)[0])
            p64_cpu = torch.randperm(31668, generator=torch.Generator().manual_seed(42))[:64]
            shuffle_cpu = torch.randperm(31668, generator=torch.Generator().manual_seed(101))
            rows, p64_rows, shuffle = ledger('coordinate_setup_H2D', lambda:
                (rows_cpu.cuda(), p64_cpu.cuda(), shuffle_cpu.cuda()),
                matched_rows=1344, P64_rows=64, shuffle_rows=31668, dtype='int64')
            report['row_coordinates'] = writer.tensor(out / 'COORDINATES.safetensors',
                                                      {'matched': rows_cpu, 'P64': p64_rows.cpu(), 'shuffle': shuffle_cpu})
            seen = {'paths': set(), 'inodes': set()}
            for ei, expert in enumerate(['high_noise_model', 'low_noise_model']):
                steps = [0, 1] if ei == 0 else [2, 3]
                for block in range(40):
                    check()
                    site = {'expert': expert, 'block': block, 'steps': steps,
                            'M': 31668, 'D': 5120, 'H': 13824,
                            'source_ordinal': p['source']['case']['native_source_ordinal']}
                    ledger.context = site
                    packets, metadata = [], []
                    for step in steps:
                        ref = p['source']['metadata'][f'step{step}_block{block:02d}']
                        meta = bound_json(hf, ref, budget)
                        guard.validate_packet(meta, {'step': step, 'block': block, 'metadata': ref},
                            {'case': p['source']['case'], 'cache': p['cache']}, hf,
                            hf / p['source']['results_HF_relative'], seen)
                        need(meta['post_process_branch'] == 'infer_add_', 'original BF16 postwrite branch')
                        sampled = bound_tensors(hf, meta['payload'], budget, meta['tensor_schema'])
                        full = bound_tensors(hf, meta['full_V1'], budget,
                            {'V1': {'shape': [31668, 5120], 'dtype': 'bfloat16',
                                    'raw_tensor_sha256': meta['full_V1']['raw_tensor_sha256']}})
                        packets.append((sampled, full['V1']))
                        metadata.append(meta)
                    c1, c2 = cache_values[expert][f'blocks.{block}.ffn.0'], cache_values[expert][f'blocks.{block}.ffn.2']
                    w1c, w2c = c1['weight_qdq'], c2['weight_qdq']
                    b1c, b2c = biases[expert][f'blocks.{block}.ffn.0.bias'], biases[expert][f'blocks.{block}.ffn.2.bias']
                    need(w1c.dtype == w2c.dtype == torch.bfloat16
                         and list(w1c.shape) == [13824, 5120] and list(w2c.shape) == [5120, 13824], 'effective W geometry')
                    weight_hash = writer.raw_sha(w2c.contiguous())
                    need(all(m['effective_W2']['runtime_operand_sha256'] == weight_hash for m in metadata), 'actual captured W2')
                    need(all(torch.equal(sampled['b2'], b2c) for sampled, _ in packets), 'actual checkpoint/runtime b2')
                    smooth = c2['smooth_scale']
                    need(smooth is not None and smooth.numel() == 13824 and bool(torch.isfinite(smooth).all())
                         and bool((smooth > 0).all()), 'original positive smooth2')
                    w1, b1, w2, b2, smooth2 = ledger('effective_operands_to_GPU',
                        lambda: tuple(v.cuda() for v in (w1c, b1c, w2c, b2c, smooth)),
                        W1=[13824, 5120], W2=[5120, 13824], dtype='bfloat16_weights_FP32_smooth2')
                    replayed, native_checks = [], []
                    for sampled, v1 in packets:
                        native_v1 = ledger('native_input_H2D', lambda: v1.cuda(), bytes=31668 * 5120 * 2)
                        native = native_replay(native_v1, w1, b1, w2, b2, smooth2, native_hif4_activation_qdq, ledger)
                        native_checks.append(ledger('native_sample_verify', lambda: verify_actual_samples(native, sampled, rows)))
                        replayed.append(native)
                    old, current = replayed
                    gate = ledger('current_gate_H2D', lambda: packets[1][0]['gate_broadcast'].cuda().reshape(-1), bytes=5120 * 2)
                    scores, proxy = proxy_scores(old['V1'], current['V1'], w1, b1, smooth2, w2, gate, ledger)
                    p64 = p64_score(current['V1'], old['H'], w1, b1, p64_rows, ledger)
                    mean_score = mean_m64_score(old['V1'], current['V1'], proxy['weight'], b1,
                                               smooth2, proxy['sensitivity'], ledger)
                    def auxiliary_scores():
                        delta = current['V2'].float() - old['V2'].float()
                        random = torch.randperm(216, generator=torch.Generator().manual_seed(42000 + ei * 40 + block)).argsort().cuda()
                        scores.update(paid_delta_diagonal=delta.abs() * proxy['sensitivity'],
                            p64_rawh=p64.reshape(1, -1).expand(31668, -1),
                            calibration_static=static[ei, block].float().repeat_interleave(64).cuda().reshape(1, -1).expand(31668, -1),
                            random=random.float().repeat_interleave(64).reshape(1, -1).expand(31668, -1),
                            mean_m64=mean_score, oldY=delta.new_zeros(1, 13824).expand(31668, -1))
                        return delta
                    delta = ledger('paid_delta_and_auxiliary_scores', auxiliary_scores)
                    ranking = ledger('global_ranking_diagnostic', lambda: ranking_diagnostic(scores))
                    full = ledger('full_delta_W2_FP32', lambda: F.linear(delta, w2.float()),
                                  shape=[31668, 13824, 5120], TF32=False)
                    metrics, packed = [], {}
                    for spec in arm_specs():
                        ledger.context = {**site, 'arm': spec['name'], 'is_native_skip': False}
                        mask, indices = ledger('support_select', lambda: arm_mask(spec, scores, 5120, shuffle))
                        statistics = ledger('support_statistics', lambda: support_summary(mask, grid=p['grid']))
                        if indices is not None:
                            packed[spec['name']] = ledger('candidate_mask_pack_and_D2H',
                                lambda: pack_selected(indices, 13824 // spec['hidden_group']).cpu())
                            retained = ledger('candidate_W2_FP32', lambda: F.linear(delta * mask, w2.float()),
                                              shape=[31668, 13824, 5120], zero_masked_dense=True, TF32=False)
                        else:
                            retained = torch.zeros_like(full)
                        quality = ledger('signed_output_diagnostic', lambda: signed_output_metrics(
                            old['Y'], current['Y'], delta, w2, gate, mask,
                            full_delta_output=full, retained_delta_output=retained))
                        ac = ledger('latent_feature_AC', lambda: latent_ac_metrics(
                            (current['Y'].float() - old['Y'].float()) * gate,
                            retained * gate, p['grid']))
                        post = ledger('sample_postwrite', lambda: sample_postwrite(old['Y'], retained, rows, packets[1][0]))
                        metrics.append({'spec': spec, 'support': statistics, 'signed_output': quality,
                                        'feature_AC': ac, 'sample_postwrite': post})
                        del mask, indices, retained
                    packed.update(ledger('proxy_parameters_D2H', lambda:
                        {'proxy_gamma': proxy['gamma'].cpu(), 'proxy_beta': proxy['beta'].cpu(),
                         'current_gate_W2_sensitivity': proxy['sensitivity'].cpu()}))
                    artifact = writer.tensor(out / f'{expert}_block{block:02d}_MASKS_AND_PROXY.safetensors', packed)
                    result = {**site, 'native_sample_checks': native_checks,
                              'actual_effective_W2_raw_SHA': weight_hash,
                              'effective_W1_raw_SHA': writer.raw_sha(w1c.contiguous()),
                              'original_bias_raw_SHA': [writer.raw_sha(b1c), writer.raw_sha(b2c)],
                              'smooth2_raw_SHA': writer.raw_sha(smooth.contiguous()),
                              'mask_and_proxy': artifact, 'mask_unit_layout': 'LITTLE_ENDIAN_UINT8_ROWGROUP_UNITS',
                              'mask_schema': {name: {'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.'),
                                                    'raw_tensor_sha256': writer.raw_sha(value.contiguous())} for name, value in packed.items()},
                              'global_ranking_vs_paid_diagonal': ranking, 'arms': metrics,
                              'threshold_original_arm': 'MISSING_ORIGINAL_PARAMETER_PRODUCER_NOT_REPLACED_BY_ADAPTER',
                              'four_step_main_MAC_algebra': four_step_cost_identity(5120, proxy_full_w1_calls=4)}
                    writer.json(out / f'{expert}_block{block:02d}_RESULT.json', result)
                    # Writer.json intentionally returns no ref; bind its already
                    # charged durable output identity without a second file read.
                    path = out / f'{expert}_block{block:02d}_RESULT.json'
                    record = writer.files[str(path.resolve())]
                    report['site_manifest'].append({'expert': expert, 'block': block,
                        'result': {'path': str(path.relative_to(hf)), 'bytes': record['bytes'], 'sha256': record['sha256']}})
                    calls_path = out / f'{expert}_block{block:02d}_CALLS.json'
                    writer.json(calls_path, {'records': ledger.records})
                    cr = writer.files[str(calls_path.resolve())]
                    report['call_record_files'].append({'path': str(calls_path.relative_to(hf)), 'bytes': cr['bytes'], 'sha256': cr['sha256']})
                    ledger.records = []
                    report['completed_pairs'] += 1
                    report['call_counts'] = dict(ledger.counts)
                    persist()
                    del packets, metadata, old, current, replayed, native, w1, b1, w2, b2, smooth2
                    del delta, full, scores, proxy, p64, mean_score, packed, ranking
            need(ledger.counts == CALL_LIMITS and len(seen['paths']) == 480 and report['completed_pairs'] == 80,
                 'exact all80 pair coverage and finite call counts')
            report['actual_native_import_origins'] = validate_import_origins(p)
            report['status'] = 'COMPLETE_FULLM_ALL80_58ARM_DISCOVERY_EXIT_PENDING'
    except BaseException as error:
        report.update(status='FAILED_FULLM_DISCOVERY_PRESERVED_NO_RETRY', error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        guard.CONTROL_READ = None
        signal.alarm(0)
        writer.check = lambda: None  # Bounded postmortem metadata, no numerical continuation.
        writer.finish_pending()
        if monitor:
            monitor.close()
            report['assigned_GPU_monitor'] = monitor.report()
            if report['assigned_GPU_monitor']['violations']:
                report['status'] = 'FAILED_ASSIGNED_GPU_MONITOR_PRESERVED_NO_RETRY'
        report.update(input_read_ledger=budget.summary(), wall_seconds=time.monotonic() - started,
                      input_explicit_memory_raw_SHA_bytes=getattr(budget, 'explicit_memory_raw_SHA_bytes', 0),
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      output_accounting=writer.summary())
        if 'torch' in sys.modules:
            report['peak_CUDA_reserved_bytes'] = sys.modules['torch'].cuda.max_memory_reserved()
        persist()
        writer.journal.close()
        if monitor: monitor.check()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--job-spec', type=Path, required=True)
    ap.add_argument('--private-assets', type=Path, required=True)
    run(ap.parse_args())


if __name__ == '__main__': main()
