"""One separately registered full-model dynamic-row experiment, no implicit retry."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time

from row_dispatch_contract import ARMS, Budget, CAPS, METHOD, ROOT, atomic, read, require_registration, sha


def resource_check(root, protocol):
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if (used > protocol['resources']['new_output_bytes_max'] or
            shutil.disk_usage(root).free < protocol['resources']['free_disk_reserve_bytes']):
        raise RuntimeError('Full160 output/disk reserve exceeded')
    return used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-view-receipt', type=Path, required=True)
    args = parser.parse_args()
    protocol = require_registration()  # Before torch, models or CUDA imports.
    view = read(args.source_view_receipt)
    if (view['status'] != 'RUNNING_CHILD_WITH_VERIFIED_SOURCE_VIEW' or
            view['view_manifest_sha256'] != protocol['source_view_manifest_sha256'] or
            view['source_inventory_sha256'] != protocol['source_view_inventory_sha256']):
        raise RuntimeError('Exact verified dependency source view is required')
    import torch
    import torch.nn.functional as F
    from historical_source_snapshot import HistoricalSources
    from resident_ffn_runtime import FullFFNProfiler, make_resident, unused_raw_placeholders
    from restored_assets import verified_restored_assets
    from run_ar2_restart import tensor_sha
    from ub1_native_replay import UB1Positions
    from ub1_row_dispatch import RowDispatch, load_metadata, make_metadata_resident
    from worker_identity import require_worker
    require_worker()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('One exclusively locked GPU required')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(os.environ['HF_HOME']) / 'derived/ffnds/R16V0_full160_row_dispatch'
    if root.exists():
        raise RuntimeError('Preserve the first attempt; no automatic retry')
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < (81 << 30):
        raise RuntimeError('Insufficient HF disk reserve')
    root.mkdir()
    budget = Budget(root)
    report = {'id': METHOD, 'status': 'PREPARING_FIXED_FULL160_EXPERIMENT',
              'protocol_sha256': sha(ROOT / protocol['self_path']), 'requests': [],
              'native_microbenchmarks': [], 'new_training_steps': 0,
              'official_quality_or_Pareto_acceptance': False,
              'intermediate_latent_reset_to_DQ': False}
    report_path = root / 'RUN.json'
    atomic(report_path, report)
    try:
        history = HistoricalSources()
        print('Full160: checking original frozen model and coefficient assets', flush=True)
        kwargs, cache, assets = verified_restored_assets(history)
        policy_path = ROOT / 'runs/worker_restart_20261020/ub1_step_budget_fit/UB1_policy.json'
        positions = UB1Positions(policy_path, history,
                                 expected_policy_sha256=protocol['UB1_policy_sha256'])
        metadata = load_metadata(positions)
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        runtime = adapter.HIF4CacheRuntime(cache['path'])
        for (step, block), p in metadata.items():
            expert = 'high_noise_model' if step < 2 else 'low_noise_model'
            if tensor_sha(runtime.records[expert][f'blocks.{block}.ffn.2']['weight_qdq']) != p['W2_tensor_sha256']:
                raise RuntimeError('Existing UB1 coefficient/model weight mismatch')
        del positions
        with unused_raw_placeholders(adapter) as audit:
            pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        if len(audit['names']) != 400 or audit['calls'] < 800:
            raise RuntimeError('Original dense placeholder coverage differs')
        report['residency'] = make_resident(runtime)
        report['resident_auxiliary_bytes'] = make_metadata_resident(metadata)
        from row_dispatch_kernels import gated_energy
        probe = (torch.arange(64 * 5120, device='cuda').reshape(64, 5120) % 127).bfloat16() / 127
        gate_probe = (torch.arange(5120, device='cuda') % 31).bfloat16() / 31
        tested_energy = gated_energy(probe.contiguous(), gate_probe)
        expected_energy = ((probe.float() * gate_probe.float()).square()).sum(1)
        error = float(((tested_energy - expected_energy).abs() / expected_energy.clamp_min(1.)).max())
        if error > 1e-6:
            raise RuntimeError('Fused selector energy differs from direct FP32 reference')
        report['selector_fusion_control_max_relative_error'] = error
        del probe, gate_probe, tested_energy, expected_energy
        report['assets'] = assets
        sample = dict(read(ROOT / '.runtime/inputs/calibration24/canonical.json')[0])
        image = sample['img_paths'][0]
        if sample['calibration_seed'] != 42 or sha(image) != protocol['image_sha256']:
            raise RuntimeError('Frozen source image or seed changed')
        report['image_sha256'] = sha(image)
        report['prompt_sha256'] = hashlib.sha256(sample['prompt'].encode()).hexdigest()
        adapter.activate_lightx2v_hooks(runtime=runtime)
        profiler = FullFFNProfiler(adapter)
        profiler.install()
        scheduler = pipe.runner.scheduler
        old_post = scheduler.step_post
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        try:
            with RowDispatch(runtime, metadata) as dispatch:
                native_ffn = WanTransformerInfer.infer_ffn
                native_post = WanTransformerInfer.post_process
                bench_counter = {k: 0 for k in runtime.counters}

                def reference_ub1(infer, phase, x, attention, shift, scale, gate):
                    """Independent existing UB1 formula, retaining full native M."""
                    step, block = int(scheduler.step_index), int(infer.block_idx)
                    p = metadata[step, block]
                    expert = 'high_noise_model' if step < 2 else 'low_noise_model'
                    r0 = runtime.records[expert][f'blocks.{block}.ffn.0']
                    r2 = runtime.records[expert][f'blocks.{block}.ffn.2']
                    x.add_(attention)
                    if hasattr(phase, 'smooth_norm2_weight') or infer.sensitive_layer_dtype != infer.infer_dtype:
                        raise RuntimeError('Unexpected norm or writeback branch for independent UB1 control')
                    norm = phase.norm2.apply(x)
                    norm = infer.modulate_func(norm, scale=scale, shift=shift).squeeze()
                    qdq = adapter.native_hif4_activation_qdq
                    v1 = qdq((norm.float() / r0['smooth_scale'].float()).bfloat16(), force_fp32=True).bfloat16()
                    ids = p['channels']
                    b0 = phase.ffn_0._get_actual_bias().to(x.device).index_select(0, ids)
                    z = F.linear(v1, r0['weight_qdq'].index_select(0, ids).contiguous(), b0)
                    hidden = F.gelu(z, approximate='tanh')
                    v2 = qdq((hidden.float() / r2['smooth_scale'].float().index_select(0, ids)).bfloat16(), force_fp32=True).bfloat16()
                    v2 = (v2.float() * p['alpha']).bfloat16()
                    from reduced_rank_residual import predict_y
                    y = F.linear(v2, r2['weight_qdq'].index_select(1, ids).contiguous(), p['b2']) + predict_y(v1, p['residual'])
                    # Avoid profiler wrappers: this independent control did not
                    # enter a profiler FFN region.
                    return original_post(infer, x, y, gate)

                def benchmark(infer, phase, x, attention, shift, scale, gate):
                    step = int(scheduler.step_index)
                    profile_size, event_size = len(profiler.rows), len(dispatch.events)
                    counters = dict(runtime.counters)
                    dispatch.benchmarking = True
                    result = {'step': step, 'block': 20, 'source_index': 0,
                              'actual_full_native_rows': 31668, 'arms': {}, 'timing_orders': [],
                              'boundary': 'native infer_ffn input through original post_process; input clone excluded',
                              'shared_AdaLN_preparation_excluded': True}

                    def execute(arm, stage):
                        xx, aa = x.clone(), attention.clone()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        with budget.call(f's{step}_{stage}_{arm}', 'standalone_FFN_forwards'), dispatch.action(arm):
                            started = time.monotonic()
                            begin.record()
                            before_gate = dispatch.gate
                            dispatch.gate = gate
                            try:
                                y = original_ffn(infer, phase, xx, aa, shift, scale, gate)
                                output = original_post(infer, xx, y, gate)
                            finally:
                                dispatch.gate = before_gate
                            end.record()
                            end.synchronize()
                            elapsed = (time.monotonic() - started) * 1000
                        if not bool(torch.isfinite(output).all()):
                            raise RuntimeError('Nonfinite native standalone output')
                        return output, begin.elapsed_time(end), elapsed

                    truth = None
                    ub1_output = None
                    for arm in ARMS:
                        output, _, _ = execute(arm, 'control')
                        if arm == 'dense_DQ':
                            truth = output
                        else:
                            sse = float((output.float() - truth.float()).square().double().sum())
                            result['arms'][arm] = {'native_post_SSE_to_dense': sse, 'cuda_ms': [], 'host_ms': []}
                            if arm == 'UB1':
                                ub1_output = output
                        if arm == 'dense_DQ':
                            result['arms'][arm] = {'native_post_SSE_to_dense': 0., 'cuda_ms': [], 'host_ms': []}
                    # The existing frozen UB1 is an independent formula control,
                    # not merely the new hook in a different mode.
                    xx, aa = x.clone(), attention.clone()
                    with budget.call(f's{step}_independent_UB1', 'standalone_FFN_forwards'):
                        reference = reference_ub1(infer, phase, xx, aa, shift, scale, gate)
                    mismatches = int((reference.view(torch.int16) != ub1_output.view(torch.int16)).sum())
                    result['UB1_reference_bit_mismatches'] = mismatches
                    if mismatches:
                        raise RuntimeError('Compact native UB1 differs from original formula')
                    del truth, ub1_output, reference, output
                    for warmup in range(2):
                        for arm in ARMS:
                            output, _, _ = execute(arm, f'warm{warmup}')
                            del output
                    for repeat in range(5):
                        order = ARMS[repeat:] + ARMS[:repeat]
                        if repeat % 2:
                            order = list(reversed(order))
                        result['timing_orders'].append(order)
                        for arm in order:
                            output, cuda_ms, host_ms = execute(arm, f'timed{repeat}')
                            result['arms'][arm]['cuda_ms'].append(cuda_ms)
                            result['arms'][arm]['host_ms'].append(host_ms)
                            del output
                    for arm, values in result['arms'].items():
                        values['median_cuda_ms'] = statistics.median(values['cuda_ms'])
                        values['median_host_ms'] = statistics.median(values['host_ms'])
                    result['dispatch_events'] = summarize_events(dispatch.events[event_size:])
                    report['native_microbenchmarks'].append(result)
                    atomic(report_path, report)
                    del profiler.rows[profile_size:]
                    del dispatch.events[event_size:]
                    for k in counters:
                        bench_counter[k] += runtime.counters[k] - counters[k]
                    dispatch.benchmarking = False
                    print(f'Full160: original step{step}/block20 native FFN benchmark completed', flush=True)

                def ffn(infer, phase, x, attention, shift, scale, gate=None):
                    if dispatch.arm == 'dense_DQ' and infer.block_idx == 20 and not dispatch.benchmarking:
                        benchmark(infer, phase, x, attention, shift, scale, gate)
                    return native_ffn(infer, phase, x, attention, shift, scale, gate)

                # Preserve the real original writeback function for the separate
                # UB1 formula (FullFFNProfiler wraps post_process).
                original_post = next(old for cls, name, old in profiler.originals if name == 'post_process')
                original_ffn = next(old for cls, name, old in profiler.originals if name == 'infer_ffn')
                WanTransformerInfer.infer_ffn = ffn
                try:
                    for arm in ARMS:
                        resource_check(root, protocol)
                        profiler.begin_request()
                        dispatch.events.clear()
                        before = dict(runtime.counters)
                        bench_before = dict(bench_counter)
                        scheduler.generator = None
                        trajectory, initial = [], []

                        def post():
                            step = int(scheduler.step_index)
                            if step == 0:
                                initial.append(tensor_sha(scheduler.latents.detach().float().cpu().contiguous()))
                            result = old_post()
                            latent = scheduler.latents.detach().float().cpu().contiguous()
                            if not bool(torch.isfinite(latent).all()):
                                raise RuntimeError('Nonfinite free student state')
                            trajectory.append({'step': step, 'shape': list(latent.shape), 'sha256': tensor_sha(latent)})
                            print(f'Full160: {arm} completed free student step{step}', flush=True)
                            return result

                        scheduler.step_post = post
                        torch.cuda.reset_peak_memory_stats()
                        began = time.monotonic()
                        video = root / f'{arm}.mp4'
                        with budget.call(arm, 'full_generations'), dispatch.action(arm), torch.inference_mode():
                            pipe.generate(seed=42, image_path=image, prompt=sample['prompt'],
                                          negative_prompt='', save_result_path=str(video))
                            torch.cuda.synchronize()
                        scheduler.step_post = old_post
                        count = {k: runtime.counters[k] - before[k] - (bench_counter[k] - bench_before[k]) for k in before}
                        if (count != {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600,
                                      'cache_misses': 0, 'weight_transfers': 0} or audit['fallback_calls'] or
                                [r['step'] for r in trajectory] != list(range(4)) or len(initial) != 1):
                            raise RuntimeError('Incomplete original model/trajectory coverage')
                        events = summarize_events(dispatch.events)
                        if arm != 'dense_DQ' and (len(events) != 160 or
                                {(e['step'], e['block']) for e in events} != {(s, b) for s in range(4) for b in range(40)}):
                            raise RuntimeError('Candidate failed actual full160 execution')
                        peak = torch.cuda.max_memory_allocated()
                        if peak > protocol['resources']['peak_allocated_bytes_max']:
                            raise RuntimeError('Peak native allocation cap exceeded')
                        request = {'arm': arm, 'trajectory': trajectory, 'initial_noise_sha256': initial[0],
                                   'video_sha256': sha(video), 'video_HF_relative': str(video.relative_to(Path(os.environ['HF_HOME']))),
                                   'counter_delta_excluding_debited_benchmarks': count,
                                   'dispatch_events': events, 'FFN_timing': profiler.report(),
                                   'peak_allocated_bytes': peak, 'elapsed_seconds_with_observation': time.monotonic() - began,
                                   'whole_generation_performance_benchmark': False,
                                   'free_student_states_no_reset': True}
                        if arm == 'dense_DQ':
                            old_dense = read(ROOT / 'research/20261023/R15V0_native_rollout_results.json')['requests'][0]
                            if ([r['sha256'] for r in trajectory] != [r['raw_tensor_sha256'] for r in old_dense['trajectory']] or
                                    initial[0] != old_dense['initial_noise_sha256']):
                                raise RuntimeError('Dense benchmark instrumentation changed the original free trajectory')
                            request['original_dense_four_latents_bitwise_equal'] = True
                        report['requests'].append(request)
                        report['new_output_bytes'] = resource_check(root, protocol)
                        atomic(report_path, report)
                finally:
                    WanTransformerInfer.infer_ffn = native_ffn
        finally:
            scheduler.step_post = old_post
            profiler.restore()
            adapter.deactivate_lightx2v_hooks()
        if (len(report['requests']) != 5 or len(report['native_microbenchmarks']) != 4 or
                len({r['initial_noise_sha256'] for r in report['requests']}) != 1 or budget.state['used'] != CAPS):
            raise RuntimeError('Incomplete fixed generation/benchmark budget or mismatched noise')
        report['status'] = 'COMPLETE_FULL160_GENERATION_AND_NATIVE_MICROBENCHMARK_AWAITING_VIDEO_REVIEW'
    except BaseException as error:
        report.update(status='FAILED_FIXED_SCOPE_DEBIT_PRESERVED', error_type=type(error).__name__)
        raise
    finally:
        report['budget'] = budget.state
        atomic(report_path, report)
        budget.close()
    print(json.dumps({'status': report['status'], 'full_generations': len(report['requests'])}), flush=True)


def summarize_events(events):
    """Persist row-set identities, never the full GPU arrays or private paths."""
    from run_ar2_restart import tensor_sha
    result = []
    for event in events:
        row = {k: v for k, v in event.items() if k != 'dense_row_ids'}
        row['dense_row_ids_sha256'] = tensor_sha(event['dense_row_ids'].cpu().contiguous())
        result.append(row)
    return result


if __name__ == '__main__':
    main()
