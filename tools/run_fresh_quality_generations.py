"""Execute one registered frozen-algorithm fresh cohort and balanced timing."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from fresh_quality_contract import ARMS, Budget, METHOD, ROOT, atomic, generation_jobs, read, require_registration, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-view-receipt', type=Path, required=True)
    args = parser.parse_args()
    p = require_registration('GENERATIONS')
    view = read(args.source_view_receipt)
    if (view['status'] != 'RUNNING_CHILD_WITH_VERIFIED_SOURCE_VIEW' or
            view['view_manifest_sha256'] != p['generation_source_view_sha256']):
        raise RuntimeError('Exact original generation dependency source view required')
    import torch
    from historical_source_snapshot import HistoricalSources
    from assigned_gpu_resources import AssignedGPUResources
    from native_ffn_events import NativeFFNEvents
    from resident_ffn_runtime import make_resident, unused_raw_placeholders
    from restored_assets import verified_restored_assets
    from run_ar2_restart import tensor_sha
    from run_full160_row_dispatch import summarize_events
    from ub1_native_replay import UB1Positions
    from ub1_row_dispatch import RowDispatch, load_metadata, make_metadata_resident
    from worker_identity import require_worker
    require_worker()
    if torch.cuda.device_count() != 1:
        raise RuntimeError('One locked idle GPU required')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    hf = Path(os.environ['HF_HOME'])
    root = hf / 'derived/ffnds/R17E0_fresh_quality'
    if root.exists():
        raise RuntimeError('Preserve first generation attempt; no implicit retry')
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 85 << 30:
        raise RuntimeError('Insufficient HF output reserve')
    root.mkdir()
    budget = Budget(root)
    report = {'id': METHOD, 'status': 'PREPARING_FIXED_FRESH_COHORT',
              'protocol_sha256': sha(ROOT / p['self_path']), 'completed': [],
              'new_training_steps': 0, 'parameter_or_fraction_selection': False,
              'official_quality_or_Pareto_acceptance': False}
    atomic(root / 'GENERATIONS.json', report)
    monitor = None
    try:
        history = HistoricalSources()
        print('Fresh23: verifying original frozen model and UB1 assets', flush=True)
        kwargs, cache, assets = verified_restored_assets(history)
        position_store = UB1Positions(ROOT / 'runs/worker_restart_20261020/ub1_step_budget_fit/UB1_policy.json', history,
                                      expected_policy_sha256=p['UB1_policy_sha256'])
        metadata = load_metadata(position_store)
        del position_store
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        runtime = adapter.HIF4CacheRuntime(cache['path'])
        for (step, block), item in metadata.items():
            expert = 'high_noise_model' if step < 2 else 'low_noise_model'
            if tensor_sha(runtime.records[expert][f'blocks.{block}.ffn.2']['weight_qdq']) != item['W2_tensor_sha256']:
                raise RuntimeError('Frozen support coefficient/model mismatch')
        with unused_raw_placeholders(adapter) as audit:
            pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        if len(audit['names']) != 400 or audit['calls'] < 800:
            raise RuntimeError('Original placeholder geometry changed')
        report['residency'] = make_resident(runtime)
        report['auxiliary_resident_bytes'] = make_metadata_resident(metadata)
        report['assets'] = assets
        sources = read(ROOT / '.runtime/inputs/r17_fresh23/canonical.json')
        old = read(ROOT / '.runtime/inputs/calibration24/canonical.json')[0]
        performance = {'seed': 42, 'image_path': old['img_paths'][0], 'prompt': old['prompt']}
        preserved = read(ROOT / 'research/20261023/R16V0_native_results.json')
        expected_warm = {r['arm']: r for r in preserved['requests']}
        grids, warm_trajectories, initial_noises = {}, {}, {}
        scheduler = pipe.runner.scheduler
        original_step_post = scheduler.step_post
        adapter.activate_lightx2v_hooks(runtime=runtime)
        from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
        old_block = WanTransformerInfer.infer_block
        observed_grids = set()

        def block(infer, weights, x, pre):
            observed_grids.add(tuple(int(v) for v in pre.grid_sizes.tuple))
            return old_block(infer, weights, x, pre)

        WanTransformerInfer.infer_block = block
        monitor = AssignedGPUResources()
        monitor.start()
        try:
            for number, job in enumerate(generation_jobs()):
                monitor.check()
                source, arm = job['source'], job['arm']
                sample = performance if source < 0 else sources[source]
                if sha(sample['image_path']) != (p['performance_image_sha256'] if source < 0 else sample['input_provenance']['image_sha256']):
                    raise RuntimeError('Frozen source image bytes changed')
                used = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
                if used > 4 << 30 or shutil.disk_usage(root).free < 80 << 30:
                    raise RuntimeError('Fresh generation output/disk cap exceeded')
                expected_grid = grids.get(source, (21, 29, 52))
                if arm != 'dense_DQ' and source not in grids:
                    raise RuntimeError('Original native dense source geometry must be observed before sparse execution')
                dispatch = None if arm == 'dense_DQ' else RowDispatch(runtime, metadata, arm,
                                     expected_shape=(math.prod(expected_grid), 5120, 13824))
                scheduler.generator = None
                snapshots, initial = [], []

                def post():
                    step = int(scheduler.step_index)
                    if step == 0:
                        initial.append(scheduler.latents.detach().float().clone())
                    result = original_step_post()
                    snapshots.append((step, scheduler.latents.detach().float().clone()))
                    return result

                scheduler.step_post = post
                observed_grids.clear()
                before = dict(runtime.counters)
                folder = root / job['name']
                folder.mkdir()
                video = folder / 'output.mp4'
                torch.cuda.reset_peak_memory_stats()
                # Prepare the static index masks before any request timing.
                # Latent GPU clones and event instrumentation remain charged.
                with (dispatch if dispatch else nullcontext()), NativeFFNEvents() as events:
                    torch.cuda.synchronize()
                    with budget.call(job['name'], 'full_generations'), torch.inference_mode():
                        # Budget bookkeeping is excluded from the actual model
                        # measurement; the enclosing job remains pre-debited.
                        start = time.perf_counter()
                        pipe.generate(seed=42, image_path=sample['image_path'], prompt=sample['prompt'],
                                      negative_prompt='', save_result_path=str(video))
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                scheduler.step_post = original_step_post
                monitor.check()
                peak = torch.cuda.max_memory_allocated()
                if peak > 79 << 30 or len(observed_grids) != 1 or len(initial) != 1 or [s for s, _ in snapshots] != list(range(4)):
                    raise RuntimeError('Native geometry, trajectory or memory cap invalid')
                actual_grid = next(iter(observed_grids))
                if source in grids and grids[source] != actual_grid:
                    raise RuntimeError('Paired source grid changed')
                grids[source] = actual_grid
                counters = {k: runtime.counters[k] - before[k] for k in before}
                if counters != {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600, 'cache_misses': 0, 'weight_transfers': 0} or audit['fallback_calls']:
                    raise RuntimeError('Missing native runtime coverage or placeholder fallback')
                trajectory = []
                for step, latent in snapshots:
                    cpu = latent.cpu().contiguous()
                    if not bool(torch.isfinite(cpu).all()):
                        raise RuntimeError('Nonfinite actual free student state')
                    trajectory.append({'step': step, 'shape': list(cpu.shape), 'sha256': tensor_sha(cpu)})
                noise_sha = tensor_sha(initial[0].cpu().contiguous())
                if source in initial_noises and initial_noises[source] != noise_sha:
                    raise RuntimeError('Paired source initial noise differs across arms or repeats')
                initial_noises[source] = noise_sha
                del snapshots, initial, cpu, latent
                hashes = [r['sha256'] for r in trajectory]
                if job['phase'] == 'warmup':
                    expected = expected_warm[arm]
                    if hashes != [r['sha256'] for r in expected['trajectory']] or noise_sha != expected['initial_noise_sha256']:
                        raise RuntimeError('New event/clone instrumentation changed preserved four-step trajectory')
                    warm_trajectories[arm] = hashes
                if job['phase'] == 'timing' and hashes != warm_trajectories[arm]:
                    raise RuntimeError('Timed native repetition changed its fixed warm trajectory')
                recorded_events = summarize_events(dispatch.events) if dispatch else []
                if dispatch and (len(recorded_events) != 160 or
                        {(e['step'], e['block']) for e in recorded_events} != {(s, b) for s in range(4) for b in range(40)}):
                    raise RuntimeError('Not an actual full160 candidate deployment')
                routing = None
                if dispatch and arm == 'dynamic_residual':
                    import numpy as np
                    masks = np.zeros((4, 40, math.prod(actual_grid)), dtype=np.bool_)
                    for event in dispatch.events:
                        masks[event['step'], event['block'], event['dense_row_ids'].cpu().numpy()] = True
                    trace = folder / 'routing.npz'
                    np.savez_compressed(trace, grid=np.asarray(actual_grid),
                                        packed=np.packbits(masks, axis=-1, bitorder='little'))
                    routing = {'HF_relative': str(trace.relative_to(hf)), 'sha256': sha(trace),
                               'bitorder': 'little', 'shape_before_packbits': list(masks.shape),
                               'CPU_copy_and_compression_outside_timing': True}
                record = {'job': job, 'free_student_states_no_reset': True, 'trajectory': trajectory,
                          'initial_noise_sha256': noise_sha, 'grid': list(actual_grid),
                          'source_image_sha256': sha(sample['image_path']),
                          'prompt_sha256': hashlib.sha256(sample['prompt'].encode()).hexdigest(),
                          'video_HF_relative': str(video.relative_to(hf)), 'video_sha256': sha(video),
                          'actual_native_counter_delta': counters, 'dispatch_events': recorded_events,
                          'native_FFN_timing': events.report_after_request_synchronized(),
                          'generation_wall_seconds': elapsed, 'whole_generation_performance_benchmark': job['phase'] == 'timing',
                          'GPU_latent_clone_observation_charged': True, 'CPU_tensor_hashes_outside_timing': True,
                          'peak_allocated_bytes': peak}
                record['routing_trace'] = routing
                atomic(folder / 'REQUEST.json', record)
                report['completed'].append({'job': job, 'request_HF_relative': str((folder / 'REQUEST.json').relative_to(hf)),
                                            'request_sha256': sha(folder / 'REQUEST.json')})
                report['status'] = 'RUNNING_FIXED_FRESH_QUALITY_AND_BALANCED_SYSTEM_TIMING'
                atomic(root / 'GENERATIONS.json', report)
                print(f'Fresh23: completed{number+1}/145 {job["name"]} wall={elapsed:.3f}s', flush=True)
        finally:
            scheduler.step_post = original_step_post
            WanTransformerInfer.infer_block = old_block
            adapter.deactivate_lightx2v_hooks()
        if len(report['completed']) != 145 or budget.state['used']['full_generations'] != 145:
            raise RuntimeError('Incomplete fixed generation plan')
        report['status'] = 'COMPLETE_145_GENERATIONS_AWAITING_FRESH_OFFICIAL_METRICS'
    except BaseException as error:
        report.update(status='FAILED_FIXED_GENERATION_SCOPE_DEBIT_PRESERVED', error_type=type(error).__name__)
        raise
    finally:
        if monitor is not None:
            monitor.close()
            report['resources'] = monitor.report()
            if monitor.violations:
                report['status'] = 'FAILED_SAMPLED_GPU_ISOLATION_DEBIT_PRESERVED'
        report['budget'] = budget.state
        atomic(root / 'GENERATIONS.json', report)
        budget.close()
    if report['status'] != 'COMPLETE_145_GENERATIONS_AWAITING_FRESH_OFFICIAL_METRICS':
        raise RuntimeError('Fresh generation process did not pass its complete fixed controls')
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
