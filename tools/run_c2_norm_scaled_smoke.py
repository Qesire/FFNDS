"""Pending finite paired C2 video smoke; refuses to run without new authority."""
import argparse
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
from c2_contract import ARMS, CAPS, METHOD, ROOT, Budget, atomic, read, require_registration, sha


def close_scope(report, complete):
    decision = read(ROOT / 'research/CURRENT_DECISION.json'); triage = read(ROOT / 'research/20261021/method_triage.json')
    registration = decision['registered_c2_smoke']
    if not registration['active']: raise RuntimeError('C2 already closed')
    status = 'COMPLETE_C2_NATIVE_VIDEOS_PENDING_FULL_REVIEW' if complete else 'STOPPED_C2_FIXED_SCOPE_NO_RETRY'
    summary = {'id': METHOD, 'status': status, 'actual_budget': report['budget']['used'],
               'completed_videos': len(report['requests']), 'automatic_followup': False,
               'quality_or_Pareto_acceptance': False, 'protocol_sha256': registration['protocol_sha256'],
               'error_type': report.get('error_type')}
    path = ROOT / 'research/20261023/C2N0_execution_summary.json'
    if path.exists(): raise RuntimeError('Preserve first C2 result')
    atomic(path, summary)
    registration.update(active=False, result=str(path.relative_to(ROOT)), fixed_scope_finished=True)
    decision['execution_state'] = registration['preserved_model_closure']; decision['eligible_model_method_ids'] = []
    decision['latest_c2_native_smoke'] = summary
    triage['policy']['execution_state'] = decision['execution_state']
    row = next(x for x in triage['methods'] if x['id'] == METHOD)
    row.update(status=status, model_execution_eligibility='NOT_ELIGIBLE_FIXED_SCOPE_FINISHED', result=str(path.relative_to(ROOT)))
    atomic(ROOT / 'research/CURRENT_DECISION.json', decision); atomic(ROOT / 'research/20261021/method_triage.json', triage)
    candidate_path = ROOT / 'research/candidate_rollout/registry/C2N0_norm_scaled_energy.json'
    candidate = read(candidate_path)
    candidate.update(status='ROLLOUT_RUNNING' if complete else 'CLOSED',
                     execution_stage=status, actual_result=str(path.relative_to(ROOT)))
    atomic(candidate_path, candidate)
    subprocess.run(['python3', str(ROOT / 'tools/check_research_routes.py')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-view-receipt', type=Path, required=True)
    args = parser.parse_args(); p = require_registration()
    view = read(args.source_view_receipt)
    if (view['view_manifest_sha256'] != p['source_view_manifest_sha256']
            or view['status'] != 'RUNNING_CHILD_WITH_VERIFIED_SOURCE_VIEW'):
        raise RuntimeError('Exact verified dependency view required')
    import numpy as np
    import torch
    from assigned_gpu_resources import AssignedGPUResources
    from historical_source_snapshot import HistoricalSources
    from norm_scaled_row_dispatch import NormScaledEnergy
    from prew1_norm_statistics import layer_norm_with_statistics
    from resident_ffn_runtime import make_resident, unused_raw_placeholders
    from restored_assets import verified_restored_assets
    from run_ar2_restart import tensor_sha
    from ub1_native_replay import UB1Positions
    from ub1_row_dispatch import RowDispatch, load_metadata, make_metadata_resident
    from worker_identity import require_worker
    require_worker()
    if torch.cuda.device_count() != 1: raise RuntimeError('One locked GPU required')
    torch.set_num_threads(8); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    hf = Path(os.environ['HF_HOME']); root = hf / 'derived/ffnds/C2N0_norm_scaled_smoke'
    if root.exists(): raise RuntimeError('Preserve first C2 attempt')
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 81 << 30: raise RuntimeError('HF disk reserve insufficient')
    root.mkdir(); budget = Budget(root); monitor = None; complete = False
    report = {'id': METHOD, 'status': 'PREPARING_C2_FIXED_SCOPE', 'requests': [],
              'protocol_sha256': sha(ROOT / p['self_path']), 'quality_or_speed_acceptance': False,
              'new_training_updates': 0, 'automatic_retry': False}
    atomic(root / 'RUN.json', report)
    try:
        history = HistoricalSources(); kwargs, cache, assets = verified_restored_assets(history)
        positions = UB1Positions(ROOT / 'runs/worker_restart_20261020/ub1_step_budget_fit/UB1_policy.json',
                                 history, expected_policy_sha256=p['UB1_policy_sha256'])
        metadata = load_metadata(positions); del positions
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        runtime = adapter.HIF4CacheRuntime(cache['path'])
        for (step, block), item in metadata.items():
            expert = 'high_noise_model' if step < 2 else 'low_noise_model'
            if tensor_sha(runtime.records[expert][f'blocks.{block}.ffn.2']['weight_qdq']) != item['W2_tensor_sha256']:
                raise RuntimeError('Original UB1/W2 mismatch')
        with unused_raw_placeholders(adapter) as placeholders: pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        if len(placeholders['names']) != 400 or placeholders['calls'] < 800: raise RuntimeError('Placeholder coverage differs')
        report['residency'] = make_resident(runtime); report['auxiliary_bytes'] = make_metadata_resident(metadata)
        report['assets'] = assets
        monitor = AssignedGPUResources(); monitor.start()
        probe = ((torch.arange(64 * 5120, device='cuda').reshape(64, 5120) % 127) / 31).bfloat16()
        with budget.call('norm_reference', 'standalone_normalization_calls'):
            expected = torch.nn.functional.layer_norm(probe, (5120,), None, None, 1e-6)
        with budget.call('norm_with_statistics', 'standalone_normalization_calls'):
            observed, _, rstd = layer_norm_with_statistics(probe, infer_dtype=torch.bfloat16, sensitive_layer_dtype=torch.bfloat16)
        if not torch.equal(expected, observed): raise RuntimeError('GPU normalization bit control failed')
        from row_dispatch_kernels import gated_energy
        gate = ((torch.arange(5120, device='cuda') % 31) / 31).bfloat16()
        with budget.call('direct_energy_reference', 'standalone_selector_calls'):
            direct = ((probe.float() * gate.float()) ** 2).sum(1)
        with budget.call('old_energy', 'standalone_selector_calls'): original = gated_energy(probe, gate)
        with budget.call('scaled_energy', 'standalone_selector_calls'):
            scaled = gated_energy(probe, gate) * rstd.reshape(-1).square()
        error = float(((original - direct).abs() / direct.clamp_min(1)).max())
        scaled_error = float(((scaled - direct * rstd.reshape(-1).square()).abs() /
                              (direct * rstd.reshape(-1).square()).clamp_min(1)).max())
        if max(error, scaled_error) > 1e-6: raise RuntimeError('Selector numeric control failed')
        report['GPU_numeric_controls'] = {'norm_bitwise_equal': True, 'old_energy_max_relative_error': error,
                                          'scaled_energy_max_relative_error': scaled_error}
        del probe, gate, direct, original, scaled, observed, expected, rstd
        fresh = read(ROOT / '.runtime/inputs/r17_fresh23/canonical.json')[6]
        old = read(ROOT / '.runtime/inputs/calibration24/canonical.json')[0]
        old_reference = next(r for r in read(ROOT / 'research/20261023/R16V0_native_results.json')['requests'] if r['arm'] == 'UB1')
        scheduler = pipe.runner.scheduler; old_post = scheduler.step_post; fresh_noise = None
        adapter.activate_lightx2v_hooks(runtime=runtime)
        try:
            for name in ARMS:
                monitor.check()
                if (sum(f.stat().st_size for f in root.rglob('*') if f.is_file()) > 1 << 29
                        or shutil.disk_usage(root).free < 80 << 30): raise RuntimeError('C2 output/disk cap exceeded')
                control = name == 'control_observed_UB1'; sample = old if control else fresh
                image = sample.get('image_path') or sample['img_paths'][0]
                if sha(image) != sample['input_provenance']['image_sha256']: raise RuntimeError('Source image changed')
                arm = 'UB1' if control else 'dynamic_residual' if name == 'norm_scaled_energy' else name
                folder = root / name; folder.mkdir(); snapshots, initial = [], []
                def post():
                    step = int(scheduler.step_index)
                    if step == 0: initial.append(scheduler.latents.detach().float().clone())
                    value = old_post(); snapshots.append((step, scheduler.latents.detach().float().clone())); return value
                scheduler.step_post = post; scheduler.generator = None
                before = dict(runtime.counters); torch.cuda.reset_peak_memory_stats()
                with RowDispatch(runtime, metadata, arm) as dispatch:
                    observer = NormScaledEnergy(dispatch, 'observe_UB1' if control else 'scaled') if control or name == 'norm_scaled_energy' else None
                    with observer if observer is not None else nullcontext():
                        started = time.monotonic()
                        with budget.call(name, 'full_generations'), torch.inference_mode():
                            pipe.generate(seed=42, image_path=image, prompt=sample['prompt'], negative_prompt='',
                                          save_result_path=str(folder / 'output.mp4'))
                            torch.cuda.synchronize()
                        wall = time.monotonic() - started
                    peak = torch.cuda.max_memory_allocated()
                    if peak > 79 << 30: raise RuntimeError('C2 GPU allocation cap exceeded')
                    if [s for s, _ in snapshots] != list(range(4)) or len(initial) != 1: raise RuntimeError('Incomplete trajectory')
                    trajectory = [{'step': s, 'sha256': tensor_sha(v.cpu().contiguous())} for s, v in snapshots]
                    noise = tensor_sha(initial[0].cpu().contiguous())
                    if control:
                        reference = [{'step': r['step'], 'sha256': r['sha256']} for r in old_reference['trajectory']]
                        if trajectory != reference or noise != old_reference['initial_noise_sha256']:
                            raise RuntimeError('All160 norm-observer UB1 full trajectory differs from original')
                    else:
                        if fresh_noise is None: fresh_noise = noise
                        if noise != fresh_noise: raise RuntimeError('Paired fresh initial noise mismatch')
                    counts = {k: runtime.counters[k] - before[k] for k in before}
                    if counts != {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600, 'cache_misses': 0, 'weight_transfers': 0} or placeholders['fallback_calls']:
                        raise RuntimeError('Original native all160 coverage failed')
                    if observer is not None and len(observer.events) != 160: raise RuntimeError('Missing actual norm/score positions')
                    if observer is not None and {(e['step'], e['block']) for e in observer.events} != {(s, b) for s in range(4) for b in range(40)}:
                        raise RuntimeError('Native norm/score position grid differs')
                    if arm != 'dense_DQ' and len(dispatch.events) != 160: raise RuntimeError('Missing physical row dispatch')
                    if arm in ('dynamic_residual', 'fixed_stride'):
                        masks = []
                        for e in dispatch.events:
                            if (e['dense_rows'], e['sparse_rows']) != (7917, 23751): raise RuntimeError('Matched row budget changed')
                            bits = np.zeros(31668, dtype=np.uint8); bits[e['dense_row_ids'].cpu().numpy()] = 1
                            masks.append(np.packbits(bits))
                        with (folder / 'ROUTING.npz').open('xb') as f: np.savez_compressed(f, packed_dense_rows=np.stack(masks))
                    request = {'name': name, 'source_index': -1 if control else 6, 'source_group': sample.get('source_group'),
                               'image_sha256': sha(image), 'prompt_sha256': hashlib.sha256(sample['prompt'].encode()).hexdigest(),
                               'trajectory': trajectory, 'initial_noise_sha256': noise, 'original_UB1_four_state_identity': control,
                               'counter_delta': counts, 'physical_row_dispatch_sites': len(dispatch.events),
                               'norm_statistic_events': [] if observer is None else observer.events,
                               'video_sha256': sha(folder / 'output.mp4'), 'video_HF_relative': str((folder / 'output.mp4').relative_to(hf)),
                               'routing_sha256': sha(folder / 'ROUTING.npz') if (folder / 'ROUTING.npz').exists() else None,
                               'free_student_no_resets': True, 'wall_seconds_one_observed_request_not_repeated_speed': wall,
                               'peak_allocated_bytes': peak, 'quality_verdict': 'PENDING_FULL_VIDEO_REVIEW'}
                    atomic(folder / 'REQUEST.json', request)
                scheduler.step_post = old_post
                report['requests'].append(request); report['budget'] = budget.state; report['status'] = 'RUNNING_C2_FIXED_NATIVE_SMOKE'
                atomic(root / 'RUN.json', report); monitor.check()
                del snapshots, initial, dispatch, observer
                print(f'C2 complete video{len(report["requests"])}/6: {name}', flush=True)
        finally:
            scheduler.step_post = old_post; adapter.deactivate_lightx2v_hooks()
        if budget.state['used'] != CAPS: raise RuntimeError('C2 exact budget incomplete')
        complete = True; report['status'] = 'COMPLETE_C2_NATIVE_VIDEOS_PENDING_FULL_REVIEW'
    except BaseException as error:
        report.update(status='STOPPED_C2_FIXED_SCOPE_NO_RETRY', error_type=type(error).__name__, error_message=str(error)); raise
    finally:
        if monitor is not None:
            monitor.close(); report['resources'] = monitor.report()
            if monitor.violations: complete = False; report['status'] = 'FAILED_C2_GPU_ISOLATION'
        report['budget'] = budget.state; atomic(root / 'RUN.json', report); budget.close(); close_scope(report, complete)


if __name__ == '__main__': main()
