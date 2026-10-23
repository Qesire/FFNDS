"""One finite native C1 probe, frozen before entry; no automatic retry."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time

from c1_contract import ARMS, BLOCKS, CAPS, METHOD, ROOT, Budget, atomic, read, require_registration, sha


def close_scope(report, success):
    decision = read(ROOT / 'research/CURRENT_DECISION.json')
    triage = read(ROOT / 'research/20261021/method_triage.json')
    registration = decision['registered_c1_diagnostic']
    if not registration['active']: raise RuntimeError('C1 already closed; preserve first result')
    status = 'COMPLETED_FIXED_NATIVE_C1_DIAGNOSTIC' if success else 'STOPPED_FIXED_NATIVE_C1_NO_RETRY'
    summary = {'id': METHOD, 'status': status, 'actual_budget': report['budget']['used'],
               'completed_generations': len(report['requests']), 'completed_finite_scopes': len(report['finite_scopes']),
               'original_protocol_sha256': registration['protocol_sha256'],
               'HF_relative_root': 'derived/ffnds/C1N0_native_information',
               'official_video_quality_or_speed_acceptance': False, 'automatic_followup': False,
               'error_type': report.get('error_type')}
    target = ROOT / 'research/20261023/C1N0_execution_summary.json'
    if target.exists(): raise RuntimeError('Preserve first C1 summary')
    atomic(target, summary)
    registration.update(active=False, result=str(target.relative_to(ROOT)), fixed_scope_finished=True)
    decision['execution_state'] = registration['preserved_model_closure']; decision['eligible_model_method_ids'] = []
    decision['latest_c1_native_diagnostic'] = summary
    triage['policy']['execution_state'] = decision['execution_state']
    triage['policy']['execution_note'] = 'C1 finite native information diagnostic completed or stopped; no automatic retry/followup. All earlier closures preserved.'
    row = next(x for x in triage['methods'] if x['id'] == METHOD)
    row.update(status=status, model_execution_eligibility='NOT_ELIGIBLE_FIXED_SCOPE_FINISHED', result=str(target.relative_to(ROOT)))
    triage['latest_c1_native_diagnostic'] = summary
    atomic(ROOT / 'research/CURRENT_DECISION.json', decision); atomic(ROOT / 'research/20261021/method_triage.json', triage)
    subprocess.run(['python3', str(ROOT / 'tools/check_research_routes.py')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-view-receipt', type=Path, required=True)
    args = parser.parse_args(); p = require_registration()
    if read(args.source_view_receipt)['view_manifest_sha256'] != p['source_view_manifest_sha256']:
        raise RuntimeError('Original verified dependency source view required')
    import numpy as np
    import torch
    from assigned_gpu_resources import AssignedGPUResources
    from c1_native_probe import Probe, prepare_actions
    from historical_source_snapshot import HistoricalSources
    from prew1_norm_statistics import layer_norm_with_statistics
    from resident_ffn_runtime import make_resident, unused_raw_placeholders
    from restored_assets import verified_restored_assets
    from run_ar2_restart import tensor_sha
    from ub1_native_replay import UB1Positions
    from ub1_row_dispatch import RowDispatch, load_metadata, make_metadata_resident
    from worker_identity import require_worker
    require_worker()
    if torch.cuda.device_count() != 1: raise RuntimeError('One locked GPU required')
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    hf = Path(os.environ['HF_HOME']); root = hf / 'derived/ffnds/C1N0_native_information'
    if root.exists(): raise RuntimeError('Preserve first C1 model attempt')
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < 82 << 30: raise RuntimeError('Insufficient HF reserve')
    root.mkdir(); budget = Budget(root); monitor = None
    report = {'id': METHOD, 'status': 'PREPARING_FROZEN_NATIVE_C1', 'protocol_sha256': sha(ROOT / p['self_path']),
              'requests': [], 'finite_scopes': [], 'automatic_retry': False,
              'new_training_updates': 0, 'new_weight_factorizations': 0,
              'candidate_free_video_or_official_quality_or_speed_result': False}
    atomic(root / 'RUN.json', report)
    success = False
    try:
        print('C1: verifying original model/cache assets and existing UB1 parameters', flush=True)
        history = HistoricalSources(); kwargs, cache, assets = verified_restored_assets(history)
        positions = UB1Positions(ROOT / 'runs/worker_restart_20261020/ub1_step_budget_fit/UB1_policy.json',
                                 history, expected_policy_sha256=p['UB1_policy_sha256'])
        metadata = load_metadata(positions)
        from HiFloat4.ptq.adapters import lightx2v_wan_distill as adapter
        runtime = adapter.HIF4CacheRuntime(cache['path'])
        for (step, block), item in metadata.items():
            expert = 'high_noise_model' if step < 2 else 'low_noise_model'
            if tensor_sha(runtime.records[expert][f'blocks.{block}.ffn.2']['weight_qdq']) != item['W2_tensor_sha256']:
                raise RuntimeError('Original coefficient/W2 mismatch')
        actions, action_records = prepare_actions(positions, metadata, runtime)
        del positions
        atomic(root / 'ACTION_PARAMETERS.json', {'actions': action_records,
            'rule': 'Two fixed source-blind whole64-group swaps; BF16 alpha before W2; calibration-only moment-centered folded bias; originalR256 unchanged',
            'new_fit_or_factorization': False})
        report['action_parameters_sha256'] = sha(root / 'ACTION_PARAMETERS.json')
        with unused_raw_placeholders(adapter) as audit:
            pipe = adapter.build_lightx2v_distill_pipeline(**kwargs)
        if len(audit['names']) != 400 or audit['calls'] < 800: raise RuntimeError('Original placeholder coverage differs')
        report['residency'] = make_resident(runtime)
        report['auxiliary_bytes'] = make_metadata_resident(metadata)
        for arm in ('swap_A', 'swap_B'):
            for step in range(4):
                for block in BLOCKS:
                    for name in ('channels', 'b2'):
                        actions[arm][step, block][name] = actions[arm][step, block][name].cuda().contiguous()
        report['assets'] = assets
        monitor = AssignedGPUResources(); monitor.start()
        # An actual GPU normalization check precedes model generation.
        probe_x = ((torch.arange(32 * 5120, device='cuda').reshape(32, 5120) % 127) / 31).bfloat16()
        with budget.call('GPU_norm_reference', 'standalone_normalization_calls'):
            expected = torch.nn.functional.layer_norm(probe_x, (5120,), None, None, 1e-6)
        with budget.call('GPU_norm_statistics', 'standalone_normalization_calls'):
            observed, _, _ = layer_norm_with_statistics(probe_x, infer_dtype=torch.bfloat16, sensitive_layer_dtype=torch.bfloat16)
        if not torch.equal(expected, observed): raise RuntimeError('GPU norm-statistics bit control failed')
        report['GPU_norm_statistics_bitwise_identity'] = True
        del probe_x, expected, observed
        sources = read(ROOT / '.runtime/inputs/r17_fresh23/canonical.json')
        old = read(ROOT / '.runtime/inputs/calibration24/canonical.json')[0]
        expected_old = next(x for x in read(ROOT / 'research/20261023/R16V0_native_results.json')['requests'] if x['arm'] == 'UB1')
        jobs = [('control_plain', 'plain', -1), ('control_observed', 'identity', -1)]
        jobs += [(f'source{i:02d}', 'diagnostic', i) for i in p['sources']]
        adapter.activate_lightx2v_hooks(runtime=runtime)
        scheduler = pipe.runner.scheduler; old_post = scheduler.step_post
        try:
            with RowDispatch(runtime, metadata, 'UB1') as dispatch, Probe(dispatch, actions, budget, root) as observer:
                for name, mode, source in jobs:
                    monitor.check()
                    if sum(f.stat().st_size for f in root.rglob('*') if f.is_file()) > 1 << 30:
                        raise RuntimeError('C1 output cap exceeded')
                    if shutil.disk_usage(root).free < 80 << 30: raise RuntimeError('C1 free disk reserve exceeded')
                    sample = old if source < 0 else sources[source]
                    image = sample.get('image_path') or sample['img_paths'][0]
                    if sha(image) != sample['input_provenance']['image_sha256']: raise RuntimeError('Source image bytes changed')
                    folder = root / name; folder.mkdir(); video = folder / 'output.mp4'
                    observer.request, observer.mode, observer.source = name, mode, source
                    before, before_extra = dict(runtime.counters), dict(observer.extra_counters)
                    snapshots, initial = [], []
                    def post():
                        step = int(scheduler.step_index)
                        if step == 0: initial.append(scheduler.latents.detach().float().clone())
                        value = old_post(); snapshots.append((step, scheduler.latents.detach().float().clone())); return value
                    scheduler.step_post = post; scheduler.generator = None
                    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
                    with budget.call(name, 'full_generations'), torch.inference_mode():
                        pipe.generate(seed=42, image_path=image, prompt=sample['prompt'], negative_prompt='', save_result_path=str(video))
                        torch.cuda.synchronize()
                    scheduler.step_post = old_post
                    if observer.pending or observer.active is not None: raise RuntimeError('Incomplete next-block readout')
                    if [s for s, _ in snapshots] != list(range(4)) or len(initial) != 1: raise RuntimeError('Incomplete original trajectory')
                    trajectory = [{'step': s, 'sha256': tensor_sha(t.cpu().contiguous())} for s, t in snapshots]
                    noise = tensor_sha(initial[0].cpu().contiguous())
                    if source < 0 and (trajectory != [{'step': r['step'], 'sha256': r['sha256']} for r in expected_old['trajectory']]
                                       or noise != expected_old['initial_noise_sha256']):
                        raise RuntimeError('Original full four-state UB1/observation bit identity failed')
                    counters = {k: runtime.counters[k] - before[k] - (observer.extra_counters[k] - before_extra[k]) for k in before}
                    if counters != {'linear_forward_calls': 1600, 'activation_qdq_calls': 1600, 'cache_misses': 0, 'weight_transfers': 0} or audit['fallback_calls']:
                        raise RuntimeError('Original all160 native baseline coverage failed')
                    peak = torch.cuda.max_memory_allocated()
                    if peak > 79 << 30: raise RuntimeError('C1 peak allocation cap exceeded')
                    monitor.check()
                    request = {'name': name, 'mode': mode, 'source_index': source, 'source_group': sample.get('source_group'),
                        'image_sha256': sha(image), 'prompt_sha256': hashlib.sha256(sample['prompt'].encode()).hexdigest(),
                        'initial_noise_sha256': noise, 'trajectory': trajectory, 'main_native_counter_delta': counters,
                        'video_sha256': sha(video), 'video_HF_relative': str(video.relative_to(hf)),
                        'free_UB1_baseline_no_resets': True, 'candidate_free_video': False,
                        'wall_seconds_includes_diagnostics_not_a_speed_benchmark': time.monotonic() - started,
                        'peak_allocated_bytes': peak, 'original_four_latent_identity_required_and_passed': source < 0}
                    atomic(folder / 'REQUEST.json', request)
                    report['requests'].append(request); report['finite_scopes'] = observer.rows
                    report['status'] = 'RUNNING_FINITE_C1_NATIVE_DIAGNOSTIC'
                    report['budget'] = budget.state; atomic(root / 'RUN.json', report)
                    print(f'C1 completed generation{len(report["requests"])}/8; finite native scopes{len(observer.rows)}/72', flush=True)
                    del snapshots, initial, request
        finally:
            scheduler.step_post = old_post; adapter.deactivate_lightx2v_hooks()
        if budget.state['used'] != CAPS or len(report['finite_scopes']) != 72: raise RuntimeError('Exact finite C1 scope incomplete')
        report['status'] = 'COMPLETE_NATIVE_C1_FIXED_SCOPE'
        atomic(root / 'RUN.json', report)
        # Separate CPU analysis consumes only completed saved numbers/features.
        from analyze_c1_native import analyze
        analyze(root)
        success = True
    except BaseException as error:
        report.update(status='STOPPED_NATIVE_C1_NO_RETRY', error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        if monitor is not None:
            monitor.close(); report['resources'] = monitor.report()
            if monitor.violations:
                success = False; report['status'] = 'FAILED_GPU_ISOLATION_NO_RETRY'
        report['budget'] = budget.state; atomic(root / 'RUN.json', report); budget.close()
        close_scope(report, success)


if __name__ == '__main__': main()
